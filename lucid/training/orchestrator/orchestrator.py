"""Phase 1 training orchestrator.

This module keeps the training loop intentionally small:

sample episode -> run pipeline -> validate -> assign blame -> propose local patch
-> shadow test -> promote/reject -> maintain failure replay -> write audit.

The larger specs in ``superintelligence/`` describe later-scale systems
(quantization lifecycle, corpus factories, large canary management). This file
implements the MVP loop needed to prove the architecture without pretending
those later systems already exist.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4


class PipelineStage(Protocol):
    """Minimal contract for injected cognitive stages."""

    def run(self, stage_input: Any) -> Any: ...


Perception = PipelineStage
CueEncoder = PipelineStage
DMF = PipelineStage
Binding = PipelineStage
ContextOp = PipelineStage
Interference = PipelineStage
Lucidity = PipelineStage
Projector = PipelineStage
Decoder = PipelineStage


@dataclass(slots=True)
class TrainingEpisode:
    episode_id: str
    raw_input: Any
    modality: str
    task_intent: str
    context: dict
    constraints: dict
    expected_output: Any
    validator_type: str
    metadata: dict
    allowed_tools: list[str] | None = None


@dataclass(slots=True)
class RunLog:
    episode_id: str
    raw_input: Any
    evidence_graph: dict
    cue_cloud: dict
    active_traces: list
    trace_clusters: list
    candidate_bindings: list
    context_frames: list
    scoped_trace_assignments: dict
    interference_edges: list
    active_basins: list
    basin_assemblies: dict
    lucidity_features: dict
    lucidity_decision: str
    lucidity_margin: float
    projection_result: Any
    decoder_output: Any
    validator_result: dict
    cost_metrics: dict


@dataclass(slots=True)
class ValidationResult:
    success: bool
    score: float
    failure_signals: list
    expected_state: Any
    confidence: float


@dataclass(slots=True)
class FailureDiagnosis:
    primary_module: str
    secondary_modules: list
    blame_confidence: float
    evidence: dict
    recommended_update_level: int


@dataclass(slots=True)
class UpdateProposal:
    patch_type: str
    target_objects: list
    update_level: int
    expected_fix: str
    risk_level: str
    shadow_test_bundle: list


@dataclass(slots=True)
class Patch:
    patch_id: str
    patch_type: str
    target_objects: list
    delta: dict
    reason: str
    update_level: int
    created_at: str


@dataclass(slots=True)
class PatchResult:
    patch_id: str
    fixed_target: bool
    retention_passed: bool
    cost_delta: float
    quality_delta: float
    promoted: bool
    episode_shadow_passed: bool
    retention_suite_version: str
    notes: str
    regressed_episode_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class UpdateRegion:
    subsystem: str
    target_ids: list[str]
    update_kind: str
    magnitude: str


@dataclass(slots=True)
class ResponsibilityAssignment:
    primary_blame: str
    secondary_blame: list[str]
    confidence: float
    evidence: dict


@dataclass(slots=True)
class ConsolidationDirective:
    action: str
    target_ids: list[str]
    reason: str
    heat_tier: str = ""
    precision_tier: str = ""


@dataclass(slots=True)
class TrainingGovernorInput:
    lucidity_decision: str
    task_outcome: str
    validation_score: float
    failure_signals: list
    top_margin: float
    active_trace_ids: list[str]
    active_basin_ids: list[str]
    touched_subsystems: list[str]
    iteration_cost: dict
    heat_state_snapshot: dict


@dataclass(slots=True)
class TrainingGovernorDecision:
    action: str
    reason: str
    governor_input: TrainingGovernorInput
    update_regions: list[UpdateRegion] = field(default_factory=list)
    responsibility_assignment: ResponsibilityAssignment | None = None
    consolidation_directives: list[ConsolidationDirective] = field(default_factory=list)
    max_modules_updated: int = 1
    human_override: bool = False
    audit_log: dict = field(default_factory=dict)


@dataclass(slots=True)
class MarginMetric:
    target_id: str
    success_count: int = 0
    failure_count: int = 0
    no_update_count: int = 0
    total_success_margin: float = 0.0
    total_failure_margin: float = 0.0
    last_action: str = ""

    def record(self, *, success: bool, margin: float, action: str) -> None:
        self.last_action = action
        if success:
            self.success_count += 1
            self.total_success_margin += margin
        else:
            self.failure_count += 1
            self.total_failure_margin += margin
        if action == "NO_UPDATE":
            self.no_update_count += 1

    def as_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "no_update_count": self.no_update_count,
            "mean_margin_on_success": self.total_success_margin / self.success_count
            if self.success_count
            else 0.0,
            "mean_margin_on_fail": self.total_failure_margin / self.failure_count
            if self.failure_count
            else 0.0,
            "last_action": self.last_action,
        }


@dataclass(slots=True)
class FailureReplayEntry:
    episode_id: str
    run_log_id_last_failure: str
    patch_ids_applied: list
    shadow_passed: bool
    consecutive_successes: int
    replay_priority: int
    entered_at: str
    last_attempt_at: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if callable(value):
        return {"callable": getattr(value, "__qualname__", type(value).__name__)}
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _safe_path_part(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return (clean.strip("_") or "episode")[:80]


def _as_dict(value: Any, key: str) -> dict:
    if isinstance(value, dict):
        return value
    return {key: value}


def _item_id(value: Any, preferred_keys: tuple[str, ...]) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in preferred_keys:
            raw = value.get(key)
            if raw:
                return str(raw)
        return str(value)
    for key in preferred_keys:
        raw = getattr(value, key, "")
        if raw:
            return str(raw)
    return str(value)


def _ids_from_items(items: list, preferred_keys: tuple[str, ...]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for item in items:
        item_id = _item_id(item, preferred_keys).strip()
        if item_id and item_id not in seen:
            ids.append(item_id)
            seen.add(item_id)
    return ids


class TrainingGovernor:
    """Cost-control policy that turns run outcomes into auditable update choices."""

    PATCH_TO_REGION = {
        "PerceptionPatch": ("perception", "store_evidence_targets"),
        "CueEncoderPatch": ("cue_encoder", "store_high_recall_cue_targets"),
        "TracePatch": ("dmf", "strengthen_or_seed_trace"),
        "InterferencePatch": ("interference", "adjust_support_or_conflict"),
        "BindingPatch": ("binding", "adjust_affordance"),
        "ContextPatch": ("context_op", "adjust_scope_gate"),
        "BasinPatch": ("basins", "adjust_basin_links"),
        "LucidityPatch": ("lucidity", "adjust_commit_policy"),
        "ProjectorPatch": ("projector", "store_projection_example"),
        "DecoderPatch": ("decoder", "add_decoder_correction_pair"),
    }

    def __init__(
        self,
        *,
        high_margin_threshold: float = 0.75,
        max_modules_updated: int = 1,
        max_modules_updated_when_uncertain: int = 2,
    ) -> None:
        self.high_margin_threshold = high_margin_threshold
        self.max_modules_updated = max_modules_updated
        self.max_modules_updated_when_uncertain = max_modules_updated_when_uncertain
        self._margin_metrics: dict[str, MarginMetric] = {}
        self._action_counts: dict[str, int] = {}

    def observe(self, run_log: RunLog, validation: ValidationResult) -> TrainingGovernorDecision:
        governor_input = self._input_from_run(run_log, validation, [])
        if validation.success:
            if (
                run_log.lucidity_decision == "commit"
                and run_log.lucidity_margin >= self.high_margin_threshold
            ):
                decision = TrainingGovernorDecision(
                    action="NO_UPDATE",
                    reason="high_margin_stable_pass",
                    governor_input=governor_input,
                    consolidation_directives=self._stability_directives(governor_input),
                    audit_log={
                        "margin_threshold": self.high_margin_threshold,
                        "protected_ids": governor_input.active_trace_ids
                        + governor_input.active_basin_ids,
                    },
                )
                self.record_decision(decision, validation)
                return decision
            decision = TrainingGovernorDecision(
                action="DEFER",
                reason="low_margin_success_observe_more",
                governor_input=governor_input,
                audit_log={"margin_threshold": self.high_margin_threshold},
            )
            self.record_decision(decision, validation)
            return decision

        return TrainingGovernorDecision(
            action="UPDATE",
            reason="needs_responsibility_assignment",
            governor_input=governor_input,
            audit_log={"margin_threshold": self.high_margin_threshold},
        )

    def decide_update(
        self,
        run_log: RunLog,
        validation: ValidationResult,
        diagnosis: FailureDiagnosis,
        proposal: UpdateProposal,
    ) -> TrainingGovernorDecision:
        responsibility = ResponsibilityAssignment(
            primary_blame=diagnosis.primary_module,
            secondary_blame=diagnosis.secondary_modules,
            confidence=diagnosis.blame_confidence,
            evidence=diagnosis.evidence,
        )
        update_regions = self._regions_for(proposal)
        max_modules = (
            self.max_modules_updated_when_uncertain
            if diagnosis.blame_confidence < 0.65
            else self.max_modules_updated
        )
        update_regions = self._cap_update_regions(update_regions, max_modules)
        governor_input = self._input_from_run(
            run_log,
            validation,
            [region.subsystem for region in update_regions],
        )

        if proposal.update_level <= 0 or not update_regions:
            decision = TrainingGovernorDecision(
                action="NO_UPDATE",
                reason="no_safe_local_update_region",
                governor_input=governor_input,
                responsibility_assignment=responsibility,
                max_modules_updated=max_modules,
                audit_log={"proposal_update_level": proposal.update_level},
            )
            self.record_decision(decision, validation)
            return decision

        decision = TrainingGovernorDecision(
            action="UPDATE",
            reason="targeted_active_region_update",
            governor_input=governor_input,
            update_regions=update_regions,
            responsibility_assignment=responsibility,
            max_modules_updated=max_modules,
            audit_log={
                "proposal_update_level": proposal.update_level,
                "patch_type": proposal.patch_type,
            },
        )
        self.record_decision(decision, validation)
        return decision

    def record_decision(
        self,
        decision: TrainingGovernorDecision,
        validation: ValidationResult,
    ) -> None:
        self._action_counts[decision.action] = self._action_counts.get(decision.action, 0) + 1
        target_ids = (
            decision.governor_input.active_basin_ids
            or decision.governor_input.active_trace_ids
            or ["global"]
        )
        for target_id in target_ids[:8]:
            metric = self._margin_metrics.setdefault(target_id, MarginMetric(target_id))
            metric.record(
                success=validation.success,
                margin=decision.governor_input.top_margin,
                action=decision.action,
            )

    def metrics(self) -> dict:
        return {
            "action_counts": dict(sorted(self._action_counts.items())),
            "margin_metrics": {
                target_id: metric.as_dict()
                for target_id, metric in sorted(self._margin_metrics.items())
            },
        }

    @staticmethod
    def _input_from_run(
        run_log: RunLog,
        validation: ValidationResult,
        touched_subsystems: list[str],
    ) -> TrainingGovernorInput:
        return TrainingGovernorInput(
            lucidity_decision=run_log.lucidity_decision,
            task_outcome="success" if validation.success else "fail",
            validation_score=validation.score,
            failure_signals=validation.failure_signals,
            top_margin=run_log.lucidity_margin,
            active_trace_ids=_ids_from_items(run_log.active_traces, ("trace_id", "id")),
            active_basin_ids=_ids_from_items(run_log.active_basins, ("basin_id", "id")),
            touched_subsystems=touched_subsystems,
            iteration_cost=run_log.cost_metrics,
            heat_state_snapshot={},
        )

    @staticmethod
    def _stability_directives(
        governor_input: TrainingGovernorInput,
    ) -> list[ConsolidationDirective]:
        targets = governor_input.active_trace_ids + governor_input.active_basin_ids
        if not targets:
            return []
        return [
            ConsolidationDirective(
                action="CANDIDATE_FOR_QUANTIZATION",
                target_ids=targets[:16],
                reason="high_margin_success_observed",
                heat_tier="candidate_cold",
                precision_tier="measure_before_quantize",
            )
        ]

    def _regions_for(self, proposal: UpdateProposal) -> list[UpdateRegion]:
        mapping = self.PATCH_TO_REGION.get(proposal.patch_type)
        if mapping is None:
            return []
        subsystem, update_kind = mapping
        return [
            UpdateRegion(
                subsystem=subsystem,
                target_ids=[str(target) for target in proposal.target_objects],
                update_kind=update_kind,
                magnitude="low",
            )
        ]

    @staticmethod
    def _cap_update_regions(
        regions: list[UpdateRegion],
        max_modules: int,
    ) -> list[UpdateRegion]:
        selected: list[UpdateRegion] = []
        used: set[str] = set()
        for region in regions:
            if region.subsystem in used:
                selected.append(region)
                continue
            if len(used) >= max_modules:
                continue
            selected.append(region)
            used.add(region.subsystem)
        return selected


class TrainingAuditLogger:
    """Writes one human-readable and machine-readable folder per training step."""

    def __init__(self, base_dir: str | Path = "audit/training") -> None:
        self.base_dir = Path(base_dir)

    def write_step(
        self,
        *,
        step_index: int,
        action: str,
        episode: TrainingEpisode | None,
        run_log: RunLog | None = None,
        validation: ValidationResult | None = None,
        diagnosis: FailureDiagnosis | None = None,
        proposal: UpdateProposal | None = None,
        patch: Patch | None = None,
        patch_result: PatchResult | None = None,
        governor_decision: TrainingGovernorDecision | None = None,
        status: dict | None = None,
        error_message: str = "",
    ) -> Path:
        episode_id = episode.episode_id if episode else "unknown"
        step_dir = self.base_dir / f"step_{step_index:06d}_{_safe_path_part(episode_id)}"
        step_dir.mkdir(parents=True, exist_ok=True)

        artifacts = {
            "episode": episode,
            "run_log": run_log,
            "validation": validation,
            "diagnosis": diagnosis,
            "proposal": proposal,
            "patch": patch,
            "patch_result": patch_result,
            "governor_decision": governor_decision,
            "status": status,
        }
        files: dict[str, str] = {}
        for name, payload in artifacts.items():
            if payload is None:
                continue
            file_name = f"{name}.json"
            _write_json(step_dir / file_name, payload)
            files[name] = file_name

        manifest = {
            "schema_version": 1,
            "created_at": _utc_now_iso(),
            "step_index": step_index,
            "action": action,
            "episode_id": episode_id,
            "validation_success": validation.success if validation else None,
            "validation_score": validation.score if validation else None,
            "lucidity_decision": run_log.lucidity_decision if run_log else "",
            "lucidity_margin": run_log.lucidity_margin if run_log else None,
            "patch_id": patch.patch_id if patch else "",
            "patch_promoted": patch_result.promoted if patch_result else None,
            "governor_action": governor_decision.action if governor_decision else "",
            "governor_reason": governor_decision.reason if governor_decision else "",
            "error_message": error_message,
            "files": files,
        }
        _write_json(step_dir / "manifest.json", manifest)

        lines = [
            f"{action} - {episode_id}",
            "=" * (len(action) + len(episode_id) + 3),
            "",
            f"step: {step_index}",
            f"validation_success: {manifest['validation_success']}",
            f"lucidity: {manifest['lucidity_decision'] or '-'}",
            f"governor: {manifest['governor_action'] or '-'}",
            f"governor_reason: {manifest['governor_reason'] or '-'}",
            f"patch: {manifest['patch_id'] or '-'}",
            f"error: {error_message or '-'}",
            "",
            "files:",
            *[f"- {name}: {file_name}" for name, file_name in files.items()],
            "",
        ]
        (step_dir / "README.txt").write_text("\n".join(lines), encoding="utf-8")
        return step_dir


class RunExecutor:
    """Runs an episode through the injected pipeline stages."""

    def __init__(
        self,
        perception: Perception,
        cue_encoder: CueEncoder,
        dmf: DMF,
        binding: Binding,
        context_op: ContextOp,
        interference: Interference,
        lucidity: Lucidity,
        projector: Projector,
        decoder: Decoder,
    ) -> None:
        self.perception = perception
        self.cue_encoder = cue_encoder
        self.dmf = dmf
        self.binding = binding
        self.context_op = context_op
        self.interference = interference
        self.lucidity = lucidity
        self.projector = projector
        self.decoder = decoder

    @staticmethod
    def _derive_lucidity_decision(margin: float) -> str:
        if margin >= 0.75:
            return "commit"
        if margin < 0.40:
            return "reject"
        return "test_consequence"

    def _run_from_states(self, episode: TrainingEpisode, state: dict | None = None) -> RunLog:
        state = state or {}
        evidence_graph = _as_dict(
            state.get("evidence_graph") or self.perception.run(episode.raw_input),
            "evidence",
        )
        cue_cloud = _as_dict(
            state.get("cue_cloud") or self.cue_encoder.run(evidence_graph),
            "cue",
        )
        dmf_out = _as_dict(state.get("dmf_out") or self.dmf.run(cue_cloud), "dmf")
        active_traces = state.get("active_traces", dmf_out.get("active_traces", []))
        trace_clusters = state.get("trace_clusters", dmf_out.get("trace_clusters", []))

        binding_out = _as_dict(
            state.get("binding_out")
            or self.binding.run(
                {
                    "dmf_out": dmf_out,
                    "active_traces": active_traces,
                    "trace_clusters": trace_clusters,
                    "episode_context": episode.context,
                }
            ),
            "binding",
        )
        candidate_bindings = state.get(
            "candidate_bindings",
            binding_out.get("candidate_bindings", []),
        )

        context_out = _as_dict(
            state.get("context_out")
            or self.context_op.run(
                {
                    "candidate_bindings": candidate_bindings,
                    "context": episode.context,
                    "constraints": episode.constraints,
                }
            ),
            "context",
        )
        context_frames = state.get("context_frames", context_out.get("context_frames", []))
        scoped_trace_assignments = state.get(
            "scoped_trace_assignments",
            context_out.get("scoped_trace_assignments", {}),
        )

        interference_out = _as_dict(
            state.get("interference_out")
            or self.interference.run(
                {
                    "context_frames": context_frames,
                    "active_traces": active_traces,
                    "candidate_bindings": candidate_bindings,
                }
            ),
            "interference",
        )
        interference_edges = state.get(
            "interference_edges",
            interference_out.get("interference_edges", []),
        )
        active_basins = state.get("active_basins", interference_out.get("active_basins", []))
        basin_assemblies = state.get(
            "basin_assemblies",
            interference_out.get("basin_assemblies", {}),
        )

        lucidity_out = state.get("lucidity_out")
        if lucidity_out is None:
            lucidity_out = self.lucidity.run(
                {
                    "active_basins": active_basins,
                    "interference_edges": interference_edges,
                    "task_intent": episode.task_intent,
                }
            )
        if isinstance(lucidity_out, tuple) and len(lucidity_out) >= 2:
            lucidity_decision = str(lucidity_out[0])
            lucidity_margin = float(lucidity_out[1])
        else:
            lucidity_margin = float(lucidity_out) if isinstance(lucidity_out, (int, float)) else 0.0
            lucidity_decision = self._derive_lucidity_decision(lucidity_margin)
        if lucidity_decision not in {"commit", "reject", "test_consequence"}:
            lucidity_decision = self._derive_lucidity_decision(lucidity_margin)

        projection_result = None
        projector_called = False
        if lucidity_decision == "test_consequence":
            projector_called = True
            projection_result = self.projector.run(
                {
                    "basins": active_basins,
                    "assemblies": basin_assemblies,
                    "constraints": episode.constraints,
                }
            )

        decoder_output = self.decoder.run(
            {
                "decision": lucidity_decision,
                "margin": lucidity_margin,
                "basin_assemblies": basin_assemblies,
                "projection_result": projection_result,
            }
        )

        return RunLog(
            episode_id=episode.episode_id,
            raw_input=episode.raw_input,
            evidence_graph=evidence_graph,
            cue_cloud=cue_cloud,
            active_traces=active_traces,
            trace_clusters=trace_clusters,
            candidate_bindings=candidate_bindings,
            context_frames=context_frames,
            scoped_trace_assignments=scoped_trace_assignments,
            interference_edges=interference_edges,
            active_basins=active_basins,
            basin_assemblies=basin_assemblies,
            lucidity_features={"raw_output": lucidity_out},
            lucidity_decision=lucidity_decision,
            lucidity_margin=lucidity_margin,
            projection_result=projection_result,
            decoder_output=decoder_output,
            validator_result={},
            cost_metrics={
                "stages_run": 9 if projector_called else 8,
                "projector_called": projector_called,
                "traces_activated": len(active_traces),
                "basins_activated": len(active_basins),
            },
        )

    def run(self, episode: TrainingEpisode, mode: str = "training_observation") -> RunLog:
        _ = mode
        return self._run_from_states(episode, {})


class ExactMatchValidator:
    def evaluate(self, run_log: RunLog, episode: TrainingEpisode) -> ValidationResult:
        success = run_log.decoder_output == episode.expected_output
        return ValidationResult(
            success=success,
            score=1.0 if success else 0.0,
            failure_signals=[] if success else ["exact_match_failed"],
            expected_state=episode.expected_output,
            confidence=1.0,
        )


class UnitTestValidator:
    def evaluate(self, run_log: RunLog, episode: TrainingEpisode) -> ValidationResult:
        if not callable(episode.expected_output):
            return ValidationResult(False, 0.0, ["expected_output_not_callable"], None, 1.0)
        try:
            success = bool(episode.expected_output(run_log.decoder_output))
        except Exception as exc:  # noqa: BLE001 - validators must report user-test failures
            return ValidationResult(
                False,
                0.0,
                [f"unit_test_exception:{type(exc).__name__}"],
                "callable(expected_output)",
                1.0,
            )
        return ValidationResult(
            success=success,
            score=1.0 if success else 0.0,
            failure_signals=[] if success else ["unit_test_failed"],
            expected_state="callable(expected_output)",
            confidence=1.0,
        )


class SelfConsistencyValidator:
    def evaluate(self, run_log: RunLog, episode: TrainingEpisode) -> ValidationResult:
        _ = episode
        success = run_log.lucidity_margin >= 0.6 and run_log.decoder_output is not None
        return ValidationResult(
            success=success,
            score=1.0 if success else max(0.0, min(1.0, run_log.lucidity_margin)),
            failure_signals=[] if success else ["self_consistency_failed"],
            expected_state=None,
            confidence=0.8,
        )


class ValidatorFactory:
    def __init__(self) -> None:
        self._validators = {
            "exact_match": ExactMatchValidator(),
            "unit_test": UnitTestValidator(),
            "self_consistency": SelfConsistencyValidator(),
        }

    def get(self, validator_type: str) -> Any:
        return self._validators.get(validator_type, self._validators["self_consistency"])


class BlameAssigner:
    """Deterministic responsibility classifier for Phase 1."""

    def diagnose(self, run_log: RunLog, validation: ValidationResult) -> FailureDiagnosis:
        if not run_log.evidence_graph:
            return FailureDiagnosis("perception", [], 0.90, {"reason": "missing_evidence"}, 9)
        if not run_log.active_traces:
            return FailureDiagnosis("cue_encoder_or_DMF", ["dmf"], 0.85, {"reason": "no_traces"}, 2)
        if not run_log.candidate_bindings:
            return FailureDiagnosis("binding", [], 0.85, {"reason": "no_bindings"}, 4)
        if not run_log.context_frames:
            return FailureDiagnosis("context_op", [], 0.80, {"reason": "no_context"}, 5)
        if not run_log.active_basins:
            return FailureDiagnosis(
                "interference_or_basin",
                ["basins"],
                0.75,
                {"reason": "no_basins"},
                3,
            )
        if "decoder_render_mismatch" in validation.failure_signals:
            return FailureDiagnosis(
                "decoder",
                [],
                0.85,
                {"reason": "committed_state_render_mismatch"},
                9,
            )
        if run_log.lucidity_decision == "reject" and validation.expected_state is not None:
            return FailureDiagnosis(
                "lucidity_too_strict",
                ["lucidity"],
                0.70,
                {"reason": "false_reject"},
                8,
            )
        if run_log.lucidity_decision == "commit" and not validation.success:
            return FailureDiagnosis(
                "lucidity_too_loose",
                ["lucidity"],
                0.70,
                {"reason": "false_commit"},
                8,
            )
        if run_log.decoder_output is None:
            return FailureDiagnosis("decoder", [], 0.85, {"reason": "empty_output"}, 9)
        return FailureDiagnosis("unknown", [], 0.20, {"reason": "unclassified"}, 0)


class UpdatePlanner:
    """Chooses the smallest local update allowed by the diagnosis."""

    MODULE_TO_LEVEL = {
        "perception": 9,
        "cue_encoder": 2,
        "dmf": 2,
        "cue_encoder_or_DMF": 2,
        "binding": 4,
        "context_op": 5,
        "interference_or_basin": 3,
        "basins": 6,
        "lucidity_too_strict": 8,
        "lucidity_too_loose": 8,
        "projector": 8,
        "decoder": 9,
        "unknown": 0,
    }
    MODULE_TO_PATCH = {
        "perception": "PerceptionPatch",
        "cue_encoder": "CueEncoderPatch",
        "dmf": "TracePatch",
        "basins": "BasinPatch",
        "projector": "ProjectorPatch",
    }
    LEVEL_TO_PATCH = {
        0: "NoPatch",
        2: "TracePatch",
        3: "InterferencePatch",
        4: "BindingPatch",
        5: "ContextPatch",
        6: "BasinPatch",
        8: "LucidityPatch",
        9: "DecoderPatch",
    }

    def plan(self, diagnosis: FailureDiagnosis, run_log: RunLog) -> UpdateProposal:
        level = min(
            self.MODULE_TO_LEVEL.get(diagnosis.primary_module, 0),
            diagnosis.recommended_update_level or 10,
        )
        patch_type = self.MODULE_TO_PATCH.get(
            diagnosis.primary_module,
            self.LEVEL_TO_PATCH.get(level, "NoPatch"),
        )
        targets = self._targets_for(run_log)
        if level == 0:
            targets = [run_log.episode_id]
        risk = "high" if level >= 8 else "medium" if level >= 5 else "low"
        return UpdateProposal(
            patch_type=patch_type,
            target_objects=targets,
            update_level=level,
            expected_fix=f"repair_{diagnosis.primary_module}",
            risk_level=risk,
            shadow_test_bundle=[run_log.episode_id],
        )

    @staticmethod
    def _targets_for(run_log: RunLog) -> list:
        targets = [str(item) for item in run_log.active_traces[:1]]
        if run_log.active_basins:
            basin = run_log.active_basins[0]
            if isinstance(basin, dict):
                targets.append(str(basin.get("basin_id", basin.get("id", "basin_unknown"))))
            else:
                targets.append(str(basin))
        return (targets or [run_log.episode_id])[:2]


class PatchBuilder:
    def build(
        self,
        proposal: UpdateProposal,
        run_log: RunLog,
        governor_decision: TrainingGovernorDecision,
    ) -> Patch:
        delta = {
            "kind": proposal.patch_type,
            "episode_id": run_log.episode_id,
            "magnitude": "low",
            "active_only": True,
            "governor_action": governor_decision.action,
            "governor_reason": governor_decision.reason,
            "update_regions": _jsonable(governor_decision.update_regions),
        }
        if proposal.patch_type == "DecoderPatch":
            delta["decoder_only"] = True
            delta["correction_pair"] = {
                "wrong_output": _jsonable(run_log.decoder_output),
                "expected_output": _jsonable(run_log.validator_result.get("expected_state", "")),
                "update_scope": "decoder_only",
            }
        return Patch(
            patch_id=str(uuid4()),
            patch_type=proposal.patch_type,
            target_objects=proposal.target_objects,
            delta=delta,
            reason=proposal.expected_fix,
            update_level=proposal.update_level,
            created_at=_utc_now_iso(),
        )


class FailureReplayStore:
    """Tracks unresolved failures and enforces the three-part clear rule."""

    def __init__(self) -> None:
        self.entries: dict[str, FailureReplayEntry] = {}
        self._cleared_without_shadow_pass = 0

    def add_or_refresh(self, run_log: RunLog) -> None:
        now = _utc_now_iso()
        entry = self.entries.get(run_log.episode_id)
        if entry is None:
            self.entries[run_log.episode_id] = FailureReplayEntry(
                episode_id=run_log.episode_id,
                run_log_id_last_failure=f"{run_log.episode_id}:{now}",
                patch_ids_applied=[],
                shadow_passed=False,
                consecutive_successes=0,
                replay_priority=1,
                entered_at=now,
                last_attempt_at=now,
            )
            return
        entry.run_log_id_last_failure = f"{run_log.episode_id}:{now}"
        entry.shadow_passed = False
        entry.consecutive_successes = 0
        entry.last_attempt_at = now

    def on_patch_rejected(self, episode_id: str) -> None:
        if episode_id in self.entries:
            entry = self.entries[episode_id]
            entry.replay_priority = min(10, entry.replay_priority + 1)
            entry.last_attempt_at = _utc_now_iso()

    def on_patch_promoted(self, episode_id: str, patch_id: str) -> None:
        entry = self.entries.get(episode_id)
        if entry is None:
            return
        entry.patch_ids_applied.append(patch_id)
        entry.shadow_passed = False
        entry.consecutive_successes = 0
        entry.last_attempt_at = _utc_now_iso()

    def on_episode_shadow_passed(self, episode_id: str) -> None:
        if episode_id in self.entries:
            self.entries[episode_id].shadow_passed = True

    def record_success(self, episode_id: str) -> None:
        if episode_id in self.entries:
            self.entries[episode_id].consecutive_successes += 1

    def try_clear(self, episode_id: str) -> bool:
        entry = self.entries.get(episode_id)
        if entry is None:
            return False
        if entry.patch_ids_applied and entry.shadow_passed and entry.consecutive_successes >= 3:
            del self.entries[episode_id]
            return True
        if entry.patch_ids_applied and entry.consecutive_successes >= 3 and not entry.shadow_passed:
            self._cleared_without_shadow_pass += 1
        return False

    def contains(self, episode_id: str) -> bool:
        return episode_id in self.entries

    def metrics(self) -> dict:
        return {
            "failure_replay_queue_depth": len(self.entries),
            "episodes_stuck_past_5_attempts": sum(
                1 for entry in self.entries.values() if entry.replay_priority >= 5
            ),
            "cleared_without_shadow_pass": self._cleared_without_shadow_pass,
        }


class EpisodeScheduler:
    def __init__(self, episodes: list[TrainingEpisode], replay: FailureReplayStore) -> None:
        self.episodes = list(episodes)
        self.replay = replay

    def sample(self) -> TrainingEpisode:
        if not self.episodes:
            raise RuntimeError("No episodes available")
        replay_ids = set(self.replay.entries)
        replay_episodes = [episode for episode in self.episodes if episode.episode_id in replay_ids]
        if replay_episodes:
            return random.choice(replay_episodes)
        return random.choice(self.episodes)


class RetentionSuiteManager:
    def __init__(self, episodes: list[TrainingEpisode], phase: int = 1) -> None:
        self.phase = phase
        self.suite_version = f"phase{phase}"
        self.episode_family = {
            episode.episode_id: str(
                episode.metadata.get("task_family")
                or episode.metadata.get("recipe")
                or episode.modality
                or "unknown"
            )
            for episode in episodes
        }
        self.regression_ids: list[str] = []
        self.canary_ids = self._seed_canaries(episodes)

    def max_shadow_episodes(self) -> int:
        if self.phase <= 1:
            return 500
        if self.phase <= 3:
            return 200
        if self.phase == 4:
            return 80
        return 50

    def coverage_map(self) -> dict[str, int]:
        coverage: dict[str, int] = {}
        for episode_id in self.canary_ids:
            family = self.episode_family.get(episode_id, "unknown")
            coverage[family] = coverage.get(family, 0) + 1
        return dict(sorted(coverage.items()))

    def select_shadow_bundle(self, target_episode_id: str) -> list[str]:
        cap = self.max_shadow_episodes()
        target_family = self.episode_family.get(target_episode_id, "unknown")
        bundle = [target_episode_id]

        for episode_id, family in self.episode_family.items():
            if family == target_family and episode_id != target_episode_id:
                bundle.append(episode_id)
            if len(bundle) >= min(cap, 6):
                break

        for episode_id in self.regression_ids:
            if episode_id not in bundle:
                bundle.append(episode_id)
            if len(bundle) >= cap:
                return bundle

        for episode_id in self.canary_ids:
            if episode_id not in bundle:
                bundle.append(episode_id)
            if len(bundle) >= cap:
                break
        return bundle

    def on_patch_result(self, bundle: list[str], patch_result: PatchResult) -> None:
        _ = bundle
        for episode_id in patch_result.regressed_episode_ids:
            if episode_id not in self.regression_ids:
                self.regression_ids.append(episode_id)
            if episode_id not in self.canary_ids:
                self.canary_ids.append(episode_id)

    def snapshot(self) -> dict:
        return {
            "suite_version": self.suite_version,
            "canary_count": len(self.canary_ids),
            "family_coverage": self.coverage_map(),
            "episode_ids": list(self.canary_ids),
            "regression_ids": list(self.regression_ids),
            "max_shadow_episodes": self.max_shadow_episodes(),
        }

    def _seed_canaries(self, episodes: list[TrainingEpisode]) -> list[str]:
        cap = self.max_shadow_episodes()
        canaries: list[str] = []
        seen_families: set[str] = set()
        for episode in episodes:
            family = self.episode_family.get(episode.episode_id, "unknown")
            if family not in seen_families:
                canaries.append(episode.episode_id)
                seen_families.add(family)
            if len(canaries) >= cap:
                return canaries
        for episode in episodes:
            if episode.episode_id not in canaries:
                canaries.append(episode.episode_id)
            if len(canaries) >= cap:
                break
        return canaries


class ShadowEvaluator:
    def __init__(
        self,
        executor: RunExecutor,
        validator_factory: ValidatorFactory,
        *,
        max_cost_increase: float = 0.05,
    ) -> None:
        self.executor = executor
        self.validator_factory = validator_factory
        self.max_cost_increase = max_cost_increase

    def test(
        self,
        patch: Patch,
        bundle: list[str],
        live_state: dict,
        episode_store: dict[str, TrainingEpisode],
    ) -> PatchResult:
        shadow_state = copy.deepcopy(live_state)
        shadow_state.setdefault("patches", {})[patch.patch_id] = _jsonable(patch.delta)

        live_scores: list[float] = []
        shadow_scores: list[float] = []
        live_costs: list[float] = []
        shadow_costs: list[float] = []
        regressed_episode_ids: list[str] = []
        target_fixed = False
        for index, episode_id in enumerate(bundle):
            episode = episode_store[episode_id]
            live_run = self.executor._run_from_states(episode, live_state.get("injected", {}))
            shadow_run = self.executor._run_from_states(episode, shadow_state.get("injected", {}))
            validator = self.validator_factory.get(episode.validator_type)
            live_result = validator.evaluate(live_run, episode)
            shadow_result = validator.evaluate(shadow_run, episode)
            live_scores.append(live_result.score)
            shadow_scores.append(shadow_result.score)
            live_costs.append(float(live_run.cost_metrics.get("stages_run", 0.0)))
            shadow_costs.append(float(shadow_run.cost_metrics.get("stages_run", 0.0)))
            if index == 0:
                target_fixed = shadow_result.score > live_result.score
            elif shadow_result.score < live_result.score - 0.01:
                regressed_episode_ids.append(episode_id)

        retention_passed = not regressed_episode_ids
        quality_delta = (shadow_scores[0] - live_scores[0]) if live_scores else 0.0
        live_cost = sum(live_costs) / max(1, len(live_costs))
        shadow_cost = sum(shadow_costs) / max(1, len(shadow_costs))
        cost_delta = shadow_cost - live_cost
        cost_passed = cost_delta <= self.max_cost_increase
        promoted = target_fixed and retention_passed and cost_passed
        if promoted:
            notes = "ok"
        elif not target_fixed:
            notes = "target_not_improved"
        elif not cost_passed:
            notes = "cost_regressed"
        else:
            notes = "retention_regressed"
        return PatchResult(
            patch_id=patch.patch_id,
            fixed_target=target_fixed,
            retention_passed=retention_passed,
            cost_delta=cost_delta,
            quality_delta=quality_delta,
            promoted=promoted,
            episode_shadow_passed=False,
            retention_suite_version="active",
            notes=notes,
            regressed_episode_ids=regressed_episode_ids,
        )

    def episode_shadow_passes(
        self,
        episode_id: str,
        live_state: dict,
        episode_store: dict[str, TrainingEpisode],
    ) -> bool:
        episode = episode_store[episode_id]
        run_log = self.executor._run_from_states(
            episode,
            live_state.get("injected", {}),
        )
        validator = self.validator_factory.get(episode.validator_type)
        return validator.evaluate(run_log, episode).success


class PromotionManager:
    def __init__(self) -> None:
        self.patch_history: list[Patch] = []
        self.rejected_patch_history: list[dict] = []

    def promote(
        self,
        patch: Patch,
        shadow_result: PatchResult,
        target_episode_id: str,
        live_state: dict,
        replay: FailureReplayStore,
        shadow: ShadowEvaluator,
        episode_store: dict[str, TrainingEpisode],
    ) -> PatchResult:
        live_state.setdefault("patches", {})[patch.patch_id] = _jsonable(patch.delta)
        self.patch_history.append(patch)
        replay.on_patch_promoted(target_episode_id, patch.patch_id)
        episode_shadow_passed = shadow.episode_shadow_passes(
            target_episode_id,
            live_state,
            episode_store,
        )
        if episode_shadow_passed:
            replay.on_episode_shadow_passed(target_episode_id)
        return PatchResult(
            patch_id=patch.patch_id,
            fixed_target=True,
            retention_passed=shadow_result.retention_passed,
            cost_delta=shadow_result.cost_delta,
            quality_delta=shadow_result.quality_delta,
            promoted=True,
            episode_shadow_passed=episode_shadow_passed,
            retention_suite_version=shadow_result.retention_suite_version,
            notes="promoted",
            regressed_episode_ids=shadow_result.regressed_episode_ids,
        )

    def reject(
        self,
        patch: Patch,
        reason: str,
        replay: FailureReplayStore,
        run_log: RunLog,
    ) -> None:
        self.rejected_patch_history.append({"patch": patch, "reason": reason})
        replay.add_or_refresh(run_log)
        replay.on_patch_rejected(run_log.episode_id)


class TrainingOrchestrator:
    def __init__(
        self,
        perception: Perception,
        cue_encoder: CueEncoder,
        dmf: DMF,
        binding: Binding,
        context_op: ContextOp,
        interference: Interference,
        lucidity: Lucidity,
        projector: Projector,
        decoder: Decoder,
        episodes: list[TrainingEpisode],
        phase: int = 1,
        debug: bool = False,
        audit_base_dir: str | Path = "audit/training",
    ) -> None:
        self.phase = phase
        self.debug = debug
        self.live_state: dict = {}
        self.episode_store = {episode.episode_id: episode for episode in episodes}
        self.step_index = 0

        self.failure_replay_store = FailureReplayStore()
        self.scheduler = EpisodeScheduler(episodes, self.failure_replay_store)
        self.executor = RunExecutor(
            perception,
            cue_encoder,
            dmf,
            binding,
            context_op,
            interference,
            lucidity,
            projector,
            decoder,
        )
        self.validator_factory = ValidatorFactory()
        self.governor = TrainingGovernor()
        self.blame_assigner = BlameAssigner()
        self.update_planner = UpdatePlanner()
        self.patch_builder = PatchBuilder()
        self.retention_suite = RetentionSuiteManager(episodes, phase)
        self.shadow_evaluator = ShadowEvaluator(self.executor, self.validator_factory)
        self.promotion_manager = PromotionManager()
        self.audit_logger = TrainingAuditLogger(audit_base_dir)

        self._run_history: list[tuple[RunLog, ValidationResult]] = []
        self._update_history: list[int] = []
        self._governor_action_history: list[str] = []
        self._last_run_log: RunLog | None = None
        self._last_patch: Patch | None = None

    def run_one_step(self) -> None:
        self.step_index += 1
        episode = None
        run_log = None
        validation = None
        diagnosis = None
        proposal = None
        patch = None
        patch_result = None
        governor_decision = None
        action = "exception"
        error_message = ""

        try:
            episode = self.scheduler.sample()
            run_log = self.executor.run(episode)
            validator = self.validator_factory.get(episode.validator_type)
            validation = validator.evaluate(run_log, episode)
            run_log.validator_result = _jsonable(validation)

            self._run_history.append((run_log, validation))
            self._last_run_log = run_log
            self._last_patch = None

            governor_decision = self.governor.observe(run_log, validation)
            if governor_decision.action in {"NO_UPDATE", "DEFER"}:
                self._governor_action_history.append(governor_decision.action)
                self._update_history.append(0)
                action = f"governor_{governor_decision.action.lower()}"
                if validation.success and self.failure_replay_store.contains(episode.episode_id):
                    self.failure_replay_store.record_success(episode.episode_id)
                    self.failure_replay_store.try_clear(episode.episode_id)
                return

            diagnosis = self.blame_assigner.diagnose(run_log, validation)
            proposal = self.update_planner.plan(diagnosis, run_log)
            governor_decision = self.governor.decide_update(
                run_log,
                validation,
                diagnosis,
                proposal,
            )
            self._governor_action_history.append(governor_decision.action)

            if governor_decision.action != "UPDATE":
                self._update_history.append(0)
                action = f"governor_{governor_decision.action.lower()}"
                self.failure_replay_store.add_or_refresh(run_log)
                return

            self._update_history.append(proposal.update_level)
            patch = self.patch_builder.build(proposal, run_log, governor_decision)
            self._last_patch = patch
            bundle = self.retention_suite.select_shadow_bundle(run_log.episode_id)
            shadow_result = self.shadow_evaluator.test(
                patch,
                bundle,
                self.live_state,
                self.episode_store,
            )
            if shadow_result.promoted:
                patch_result = self.promotion_manager.promote(
                    patch,
                    shadow_result,
                    run_log.episode_id,
                    self.live_state,
                    self.failure_replay_store,
                    self.shadow_evaluator,
                    self.episode_store,
                )
                self.retention_suite.on_patch_result(bundle, patch_result)
                action = "patch_promoted"
            else:
                patch_result = shadow_result
                self.retention_suite.on_patch_result(bundle, patch_result)
                self.promotion_manager.reject(
                    patch,
                    shadow_result.notes,
                    self.failure_replay_store,
                    run_log,
                )
                action = "patch_rejected"
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.audit_logger.write_step(
                step_index=self.step_index,
                action=action,
                episode=episode,
                run_log=run_log,
                validation=validation,
                diagnosis=diagnosis,
                proposal=proposal,
                patch=patch,
                patch_result=patch_result,
                governor_decision=governor_decision,
                status=self.get_status(),
                error_message=error_message,
            )

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.run_one_step()

    def get_status(self) -> dict:
        total = len(self._run_history)
        successes = sum(1 for _run, validation in self._run_history if validation.success)
        projector_calls = sum(
            1
            for run, _validation in self._run_history
            if run.cost_metrics.get("projector_called", False)
        )
        patch_count = len(self.promotion_manager.patch_history)
        rejected_count = len(self.promotion_manager.rejected_patch_history)
        return {
            "metrics": {
                "success_rate": successes / total if total else 0.0,
                "updates_per_episode": sum(1 for level in self._update_history if level > 0) / total
                if total
                else 0.0,
                "projector_calls_per_episode": projector_calls / total if total else 0.0,
                "no_update_rate": sum(1 for level in self._update_history if level == 0) / total
                if total
                else 0.0,
                "defer_rate": self._governor_action_history.count("DEFER") / total
                if total
                else 0.0,
                "patch_promotion_rate": patch_count / max(1, patch_count + rejected_count),
            },
            "governor": self.governor.metrics(),
            "failure_replay": self.failure_replay_store.metrics(),
            "retention_suite": self.retention_suite.snapshot(),
            "live_state_keys": list(self.live_state.keys()),
            "patch_history_count": len(self.promotion_manager.patch_history),
            "rejected_patch_count": len(self.promotion_manager.rejected_patch_history),
        }
