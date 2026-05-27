from dataclasses import dataclass
from uuid import uuid4
from datetime import datetime, timezone
import copy
import random
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from abc import ABC, abstractmethod

from pipeline import (
    Perception,
    CueEncoder,
    DMF,
    Binding,
    ContextOp,
    Interference,
    Lucidity,
    Projector,
    Decoder,
)


@dataclass
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


@dataclass
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


@dataclass
class ValidationResult:
    success: bool
    score: float
    failure_signals: list
    expected_state: Any
    confidence: float


@dataclass
class FailureDiagnosis:
    primary_module: str
    secondary_modules: list
    blame_confidence: float
    evidence: dict
    recommended_update_level: int


@dataclass
class UpdateProposal:
    patch_type: str
    target_objects: list
    update_level: int
    expected_fix: str
    risk_level: str
    shadow_test_bundle: list


@dataclass
class Patch:
    patch_id: str
    patch_type: str
    target_objects: list
    delta: dict
    reason: str
    update_level: int
    created_at: str


@dataclass
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


@dataclass
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
    return datetime.now(timezone.utc).isoformat()


class BaseValidator(ABC):
    @abstractmethod
    def evaluate(self, run_log: RunLog, episode: TrainingEpisode) -> ValidationResult:
        raise NotImplementedError


class ExactMatchValidator(BaseValidator):
    def evaluate(self, run_log: RunLog, episode: TrainingEpisode) -> ValidationResult:
        success = run_log.decoder_output == episode.expected_output
        return ValidationResult(
            success=success,
            score=1.0 if success else 0.0,
            failure_signals=[] if success else ["exact_match_failed"],
            expected_state=episode.expected_output,
            confidence=1.0,
        )


class UnitTestValidator(BaseValidator):
    def evaluate(self, run_log: RunLog, episode: TrainingEpisode) -> ValidationResult:
        if not callable(episode.expected_output):
            return ValidationResult(
                success=False,
                score=0.0,
                failure_signals=["expected_output_not_callable"],
                expected_state=None,
                confidence=1.0,
            )
        try:
            success = bool(episode.expected_output(run_log.decoder_output))
        except Exception as exc:
            return ValidationResult(
                success=False,
                score=0.0,
                failure_signals=[f"unit_test_exception:{type(exc).__name__}"],
                expected_state="callable(expected_output)",
                confidence=1.0,
            )
        return ValidationResult(
            success=success,
            score=1.0 if success else 0.0,
            failure_signals=[] if success else ["unit_test_failed"],
            expected_state="callable(expected_output)",
            confidence=1.0,
        )


class SelfConsistencyValidator(BaseValidator):
    def evaluate(self, run_log: RunLog, episode: TrainingEpisode) -> ValidationResult:
        success = run_log.lucidity_margin > 0.6 and run_log.decoder_output is not None
        return ValidationResult(
            success=success,
            score=1.0 if success else max(0.0, min(1.0, run_log.lucidity_margin)),
            failure_signals=[] if success else ["self_consistency_failed"],
            expected_state=episode.expected_output,
            confidence=0.8,
        )


class DecoderFaithfulnessValidator(BaseValidator):
    def _flatten(self, obj: Any, prefix: str = "") -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else str(key)
                flat.update(self._flatten(value, new_prefix))
            return flat
        if isinstance(obj, list):
            for idx, value in enumerate(obj):
                new_prefix = f"{prefix}[{idx}]"
                flat.update(self._flatten(value, new_prefix))
            return flat
        flat[prefix or "value"] = obj
        return flat

    def evaluate(self, run_log: RunLog, episode: TrainingEpisode) -> ValidationResult:
        basin_flat = self._flatten(run_log.basin_assemblies if run_log.basin_assemblies is not None else {})
        output_flat = self._flatten(run_log.decoder_output if run_log.decoder_output is not None else {})
        if not basin_flat:
            return ValidationResult(
                success=False,
                score=0.0,
                failure_signals=["empty_basin_assembly"],
                expected_state=run_log.basin_assemblies,
                confidence=0.7,
            )
        total = 0
        match = 0
        for key, value in basin_flat.items():
            total += 1
            if output_flat.get(key) == value:
                match += 1
        score = (match / total) if total else 0.0
        return ValidationResult(
            success=score >= 0.8,
            score=score,
            failure_signals=[] if score >= 0.8 else ["decoder_faithfulness_failed"],
            expected_state=run_log.basin_assemblies,
            confidence=0.75,
        )


class ValidatorFactory:
    def __init__(self) -> None:
        self._validators = {
            "exact_match": ExactMatchValidator(),
            "unit_test": UnitTestValidator(),
            "self_consistency": SelfConsistencyValidator(),
            "decoder_faithfulness": DecoderFaithfulnessValidator(),
        }

    def get(self, validator_type: str) -> BaseValidator:
        return self._validators.get(validator_type, SelfConsistencyValidator())


class RunExecutor:
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
    ):
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
    def _as_dict(value: Any, key: str) -> dict:
        if isinstance(value, dict):
            return value
        return {key: value}

    @staticmethod
    def _derive_lucidity_decision(margin: float) -> str:
        if margin >= 0.75:
            return "commit"
        if margin < 0.40:
            return "reject"
        return "test_consequence"

    def _run_from_states(self, episode: TrainingEpisode, state: dict) -> RunLog:
        evidence_graph = state.get("evidence_graph")
        if evidence_graph is None:
            evidence_graph = self.perception.run(episode.raw_input)
        evidence_graph = self._as_dict(evidence_graph, "evidence")

        cue_cloud = state.get("cue_cloud")
        if cue_cloud is None:
            cue_cloud = self.cue_encoder.run(evidence_graph)
        cue_cloud = self._as_dict(cue_cloud, "cue")

        dmf_out = state.get("dmf_out")
        if dmf_out is None:
            dmf_out = self.dmf.run(cue_cloud)
        dmf_out = self._as_dict(dmf_out, "dmf")
        active_traces = state.get("active_traces", dmf_out.get("active_traces", []))
        trace_clusters = state.get("trace_clusters", dmf_out.get("trace_clusters", []))

        binding_in = {
            "dmf_out": dmf_out,
            "active_traces": active_traces,
            "trace_clusters": trace_clusters,
            "episode_context": episode.context,
        }
        binding_out = state.get("binding_out")
        if binding_out is None:
            binding_out = self.binding.run(binding_in)
        binding_out = self._as_dict(binding_out, "binding")
        candidate_bindings = state.get("candidate_bindings", binding_out.get("candidate_bindings", []))

        context_in = {
            "candidate_bindings": candidate_bindings,
            "context": episode.context,
            "constraints": episode.constraints,
        }
        context_out = state.get("context_out")
        if context_out is None:
            context_out = self.context_op.run(context_in)
        context_out = self._as_dict(context_out, "context")
        context_frames = state.get("context_frames", context_out.get("context_frames", []))
        scoped_trace_assignments = state.get(
            "scoped_trace_assignments", context_out.get("scoped_trace_assignments", {})
        )

        interference_in = {
            "context_frames": context_frames,
            "active_traces": active_traces,
            "candidate_bindings": candidate_bindings,
        }
        interference_out = state.get("interference_out")
        if interference_out is None:
            interference_out = self.interference.run(interference_in)
        interference_out = self._as_dict(interference_out, "interference")
        interference_edges = state.get("interference_edges", interference_out.get("interference_edges", []))
        active_basins = state.get("active_basins", interference_out.get("active_basins", []))
        basin_assemblies = state.get("basin_assemblies", interference_out.get("basin_assemblies", {}))

        lucidity_in = {
            "active_basins": active_basins,
            "interference_edges": interference_edges,
            "task_intent": episode.task_intent,
        }
        lucidity_out = state.get("lucidity_out")
        if lucidity_out is None:
            lucidity_out = self.lucidity.run(lucidity_in)

        lucidity_decision = "reject"
        lucidity_margin = 0.0
        lucidity_features = {}
        if isinstance(lucidity_out, tuple) and len(lucidity_out) >= 2:
            lucidity_decision = str(lucidity_out[0])
            lucidity_margin = float(lucidity_out[1])
        else:
            lucidity_margin = float(lucidity_out) if isinstance(lucidity_out, (int, float)) else 0.0
            lucidity_decision = self._derive_lucidity_decision(lucidity_margin)
        if lucidity_decision not in {"commit", "reject", "test_consequence"}:
            lucidity_decision = self._derive_lucidity_decision(lucidity_margin)
        lucidity_features = {"raw_output": lucidity_out}

        projection_result = None
        projector_called = False
        if lucidity_decision == "test_consequence":
            projector_called = True
            projection_result = self.projector.run(
                {"basins": active_basins, "assemblies": basin_assemblies, "constraints": episode.constraints}
            )

        decoder_in = {
            "decision": lucidity_decision,
            "margin": lucidity_margin,
            "basin_assemblies": basin_assemblies,
            "projection_result": projection_result,
        }
        decoder_output = self.decoder.run(decoder_in)

        cost_metrics = {
            "stages_run": 9 if projector_called else 8,
            "projector_called": projector_called,
            "traces_activated": len(active_traces),
            "basins_activated": len(active_basins),
        }

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
            lucidity_features=lucidity_features,
            lucidity_decision=lucidity_decision,
            lucidity_margin=lucidity_margin,
            projection_result=projection_result,
            decoder_output=decoder_output,
            validator_result={},
            cost_metrics=cost_metrics,
        )

    def run(self, episode: TrainingEpisode, mode: str) -> RunLog:
        _ = mode
        return self._run_from_states(episode, {})


class BlameAssigner:
    @staticmethod
    def _has_expected_evidence(evidence_graph: dict) -> bool:
        if not evidence_graph:
            return False
        expected_keys = {"entities", "events", "relations"}
        if any(key in evidence_graph for key in expected_keys):
            return True
        return len(evidence_graph.keys()) > 0

    @staticmethod
    def _top_basin_id(active_basins: list) -> Optional[str]:
        if not active_basins:
            return None
        first = active_basins[0]
        if isinstance(first, dict):
            return str(first.get("basin_id", first.get("id", ""))) or None
        return str(first)

    def diagnose(self, run_log: RunLog, validation: ValidationResult) -> FailureDiagnosis:
        if not self._has_expected_evidence(run_log.evidence_graph):
            return FailureDiagnosis("perception", [], 0.9, {"rule": 1}, 9)
        if len(run_log.active_traces) < 1:
            return FailureDiagnosis("cue_encoder_or_DMF", ["cue_encoder", "dmf"], 0.85, {"rule": 2}, 2)
        if not run_log.candidate_bindings:
            return FailureDiagnosis("binding", [], 0.85, {"rule": 3}, 4)
        if not run_log.context_frames or (
            isinstance(run_log.scoped_trace_assignments, dict) and not run_log.scoped_trace_assignments
        ):
            return FailureDiagnosis("context_op", [], 0.8, {"rule": 4}, 5)

        expected_basin = None
        if isinstance(validation.expected_state, dict):
            expected_basin = validation.expected_state.get("basin_id")
        top_basin = self._top_basin_id(run_log.active_basins)
        if run_log.context_frames and expected_basin and top_basin and expected_basin != top_basin:
            return FailureDiagnosis(
                "interference_or_basin", ["interference", "basin"], 0.75, {"rule": 5}, 3
            )
        if (
            expected_basin
            and top_basin == expected_basin
            and run_log.lucidity_margin >= 0.7
            and run_log.lucidity_decision == "reject"
        ):
            return FailureDiagnosis("lucidity_too_strict", ["lucidity"], 0.7, {"rule": 6}, 8)
        if run_log.lucidity_decision == "commit" and not validation.success and validation.failure_signals:
            return FailureDiagnosis("lucidity_too_loose", ["lucidity"], 0.7, {"rule": 7}, 8)
        if run_log.decoder_output is None:
            return FailureDiagnosis("decoder", [], 0.85, {"rule": 8}, 9)
        if isinstance(run_log.basin_assemblies, dict) and isinstance(run_log.decoder_output, dict):
            basin_keys = set(run_log.basin_assemblies.keys())
            if basin_keys and any(
                key in basin_keys and run_log.decoder_output.get(key) != run_log.basin_assemblies.get(key)
                for key in basin_keys
            ):
                return FailureDiagnosis("decoder", [], 0.8, {"rule": 8}, 9)
        return FailureDiagnosis("unknown", [], 0.2, {"rule": 9}, 0)


class DiagnosticForcer:
    def __init__(self, executor: RunExecutor, validator_factory: ValidatorFactory):
        self.executor = executor
        self.validator_factory = validator_factory

    def _evaluate_forced(self, episode: TrainingEpisode, state: dict) -> ValidationResult:
        forced_log = self.executor._run_from_states(episode, state)
        validator = self.validator_factory.get(episode.validator_type)
        return validator.evaluate(forced_log, episode)

    def run_forced_tests(self, run_log: RunLog, episode: TrainingEpisode) -> FailureDiagnosis:
        gold_states = episode.metadata.get("gold_states", {}) if isinstance(episode.metadata, dict) else {}
        tests: List[Tuple[str, dict, str, int]] = [
            (
                "perception",
                {"evidence_graph": gold_states.get("evidence_graph", run_log.evidence_graph)},
                "A",
                9,
            ),
            (
                "cue_encoder_or_DMF",
                {"active_traces": gold_states.get("active_traces", run_log.active_traces)},
                "B",
                2,
            ),
            (
                "binding",
                {"candidate_bindings": gold_states.get("candidate_bindings", run_log.candidate_bindings)},
                "C",
                4,
            ),
            (
                "context_op",
                {"context_frames": gold_states.get("context_frames", run_log.context_frames)},
                "D",
                5,
            ),
            (
                "interference_or_basin",
                {"active_basins": gold_states.get("active_basins", run_log.active_basins)},
                "E",
                3,
            ),
        ]
        for module, state, test_name, level in tests:
            validation = self._evaluate_forced(episode, state)
            if validation.success:
                return FailureDiagnosis(
                    primary_module=module,
                    secondary_modules=[],
                    blame_confidence=0.75,
                    evidence={"forced_test": test_name, "success": True},
                    recommended_update_level=level,
                )

        forced_decoder_state = {
            "active_basins": gold_states.get("active_basins", run_log.active_basins),
            "basin_assemblies": gold_states.get("basin_assemblies", run_log.basin_assemblies),
        }
        validation_f = self._evaluate_forced(episode, forced_decoder_state)
        if not validation_f.success:
            return FailureDiagnosis(
                primary_module="decoder",
                secondary_modules=[],
                blame_confidence=0.8,
                evidence={"forced_test": "F", "success": False},
                recommended_update_level=9,
            )
        return FailureDiagnosis(
            primary_module="unknown",
            secondary_modules=[],
            blame_confidence=0.3,
            evidence={"forced_test": "none"},
            recommended_update_level=0,
        )


class UpdatePlanner:
    UPDATE_LEVELS = {
        0: "no_update",
        1: "counter_stat_update",
        2: "trace_reinforcement",
        3: "interference_edge_update",
        4: "binding_affordance_update",
        5: "context_scope_update",
        6: "basin_link_update",
        7: "basin_split_merge",
        8: "tiny_policy_update",
        9: "gradient_patch",
        10: "global_recalibration",
    }

    MODULE_TO_LEVEL = {
        "perception": 9,
        "cue_encoder_or_DMF": 2,
        "binding": 4,
        "context_op": 5,
        "interference_or_basin": 3,
        "lucidity_too_strict": 8,
        "lucidity_too_loose": 8,
        "decoder": 9,
        "unknown": 0,
    }

    LEVEL_TO_PATCH = {
        0: "NoPatch",
        1: "TracePatch",
        2: "TracePatch",
        3: "InterferencePatch",
        4: "BindingPatch",
        5: "ContextPatch",
        6: "BasinPatch",
        7: "BasinPatch",
        8: "LucidityPatch",
        9: "DecoderPatch",
        10: "GlobalPatch",
    }

    def _target_objects(self, diagnosis: FailureDiagnosis, run_log: RunLog) -> list:
        candidates = []
        candidates.extend([str(x) for x in run_log.active_traces[:2]])
        basin_ids = []
        for basin in run_log.active_basins[:2]:
            if isinstance(basin, dict):
                basin_ids.append(str(basin.get("basin_id", basin.get("id", "unknown_basin"))))
            else:
                basin_ids.append(str(basin))
        candidates.extend(basin_ids)
        if not candidates:
            candidates = [run_log.episode_id]
        return candidates[:2]

    def plan(self, diagnosis: FailureDiagnosis, run_log: RunLog) -> UpdateProposal:
        level = self.MODULE_TO_LEVEL.get(diagnosis.primary_module, diagnosis.recommended_update_level)
        if diagnosis.recommended_update_level > 0:
            level = min(level, diagnosis.recommended_update_level)

        if level == 10 and not run_log.cost_metrics.get("force_global", False):
            level = 9
        risk_level = "low"
        if level >= 8:
            risk_level = "high"
        elif level >= 5:
            risk_level = "medium"

        if risk_level == "high" and diagnosis.blame_confidence < 0.6:
            level = max(1, level - 2)
            risk_level = "medium" if level >= 5 else "low"
        if level == 10 and not run_log.cost_metrics.get("force_global", False):
            level = 9

        patch_type = self.LEVEL_TO_PATCH.get(level, "NoPatch")
        targets = self._target_objects(diagnosis, run_log)
        if level == 0:
            patch_type = "NoPatch"
            targets = [run_log.episode_id]

        return UpdateProposal(
            patch_type=patch_type,
            target_objects=targets,
            update_level=level,
            expected_fix=f"repair_{diagnosis.primary_module}",
            risk_level=risk_level,
            shadow_test_bundle=[run_log.episode_id],
        )


class PatchBuilder:
    @staticmethod
    def _pick_trace(run_log: RunLog) -> str:
        return str(run_log.active_traces[0]) if run_log.active_traces else "trace_unknown"

    @staticmethod
    def _pick_basin(run_log: RunLog) -> str:
        if not run_log.active_basins:
            return "basin_unknown"
        basin = run_log.active_basins[0]
        if isinstance(basin, dict):
            return str(basin.get("basin_id", basin.get("id", "basin_unknown")))
        return str(basin)

    @staticmethod
    def _pick_context(run_log: RunLog) -> str:
        if not run_log.context_frames:
            return "context_default"
        frame = run_log.context_frames[0]
        if isinstance(frame, dict):
            return str(frame.get("frame_id", frame.get("id", "context_default")))
        return str(frame)

    def build(self, proposal: UpdateProposal, run_log: RunLog) -> Patch:
        patch_id = str(uuid4())
        created_at = _utc_now_iso()
        patch_type = proposal.patch_type
        delta: Dict[str, Any]

        if patch_type == "TracePatch":
            delta = {
                "trace_id": self._pick_trace(run_log),
                "delta_strength": 0.05,
                "reason": proposal.expected_fix,
            }
        elif patch_type == "InterferencePatch":
            delta = {
                "edge": (self._pick_trace(run_log), self._pick_basin(run_log)),
                "scope": self._pick_context(run_log),
                "delta_positive": 0.04,
                "delta_negative": 0.0,
            }
        elif patch_type == "ContextPatch":
            delta = {
                "pattern": list(run_log.evidence_graph.keys())[:5],
                "reduce": "global_leak",
                "increase": "local_scope_bridge",
                "weight_delta": 0.05,
            }
        elif patch_type == "BindingPatch":
            delta = {
                "frame_id": self._pick_context(run_log),
                "affordance_delta": 0.05,
            }
        elif patch_type == "BasinPatch":
            delta = {
                "basin_id": self._pick_basin(run_log),
                "operation": "link_adjust",
                "parameters": {"delta": 0.05},
            }
        elif patch_type == "LucidityPatch":
            if diagnosis_module := proposal.expected_fix:
                if diagnosis_module.endswith("lucidity_too_strict"):
                    direction = "looser"
                elif diagnosis_module.endswith("lucidity_too_loose"):
                    direction = "stricter"
                else:
                    direction = "stricter"
            else:
                direction = "stricter"
            delta = {"threshold_delta": 0.03, "direction": direction}
        elif patch_type == "DecoderPatch":
            delta = {"output_pattern": "default", "correction": proposal.expected_fix}
        elif patch_type == "PerceptionPatch":
            delta = {"missing_evidence_type": "entity_or_event", "sensitivity_delta": 0.04}
        elif patch_type == "CueEncoderPatch":
            delta = {"cue_pattern": "low_activation", "activation_delta": 0.04}
        elif patch_type == "GlobalPatch":
            delta = {"recalibrate": True, "scope": "global"}
        else:
            delta = {"noop": True}

        return Patch(
            patch_id=patch_id,
            patch_type=patch_type,
            target_objects=proposal.target_objects,
            delta=delta,
            reason=proposal.expected_fix,
            update_level=proposal.update_level,
            created_at=created_at,
        )


class RetentionSuiteManager:
    PHASE_BUDGETS = {1: 500, 2: 200, 3: 200, 4: 100, 5: 65}
    PHASE_BUDGETS_HIGH_RISK = {5: 120}

    def __init__(self):
        self.canary_episodes: list = []
        self.suite_version: str = "v1"
        self.family_coverage_map: dict = {}
        self.last_audit_at: str = ""
        self.regression_log: list = []
        self.blocked_families: set = set()

    def select_shadow_bundle(self, patch: Patch, target_episode_id: str, phase: int) -> list:
        max_budget = self.PHASE_BUDGETS_HIGH_RISK.get(phase, self.PHASE_BUDGETS.get(phase, 65))
        if phase in self.PHASE_BUDGETS_HIGH_RISK and patch.update_level < 8:
            max_budget = self.PHASE_BUDGETS.get(phase, max_budget)
        if max_budget < 1:
            max_budget = 1

        selected = [target_episode_id]
        if target_episode_id in self.canary_episodes:
            selected = [target_episode_id]

        sampled = self.sample(max_budget - 1)
        for episode_id in sampled:
            if episode_id != target_episode_id and episode_id not in selected:
                selected.append(episode_id)
            if len(selected) >= max_budget:
                break

        if len(selected) > max_budget:
            raise ValueError(f"Shadow bundle exceeds phase budget {max_budget}")
        return selected

    def add_episode(self, episode: TrainingEpisode, reason: str):
        _ = reason
        if episode.episode_id not in self.canary_episodes:
            self.canary_episodes.append(episode.episode_id)
        family = str(episode.metadata.get("task_family", episode.task_intent)) if episode.metadata else episode.task_intent
        family_set = self.family_coverage_map.setdefault(family, set())
        family_set.add(episode.episode_id)
        if family in self.blocked_families:
            self.blocked_families.remove(family)

    def on_promote(self, patch_result: PatchResult):
        if not patch_result.retention_passed:
            self.regression_log.append(
                {
                    "patch_id": patch_result.patch_id,
                    "time": _utc_now_iso(),
                    "notes": patch_result.notes,
                }
            )

    def on_corpus_growth(self, new_family: str):
        if new_family not in self.family_coverage_map:
            self.blocked_families.add(new_family)

    def audit_suite_vs_corpus(self, corpus_sample: list):
        now = datetime.now(timezone.utc)
        self.last_audit_at = now.isoformat()
        sample_families = defaultdict(list)
        for episode in corpus_sample:
            family = str(episode.metadata.get("task_family", episode.task_intent)) if episode.metadata else episode.task_intent
            sample_families[family].append(episode.episode_id)
            if family not in self.family_coverage_map or len(self.family_coverage_map.get(family, [])) == 0:
                self.blocked_families.add(family)

        # Demote stale canaries older than 90 days only if family has alternate entries.
        canary_ages = []
        for episode in corpus_sample:
            if episode.episode_id in self.canary_episodes:
                created_at = episode.metadata.get("created_at") if episode.metadata else None
                if created_at:
                    try:
                        age_days = (now - datetime.fromisoformat(created_at)).days
                        canary_ages.append((episode, age_days))
                    except ValueError:
                        continue
        for episode, age_days in canary_ages:
            family = str(episode.metadata.get("task_family", episode.task_intent)) if episode.metadata else episode.task_intent
            family_entries = self.family_coverage_map.get(family, set())
            if age_days > 90 and len(family_entries) > 1 and episode.episode_id in self.canary_episodes:
                self.canary_episodes.remove(episode.episode_id)
                family_entries.discard(episode.episode_id)

    def sample(self, budget: int) -> list:
        if budget <= 0:
            return []
        if not self.canary_episodes:
            return []

        selected: List[str] = []
        family_keys = list(self.family_coverage_map.keys())
        random.shuffle(family_keys)
        for family in family_keys:
            family_ids = list(self.family_coverage_map.get(family, []))
            if family_ids and len(selected) < budget:
                selected.append(random.choice(family_ids))
        if len(selected) < budget:
            remaining = [eid for eid in self.canary_episodes if eid not in selected]
            random.shuffle(remaining)
            selected.extend(remaining[: max(0, budget - len(selected))])
        return selected[:budget]


class ShadowEvaluator:
    def __init__(self, executor: RunExecutor, validator_factory: ValidatorFactory):
        self.executor = executor
        self.validator_factory = validator_factory

    @staticmethod
    def _safe_score(validation: ValidationResult) -> float:
        return float(validation.score) if isinstance(validation.score, (int, float)) else 0.0

    @staticmethod
    def _extract_false_commit(run_log: RunLog, validation: ValidationResult) -> bool:
        return run_log.lucidity_decision == "commit" and not validation.success

    @staticmethod
    def _extract_basin_pollution(run_log: RunLog) -> float:
        if not run_log.active_basins:
            return 0.0
        polluted = 0
        for basin in run_log.active_basins:
            if isinstance(basin, dict) and basin.get("polluted", False):
                polluted += 1
        return polluted / max(1, len(run_log.active_basins))

    @staticmethod
    def _apply_patch_to_state(patch: Patch, state: dict):
        patches = state.setdefault("patches", {})
        patches[patch.patch_id] = copy.deepcopy(patch.delta)
        state["last_patch_type"] = patch.patch_type

    def _run_episode_with_state(
        self, episode: TrainingEpisode, state: dict
    ) -> Tuple[RunLog, ValidationResult]:
        run_log = self.executor._run_from_states(episode, state.get("injected", {}))
        validator = self.validator_factory.get(episode.validator_type)
        validation = validator.evaluate(run_log, episode)
        run_log.validator_result = vars(validation)
        return run_log, validation

    def test(self, patch: Patch, bundle: list, live_state: dict, episode_store: dict) -> PatchResult:
        shadow_state = copy.deepcopy(live_state)
        self._apply_patch_to_state(patch, shadow_state)

        if not bundle:
            return PatchResult(
                patch_id=patch.patch_id,
                fixed_target=False,
                retention_passed=False,
                cost_delta=0.0,
                quality_delta=0.0,
                promoted=False,
                episode_shadow_passed=False,
                retention_suite_version="v1",
                notes="empty_bundle",
            )

        target_id = bundle[0]
        shadow_scores = {}
        live_scores = {}
        shadow_costs = []
        live_costs = []
        shadow_false_commits = 0
        live_false_commits = 0
        shadow_pollution = 0.0
        live_pollution = 0.0

        for episode_id in bundle:
            episode = episode_store[episode_id]
            shadow_run, shadow_validation = self._run_episode_with_state(episode, shadow_state)
            live_run, live_validation = self._run_episode_with_state(episode, live_state)
            shadow_scores[episode_id] = self._safe_score(shadow_validation)
            live_scores[episode_id] = self._safe_score(live_validation)
            shadow_costs.append(float(shadow_run.cost_metrics.get("stages_run", 0)))
            live_costs.append(float(live_run.cost_metrics.get("stages_run", 0)))
            shadow_false_commits += 1 if self._extract_false_commit(shadow_run, shadow_validation) else 0
            live_false_commits += 1 if self._extract_false_commit(live_run, live_validation) else 0
            shadow_pollution += self._extract_basin_pollution(shadow_run)
            live_pollution += self._extract_basin_pollution(live_run)

        fixed_target = shadow_scores.get(target_id, 0.0) > live_scores.get(target_id, 0.0)

        retention_passed = True
        for eid in bundle[1:]:
            base = live_scores.get(eid, 0.0)
            new = shadow_scores.get(eid, 0.0)
            if base - new > 0.01:
                retention_passed = False
                break

        avg_live_cost = (sum(live_costs) / len(live_costs)) if live_costs else 0.0
        avg_shadow_cost = (sum(shadow_costs) / len(shadow_costs)) if shadow_costs else 0.0
        cost_delta = ((avg_shadow_cost - avg_live_cost) / avg_live_cost) if avg_live_cost > 0 else 0.0
        cost_ok = cost_delta <= 0.05

        denom = max(1, len(bundle))
        live_false_commit_rate = live_false_commits / denom
        shadow_false_commit_rate = shadow_false_commits / denom
        lucidity_ok = shadow_false_commit_rate <= live_false_commit_rate

        live_pollution_rate = live_pollution / denom
        shadow_pollution_rate = shadow_pollution / denom
        basin_ok = shadow_pollution_rate <= live_pollution_rate

        promoted = all([fixed_target, retention_passed, cost_ok, lucidity_ok, basin_ok])
        failed_conditions = []
        if not fixed_target:
            failed_conditions.append("fixed_target")
        if not retention_passed:
            failed_conditions.append("retention")
        if not cost_ok:
            failed_conditions.append("cost")
        if not lucidity_ok:
            failed_conditions.append("lucidity_calibration")
        if not basin_ok:
            failed_conditions.append("basin_pollution")
        notes = "ok" if promoted else ",".join(failed_conditions)

        quality_delta = shadow_scores.get(target_id, 0.0) - live_scores.get(target_id, 0.0)
        return PatchResult(
            patch_id=patch.patch_id,
            fixed_target=fixed_target,
            retention_passed=retention_passed,
            cost_delta=cost_delta,
            quality_delta=quality_delta,
            promoted=promoted,
            episode_shadow_passed=False,
            retention_suite_version="v1",
            notes=notes,
        )

    def run_episode_shadow(self, episode_id: str, live_state_after_promote: dict, episode_store: dict) -> bool:
        episode = episode_store[episode_id]
        run_log = self.executor._run_from_states(episode, live_state_after_promote.get("injected", {}))
        validator = self.validator_factory.get(episode.validator_type)
        validation = validator.evaluate(run_log, episode)
        return validation.success


class FailureReplayStore:
    def __init__(self):
        self.entries: dict = {}
        self._cleared_without_shadow_pass: int = 0

    def add_or_refresh(self, run_log: RunLog):
        now = _utc_now_iso()
        run_log_id = f"{run_log.episode_id}:{now}"
        if run_log.episode_id not in self.entries:
            self.entries[run_log.episode_id] = FailureReplayEntry(
                episode_id=run_log.episode_id,
                run_log_id_last_failure=run_log_id,
                patch_ids_applied=[],
                shadow_passed=False,
                consecutive_successes=0,
                replay_priority=1,
                entered_at=now,
                last_attempt_at=now,
            )
        else:
            entry = self.entries[run_log.episode_id]
            entry.run_log_id_last_failure = run_log_id
            entry.last_attempt_at = now

    def record_success(self, episode_id: str):
        if episode_id in self.entries:
            self.entries[episode_id].consecutive_successes += 1

    def on_patch_promoted(self, episode_id: str, patch_id: str):
        if episode_id not in self.entries:
            return
        entry = self.entries[episode_id]
        entry.patch_ids_applied.append(patch_id)
        entry.consecutive_successes = 0
        entry.shadow_passed = False
        entry.last_attempt_at = _utc_now_iso()

    def on_episode_shadow_passed(self, episode_id: str):
        if episode_id in self.entries:
            self.entries[episode_id].shadow_passed = True

    def on_patch_rejected(self, episode_id: str):
        if episode_id in self.entries:
            entry = self.entries[episode_id]
            entry.replay_priority = min(10, entry.replay_priority + 1)
            entry.last_attempt_at = _utc_now_iso()

    def try_clear(self, episode_id: str) -> bool:
        if episode_id not in self.entries:
            return False
        entry = self.entries[episode_id]
        can_clear = (
            len(entry.patch_ids_applied) > 0
            and entry.shadow_passed is True
            and entry.consecutive_successes >= 3
        )
        if can_clear:
            if not entry.shadow_passed:
                self._cleared_without_shadow_pass += 1
                raise RuntimeError("attempted clear without shadow pass")
            del self.entries[episode_id]
            return True
        return False

    def contains(self, episode_id: str) -> bool:
        return episode_id in self.entries

    def get_priority_weight(self, episode_id: str) -> float:
        if episode_id not in self.entries:
            return 1.0
        entry = self.entries[episode_id]
        age_hours = 1.0
        try:
            age_hours = max(
                1.0,
                (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(entry.entered_at.replace("Z", "+00:00"))
                ).total_seconds()
                / 3600.0,
            )
        except ValueError:
            age_hours = 1.0
        age_term = min(3.0, 1.0 + age_hours / 48.0)
        priority_term = 1.0 + (entry.replay_priority / 10.0)
        streak_term = 1.0 + max(0.0, (3 - entry.consecutive_successes) * 0.2)
        return age_term * priority_term * streak_term

    def metrics(self) -> dict:
        stuck = 0
        for entry in self.entries.values():
            if entry.replay_priority >= 5:
                stuck += 1
        return {
            "failure_replay_queue_depth": len(self.entries),
            "episodes_stuck_past_5_attempts": stuck,
            "cleared_without_shadow_pass": self._cleared_without_shadow_pass,
        }


class PromotionManager:
    def __init__(self):
        self.patch_history: list = []
        self.rejected_patch_history: list = []

    @staticmethod
    def _apply_patch_delta(live_state: dict, patch: Patch):
        live_state.setdefault("applied_patches", {})[patch.patch_id] = copy.deepcopy(patch.delta)
        live_state["last_patch"] = patch.patch_type

    def apply(
        self,
        patch: Patch,
        live_state: dict,
        failure_replay_store: FailureReplayStore,
        shadow_evaluator: ShadowEvaluator,
        episode_store: dict,
    ) -> PatchResult:
        self._apply_patch_delta(live_state, patch)
        self.patch_history.append(patch)

        episode_shadow_passed_any = False
        for target_episode_id in patch.target_objects:
            if target_episode_id in episode_store:
                failure_replay_store.on_patch_promoted(target_episode_id, patch.patch_id)
                passed = shadow_evaluator.run_episode_shadow(target_episode_id, live_state, episode_store)
                if passed:
                    failure_replay_store.on_episode_shadow_passed(target_episode_id)
                    episode_shadow_passed_any = True

        return PatchResult(
            patch_id=patch.patch_id,
            fixed_target=True,
            retention_passed=True,
            cost_delta=0.0,
            quality_delta=0.0,
            promoted=True,
            episode_shadow_passed=episode_shadow_passed_any,
            retention_suite_version="v1",
            notes="promoted",
        )

    def reject(self, patch: Patch, reason: str, failure_replay_store: FailureReplayStore, run_log: RunLog):
        self.rejected_patch_history.append({"patch": patch, "reason": reason})
        failure_replay_store.add_or_refresh(run_log)
        failure_replay_store.on_patch_rejected(run_log.episode_id)


class EpisodeScheduler:
    QUEUE_NAMES = [
        "real_text_chat",
        "synthetic_structure",
        "verifiable_task",
        "multimodal",
        "agent_episode",
        "failure_replay",
        "retention_test",
    ]
    MAX_QUEUE_SHARE = 0.40
    RETENTION_TEST_MAX_GAP = 50

    def __init__(self, failure_replay_store: FailureReplayStore):
        self.queues: dict = {name: [] for name in self.QUEUE_NAMES}
        self.failure_replay_store = failure_replay_store
        self.module_failure_rates: dict = {}
        self.sample_history: list = []
        self.steps_since_retention_test: int = 0
        self._episode_to_queue: dict = {}

    def _queue_fraction_last100(self, queue_name: str) -> float:
        recent = self.sample_history[-100:]
        if not recent:
            return 0.0
        count = sum(1 for q in recent if q == queue_name)
        return count / len(recent)

    def _episode_weight(self, queue_name: str, episode: TrainingEpisode) -> float:
        if self._queue_fraction_last100(queue_name) > self.MAX_QUEUE_SHARE:
            return 0.0
        metadata = episode.metadata if isinstance(episode.metadata, dict) else {}
        failure_rate = float(self.module_failure_rates.get(metadata.get("primary_module", "unknown"), 0.0))
        novelty_value = float(metadata.get("novelty_value", 0.0))
        undertrained = float(metadata.get("undertrained_module_value", 0.0))
        retention_risk = float(metadata.get("retention_risk", 0.0))
        overfitting_risk = float(metadata.get("overfitting_risk", 0.0))
        weight = 1.0 + failure_rate * 3.0 + novelty_value * 1.5 + undertrained * 2.0 + retention_risk - overfitting_risk
        if queue_name == "failure_replay":
            weight *= self.failure_replay_store.get_priority_weight(episode.episode_id)
        return max(0.0, weight)

    def sample(self) -> TrainingEpisode:
        if self.steps_since_retention_test >= self.RETENTION_TEST_MAX_GAP and self.queues["retention_test"]:
            self.steps_since_retention_test = 0
            episode = random.choice(self.queues["retention_test"])
            self.sample_history.append("retention_test")
            self.sample_history = self.sample_history[-100:]
            return episode

        choices: List[Tuple[TrainingEpisode, str, float]] = []
        for queue_name, episodes in self.queues.items():
            for episode in episodes:
                w = self._episode_weight(queue_name, episode)
                if w > 0.0:
                    choices.append((episode, queue_name, w))
        if not choices:
            for queue_name in self.QUEUE_NAMES:
                if self.queues[queue_name]:
                    episode = random.choice(self.queues[queue_name])
                    self.sample_history.append(queue_name)
                    self.sample_history = self.sample_history[-100:]
                    self.steps_since_retention_test += 1
                    return episode
            raise RuntimeError("No episodes available in any queue")

        total_weight = sum(item[2] for item in choices)
        r = random.uniform(0.0, total_weight)
        upto = 0.0
        selected_episode = choices[-1][0]
        selected_queue = choices[-1][1]
        for episode, queue_name, weight in choices:
            upto += weight
            if upto >= r:
                selected_episode = episode
                selected_queue = queue_name
                break
        self.sample_history.append(selected_queue)
        self.sample_history = self.sample_history[-100:]
        self.steps_since_retention_test = 0 if selected_queue == "retention_test" else self.steps_since_retention_test + 1
        return selected_episode

    def add_to_queue(self, queue_name: str, episode: TrainingEpisode):
        if queue_name not in self.queues:
            raise ValueError(f"Unknown queue: {queue_name}")
        if episode.episode_id not in self._episode_to_queue:
            self.queues[queue_name].append(episode)
            self._episode_to_queue[episode.episode_id] = queue_name

    def add_or_refresh_failure_replay(self, run_log: RunLog):
        self.failure_replay_store.add_or_refresh(run_log)
        queue_name = self._episode_to_queue.get(run_log.episode_id)
        if queue_name == "failure_replay":
            return
        source_episode = None
        for queue in self.queues.values():
            for episode in queue:
                if episode.episode_id == run_log.episode_id:
                    source_episode = episode
                    break
            if source_episode is not None:
                break
        if source_episode is not None and run_log.episode_id not in [ep.episode_id for ep in self.queues["failure_replay"]]:
            self.queues["failure_replay"].append(source_episode)
            self._episode_to_queue[run_log.episode_id] = "failure_replay"

    def update_module_failure_rate(self, module: str, failed: bool):
        old = float(self.module_failure_rates.get(module, 0.0))
        self.module_failure_rates[module] = 0.9 * old + 0.1 * (1.0 if failed else 0.0)

    def get_queue_stats(self) -> dict:
        recent = self.sample_history[-100:]
        distribution = {}
        denom = len(recent) if recent else 1
        for q in self.QUEUE_NAMES:
            distribution[q] = sum(1 for x in recent if x == q) / denom
        return {
            "per_queue_counts": {k: len(v) for k, v in self.queues.items()},
            "last_100_distribution": distribution,
            "module_failure_rates": dict(self.module_failure_rates),
        }


class QuantizerFreezer:
    LIFECYCLE = ["provisional", "plastic", "stable", "quantized", "frozen"]
    ADVANCE_THRESHOLDS = {"plastic": 5, "stable": 10, "quantized": 20}

    def __init__(self, failure_replay_store: FailureReplayStore):
        self.lifecycle_store: dict = {}
        self.consecutive_successes: dict = {}
        self.creation_step: dict = {}
        self.current_step: int = 0
        self.failure_replay_store = failure_replay_store
        self._thaw_log: list = []

    def register(self, object_id: str):
        if object_id not in self.lifecycle_store:
            self.lifecycle_store[object_id] = "provisional"
            self.consecutive_successes[object_id] = 0
            self.creation_step[object_id] = self.current_step

    def _advance(self, object_id: str):
        state = self.lifecycle_store.get(object_id, "provisional")
        idx = self.LIFECYCLE.index(state)
        if idx >= len(self.LIFECYCLE) - 1:
            return
        next_state = self.LIFECYCLE[idx + 1]
        if next_state in self.ADVANCE_THRESHOLDS:
            needed = self.ADVANCE_THRESHOLDS[next_state]
            if self.consecutive_successes.get(object_id, 0) < needed:
                return
        self.lifecycle_store[object_id] = next_state

    def maybe_freeze(self, run_log: RunLog):
        self.current_step += 1
        object_ids = []
        object_ids.extend(str(x) for x in run_log.active_traces)
        object_ids.extend(
            str(x.get("basin_id", x.get("id", "basin_unknown"))) if isinstance(x, dict) else str(x)
            for x in run_log.active_basins
        )
        object_ids.extend(str(x) for x in run_log.interference_edges)
        for object_id in object_ids:
            self.ensure_thawed(object_id, reason="tracking")
            if (self.current_step - self.creation_step.get(object_id, self.current_step)) < 10:
                continue
            if self.failure_replay_store.contains(run_log.episode_id):
                continue
            self.consecutive_successes[object_id] = self.consecutive_successes.get(object_id, 0) + 1
            self._advance(object_id)

    def maybe_quantize_stable_parts(self):
        for object_id in list(self.lifecycle_store.keys()):
            self._advance(object_id)

    def thaw(self, object_id: str, reason: str):
        self.lifecycle_store[object_id] = "plastic"
        self.consecutive_successes[object_id] = 0
        self._thaw_log.append({"object_id": object_id, "reason": reason, "time": _utc_now_iso()})

    def ensure_thawed(self, object_id: str, reason: str):
        if object_id not in self.lifecycle_store:
            self.register(object_id)
            return
        if self.lifecycle_store[object_id] == "frozen":
            self.thaw(object_id, reason)

    def get_compression_stats(self) -> dict:
        counts = defaultdict(int)
        for state in self.lifecycle_store.values():
            counts[state] += 1
        compressed = counts.get("quantized", 0) + counts.get("frozen", 0)
        return {
            "state_counts": dict(counts),
            "estimated_memory_saving": compressed * 0.6,
            "thaw_events": len(self._thaw_log),
        }


class MetricsTracker:
    def __init__(self, failure_replay_store: FailureReplayStore, scheduler: EpisodeScheduler):
        self.failure_replay_store = failure_replay_store
        self.scheduler = scheduler
        self._run_history: list = []
        self._patch_history: list = []
        self._update_history: list = []
        self._alerts: list = []
        self._last_retention_score: float = 1.0

    def record_run(self, run_log: RunLog, validation: ValidationResult):
        self._run_history.append({"run_log": run_log, "validation": validation})
        self._run_history = self._run_history[-500:]

    def record_patch(self, patch: Patch, result: PatchResult):
        self._patch_history.append({"patch": patch, "result": result})
        self._patch_history = self._patch_history[-500:]

    def record_update(self, module: str, update_level: int):
        self._update_history.append({"module": module, "update_level": update_level})
        self._update_history = self._update_history[-1000:]

    def _rolling(self, n: int = 100) -> list:
        return self._run_history[-n:]

    def _patch_rates(self) -> Tuple[float, float]:
        if not self._patch_history:
            return 0.0, 0.0
        promoted = sum(1 for item in self._patch_history if item["result"].promoted)
        regressions = sum(1 for item in self._patch_history if not item["result"].retention_passed)
        total = len(self._patch_history)
        return promoted / total, regressions / total

    def get_report(self) -> dict:
        rolling = self._rolling(100)
        if not rolling:
            return {
                "active_traces_per_episode": 0.0,
                "active_basins_per_episode": 0.0,
                "projector_calls_per_episode": 0.0,
                "updates_per_episode": 0.0,
                "success_rate": 0.0,
                "lucidity_false_commit_rate": 0.0,
                "lucidity_false_reject_rate": 0.0,
                "decoder_faithfulness": 0.0,
                "patch_promotion_rate": 0.0,
                "patch_regression_rate": 0.0,
                "cost_per_successful_learning_event": 0.0,
                "failure_replay_queue_depth": self.failure_replay_store.metrics()["failure_replay_queue_depth"],
                "episodes_stuck_past_5_attempts": self.failure_replay_store.metrics()["episodes_stuck_past_5_attempts"],
                "no_update_rate": 0.0,
            }

        active_traces_avg = sum(len(item["run_log"].active_traces) for item in rolling) / len(rolling)
        active_basins_avg = sum(len(item["run_log"].active_basins) for item in rolling) / len(rolling)
        projector_calls_avg = (
            sum(1 for item in rolling if item["run_log"].cost_metrics.get("projector_called", False)) / len(rolling)
        )
        update_window = self._update_history[-100:]
        updates_per_episode = (sum(1 for u in update_window if u["update_level"] > 0) / len(rolling)) if rolling else 0.0
        success_rate = sum(1 for item in rolling if item["validation"].success) / len(rolling)
        false_commit_rate = (
            sum(
                1
                for item in rolling
                if item["run_log"].lucidity_decision == "commit" and not item["validation"].success
            )
            / len(rolling)
        )
        false_reject_rate = (
            sum(
                1
                for item in rolling
                if item["run_log"].lucidity_decision == "reject" and item["validation"].success
            )
            / len(rolling)
        )
        faithfulness_scores = []
        for item in rolling:
            validator_result = item["run_log"].validator_result
            if isinstance(validator_result, dict):
                faithfulness_scores.append(float(validator_result.get("score", item["validation"].score)))
            else:
                faithfulness_scores.append(float(item["validation"].score))
        decoder_faithfulness = (sum(faithfulness_scores) / len(faithfulness_scores)) if faithfulness_scores else 0.0
        promotion_rate, regression_rate = self._patch_rates()
        successful_events = sum(1 for u in update_window if u["update_level"] > 0)
        total_stage_cost = sum(float(item["run_log"].cost_metrics.get("stages_run", 0)) for item in rolling)
        cost_per_learning = total_stage_cost / max(1, successful_events)
        no_update_rate = (
            sum(1 for u in update_window if u["update_level"] == 0) / len(update_window) if update_window else 0.0
        )
        replay_metrics = self.failure_replay_store.metrics()
        return {
            "active_traces_per_episode": active_traces_avg,
            "active_basins_per_episode": active_basins_avg,
            "projector_calls_per_episode": projector_calls_avg,
            "updates_per_episode": updates_per_episode,
            "success_rate": success_rate,
            "lucidity_false_commit_rate": false_commit_rate,
            "lucidity_false_reject_rate": false_reject_rate,
            "decoder_faithfulness": decoder_faithfulness,
            "patch_promotion_rate": promotion_rate,
            "patch_regression_rate": regression_rate,
            "cost_per_successful_learning_event": cost_per_learning,
            "failure_replay_queue_depth": replay_metrics["failure_replay_queue_depth"],
            "episodes_stuck_past_5_attempts": replay_metrics["episodes_stuck_past_5_attempts"],
            "no_update_rate": no_update_rate,
        }

    def get_alerts(self) -> list:
        report = self.get_report()
        replay_metrics = self.failure_replay_store.metrics()
        alerts = []
        if report["episodes_stuck_past_5_attempts"] > 0 and report["patch_promotion_rate"] > 0.5:
            alerts.append("Patches promoted but failures not resolving — check 3-condition clear logic")
        if replay_metrics["cleared_without_shadow_pass"] > 0:
            alerts.append("CRITICAL: failure replay cleared without episode shadow pass")
        queue_stats = self.scheduler.get_queue_stats()
        for name, frac in queue_stats["last_100_distribution"].items():
            if frac > 0.40:
                alerts.append(f"Scheduler imbalance on queue: {name}")
        retention_score = 1.0 - report["patch_regression_rate"]
        if (self._last_retention_score - retention_score) > 0.02:
            alerts.append("Retention regression — review last promoted patch")
        self._last_retention_score = retention_score
        self._alerts = alerts
        return list(alerts)


class InvariantChecker:
    def check(
        self,
        failure_replay_store: FailureReplayStore,
        metrics_tracker: MetricsTracker,
        last_run_log: RunLog,
        last_patch: Patch = None,
    ):
        _ = metrics_tracker
        assert failure_replay_store.metrics()["cleared_without_shadow_pass"] == 0, (
            "cleared_without_shadow_pass is non-zero"
        )

        for entry in failure_replay_store.entries.values():
            all_met = (
                len(entry.patch_ids_applied) > 0
                and entry.shadow_passed
                and entry.consecutive_successes >= 3
            )
            assert not all_met, f"Entry {entry.episode_id} meets all 3 clear conditions but was not cleared"

        if last_run_log.lucidity_margin >= 0.75:
            assert last_run_log.projection_result is None, "Projector was called despite high lucidity margin"

        if last_patch is not None:
            assert len(last_patch.target_objects) <= 2, (
                f"Patch targets {len(last_patch.target_objects)} objects — max is 2"
            )


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
        episodes: list,
        phase: int = 1,
        debug: bool = False,
    ):
        self.phase = phase
        self.debug = debug
        self.live_state: dict = {}
        self.episode_store: dict = {ep.episode_id: ep for ep in episodes}

        self.failure_replay_store = FailureReplayStore()
        self.executor = RunExecutor(
            perception, cue_encoder, dmf, binding, context_op, interference, lucidity, projector, decoder
        )
        self.validator_factory = ValidatorFactory()
        self.blame_assigner = BlameAssigner()
        self.diagnostic_forcer = DiagnosticForcer(self.executor, self.validator_factory)
        self.update_planner = UpdatePlanner()
        self.patch_builder = PatchBuilder()
        self.retention_suite_manager = RetentionSuiteManager()
        self.shadow_evaluator = ShadowEvaluator(self.executor, self.validator_factory)
        self.promotion_manager = PromotionManager()
        self.scheduler = EpisodeScheduler(self.failure_replay_store)
        self.quantizer_freezer = QuantizerFreezer(self.failure_replay_store)
        self.metrics_tracker = MetricsTracker(self.failure_replay_store, self.scheduler)
        self.invariant_checker = InvariantChecker()

        self._last_run_log: Optional[RunLog] = None
        self._last_patch: Optional[Patch] = None

        for ep in episodes:
            queue = self._modality_to_queue(ep.modality)
            self.scheduler.add_to_queue(queue, ep)
            self.retention_suite_manager.add_episode(ep, reason="seed")

    def _modality_to_queue(self, modality: str) -> str:
        return {
            "text": "real_text_chat",
            "code": "verifiable_task",
            "multimodal": "multimodal",
            "agent": "agent_episode",
        }.get(modality, "synthetic_structure")

    def run_one_step(self):
        episode = self.scheduler.sample()
        run_log = self.executor.run(episode, mode="training_observation")

        validator = self.validator_factory.get(episode.validator_type)
        validation = validator.evaluate(run_log, episode)
        run_log.validator_result = vars(validation)

        self.metrics_tracker.record_run(run_log, validation)
        self._last_run_log = run_log
        self._last_patch = None

        if validation.success and run_log.lucidity_margin >= 0.75:
            self.metrics_tracker.record_update("none", 0)
            self.quantizer_freezer.maybe_freeze(run_log)
            self.scheduler.update_module_failure_rate("none", failed=False)
            if self.failure_replay_store.contains(episode.episode_id):
                self.failure_replay_store.record_success(episode.episode_id)
                self.failure_replay_store.try_clear(episode.episode_id)
            if self.debug:
                self.invariant_checker.check(self.failure_replay_store, self.metrics_tracker, run_log, None)
            return

        diagnosis = self.blame_assigner.diagnose(run_log, validation)
        if diagnosis.blame_confidence < 0.5:
            diagnosis = self.diagnostic_forcer.run_forced_tests(run_log, episode)

        self.scheduler.update_module_failure_rate(diagnosis.primary_module, failed=True)
        proposal = self.update_planner.plan(diagnosis, run_log)
        self.metrics_tracker.record_update(diagnosis.primary_module, proposal.update_level)

        if proposal.update_level == 0:
            self.scheduler.add_or_refresh_failure_replay(run_log)
            if self.debug:
                self.invariant_checker.check(self.failure_replay_store, self.metrics_tracker, run_log, None)
            return

        patch = self.patch_builder.build(proposal, run_log)
        self._last_patch = patch

        for obj_id in patch.target_objects:
            self.quantizer_freezer.ensure_thawed(obj_id, reason=f"patch {patch.patch_id}")

        bundle = self.retention_suite_manager.select_shadow_bundle(patch, episode.episode_id, self.phase)
        shadow_result = self.shadow_evaluator.test(patch, bundle, self.live_state, self.episode_store)

        if shadow_result.promoted:
            patch_result = self.promotion_manager.apply(
                patch, self.live_state, self.failure_replay_store, self.shadow_evaluator, self.episode_store
            )
            self.retention_suite_manager.on_promote(patch_result)
            self.metrics_tracker.record_patch(patch, patch_result)
        else:
            self.promotion_manager.reject(patch, shadow_result.notes, self.failure_replay_store, run_log)
            self.scheduler.add_or_refresh_failure_replay(run_log)
            self.metrics_tracker.record_patch(patch, shadow_result)

        self.quantizer_freezer.maybe_quantize_stable_parts()
        for alert in self.metrics_tracker.get_alerts():
            print(f"[ALERT] {alert}")

        if self.debug:
            self.invariant_checker.check(self.failure_replay_store, self.metrics_tracker, run_log, patch)

    def run(self, n_steps: int):
        for _ in range(n_steps):
            self.run_one_step()

    def get_status(self) -> dict:
        return {
            "metrics": self.metrics_tracker.get_report(),
            "queue_stats": self.scheduler.get_queue_stats(),
            "compression": self.quantizer_freezer.get_compression_stats(),
            "failure_replay": self.failure_replay_store.metrics(),
            "alerts": self.metrics_tracker.get_alerts(),
            "live_state_keys": list(self.live_state.keys()),
            "patch_history_count": len(self.promotion_manager.patch_history),
            "rejected_patch_count": len(self.promotion_manager.rejected_patch_history),
        }
