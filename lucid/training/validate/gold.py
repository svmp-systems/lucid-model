"""Validate pipeline runs and module stores against episode gold."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lucid.cognition.pipe_orchestrator.runner import OrchestratorConfig, OrchestratorRunner
from lucid.cognition.input.perception import PerceptionConfig
from lucid.ir.common import LucidityDecision, Modality, TaskIntent
from lucid.ir.pipeline import PipelineRun
from lucid.ir.serde import from_dict, from_json, to_dict
from lucid.ir.training import Episode
from lucid.training.corpus import adapters
from lucid.training.checkpoint.store import checkpoint_summary, load_checkpoint
from lucid.runtime.paths import DEFAULT_AUDIT_VALIDATION
from lucid.training.corpus.output import read_episodes
from lucid.training.loop.orchestrator import RunLog, TrainingEpisode, ValidationResult


@dataclass(slots=True)
class GoldValidationReport:
    episode_id: str
    success: bool
    score: float
    failure_signals: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class L3ModuleReport:
    module: str
    episodes_checked: int = 0
    episodes_with_gold: int = 0
    store_count: int = 0
    passed: bool = True
    notes: list[str] = field(default_factory=list)


def _normalize_lucidity(raw: str) -> str:
    text = (raw or "").strip().upper()
    if text.startswith("LUCIDITYDECISION."):
        text = text.split(".", 1)[-1]
    return text


def _grid_equal(left: Any, right: Any) -> bool:
    if not isinstance(left, list) or not isinstance(right, list):
        return left == right
    if len(left) != len(right):
        return False
    for row_left, row_right in zip(left, right, strict=False):
        if list(row_left) != list(row_right):
            return False
    return True


class GoldEpisodeValidator:
    """Score a pipeline run against generator gold on an Episode."""

    def evaluate_run(self, run: PipelineRun, episode: Episode) -> GoldValidationReport:
        signals: list[str] = []
        details: dict[str, Any] = {"episode_id": episode.episode_id, "template_id": episode.template_id}

        gold_lucidity = _normalize_lucidity(episode.gold.lucidity_target)
        if run.lucidity_output is not None:
            decision = run.lucidity_output.decision
            actual = _normalize_lucidity(
                decision.value if hasattr(decision, "value") else str(decision)
            )
            details["lucidity"] = {"expected": gold_lucidity, "actual": actual}
            if gold_lucidity and actual != gold_lucidity:
                signals.append("lucidity_decision_mismatch")

        expected = episode.gold.expected_answer
        if expected is not None and run.decoder_output is not None:
            if isinstance(expected, list) and expected and isinstance(expected[0], list):
                actual_grid = run.decoder_output.surface_grid
                details["decoder_grid"] = {"expected": expected, "actual": actual_grid}
                if not _grid_equal(actual_grid, expected):
                    signals.append("decoder_grid_mismatch")
            elif isinstance(expected, str):
                details["decoder_text"] = {
                    "expected_preview": expected[:80],
                    "actual_preview": (run.decoder_output.surface_text or "")[:80],
                }
                if expected and expected not in (run.decoder_output.surface_text or ""):
                    signals.append("decoder_text_mismatch")

        if (
            gold_lucidity == "COMMIT"
            and str(episode.task_intent).endswith("SOLVE_GRID")
            and run.projector_output is None
            and run.lucidity_output is not None
            and _normalize_lucidity(str(run.lucidity_output.decision)) == "REQUEST_PROJECTION"
        ):
            pass
        elif (
            str(episode.task_intent).endswith("SOLVE_GRID")
            and gold_lucidity == "COMMIT"
            and run.projector_output is None
        ):
            signals.append("grid_missing_projection")

        success = not signals
        return GoldValidationReport(
            episode_id=episode.episode_id,
            success=success,
            score=1.0 if success else 0.0,
            failure_signals=signals,
            details=details,
        )

    def evaluate_run_log(self, run_log: RunLog, episode: Episode) -> ValidationResult:
        report = self._run_log_report(run_log, episode)
        return ValidationResult(
            success=report.success,
            score=report.score,
            failure_signals=report.failure_signals,
            expected_state=episode.gold.expected_answer,
            confidence=report.score,
        )

    def _run_log_report(self, run_log: RunLog, episode: Episode) -> GoldValidationReport:
        signals: list[str] = []
        gold_lucidity = _normalize_lucidity(episode.gold.lucidity_target)
        actual = _normalize_lucidity(run_log.lucidity_decision)
        if gold_lucidity and actual != gold_lucidity:
            signals.append("lucidity_decision_mismatch")
        expected = episode.gold.expected_answer
        if expected is not None and run_log.decoder_output != expected:
            if isinstance(expected, list):
                if not _grid_equal(run_log.decoder_output, expected):
                    signals.append("decoder_render_mismatch")
            else:
                signals.append("decoder_render_mismatch")
        return GoldValidationReport(
            episode_id=episode.episode_id,
            success=not signals,
            score=1.0 if not signals else 0.0,
            failure_signals=signals,
        )


class L3ModuleGoldValidator:
    """Check checkpoint stores are populated for phase-1 gold coverage."""

    MODULE_STORE: dict[str, tuple[str, str]] = {
        "perception": ("perception_examples", "examples"),
        "cue_encoder": ("cue_encoder_map", "cue_targets"),
        "dmf": ("tracebank", "records"),
        "binding": ("binding_affordances", "patterns"),
        "context-op": ("context_policy", "scope_patterns"),
        "interference": ("interference_graph", "gates"),
        "basins": ("basin_bank", "records"),
        "lucidity": ("lucidity_policy", "decision_counts"),
        "projector": ("projector_examples", "examples"),
        "decoder": ("decoder_adapter", "render_targets"),
    }

    def evaluate_checkpoint(
        self,
        checkpoint: str | Path,
        episodes: list[Episode],
    ) -> list[L3ModuleReport]:
        state = load_checkpoint(checkpoint, create=True)
        summary = checkpoint_summary(state)
        reports: list[L3ModuleReport] = []
        for module, (store_name, key) in self.MODULE_STORE.items():
            store = state.ensure_store(store_name)
            if key == "decision_counts":
                count = len(store.get("decision_counts") or {})
            else:
                rows = store.get(key, [])
                count = len(rows) if isinstance(rows, list) else len(rows or {})
            with_gold = self._episodes_with_gold(module, episodes)
            passed = count > 0 or not with_gold
            reports.append(
                L3ModuleReport(
                    module=module,
                    episodes_checked=len(episodes),
                    episodes_with_gold=with_gold,
                    store_count=count,
                    passed=passed,
                    notes=[] if passed else [f"empty_store:{store_name}"],
                )
            )
        _ = summary
        return reports

    @staticmethod
    def _episodes_with_gold(module: str, episodes: list[Episode]) -> int:
        count = 0
        for episode in episodes:
            if module == "perception" and any(adapters.perception_targets(episode).values()):
                count += 1
            elif module == "cue_encoder" and adapters.cue_encoder_targets(episode)["trace_targets"]:
                count += 1
            elif module == "dmf" and adapters.dmf_targets(episode):
                count += 1
            elif module == "binding" and adapters.binding_targets(episode):
                count += 1
            elif module == "context-op" and adapters.context_targets(episode)["scope_assignments"]:
                count += 1
            elif module == "interference" and adapters.interference_targets(episode):
                count += 1
            elif module == "basins" and adapters.basin_targets(episode):
                count += 1
            elif module == "lucidity" and adapters.lucidity_target(episode)["decision"]:
                count += 1
            elif module == "projector" and adapters.projector_target(episode):
                count += 1
            elif module == "decoder" and adapters.decoder_target(episode)["expected_answer"] is not None:
                count += 1
        return count


def validate_episode_pack(
    episodes_path: str | Path,
    *,
    checkpoint: str = "",
    audit_dir: str = DEFAULT_AUDIT_VALIDATION,
    limit: int = 0,
) -> dict[str, Any]:
    episodes = read_episodes(episodes_path)
    if limit > 0:
        episodes = episodes[:limit]

    runner = OrchestratorRunner(
        config=OrchestratorConfig(
            audit_base_dir=audit_dir,
            perception=PerceptionConfig(backend="rule", write_audit=False),
            checkpoint=checkpoint or None,
        )
    )
    validator = GoldEpisodeValidator()
    results: list[dict[str, Any]] = []
    crashes = 0
    for episode in episodes:
        try:
            run = runner.run_episode(episode)
            report = validator.evaluate_run(run, episode)
            results.append(to_dict(report))
        except Exception as exc:  # noqa: BLE001
            crashes += 1
            results.append(
                {
                    "episode_id": episode.episode_id,
                    "success": False,
                    "score": 0.0,
                    "failure_signals": [f"crash:{type(exc).__name__}"],
                    "details": {"error": str(exc)},
                }
            )

    successes = sum(1 for row in results if row.get("success"))
    l3 = []
    if checkpoint:
        l3 = [to_dict(row) for row in L3ModuleGoldValidator().evaluate_checkpoint(checkpoint, episodes)]

    payload = {
        "episodes": len(episodes),
        "successes": successes,
        "crashes": crashes,
        "success_rate": successes / len(episodes) if episodes else 0.0,
        "l3_module_gold": l3,
        "results": results,
    }
    return payload


def to_training_episode(episode: Episode) -> TrainingEpisode:
    task = episode.task_intent.value if hasattr(episode.task_intent, "value") else str(episode.task_intent)
    modality = episode.modality.value if hasattr(episode.modality, "value") else str(episode.modality)
    return TrainingEpisode(
        episode_id=episode.episode_id,
        raw_input=episode.raw_input,
        modality=modality,
        task_intent=task,
        context={},
        constraints={},
        expected_output=episode.gold.expected_answer,
        validator_type="gold_episode",
        metadata={
            "template_id": episode.template_id,
            "episode_json": to_dict(episode),
        },
    )


def episode_from_training(training: TrainingEpisode) -> Episode:
    raw = training.metadata.get("episode_json")
    if isinstance(raw, dict):
        return from_dict(raw, Episode)
    if isinstance(raw, str) and raw.strip():
        return from_json(raw, Episode)
    return Episode(
        episode_id=training.episode_id,
        raw_input=training.raw_input,
        modality=Modality(training.modality),
        task_intent=TaskIntent(training.task_intent),
    )
