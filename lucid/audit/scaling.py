"""Scaling observatory — cost/quality receipts under audit/scaling/ (scoped spec)."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from lucid.ir.pipeline import PipelineRun
from lucid.ir.serde import to_dict
from lucid.ir.training import Episode

# --- types ---


@dataclass(slots=True)
class ScalingPoint:
    point_id: str
    timestamp_utc: str
    scale_id: str
    event_type: str
    training_mode: str
    module_under_test: str
    run_kind: str
    episode_id: str = ""
    run_id: str = ""
    corpus_slice: str = ""
    build_phase: int = 1
    validator_success: bool | None = None
    validator_score: float | None = None
    module_gold_score: float | None = None
    lucidity_decision: str = ""
    audit_complete: bool | None = None
    governor_action: str = ""
    patch_promoted: bool | None = None
    patch_rejected: bool | None = None
    retention_canary_pass: bool | None = None
    wall_time_ms: float = 0.0
    gpu_seconds: float = 0.0
    active_trace_count: int | None = None
    active_basin_count: int | None = None
    retrieval_budget: int | None = None
    config_hash: str = ""
    git_sha: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScalingConfig:
    enabled: bool = True
    data_dir: Path = Path("audit/scaling")
    build_phase: int = 1
    hardware_class: str = "local"

    @property
    def points_path(self) -> Path:
        return self.data_dir / "points.jsonl"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @classmethod
    def from_env(cls) -> ScalingConfig:
        enabled = os.environ.get("LUCID_SCALING", "1").strip().lower() not in ("0", "false", "no", "off")
        base = os.environ.get("LUCID_SCALING_DIR", "audit/scaling").strip() or "audit/scaling"
        try:
            phase = int(os.environ.get("LUCID_BUILD_PHASE", "1"))
        except ValueError:
            phase = 1
        return cls(
            enabled=enabled,
            data_dir=Path(base),
            build_phase=phase,
            hardware_class=os.environ.get("LUCID_HARDWARE_CLASS", "local").strip() or "local",
        )


@dataclass(slots=True)
class ScalingSummary:
    scale_id: str
    point_count: int
    validator_success_rate: float | None
    module_gold_mean: float | None
    wall_time_ms_p50: float | None
    cost_per_success_wall_s: float | None
    plateau_warning: bool
    falloff_warning: bool


# --- storage ---

_recorder: ScalingRecorder | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_scale_id(*, module_under_test: str, training_mode: str, corpus_slice: str, config_hash: str) -> str:
    return ":".join(
        [
            module_under_test or "none",
            training_mode or "unknown",
            corpus_slice or "default",
            config_hash or "none",
        ]
    )


def config_hash_from_mapping(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def try_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        return out.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def iter_points(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _append_point(path: Path, point: ScalingPoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_dict(point), ensure_ascii=False, default=str))
        handle.write("\n")


class ScalingRecorder:
    def __init__(self, config: ScalingConfig | None = None) -> None:
        self.config = config or ScalingConfig.from_env()

    def record(self, point: ScalingPoint) -> None:
        if self.config.enabled:
            _append_point(self.config.points_path, point)


def record_point(point: ScalingPoint, *, config: ScalingConfig | None = None) -> None:
    if config is not None:
        ScalingRecorder(config).record(point)
        return
    global _recorder
    if _recorder is None:
        _recorder = ScalingRecorder()
    _recorder.record(point)


# --- extract (pipeline / trainer / orchestrator) ---


def _corpus_slice(episode: Episode | None) -> str:
    if episode is None:
        return ""
    if episode.template_id:
        return str(episode.template_id)
    meta = episode.meta or {}
    return str(meta.get("recipe") or meta.get("template") or "")


def point_from_trainer_step(
    *,
    module_under_test: str,
    corpus_slice: str,
    module_gold_score: float,
    wall_time_ms: float = 0.0,
    config: ScalingConfig | None = None,
) -> ScalingPoint:
    cfg = config or ScalingConfig.from_env()
    cfg_hash = config_hash_from_mapping({"module": module_under_test, "corpus": corpus_slice})
    return ScalingPoint(
        point_id=str(uuid4()),
        timestamp_utc=utc_now_iso(),
        scale_id=build_scale_id(
            module_under_test=module_under_test,
            training_mode="calibrate",
            corpus_slice=corpus_slice,
            config_hash=cfg_hash,
        ),
        event_type="trainer_step",
        training_mode="calibrate",
        module_under_test=module_under_test,
        run_kind="train",
        corpus_slice=corpus_slice,
        build_phase=cfg.build_phase,
        module_gold_score=module_gold_score,
        wall_time_ms=wall_time_ms,
        config_hash=cfg_hash,
        git_sha=try_git_sha(),
        provenance={"source": "module_trainer"},
    )


def point_from_pipeline_run(
    run: PipelineRun,
    *,
    config: ScalingConfig | None = None,
    audit_complete: bool | None = None,
) -> ScalingPoint:
    cfg = config or ScalingConfig.from_env()
    ctx = run.context
    episode = ctx.episode
    if audit_complete is None:
        audit_complete = bool(run.stage_results) and all(r.success for r in run.stage_results)

    training_mode = "full_loop" if ctx.mode in ("training_observation", "train") else "inference_only"
    run_kind = "train" if training_mode == "full_loop" else "inference"
    budget = int(run.cue_encoder_input.retrieval_budget) if run.cue_encoder_input else 128
    cfg_hash = config_hash_from_mapping(
        {
            "mode": ctx.mode,
            "retrieval_budget": budget,
            "task_intent": ctx.task_intent.value if hasattr(ctx.task_intent, "value") else str(ctx.task_intent),
        }
    )
    corpus = _corpus_slice(episode) or "episode"

    gold_score: float | None = None
    validator_success: bool | None = None
    if episode is not None and episode.gold is not None:
        target = (episode.gold.lucidity_target or "").strip().lower().replace(" ", "_")
        actual = ""
        if run.lucidity_output is not None:
            d = run.lucidity_output.decision
            actual = d.value if hasattr(d, "value") else str(d)
        if target:
            validator_success = actual == target
            gold_score = 1.0 if validator_success else 0.0

    lucidity_decision = ""
    if run.lucidity_output is not None:
        d = run.lucidity_output.decision
        lucidity_decision = d.value if hasattr(d, "value") else str(d)

    return ScalingPoint(
        point_id=str(uuid4()),
        timestamp_utc=utc_now_iso(),
        scale_id=build_scale_id(
            module_under_test="pipeline",
            training_mode=training_mode,
            corpus_slice=corpus,
            config_hash=cfg_hash,
        ),
        event_type="pipeline_run",
        training_mode=training_mode,
        module_under_test="pipeline",
        run_kind=run_kind,
        episode_id=episode.episode_id if episode is not None else "",
        run_id=ctx.run_id,
        corpus_slice=corpus,
        build_phase=cfg.build_phase,
        validator_success=validator_success,
        validator_score=gold_score,
        module_gold_score=gold_score,
        lucidity_decision=lucidity_decision,
        audit_complete=audit_complete,
        wall_time_ms=float(run.cost_metrics.wall_time_ms),
        active_trace_count=len(run.dmf_output.active_traces) if run.dmf_output else None,
        active_basin_count=len(run.basin_output.candidate_basin_states) if run.basin_output else None,
        retrieval_budget=budget,
        config_hash=cfg_hash,
        git_sha=try_git_sha(),
        provenance={"source": "orchestrator_runner"},
    )


def point_from_orchestrator_step(
    *,
    action: str,
    episode: Any | None,
    run_log: Any | None,
    validation: Any | None,
    diagnosis: Any | None,
    patch_result: Any | None,
    step_index: int,
    config: ScalingConfig | None = None,
) -> ScalingPoint:
    cfg = config or ScalingConfig.from_env()
    corpus = ""
    if episode is not None:
        meta = getattr(episode, "metadata", None) or {}
        if isinstance(meta, dict):
            corpus = str(meta.get("recipe") or meta.get("template_id") or "")
    cfg_hash = config_hash_from_mapping({"phase": cfg.build_phase, "orchestrator": "mvp"})
    validator_success = validator_score = None
    if validation is not None:
        validator_success = bool(getattr(validation, "success", False))
        validator_score = float(getattr(validation, "score", 0.0))
    patch_promoted = patch_rejected = retention_pass = None
    if patch_result is not None:
        patch_promoted = bool(getattr(patch_result, "promoted", False))
        patch_rejected = not patch_promoted and action == "patch_rejected"
        retention_pass = bool(getattr(patch_result, "retention_passed", False))
    lucidity_decision = ""
    if run_log is not None:
        lucidity_decision = str(getattr(run_log, "lucidity_decision", "") or "")
    blame = str(getattr(diagnosis, "primary_module", "") or "") if diagnosis else ""
    return ScalingPoint(
        point_id=str(uuid4()),
        timestamp_utc=utc_now_iso(),
        scale_id=build_scale_id(
            module_under_test="pipeline",
            training_mode="full_loop",
            corpus_slice=corpus or "orchestrator",
            config_hash=cfg_hash,
        ),
        event_type="orchestrator_step",
        training_mode="full_loop",
        module_under_test="pipeline",
        run_kind="train",
        episode_id=getattr(episode, "episode_id", "") if episode is not None else "",
        corpus_slice=corpus,
        build_phase=cfg.build_phase,
        validator_success=validator_success,
        validator_score=validator_score,
        lucidity_decision=lucidity_decision,
        governor_action="NO_UPDATE" if action == "no_update_success" else ("UPDATE" if action.startswith("patch_") else ""),
        patch_promoted=patch_promoted,
        patch_rejected=patch_rejected,
        retention_canary_pass=retention_pass,
        config_hash=cfg_hash,
        git_sha=try_git_sha(),
        provenance={
            "source": "training_orchestrator",
            "action": action,
            "step_index": step_index,
            "primary_blame_module": blame,
        },
    )


def record_pipeline_run(run: PipelineRun, **kwargs: Any) -> None:
    record_point(point_from_pipeline_run(run, **kwargs))


def record_orchestrator_step(**kwargs: Any) -> None:
    record_point(point_from_orchestrator_step(**kwargs))


# --- summarize / export (derived metrics, not stored on append) ---


def load_points(path: Path, *, scale_id: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_points(path):
        if scale_id and row.get("scale_id") != scale_id:
            continue
        rows.append(row)
    return rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def summarize_points(points: list[dict[str, Any]], *, scale_id: str = "") -> ScalingSummary:
    sid = scale_id or (str(points[0].get("scale_id")) if points else "")
    validator_rates = [float(p["validator_success"]) for p in points if p.get("validator_success") is not None]
    gold_scores = [float(p["module_gold_score"]) for p in points if p.get("module_gold_score") is not None]
    wall_times = [float(p["wall_time_ms"]) for p in points if p.get("wall_time_ms")]
    success_count = sum(1 for p in points if p.get("validator_success") is True)
    cost_per_success = None
    if success_count and wall_times:
        cost_per_success = (sum(wall_times) / 1000.0) / success_count
    plateau = falloff = False
    if len(gold_scores) >= 6:
        r, p = _mean(gold_scores[-3:]), _mean(gold_scores[-6:-3])
        if r is not None and p is not None and abs(r - p) < 0.01:
            plateau = True
    if len(validator_rates) >= 6:
        r, p = _mean(validator_rates[-3:]) or 0.0, _mean(validator_rates[-6:-3]) or 0.0
        if p - r > 0.05:
            falloff = True
    return ScalingSummary(
        scale_id=sid,
        point_count=len(points),
        validator_success_rate=_mean(validator_rates),
        module_gold_mean=_mean(gold_scores),
        wall_time_ms_p50=_median(wall_times),
        cost_per_success_wall_s=cost_per_success,
        plateau_warning=plateau,
        falloff_warning=falloff,
    )


def format_summary(summary: ScalingSummary) -> str:
    lines = [f"scale_id: {summary.scale_id or '(all)'}", f"  points: {summary.point_count}"]
    if summary.module_gold_mean is not None:
        lines.append(f"  module_gold_mean: {summary.module_gold_mean:.3f}")
    if summary.validator_success_rate is not None:
        lines.append(f"  validator_success_rate: {summary.validator_success_rate:.3f}")
    if summary.wall_time_ms_p50 is not None:
        lines.append(f"  wall_time_ms p50: {summary.wall_time_ms_p50:.1f}")
    if summary.cost_per_success_wall_s is not None:
        lines.append(f"  cost_per_success (wall s): {summary.cost_per_success_wall_s:.3f}")
    lines.append(f"  plateau_warning: {summary.plateau_warning}")
    lines.append(f"  falloff_warning: {summary.falloff_warning}")
    return "\n".join(lines)


def summarize_file(config: ScalingConfig | None = None, *, scale_id: str | None = None) -> ScalingSummary:
    cfg = config or ScalingConfig.from_env()
    return summarize_points(load_points(cfg.points_path, scale_id=scale_id), scale_id=scale_id or "")


def export_summary_csv(points: list[dict[str, Any]], out_path: Path) -> None:
    by_scale: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        by_scale.setdefault(str(point.get("scale_id") or ""), []).append(point)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scale_id",
        "point_count",
        "validator_success_rate",
        "module_gold_mean",
        "wall_time_ms_p50",
        "cost_per_success_wall_s",
        "plateau_warning",
        "falloff_warning",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sid, group in sorted(by_scale.items()):
            s = summarize_points(group, scale_id=sid)
            writer.writerow(
                {
                    "scale_id": sid,
                    "point_count": s.point_count,
                    "validator_success_rate": s.validator_success_rate,
                    "module_gold_mean": s.module_gold_mean,
                    "wall_time_ms_p50": s.wall_time_ms_p50,
                    "cost_per_success_wall_s": s.cost_per_success_wall_s,
                    "plateau_warning": s.plateau_warning,
                    "falloff_warning": s.falloff_warning,
                }
            )
