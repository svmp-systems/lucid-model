"""Write per-stage audit JSON: machine fields + human summary in every file."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from lucid.ir.common import AuditEnvelope, Provenance
from lucid.ir.pipeline import PipelineRun, RunContext, StageName
from lucid.ir.serde import from_dict, to_dict, to_json

SCHEMA_VERSION = 1

_PIPELINE_STAGE_FIELDS: tuple[tuple[str, str, str], ...] = (
    (StageName.PERCEPTION.value, "perception_input", "evidence_graph"),
    (StageName.CUE_ENCODER.value, "cue_encoder_input", "cue_cloud"),
    (StageName.DMF.value, "dmf_input", "dmf_output"),
    (StageName.BINDING.value, "binding_input", "binding_output"),
    (StageName.CONTEXT_OP.value, "context_op_input", "context_op_output"),
    (StageName.INTERFERENCE.value, "interference_input", "interference_output"),
    (StageName.BASINS.value, "basin_input", "basin_output"),
    (StageName.LUCIDITY.value, "lucidity_input", "lucidity_output"),
    (StageName.PROJECTOR.value, "projector_input", "projector_output"),
    (StageName.DECODER.value, "decoder_input", "decoder_output"),
)


# --- Run index (was manifest.py) ---


@dataclass(slots=True)
class StageAuditRef:
    stage_name: str
    file_name: str
    stage_index: int = 0
    occurrence: int = 1
    input_hash: str = ""
    output_hash: str = ""
    duration_ms: float = 0.0
    success: bool = True
    error_message: str = ""


@dataclass(slots=True)
class RunAuditManifest:
    run_id: str
    session_id: str = ""
    turn_index: int = 0
    mode: str = "inference"
    task_intent: str = ""
    episode_id: str = ""
    created_at: str = ""
    stages: list[StageAuditRef] = field(default_factory=list)
    lucidity_decision: str = ""
    wall_time_ms: float = 0.0
    run_content_hash: str = ""
    summary: dict[str, Any] = field(default_factory=dict)


# --- Stable hashes (was hash_utils.py) ---


def canonical_json(obj: Any) -> str:
    return json.dumps(to_dict(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


# --- Paths (was paths.py; session layout kept for Phase 2) ---


def resolve_run_dir(base_dir: Path | str, context: RunContext) -> Path:
    """``base_dir`` is the directory that directly contains run folders."""
    base = Path(base_dir)
    if context.session_id:
        return base / context.session_id / f"turn_{context.turn_index:04d}" / context.run_id
    return base / context.run_id


# --- Human summaries (embedded in JSON files) ---


def summarize_stage_output(stage_name: str, output: Any) -> dict[str, Any]:
    """Short summary block stored inside each stage .json file."""
    if output is None:
        return {"headline": "no output", "lines": ["(no output)"]}

    data = to_dict(output) if not isinstance(output, dict) else output
    lines: list[str] = []

    if stage_name == "perception":
        units = data.get("candidate_units") or []
        flags = data.get("uncertainty_flags") or []
        lines.append(f"candidate_units: {len(units)}")
        lines.append(f"uncertainty_flags: {len(flags)}")
        surfaces = [u.get("surface", "") for u in units[:6] if u.get("surface")]
        if surfaces:
            lines.append(f"surfaces: {', '.join(surfaces)}")
        headline = f"{len(units)} units"
        if surfaces:
            headline += f" ({', '.join(surfaces[:3])})"

    elif stage_name == "cue_encoder":
        prim = len(data.get("primitive_trace_activations") or [])
        rel = len(data.get("relational_trace_activations") or [])
        lines.append(f"primitive_activations: {prim}")
        lines.append(f"relational_activations: {rel}")
        headline = f"{prim} primitive, {rel} relational activations"

    elif stage_name == "dmf":
        n = len(data.get("active_traces") or [])
        margin = data.get("top_margin", 0.0)
        lines.append(f"active_traces: {n}")
        lines.append(f"top_margin: {margin}")
        headline = f"{n} active traces, margin {margin}"

    elif stage_name == "binding":
        n = len(data.get("candidate_frames") or [])
        score = data.get("binding_stability_score", 0.0)
        lines.append(f"candidate_frames: {n}")
        lines.append(f"binding_stability_score: {score}")
        headline = f"{n} candidate frames"

    elif stage_name == "context_op":
        ctx = len(data.get("context_frames") or [])
        scoped = len(data.get("scoped_trace_assignments") or [])
        links = len(data.get("frame_links") or [])
        gates = len(data.get("interference_gates") or [])
        pressure = len(data.get("local_basin_pressures") or [])
        lines.append(f"context_frames: {ctx}")
        lines.append(f"scoped_trace_assignments: {scoped}")
        lines.append(f"frame_links: {links}")
        lines.append(f"interference_gates: {gates}")
        lines.append(f"local_basin_pressures: {pressure}")
        notes = data.get("audit_notes") or []
        if notes:
            lines.append(f"audit: {notes[0]}")
        headline = f"{ctx} context frames, {scoped} scoped traces, {gates} gates"

    elif stage_name == "interference":
        tt = len(data.get("trace_trace_edges") or [])
        tf = len(data.get("trace_frame_edges") or [])
        fb = len(data.get("frame_basin_edges") or [])
        scoped = len(data.get("scoped_basin_energy_deltas") or [])
        conflicts = len(data.get("conflict_reports") or [])
        lines.append(f"trace_trace_edges: {tt}")
        lines.append(f"trace_frame_edges: {tf}")
        lines.append(f"frame_basin_edges: {fb}")
        lines.append(f"scoped_basin_energy_deltas: {scoped}")
        lines.append(f"conflict_reports: {conflicts}")
        notes = data.get("audit_notes") or []
        if notes:
            lines.append(f"audit: {notes[0]}")
        headline = f"{tt} trace edges, {scoped} scoped basin deltas, {conflicts} conflicts"

    elif stage_name == "basins":
        basins = data.get("candidate_basin_states") or []
        summary = data.get("competition_summary") or {}
        top = summary.get("top_basin_id", "")
        margin = summary.get("top_margin", 0.0)
        lines.append(f"candidate_basin_states: {len(basins)}")
        lines.append(f"top_basin: {top}")
        lines.append(f"top_margin: {margin}")
        headline = f"{len(basins)} basins; top {top or '-'} (margin {margin})"

    elif stage_name == "lucidity":
        decision = data.get("decision", "")
        committed = data.get("committed_state") or {}
        primary = committed.get("primary_basin_id", "") if isinstance(committed, dict) else ""
        lines.append(f"decision: {decision}")
        if primary:
            lines.append(f"primary_basin_id: {primary}")
        headline = f"decision: {decision}"

    elif stage_name == "projector":
        n = len(data.get("rollouts") or [])
        best = data.get("best_rollout_id", "")
        lines.append(f"rollouts: {n}")
        lines.append(f"best_rollout_id: {best}")
        headline = f"{n} rollouts"

    elif stage_name == "decoder":
        refused = data.get("refused", False)
        grid = data.get("surface_grid")
        if isinstance(grid, list) and grid and isinstance(grid[0], list):
            rows = len(grid)
            cols = len(grid[0]) if grid[0] else 0
            lines.append(f"surface_grid: {rows}x{cols}")
            headline = f"grid {rows}x{cols}"
        else:
            text = (data.get("surface_text") or "").strip()
            lines.append(f"refused: {refused}")
            if text:
                preview = text[:120] + ("…" if len(text) > 120 else "")
                lines.append(f"surface_text: {preview}")
            headline = "refused" if refused else (text[:60] + "…" if len(text) > 60 else text or "empty")

    else:
        keys = ", ".join(sorted(data.keys())[:8])
        lines.append(f"fields: {keys}")
        headline = stage_name

    return {"headline": headline, "lines": lines}


def _build_manifest_summary(manifest: RunAuditManifest) -> dict[str, Any]:
    stage_lines = []
    for ref in manifest.stages:
        mark = "ok" if ref.success else "FAIL"
        ms = f"{ref.duration_ms:.0f}ms" if ref.duration_ms else "-"
        label = ref.stage_name if ref.occurrence <= 1 else f"{ref.stage_name}#{ref.occurrence}"
        stage_lines.append(f"  {label:<14} {mark:<4} {ms:>8}")

    lucidity = manifest.lucidity_decision or "(none)"
    headline = f"{lucidity} · {len(manifest.stages)} stages · {manifest.wall_time_ms:.0f}ms total"

    return {
        "headline": headline,
        "lines": [
            f"run_id: {manifest.run_id}",
            f"task: {manifest.task_intent} · mode: {manifest.mode}",
            f"episode: {manifest.episode_id or '-'}",
            "",
            "stages:",
            *stage_lines,
        ],
    }


def _stage_record(
    envelope: AuditEnvelope,
    *,
    duration_ms: float,
    success: bool,
    error_message: str,
    stage_index: int = 0,
    occurrence: int = 1,
) -> dict[str, Any]:
    """On-disk shape: meta + summary first, then input/output for machines."""
    output = envelope.payload.get("output")
    summary = summarize_stage_output(envelope.stage_name, output)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": envelope.run_id,
        "stage_name": envelope.stage_name,
        "stage_index": stage_index,
        "occurrence": occurrence,
        "timestamp": envelope.timestamp,
        "adapter_version": envelope.adapter_version,
        "input_hash": envelope.input_hash,
        "output_hash": envelope.payload.get("output_hash", ""),
        "duration_ms": duration_ms,
        "success": success,
        "error_message": error_message,
        "summary": summary,
        "input": envelope.payload.get("input"),
        "output": output,
        "provenance": to_dict(envelope.provenance) if envelope.provenance else None,
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stage_file_name(stage_name: str, occurrence: int = 1) -> str:
    if occurrence <= 1:
        return f"{stage_name}.json"
    return f"{stage_name}_{occurrence:02d}.json"


def _stage_key(name: StageName | str) -> str:
    if isinstance(name, Enum):
        return str(name.value)
    return str(name)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(to_json(data), encoding="utf-8")


class AuditLogger:
    """Persists stage audits as readable JSON under audit/runs/{run_id}/."""

    def __init__(self, base_dir: Path | str = "audit", *, adapter_version: str = "0.1.0") -> None:
        self.base_dir = Path(base_dir)
        self.adapter_version = adapter_version

    def run_directory(self, context: RunContext) -> Path:
        return resolve_run_dir(self.base_dir, context)

    def write_stage(
        self,
        *,
        run_dir: Path,
        context: RunContext,
        stage_name: str,
        stage_input: Any,
        stage_output: Any,
        file_name: str | None = None,
        stage_index: int = 0,
        occurrence: int = 1,
        duration_ms: float = 0.0,
        success: bool = True,
        error_message: str = "",
        provenance: Provenance | None = None,
    ) -> AuditEnvelope:
        run_dir.mkdir(parents=True, exist_ok=True)

        input_hash = content_hash(stage_input) if stage_input is not None else ""
        output_hash = content_hash(stage_output) if stage_output is not None else ""

        envelope = AuditEnvelope(
            run_id=context.run_id,
            stage_name=stage_name,
            timestamp=_utc_now_iso(),
            input_hash=input_hash,
            payload={
                "input": to_dict(stage_input),
                "output": to_dict(stage_output),
                "output_hash": output_hash,
            },
            adapter_version=self.adapter_version,
            provenance=provenance,
        )

        record = _stage_record(
            envelope,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            stage_index=stage_index,
            occurrence=occurrence,
        )
        _write_json(run_dir / (file_name or _stage_file_name(stage_name, occurrence)), record)
        return envelope

    def write_pipeline_run(self, run: PipelineRun) -> Path:
        context = run.context
        run_dir = self.run_directory(context)
        run_dir.mkdir(parents=True, exist_ok=True)

        refs: list[StageAuditRef] = []
        output_hashes: list[str] = []

        if run.stage_records:
            for record in run.stage_records:
                stage_name = _stage_key(record.stage_name)
                result = None
                if 0 <= record.stage_index < len(run.stage_results):
                    candidate = run.stage_results[record.stage_index]
                    if _stage_key(candidate.stage_name) == stage_name:
                        result = candidate

                duration_ms = result.duration_ms if result else 0.0
                success = result.success if result else record.output_payload is not None
                error_message = result.error_message if result else ""
                occurrence = record.occurrence or 1
                file_name = _stage_file_name(stage_name, occurrence)

                envelope = self.write_stage(
                    run_dir=run_dir,
                    context=context,
                    stage_name=stage_name,
                    stage_input=record.input_payload,
                    stage_output=record.output_payload,
                    file_name=file_name,
                    stage_index=record.stage_index,
                    occurrence=occurrence,
                    duration_ms=duration_ms,
                    success=success,
                    error_message=error_message,
                )

                output_hash = envelope.payload.get("output_hash", "")
                if output_hash:
                    output_hashes.append(output_hash)

                refs.append(
                    StageAuditRef(
                        stage_name=stage_name,
                        file_name=file_name,
                        stage_index=record.stage_index,
                        occurrence=occurrence,
                        input_hash=envelope.input_hash,
                        output_hash=output_hash,
                        duration_ms=duration_ms,
                        success=success,
                        error_message=error_message,
                    )
                )

                if result is not None:
                    result.audit = envelope
        else:
            stage_results_by_name = {
                _stage_key(result.stage_name): result for result in run.stage_results
            }

            for stage_name, input_attr, output_attr in _PIPELINE_STAGE_FIELDS:
                stage_input = getattr(run, input_attr)
                stage_output = getattr(run, output_attr)
                if stage_input is None and stage_output is None:
                    continue

                result = stage_results_by_name.get(stage_name)
                duration_ms = result.duration_ms if result else 0.0
                success = result.success if result else stage_output is not None
                error_message = result.error_message if result else ""
                file_name = _stage_file_name(stage_name)

                envelope = self.write_stage(
                    run_dir=run_dir,
                    context=context,
                    stage_name=stage_name,
                    stage_input=stage_input,
                    stage_output=stage_output,
                    file_name=file_name,
                    duration_ms=duration_ms,
                    success=success,
                    error_message=error_message,
                )

                output_hash = envelope.payload.get("output_hash", "")
                if output_hash:
                    output_hashes.append(output_hash)

                refs.append(
                    StageAuditRef(
                        stage_name=stage_name,
                        file_name=file_name,
                        input_hash=envelope.input_hash,
                        output_hash=output_hash,
                        duration_ms=duration_ms,
                        success=success,
                        error_message=error_message,
                    )
                )

                if result is not None:
                    result.audit = envelope

        lucidity_decision = ""
        if run.lucidity_output is not None:
            lucidity_decision = run.lucidity_output.decision.value

        task_intent = context.task_intent
        task_intent_str = task_intent.value if hasattr(task_intent, "value") else str(task_intent)

        manifest = RunAuditManifest(
            run_id=context.run_id,
            session_id=context.session_id,
            turn_index=context.turn_index,
            mode=context.mode,
            task_intent=task_intent_str,
            episode_id=context.episode.episode_id if context.episode else "",
            created_at=_utc_now_iso(),
            stages=refs,
            lucidity_decision=lucidity_decision,
            wall_time_ms=run.cost_metrics.wall_time_ms,
            run_content_hash=content_hash(output_hashes),
        )
        manifest.summary = _build_manifest_summary(manifest)

        manifest_record = {
            "schema_version": SCHEMA_VERSION,
            **to_dict(manifest),
        }
        _write_json(run_dir / "manifest.json", manifest_record)

        readme_lines = manifest.summary.get("lines", [])
        headline = manifest.summary.get("headline", "")
        readme_text = "\n".join([headline, "=" * len(headline), ""] + readme_lines) + "\n"
        (run_dir / "README.txt").write_text(readme_text, encoding="utf-8")

        context.audit_dir = str(run_dir)
        return run_dir

    def load_manifest(self, run_dir: Path | str) -> RunAuditManifest:
        data = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
        return from_dict(data, RunAuditManifest)

    def load_stage_record(self, run_dir: Path | str, stage_name: str) -> dict[str, Any]:
        return json.loads((Path(run_dir) / _stage_file_name(stage_name)).read_text(encoding="utf-8"))

    def load_stage_envelope(self, run_dir: Path | str, stage_name: str) -> AuditEnvelope:
        """Rebuild AuditEnvelope from on-disk stage record (for older callers)."""
        record = self.load_stage_record(run_dir, stage_name)
        return AuditEnvelope(
            run_id=record["run_id"],
            stage_name=record["stage_name"],
            timestamp=record["timestamp"],
            input_hash=record.get("input_hash", ""),
            payload={
                "input": record.get("input"),
                "output": record.get("output"),
                "output_hash": record.get("output_hash", ""),
            },
            adapter_version=record.get("adapter_version", ""),
            provenance=from_dict(record["provenance"], Provenance) if record.get("provenance") else None,
        )
