"""Durable audit writer for direct cue encoder runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from lucid.audit.logger import content_hash
from lucid.ir.cue import CueCloud, CueEncoderInput
from lucid.ir.serde import to_dict


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(to_dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_cue_encoder_audit(
    *,
    audit_base_dir: str | Path,
    cue_input: CueEncoderInput,
    cue_cloud: CueCloud,
    details: dict[str, Any] | None = None,
) -> Path:
    run_id = uuid4().hex
    run_dir = Path(audit_base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "input": "input.json",
        "output": "output.json",
        "manifest": "manifest.json",
        "readme": "README.txt",
    }
    _write_json(run_dir / files["input"], cue_input)
    _write_json(run_dir / files["output"], cue_cloud)

    primitive = cue_cloud.primitive_trace_activations
    relational = cue_cloud.relational_trace_activations
    manifest = {
        "schema_version": 1,
        "created_at": _utc_now_iso(),
        "run_id": run_id,
        "stage_name": "cue_encoder",
        "input_hash": content_hash(cue_input),
        "output_hash": content_hash(cue_cloud),
        "primitive_activation_count": len(primitive),
        "relational_activation_count": len(relational),
        "ambiguity_policy": str(cue_cloud.ambiguity_policy),
        "retrieval_budget_used": cue_cloud.retrieval_budget_used,
        "files": files,
        "details": details or {},
    }
    _write_json(run_dir / files["manifest"], manifest)

    top = ", ".join(req.trace_id for req in primitive[:6]) or "-"
    lines = [
        "cue_encoder direct run",
        "======================",
        "",
        f"run_id: {run_id}",
        f"primitive_activations: {len(primitive)}",
        f"relational_activations: {len(relational)}",
        f"ambiguity_policy: {cue_cloud.ambiguity_policy}",
        f"top_cues: {top}",
        "",
        "files:",
        *[f"- {name}: {file_name}" for name, file_name in files.items()],
        "",
    ]
    (run_dir / files["readme"]).write_text("\n".join(lines), encoding="utf-8")
    return run_dir
