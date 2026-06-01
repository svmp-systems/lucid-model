"""Audit writer for direct binding runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from lucid.audit.logger import content_hash
from lucid.ir.binding import BindingInput, BindingOutput
from lucid.ir.serde import to_dict


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(to_dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_binding_audit(
    *,
    audit_base_dir: str | Path,
    binding_input: BindingInput,
    binding_output: BindingOutput,
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
    _write_json(run_dir / files["input"], binding_input)
    _write_json(run_dir / files["output"], binding_output)

    frames = binding_output.candidate_frames
    manifest = {
        "schema_version": 1,
        "created_at": _utc_now_iso(),
        "run_id": run_id,
        "stage_name": "binding",
        "input_hash": content_hash(binding_input),
        "output_hash": content_hash(binding_output),
        "candidate_frame_count": len(frames),
        "frame_ids": [frame.frame_id for frame in frames],
        "binding_stability_score": binding_output.binding_stability_score,
        "competition_edge_count": len(binding_output.frame_competition_edges),
        "audit_notes": list(binding_output.audit_notes),
        "files": files,
        "details": details or {},
    }
    _write_json(run_dir / files["manifest"], manifest)

    readme = (
        f"Binding audit {run_id}\n"
        f"Frames: {len(frames)}  Stability: {binding_output.binding_stability_score:.3f}\n"
        f"Frame ids: {', '.join(frame.frame_id for frame in frames) or '(none)'}\n"
    )
    (run_dir / files["readme"]).write_text(readme, encoding="utf-8")
    return run_dir
