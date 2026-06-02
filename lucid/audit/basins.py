"""Audit writer for direct basin runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from lucid.audit.logger import content_hash
from lucid.ir.basins import BasinInput, BasinOutput
from lucid.ir.serde import to_dict


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(to_dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_basins_audit(
    *,
    audit_base_dir: str | Path,
    basin_input: BasinInput,
    basin_output: BasinOutput,
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
    _write_json(run_dir / files["input"], basin_input)
    _write_json(run_dir / files["output"], basin_output)

    states = basin_output.candidate_basin_states
    summary = basin_output.competition_summary
    manifest = {
        "schema_version": 1,
        "created_at": _utc_now_iso(),
        "run_id": run_id,
        "stage_name": "basins",
        "input_hash": content_hash(basin_input),
        "output_hash": content_hash(basin_output),
        "candidate_basin_count": len(states),
        "basin_ids": [state.basin_id for state in states],
        "top_basin_id": summary.top_basin_id,
        "top_margin": summary.top_margin,
        "active_basin_count": summary.active_basin_count,
        "unresolved_conflict_count": len(basin_output.unresolved_conflicts),
        "audit_notes": list(basin_output.audit_notes),
        "files": files,
        "details": details or {},
    }
    _write_json(run_dir / files["manifest"], manifest)

    readme = (
        f"Basins audit {run_id}\n"
        f"Candidates: {len(states)}  Top: {summary.top_basin_id or '-'}  "
        f"Margin: {summary.top_margin:.3f}\n"
        f"Conflicts: {len(basin_output.unresolved_conflicts)}\n"
    )
    (run_dir / files["readme"]).write_text(readme, encoding="utf-8")
    return run_dir
