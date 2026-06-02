"""Audit writer for direct basin runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lucid.audit.direct_run import write_smoke_run
from lucid.ir.basins import BasinInput, BasinOutput


def write_basins_audit(
    *,
    audit_base_dir: str | Path,
    basin_input: BasinInput,
    basin_output: BasinOutput,
    details: dict[str, Any] | None = None,
) -> Path:
    label = str((details or {}).get("fixture") or "run")

    def _extra(_inp: BasinInput, out: BasinOutput) -> dict[str, Any]:
        summary = out.competition_summary
        return {
            "candidate_basin_count": len(out.candidate_basin_states),
            "basin_ids": [state.basin_id for state in out.candidate_basin_states],
            "top_basin_id": summary.top_basin_id,
            "top_margin": summary.top_margin,
            "active_basin_count": summary.active_basin_count,
            "unresolved_conflict_count": len(out.unresolved_conflicts),
            "audit_notes": list(out.audit_notes),
        }

    def _readme(module: str, run_label: str, _inp: BasinInput, out: BasinOutput) -> list[str]:
        summary = out.competition_summary
        return [
            f"smoke run: {module}",
            "",
            f"label: {run_label}",
            f"candidate_basins: {len(out.candidate_basin_states)}",
            f"top_basin: {summary.top_basin_id or '-'}",
            f"top_margin: {summary.top_margin:.3f}",
            f"conflicts: {len(out.unresolved_conflicts)}",
        ]

    return write_smoke_run(
        module="basins",
        label=label,
        stage_input=basin_input,
        stage_output=basin_output,
        audit_base_dir=audit_base_dir,
        build_manifest_extra=_extra,
        build_readme_lines=_readme,
        details=details,
    )
