"""Audit writer for direct binding runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lucid.audit.direct_run import write_smoke_run
from lucid.ir.binding import BindingInput, BindingOutput


def write_binding_audit(
    *,
    audit_base_dir: str | Path,
    binding_input: BindingInput,
    binding_output: BindingOutput,
    details: dict[str, Any] | None = None,
) -> Path:
    label = str((details or {}).get("fixture") or "run")

    def _extra(_inp: BindingInput, out: BindingOutput) -> dict[str, Any]:
        return {
            "candidate_frame_count": len(out.candidate_frames),
            "frame_ids": [frame.frame_id for frame in out.candidate_frames],
            "binding_stability_score": out.binding_stability_score,
            "competition_edge_count": len(out.frame_competition_edges),
            "audit_notes": list(out.audit_notes),
        }

    def _readme(module: str, run_label: str, _inp: BindingInput, out: BindingOutput) -> list[str]:
        return [
            f"smoke run: {module}",
            "",
            f"label: {run_label}",
            f"candidate_frames: {len(out.candidate_frames)}",
            f"binding_stability_score: {out.binding_stability_score:.3f}",
            f"frame_ids: {', '.join(f.frame_id for f in out.candidate_frames) or '(none)'}",
        ]

    return write_smoke_run(
        module="binding",
        label=label,
        stage_input=binding_input,
        stage_output=binding_output,
        audit_base_dir=audit_base_dir,
        build_manifest_extra=_extra,
        build_readme_lines=_readme,
        details=details,
    )
