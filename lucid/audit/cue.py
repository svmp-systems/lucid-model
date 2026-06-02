"""Audit writer for direct cue encoder runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lucid.audit.direct_run import write_smoke_run
from lucid.ir.cue import CueCloud, CueEncoderInput


def write_cue_encoder_audit(
    *,
    audit_base_dir: str | Path,
    cue_input: CueEncoderInput,
    cue_cloud: CueCloud,
    details: dict[str, Any] | None = None,
) -> Path:
    label = str((details or {}).get("fixture") or "run")

    def _extra(_inp: CueEncoderInput, out: CueCloud) -> dict[str, Any]:
        primitive = out.primitive_trace_activations
        relational = out.relational_trace_activations
        return {
            "primitive_activation_count": len(primitive),
            "relational_activation_count": len(relational),
            "ambiguity_policy": str(out.ambiguity_policy),
            "retrieval_budget_used": out.retrieval_budget_used,
            "top_cue_ids": [req.trace_id for req in primitive[:8]],
        }

    def _readme(module: str, run_label: str, _inp: CueEncoderInput, out: CueCloud) -> list[str]:
        primitive = out.primitive_trace_activations
        top = ", ".join(req.trace_id for req in primitive[:6]) or "-"
        return [
            f"smoke run: {module}",
            "",
            f"label: {run_label}",
            f"primitive_activations: {len(primitive)}",
            f"relational_activations: {len(out.relational_trace_activations)}",
            f"top_cues: {top}",
        ]

    return write_smoke_run(
        module="cue_encoder",
        label=label,
        stage_input=cue_input,
        stage_output=cue_cloud,
        audit_base_dir=audit_base_dir,
        build_manifest_extra=_extra,
        build_readme_lines=_readme,
        details=details,
    )
