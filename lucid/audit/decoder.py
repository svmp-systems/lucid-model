"""Audit writer for direct decoder smoke runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lucid.audit.direct_run import write_smoke_run
from lucid.ir.expression import DecoderInput, DecoderOutput


def write_decoder_audit(
    *,
    audit_base_dir: str | Path,
    decoder_input: DecoderInput,
    decoder_output: DecoderOutput,
    details: dict[str, Any] | None = None,
) -> Path:
    label = str((details or {}).get("fixture") or "run")

    def _extra(_inp: DecoderInput, out: DecoderOutput) -> dict[str, Any]:
        report = out.faithfulness_report
        return {
            "render_mode": out.render_mode,
            "refused": out.refused,
            "faithfulness_passed": report.passed,
            "policy_violations": list(report.policy_violations),
            "sentence_count": len(out.sentence_refs),
            "surface_preview": (out.surface_text or "")[:160],
            "has_grid": out.surface_grid is not None,
        }

    def _readme(module: str, run_label: str, _inp: DecoderInput, out: DecoderOutput) -> list[str]:
        report = out.faithfulness_report
        lines = [
            f"smoke run: {module}",
            "",
            f"label: {run_label}",
            f"render_mode: {out.render_mode or '-'}",
            f"faithfulness: {'pass' if report.passed else 'fail'}",
        ]
        if out.surface_text:
            lines.append(f"text: {out.surface_text[:120]}")
        if out.surface_grid is not None:
            lines.append(f"grid_rows: {len(out.surface_grid)}")
        return lines

    return write_smoke_run(
        module="decoder",
        label=label,
        stage_input=decoder_input,
        stage_output=decoder_output,
        audit_base_dir=audit_base_dir,
        build_manifest_extra=_extra,
        build_readme_lines=_readme,
        details=details,
    )
