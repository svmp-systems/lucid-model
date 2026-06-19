"""CLI smoke audit writers for direct module runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lucid.audit.direct_run import write_smoke_run
from lucid.ir.basins import BasinInput, BasinOutput
from lucid.ir.binding import BindingInput, BindingOutput
from lucid.ir.cue import CueCloud, CueEncoderInput
from lucid.ir.expression import DecoderInput, DecoderOutput
from lucid.ir.lucidity import LucidityInput, LucidityOutput


def _run_label(details: dict[str, Any] | None) -> str:
    return str((details or {}).get("fixture") or "run")


def write_basins_audit(
    *,
    audit_base_dir: str | Path,
    basin_input: BasinInput,
    basin_output: BasinOutput,
    details: dict[str, Any] | None = None,
) -> Path:
    label = _run_label(details)

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


def write_binding_audit(
    *,
    audit_base_dir: str | Path,
    binding_input: BindingInput,
    binding_output: BindingOutput,
    details: dict[str, Any] | None = None,
) -> Path:
    label = _run_label(details)

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


def write_cue_encoder_audit(
    *,
    audit_base_dir: str | Path,
    cue_input: CueEncoderInput,
    cue_cloud: CueCloud,
    details: dict[str, Any] | None = None,
) -> Path:
    label = _run_label(details)

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


def write_decoder_audit(
    *,
    audit_base_dir: str | Path,
    decoder_input: DecoderInput,
    decoder_output: DecoderOutput,
    details: dict[str, Any] | None = None,
) -> Path:
    label = _run_label(details)

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


def write_lucidity_audit(
    *,
    audit_base_dir: str | Path,
    lucidity_input: LucidityInput,
    lucidity_output: LucidityOutput,
    details: dict[str, Any] | None = None,
) -> Path:
    label = _run_label(details)

    def _extra(_inp: LucidityInput, out: LucidityOutput) -> dict[str, Any]:
        checks = out.check_results
        return {
            "decision": out.decision.value,
            "pass_kind": _inp.pass_kind,
            "task_intent": _inp.task_intent,
            "margin": out.confidence_summary.margin,
            "coverage": out.confidence_summary.coverage,
            "coherence": out.confidence_summary.coherence,
            "checks_passed": {
                "margin": checks.margin_check.passed if checks.margin_check else None,
                "coverage": checks.coverage_check.passed if checks.coverage_check else None,
                "coherence": checks.coherence_check.passed if checks.coherence_check else None,
                "binding": checks.binding_stability_check.passed if checks.binding_stability_check else None,
                "scope": checks.scope_check.passed if checks.scope_check else None,
                "projection_fit": checks.projection_fit_check.passed if checks.projection_fit_check else None,
                "contradiction": checks.contradiction_check.passed if checks.contradiction_check else None,
                "maturity": checks.maturity_check.passed if checks.maturity_check else None,
                "risk": checks.risk_check.passed if checks.risk_check else None,
            },
            "audit_notes": list(out.audit_notes),
        }

    def _readme(_module: str, run_label: str, _inp: LucidityInput, out: LucidityOutput) -> list[str]:
        return [
            "smoke run: lucidity",
            "",
            f"label: {run_label}",
            f"decision: {out.decision.value}",
            f"margin: {out.confidence_summary.margin:.3f}",
            f"coverage: {out.confidence_summary.coverage:.3f}",
        ]

    return write_smoke_run(
        module="lucidity",
        label=label,
        stage_input=lucidity_input,
        stage_output=lucidity_output,
        audit_base_dir=audit_base_dir,
        build_manifest_extra=_extra,
        build_readme_lines=_readme,
        details=details,
    )
