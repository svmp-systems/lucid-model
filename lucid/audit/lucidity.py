"""Audit writer for direct lucidity runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lucid.audit.direct_run import write_smoke_run
from lucid.ir.lucidity import LucidityInput, LucidityOutput


def write_lucidity_audit(
    *,
    audit_base_dir: str | Path,
    lucidity_input: LucidityInput,
    lucidity_output: LucidityOutput,
    details: dict[str, Any] | None = None,
) -> Path:
    label = str((details or {}).get("fixture") or "run")

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
