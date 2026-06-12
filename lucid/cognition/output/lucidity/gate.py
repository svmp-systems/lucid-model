"""Lucidity gate — run checks, decide, build render packet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lucid.cognition.output.lucidity.checks import run_checks
from lucid.cognition.output.lucidity.config import LucidityConfig
from lucid.cognition.output.lucidity.decide import decide
from lucid.cognition.output.lucidity.render_packet import attach_render_packet
from lucid.ir.lucidity import LucidityInput, LucidityOutput


@dataclass(slots=True)
class LucidityGateConfig:
    thresholds: LucidityConfig | None = None
    checkpoint: str | Path | None = None


def run_lucidity(
    inp: LucidityInput,
    *,
    config: LucidityGateConfig | None = None,
    ctx: object | None = None,
) -> LucidityOutput:
    """Controlled collapse: evaluate upstream state and emit decision + decoder script."""
    cfg = (config.thresholds if config and config.thresholds else None) or LucidityConfig()
    checkpoint = config.checkpoint if config else None
    if checkpoint is None and ctx is not None:
        extra = getattr(ctx, "extra", None)
        if isinstance(extra, dict):
            checkpoint = extra.get("checkpoint") or extra.get("lucidity_checkpoint")
    if checkpoint:
        from lucid.cognition.pipe_orchestrator.checkpoint_runtime import lucidity_config_overrides

        template_id = ""
        if ctx is not None:
            extra = getattr(ctx, "extra", None)
            if isinstance(extra, dict):
                template_id = str(extra.get("template_id") or "")
        overrides = lucidity_config_overrides(checkpoint, template_id=template_id)
        if overrides.get("margin_threshold_answer") is not None:
            cfg.margin_threshold_answer = float(overrides["margin_threshold_answer"])
        if overrides.get("coverage_threshold") is not None:
            cfg.coverage_threshold = float(overrides["coverage_threshold"])
    checks, confidence = run_checks(inp, cfg)
    output = decide(inp, checks, confidence, cfg)
    output.check_results = checks
    output.confidence_summary = confidence
    return attach_render_packet(output, lucidity_input=inp)
