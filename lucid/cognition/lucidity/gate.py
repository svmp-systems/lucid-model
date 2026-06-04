"""Lucidity gate — run checks, decide, build render packet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lucid.cognition.lucidity.checks import run_checks
from lucid.cognition.lucidity.config import LucidityConfig
from lucid.cognition.lucidity.decide import decide
from lucid.cognition.lucidity.render_packet import attach_render_packet
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
    _ = ctx
    cfg = (config.thresholds if config and config.thresholds else None) or LucidityConfig()
    checks, confidence = run_checks(inp, cfg)
    output = decide(inp, checks, confidence, cfg)
    output.check_results = checks
    output.confidence_summary = confidence
    return attach_render_packet(output, lucidity_input=inp)
