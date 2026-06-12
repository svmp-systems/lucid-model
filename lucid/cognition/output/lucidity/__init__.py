"""Lucidity gate — checks, decision, render packet for the decoder."""

from lucid.cognition.output.lucidity.checks import run_checks
from lucid.cognition.output.lucidity.config import LucidityConfig
from lucid.cognition.output.lucidity.gate import LucidityGateConfig, run_lucidity
from lucid.cognition.output.lucidity.render_packet import attach_render_packet, build_render_packet

__all__ = [
    "LucidityConfig",
    "LucidityGateConfig",
    "attach_render_packet",
    "build_render_packet",
    "run_checks",
    "run_lucidity",
]
