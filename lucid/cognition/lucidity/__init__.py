"""Lucidity gate — checks, decision, render packet for the decoder."""

from lucid.cognition.lucidity.checks import run_checks
from lucid.cognition.lucidity.config import LucidityConfig
from lucid.cognition.lucidity.gate import LucidityGateConfig, run_lucidity
from lucid.cognition.lucidity.render_packet import attach_render_packet, build_render_packet

__all__ = [
    "LucidityConfig",
    "LucidityGateConfig",
    "attach_render_packet",
    "build_render_packet",
    "run_checks",
    "run_lucidity",
]
