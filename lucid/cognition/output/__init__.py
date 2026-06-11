"""Output stages: lucidity gate, optional projector, decoder surfaces."""

from lucid.cognition.output.decoder import run_decoder
from lucid.cognition.output.lucidity import (
    LucidityGateConfig,
    attach_render_packet,
    build_render_packet,
    run_lucidity,
)
from lucid.cognition.output.projector import run_projector

__all__ = [
    "LucidityGateConfig",
    "attach_render_packet",
    "build_render_packet",
    "run_decoder",
    "run_lucidity",
    "run_projector",
]
