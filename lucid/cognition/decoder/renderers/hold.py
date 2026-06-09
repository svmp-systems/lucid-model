"""Hold mode — no external answer."""

from __future__ import annotations

from lucid.ir.expression import DecoderOutput


def render_hold() -> DecoderOutput:
    return DecoderOutput(
        surface_text="",
        render_mode="hold",
        audit_notes=["decoder:hold"],
    )
