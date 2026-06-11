"""Refusal mode — safe cannot-answer wording."""

from __future__ import annotations

from lucid.cognition.output.decoder.compose import build_compose_plan
from lucid.ir.expression import DecoderOutput, SentenceRef
from lucid.ir.lucidity import LucidityRenderPacket


def render_refusal(packet: LucidityRenderPacket) -> DecoderOutput:
    plan = build_compose_plan(packet)
    reason = "I cannot answer safely with the evidence available."
    for bullet in plan.bullets:
        raw = bullet.payload.get("refusal_reason")
        if isinstance(raw, str) and raw.strip():
            reason = raw.strip()
            if not reason.endswith("."):
                reason += "."
            break

    sentence_refs = [
        SentenceRef(sentence_id="s0", unit_ids=["refusal"]),
    ]
    return DecoderOutput(
        surface_text=reason,
        render_mode="refusal",
        refused=True,
        refusal_reason=reason,
        sentence_refs=sentence_refs,
        audit_notes=["decoder:refusal"],
    )
