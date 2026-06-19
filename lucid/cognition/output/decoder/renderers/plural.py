"""Plural mode — show live alternatives without picking a winner."""

from __future__ import annotations

from lucid.cognition.output.decoder.compose import build_compose_plan
from lucid.cognition.output.decoder.faithfulness import collect_cited_refs
from lucid.cognition.output.decoder.renderers.text import _render_bullet
from lucid.ir.expression import DecoderOutput, SentenceRef
from lucid.ir.lucidity import LucidityRenderPacket, SourceRef


def _phrase_alternative(alt: dict) -> str:
    hint = str(alt.get("narrative_hint") or alt.get("basin_id") or "another reading").strip()
    frame = str(alt.get("scope_frame_id") or "").strip()
    if frame:
        return f"{hint} (scope {frame})"
    return hint


def render_plural(packet: LucidityRenderPacket) -> DecoderOutput:
    plan = build_compose_plan(packet)
    sentences: list[str] = []
    sentence_refs: list[SentenceRef] = []

    for index, bullet in enumerate(plan.bullets):
        text, refs = _render_bullet(bullet, packet=packet)
        if text:
            sentences.append(text)
            sentence_refs.append(
                SentenceRef(
                    sentence_id=f"s{index}",
                    source_refs=refs,
                    unit_ids=list(bullet.unit_ids),
                )
            )

    alts = plan.preserved_alternatives
    if alts:
        parts = [_phrase_alternative(item) for item in alts[:4]]
        alt_refs: list[SourceRef] = []
        for item in alts:
            raw_refs = item.get("source_refs") or []
            for ref in raw_refs:
                if isinstance(ref, SourceRef):
                    alt_refs.append(ref)
                elif isinstance(ref, dict) and ref.get("ref_id"):
                    alt_refs.append(
                        SourceRef(
                            ref_type=str(ref.get("ref_type") or "basin"),
                            ref_id=str(ref["ref_id"]),
                            scope_frame_id=str(ref.get("scope_frame_id") or ""),
                            role=str(ref.get("role") or ""),
                        )
                    )
        lead = "This does not force a single reading."
        if len(parts) == 1:
            body = f"One live reading is {parts[0]}."
        elif len(parts) == 2:
            body = f"One reading is {parts[0]}, and another is {parts[1]}."
        else:
            body = "Live readings include " + ", ".join(parts[:-1]) + f", and {parts[-1]}."
        sentences.append(f"{lead} {body}")
        sentence_refs.append(
            SentenceRef(
                sentence_id=f"s{len(sentence_refs)}",
                source_refs=alt_refs,
                unit_ids=[],
            )
        )

    surface = " ".join(sentences).strip()
    if not surface:
        surface = "Multiple interpretations remain possible; none is committed."

    return DecoderOutput(
        surface_text=surface,
        render_mode="plural",
        cited_refs=collect_cited_refs(sentence_refs),
        sentence_refs=sentence_refs,
        uncertainty_presentation={"alternatives": len(alts)},
        audit_notes=["decoder:plural"],
    )
