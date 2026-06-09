"""Text renderers — committed and uncertainty modes."""

from __future__ import annotations

from lucid.cognition.decoder.compose import ComposeBullet, build_compose_plan
from lucid.cognition.decoder.faithfulness import collect_cited_refs
from lucid.ir.expression import DecoderOutput, SentenceRef
from lucid.cognition.decoder.phrases import humanize
from lucid.ir.lucidity import LucidityRenderPacket, RenderUnit, SourceRef


def _source_refs_for_units(units: list[RenderUnit]) -> list[SourceRef]:
    refs: list[SourceRef] = []
    for unit in units:
        refs.extend(unit.source_refs)
    return refs


def _source_refs_for_bullets(packet: LucidityRenderPacket, bullet: ComposeBullet) -> list[SourceRef]:
    units = [unit for unit in packet.approved_units if unit.unit_id in bullet.unit_ids]
    return _source_refs_for_units(units)


def _render_bullet(bullet: ComposeBullet, *, packet: LucidityRenderPacket) -> tuple[str, list[SourceRef]]:
    payload = bullet.payload
    refs = _source_refs_for_bullets(packet, bullet)

    if bullet.text_intent == "refusal":
        reason = payload.get("refusal_reason", "cannot answer safely")
        return f"I cannot answer safely: {humanize(reason)}.", refs

    if bullet.unit_type == "caveat":
        note = payload.get("summary") or payload.get("kayaking_scope") or payload.get("note")
        if note == "separate_event" or "separate" in str(note).lower():
            return "An earlier part of the text describes a separate event and should not be merged with this reading.", refs
        return f"Note: {humanize(note)}.", refs

    if "bank_sense" in payload:
        frame = payload.get("scope_frame_id") or ""
        sense = humanize(payload["bank_sense"])
        if frame:
            return f"In scope {frame}, bank means {sense}.", refs
        return f"Here, bank means {sense}.", refs

    if "summary" in payload:
        return f"{humanize(payload['summary'])}.", refs

    if "subject_ref" in payload and "predicate_ref" in payload:
        subj = payload.get("subject_ref", "")
        pred = payload.get("predicate_ref", "")
        return f"The committed reading links {subj} to {pred}.", refs

    if payload:
        pairs = ", ".join(f"{key}={humanize(val)}" for key, val in payload.items())
        return f"Approved: {pairs}.", refs

    return "", refs


def render_text_committed(packet: LucidityRenderPacket) -> DecoderOutput:
    plan = build_compose_plan(packet)
    sentences: list[str] = []
    sentence_refs: list[SentenceRef] = []

    for index, bullet in enumerate(plan.bullets):
        if bullet.unit_type == "artifact":
            continue
        text, refs = _render_bullet(bullet, packet=packet)
        if not text:
            continue
        sentences.append(text)
        sentence_refs.append(
            SentenceRef(
                sentence_id=f"s{index}",
                source_refs=refs,
                unit_ids=list(bullet.unit_ids),
            )
        )

    surface = " ".join(sentences).strip()
    if not surface:
        surface = "Committed, but no renderable units were provided."

    return DecoderOutput(
        surface_text=surface,
        render_mode="committed",
        cited_refs=collect_cited_refs(sentence_refs),
        sentence_refs=sentence_refs,
        audit_notes=["decoder:text_committed"],
    )


def render_text_uncertainty(packet: LucidityRenderPacket) -> DecoderOutput:
    out = render_text_committed(packet)
    out.render_mode = "uncertainty"
    if out.surface_text and "not fully" not in out.surface_text.lower():
        out.surface_text = out.surface_text.rstrip(".") + ", but confidence is limited."
    out.uncertainty_presentation = {"style": "concise"}
    out.audit_notes = ["decoder:text_uncertainty"]
    return out


def render_literal_fallback(packet: LucidityRenderPacket) -> DecoderOutput:
    """Boring literal render when faithfulness or polish fails."""
    parts: list[str] = []
    refs: list[SentenceRef] = []
    for index, unit in enumerate(packet.approved_units):
        if unit.unit_type == "artifact":
            continue
        payload = ", ".join(f"{k}={v}" for k, v in unit.payload.items())
        parts.append(f"[{unit.unit_type}] {payload}")
        refs.append(
            SentenceRef(
                sentence_id=f"lit{index}",
                source_refs=list(unit.source_refs),
                unit_ids=[unit.unit_id],
            )
        )
    surface = " ".join(parts) if parts else "Unable to render safely."
    return DecoderOutput(
        surface_text=surface,
        render_mode=packet.render_mode,
        cited_refs=collect_cited_refs(refs),
        sentence_refs=refs,
        audit_notes=["decoder:literal_fallback"],
    )
