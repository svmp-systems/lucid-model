"""Text renderers — committed and uncertainty modes."""

from __future__ import annotations

from lucid.cognition.output.decoder.compose import ComposeBullet, build_compose_plan
from lucid.cognition.output.decoder.faithfulness import collect_cited_refs
from lucid.ir.expression import DecoderOutput, SentenceRef
from lucid.cognition.output.decoder.phrases import humanize
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

    if bullet.unit_type == "frame_summary":
        surfaces = [str(item).strip() for item in payload.get("member_evidence_surfaces") or [] if str(item).strip()]
        phrase = str(payload.get("summary") or "").strip() or ", ".join(surfaces)
        if not phrase:
            return "", refs
        unresolved = {str(item).strip().lower() for item in payload.get("unresolved_slots") or []}
        if "bank_sense" in unresolved or "bank" in unresolved:
            return f"{phrase.rstrip('.')}, but the sense of \"bank\" is not fully settled.", refs
        return f"{phrase.rstrip('.')}.", refs

    if bullet.unit_type == "claim" and "basin_id" in payload and "energy" in payload:
        return "", refs

    if "bank_sense" in payload:
        frame = payload.get("scope_frame_id") or ""
        sense = humanize(payload["bank_sense"])
        if frame:
            return f"In scope {frame}, bank means {sense}.", refs
        return f"Here, bank means {sense}.", refs

    if "summary" in payload:
        return f"{humanize(payload['summary'])}.", refs

    if payload.get("speech_kind") and payload.get("summary"):
        return f"{humanize(payload['summary'])}.", refs

    if "subject_ref" in payload and "predicate_ref" in payload:
        subj = payload.get("subject_ref", "")
        pred = payload.get("predicate_ref", "")
        return f"The committed reading links {subj} to {pred}.", refs

    if "subject" in payload and "relation" in payload and "target" in payload:
        from lucid.cognition.output.decoder.fluent import realize_relation_group

        target = payload["target"]
        targets = target if isinstance(target, list) else [target]
        text = realize_relation_group(
            str(payload.get("subject") or ""),
            str(payload.get("relation") or ""),
            [str(item) for item in targets if str(item).strip()],
        )
        if text:
            return text, refs

    if payload:
        summary = str(payload.get("summary") or "").strip()
        if summary:
            return f"{humanize(summary)}.", refs
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
    """Readable fallback when polish or strict faithfulness checks fail."""
    out = render_text_committed(packet)
    chunks = [part.strip().rstrip(".") for part in out.surface_text.split(".") if part.strip()]
    if len(chunks) > 1:
        merged = ", ".join(chunks[:-1]) + f", and {chunks[-1]}"
        bank_note = any("bank" in chunk.lower() and "not fully settled" in chunk.lower() for chunk in chunks)
        if bank_note and "not fully settled" not in merged.lower():
            merged += ", though the sense of \"bank\" is not fully settled"
        out.surface_text = f"{merged}."
    out.audit_notes = ["decoder:literal_fallback"]
    return out
