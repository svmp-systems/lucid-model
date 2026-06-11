"""Surface realization from a realization program."""

from __future__ import annotations

from dataclasses import dataclass, field

from lucid.cognition.output.decoder.phrases import humanize
from lucid.cognition.output.decoder.realization_ops import RealizationOp, RealizationProgram
from lucid.ir.expression import DecoderOutput, SentenceRef
from lucid.ir.lucidity import RenderConstraints, SourceRef


@dataclass(slots=True)
class RealizedSentence:
    text: str
    op_id: str
    source_refs: list[SourceRef] = field(default_factory=list)
    unit_ids: list[str] = field(default_factory=list)


def _realize_bank_sense(payload: dict) -> str | None:
    if "bank_sense" not in payload:
        return None
    sense = humanize(payload["bank_sense"])
    scope = str(payload.get("scope_frame_id") or "").strip()
    if sense.lower().startswith(("a ", "an ", "the ")):
        phrase = f"as {sense}"
    else:
        phrase = f"in the {sense} sense"
    if scope:
        return f"In scope {scope}, bank is being used {phrase}."
    return f"Here, bank is being used {phrase}."


def _realize_summary(payload: dict) -> str | None:
    if "summary" not in payload:
        return None
    return f"{humanize(payload['summary'])}."


def _join_surface_phrase(surfaces: list[str]) -> str:
    clean = [str(item).strip() for item in surfaces if str(item).strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return ", ".join(clean[:-1]) + f", and {clean[-1]}"


def _bank_sense_unresolved(payload: dict) -> bool:
    unresolved = {str(item).strip().lower() for item in payload.get("unresolved_slots") or []}
    return "bank_sense" in unresolved or "bank" in unresolved


def _realize_frame_summary(payload: dict) -> str:
    summary = str(payload.get("summary") or "").strip()
    surfaces = [str(item).strip() for item in payload.get("member_evidence_surfaces") or [] if str(item).strip()]
    phrase = summary or _join_surface_phrase(surfaces)
    if not phrase:
        return ""
    if _bank_sense_unresolved(payload):
        return f"{phrase.rstrip('.')}, but the sense of \"bank\" is not fully settled."
    return f"{phrase.rstrip('.')}."


def _realize_basin_primary(payload: dict) -> str:
    """Skip raw basin telemetry when frame summaries carry readable content."""
    if "energy" in payload and "basin_id" in payload:
        return ""
    return _realize_claim(payload)


def _realize_claim(payload: dict) -> str:
    bank = _realize_bank_sense(payload)
    if bank:
        return bank
    summary = _realize_summary(payload)
    if summary:
        return summary
    if "subject_ref" in payload and "predicate_ref" in payload:
        return (
            f"The approved reading links {humanize(payload['subject_ref'])} "
            f"to {humanize(payload['predicate_ref'])}."
        )
    if "basin_id" in payload and "energy" in payload:
        return _realize_basin_primary(payload)
    internal_keys = {
        "frame_id",
        "frame_type",
        "basin_id",
        "roles",
        "member_evidence_refs",
        "member_evidence_surfaces",
        "unresolved_slots",
        "supporting_trace_ids",
        "supporting_frame_ids",
        "scope_frame_ids",
        "energy",
        "margin_vs_next",
    }
    if set(payload.keys()).issubset(internal_keys):
        return ""
    pairs = ", ".join(f"{humanize(key)}: {humanize(value)}" for key, value in payload.items())
    return f"The approved reading is {pairs}." if pairs else ""


def _realize_scope_boundary(payload: dict) -> str:
    note = payload.get("kayaking_scope") or payload.get("summary") or payload.get("note")
    if str(note).strip().lower() == "separate_event" or "separate" in str(note).lower():
        return "The kayaking context belongs to a separate earlier event, so it is not merged with this reading."
    return f"Boundary note: {humanize(note)}."


def _realize_alternatives(payload: dict) -> str:
    alternatives = payload.get("alternatives") or []
    labels = [str(item.get("label") or "another reading") for item in alternatives]
    if not labels:
        return "Multiple interpretations remain live; none is committed."
    if len(labels) == 1:
        return f"This does not force a single reading; one live reading is {labels[0]}."
    if len(labels) == 2:
        return f"This does not force a single reading; one reading is {labels[0]}, and another is {labels[1]}."
    return "This does not force a single reading; live readings include " + ", ".join(labels[:-1]) + f", and {labels[-1]}."


def _realize_refusal(payload: dict) -> str:
    reason = str(payload.get("refusal_reason") or "I cannot answer safely with the evidence available.")
    return reason if reason.endswith(".") else reason + "."


def _realize_action(payload: dict) -> str:
    action_type = humanize(payload.get("action_type", "approved action"))
    target = payload.get("target_ref")
    if target:
        return f"The approved action is {action_type} for {humanize(target)}."
    return f"The approved action is {action_type}."


def _text_for_op(op: RealizationOp) -> str:
    if op.op_type == "realize_frame_summary":
        return _realize_frame_summary(op.payload)
    if op.op_type in {"realize_claim", "realize_reason", "realize_literal"}:
        return _realize_claim(op.payload)
    if op.op_type == "realize_scope_boundary":
        return _realize_scope_boundary(op.payload)
    if op.op_type == "realize_alternatives":
        return _realize_alternatives(op.payload)
    if op.op_type == "realize_refusal":
        return _realize_refusal(op.payload)
    if op.op_type == "realize_action":
        return _realize_action(op.payload)
    return _realize_claim(op.payload)


def _compose_committed_surface(realized: list[RealizedSentence], program: RealizationProgram) -> str | None:
    """Merge frame evidence into one readable line when lucidity committed."""
    if program.render_mode != "committed" or not realized:
        return None
    chunks: list[str] = []
    bank_unresolved = False
    for sentence in realized:
        text = sentence.text.strip()
        if not text:
            continue
        if "bank" in text.lower() and "not fully settled" in text.lower():
            bank_unresolved = True
            text = text.split(", but the sense")[0].strip().rstrip(".")
        chunks.append(text.rstrip("."))
    if len(chunks) <= 1:
        return None
    merged = ". ".join(chunks)
    if not merged:
        return None
    if bank_unresolved:
        return f"{merged}, though the sense of \"bank\" is not fully settled."
    return f"{merged}."


def _clip_sentences(sentences: list[RealizedSentence], constraints: RenderConstraints) -> list[RealizedSentence]:
    max_sentences = constraints.max_sentences or 0
    if max_sentences > 0:
        return sentences[:max_sentences]
    return sentences


def realize_surface(program: RealizationProgram, constraints: RenderConstraints) -> DecoderOutput:
    """Linearize realization operations into surface output and sentence refs."""
    realized: list[RealizedSentence] = []
    for op in program.ops:
        if op.op_type == "realize_artifact":
            continue
        text = _text_for_op(op).strip()
        if not text:
            continue
        realized.append(
            RealizedSentence(
                text=text,
                op_id=op.op_id,
                source_refs=list(op.source_refs),
                unit_ids=list(op.source_unit_ids),
            )
        )
    realized = _clip_sentences(realized, constraints)

    sentence_refs = [
        SentenceRef(
            sentence_id=f"s{index}",
            source_refs=sentence.source_refs,
            unit_ids=sentence.unit_ids,
        )
        for index, sentence in enumerate(realized)
    ]
    composed = _compose_committed_surface(realized, program)
    if composed:
        surface = composed
        consumed_unit_ids = [
            uid
            for op in program.ops
            if op.op_type != "realize_artifact"
            for uid in op.source_unit_ids
        ]
        consumed_refs: list[SourceRef] = []
        seen_refs: set[tuple[str, str, str]] = set()
        for op in program.ops:
            if op.op_type == "realize_artifact":
                continue
            for ref in op.source_refs:
                key = (ref.ref_type, ref.ref_id, ref.scope_frame_id)
                if key in seen_refs:
                    continue
                seen_refs.add(key)
                consumed_refs.append(ref)
        sentence_refs = [
            SentenceRef(
                sentence_id="s0",
                source_refs=consumed_refs or [ref for sentence in realized for ref in sentence.source_refs],
                unit_ids=consumed_unit_ids or [uid for sentence in realized for uid in sentence.unit_ids],
            )
        ]
    else:
        surface = " ".join(sentence.text for sentence in realized).strip()
    if not surface:
        surface = "No approved text content was available to render."

    notes = ["decoder:semantic_transducer"]
    if composed:
        notes.append("decoder:composed_committed")

    return DecoderOutput(
        surface_text=surface,
        render_mode=program.render_mode,
        sentence_refs=sentence_refs,
        structured_payload={
            "realization_program": [
                {
                    "op_id": op.op_id,
                    "op_type": op.op_type,
                    "step_id": op.step_id,
                    "unit_ids": op.source_unit_ids,
                }
                for op in program.ops
            ]
        },
        audit_notes=notes,
    )
