"""Surface realization from a realization program."""

from __future__ import annotations

from dataclasses import dataclass, field

from lucid.cognition.decoder.phrases import humanize
from lucid.cognition.decoder.realization_ops import RealizationOp, RealizationProgram
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
    pairs = ", ".join(f"{humanize(key)}: {humanize(value)}" for key, value in payload.items())
    return f"The approved reading is {pairs}." if pairs else "The approved reading is committed."


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
    surface = " ".join(sentence.text for sentence in realized).strip()
    if not surface:
        surface = "No approved text content was available to render."

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
        audit_notes=["decoder:semantic_transducer"],
    )
