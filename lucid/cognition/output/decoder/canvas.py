"""Minimal decoder path: approved packet -> rough canvas -> faithful surface."""

from __future__ import annotations

from dataclasses import dataclass, field

from lucid.cognition.output.decoder.faithfulness import collect_cited_refs
from lucid.cognition.output.decoder.phrases import humanize
from lucid.ir.expression import DecoderOutput, SentenceRef
from lucid.ir.lucidity import LucidityRenderPacket, RenderConstraints, SourceRef


@dataclass(slots=True)
class CanvasLine:
    line_id: str
    text: str
    unit_ids: list[str] = field(default_factory=list)
    source_refs: list[SourceRef] = field(default_factory=list)
    required: bool = True


@dataclass(slots=True)
class RenderCanvas:
    packet_id: str
    render_mode: str
    lines: list[CanvasLine] = field(default_factory=list)


def _sentence(text: str) -> str:
    cleaned = " ".join(str(text).strip().split())
    if not cleaned:
        return ""
    return cleaned if cleaned.endswith((".", "!", "?")) else cleaned + "."


def _relation_sentence(payload: dict) -> str:
    subject = humanize(payload.get("subject", "")).strip()
    relation = str(payload.get("relation") or "").strip().lower()
    target = humanize(payload.get("target", "")).strip()
    if not subject or not relation or not target:
        return ""
    if relation in {"type_of", "is_a", "kind_of"}:
        return _sentence(f"{subject} is {target}")
    if relation in {"property", "has_property"}:
        if target.lower().startswith("can "):
            return _sentence(f"{subject} {target}")
        return _sentence(f"{subject} has {target}")
    if relation in {"can", "capability"}:
        return _sentence(f"{subject} can {target}")
    if relation in {"challenge", "limitation"}:
        return _sentence(f"{subject} is limited by {target}")
    if relation in {"uses", "use"}:
        return _sentence(f"{subject} uses {target}")
    if relation in {"enables", "supports"}:
        return _sentence(f"{subject} supports {target}")
    return _sentence(f"{subject} {humanize(relation)} {target}")


def _line_from_payload(payload: dict) -> str:
    relation = _relation_sentence(payload)
    if relation:
        return relation
    if "bank_sense" in payload:
        sense = humanize(payload["bank_sense"])
        scope = str(payload.get("scope_frame_id") or "").strip()
        if scope:
            return _sentence(f"In scope {scope}, bank is being used in the {sense} sense")
        return _sentence(f"Here, bank is being used in the {sense} sense")
    if "summary" in payload:
        unresolved = payload.get("unresolved_slots") or []
        if isinstance(unresolved, list) and unresolved:
            return _sentence(f"{humanize(payload['summary'])} is not fully settled")
        return _sentence(humanize(payload["summary"]))
    if "refusal_reason" in payload:
        return _sentence(str(payload["refusal_reason"]))
    if "action_type" in payload:
        action = humanize(payload.get("action_type", "approved action"))
        target = humanize(payload.get("target_ref", "")).strip()
        return _sentence(
            f"The approved action is {action} for {target}"
            if target
            else f"The approved action is {action}"
        )
    return ""


def build_canvas(packet: LucidityRenderPacket) -> RenderCanvas:
    canvas = RenderCanvas(packet_id=packet.packet_id, render_mode=packet.render_mode)
    for unit in packet.approved_units:
        if unit.unit_type == "artifact":
            continue
        text = _line_from_payload(dict(unit.payload))
        if not text:
            continue
        canvas.lines.append(
            CanvasLine(
                line_id=unit.unit_id,
                text=text,
                unit_ids=[unit.unit_id],
                source_refs=list(unit.source_refs),
                required=unit.required,
            )
        )

    if packet.render_mode == "plural" and packet.preserved_alternatives:
        labels = [
            humanize(
                alt.get("narrative_hint")
                or alt.get("basin_id")
                or alt.get("hypothesis_id")
                or "another reading"
            )
            for alt in packet.preserved_alternatives
        ]
        if len(labels) == 1:
            text = f"This does not force a single reading; one live reading is {labels[0]}."
        elif len(labels) == 2:
            text = (
                "This does not force a single reading; one reading is "
                f"{labels[0]}, and another is {labels[1]}."
            )
        else:
            text = (
                "This does not force a single reading; live readings include "
                + ", ".join(labels[:-1])
                + f", and {labels[-1]}."
            )
        refs: list[SourceRef] = []
        for alt in packet.preserved_alternatives:
            for ref in alt.get("source_refs") or []:
                if isinstance(ref, SourceRef):
                    refs.append(ref)
        canvas.lines.append(CanvasLine(line_id="plural-alternatives", text=text, source_refs=refs))
    return canvas


def realize_canvas(canvas: RenderCanvas, constraints: RenderConstraints) -> DecoderOutput:
    lines = list(canvas.lines)
    required = [line for line in lines if line.required]
    optional = [line for line in lines if not line.required]
    max_sentences = constraints.max_sentences or 0
    if max_sentences > 0:
        lines = required + optional[: max(0, max_sentences - len(required))]

    sentences: list[str] = []
    sentence_refs: list[SentenceRef] = []
    for index, line in enumerate(lines):
        text = _sentence(line.text)
        if not text:
            continue
        sentences.append(text)
        sentence_refs.append(
            SentenceRef(
                sentence_id=f"s{index}",
                source_refs=list(line.source_refs),
                unit_ids=list(line.unit_ids),
            )
        )
    surface = " ".join(sentences).strip()
    if not surface:
        surface = "No approved text content was available to render."
    return DecoderOutput(
        surface_text=surface,
        render_mode=canvas.render_mode,
        cited_refs=collect_cited_refs(sentence_refs),
        sentence_refs=sentence_refs,
        structured_payload={
            "canvas": [
                {
                    "line_id": line.line_id,
                    "unit_ids": line.unit_ids,
                }
                for line in lines
            ]
        },
        audit_notes=["decoder:canvas_builder", "decoder:tiny_denoising_realizer"],
    )
