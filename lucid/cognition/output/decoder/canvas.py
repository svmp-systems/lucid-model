"""Minimal decoder path: approved packet -> fluent lines -> faithful surface."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lucid.cognition.output.decoder.faithfulness import collect_cited_refs
from lucid.cognition.output.decoder.fluent import compose_fluent_lines
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


def _split_sentences(text: str) -> list[str]:
    return [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text.strip()) if chunk.strip()]


def build_canvas(packet: LucidityRenderPacket) -> RenderCanvas:
    canvas = RenderCanvas(packet_id=packet.packet_id, render_mode=packet.render_mode)
    for index, line in enumerate(compose_fluent_lines(packet.approved_units)):
        canvas.lines.append(
            CanvasLine(
                line_id=line.unit_ids[0] if line.unit_ids else f"fluent-{index}",
                text=line.text,
                unit_ids=list(line.unit_ids),
                source_refs=list(line.source_refs),
                required=line.required,
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
    sentence_index = 0
    for line in lines:
        chunks = _split_sentences(_sentence(line.text))
        if not chunks:
            continue
        for chunk in chunks:
            text = _sentence(chunk)
            if not text:
                continue
            sentences.append(text)
            sentence_refs.append(
                SentenceRef(
                    sentence_id=f"s{sentence_index}",
                    source_refs=list(line.source_refs),
                    unit_ids=list(line.unit_ids),
                )
            )
            sentence_index += 1
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
        audit_notes=["decoder:canvas_builder", "decoder:fluent_realizer"],
    )
