"""Grid mode — copy approved grid artifact."""

from __future__ import annotations

from lucid.cognition.decoder.faithfulness import collect_cited_refs
from lucid.ir.expression import DecoderOutput, SentenceRef
from lucid.ir.lucidity import LucidityRenderPacket, SourceRef


def render_grid(packet: LucidityRenderPacket) -> DecoderOutput:
    grid: list[list[int]] | None = None
    unit_ids: list[str] = []
    refs: list[SourceRef] = []

    for unit in packet.approved_units:
        if unit.unit_type != "artifact":
            continue
        candidate = unit.payload.get("grid_output")
        if isinstance(candidate, list) and candidate and isinstance(candidate[0], list):
            grid = [[int(cell) for cell in row] for row in candidate]
            unit_ids.append(unit.unit_id)
            refs.extend(unit.source_refs)
            break

    sentence_refs = [SentenceRef(sentence_id="grid", unit_ids=unit_ids, source_refs=refs)]
    return DecoderOutput(
        surface_grid=grid,
        render_mode="committed",
        cited_refs=collect_cited_refs(sentence_refs),
        sentence_refs=sentence_refs,
        structured_payload={"grid_output": grid} if grid is not None else None,
        audit_notes=["decoder:grid"],
    )
