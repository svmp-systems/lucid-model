"""Semantic graph construction from lucidity render packets.

This is the decoder's meaning layer. It preserves approved content as typed
nodes instead of turning it straight into template text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lucid.ir.lucidity import ExplicitOmission, LucidityRenderPacket, SourceRef


@dataclass(slots=True)
class SemanticNode:
    node_id: str
    node_type: str
    scope_frame_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    required: bool = True
    source_refs: list[SourceRef] = field(default_factory=list)
    source_unit_ids: list[str] = field(default_factory=list)
    text_intent: str = "answer"


@dataclass(slots=True)
class SemanticEdge:
    source_id: str
    target_id: str
    relation: str
    source_refs: list[SourceRef] = field(default_factory=list)


@dataclass(slots=True)
class SemanticGraph:
    graph_id: str
    render_mode: str
    output_format: str
    nodes: list[SemanticNode] = field(default_factory=list)
    edges: list[SemanticEdge] = field(default_factory=list)
    explicit_omissions: list[ExplicitOmission] = field(default_factory=list)
    provenance_chain: list[str] = field(default_factory=list)


def build_semantic_graph(packet: LucidityRenderPacket) -> SemanticGraph:
    """Translate approved render units and alternatives into a semantic graph."""
    graph = SemanticGraph(
        graph_id=packet.packet_id,
        render_mode=packet.render_mode,
        output_format=packet.output_format,
        explicit_omissions=list(packet.explicit_omissions),
        provenance_chain=list(packet.provenance_chain),
    )

    for unit in packet.approved_units:
        graph.nodes.append(
            SemanticNode(
                node_id=unit.unit_id,
                node_type=unit.unit_type,
                scope_frame_id=unit.scope_frame_id,
                payload=dict(unit.payload),
                confidence=unit.confidence,
                required=unit.required,
                source_refs=list(unit.source_refs),
                source_unit_ids=[unit.unit_id],
                text_intent=unit.text_intent,
            )
        )

    for index, alt in enumerate(packet.preserved_alternatives):
        node_id = str(alt.get("hypothesis_id") or alt.get("basin_id") or f"alt-{index}")
        refs: list[SourceRef] = []
        for ref in alt.get("source_refs") or []:
            if isinstance(ref, SourceRef):
                refs.append(ref)
            elif isinstance(ref, dict) and ref.get("ref_id"):
                refs.append(
                    SourceRef(
                        ref_type=str(ref.get("ref_type") or "basin"),
                        ref_id=str(ref["ref_id"]),
                        scope_frame_id=str(ref.get("scope_frame_id") or ""),
                        role=str(ref.get("role") or "supports"),
                    )
                )
        graph.nodes.append(
            SemanticNode(
                node_id=node_id,
                node_type="alternative",
                scope_frame_id=str(alt.get("scope_frame_id") or ""),
                payload=dict(alt),
                confidence=float(alt.get("confidence") or 0.0),
                required=packet.render_mode == "plural",
                source_refs=refs,
                source_unit_ids=[],
                text_intent="answer",
            )
        )

    return graph
