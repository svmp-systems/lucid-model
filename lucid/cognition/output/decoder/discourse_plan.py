"""Discourse planning over approved semantic nodes."""

from __future__ import annotations

from dataclasses import dataclass, field

from lucid.cognition.output.decoder.fluent import extract_relation_claims, group_relation_claims
from lucid.cognition.output.decoder.semantic_graph import SemanticGraph, SemanticNode
from lucid.ir.lucidity import RenderConstraints, RenderUnit


@dataclass(slots=True)
class DiscourseStep:
    step_id: str
    function: str
    nodes: list[SemanticNode] = field(default_factory=list)
    required: bool = True


@dataclass(slots=True)
class DiscoursePlan:
    plan_id: str
    render_mode: str
    output_format: str
    steps: list[DiscourseStep] = field(default_factory=list)


def _node_priority(node: SemanticNode) -> tuple[int, str]:
    by_type = {
        "claim": 0,
        "artifact": 0,
        "action": 0,
        "frame_summary": 1,
        "alternative": 2,
        "caveat": 3,
    }
    by_intent = {
        "answer": 0,
        "reason": 1,
        "caveat": 2,
        "next_step": 3,
        "refusal": 0,
    }
    return (by_type.get(node.node_type, 8) + by_intent.get(node.text_intent, 8), node.node_id)


def _step_function(node: SemanticNode, render_mode: str) -> str:
    if node.text_intent == "refusal" or render_mode == "refusal":
        return "refusal"
    if render_mode == "plural" or node.node_type == "alternative":
        return "alternative"
    if node.node_type == "caveat":
        return "scope_boundary"
    if node.node_type == "artifact":
        return "artifact"
    if node.node_type == "action":
        return "action"
    if node.text_intent == "reason":
        return "reason"
    if node.node_type == "frame_summary":
        return "frame_summary"
    return "answer"


def _node_to_render_unit(node: SemanticNode) -> RenderUnit:
    return RenderUnit(
        unit_id=node.node_id,
        unit_type=node.node_type,
        scope_frame_id=node.scope_frame_id,
        text_intent=node.text_intent,
        payload=dict(node.payload),
        confidence=node.confidence,
        required=node.required,
        source_refs=list(node.source_refs),
    )


def _merge_relation_nodes(nodes: list[SemanticNode]) -> list[SemanticNode]:
    """Collapse repeated subject/relation claims into one node with multiple targets."""
    relation_nodes = [
        node
        for node in nodes
        if str(node.payload.get("subject") or "").strip() and str(node.payload.get("relation") or "").strip()
    ]
    other_nodes = [node for node in nodes if node not in relation_nodes]
    if not relation_nodes:
        return list(nodes)

    claims = extract_relation_claims([_node_to_render_unit(node) for node in relation_nodes])
    grouped = group_relation_claims(claims)
    merged_by_anchor: dict[str, SemanticNode] = {}
    member_to_anchor: dict[str, str] = {}

    for _subject, _relation, group in grouped:
        members = [node for node in relation_nodes if node.node_id in {claim.unit_id for claim in group}]
        if not members:
            continue
        anchor_id = members[0].node_id
        seen_refs: set[tuple[str, str, str]] = set()
        refs: list = []
        for node in members:
            member_to_anchor[node.node_id] = anchor_id
            for ref in node.source_refs:
                key = (ref.ref_type, ref.ref_id, ref.scope_frame_id)
                if key in seen_refs:
                    continue
                seen_refs.add(key)
                refs.append(ref)
        merged_by_anchor[anchor_id] = SemanticNode(
            node_id=anchor_id,
            node_type=members[0].node_type,
            scope_frame_id=members[0].scope_frame_id,
            payload={
                **dict(members[0].payload),
                "target": [claim.target for claim in group],
            },
            confidence=max(node.confidence for node in members),
            required=any(node.required for node in members),
            source_refs=refs,
            source_unit_ids=[node.node_id for node in members],
            text_intent=members[0].text_intent,
        )

    ordered: list[SemanticNode] = []
    seen: set[str] = set()
    for node in nodes:
        if node in other_nodes:
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            ordered.append(node)
            continue
        anchor_id = member_to_anchor.get(node.node_id, node.node_id)
        if anchor_id in seen:
            continue
        seen.add(anchor_id)
        ordered.append(merged_by_anchor.get(anchor_id, node))
    return ordered


def plan_discourse(graph: SemanticGraph, constraints: RenderConstraints) -> DiscoursePlan:
    """Order content by communicative function without inventing content."""
    steps: list[DiscourseStep] = []
    nodes = _merge_relation_nodes(sorted(graph.nodes, key=_node_priority))
    if graph.render_mode == "plural":
        alternatives = [node for node in nodes if node.node_type == "alternative"]
        others = [node for node in nodes if node.node_type != "alternative"]
        if alternatives:
            steps.append(
                DiscourseStep(
                    step_id="plural-alternatives",
                    function="alternative",
                    nodes=alternatives,
                    required=True,
                )
            )
        nodes = others

    for node in nodes:
        steps.append(
            DiscourseStep(
                step_id=node.node_id,
                function=_step_function(node, graph.render_mode),
                nodes=[node],
                required=node.required,
            )
        )

    max_steps = constraints.max_sentences or 0
    if max_steps > 0:
        required = [step for step in steps if step.required]
        optional = [step for step in steps if not step.required]
        steps = required + optional[: max(0, max_steps - len(required))]

    return DiscoursePlan(
        plan_id=graph.graph_id,
        render_mode=graph.render_mode,
        output_format=graph.output_format,
        steps=steps,
    )
