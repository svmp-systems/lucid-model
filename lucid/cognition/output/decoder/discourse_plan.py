"""Discourse planning over approved semantic nodes."""

from __future__ import annotations

from dataclasses import dataclass, field

from lucid.cognition.output.decoder.semantic_graph import SemanticGraph, SemanticNode
from lucid.ir.lucidity import RenderConstraints


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


def plan_discourse(graph: SemanticGraph, constraints: RenderConstraints) -> DiscoursePlan:
    """Order content by communicative function without inventing content."""
    steps: list[DiscourseStep] = []
    nodes = sorted(graph.nodes, key=_node_priority)

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
