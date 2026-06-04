"""Structural faithfulness over semantic graph and realization program."""

from __future__ import annotations

from lucid.cognition.decoder.realization_ops import RealizationProgram
from lucid.cognition.decoder.semantic_graph import SemanticGraph
from lucid.ir.expression import FaithfulnessReport
from lucid.ir.lucidity import DecoderPolicy


def check_structural_faithfulness(
    *,
    graph: SemanticGraph,
    program: RealizationProgram,
    policy: DecoderPolicy,
) -> FaithfulnessReport:
    """Check operation-level safety, independent of fragile surface tokens."""
    violations: list[str] = []
    omitted: list[str] = []

    approved_node_ids = {node.node_id for node in graph.nodes}
    required_unit_ids = {
        unit_id
        for node in graph.nodes
        if node.required
        for unit_id in (node.source_unit_ids or [node.node_id])
    }
    consumed_unit_ids = {
        unit_id
        for op in program.ops
        for unit_id in op.source_unit_ids
    }

    for op in program.ops:
        if op.source_unit_ids:
            unknown = set(op.source_unit_ids) - approved_node_ids - required_unit_ids
            if unknown:
                violations.append("unknown_unit:" + ",".join(sorted(unknown)))
        if op.function == "alternative" and policy.forbid_single_answer:
            continue
        if op.function == "answer" and graph.render_mode == "plural" and policy.forbid_single_answer:
            violations.append("answer_in_plural_mode")

    missing = required_unit_ids - consumed_unit_ids
    if graph.render_mode == "committed" and missing:
        omitted.extend(sorted(missing))

    passed = not violations and not omitted
    score = 1.0 if passed else max(0.0, 1.0 - 0.2 * (len(violations) + len(omitted)))
    return FaithfulnessReport(
        passed=passed,
        omitted_required_units=omitted,
        policy_violations=violations,
        reparse_match_score=score,
    )
