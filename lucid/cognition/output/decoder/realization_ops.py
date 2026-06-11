"""Meaning-preserving realization operations."""

from __future__ import annotations

from dataclasses import dataclass, field

from lucid.cognition.output.decoder.discourse_plan import DiscoursePlan, DiscourseStep
from lucid.cognition.output.decoder.phrases import humanize
from lucid.ir.lucidity import SourceRef


@dataclass(slots=True)
class RealizationOp:
    op_id: str
    op_type: str
    step_id: str
    function: str
    payload: dict = field(default_factory=dict)
    source_refs: list[SourceRef] = field(default_factory=list)
    source_unit_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RealizationProgram:
    program_id: str
    render_mode: str
    output_format: str
    ops: list[RealizationOp] = field(default_factory=list)


def _refs_for_step(step: DiscourseStep) -> list[SourceRef]:
    refs: list[SourceRef] = []
    seen: set[tuple[str, str, str]] = set()
    for node in step.nodes:
        for ref in node.source_refs:
            key = (ref.ref_type, ref.ref_id, ref.scope_frame_id)
            if key in seen:
                continue
            seen.add(key)
            refs.append(ref)
    return refs


def _unit_ids_for_step(step: DiscourseStep) -> list[str]:
    ids: list[str] = []
    for node in step.nodes:
        ids.extend(node.source_unit_ids)
    return list(dict.fromkeys(ids))


def _alternative_payload(step: DiscourseStep) -> dict:
    alternatives = []
    for node in step.nodes:
        hint = (
            node.payload.get("narrative_hint")
            or node.payload.get("basin_id")
            or node.payload.get("hypothesis_id")
            or "another reading"
        )
        scope = node.scope_frame_id or node.payload.get("scope_frame_id") or ""
        alternatives.append({"label": humanize(hint), "scope": scope})
    return {"alternatives": alternatives}


def _payload_for_step(step: DiscourseStep) -> dict:
    if step.function == "alternative":
        return _alternative_payload(step)
    if not step.nodes:
        return {}
    payload = dict(step.nodes[0].payload)
    if step.nodes[0].scope_frame_id and "scope_frame_id" not in payload:
        payload["scope_frame_id"] = step.nodes[0].scope_frame_id
    return payload


def plan_realization(plan: DiscoursePlan) -> RealizationProgram:
    """Choose realization operations over discourse steps."""
    ops: list[RealizationOp] = []
    for index, step in enumerate(plan.steps):
        op_type = {
            "answer": "realize_claim",
            "reason": "realize_reason",
            "frame_summary": "realize_frame_summary",
            "scope_boundary": "realize_scope_boundary",
            "alternative": "realize_alternatives",
            "refusal": "realize_refusal",
            "artifact": "realize_artifact",
            "action": "realize_action",
        }.get(step.function, "realize_literal")
        ops.append(
            RealizationOp(
                op_id=f"op-{index}",
                op_type=op_type,
                step_id=step.step_id,
                function=step.function,
                payload=_payload_for_step(step),
                source_refs=_refs_for_step(step),
                source_unit_ids=_unit_ids_for_step(step),
            )
        )
    return RealizationProgram(
        program_id=plan.plan_id,
        render_mode=plan.render_mode,
        output_format=plan.output_format,
        ops=ops,
    )
