"""Build CommittedState and preserved hypotheses from lucidity inputs."""

from __future__ import annotations

from uuid import uuid4

from lucid.cognition.output.lucidity.config import normalize_task_intent
from lucid.ir.basins import BasinAssembly, BasinOutput, CandidateBasinState
from lucid.ir.binding import CandidateFrame
from lucid.ir.common import CommitShape
from lucid.ir.context_op import ContextFrame
from lucid.ir.lucidity import (
    CommittedState,
    FrameCommit,
    LucidityInput,
    PreservedHypothesis,
    RenderUnit,
    SourceRef,
)
from lucid.ir.projector import ProjectorOutput, ProjectorRollout


def _top_basin_states(basins: BasinOutput, limit: int = 3) -> list[CandidateBasinState]:
    ranked = sorted(
        basins.candidate_basin_states,
        key=lambda state: state.energy,
        reverse=True,
    )
    return ranked[:limit]


def preserved_hypotheses_from_basins(inp: LucidityInput, *, limit: int = 3) -> list[PreservedHypothesis]:
    rows: list[PreservedHypothesis] = []
    for state in _top_basin_states(inp.basin_output, limit=limit):
        frame_id = state.supporting_frame_ids[0] if state.supporting_frame_ids else ""
        hint_parts = [state.basin_id]
        if state.supporting_frame_ids:
            hint_parts.append("frames=" + ",".join(state.supporting_frame_ids[:3]))
        if state.supporting_trace_ids:
            hint_parts.append("traces=" + ",".join(state.supporting_trace_ids[:5]))
        rows.append(
            PreservedHypothesis(
                hypothesis_id=state.basin_id or str(uuid4()),
                frame_id=frame_id,
                basin_id=state.basin_id,
                narrative_hint="; ".join(part for part in hint_parts if part),
                confidence=max(0.0, state.energy),
            )
        )
    return rows


def _frame_for_scope(frames: list[CandidateFrame], context_frames: list[ContextFrame]) -> dict[str, CandidateFrame]:
    mapping: dict[str, CandidateFrame] = {}
    for ctx in context_frames:
        members = ctx.member_frame_ids or [frame.frame_id for frame in frames]
        for frame_id in members:
            frame = next((item for item in frames if item.frame_id == frame_id), None)
            if frame is not None:
                mapping[ctx.context_frame_id] = frame
    if not mapping:
        for frame in frames:
            mapping[frame.frame_id] = frame
    return mapping


def _pick_commit_shape(
    inp: LucidityInput,
    *,
    assembly: BasinAssembly | None,
) -> CommitShape:
    task = normalize_task_intent(inp.task_intent)
    if task == "solve_grid" and assembly is not None:
        return CommitShape.ASSEMBLY

    scoped = _frame_for_scope(inp.binding_output.candidate_frames, inp.context_op_output.context_frames)
    if len(scoped) > 1:
        return CommitShape.PER_FRAME
    if assembly is not None and len(assembly.member_basin_ids) > 1:
        return CommitShape.ASSEMBLY
    return CommitShape.SINGLE


def _best_assembly(basins: BasinOutput) -> BasinAssembly | None:
    if not basins.basin_assemblies:
        return None
    return max(basins.basin_assemblies, key=lambda item: item.combined_energy)


def _rollout_artifact(projection: ProjectorOutput | None) -> tuple[dict, ProjectorRollout | None]:
    if projection is None:
        return {}, None
    rollout = None
    if projection.best_rollout_id:
        rollout = next((r for r in projection.rollouts if r.rollout_id == projection.best_rollout_id), None)
    if rollout is None and projection.rollouts:
        rollout = projection.rollouts[0]
    artifact = dict(rollout.implied_artifact) if rollout is not None else {}
    return artifact, rollout


def _unit_surfaces(inp: LucidityInput) -> dict[str, str]:
    return {
        unit.unit_id: unit.surface
        for unit in inp.perceptual_evidence_graph.candidate_units
        if unit.unit_id and unit.surface
    }


def _graph_node_label(frame: CandidateFrame, node_id: str) -> str:
    for graph in frame.local_graphs:
        for node in graph.nodes:
            if node.node_id == node_id:
                return node.label or node.node_id
    return node_id


def _norm_label(value: object) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", " ").split())


def _graph_relation_render_units(inp: LucidityInput, *, subject_filter: str = "") -> list[RenderUnit]:
    units: list[RenderUnit] = []
    index = 0
    seen_claims: set[tuple[str, str, str]] = set()
    subject_filter_key = _norm_label(subject_filter)
    for frame in inp.binding_output.candidate_frames:
        for graph in frame.local_graphs:
            for edge in graph.edges:
                if edge.edge_kind != "relation" or not edge.label:
                    continue
                if edge.edge_id.startswith("alias_"):
                    continue
                subject = _graph_node_label(frame, edge.source_id)
                target = _graph_node_label(frame, edge.target_id)
                if not subject or not target:
                    continue
                if subject_filter_key and _norm_label(subject) != subject_filter_key:
                    continue
                claim_key = (
                    subject.strip().lower(),
                    edge.label.strip().lower(),
                    target.strip().lower(),
                )
                if claim_key in seen_claims:
                    continue
                seen_claims.add(claim_key)
                refs = [
                    SourceRef(ref_type="frame", ref_id=frame.frame_id, role="supports"),
                    *[
                        SourceRef(ref_type="source", ref_id=ref, role="supports")
                        for ref in edge.provenance_refs
                        if ref
                    ],
                    *[
                        SourceRef(ref_type="evidence", ref_id=ref, role="supports")
                        for ref in edge.source_unit_ids
                        if ref
                    ],
                ]
                units.append(
                    RenderUnit(
                        unit_id=f"graph-claim-{index}",
                        unit_type="claim",
                        scope_frame_id=frame.frame_id,
                        text_intent="answer",
                        payload={
                            "subject": subject,
                            "relation": edge.label,
                            "target": target,
                            "summary": f"{subject} {edge.label} {target}",
                            "graph_id": graph.graph_id,
                            "edge_id": edge.edge_id,
                            "inferred": edge.inferred,
                        },
                        confidence=edge.confidence,
                        required=True,
                        source_refs=refs,
                    )
                )
                index += 1
    return units


def _source_refs_from_ids(
    refs: list[str],
    *,
    ref_type: str,
    role: str = "supports",
) -> list[SourceRef]:
    return [
        SourceRef(ref_type=ref_type, ref_id=ref, role=role)
        for ref in refs
        if str(ref).strip()
    ]


def _basin_relation_render_units(state: CandidateBasinState | None) -> list[RenderUnit]:
    if state is None:
        return []
    payload = dict(state.quantized_payload)
    relations = payload.get("relations") if isinstance(payload.get("relations"), list) else []
    subject = str(payload.get("canonical_label") or payload.get("concept_id") or state.basin_id).strip()
    if not subject:
        return []
    units: list[RenderUnit] = []
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            continue
        rel_name = str(relation.get("relation") or "").strip()
        target = str(relation.get("target") or "").strip()
        if not rel_name or not target:
            continue
        rel_sources = [
            str(ref)
            for ref in relation.get("source_refs") or state.source_refs
            if str(ref).strip()
        ]
        refs = [
            SourceRef(ref_type="basin", ref_id=state.basin_id, role="supports"),
            *[
                SourceRef(ref_type="source", ref_id=ref, role="supports")
                for ref in rel_sources
            ],
        ]
        if index < len(state.relation_handles):
            refs.append(
                SourceRef(
                    ref_type="evidence",
                    ref_id=state.relation_handles[index],
                    role="supports",
                )
            )
        units.append(
            RenderUnit(
                unit_id=f"basin-claim-{index}",
                unit_type="claim",
                scope_frame_id=state.scope_frame_ids[0] if state.scope_frame_ids else "",
                text_intent="answer",
                payload={
                    "subject": subject,
                    "relation": rel_name,
                    "target": target,
                    "basin_id": state.basin_id,
                    "source": "basin_quantized_payload",
                },
                confidence=max(0.0, min(1.0, float(relation.get("confidence", state.energy)))),
                required=True,
                source_refs=refs,
            )
        )
    return units


def _frame_render_unit(
    *,
    index: int,
    frame: CandidateFrame,
    ctx_id: str,
    basin_id: str,
    inp: LucidityInput,
    required: bool = True,
) -> RenderUnit:
    surfaces = _unit_surfaces(inp)
    role_rows: list[dict] = []
    refs: list[SourceRef] = [
        SourceRef(ref_type="frame", ref_id=frame.frame_id, scope_frame_id=ctx_id, role="supports"),
    ]
    if basin_id:
        refs.append(SourceRef(ref_type="basin", ref_id=basin_id, scope_frame_id=ctx_id, role="supports"))

    for role, trace_id in sorted(frame.role_assignments.items()):
        evidence_refs = frame.slot_evidence_refs.get(role, [])
        role_rows.append(
            {
                "role": role,
                "trace_id": trace_id,
                "evidence_refs": list(evidence_refs),
                "evidence_surfaces": [surfaces[ref] for ref in evidence_refs if ref in surfaces],
            }
        )
        if trace_id:
            refs.append(SourceRef(ref_type="trace", ref_id=trace_id, scope_frame_id=ctx_id, role="supports"))
        for ref in evidence_refs:
            refs.append(SourceRef(ref_type="evidence", ref_id=ref, scope_frame_id=ctx_id, role="supports"))

    ordered_surfaces: list[str] = []
    seen_surfaces: set[str] = set()
    for role in sorted(frame.role_assignments.keys()):
        for ref in frame.slot_evidence_refs.get(role, []):
            surface = surfaces.get(ref)
            if surface and surface not in seen_surfaces:
                ordered_surfaces.append(surface)
                seen_surfaces.add(surface)
                break
    member_surfaces = ordered_surfaces or [
        surfaces[ref] for ref in frame.member_evidence_refs if ref in surfaces
    ]
    summary = " ".join(member_surfaces).strip()
    payload = {
        "frame_id": frame.frame_id,
        "frame_type": frame.frame_type,
        "basin_id": basin_id,
        "roles": role_rows,
        "member_evidence_refs": list(frame.member_evidence_refs),
        "member_evidence_surfaces": member_surfaces,
        "unresolved_slots": list(frame.unresolved_slot_names),
    }
    if summary:
        payload["summary"] = summary
    return RenderUnit(
        unit_id=f"frame-{index}",
        unit_type="frame_summary",
        scope_frame_id=ctx_id,
        text_intent="answer",
        payload=payload,
        confidence=frame.confidence,
        required=required,
        source_refs=refs,
    )


def _committed_render_units(
    *,
    inp: LucidityInput,
    frame_commits: list[FrameCommit],
    artifact: dict,
    primary_basin_id: str,
    assembly_ids: list[str],
) -> list[RenderUnit]:
    units: list[RenderUnit] = []
    frame_by_id = {frame.frame_id: frame for frame in inp.binding_output.candidate_frames}
    scoped = _frame_for_scope(inp.binding_output.candidate_frames, inp.context_op_output.context_frames)
    primary_state = (
        next(
            (item for item in inp.basin_output.candidate_basin_states if item.basin_id == primary_basin_id),
            None,
        )
        if primary_basin_id
        else None
    )
    primary_payload = primary_state.quantized_payload if primary_state is not None else {}
    primary_subject = ""
    if isinstance(primary_payload, dict):
        primary_subject = str(
            primary_payload.get("concept_id") or primary_payload.get("canonical_label") or ""
        )
    graph_units = _graph_relation_render_units(inp, subject_filter=primary_subject)
    basin_relation_units: list[RenderUnit] = []

    if primary_basin_id:
        state = primary_state
        basin_refs = [SourceRef(ref_type="basin", ref_id=primary_basin_id, role="supports")]
        if state is not None:
            basin_refs.extend(_source_refs_from_ids(state.source_refs, ref_type="source"))
            basin_refs.extend(_source_refs_from_ids(state.evidence_handles, ref_type="evidence"))
        units.append(
            RenderUnit(
                unit_id="basin-primary",
                unit_type="claim",
                text_intent="answer",
                payload={
                    "basin_id": primary_basin_id,
                    "energy": state.energy if state is not None else 0.0,
                    "margin_vs_next": state.margin_vs_next if state is not None else 0.0,
                    "supporting_trace_ids": list(state.supporting_trace_ids) if state is not None else [],
                    "supporting_frame_ids": list(state.supporting_frame_ids) if state is not None else [],
                    "scope_frame_ids": list(state.scope_frame_ids) if state is not None else [],
                    "evidence_handles": list(state.evidence_handles) if state is not None else [],
                    "relation_handles": list(state.relation_handles) if state is not None else [],
                    "source_refs": list(state.source_refs) if state is not None else [],
                    "trust_score": state.trust_score if state is not None else 0.0,
                    "heat_tier": state.heat_tier if state is not None else "",
                    "quantized_payload": dict(state.quantized_payload) if state is not None else {},
                },
                confidence=state.energy if state is not None else 0.0,
                required=not artifact and not frame_commits and not graph_units,
                source_refs=basin_refs,
            )
        )
        if state is not None and not graph_units:
            basin_relation_units = _basin_relation_render_units(state)
            units.extend(basin_relation_units)

    units.extend(graph_units)
    has_relation_units = bool(graph_units or basin_relation_units)

    for index, commit in enumerate(frame_commits):
        if has_relation_units:
            continue
        frame = frame_by_id.get(commit.context_frame_id) or scoped.get(commit.context_frame_id)
        if frame is not None:
            units.append(
                _frame_render_unit(
                    index=index,
                    frame=frame,
                    ctx_id=commit.context_frame_id,
                    basin_id=commit.basin_id,
                    inp=inp,
                    required=not artifact,
                )
            )

    if artifact:
        units.append(
            RenderUnit(
                unit_id="projection-artifact",
                unit_type="artifact",
                text_intent="answer",
                payload=dict(artifact),
                confidence=1.0,
                required=True,
                source_refs=[
                    SourceRef(ref_type="projection", ref_id=aid, role="supports")
                    for aid in assembly_ids
                ],
            )
        )
    return units


def build_committed_state(
    inp: LucidityInput,
    *,
    projection: ProjectorOutput | None = None,
) -> CommittedState:
    summary = inp.basin_output.competition_summary
    assembly = _best_assembly(inp.basin_output)
    shape = _pick_commit_shape(inp, assembly=assembly)
    artifact, rollout = _rollout_artifact(projection or inp.projection_output)

    frame_commits: list[FrameCommit] = []
    scoped = _frame_for_scope(inp.binding_output.candidate_frames, inp.context_op_output.context_frames)
    for ctx_id, frame in scoped.items():
        basin_id = summary.top_basin_id
        for state in inp.basin_output.candidate_basin_states:
            if frame.frame_id in state.supporting_frame_ids or ctx_id in state.scope_frame_ids:
                basin_id = state.basin_id
                break
        frame_commits.append(
            FrameCommit(
                context_frame_id=ctx_id,
                frame_type=frame.frame_type,
                basin_id=basin_id,
                role_map=dict(frame.role_assignments),
                scope_notes=next(
                    (ctx.scope_notes for ctx in inp.context_op_output.context_frames if ctx.context_frame_id == ctx_id),
                    "",
                ),
            )
        )

    member_ids: list[str] = []
    assembly_ids: list[str] = []
    if assembly is not None:
        assembly_ids = [assembly.assembly_id]
        member_ids = list(assembly.member_basin_ids)

    render_units = _committed_render_units(
        inp=inp,
        frame_commits=frame_commits,
        artifact=artifact,
        primary_basin_id=summary.top_basin_id,
        assembly_ids=assembly_ids,
    )

    return CommittedState(
        commit_id=str(uuid4()),
        commit_shape=shape,
        primary_basin_id=summary.top_basin_id,
        assembly_ids=assembly_ids,
        member_basin_ids=member_ids,
        frame_commits=frame_commits,
        render_units=render_units,
        projection_artifact=artifact,
        provenance_chain=[rollout.rollout_id] if rollout is not None else [],
    )


def projector_targets(inp: LucidityInput) -> list[str]:
    targets: list[str] = []
    assembly = _best_assembly(inp.basin_output)
    if assembly is not None and assembly.assembly_id:
        targets.append(assembly.assembly_id)
    summary = inp.basin_output.competition_summary
    if summary.top_basin_id:
        targets.append(summary.top_basin_id)
    if summary.second_basin_id:
        targets.append(summary.second_basin_id)
    if not targets:
        targets = [state.basin_id for state in _top_basin_states(inp.basin_output, limit=2) if state.basin_id]
    deduped: list[str] = []
    for item in targets:
        if item and item not in deduped:
            deduped.append(item)
    return deduped or ["asy_grid_candidate"]
