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

    if primary_basin_id:
        state = next(
            (item for item in inp.basin_output.candidate_basin_states if item.basin_id == primary_basin_id),
            None,
        )
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
                },
                confidence=state.energy if state is not None else 0.0,
                required=not artifact and not frame_commits,
                source_refs=[SourceRef(ref_type="basin", ref_id=primary_basin_id, role="supports")],
            )
        )

    for index, commit in enumerate(frame_commits):
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
