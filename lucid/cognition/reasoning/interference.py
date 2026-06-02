"""Interference: scoped support and conflict pressure.

This stage is deterministic and local-only. It reads context-op scopes/gates,
binding frames, and DMF traces, then emits soft support/conflict edges and
scoped basin pressure for the basin stage. It does not choose a basin, delete a
hypothesis, project, or decode.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

from lucid.ir.binding import CandidateFrame
from lucid.ir.context_op import ContextFrame, InterferenceGate, LocalBasinPressure
from lucid.ir.dmf import ActiveTrace, ConflictSignal, TraceCluster
from lucid.ir.interference import (
    BasinBasinEdge,
    BasinEnergyDelta,
    ConflictReport,
    FrameBasinEdge,
    InterferenceInput,
    InterferenceOutput,
    LearnedInterferenceLink,
    TraceFrameEdge,
    TraceTraceEdge,
)


def run_interference(inp: InterferenceInput) -> InterferenceOutput:
    """Compute local support/conflict pressure for every context frame."""
    return InterferenceOperator().run(inp)


class InterferenceOperator:
    """Deterministic local interference implementation."""

    def run(self, inp: InterferenceInput) -> InterferenceOutput:
        active = {trace.trace_id: trace for trace in inp.dmf_output.active_traces}
        frames_by_context = _frames_by_context(inp.context_frames, inp.candidate_frames)
        gates = {gate.scope_frame_id: gate for gate in inp.interference_gates}
        scoped_assignments = _assignments_by_context(inp)
        clusters = _clusters_by_trace(inp.dmf_output.trace_clusters)
        pressures = {p.context_frame_id: p for p in inp.local_basin_pressures}

        out = InterferenceOutput()
        for context in inp.context_frames:
            scope_id = context.context_frame_id
            allowed, blocked = self._scope_trace_sets(
                context,
                frames_by_context.get(scope_id, []),
                gates.get(scope_id),
                scoped_assignments.get(scope_id, set()),
                active,
            )
            scoped_frames = frames_by_context.get(scope_id, [])

            self._apply_learned_links(out, scope_id, allowed, blocked, inp.learned_interference_links)
            self._apply_cluster_support(out, scope_id, allowed, clusters, active)
            self._apply_conflicts(out, scope_id, allowed, blocked, inp.dmf_output.conflict_signals)
            frame_scores = self._apply_frame_support(out, scope_id, scoped_frames, allowed, blocked, active)
            self._apply_frame_basin_pressure(out, scope_id, scoped_frames, frame_scores)
            self._apply_local_basin_pressure(out, scope_id, pressures.get(scope_id))
            self._apply_basin_relations(out, scope_id)

        self._finalize(out, inp)
        return out

    def _scope_trace_sets(
        self,
        context: ContextFrame,
        frames: list[CandidateFrame],
        gate: InterferenceGate | None,
        assigned_trace_ids: set[str],
        active: dict[str, ActiveTrace],
    ) -> tuple[set[str], set[str]]:
        if gate is not None:
            blocked = set(gate.blocked_trace_ids)
            allowed = set(gate.allowed_trace_ids) - blocked
            return allowed, blocked

        allowed = set(assigned_trace_ids)
        for frame in frames:
            allowed.update(_frame_trace_ids(frame))
        if not allowed and not context.member_frame_ids:
            allowed.update(active)
        return allowed, set()

    def _apply_learned_links(
        self,
        out: InterferenceOutput,
        scope_id: str,
        allowed: set[str],
        blocked: set[str],
        links: list[LearnedInterferenceLink],
    ) -> None:
        for link in links:
            if link.scope_hint and link.scope_hint != scope_id:
                continue
            if link.source_id in blocked or link.target_id in blocked:
                continue
            if link.source_id not in allowed or link.target_id not in allowed:
                continue
            delta = _round(link.weight)
            edge = TraceTraceEdge(link.source_id, link.target_id, delta, scope_frame_id=scope_id)
            if delta >= 0:
                out.trace_trace_edges.append(edge)
            else:
                out.trace_trace_edges.append(edge)
                out.conflict_reports.append(
                    ConflictReport(
                        scope_frame_id=scope_id,
                        conflict_type="learned_trace_conflict",
                        members=[link.source_id, link.target_id],
                        severity=min(1.0, abs(delta)),
                    )
                )

    def _apply_cluster_support(
        self,
        out: InterferenceOutput,
        scope_id: str,
        allowed: set[str],
        clusters: dict[str, TraceCluster],
        active: dict[str, ActiveTrace],
    ) -> None:
        for trace_a, trace_b in combinations(sorted(allowed), 2):
            cluster_a = clusters.get(trace_a)
            cluster_b = clusters.get(trace_b)
            if cluster_a is None or cluster_b is None:
                continue
            if cluster_a.cluster_id != cluster_b.cluster_id:
                continue
            activation = (_activation(trace_a, active) + _activation(trace_b, active)) / 2.0
            delta = _round(min(cluster_a.cluster_strength, cluster_b.cluster_strength, activation) * 0.35)
            if delta > 0:
                out.trace_trace_edges.append(
                    TraceTraceEdge(trace_a, trace_b, delta, scope_frame_id=scope_id)
                )

    def _apply_conflicts(
        self,
        out: InterferenceOutput,
        scope_id: str,
        allowed: set[str],
        blocked: set[str],
        conflicts: list[ConflictSignal],
    ) -> None:
        for conflict in conflicts:
            signal_scope = _context_id(conflict.scope_frame_id) if conflict.scope_frame_id else ""
            if signal_scope and signal_scope != scope_id:
                continue
            members = {conflict.trace_id_a, conflict.trace_id_b}
            if members & blocked:
                continue
            if not members <= allowed:
                continue
            delta = -_round(min(1.0, max(0.0, conflict.severity)) * 0.7)
            out.trace_trace_edges.append(
                TraceTraceEdge(conflict.trace_id_a, conflict.trace_id_b, delta, scope_frame_id=scope_id)
            )
            out.conflict_reports.append(
                ConflictReport(
                    scope_frame_id=scope_id,
                    conflict_type="dmf_trace_conflict",
                    members=sorted(members),
                    severity=_round(conflict.severity),
                )
            )

    def _apply_frame_support(
        self,
        out: InterferenceOutput,
        scope_id: str,
        frames: list[CandidateFrame],
        allowed: set[str],
        blocked: set[str],
        active: dict[str, ActiveTrace],
    ) -> dict[str, float]:
        frame_scores: dict[str, float] = {}
        for frame in frames:
            support_values: list[float] = []
            for trace_id in _frame_trace_ids(frame):
                if trace_id in blocked or trace_id not in allowed:
                    continue
                value = _round(_activation(trace_id, active) * max(0.05, frame.confidence))
                if trace_id in frame.conflicting_trace_ids:
                    out.trace_frame_edges.append(TraceFrameEdge(trace_id, frame.frame_id, -value))
                    out.conflict_reports.append(
                        ConflictReport(
                            scope_frame_id=scope_id,
                            conflict_type="trace_frame_conflict",
                            members=[trace_id, frame.frame_id],
                            severity=value,
                        )
                    )
                else:
                    out.trace_frame_edges.append(TraceFrameEdge(trace_id, frame.frame_id, value))
                    support_values.append(value)

            if support_values:
                frame_scores[frame.frame_id] = _round(sum(support_values) / len(support_values))
        return frame_scores

    def _apply_frame_basin_pressure(
        self,
        out: InterferenceOutput,
        scope_id: str,
        frames: list[CandidateFrame],
        frame_scores: dict[str, float],
    ) -> None:
        for frame in frames:
            score = frame_scores.get(frame.frame_id, _round(frame.confidence * 0.5))
            if score <= 0:
                continue
            basin_id = _basin_id(f"{frame.frame_type}_frame")
            delta = _round(score * 0.18)
            out.frame_basin_edges.append(FrameBasinEdge(frame.frame_id, basin_id, delta))
            _add_basin_delta(
                out,
                scope_id,
                basin_id,
                delta,
                [f"frame:{frame.frame_id}", f"frame_type:{frame.frame_type}"],
            )

            for unresolved in frame.unresolved_slot_names:
                conflict_basin = _basin_id(f"unresolved_{unresolved}")
                conflict_delta = _round(score * 0.06)
                out.frame_basin_edges.append(
                    FrameBasinEdge(frame.frame_id, conflict_basin, conflict_delta)
                )
                _add_basin_delta(
                    out,
                    scope_id,
                    conflict_basin,
                    conflict_delta,
                    [f"unresolved_slot:{unresolved}", f"frame:{frame.frame_id}"],
                )

    def _apply_local_basin_pressure(
        self,
        out: InterferenceOutput,
        scope_id: str,
        pressure: LocalBasinPressure | None,
    ) -> None:
        if pressure is None:
            return
        for family, weight in sorted(pressure.basin_family_hints.items()):
            delta = _round(max(0.0, min(1.0, weight)) * 0.16)
            if delta <= 0:
                continue
            basin_id = _basin_id(family)
            _add_basin_delta(out, scope_id, basin_id, delta, [f"local_prior:{family}"])

    def _apply_basin_relations(self, out: InterferenceOutput, scope_id: str) -> None:
        basin_ids = sorted(
            {
                delta.basin_id
                for delta in out.scoped_basin_energy_deltas
                if delta.scope_frame_id == scope_id
            }
        )
        if len(basin_ids) < 2:
            return

        cooperating = [
            basin_id
            for basin_id in basin_ids
            if _is_cooperation_family(basin_id)
        ]
        if len(cooperating) >= 2:
            out.cooperation_maps[scope_id] = cooperating
            for basin_a, basin_b in combinations(cooperating, 2):
                out.basin_basin_edges.append(
                    BasinBasinEdge(basin_a, basin_b, 0.24, relation="cooperate")
                )

        competing = [
            basin_id
            for basin_id in basin_ids
            if _is_competition_family(basin_id)
        ]
        if len(competing) >= 2:
            out.competition_maps[scope_id] = competing
            for basin_a, basin_b in combinations(competing, 2):
                out.basin_basin_edges.append(
                    BasinBasinEdge(basin_a, basin_b, -0.22, relation="compete")
                )
            out.conflict_reports.append(
                ConflictReport(
                    scope_frame_id=scope_id,
                    conflict_type="basin_family_competition",
                    members=competing,
                    severity=0.44,
                )
            )

    def _finalize(self, out: InterferenceOutput, inp: InterferenceInput) -> None:
        out.trace_trace_edges.sort(key=lambda e: (e.scope_frame_id, e.trace_id_a, e.trace_id_b, e.delta))
        out.trace_frame_edges.sort(key=lambda e: (e.frame_id, e.trace_id, e.delta))
        out.frame_basin_edges.sort(key=lambda e: (e.frame_id, e.basin_id, e.delta))
        out.basin_basin_edges.sort(key=lambda e: (e.relation, e.basin_id_a, e.basin_id_b))
        out.scoped_basin_energy_deltas.sort(key=lambda e: (e.scope_frame_id, e.basin_id, e.delta))
        out.conflict_reports.sort(key=lambda e: (e.scope_frame_id, e.conflict_type, e.members))
        out.basin_energy_deltas = dict(sorted(out.basin_energy_deltas.items()))
        out.cooperation_maps = {k: sorted(v) for k, v in sorted(out.cooperation_maps.items())}
        out.competition_maps = {k: sorted(v) for k, v in sorted(out.competition_maps.items())}

        blocked_count = sum(len(gate.blocked_trace_ids) for gate in inp.interference_gates)
        out.audit_notes = [
            (
                f"frames_processed={len(inp.context_frames)} "
                f"gates_honored={len(inp.interference_gates)} blocked_traces={blocked_count}"
            ),
            (
                f"trace_edges={len(out.trace_trace_edges)} "
                f"trace_frame_edges={len(out.trace_frame_edges)} "
                f"frame_basin_edges={len(out.frame_basin_edges)}"
            ),
            (
                f"scoped_basin_deltas={len(out.scoped_basin_energy_deltas)} "
                f"conflict_reports={len(out.conflict_reports)}"
            ),
        ]


def _frames_by_context(
    context_frames: list[ContextFrame],
    candidate_frames: list[CandidateFrame],
) -> dict[str, list[CandidateFrame]]:
    by_id = {frame.frame_id: frame for frame in candidate_frames}
    first_context = context_frames[0].context_frame_id if context_frames else ""
    out: dict[str, list[CandidateFrame]] = {context.context_frame_id: [] for context in context_frames}
    assigned: set[str] = set()

    for context in context_frames:
        for member_id in context.member_frame_ids:
            frame = by_id.get(member_id)
            if frame is not None:
                out.setdefault(context.context_frame_id, []).append(frame)
                assigned.add(frame.frame_id)

    if first_context:
        for frame in candidate_frames:
            if frame.frame_id not in assigned:
                out.setdefault(first_context, []).append(frame)
    return out


def _assignments_by_context(inp: InterferenceInput) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for assignment in inp.scoped_trace_assignments:
        out.setdefault(assignment.primary_context_frame_id, set()).add(assignment.trace_id)
        for context_id in assignment.secondary_context_frame_ids:
            out.setdefault(context_id, set()).add(assignment.trace_id)
    return out


def _clusters_by_trace(clusters: list[TraceCluster]) -> dict[str, TraceCluster]:
    out: dict[str, TraceCluster] = {}
    for cluster in clusters:
        for trace_id in cluster.member_trace_ids:
            out[trace_id] = cluster
    return out


def _frame_trace_ids(frame: CandidateFrame) -> list[str]:
    values: list[str] = []
    values.extend(_string_values(frame.role_assignments.values()))
    values.extend(_string_values(frame.relation_assignments.values()))
    values.extend(frame.supporting_trace_ids)
    values.extend(frame.conflicting_trace_ids)
    return _dedupe(values)


def _string_values(values: Iterable[object]) -> list[str]:
    return [str(value) for value in values if isinstance(value, str) and value.strip()]


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _activation(trace_id: str, active: dict[str, ActiveTrace]) -> float:
    trace = active.get(trace_id)
    if trace is None:
        return 0.5
    return max(0.0, min(1.0, trace.activation))


def _add_basin_delta(
    out: InterferenceOutput,
    scope_id: str,
    basin_id: str,
    delta: float,
    reason_refs: list[str],
) -> None:
    scoped_key = f"{scope_id}::{basin_id}"
    out.basin_energy_deltas[scoped_key] = _round(out.basin_energy_deltas.get(scoped_key, 0.0) + delta)
    out.scoped_basin_energy_deltas.append(
        BasinEnergyDelta(
            scope_frame_id=scope_id,
            basin_id=basin_id,
            delta=delta,
            reason_refs=reason_refs,
        )
    )


def _basin_id(value: str) -> str:
    clean = _slug(value)
    return clean if clean.startswith("b_") else f"b_{clean}"


def _context_id(value: str) -> str:
    if not value:
        return ""
    clean = _slug(value)
    return clean if clean.startswith("cf_") else f"cf_{clean}"


def _slug(value: str) -> str:
    clean = "".join(char.lower() if char.isalnum() else "_" for char in str(value))
    clean = "_".join(part for part in clean.split("_") if part)
    return clean or "scope"


def _round(value: float) -> float:
    return round(float(value), 3)


def _is_cooperation_family(basin_id: str) -> bool:
    return any(
        token in basin_id
        for token in (
            "event_frame",
            "transform_frame",
            "position_shift",
            "attribute_change",
            "shared_referent",
        )
    )


def _is_competition_family(basin_id: str) -> bool:
    return any(
        token in basin_id
        for token in (
            "financial_destination",
            "outdoor_context",
            "water_activity",
            "unresolved",
            "word_sense",
        )
    )
