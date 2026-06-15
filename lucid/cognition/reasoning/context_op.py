"""Context operator: local scope assignment and interference gates.

This stage is deliberately deterministic. It does not classify the whole input,
choose basins, project consequences, or decode. It only turns plural binding
frames plus perceptual evidence into local scopes, trace assignments, frame
links, soft basin-family pressure, and gates for the interference stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from lucid.cognition.reasoning.cue_routes import (
    competing_units,
    evidence_cue_routes,
    has_competing_routes,
)
from lucid.cognition.memory.basin_bank import normalize_family_hint
from lucid.ir.binding import CandidateFrame
from lucid.ir.common import AmbiguityPolicy, ComputePolicy
from lucid.ir.cue import CueCloud
from lucid.ir.context_op import (
    ContextFrame,
    ContextOpInput,
    ContextOpOutput,
    FrameLink,
    InterferenceGate,
    LocalBasinPressure,
    ScopedTraceAssignment,
)
from lucid.ir.dmf import ActiveTrace
from lucid.ir.perception import CandidateRegion, PerceptualEvidenceGraph
from lucid.runtime.paths import resolve_checkpoint
from lucid.training.checkpoint.slots import resolve_checkpoint_ref
from lucid.training.checkpoint.store import STORE_FILES


@dataclass(slots=True)
class ContextOpConfig:
    checkpoint: str | Path | None = None


def run_context_op(
    inp: ContextOpInput,
    *,
    config: ContextOpConfig | None = None,
) -> ContextOpOutput:
    """Build local context scopes from binding frames and perception hints."""
    return ContextOperator(config or ContextOpConfig()).run(inp)


class ContextOperator:
    """Deterministic context-op implementation for the Lucid pipeline."""

    def __init__(self, config: ContextOpConfig) -> None:
        self._gate_patterns = _load_gate_patterns(config.checkpoint)

    def run(self, inp: ContextOpInput) -> ContextOpOutput:
        context_frames = self._build_context_frames(
            inp.binding_candidate_frames,
            inp.perceptual_evidence_graph,
            inp.prior_context_frames,
            force_widen=self._has_feedback(inp, "SEARCH_WIDER"),
        )
        assignments = self._assign_traces(context_frames, inp)
        links = self._link_frames(
            context_frames,
            assignments,
            inp.perceptual_evidence_graph,
            inp.binding_candidate_frames,
        )
        gates = self._build_gates(context_frames, assignments, links, inp)
        pressures = self._build_local_pressures(context_frames, assignments, inp)
        ambiguity_policy = self._ambiguity_policy(inp)
        compute_policy = self._compute_policy(inp, ambiguity_policy)
        audit_notes = self._audit_notes(context_frames, assignments, links, gates, pressures, inp)

        return ContextOpOutput(
            context_frames=context_frames,
            scoped_trace_assignments=assignments,
            frame_links=links,
            interference_gates=gates,
            local_basin_pressures=pressures,
            ambiguity_policy=ambiguity_policy,
            compute_policy=compute_policy,
            audit_notes=audit_notes,
        )

    @staticmethod
    def _has_feedback(inp: ContextOpInput, token: str) -> bool:
        return any(item.strip().upper() == token for item in inp.lucidity_feedback)

    def _build_context_frames(
        self,
        frames: list[CandidateFrame],
        graph: PerceptualEvidenceGraph,
        prior_context_frames: list[ContextFrame],
        *,
        force_widen: bool,
    ) -> list[ContextFrame]:
        out: list[ContextFrame] = []
        seen: set[str] = set()

        for prior in prior_context_frames:
            if prior.context_frame_id in seen:
                continue
            out.append(prior)
            seen.add(prior.context_frame_id)

        for frame in frames:
            context_id = _context_id(frame.frame_id)
            if context_id in seen:
                continue
            out.append(
                ContextFrame(
                    context_frame_id=context_id,
                    member_frame_ids=[frame.frame_id],
                    scope_notes=_frame_scope_notes(frame),
                    heat_policy="widen" if force_widen else "active",
                )
            )
            seen.add(context_id)

        if out:
            return out

        for region in graph.candidate_regions:
            context_id = _context_id(region.region_id)
            if context_id in seen:
                continue
            out.append(
                ContextFrame(
                    context_frame_id=context_id,
                    scope_notes=_region_scope_notes(region),
                    heat_policy="widen" if force_widen else "active",
                )
            )
            seen.add(context_id)

        if not out and graph.candidate_units:
            out.append(
                ContextFrame(
                    context_frame_id="cf_evidence",
                    scope_notes=f"evidence fallback; units:{len(graph.candidate_units)}",
                    heat_policy="widen" if force_widen else "active",
                )
            )

        return out

    def _assign_traces(
        self,
        context_frames: list[ContextFrame],
        inp: ContextOpInput,
    ) -> list[ScopedTraceAssignment]:
        if not context_frames:
            return []

        context_by_member = {
            member_id: context.context_frame_id
            for context in context_frames
            for member_id in context.member_frame_ids
        }
        trace_context_weights: dict[str, dict[str, float]] = {}

        for frame in inp.binding_candidate_frames:
            context_id = context_by_member.get(frame.frame_id, context_frames[0].context_frame_id)
            for trace_id in _frame_trace_ids(frame):
                weight = _trace_weight(trace_id, inp.dmf_output.active_traces)
                _add_weight(trace_context_weights, trace_id, context_id, weight)

        region_contexts = {
            context.context_frame_id: context
            for context in context_frames
            if not context.member_frame_ids
        }
        if region_contexts:
            for trace in inp.dmf_output.active_traces:
                if trace.trace_id in trace_context_weights:
                    continue
                context_id = self._region_context_for_trace(trace.trace_id, region_contexts)
                _add_weight(trace_context_weights, trace.trace_id, context_id, trace.activation)

        if not trace_context_weights:
            primary = context_frames[0].context_frame_id
            for trace in inp.dmf_output.active_traces:
                _add_weight(trace_context_weights, trace.trace_id, primary, trace.activation)

        for conflict in inp.dmf_output.conflict_signals:
            if conflict.scope_frame_id:
                scope_id = _context_id(conflict.scope_frame_id)
                if scope_id in {frame.context_frame_id for frame in context_frames}:
                    for trace_id in (conflict.trace_id_a, conflict.trace_id_b):
                        _add_weight(trace_context_weights, trace_id, scope_id, 0.25)

        assignments: list[ScopedTraceAssignment] = []
        for trace_id, context_weights in sorted(trace_context_weights.items()):
            ranked = sorted(context_weights.items(), key=lambda item: (-item[1], item[0]))
            primary_id, primary_weight = ranked[0]
            secondary_ids = [context_id for context_id, weight in ranked[1:] if weight >= 0.2]
            assignments.append(
                ScopedTraceAssignment(
                    trace_id=trace_id,
                    primary_context_frame_id=primary_id,
                    secondary_context_frame_ids=secondary_ids,
                    weight=round(primary_weight, 3),
                )
            )

        return assignments

    def _region_context_for_trace(
        self,
        trace_id: str,
        region_contexts: dict[str, ContextFrame],
    ) -> str:
        token = _clean_token(trace_id)
        for context_id, context in region_contexts.items():
            notes = context.scope_notes.lower()
            if token and token in notes:
                return context_id
        return sorted(region_contexts)[0]

    def _link_frames(
        self,
        context_frames: list[ContextFrame],
        assignments: list[ScopedTraceAssignment],
        graph: PerceptualEvidenceGraph,
        candidate_frames: list[CandidateFrame],
    ) -> list[FrameLink]:
        links: dict[tuple[str, str, str], float] = {}

        for assignment in assignments:
            for secondary_id in assignment.secondary_context_frame_ids:
                key = _ordered_link(
                    assignment.primary_context_frame_id,
                    secondary_id,
                    "shared_trace",
                )
                links[key] = max(links.get(key, 0.0), min(1.0, assignment.weight))

        unit_to_context = self._unit_context_map(context_frames, graph, candidate_frames)
        for hint in graph.reference_hints:
            sources = unit_to_context.get(hint.source_unit_id, set())
            targets = unit_to_context.get(hint.target_unit_id, set())
            for source in sources:
                for target in targets:
                    if source and target and source != target:
                        key = _ordered_link(source, target, hint.reference_type or "reference")
                        links[key] = max(links.get(key, 0.0), hint.confidence or 0.5)

        for hint in graph.arrangement_hints:
            sources = unit_to_context.get(hint.source_unit_id, set())
            targets = unit_to_context.get(hint.target_unit_id, set())
            for source in sources:
                for target in targets:
                    if source and target and source != target:
                        key = _ordered_link(source, target, hint.hint_type or "arrangement")
                        links[key] = max(links.get(key, 0.0), hint.weight or 0.5)

        return [
            FrameLink(source_frame_id=a, target_frame_id=b, link_type=kind, weight=round(weight, 3))
            for (a, b, kind), weight in sorted(links.items())
        ]

    def _unit_context_map(
        self,
        context_frames: list[ContextFrame],
        graph: PerceptualEvidenceGraph,
        candidate_frames: list[CandidateFrame],
    ) -> dict[str, set[str]]:
        by_region = {
            region.region_id: set(region.member_unit_ids)
            for region in graph.candidate_regions
        }
        by_frame = {
            frame.frame_id: set(frame.member_evidence_refs)
            for frame in candidate_frames
        }
        unit_to_context: dict[str, set[str]] = {}
        for context in context_frames:
            for member_id in context.member_frame_ids:
                for unit_id in by_frame.get(member_id, set()):
                    unit_to_context.setdefault(unit_id, set()).add(context.context_frame_id)
                for unit_id in by_region.get(member_id, set()):
                    unit_to_context.setdefault(unit_id, set()).add(context.context_frame_id)
            for region_id, unit_ids in by_region.items():
                if _context_id(region_id) == context.context_frame_id:
                    for unit_id in unit_ids:
                        unit_to_context.setdefault(unit_id, set()).add(context.context_frame_id)
        return unit_to_context

    def _build_gates(
        self,
        context_frames: list[ContextFrame],
        assignments: list[ScopedTraceAssignment],
        links: list[FrameLink],
        inp: ContextOpInput,
    ) -> list[InterferenceGate]:
        assigned_by_context: dict[str, set[str]] = {
            frame.context_frame_id: set() for frame in context_frames
        }
        secondary_by_context: dict[str, set[str]] = {
            frame.context_frame_id: set() for frame in context_frames
        }
        all_traces = {assignment.trace_id for assignment in assignments}

        for assignment in assignments:
            assigned_by_context.setdefault(assignment.primary_context_frame_id, set()).add(
                assignment.trace_id
            )
            for secondary_id in assignment.secondary_context_frame_ids:
                secondary_by_context.setdefault(secondary_id, set()).add(assignment.trace_id)

        linked_contexts: dict[str, set[str]] = {
            frame.context_frame_id: set() for frame in context_frames
        }
        for link in links:
            linked_contexts.setdefault(link.source_frame_id, set()).add(link.target_frame_id)
            linked_contexts.setdefault(link.target_frame_id, set()).add(link.source_frame_id)

        conflict_pairs = {
            tuple(sorted((conflict.trace_id_a, conflict.trace_id_b)))
            for conflict in inp.dmf_output.conflict_signals
            if conflict.severity >= 0.5
        }

        gates: list[InterferenceGate] = []
        for context in context_frames:
            allowed = set(assigned_by_context.get(context.context_frame_id, set()))
            allowed |= secondary_by_context.get(context.context_frame_id, set())
            for linked_id in linked_contexts.get(context.context_frame_id, set()):
                allowed |= secondary_by_context.get(linked_id, set())

            blocked = set()
            for trace_id in all_traces - allowed:
                if self._should_block_trace(trace_id, allowed, conflict_pairs, context, inp):
                    blocked.add(trace_id)

            allowed, blocked = self._apply_learned_gate_patterns(
                context.context_frame_id,
                allowed,
                blocked,
            )
            gates.append(
                InterferenceGate(
                    gate_id=f"gate_{context.context_frame_id}",
                    scope_frame_id=context.context_frame_id,
                    allowed_trace_ids=sorted(allowed),
                    blocked_trace_ids=sorted(blocked),
                    reason=_gate_reason(context, allowed, blocked),
                )
            )
        return gates

    def _apply_learned_gate_patterns(
        self,
        scope_frame_id: str,
        allowed: set[str],
        blocked: set[str],
    ) -> tuple[set[str], set[str]]:
        for pattern in self._gate_patterns:
            pattern_scope = _context_id(str(pattern.get("scope_frame_id") or ""))
            if pattern_scope != scope_frame_id:
                continue
            for trace_id in pattern.get("allowed_trace_ids") or []:
                token = str(trace_id).strip()
                if token:
                    allowed.add(token)
            for trace_id in pattern.get("blocked_trace_ids") or []:
                token = str(trace_id).strip()
                if token:
                    blocked.add(token)
        return allowed, blocked

    def _should_block_trace(
        self,
        trace_id: str,
        allowed: set[str],
        conflict_pairs: set[tuple[str, str]],
        context: ContextFrame,
        inp: ContextOpInput,
    ) -> bool:
        if any(tuple(sorted((trace_id, allowed_id))) in conflict_pairs for allowed_id in allowed):
            return True
        _ = context
        return False

    def _build_local_pressures(
        self,
        context_frames: list[ContextFrame],
        assignments: list[ScopedTraceAssignment],
        inp: ContextOpInput,
    ) -> list[LocalBasinPressure]:
        assignment_map = {
            assignment.trace_id: assignment for assignment in assignments
        }
        active_by_id = {trace.trace_id: trace for trace in inp.dmf_output.active_traces}
        pressures: list[LocalBasinPressure] = []

        for context in context_frames:
            hints: dict[str, float] = {}
            traces = [
                trace_id
                for trace_id, assignment in assignment_map.items()
                if assignment.primary_context_frame_id == context.context_frame_id
                or context.context_frame_id in assignment.secondary_context_frame_ids
            ]
            for trace_id in traces:
                trace = active_by_id.get(trace_id)
                labels: list[str] = []
                if trace and trace.cluster_id:
                    labels.append(trace.cluster_id)
                labels.append(_clean_token(trace_id))
                for label in labels:
                    family = normalize_family_hint(label)
                    if not family:
                        continue
                    weight = round(
                        min(1.0, (trace.activation if trace else 0.5) * 0.8),
                        3,
                    )
                    hints[family] = max(hints.get(family, 0.0), weight)

            if inp.cue_cloud is not None:
                for family, factor in _hints_from_competing_routes(
                    inp.cue_cloud,
                    context,
                    inp.binding_candidate_frames,
                ).items():
                    hints[family] = max(hints.get(family, 0.0), round(factor, 3))

            if context.member_frame_ids:
                for frame in inp.binding_candidate_frames:
                    if frame.frame_id in context.member_frame_ids:
                        family = f"{_slug(frame.frame_type)}_frame"
                        hints[family] = max(hints.get(family, 0.0), round(frame.confidence, 3))

            pressures.append(
                LocalBasinPressure(
                    context_frame_id=context.context_frame_id,
                    basin_family_hints=dict(sorted(hints.items())),
                )
            )
        return pressures

    def _ambiguity_policy(self, inp: ContextOpInput) -> AmbiguityPolicy:
        if self._has_feedback(inp, "SEARCH_WIDER"):
            return AmbiguityPolicy.FORCE_WIDEN
        if self._has_feedback(inp, "RECHECK_BINDING"):
            return AmbiguityPolicy.PRESERVE_PLURAL
        has_unresolved = any(frame.unresolved_slot_names for frame in inp.binding_candidate_frames)
        evidence_refs = {
            ref
            for frame in inp.binding_candidate_frames
            for ref in frame.member_evidence_refs
        }
        has_route_competition = has_competing_routes(inp.cue_cloud, evidence_refs)
        high_margin = inp.dmf_output.top_margin >= 0.6
        stable_frames = all(frame.confidence >= 0.65 for frame in inp.binding_candidate_frames)
        low_dmf_uncertainty = inp.dmf_output.uncertainty_summary != "high"
        if (
            inp.binding_candidate_frames
            and high_margin
            and stable_frames
            and not has_unresolved
            and not has_route_competition
            and low_dmf_uncertainty
        ):
            return AmbiguityPolicy.ALLOW_NARROW
        return AmbiguityPolicy.PRESERVE_PLURAL

    def _compute_policy(
        self,
        inp: ContextOpInput,
        ambiguity_policy: AmbiguityPolicy,
    ) -> ComputePolicy:
        policy = inp.compute_policy
        if ambiguity_policy == AmbiguityPolicy.FORCE_WIDEN:
            widened_frames = len(inp.binding_candidate_frames) * 2
            return replace(
                policy,
                max_candidate_frames=max(policy.max_candidate_frames, widened_frames),
                retrieval_budget_multiplier=max(policy.retrieval_budget_multiplier, 1.5),
                mode="deep_scope",
            )
        return policy

    def _audit_notes(
        self,
        context_frames: list[ContextFrame],
        assignments: list[ScopedTraceAssignment],
        links: list[FrameLink],
        gates: list[InterferenceGate],
        pressures: list[LocalBasinPressure],
        inp: ContextOpInput,
    ) -> list[str]:
        blocked_count = sum(len(gate.blocked_trace_ids) for gate in gates)
        pressure_count = sum(len(pressure.basin_family_hints) for pressure in pressures)
        notes = [
            (
                f"context_frames={len(context_frames)} "
                f"from binding_frames={len(inp.binding_candidate_frames)}"
            ),
            f"scoped_trace_assignments={len(assignments)}",
            (
                f"frame_links={len(links)} interference_gates={len(gates)} "
                f"blocked_traces={blocked_count}"
            ),
            f"local_basin_pressure_hints={pressure_count}",
        ]
        if inp.lucidity_feedback:
            notes.append("lucidity_feedback=" + ",".join(inp.lucidity_feedback))
        return notes


def _context_id(value: str) -> str:
    clean = _slug(value)
    return clean if clean.startswith("cf_") else f"cf_{clean}"


def _slug(value: str) -> str:
    clean = "".join(char.lower() if char.isalnum() else "_" for char in str(value))
    clean = "_".join(part for part in clean.split("_") if part)
    return clean or "scope"


def _clean_token(value: str) -> str:
    token = _slug(value)
    for prefix in ("t_", "trace_", "u_", "cf_"):
        if token.startswith(prefix):
            return token[len(prefix) :]
    return token


def _frame_scope_notes(frame: CandidateFrame) -> str:
    bits = [f"frame_type:{frame.frame_type}"]
    if frame.member_evidence_refs:
        bits.append("evidence:" + ",".join(frame.member_evidence_refs))
    if frame.unresolved_slot_names:
        bits.append("unresolved:" + ",".join(frame.unresolved_slot_names))
    return "; ".join(bits)


def _region_scope_notes(region: CandidateRegion) -> str:
    bits = [f"region:{region.role_hint or region.region_id}"]
    if region.member_unit_ids:
        bits.append("members:" + ",".join(region.member_unit_ids))
    if region.uncertainty:
        bits.append(f"uncertainty:{region.uncertainty}")
    return "; ".join(bits)


def _frame_trace_ids(frame: CandidateFrame) -> list[str]:
    trace_ids: list[str] = []
    trace_ids.extend(_string_values(frame.role_assignments.values()))
    trace_ids.extend(_string_values(frame.relation_assignments.values()))
    trace_ids.extend(frame.supporting_trace_ids)
    return _dedupe(trace_ids)


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


def _trace_weight(trace_id: str, active_traces: list[ActiveTrace]) -> float:
    for trace in active_traces:
        if trace.trace_id == trace_id:
            return max(0.05, min(1.0, trace.activation))
    return 0.5


def _add_weight(
    weights: dict[str, dict[str, float]],
    trace_id: str,
    context_id: str,
    weight: float,
) -> None:
    if not trace_id or not context_id:
        return
    by_context = weights.setdefault(trace_id, {})
    by_context[context_id] = max(by_context.get(context_id, 0.0), max(0.05, min(1.0, weight)))


def _ordered_link(source: str, target: str, kind: str) -> tuple[str, str, str]:
    a, b = sorted((source, target))
    return a, b, kind


def _gate_reason(context: ContextFrame, allowed: set[str], blocked: set[str]) -> str:
    reason = f"scope {context.context_frame_id}: allow local/linked traces"
    if blocked:
        reason += "; block cross-scope conflicts/leaks"
    if "unresolved:" in context.scope_notes:
        reason += "; preserve unresolved slots"
    if not allowed:
        reason += "; no active traces assigned"
    return reason


def _load_gate_patterns(checkpoint: str | Path | None) -> list[dict[str, Any]]:
    if not checkpoint:
        return []
    root = resolve_checkpoint(resolve_checkpoint_ref(checkpoint))
    path = root / STORE_FILES["context_policy"]
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    patterns = payload.get("gate_patterns") or []
    return [pattern for pattern in patterns if isinstance(pattern, dict)]


def _scope_evidence_refs(
    context: ContextFrame,
    frames: list[CandidateFrame],
) -> set[str]:
    evidence_refs: set[str] = set()
    for frame in frames:
        if context.member_frame_ids and frame.frame_id not in context.member_frame_ids:
            continue
        evidence_refs.update(frame.member_evidence_refs)
        for refs in frame.slot_evidence_refs.values():
            evidence_refs.update(refs)
    return evidence_refs


def _hints_from_competing_routes(
    cue_cloud: CueCloud,
    context: ContextFrame,
    frames: list[CandidateFrame],
) -> dict[str, float]:
    """Local basin pressure from plural cue routes on in-scope evidence."""

    hints: dict[str, float] = {}
    evidence_refs = _scope_evidence_refs(context, frames)
    routes = evidence_cue_routes(cue_cloud)
    competing = competing_units(cue_cloud)
    for unit_id in evidence_refs:
        if unit_id not in competing:
            continue
        for cue_key, weight in routes.get(unit_id, []):
            family = normalize_family_hint(cue_key)
            if not family:
                continue
            hints[family] = max(hints.get(family, 0.0), round(min(1.0, weight), 3))
    return hints
