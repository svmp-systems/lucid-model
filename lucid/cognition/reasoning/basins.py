"""Basins: plural hypothesis attractors scoped to context frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from lucid.ir.basins import (
    BasinAssembly,
    BasinConflict,
    BasinInput,
    BasinOutput,
    CandidateBasinState,
    CompetitionSummary,
)
from lucid.ir.binding import CandidateFrame
from lucid.ir.context_op import ContextFrame, LocalBasinPressure
from lucid.ir.interference import InterferenceOutput
from lucid.cognition.memory.basin_bank import BasinBank, BasinBankRecord, load_basin_bank, normalize_family_hint

_MIN_AFFINITY = 0.08
_MIN_PRESSURE = 0.12
_MIN_ACTIVATION = 0.12
_CONFLICT_MARGIN = 0.06
_PRIOR_STATE_SCALE = 0.12
_SUPPRESSION_SCALE = 0.25


@dataclass(slots=True)
class BasinsConfig:
    checkpoint: str | Path | None = None
    min_energy: float = 0.15


@dataclass(slots=True)
class _ScoredBasin:
    record: BasinBankRecord
    scope_id: str
    member_frames: list[str] = field(default_factory=list)
    traces: list[str] = field(default_factory=list)
    activation_score: float = 0.0
    energy: float = 0.0


def run_basins(inp: BasinInput, *, config: BasinsConfig | None = None) -> BasinOutput:
    return BasinsOperator(config or BasinsConfig()).run(inp)


class BasinsOperator:
    def __init__(self, config: BasinsConfig) -> None:
        self.config = config
        self._bank = load_basin_bank(config.checkpoint)

    def run(self, inp: BasinInput) -> BasinOutput:
        frame_by_id = {frame.frame_id: frame for frame in inp.candidate_frames}
        pressure_by_scope = {
            pressure.context_frame_id: pressure for pressure in inp.local_basin_pressures
        }
        scoped_candidates = self._shortlist(inp.context_frames, frame_by_id, pressure_by_scope)
        scored = self._score_candidates(
            scoped_candidates,
            frame_by_id,
            pressure_by_scope,
            inp.interference_output,
            inp.prior_basin_state,
        )
        states = self._build_states(scored, frame_by_id)
        states = self._filter_and_rank(states, inp.compute_policy.max_active_basins)
        states = self._assign_margins(states)
        assemblies = self._build_assemblies(states, inp.interference_output)
        conflicts = self._detect_conflicts(states)
        summary = _competition_summary(states)
        stability = _binding_stability_hint(inp.candidate_frames)
        notes = _audit_notes(
            bank=self._bank,
            states=states,
            assemblies=assemblies,
            conflicts=conflicts,
            inp=inp,
        )
        snapshot_id = inp.basin_field_snapshot_id or self._bank.snapshot_id()
        return BasinOutput(
            candidate_basin_states=states,
            basin_assemblies=assemblies,
            competition_summary=summary,
            unresolved_conflicts=conflicts,
            binding_stability_hint=stability,
            audit_notes=notes + [f"basin_field_snapshot_id={snapshot_id}"],
        )

    def _shortlist(
        self,
        context_frames: list[ContextFrame],
        frame_by_id: dict[str, CandidateFrame],
        pressure_by_scope: dict[str, LocalBasinPressure],
    ) -> dict[str, list[tuple[BasinBankRecord, list[str], list[str], float]]]:
        """scope_id -> [(record, member_frame_ids, supporting_trace_ids, activation), ...]"""

        scoped: dict[str, list[tuple[BasinBankRecord, list[str], list[str], float]]] = {}
        seen_per_scope: dict[str, set[str]] = {}

        for context in context_frames:
            scope_id = context.context_frame_id
            member_frames = [
                frame_id for frame_id in context.member_frame_ids if frame_id in frame_by_id
            ]
            scope_traces = _supporting_traces(member_frames, frame_by_id)
            scope_tokens = _activation_tokens(member_frames, frame_by_id, scope_traces)
            pressure = pressure_by_scope.get(scope_id)
            hints = dict(pressure.basin_family_hints) if pressure else {}
            seen = seen_per_scope.setdefault(scope_id, set())
            entries: list[tuple[BasinBankRecord, list[str], list[str], float]] = []

            for record in self._bank.records:
                if record.basin_id in seen:
                    continue
                matched_frames = [
                    frame_id
                    for frame_id in member_frames
                    if float(record.frame_affinities.get(frame_id, 0.0)) >= _MIN_AFFINITY
                ]
                family_key = normalize_family_hint(record.family_hint)
                pressure_weight = max(
                    (hints.get(key, 0.0) for key in hints if normalize_family_hint(key) == family_key),
                    default=0.0,
                )
                activation_score = _activation_signature_score(record, scope_tokens)
                if (
                    not matched_frames
                    and pressure_weight < _MIN_PRESSURE
                    and activation_score < _MIN_ACTIVATION
                ):
                    continue
                if not matched_frames and (
                    pressure_weight >= _MIN_PRESSURE or activation_score >= _MIN_ACTIVATION
                ):
                    matched_frames = list(member_frames)
                traces = _supporting_traces(matched_frames, frame_by_id)
                seen.add(record.basin_id)
                entries.append((record, matched_frames, traces, activation_score))

            for hint_key, hint_weight in hints.items():
                if hint_weight < _MIN_PRESSURE:
                    continue
                for record in self._bank.lookup_family(hint_key):
                    if record.basin_id in seen:
                        continue
                    matched_frames = member_frames or [
                        frame_id
                        for frame_id, frame in frame_by_id.items()
                        if float(record.frame_affinities.get(frame_id, 0.0)) >= _MIN_AFFINITY
                    ]
                    traces = _supporting_traces(matched_frames, frame_by_id)
                    activation_score = _activation_signature_score(
                        record,
                        _activation_tokens(matched_frames, frame_by_id, traces),
                    )
                    seen.add(record.basin_id)
                    entries.append((record, matched_frames, traces, activation_score))

            scoped[scope_id] = entries
        return scoped

    def _score_candidates(
        self,
        scoped: dict[str, list[tuple[BasinBankRecord, list[str], list[str], float]]],
        frame_by_id: dict[str, CandidateFrame],
        pressure_by_scope: dict[str, LocalBasinPressure],
        interference: InterferenceOutput,
        prior_basin_state: list[CandidateBasinState],
    ) -> list[_ScoredBasin]:
        scored: list[_ScoredBasin] = []
        prior_energy = _prior_energy_by_scope(prior_basin_state)
        suppression_penalty = _suppression_penalties(scoped)
        edge_boost: dict[tuple[str, str], float] = {}
        for edge in interference.frame_basin_edges:
            key = (edge.frame_id, edge.basin_id)
            edge_boost[key] = edge_boost.get(key, 0.0) + float(edge.delta)

        compete_penalty: dict[str, float] = {}
        for edge in interference.basin_basin_edges:
            if edge.relation != "compete":
                continue
            compete_penalty[edge.basin_id_a] = compete_penalty.get(edge.basin_id_a, 0.0) + abs(
                float(edge.delta)
            )
            compete_penalty[edge.basin_id_b] = compete_penalty.get(edge.basin_id_b, 0.0) + abs(
                float(edge.delta)
            )

        for scope_id, entries in scoped.items():
            pressure = pressure_by_scope.get(scope_id)
            hints = dict(pressure.basin_family_hints) if pressure else {}
            for record, member_frames, traces, activation_score in entries:
                affinity_score = 0.0
                for frame_id in member_frames:
                    frame = frame_by_id.get(frame_id)
                    affinity = float(record.frame_affinities.get(frame_id, 0.0))
                    frame_conf = frame.confidence if frame else 0.5
                    affinity_score = max(affinity_score, affinity * frame_conf)

                family_key = normalize_family_hint(record.family_hint)
                pressure_score = max(
                    (
                        hints.get(key, 0.0)
                        for key in hints
                        if normalize_family_hint(key) == family_key
                    ),
                    default=0.0,
                )
                trace_score = _trace_coherence(traces, member_frames, frame_by_id)
                interference_delta = float(interference.basin_energy_deltas.get(record.basin_id, 0.0))
                frame_edge_boost = sum(
                    edge_boost.get((frame_id, record.basin_id), 0.0)
                    for frame_id in member_frames
                )
                prior_boost = _PRIOR_STATE_SCALE * max(
                    prior_energy.get((scope_id, record.basin_id), 0.0),
                    prior_energy.get(("", record.basin_id), 0.0),
                )
                energy = (
                    0.25
                    + 0.35 * affinity_score
                    + 0.25 * pressure_score
                    + 0.15 * trace_score
                    + 0.2 * activation_score
                    + 0.1 * max(0.0, min(1.0, record.trust_score))
                    + _heat_bonus(record.heat_tier)
                    + interference_delta
                    + frame_edge_boost
                    + prior_boost
                    - compete_penalty.get(record.basin_id, 0.0)
                    - suppression_penalty.get((scope_id, record.basin_id), 0.0)
                )
                scored.append(
                    _ScoredBasin(
                        record=record,
                        scope_id=scope_id,
                        member_frames=list(member_frames),
                        traces=list(traces),
                        activation_score=round(activation_score, 4),
                        energy=round(energy, 4),
                    )
                )
        return scored

    def _build_states(
        self,
        scored: list[_ScoredBasin],
        frame_by_id: dict[str, CandidateFrame],
    ) -> list[CandidateBasinState]:
        states: list[CandidateBasinState] = []
        for item in scored:
            states.append(
                CandidateBasinState(
                    basin_id=item.record.basin_id,
                    energy=item.energy,
                    supporting_trace_ids=sorted(set(item.traces)),
                    supporting_frame_ids=sorted(set(item.member_frames)),
                    scope_frame_ids=[item.scope_id],
                    coherence_score=round(
                        _trace_coherence(item.traces, item.member_frames, frame_by_id),
                        4,
                    ),
                    activation_signature=dict(item.record.activation_signature),
                    semantic_signature=dict(item.record.semantic_signature),
                    evidence_handles=list(item.record.evidence_handles),
                    relation_handles=list(item.record.relation_handles),
                    source_refs=list(item.record.source_refs),
                    trust_score=item.record.trust_score,
                    heat_tier=item.record.heat_tier,
                    quantized_payload=dict(item.record.quantized_payload),
                )
            )
        return states

    def _filter_and_rank(
        self,
        states: list[CandidateBasinState],
        max_active: int,
    ) -> list[CandidateBasinState]:
        kept = [state for state in states if state.energy >= self.config.min_energy]
        limit = max(1, int(max_active))
        kept.sort(key=lambda item: item.energy, reverse=True)
        if len(kept) <= limit:
            return kept

        by_scope: dict[str, list[CandidateBasinState]] = {}
        for state in kept:
            by_scope.setdefault(_primary_scope(state), []).append(state)

        selected: list[CandidateBasinState] = []
        selected_keys: set[tuple[str, str]] = set()
        ranked_scopes = sorted(
            by_scope.values(),
            key=lambda scope_states: scope_states[0].energy,
            reverse=True,
        )
        for scope_states in ranked_scopes:
            if len(selected) >= limit:
                break
            state = scope_states[0]
            key = (_primary_scope(state), state.basin_id)
            selected.append(state)
            selected_keys.add(key)

        if len(selected) < limit:
            for state in kept:
                key = (_primary_scope(state), state.basin_id)
                if key in selected_keys:
                    continue
                selected.append(state)
                selected_keys.add(key)
                if len(selected) >= limit:
                    break

        selected.sort(key=lambda item: item.energy, reverse=True)
        return selected

    def _assign_margins(self, states: list[CandidateBasinState]) -> list[CandidateBasinState]:
        by_scope: dict[str, list[CandidateBasinState]] = {}
        for state in states:
            scope = state.scope_frame_ids[0] if state.scope_frame_ids else ""
            by_scope.setdefault(scope, []).append(state)

        ranked: list[CandidateBasinState] = []
        for scope_states in by_scope.values():
            scope_states.sort(key=lambda item: item.energy, reverse=True)
            for index, state in enumerate(scope_states):
                margin = 0.0
                if index + 1 < len(scope_states):
                    margin = state.energy - scope_states[index + 1].energy
                ranked.append(
                    CandidateBasinState(
                        basin_id=state.basin_id,
                        energy=state.energy,
                        assembly_id=state.assembly_id,
                        member_basin_ids=list(state.member_basin_ids),
                        supporting_trace_ids=list(state.supporting_trace_ids),
                        supporting_frame_ids=list(state.supporting_frame_ids),
                        scope_frame_ids=list(state.scope_frame_ids),
                        margin_vs_next=round(max(0.0, margin), 4),
                        coherence_score=state.coherence_score,
                        activation_signature=dict(state.activation_signature),
                        semantic_signature=dict(state.semantic_signature),
                        evidence_handles=list(state.evidence_handles),
                        relation_handles=list(state.relation_handles),
                        source_refs=list(state.source_refs),
                        trust_score=state.trust_score,
                        heat_tier=state.heat_tier,
                        quantized_payload=dict(state.quantized_payload),
                    )
                )
        ranked.sort(key=lambda item: item.energy, reverse=True)
        return ranked

    def _build_assemblies(
        self,
        states: list[CandidateBasinState],
        interference: InterferenceOutput,
    ) -> list[BasinAssembly]:
        state_by_scope_id = {(_primary_scope(state), state.basin_id): state for state in states}
        scopes = sorted({_primary_scope(state) for state in states})
        assemblies: list[BasinAssembly] = []
        seen: set[str] = set()

        for assembly_key, members in interference.cooperation_maps.items():
            base_assembly_id = (
                assembly_key if assembly_key.startswith("asy_") else f"asy_{uuid4().hex[:8]}"
            )
            for scope_id in scopes:
                member_ids = [
                    member for member in members if (scope_id, member) in state_by_scope_id
                ]
                if len(member_ids) < 2:
                    continue
                assembly_id = _scoped_assembly_id(base_assembly_id, scope_id)
                if assembly_id in seen:
                    continue
                seen.add(assembly_id)
                member_states = [state_by_scope_id[(scope_id, member)] for member in member_ids]
                combined = sum(state.energy for state in member_states)
                memory = _assembly_memory(member_states)
                assemblies.append(
                    BasinAssembly(
                        assembly_id=assembly_id,
                        member_basin_ids=member_ids,
                        combined_energy=round(combined, 4),
                        assembly_coherence=round(min(1.0, combined / len(member_ids)), 4),
                        scope_frame_ids=[scope_id] if scope_id else [],
                        evidence_handles=memory["evidence_handles"],
                        relation_handles=memory["relation_handles"],
                        source_refs=memory["source_refs"],
                        quantized_payload=memory["quantized_payload"],
                    )
                )

        for record in self._bank.records:
            for scope_id in scopes:
                if (scope_id, record.basin_id) not in state_by_scope_id:
                    continue
                partners = [
                    partner
                    for partner, weight in record.cooperation_links.items()
                    if weight > 0.0 and (scope_id, partner) in state_by_scope_id
                ]
                if not partners:
                    continue
                member_ids = [record.basin_id, *partners]
                assembly_id = _scoped_assembly_id(f"asy_{record.basin_id}", scope_id)
                if assembly_id in seen:
                    continue
                seen.add(assembly_id)
                member_states = [state_by_scope_id[(scope_id, member)] for member in member_ids]
                combined = sum(state.energy for state in member_states)
                memory = _assembly_memory(member_states)
                assemblies.append(
                    BasinAssembly(
                        assembly_id=assembly_id,
                        member_basin_ids=member_ids,
                        combined_energy=round(combined, 4),
                        assembly_coherence=round(min(1.0, combined / len(member_ids)), 4),
                        scope_frame_ids=[scope_id] if scope_id else [],
                        evidence_handles=memory["evidence_handles"],
                        relation_handles=memory["relation_handles"],
                        source_refs=memory["source_refs"],
                        quantized_payload=memory["quantized_payload"],
                    )
                )
        return assemblies

    def _detect_conflicts(self, states: list[CandidateBasinState]) -> list[BasinConflict]:
        conflicts: list[BasinConflict] = []
        by_scope: dict[str, list[CandidateBasinState]] = {}
        for state in states:
            scope = state.scope_frame_ids[0] if state.scope_frame_ids else ""
            by_scope.setdefault(scope, []).append(state)

        for scope_id, scope_states in by_scope.items():
            if len(scope_states) < 2:
                continue
            scope_states.sort(key=lambda item: item.energy, reverse=True)
            margin = scope_states[0].energy - scope_states[1].energy
            if margin <= _CONFLICT_MARGIN:
                conflicts.append(
                    BasinConflict(
                        scope_frame_id=scope_id,
                        conflict_type="low_margin_competition",
                        basin_ids=[state.basin_id for state in scope_states[:3]],
                    )
                )
        return conflicts


def _supporting_traces(
    member_frames: list[str],
    frame_by_id: dict[str, CandidateFrame],
) -> list[str]:
    traces: set[str] = set()
    for frame_id in member_frames:
        frame = frame_by_id.get(frame_id)
        if frame is None:
            continue
        traces.update(tid for tid in frame.role_assignments.values() if tid)
        traces.update(frame.supporting_trace_ids)
    return sorted(traces)


def _activation_tokens(
    member_frames: list[str],
    frame_by_id: dict[str, CandidateFrame],
    traces: list[str],
) -> set[str]:
    tokens = {normalize_family_hint(trace) for trace in traces if trace}
    for trace in traces:
        token = normalize_family_hint(trace)
        if token.startswith("t_") and len(token) > 2:
            tokens.add(token[2:])
    for frame_id in member_frames:
        frame = frame_by_id.get(frame_id)
        if frame is None:
            continue
        tokens.add(normalize_family_hint(frame.frame_id))
        tokens.add(normalize_family_hint(frame.frame_type))
        for value in list(frame.role_assignments.values()) + list(frame.relation_assignments.values()):
            token = normalize_family_hint(value)
            if token:
                tokens.add(token)
                if token.startswith("t_") and len(token) > 2:
                    tokens.add(token[2:])
        for graph in frame.local_graphs:
            tokens.add(normalize_family_hint(graph.family))
            for source_ref in graph.source_refs:
                tokens.add(normalize_family_hint(source_ref))
            for node in graph.nodes:
                tokens.add(normalize_family_hint(node.node_id))
                tokens.add(normalize_family_hint(node.label))
            for edge in graph.edges:
                tokens.add(normalize_family_hint(edge.label))
                for ref in edge.provenance_refs:
                    tokens.add(normalize_family_hint(ref))
    tokens.discard("")
    return tokens


def _activation_signature_score(record: BasinBankRecord, tokens: set[str]) -> float:
    if not record.activation_signature or not tokens:
        return 0.0
    total_weight = 0.0
    matched_weight = 0.0
    for key, raw_weight in record.activation_signature.items():
        weight = max(0.0, float(raw_weight))
        if weight <= 0.0:
            continue
        total_weight += weight
        token = normalize_family_hint(key)
        if token in tokens:
            matched_weight += weight
            continue
        if token.startswith("t_") and token[2:] in tokens:
            matched_weight += weight * 0.9
    if total_weight <= 0.0:
        return 0.0
    return round(min(1.0, matched_weight / total_weight), 4)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for value in values:
        item = str(value)
        if not item or item in seen:
            continue
        rows.append(item)
        seen.add(item)
    return rows


def _assembly_memory(states: list[CandidateBasinState]) -> dict[str, object]:
    evidence_handles = _dedupe([handle for state in states for handle in state.evidence_handles])
    relation_handles = _dedupe([handle for state in states for handle in state.relation_handles])
    source_refs = _dedupe([ref for state in states for ref in state.source_refs])
    quantized_payload: dict[str, object] = {
        "precision": "assembly_sparse_handles",
        "member_count": len(states),
        "evidence_handle_count": len(evidence_handles),
        "relation_handle_count": len(relation_handles),
    }
    return {
        "evidence_handles": evidence_handles,
        "relation_handles": relation_handles,
        "source_refs": source_refs,
        "quantized_payload": quantized_payload,
    }


def _primary_scope(state: CandidateBasinState) -> str:
    return state.scope_frame_ids[0] if state.scope_frame_ids else ""


def _safe_fragment(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return clean.strip("_") or "global"


def _scoped_assembly_id(base_id: str, scope_id: str) -> str:
    if not scope_id:
        return base_id
    return f"{base_id}__scope_{_safe_fragment(scope_id)}"


def _prior_energy_by_scope(
    prior_states: list[CandidateBasinState],
) -> dict[tuple[str, str], float]:
    energy_by_scope: dict[tuple[str, str], float] = {}
    for state in prior_states:
        if not state.basin_id:
            continue
        energy = min(1.0, max(0.0, float(state.energy)))
        for scope_id in state.scope_frame_ids or [""]:
            key = (scope_id, state.basin_id)
            energy_by_scope[key] = max(energy_by_scope.get(key, 0.0), energy)
    return energy_by_scope


def _suppression_penalties(
    scoped: dict[str, list[tuple[BasinBankRecord, list[str], list[str], float]]],
) -> dict[tuple[str, str], float]:
    penalties: dict[tuple[str, str], float] = {}
    for scope_id, entries in scoped.items():
        basin_ids = {record.basin_id for record, _frames, _traces, _activation in entries}
        for record, _frames, _traces, _activation in entries:
            for target_id, weight in record.suppression_links.items():
                if target_id not in basin_ids or target_id == record.basin_id:
                    continue
                key = (scope_id, target_id)
                penalties[key] = penalties.get(key, 0.0) + max(0.0, float(weight)) * _SUPPRESSION_SCALE
    return penalties


def _trace_coherence(
    traces: list[str],
    member_frames: list[str],
    frame_by_id: dict[str, CandidateFrame],
) -> float:
    if not member_frames:
        return 0.35
    confidences = [
        frame_by_id[frame_id].confidence
        for frame_id in member_frames
        if frame_id in frame_by_id
    ]
    base = sum(confidences) / len(confidences) if confidences else 0.35
    if traces:
        base = min(1.0, base + 0.1 * len(traces))
    return base


def _heat_bonus(heat_tier: str) -> float:
    return {
        "hot": 0.06,
        "warm": 0.04,
        "cold": 0.02,
        "stabilized": 0.04,
        "probation": 0.0,
        "quarantine": -0.01,
    }.get(str(heat_tier or "").strip().lower(), 0.0)


def _competition_summary(states: list[CandidateBasinState]) -> CompetitionSummary:
    if not states:
        return CompetitionSummary()
    ordered = sorted(states, key=lambda item: item.energy, reverse=True)
    top = ordered[0]
    top_scope = _primary_scope(top)
    second = next((state for state in ordered[1:] if _primary_scope(state) == top_scope), None)
    margin = top.energy - second.energy if second else top.energy
    return CompetitionSummary(
        top_basin_id=top.basin_id,
        second_basin_id=second.basin_id if second else "",
        top_margin=round(max(0.0, margin), 4),
        active_basin_count=len(states),
    )


def _binding_stability_hint(frames: list[CandidateFrame]) -> float:
    if not frames:
        return 0.0
    return round(sum(frame.confidence for frame in frames) / len(frames), 4)


def _audit_notes(
    *,
    bank: BasinBank,
    states: list[CandidateBasinState],
    assemblies: list[BasinAssembly],
    conflicts: list[BasinConflict],
    inp: BasinInput,
) -> list[str]:
    return [
        f"basin_bank_size={len(bank.records)}",
        f"context_frames={len(inp.context_frames)}",
        f"prior_basin_states={len(inp.prior_basin_state)}",
        f"candidate_basin_states={len(states)}",
        f"basin_assemblies={len(assemblies)}",
        f"basin_evidence_handles={sum(len(state.evidence_handles) for state in states)}",
        f"basin_source_refs={sum(len(state.source_refs) for state in states)}",
        (
            "suppression_links="
            f"{sum(len(record.suppression_links) for record in bank.records)}"
        ),
        f"unresolved_conflicts={len(conflicts)}",
        (
            "interference_deltas="
            f"{len(inp.interference_output.basin_energy_deltas)}"
        ),
    ]
