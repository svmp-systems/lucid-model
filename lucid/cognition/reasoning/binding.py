"""Binding: plural candidate frames from perception evidence and DMF activations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lucid.cognition.input.cue.encoder import normalize_cue_key
from lucid.cognition.reasoning.cue_routes import (
    competing_units as plural_cue_units,
    cue_keys_per_unit,
)
from lucid.ir.binding import (
    BindingInput,
    BindingOutput,
    CandidateFrame,
    FrameCompetitionEdge,
)
from lucid.ir.cue import CueCloud
from lucid.ir.dmf import ActiveTrace, DmfOutput
from lucid.ir.perception import (
    CandidateUnit,
    ChangeHint,
    PerceptualEvidenceGraph,
)
from lucid.memory.dmf import tracebank_from_checkpoint

_TOKEN_RE = re.compile(r"[^a-z0-9_]+")

_REGION_FRAME_HINTS: dict[str, str] = {
    "main_clause": "event_one",
    "relative_clause": "event_two",
    "legend_or_key_region": "frame_legend",
    "canvas": "frame_canvas",
}

_DEPOSIT_SURFACES = frozenset({"placed", "deposited", "put", "stored", "left"})
_LOCATION_SURFACES = frozenset({"bank", "vault", "safe", "river", "market"})


@dataclass(slots=True)
class BindingConfig:
    checkpoint: str | Path | None = None
    affordances: dict[str, Any] | None = None
    min_frame_confidence: float = 0.2
    min_trace_attachment: float = 0.12
    widen_on_recheck: bool = True


@dataclass(slots=True)
class FrameSeed:
    seed_id: str
    frame_type: str
    unit_ids: set[str] = field(default_factory=set)
    structural_tags: list[str] = field(default_factory=list)
    source: str = ""
    forced_trace_assignments: dict[str, str] = field(default_factory=dict)


def run_binding(inp: BindingInput, *, config: BindingConfig | None = None) -> BindingOutput:
    operator = BindingOperator(config or BindingConfig())
    return operator.run(inp)


class BindingOperator:
    def __init__(self, config: BindingConfig) -> None:
        self.config = config
        self._affordances = config.affordances
        if self._affordances is None and config.checkpoint:
            path = Path(config.checkpoint)
            if path.exists():
                self._affordances = _load_affordances(path)

    def run(self, inp: BindingInput) -> BindingOutput:
        graph = inp.perceptual_evidence_graph
        dmf = inp.dmf_output
        cue = inp.cue_cloud or CueCloud()
        affordances = self._affordances or _empty_affordances()
        cue_to_trace = _build_cue_to_trace_map(
            self.config.checkpoint,
            dmf.active_traces,
        )
        unit_by_id = {unit.unit_id: unit for unit in graph.candidate_units}
        unit_traces = _unit_trace_weights(cue, cue_to_trace, dmf.active_traces, unit_by_id, dmf)
        competing_units = _competing_unit_ids(cue, unit_traces, dmf.active_traces)

        seeds = _collect_seeds(graph, inp.prior_candidate_frames, affordances)
        local_seeds = _local_reading_seeds(
            graph=graph,
            unit_by_id=unit_by_id,
            unit_traces=unit_traces,
            active_traces=dmf.active_traces,
            competing_units=competing_units,
        )
        seeds.extend(local_seeds)
        if inp.prior_candidate_frames and self.config.widen_on_recheck:
            seeds.extend(_seeds_from_prior_frames(inp.prior_candidate_frames))

        frames: list[CandidateFrame] = []
        audit_notes: list[str] = []

        for seed in seeds:
            frame = self._materialize_frame(
                seed=seed,
                graph=graph,
                dmf=dmf,
                unit_by_id=unit_by_id,
                unit_traces=unit_traces,
                competing_units=competing_units,
                affordances=affordances,
                cue_to_trace=cue_to_trace,
            )
            if frame.confidence >= self.config.min_frame_confidence:
                frames.append(frame)

        frames = _dedupe_frames(frames)
        if not frames and dmf.active_traces:
            frames.append(
                self._fallback_frame(
                    graph=graph,
                    dmf=dmf,
                    unit_by_id=unit_by_id,
                    unit_traces=unit_traces,
                    competing_units=competing_units,
                    cue_to_trace=cue_to_trace,
                )
            )
            audit_notes.append("fallback_single_frame_from_active_traces")

        edges = _competition_edges(frames, dmf, graph)
        stability = _binding_stability_score(frames)

        audit_notes.extend(
            [
                f"seeds={len(seeds)}",
                f"local_reading_seeds={len(local_seeds)}",
                f"candidate_frames={len(frames)}",
                f"active_traces={len(dmf.active_traces)}",
                f"stability={stability:.3f}",
            ]
        )

        return BindingOutput(
            candidate_frames=frames,
            frame_competition_edges=edges,
            binding_stability_score=stability,
            audit_notes=audit_notes,
        )

    def _materialize_frame(
        self,
        *,
        seed: FrameSeed,
        graph: PerceptualEvidenceGraph,
        dmf: DmfOutput,
        unit_by_id: dict[str, CandidateUnit],
        unit_traces: dict[str, dict[str, float]],
        competing_units: set[str],
        affordances: dict[str, Any],
        cue_to_trace: dict[str, list[str]],
    ) -> CandidateFrame:
        slot_assignments: dict[str, str] = {}
        slot_evidence_refs: dict[str, list[str]] = {}
        slot_affinity_hints: dict[str, dict[str, float]] = {}
        member_refs: list[str] = []
        supporting: list[str] = []
        unresolved: list[str] = []

        trace_scores: dict[str, float] = {}
        for unit_id in sorted(seed.unit_ids):
            if unit_id not in unit_by_id:
                continue
            member_refs.append(unit_id)
            for trace_id, weight in unit_traces.get(unit_id, {}).items():
                trace_scores[trace_id] = max(trace_scores.get(trace_id, 0.0), weight)

        for trace in dmf.active_traces:
            if trace.trace_id and trace.trace_id not in trace_scores:
                if any(
                    trace.trace_id in unit_traces.get(uid, {})
                    for uid in seed.unit_ids
                ):
                    trace_scores[trace.trace_id] = trace.activation

        ranked_traces = sorted(trace_scores.items(), key=lambda item: (-item[1], item[0]))
        for slot_index, unit_id in enumerate(_ordered_unit_ids(member_refs, unit_by_id)):
            unit = unit_by_id[unit_id]
            slot_id = f"slot_{slot_index:02d}"
            slot_evidence_refs[slot_id] = [unit_id]
            hints = _slot_affinity_hints_for_unit(
                unit=unit,
                seed=seed,
                affordances=affordances,
                graph=graph,
            )
            if hints:
                slot_affinity_hints[slot_id] = hints
            forced_trace = seed.forced_trace_assignments.get(unit_id, "").strip()
            if forced_trace:
                slot_assignments[slot_id] = forced_trace
                supporting.append(forced_trace)
                continue
            if unit_id in competing_units:
                unresolved.append(f"{normalize_cue_key(unit.surface)}_sense")
                continue

            best_trace = _best_trace_for_unit(unit_id, ranked_traces, unit_traces)
            if not best_trace:
                continue
            trace_id = _resolve_trace_id(best_trace, cue_to_trace, dmf.active_traces)
            slot_assignments[slot_id] = trace_id
            supporting.append(trace_id)

        conflicting = sorted(
            set(_conflicting_traces_for_frame(supporting, dmf))
            | set(
                _conflicting_traces_from_unit_routes(
                    member_refs,
                    unit_traces,
                    competing_units,
                    dmf,
                )
            )
        )

        confidence = _frame_confidence(
            seed,
            ranked_traces,
            slot_assignments,
            slot_evidence_refs,
            unresolved,
            conflicting,
            dmf,
        )

        return CandidateFrame(
            frame_id=seed.seed_id,
            frame_type=seed.frame_type,
            role_assignments=slot_assignments,
            slot_evidence_refs=slot_evidence_refs,
            slot_affinity_hints=slot_affinity_hints,
            member_evidence_refs=sorted(set(member_refs)),
            confidence=round(confidence, 3),
            unresolved_slot_names=sorted(set(unresolved)),
            supporting_trace_ids=sorted(set(supporting)),
            conflicting_trace_ids=sorted(set(conflicting)),
        )

    def _fallback_frame(
        self,
        *,
        graph: PerceptualEvidenceGraph,
        dmf: DmfOutput,
        unit_by_id: dict[str, CandidateUnit],
        unit_traces: dict[str, dict[str, float]],
        competing_units: set[str],
        cue_to_trace: dict[str, list[str]],
    ) -> CandidateFrame:
        seed = FrameSeed(
            seed_id="frame_active",
            frame_type="event",
            unit_ids=set(unit_by_id),
            structural_tags=["fallback"],
            source="dmf_active",
        )
        return self._materialize_frame(
            seed=seed,
            graph=graph,
            dmf=dmf,
            unit_by_id=unit_by_id,
            unit_traces=unit_traces,
            competing_units=competing_units,
            affordances=_empty_affordances(),
            cue_to_trace=cue_to_trace,
        )


def _resolve_trace_id(
    key: str,
    cue_to_trace: dict[str, list[str]],
    active_traces: list[ActiveTrace],
) -> str:
    raw = key.strip()
    if not raw:
        return ""
    normalized = normalize_cue_key(raw)
    if normalized in cue_to_trace and cue_to_trace[normalized]:
        return cue_to_trace[normalized][0]
    for trace in active_traces:
        if trace.trace_id == raw:
            return trace.trace_id
    for trace in active_traces:
        token = normalize_cue_key(trace.trace_id)
        if token and (token in normalized or normalized in token):
            return trace.trace_id
    return raw


def _empty_affordances() -> dict[str, Any]:
    return {"patterns": [], "region_frame_hints": dict(_REGION_FRAME_HINTS)}


def _load_affordances(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint / "binding_affordances.json"
    if not path.exists():
        return _empty_affordances()
    store = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(store, dict):
        return _empty_affordances()
    store.setdefault("patterns", [])
    store.setdefault("region_frame_hints", dict(_REGION_FRAME_HINTS))
    return store


def _collect_seeds(
    graph: PerceptualEvidenceGraph,
    prior_frames: list[CandidateFrame],
    affordances: dict[str, Any],
) -> list[FrameSeed]:
    seeds: list[FrameSeed] = []
    seen: set[str] = set()
    region_hints = affordances.get("region_frame_hints") or _REGION_FRAME_HINTS

    for region in graph.candidate_regions:
        frame_id = str(region_hints.get(region.role_hint, "") or _slug_frame_id(region.region_id))
        seed_id = frame_id if frame_id.startswith("event") or frame_id.startswith("frame") else _slug_frame_id(region.region_id)
        if seed_id in seen:
            continue
        seeds.append(
            FrameSeed(
                seed_id=seed_id,
                frame_type=_frame_type_from_role_hint(region.role_hint),
                unit_ids=set(region.member_unit_ids),
                structural_tags=[f"region:{region.role_hint or region.region_id}"],
                source="candidate_region",
            )
        )
        seen.add(seed_id)

    for container in graph.candidate_containers:
        seed_id = _slug_frame_id(container.container_id)
        if seed_id in seen:
            continue
        unit_ids: set[str] = set()
        if container.interior_region_id:
            for region in graph.candidate_regions:
                if region.region_id == container.interior_region_id:
                    unit_ids.update(region.member_unit_ids)
        seeds.append(
            FrameSeed(
                seed_id=seed_id,
                frame_type=_frame_type_from_role_hint(container.kind_hint),
                unit_ids=unit_ids,
                structural_tags=[f"container:{container.kind_hint}"],
                source="candidate_container",
            )
        )
        seen.add(seed_id)

    for group in graph.grouping_hints:
        seed_id = _slug_frame_id(group.group_id)
        if seed_id in seen:
            continue
        seeds.append(
            FrameSeed(
                seed_id=seed_id,
                frame_type="event" if "clause" in group.grouping_reason else "relation",
                unit_ids=set(group.member_unit_ids),
                structural_tags=[f"group:{group.grouping_reason}"],
                source="grouping_hint",
            )
        )
        seen.add(seed_id)

    for hint in graph.change_hints:
        seed = _seed_from_change_hint(hint, graph)
        if seed and seed.seed_id not in seen:
            seeds.append(seed)
            seen.add(seed.seed_id)

    if not seeds:
        seeds.extend(_seeds_from_units_and_references(graph, seen))

    verb_like = [unit for unit in graph.candidate_units if _is_verb_like(unit)]
    if len(verb_like) >= 2 and len(seeds) < 2:
        existing_ids = {seed.seed_id for seed in seeds}
        for extra in _seeds_from_units_and_references(graph, set(seen)):
            if extra.seed_id not in existing_ids:
                seeds.append(extra)
                seen.add(extra.seed_id)
                existing_ids.add(extra.seed_id)

    if not seeds and prior_frames:
        seeds.extend(_seeds_from_prior_frames(prior_frames))

    return seeds


def _seeds_from_units_and_references(
    graph: PerceptualEvidenceGraph,
    seen: set[str],
) -> list[FrameSeed]:
    seeds: list[FrameSeed] = []
    units = graph.candidate_units
    if not units:
        return seeds

    verb_units = [
        unit
        for unit in units
        if _is_verb_like(unit)
    ]
    if len(verb_units) >= 2:
        ordered = sorted(units, key=_position_key)
        pivot = ordered.index(verb_units[1])
        first_units = {unit.unit_id for unit in ordered[:pivot]}
        second_units = {unit.unit_id for unit in ordered[pivot:]}
        for seed_id, unit_ids, tag in (
            ("event_one", first_units, "verb_split:first"),
            ("event_two", second_units, "verb_split:second"),
        ):
            if seed_id not in seen and unit_ids:
                seeds.append(
                    FrameSeed(
                        seed_id=seed_id,
                        frame_type="event",
                        unit_ids=set(unit_ids),
                        structural_tags=[tag],
                        source="verb_split",
                    )
                )
                seen.add(seed_id)
        return seeds

    if units:
        seed_id = "event_one"
        if seed_id not in seen:
            seeds.append(
                FrameSeed(
                    seed_id=seed_id,
                    frame_type="event",
                    unit_ids={unit.unit_id for unit in units},
                    structural_tags=["unit_blob"],
                    source="all_units",
                )
            )
    return seeds


def _seed_from_change_hint(
    hint: ChangeHint,
    graph: PerceptualEvidenceGraph,
) -> FrameSeed | None:
    unit_ids: set[str] = set()
    if hint.before_unit_id:
        unit_ids.add(hint.before_unit_id)
    if hint.after_unit_id:
        unit_ids.add(hint.after_unit_id)
    if not unit_ids:
        return None
    frame_type = "transform" if "shift" in hint.change_type else "attribute_map"
    return FrameSeed(
        seed_id=f"frame_{normalize_cue_key(hint.change_type)}",
        frame_type=frame_type,
        unit_ids=unit_ids,
        structural_tags=[f"change:{hint.change_type}"],
        source="change_hint",
    )


def _seeds_from_prior_frames(frames: list[CandidateFrame]) -> list[FrameSeed]:
    seeds: list[FrameSeed] = []
    for frame in frames:
        seeds.append(
            FrameSeed(
                seed_id=frame.frame_id,
                frame_type=frame.frame_type,
                unit_ids=set(frame.member_evidence_refs),
                structural_tags=["prior_frame"],
                source="prior_candidate_frames",
            )
        )
    return seeds


def _ambiguous_unit_ids(graph: PerceptualEvidenceGraph) -> set[str]:
    return {
        flag.target_id
        for flag in graph.uncertainty_flags
        if flag.target_id and flag.uncertainty_type in {"polysemy", "ambiguous"}
    }


def _local_unit_window(
    focus_unit_id: str,
    unit_by_id: dict[str, CandidateUnit],
    *,
    radius: int = 2,
) -> set[str]:
    ordered = sorted(unit_by_id.values(), key=_position_key)
    focus_index = next(
        (index for index, unit in enumerate(ordered) if unit.unit_id == focus_unit_id),
        -1,
    )
    if focus_index < 0:
        return {focus_unit_id}
    start = max(0, focus_index - radius)
    stop = min(len(ordered), focus_index + radius + 1)
    return {unit.unit_id for unit in ordered[start:stop]}


def _best_trace_per_cluster(
    trace_weights: dict[str, float],
    active_traces: list[ActiveTrace],
    *,
    min_weight: float = 0.12,
    limit: int = 4,
) -> list[str]:
    active_by_id = {trace.trace_id: trace for trace in active_traces if trace.trace_id}
    by_cluster: dict[str, tuple[str, float]] = {}
    for trace_id, weight in trace_weights.items():
        if trace_id not in active_by_id or weight < min_weight:
            continue
        trace = active_by_id[trace_id]
        cluster = trace.cluster_id or trace.trace_id
        previous = by_cluster.get(cluster)
        if previous is None or weight > previous[1]:
            by_cluster[cluster] = (trace_id, weight)
    ranked = sorted(by_cluster.values(), key=lambda item: (-item[1], item[0]))
    return [trace_id for trace_id, _weight in ranked[:limit]]


def _local_reading_seeds(
    *,
    graph: PerceptualEvidenceGraph,
    unit_by_id: dict[str, CandidateUnit],
    unit_traces: dict[str, dict[str, float]],
    active_traces: list[ActiveTrace],
    competing_units: set[str],
) -> list[FrameSeed]:
    ambiguous = _ambiguous_unit_ids(graph)
    focus_units = sorted(
        (competing_units & ambiguous) or competing_units,
        key=lambda unit_id: _position_key(unit_by_id[unit_id])
        if unit_id in unit_by_id
        else (0, unit_id, ""),
    )
    seeds: list[FrameSeed] = []
    seen: set[str] = set()
    for unit_id in focus_units:
        unit = unit_by_id.get(unit_id)
        if unit is None:
            continue
        trace_ids = _best_trace_per_cluster(unit_traces.get(unit_id, {}), active_traces)
        if len(trace_ids) < 2:
            continue
        local_units = _local_unit_window(unit_id, unit_by_id)
        surface = normalize_cue_key(unit.surface) or unit_id
        for trace_id in trace_ids:
            trace_slug = normalize_cue_key(trace_id) or _slug_frame_id(trace_id)
            seed_id = f"local_{unit_id}__{trace_slug}"
            if seed_id in seen:
                continue
            seeds.append(
                FrameSeed(
                    seed_id=seed_id,
                    frame_type="local_reading",
                    unit_ids=set(local_units),
                    structural_tags=[
                        f"local_focus:{unit_id}",
                        f"surface:{surface}",
                        f"forced_trace:{trace_id}",
                    ],
                    source="local_competing_trace",
                    forced_trace_assignments={unit_id: trace_id},
                )
            )
            seen.add(seed_id)
    return seeds


def _build_cue_to_trace_map(
    checkpoint: str | Path | None,
    active_traces: list[ActiveTrace],
) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}

    def add(key: str, trace_id: str) -> None:
        norm = normalize_cue_key(key)
        clean_trace = trace_id.strip()
        if not norm or not clean_trace:
            return
        bucket = mapping.setdefault(norm, [])
        if clean_trace not in bucket:
            bucket.append(clean_trace)

    if checkpoint:
        for record in tracebank_from_checkpoint(checkpoint):
            trace_id = record.trace_id.strip()
            if not trace_id:
                continue
            add(trace_id, trace_id)
            for key in record.cue_affinities:
                add(key, trace_id)
            if record.alias:
                add(record.alias, trace_id)
    for trace in active_traces:
        if trace.trace_id:
            add(trace.trace_id, trace.trace_id)
    return mapping


def _unit_trace_weights(
    cue: CueCloud,
    cue_to_trace: dict[str, list[str]],
    active_traces: list[ActiveTrace],
    unit_by_id: dict[str, CandidateUnit],
    dmf: DmfOutput | None = None,
) -> dict[str, dict[str, float]]:
    weights: dict[str, dict[str, float]] = {}
    activation_by_id = {trace.trace_id: trace.activation for trace in active_traces if trace.trace_id}

    def add(unit_id: str, trace_id: str, weight: float) -> None:
        if not unit_id or not trace_id:
            return
        bucket = weights.setdefault(unit_id, {})
        bucket[trace_id] = max(bucket.get(trace_id, 0.0), float(weight))

    if dmf is not None:
        for trace in active_traces:
            trace_id = trace.trace_id
            if not trace_id:
                continue
            for reason in dmf.activation_reasons.get(trace_id, []):
                contribution = _safe_float(reason.get("contribution"), 0.0)
                score = max(contribution, trace.activation * 0.5)
                refs = [
                    str(ref)
                    for ref in (
                        list(reason.get("evidence_refs") or [])
                        + list(reason.get("endpoint_unit_ids") or [])
                    )
                    if str(ref)
                ]
                for ref in refs:
                    add(ref, trace_id, score)

    for req in cue.primitive_trace_activations:
        cue_key = normalize_cue_key(req.trace_id)
        trace_ids = cue_to_trace.get(cue_key) or [req.trace_id]
        for trace_id in trace_ids:
            score = float(req.weight) * activation_by_id.get(trace_id, 0.55)
            for ref in req.evidence_refs:
                add(ref, trace_id, score)
            for unit_id, unit in unit_by_id.items():
                if cue_key and cue_key in normalize_cue_key(unit.surface):
                    add(unit_id, trace_id, score)

    for req in cue.relational_trace_activations:
        cue_key = normalize_cue_key(req.trace_id)
        trace_ids = cue_to_trace.get(cue_key) or [req.trace_id]
        for trace_id in trace_ids:
            score = float(req.weight) * activation_by_id.get(trace_id, 0.55)
            for ref in req.relation_refs:
                add(ref, trace_id, score)
            for ref in req.endpoint_unit_ids:
                add(ref, trace_id, score)

    for trace in active_traces:
        if not trace.trace_id:
            continue
        token = normalize_cue_key(trace.trace_id)
        for unit_id, unit in unit_by_id.items():
            surface = normalize_cue_key(unit.surface)
            if token and (token in surface or surface in token):
                add(unit_id, trace.trace_id, trace.activation)

    return weights


def _competing_unit_ids(
    cue: CueCloud,
    unit_traces: dict[str, dict[str, float]],
    active_traces: list[ActiveTrace],
    *,
    min_weight: float = 0.12,
) -> set[str]:
    routes = cue_keys_per_unit(cue)
    competing: set[str] = set(plural_cue_units(cue).keys())
    cluster_by_trace = {trace.trace_id: trace.cluster_id for trace in active_traces if trace.trace_id}
    active_ids = set(cluster_by_trace)

    for unit_id in set(routes) | set(unit_traces):
        strong = {
            trace_id
            for trace_id, weight in unit_traces.get(unit_id, {}).items()
            if trace_id in active_ids and weight >= min_weight
        }
        if len(strong) < 2:
            continue
        clusters = {cluster_by_trace.get(trace_id, trace_id) for trace_id in strong}
        if len(clusters) >= 2:
            competing.add(unit_id)
    return competing


def _conflicting_traces_from_unit_routes(
    member_refs: list[str],
    unit_traces: dict[str, dict[str, float]],
    competing_units: set[str],
    dmf: DmfOutput,
    *,
    min_weight: float = 0.12,
) -> list[str]:
    conflicts: set[str] = set()
    active_ids = {trace.trace_id for trace in dmf.active_traces if trace.trace_id}
    for unit_id in member_refs:
        if unit_id not in competing_units:
            continue
        strong = [
            trace_id
            for trace_id, weight in unit_traces.get(unit_id, {}).items()
            if trace_id in active_ids and weight >= min_weight
        ]
        conflicts.update(strong)
    conflicts.update(_conflicting_traces_for_frame(sorted(conflicts), dmf))
    return sorted(conflicts)


def _slot_affinity_hints_for_unit(
    *,
    unit: CandidateUnit,
    seed: FrameSeed,
    affordances: dict[str, Any],
    graph: PerceptualEvidenceGraph,
) -> dict[str, float]:
    _ = graph
    hints: dict[str, float] = {}
    patterns = affordances.get("patterns") or []
    feature_key = _unit_feature_key(unit)
    for pattern in patterns:
        if not isinstance(pattern, dict):
            continue
        if pattern.get("pattern_type") and pattern.get("pattern_type") != "slot_affinity":
            continue
        if pattern.get("feature_key") != feature_key:
            continue
        if pattern.get("frame_type") and pattern.get("frame_type") != seed.frame_type:
            continue
        hint = str(pattern.get("slot_hint") or "")
        if not hint and pattern.get("role_slot"):
            hint = _legacy_role_hint(str(pattern.get("role_slot")))
        if not hint:
            continue
        weight = _safe_float(pattern.get("weight"), 0.0)
        hints[hint] = max(hints.get(hint, 0.0), weight)

    kind = normalize_cue_key(unit.kind_hint or "span")
    surface = normalize_cue_key(unit.surface)
    if kind in {"verb", "verb_span"} or _is_verb_like(unit):
        hints["event_anchor_like"] = max(hints.get("event_anchor_like", 0.0), 0.55)
    if kind in {"noun", "noun_span", "cell"}:
        hints["object_like"] = max(hints.get("object_like", 0.0), 0.45)
    if kind in {"modifier", "gerund_clause"}:
        hints["context_like"] = max(hints.get("context_like", 0.0), 0.45)
    if surface in _DEPOSIT_SURFACES:
        hints["event_anchor_like"] = max(hints.get("event_anchor_like", 0.0), 0.6)
    if surface in _LOCATION_SURFACES:
        hints["location_like"] = max(hints.get("location_like", 0.0), 0.6)
    return dict(sorted(hints.items()))


def _unit_feature_key(unit: CandidateUnit) -> str:
    kind = normalize_cue_key(unit.kind_hint or "span")
    surface = normalize_cue_key(unit.surface)
    return f"unit:{kind}:{surface}"


def _legacy_role_hint(role_slot: str) -> str:
    normalized = normalize_cue_key(role_slot)
    if normalized in {"action", "predicate", "verb"}:
        return "event_anchor_like"
    if normalized in {"destination", "location", "source"}:
        return "location_like"
    if normalized in {"context", "modifier"}:
        return "context_like"
    if normalized in {"before", "after", "theme"}:
        return "object_like"
    return normalized


def _has_slot_hint(hints: dict[str, float], hint: str) -> bool:
    return hints.get(hint, 0.0) > 0.0


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _position_key(unit: CandidateUnit) -> tuple[int, str, str]:
    raw = str(unit.position_or_time or "").strip()
    try:
        position = int(raw)
    except ValueError:
        position = 0
    return position, unit.unit_id, unit.surface


def _ordered_unit_ids(
    unit_ids: list[str],
    unit_by_id: dict[str, CandidateUnit],
) -> list[str]:
    return sorted(
        unit_ids,
        key=lambda unit_id: _position_key(unit_by_id[unit_id])
        if unit_id in unit_by_id
        else (0, unit_id, ""),
    )


def _conflicting_traces_for_frame(
    supporting: list[str],
    dmf: DmfOutput,
) -> list[str]:
    support = set(supporting)
    conflicts: set[str] = set()
    for signal in dmf.conflict_signals:
        if not signal.trace_id_a or not signal.trace_id_b:
            continue
        if signal.trace_id_a in support and signal.trace_id_b in support:
            conflicts.add(signal.trace_id_a)
            conflicts.add(signal.trace_id_b)
    return sorted(conflicts)


def _best_trace_for_unit(
    unit_id: str,
    ranked_traces: list[tuple[str, float]],
    unit_traces: dict[str, dict[str, float]],
) -> str:
    local = unit_traces.get(unit_id, {})
    if local:
        return max(local.items(), key=lambda item: (item[1], item[0]))[0]
    _ = ranked_traces
    return ""


def _frame_confidence(
    seed: FrameSeed,
    ranked_traces: list[tuple[str, float]],
    slot_assignments: dict[str, str],
    slot_evidence_refs: dict[str, list[str]],
    unresolved: list[str],
    conflicting: list[str],
    dmf: DmfOutput,
) -> float:
    total_slots = max(1, len(slot_evidence_refs))
    assigned_slots = len(slot_assignments)
    evidence_coverage = assigned_slots / total_slots
    unique_ratio = (
        len(set(slot_assignments.values())) / assigned_slots
        if assigned_slots
        else 0.0
    )
    base = 0.2 + 0.35 * evidence_coverage + 0.2 * unique_ratio
    if ranked_traces:
        base += 0.15 * ranked_traces[0][1]
    base += 0.1 * min(1.0, dmf.coverage_score)
    base -= min(0.2, 0.05 * len(unresolved))
    base -= min(0.15, 0.05 * len(conflicting))
    if assigned_slots and unique_ratio < 1.0:
        base -= 0.1
    return max(0.0, min(1.0, base))


def _competition_edges(
    frames: list[CandidateFrame],
    dmf: DmfOutput,
    graph: PerceptualEvidenceGraph,
) -> list[FrameCompetitionEdge]:
    edges: list[FrameCompetitionEdge] = []
    conflict_pairs = {
        tuple(sorted((signal.trace_id_a, signal.trace_id_b)))
        for signal in dmf.conflict_signals
        if signal.trace_id_a and signal.trace_id_b
    }

    for index, left in enumerate(frames):
        left_traces = set(left.role_assignments.values()) | set(left.supporting_trace_ids)
        for right in frames[index + 1 :]:
            right_traces = set(right.role_assignments.values()) | set(right.supporting_trace_ids)
            left_focus = _local_focus_unit(left)
            right_focus = _local_focus_unit(right)
            if left_focus and left_focus == right_focus and left.frame_id != right.frame_id:
                edges.append(
                    FrameCompetitionEdge(
                        frame_id_a=left.frame_id,
                        frame_id_b=right.frame_id,
                        relation="compete",
                        weight=0.65,
                    )
                )
                continue
            if (
                left.frame_type == "local_reading"
                and right.frame_type == "local_reading"
                and set(left.member_evidence_refs) == set(right.member_evidence_refs)
                and left_traces
                and right_traces
                and left_traces.isdisjoint(right_traces)
            ):
                edges.append(
                    FrameCompetitionEdge(
                        frame_id_a=left.frame_id,
                        frame_id_b=right.frame_id,
                        relation="compete",
                        weight=0.65,
                    )
                )
            shared = left_traces & right_traces
            if shared:
                edges.append(
                    FrameCompetitionEdge(
                        frame_id_a=left.frame_id,
                        frame_id_b=right.frame_id,
                        relation="cooperate",
                        weight=round(min(1.0, 0.45 + 0.1 * len(shared)), 3),
                    )
                )
            competing = any(
                tuple(sorted((trace_a, trace_b))) in conflict_pairs
                for trace_a in left_traces
                for trace_b in right_traces
            )
            if competing:
                edges.append(
                    FrameCompetitionEdge(
                        frame_id_a=left.frame_id,
                        frame_id_b=right.frame_id,
                        relation="compete",
                        weight=0.7,
                    )
                )

    for hint in graph.reference_hints:
        source_frame = _frame_for_unit(frames, hint.source_unit_id)
        target_frame = _frame_for_unit(frames, hint.target_unit_id)
        if source_frame and target_frame and source_frame != target_frame:
            edges.append(
                FrameCompetitionEdge(
                    frame_id_a=source_frame,
                    frame_id_b=target_frame,
                    relation="cooperate",
                    weight=round(float(hint.confidence), 3),
                )
            )
    return edges


def _local_focus_unit(frame: CandidateFrame) -> str:
    if frame.frame_type != "local_reading" or not frame.frame_id.startswith("local_"):
        return ""
    rest = frame.frame_id[len("local_") :]
    if "__" not in rest:
        return ""
    return rest.split("__", 1)[0]


def _frame_for_unit(frames: list[CandidateFrame], unit_id: str) -> str:
    for frame in frames:
        if unit_id in frame.member_evidence_refs:
            return frame.frame_id
    return ""


def _binding_stability_score(frames: list[CandidateFrame]) -> float:
    if not frames:
        return 0.0
    confidences = sorted((frame.confidence for frame in frames), reverse=True)
    if len(confidences) == 1:
        return round(confidences[0], 3)
    margin = confidences[0] - confidences[1]
    return round(max(0.0, min(1.0, confidences[0] * 0.6 + margin * 0.4)), 3)


def _dedupe_frames(frames: list[CandidateFrame]) -> list[CandidateFrame]:
    by_id: dict[str, CandidateFrame] = {}
    for frame in frames:
        existing = by_id.get(frame.frame_id)
        if existing is None or frame.confidence > existing.confidence:
            by_id[frame.frame_id] = frame
    return sorted(by_id.values(), key=lambda frame: (-frame.confidence, frame.frame_id))


def _slug_frame_id(value: str) -> str:
    clean = _TOKEN_RE.sub("_", value.strip().lower()).strip("_")
    return clean if clean.startswith(("event_", "frame_")) else f"frame_{clean}"


def _frame_type_from_role_hint(role_hint: str) -> str:
    hint = normalize_cue_key(role_hint)
    if "transform" in hint or "shift" in hint:
        return "transform"
    if "legend" in hint or "symbol" in hint:
        return "symbol_region"
    if "relation" in hint:
        return "relation"
    return "event"


def _is_verb_like(unit: CandidateUnit) -> bool:
    kind = normalize_cue_key(unit.kind_hint)
    if kind in {"verb", "verb_span"}:
        return True
    return normalize_cue_key(unit.surface) in _DEPOSIT_SURFACES | {
        "found",
        "discovered",
        "picked",
        "sold",
        "wrapped",
        "spotted",
    }
