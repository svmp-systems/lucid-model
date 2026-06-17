"""Runtime cue encoder.

The cue encoder compiles perceptual evidence into sparse cue pressure for the
DMF. The cue names are addresses into trace cue affinities, not final meanings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lucid.ir.common import AmbiguityPolicy
from lucid.ir.cue import (
    CueCloud,
    CueEncoderInput,
    RelationalActivationRequest,
    TraceActivationRequest,
)
from lucid.ir.perception import (
    ArrangementHint,
    CandidateMarker,
    CandidateUnit,
    ChangeHint,
    PerceptualEvidenceGraph,
    ReferenceHint,
    UncertaintyFlag,
)
from lucid.runtime.paths import resolve_checkpoint
from lucid.training.checkpoint.slots import resolve_checkpoint_ref
from lucid.training.source_context import extract_session_concept_topics, parse_concept_query_with_context, resolve_concept_topic

_TOKEN_RE = re.compile(r"[^a-z0-9_]+")
_STOP_CUE_KEYS = frozenset(
    {
        "a",
        "an",
        "and",
        "be",
        "been",
        "being",
        "but",
        "i",
        "in",
        "is",
        "it",
        "of",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "which",
        "with",
    }
)
_STRUCTURAL_CUE_HINTS: dict[str, str] = {
    "a": "indefinite_reference",
    "an": "indefinite_reference",
    "and": "coordination",
    "after": "temporal_sequence",
    "before": "temporal_sequence",
    "but": "contrast",
    "he": "pronoun_coreference",
    "her": "pronoun_coreference",
    "him": "pronoun_coreference",
    "i": "deictic_speaker",
    "in": "locative_marker",
    "it": "pronoun_coreference",
    "of": "relation_marker",
    "she": "pronoun_coreference",
    "that": "reference_marker",
    "the": "definite_reference",
    "their": "pronoun_coreference",
    "them": "pronoun_coreference",
    "they": "pronoun_coreference",
    "this": "reference_marker",
    "to": "destination_marker",
    "while": "temporal_subordinate",
    "which": "relative_reference",
    "with": "association_marker",
    "you": "deictic_addressee",
}
_CONTEXTUAL_SUFFIX_TERMS = frozenset(
    {
        "algorithm",
        "bit",
        "circuit",
        "computer",
        "gate",
        "hardware",
        "mechanic",
        "particle",
        "processor",
        "state",
        "system",
    }
)


@dataclass(frozen=True, slots=True)
class CueEncoderConfig:
    checkpoint: str | Path | None = None
    cue_map: dict[str, Any] | None = None
    floor_threshold: float = 0.05
    learned_weight_multiplier: float = 0.9
    route_top_k: int = 4
    route_min_overlap: float = 0.34
    widen_min_overlap: float = 0.2
    coverage_widen_threshold: float = 0.55


@dataclass(frozen=True, slots=True)
class EvidenceFeature:
    feature_key: str
    cue_key: str
    weight: float
    evidence_refs: tuple[str, ...]
    kind: str = "primitive"
    relation_refs: tuple[str, ...] = ()
    endpoint_unit_ids: tuple[str, ...] = ()
    keep_alive: bool = True


def normalize_cue_key(value: str) -> str:
    clean = _TOKEN_RE.sub("_", value.strip().lower()).strip("_")
    return clean


def _singular_cue_key(value: str) -> str:
    key = normalize_cue_key(value)
    if key.endswith("ies") and len(key) > 4:
        return key[:-3] + "y"
    if key.endswith("s") and not key.endswith("ss") and len(key) > 3:
        return key[:-1]
    return key


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _base_weight(confidence: float = 0.0, salience: float = 0.0, fallback: float = 0.62) -> float:
    if confidence <= 0 and salience <= 0:
        return fallback
    return _clamp(0.25 + 0.55 * max(0.0, confidence) + 0.20 * max(0.0, salience))


def _load_cue_map(checkpoint: str | Path | None) -> dict[str, Any]:
    if not checkpoint:
        return {}
    raw_path = Path(checkpoint)
    if raw_path.is_file():
        path = raw_path
    else:
        root = resolve_checkpoint(resolve_checkpoint_ref(checkpoint))
        path = root / "cue_encoder_map.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_cue_map(config: CueEncoderConfig) -> dict[str, Any]:
    if config.cue_map is not None:
        return config.cue_map
    return _load_cue_map(config.checkpoint)


def feature_bitset(features: list[EvidenceFeature]) -> frozenset[str]:
    return frozenset(feature.feature_key for feature in features if feature.feature_key)


def _route_pattern(entry: dict[str, Any], anchor_key: str) -> frozenset[str]:
    raw = entry.get("feature_pattern")
    if isinstance(raw, list) and raw:
        return frozenset(str(key) for key in raw if key)
    return frozenset([anchor_key]) if anchor_key else frozenset()


def iter_promoted_routes(
    cue_map: dict[str, Any],
) -> list[tuple[str, str, frozenset[str], dict[str, Any]]]:
    routes: list[tuple[str, str, frozenset[str], dict[str, Any]]] = []
    for index_name in ("feature_index", "relation_index"):
        index = cue_map.get(index_name) or {}
        if not isinstance(index, dict):
            continue
        for anchor_key, entries in index.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict):
                    routes.append((index_name, anchor_key, _route_pattern(entry, anchor_key), entry))
    return routes


def rank_similar_routes(
    query_bits: frozenset[str],
    cue_map: dict[str, Any],
    *,
    top_k: int,
    min_overlap: float,
    exclude: set[tuple[str, str]] | None = None,
) -> list[tuple[float, str, frozenset[str], dict[str, Any]]]:
    excluded = exclude or set()
    ranked: list[tuple[float, str, frozenset[str], dict[str, Any]]] = []
    for index_name, _anchor_key, pattern, entry in iter_promoted_routes(cue_map):
        cue_key = str(entry.get("cue_key") or entry.get("trace_family") or "")
        if not cue_key or not pattern:
            continue
        identity = (index_name, cue_key)
        if identity in excluded:
            continue
        shared = query_bits & pattern
        if not shared:
            continue
        overlap = len(shared) / max(len(pattern), 1)
        if overlap < min_overlap:
            continue
        score = overlap * float(entry.get("weight", 1.0))
        ranked.append((score, index_name, pattern, entry))
    ranked.sort(key=lambda item: (-item[0], str(item[3].get("cue_key") or "")))
    return ranked[: max(0, top_k)]


def measure_cue_recall(cloud: CueCloud, gold_families: set[str]) -> dict[str, Any]:
    emitted = {request.trace_id for request in cloud.primitive_trace_activations}
    emitted.update(request.trace_id for request in cloud.relational_trace_activations)
    if not gold_families:
        return {"recall": 1.0, "missing": [], "emitted": sorted(emitted), "gold": []}
    missing = sorted(gold_families - emitted)
    recall = (len(gold_families) - len(missing)) / len(gold_families)
    return {
        "recall": recall,
        "missing": missing,
        "emitted": sorted(emitted),
        "gold": sorted(gold_families),
    }


def _unit_by_id(graph: PerceptualEvidenceGraph) -> dict[str, CandidateUnit]:
    return {unit.unit_id: unit for unit in graph.candidate_units}


def _target_has_uncertainty(flags: list[UncertaintyFlag], target_id: str) -> bool:
    return any(flag.target_id == target_id for flag in flags)


def _surface_features(
    unit: CandidateUnit,
    *,
    force_keep_alive: bool,
) -> list[EvidenceFeature]:
    surface = normalize_cue_key(unit.surface)
    if not surface:
        return []
    resolved = resolve_concept_topic(surface) or surface
    weight = _base_weight(unit.confidence, unit.salience)
    refs = (unit.unit_id,)
    if surface in _STOP_CUE_KEYS:
        features = [
            EvidenceFeature(
                feature_key=f"surface:{surface}",
                cue_key=surface,
                weight=max(0.1, weight * 0.35),
                evidence_refs=refs,
                kind="learned_only",
                keep_alive=False,
            )
        ]
        structural = _STRUCTURAL_CUE_HINTS.get(surface)
        if structural:
            features.append(
                EvidenceFeature(
                    feature_key=f"structure:{structural}",
                    cue_key=structural,
                    weight=max(0.1, weight * 0.45),
                    evidence_refs=refs,
                    kind="learned_only",
                    keep_alive=False,
                )
            )
        return features

    features = [
        EvidenceFeature(
            feature_key=f"surface:{surface}",
            cue_key=resolved,
            weight=weight,
            evidence_refs=refs,
            keep_alive=force_keep_alive or resolved != surface,
        )
    ]
    if resolved != surface:
        features.append(
            EvidenceFeature(
                feature_key=f"alias:{surface}->{resolved}",
                cue_key=resolved,
                weight=max(weight, 0.72),
                evidence_refs=refs,
                keep_alive=True,
            )
        )
    if unit.kind_hint:
        features.append(
            EvidenceFeature(
                feature_key=f"kind:{normalize_cue_key(unit.kind_hint)}",
                cue_key=resolved,
                weight=max(0.15, weight * 0.7),
                evidence_refs=refs,
                keep_alive=False,
            )
        )
    for hint in unit.type_hints:
        normalized = normalize_cue_key(hint)
        if normalized:
            features.append(
                EvidenceFeature(
                    feature_key=f"type:{normalized}",
                    cue_key=resolved,
                    weight=max(0.15, weight * 0.7),
                    evidence_refs=refs,
                    keep_alive=False,
                )
            )
    if unit.feature_signature:
        normalized = normalize_cue_key(unit.feature_signature)
        features.append(
            EvidenceFeature(
                feature_key=f"feature:{normalized}",
                cue_key=f"feature_{normalized}",
                weight=max(0.15, weight * 0.65),
                evidence_refs=refs,
                keep_alive=False,
            )
        )
    return features


def _marker_features(marker: CandidateMarker) -> list[EvidenceFeature]:
    surface = normalize_cue_key(marker.surface)
    if not surface:
        return []
    weight = _base_weight(marker.confidence, fallback=0.45)
    refs = (marker.marker_id,)
    features = [
        EvidenceFeature(
            feature_key=f"marker_surface:{surface}",
            cue_key=surface,
            weight=max(0.15, weight * 0.55),
            evidence_refs=refs,
            kind="learned_only",
            keep_alive=False,
        )
    ]
    structural = _STRUCTURAL_CUE_HINTS.get(surface)
    if structural:
        features.append(
            EvidenceFeature(
                feature_key=f"marker_structure:{structural}",
                cue_key=structural,
                weight=max(0.15, weight * 0.6),
                evidence_refs=refs,
                kind="learned_only",
                keep_alive=False,
            )
        )
    for hint in marker.marker_type_hints:
        normalized = normalize_cue_key(hint)
        if normalized:
            features.append(
                EvidenceFeature(
                    feature_key=f"marker_type:{normalized}",
                    cue_key=normalized,
                    weight=max(0.15, weight * 0.5),
                    evidence_refs=refs,
                    kind="learned_only",
                    keep_alive=False,
                )
            )
    return features


def _reference_features(hint: ReferenceHint) -> EvidenceFeature | None:
    cue_key = normalize_cue_key(hint.reference_type)
    if not cue_key:
        return None
    return EvidenceFeature(
        feature_key=f"reference:{cue_key}",
        cue_key=cue_key,
        weight=_clamp(hint.confidence or 0.5),
        evidence_refs=(hint.source_unit_id, hint.target_unit_id),
        kind="relation",
        relation_refs=(hint.reference_type,),
        endpoint_unit_ids=(hint.source_unit_id, hint.target_unit_id),
    )


def _arrangement_features(hint: ArrangementHint) -> EvidenceFeature | None:
    cue_key = normalize_cue_key(hint.hint_type)
    if not cue_key:
        return None
    return EvidenceFeature(
        feature_key=f"arrangement:{cue_key}",
        cue_key=cue_key,
        weight=_clamp(hint.weight or 0.5),
        evidence_refs=(hint.source_unit_id, hint.target_unit_id),
        kind="relation",
        relation_refs=(hint.hint_type,),
        endpoint_unit_ids=(hint.source_unit_id, hint.target_unit_id),
    )


def _change_features(
    hint: ChangeHint,
    units: dict[str, CandidateUnit],
) -> list[EvidenceFeature]:
    cue_base = normalize_cue_key(hint.change_type)
    if not cue_base:
        return []
    refs = tuple(ref for ref in (hint.before_unit_id, hint.after_unit_id) if ref)
    weight = _clamp(hint.weight or 0.6)
    features = [
        EvidenceFeature(
            feature_key=f"change:{cue_base}",
            cue_key=f"{cue_base}_like",
            weight=weight,
            evidence_refs=refs,
        )
    ]

    before = units.get(hint.before_unit_id)
    after = units.get(hint.after_unit_id)
    if before is None or after is None:
        return features
    if before.feature_signature and before.feature_signature == after.feature_signature:
        features.append(
            EvidenceFeature(
                feature_key="grid:color_preserved",
                cue_key="color_preserved_like",
                weight=max(0.1, weight * 0.9),
                evidence_refs=refs,
            )
        )
    if before.kind_hint and before.kind_hint == after.kind_hint:
        features.append(
            EvidenceFeature(
                feature_key="grid:shape_preserved",
                cue_key="shape_preserved_like",
                weight=max(0.1, weight * 0.85),
                evidence_refs=refs,
            )
        )
    return features


def evidence_features(graph: PerceptualEvidenceGraph) -> list[EvidenceFeature]:
    units = _unit_by_id(graph)
    features: list[EvidenceFeature] = []
    for unit in graph.candidate_units:
        force_keep_alive = _target_has_uncertainty(graph.uncertainty_flags, unit.unit_id)
        features.extend(_surface_features(unit, force_keep_alive=force_keep_alive))
    features.extend(_compound_unit_features(graph.candidate_units))
    for marker in graph.candidate_markers:
        features.extend(_marker_features(marker))
    for hint in graph.reference_hints:
        feature = _reference_features(hint)
        if feature is not None:
            features.append(feature)
    for hint in graph.arrangement_hints:
        feature = _arrangement_features(hint)
        if feature is not None:
            features.append(feature)
    for hint in graph.change_hints:
        features.extend(_change_features(hint, units))
    for flag in graph.uncertainty_flags:
        uncertainty = normalize_cue_key(flag.uncertainty_type)
        if not uncertainty:
            continue
        unit = units.get(flag.target_id)
        if unit is None:
            continue
        surface = normalize_cue_key(unit.surface)
        if not surface:
            continue
        weight = _base_weight(unit.confidence, unit.salience, fallback=0.5)
        refs = (unit.unit_id,)
        features.append(
            EvidenceFeature(
                feature_key=f"uncertainty:{uncertainty}:surface:{surface}",
                cue_key=surface,
                weight=max(0.2, weight * 0.85),
                evidence_refs=refs,
                kind="learned_only",
                keep_alive=True,
            )
        )
    return features


def _compound_unit_features(units: list[CandidateUnit]) -> list[EvidenceFeature]:
    ordered = sorted(units, key=lambda unit: (_unit_position(unit), unit.unit_id))
    features: list[EvidenceFeature] = []
    for width in (2, 3):
        for index in range(0, max(0, len(ordered) - width + 1)):
            group = ordered[index : index + width]
            keys = [normalize_cue_key(unit.surface) for unit in group]
            if any(not key for key in keys):
                continue
            if any(key in _STOP_CUE_KEYS for key in keys):
                continue
            phrase_key = "_".join(keys)
            if len(phrase_key) < 4:
                continue
            base = max(_base_weight(unit.confidence, unit.salience, fallback=0.62) for unit in group)
            refs = tuple(unit.unit_id for unit in group if unit.unit_id)
            features.append(
                EvidenceFeature(
                    feature_key=f"phrase:{phrase_key}",
                    cue_key=phrase_key,
                    weight=max(0.3, base * 0.96),
                    evidence_refs=refs,
                    keep_alive=any(
                        _target_has_uncertainty([], unit.unit_id) for unit in group
                    ),
                )
            )
    return features


def _unit_position(unit: CandidateUnit) -> tuple[int, str]:
    raw = str(unit.position_or_time or "")
    try:
        return (int(raw), raw)
    except ValueError:
        return (10_000, raw)


def _weak_structure_hints(graph: PerceptualEvidenceGraph) -> list[str]:
    hints: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        if value and value not in seen:
            hints.append(value)
            seen.add(value)

    if graph.candidate_units:
        add("span_like")
    for unit in graph.candidate_units:
        structural = _STRUCTURAL_CUE_HINTS.get(normalize_cue_key(unit.surface))
        if structural:
            add(f"structure:{structural}")
    for marker in graph.candidate_markers:
        structural = _STRUCTURAL_CUE_HINTS.get(normalize_cue_key(marker.surface))
        if structural:
            add(f"marker:{structural}")
    for region in graph.candidate_regions:
        role = normalize_cue_key(region.role_hint)
        if role:
            add(f"region:{role}")
    for flag in graph.uncertainty_flags:
        uncertainty = normalize_cue_key(flag.uncertainty_type)
        if uncertainty:
            add(f"uncertainty:{uncertainty}")
    for hint in graph.change_hints:
        change = normalize_cue_key(hint.change_type)
        if change:
            add(f"change:{change}")
    return hints


def _merge_primitive(
    merged: dict[str, dict[str, Any]],
    cue_key: str,
    weight: float,
    evidence_refs: tuple[str, ...],
    keep_alive: bool,
) -> None:
    cue_key = normalize_cue_key(cue_key)
    if not cue_key:
        return
    record = merged.setdefault(
        cue_key,
        {"weight": 0.0, "evidence_refs": set(), "keep_alive": False},
    )
    record["weight"] = max(float(record["weight"]), _clamp(weight))
    record["evidence_refs"].update(ref for ref in evidence_refs if ref)
    record["keep_alive"] = bool(record["keep_alive"] or keep_alive)


def _merge_relation(
    merged: dict[str, dict[str, Any]],
    cue_key: str,
    weight: float,
    relation_refs: tuple[str, ...],
    endpoint_unit_ids: tuple[str, ...],
) -> None:
    cue_key = normalize_cue_key(cue_key)
    if not cue_key:
        return
    record = merged.setdefault(
        cue_key,
        {"weight": 0.0, "relation_refs": set(), "endpoint_unit_ids": set()},
    )
    record["weight"] = max(float(record["weight"]), _clamp(weight))
    record["relation_refs"].update(ref for ref in relation_refs if ref)
    record["endpoint_unit_ids"].update(ref for ref in endpoint_unit_ids if ref)


def _learned_entries(cue_map: dict[str, Any], feature: EvidenceFeature) -> list[dict[str, Any]]:
    index_name = "relation_index" if feature.kind == "relation" else "feature_index"
    index = cue_map.get(index_name) or {}
    if not isinstance(index, dict):
        return []
    entries = index.get(feature.feature_key) or []
    return entries if isinstance(entries, list) else []


def _evidence_refs_for_pattern(
    features: list[EvidenceFeature],
    pattern_bits: frozenset[str],
) -> tuple[str, ...]:
    refs: set[str] = set()
    for feature in features:
        if feature.feature_key in pattern_bits:
            refs.update(ref for ref in feature.evidence_refs if ref)
    return tuple(sorted(refs))


def _apply_similar_routes(
    *,
    primitive: dict[str, dict[str, Any]],
    relation: dict[str, dict[str, Any]],
    features: list[EvidenceFeature],
    cue_map: dict[str, Any],
    config: CueEncoderConfig,
    min_overlap: float,
    top_k: int,
    exclude: set[tuple[str, str]],
) -> int:
    query_bits = feature_bitset(features)
    if not query_bits:
        return 0
    applied = 0
    for score, index_name, pattern, entry in rank_similar_routes(
        query_bits,
        cue_map,
        top_k=top_k,
        min_overlap=min_overlap,
        exclude=exclude,
    ):
        cue_key = str(
            entry.get("cue_key") or entry.get("trace_id") or entry.get("trace_family") or ""
        )
        if not cue_key:
            continue
        evidence_refs = _evidence_refs_for_pattern(features, query_bits & pattern)
        if not evidence_refs:
            continue
        weight = _clamp(score * config.learned_weight_multiplier)
        keep_alive = bool(entry.get("preserve_as_alternative", True))
        if index_name == "relation_index":
            _merge_relation(relation, cue_key, weight, (), evidence_refs)
        else:
            _merge_primitive(primitive, cue_key, weight, evidence_refs, keep_alive)
        exclude.add((index_name, cue_key))
        applied += 1
    return applied


def _estimate_feature_coverage(
    features: list[EvidenceFeature],
    primitive: dict[str, dict[str, Any]],
    relation: dict[str, dict[str, Any]],
) -> float:
    actionable = [feature for feature in features if feature.kind != "learned_only"]
    if not actionable:
        return 1.0
    covered = 0
    for feature in actionable:
        if feature.kind == "relation":
            if feature.cue_key in relation:
                covered += 1
        elif feature.cue_key in primitive:
            covered += 1
    return covered / len(actionable)


def _apply_learned_entries(
    *,
    primitive: dict[str, dict[str, Any]],
    relation: dict[str, dict[str, Any]],
    feature: EvidenceFeature,
    cue_map: dict[str, Any],
    multiplier: float,
    applied_exact: set[tuple[str, str]],
) -> None:
    index_name = "relation_index" if feature.kind == "relation" else "feature_index"
    for entry in _learned_entries(cue_map, feature):
        if not isinstance(entry, dict):
            continue
        cue_key = str(
            entry.get("cue_key")
            or entry.get("trace_id")
            or entry.get("trace_family")
            or ""
        )
        if not cue_key:
            continue
        applied_exact.add((index_name, cue_key))
        weight = feature.weight * float(entry.get("weight", 1.0)) * multiplier
        keep_alive = bool(entry.get("preserve_as_alternative", feature.keep_alive))
        if feature.kind == "relation":
            _merge_relation(
                relation,
                cue_key,
                weight,
                feature.relation_refs,
                feature.endpoint_unit_ids,
            )
        else:
            _merge_primitive(primitive, cue_key, weight, feature.evidence_refs, keep_alive)


def _select_primitive(
    merged: dict[str, dict[str, Any]],
    *,
    budget: int,
    floor_threshold: float,
) -> list[TraceActivationRequest]:
    records = [
        (cue_key, record)
        for cue_key, record in merged.items()
        if float(record["weight"]) >= floor_threshold
    ]
    records.sort(key=lambda item: (not bool(item[1]["keep_alive"]), -float(item[1]["weight"]), item[0]))
    selected = records[: max(0, budget)]
    return [
        TraceActivationRequest(
            trace_id=cue_key,
            weight=_clamp(float(record["weight"])),
            evidence_refs=sorted(record["evidence_refs"]),
            keep_alive=bool(record["keep_alive"]),
        )
        for cue_key, record in selected
    ]


def _select_relation(
    merged: dict[str, dict[str, Any]],
    *,
    budget: int,
    floor_threshold: float,
) -> list[RelationalActivationRequest]:
    records = [
        (cue_key, record)
        for cue_key, record in merged.items()
        if float(record["weight"]) >= floor_threshold
    ]
    records.sort(key=lambda item: (-float(item[1]["weight"]), item[0]))
    selected = records[: max(0, budget)]
    return [
        RelationalActivationRequest(
            trace_id=cue_key,
            weight=_clamp(float(record["weight"])),
            relation_refs=sorted(record["relation_refs"]),
            endpoint_unit_ids=sorted(record["endpoint_unit_ids"]),
        )
        for cue_key, record in selected
    ]


def _effective_policy(inp: CueEncoderInput, graph: PerceptualEvidenceGraph) -> AmbiguityPolicy:
    if inp.ambiguity_policy_in == AmbiguityPolicy.FORCE_WIDEN:
        return AmbiguityPolicy.FORCE_WIDEN
    if graph.uncertainty_flags:
        return AmbiguityPolicy.PRESERVE_PLURAL
    return inp.ambiguity_policy_in


def encode_cues(
    inp: CueEncoderInput,
    *,
    config: CueEncoderConfig | None = None,
) -> CueCloud:
    cfg = config or CueEncoderConfig()
    graph = inp.perceptual_evidence_graph
    learned_map_available = bool(cfg.cue_map) or bool(cfg.checkpoint)

    primitive: dict[str, dict[str, Any]] = {}
    relation: dict[str, dict[str, Any]] = {}
    features = evidence_features(graph)
    raw_text = graph.provenance.extra.get("raw_text")
    session_context = graph.provenance.extra.get("session_context")
    if isinstance(raw_text, str):
        parsed = parse_concept_query_with_context(
            raw_text.strip(),
            session_context if isinstance(session_context, dict) else None,
        )
        if parsed:
            _, concept_id, _ = parsed
            _merge_primitive(primitive, concept_id, 0.85, ("utterance",), True)
            _merge_primitive(primitive, "concept_query_like", 0.75, ("utterance",), False)

    for feature in features:
        if feature.kind == "relation":
            _merge_relation(
                relation,
                feature.cue_key,
                feature.weight,
                feature.relation_refs,
                feature.endpoint_unit_ids,
            )
        elif feature.kind != "learned_only":
            _merge_primitive(
                primitive,
                feature.cue_key,
                feature.weight,
                feature.evidence_refs,
                feature.keep_alive,
            )

    policy = _effective_policy(inp, graph)
    feature_coverage = _estimate_feature_coverage(features, primitive, relation)
    prior_dmf_coverage = inp.upstream_state.get("dmf_coverage_score")
    should_widen = policy == AmbiguityPolicy.FORCE_WIDEN
    if isinstance(prior_dmf_coverage, (int, float)) and float(prior_dmf_coverage) < cfg.coverage_widen_threshold:
        should_widen = True
    if feature_coverage < cfg.coverage_widen_threshold:
        should_widen = True

    budget = max(1, int(inp.retrieval_budget * inp.compute_policy.retrieval_budget_multiplier))
    if should_widen:
        budget = max(budget, int(budget * 1.5))

    relational_budget = min(len(relation), max(0, budget // 4))
    primitive_budget = max(1, budget - relational_budget)
    primitive_requests = _select_primitive(
        primitive,
        budget=primitive_budget,
        floor_threshold=cfg.floor_threshold,
    )
    relational_requests = _select_relation(
        relation,
        budget=relational_budget,
        floor_threshold=cfg.floor_threshold,
    )

    soft_priors: dict[str, float] = {}
    if inp.task_intent_hint:
        key = f"task:{normalize_cue_key(inp.task_intent_hint)}"
        soft_priors[key] = 0.12
    for key, weight in _session_context_priors(inp.upstream_state, primitive).items():
        soft_priors[key] = max(soft_priors.get(key, 0.0), weight)

    cloud = CueCloud(
        primitive_trace_activations=primitive_requests,
        relational_trace_activations=relational_requests,
        soft_context_priors=soft_priors,
        weak_structure_hints=_weak_structure_hints(graph),
        ambiguity_policy=policy,
        retrieval_budget_used=len(primitive_requests) + len(relational_requests) + len(soft_priors),
        suppression_list=[],
        provenance=inp.provenance,
    )
    cloud.provenance.extra["cue_encoder"] = {
        "mode": "evidence_compile",
        "learned_map_available": learned_map_available,
        "feature_count": len(features),
        "feature_coverage": round(feature_coverage, 4),
        "widen_applied": should_widen,
        "prior_dmf_coverage": prior_dmf_coverage,
        "primitive_candidate_count": len(primitive),
        "relational_candidate_count": len(relation),
    }
    return cloud


def _session_context_priors(
    upstream_state: dict[str, Any],
    primitive: dict[str, dict[str, Any]],
) -> dict[str, float]:
    session_context = upstream_state.get("session_context")
    if not isinstance(session_context, dict):
        return {}

    current_terms = {
        _singular_cue_key(key)
        for key in primitive
        if _singular_cue_key(key) in _CONTEXTUAL_SUFFIX_TERMS
    }
    if not current_terms:
        return {}

    recent_parts: list[str] = []
    for turn in session_context.get("recent_turns") or []:
        if not isinstance(turn, dict):
            continue
        recent_parts.append(str(turn.get("user_input") or ""))
        recent_parts.append(str(turn.get("assistant_output") or ""))
    for item in session_context.get("active_memories") or []:
        if isinstance(item, dict):
            recent_parts.append(str(item.get("key") or ""))
            recent_parts.append(str(item.get("value") or ""))
    recent_text = " ".join(part for part in recent_parts if part).lower()
    if not recent_text:
        return {}

    priors: dict[str, float] = {}
    for concept_id in extract_session_concept_topics(session_context):
        priors[concept_id] = max(priors.get(concept_id, 0.0), 0.42)
    context_keys = _context_topic_keys(recent_text)
    for topic_key in context_keys:
        suffix = _singular_cue_key(topic_key.rsplit("_", 1)[-1])
        if suffix in current_terms:
            priors[topic_key] = max(priors.get(topic_key, 0.0), 0.38)

    context_tokens = set(normalize_cue_key(recent_text).split("_"))
    if "quantum" in context_tokens:
        for term in current_terms:
            priors[f"quantum_{term}"] = max(priors.get(f"quantum_{term}", 0.0), 0.32)
    return priors


def _context_topic_keys(text: str) -> set[str]:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token and token not in _STOP_CUE_KEYS
    ]
    keys: set[str] = set()
    for index, token in enumerate(tokens[:-1]):
        if token != "quantum":
            continue
        suffix = _singular_cue_key(tokens[index + 1])
        if suffix in _CONTEXTUAL_SUFFIX_TERMS:
            keys.add(f"quantum_{suffix}")
    return keys
