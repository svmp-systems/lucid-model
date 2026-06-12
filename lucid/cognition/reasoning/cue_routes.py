"""Evidence-level cue routing topology.

Perception and the cue encoder only index evidence to cue keys. This module
extracts that routing graph without interpreting senses or choosing winners.
"""

from __future__ import annotations

from lucid.cognition.input.cue.encoder import normalize_cue_key
from lucid.ir.cue import CueCloud


def cue_keys_per_unit(cue_cloud: CueCloud) -> dict[str, set[str]]:
    """Distinct cue keys routed to each evidence unit."""

    by_unit: dict[str, set[str]] = {}
    for req in cue_cloud.primitive_trace_activations:
        cue_key = normalize_cue_key(req.trace_id)
        if not cue_key:
            continue
        for ref in req.evidence_refs:
            if ref:
                by_unit.setdefault(ref, set()).add(cue_key)
    for req in cue_cloud.relational_trace_activations:
        cue_key = normalize_cue_key(req.trace_id)
        if not cue_key:
            continue
        for ref in (*req.relation_refs, *req.endpoint_unit_ids):
            if ref:
                by_unit.setdefault(ref, set()).add(cue_key)
    return by_unit


def evidence_cue_routes(cue_cloud: CueCloud) -> dict[str, list[tuple[str, float]]]:
    """Per-unit cue keys with their strongest routed weight."""

    weights: dict[str, dict[str, float]] = {}
    for req in cue_cloud.primitive_trace_activations:
        cue_key = normalize_cue_key(req.trace_id)
        if not cue_key:
            continue
        weight = float(req.weight)
        for ref in req.evidence_refs:
            if not ref:
                continue
            bucket = weights.setdefault(ref, {})
            bucket[cue_key] = max(bucket.get(cue_key, 0.0), weight)
    for req in cue_cloud.relational_trace_activations:
        cue_key = normalize_cue_key(req.trace_id)
        if not cue_key:
            continue
        weight = float(req.weight)
        for ref in (*req.relation_refs, *req.endpoint_unit_ids):
            if not ref:
                continue
            bucket = weights.setdefault(ref, {})
            bucket[cue_key] = max(bucket.get(cue_key, 0.0), weight)
    return {
        ref: sorted(bucket.items(), key=lambda item: (-item[1], item[0]))
        for ref, bucket in weights.items()
    }


def competing_units(
    cue_cloud: CueCloud | None,
    *,
    min_keys: int = 2,
) -> dict[str, set[str]]:
    """Units indexed by two or more distinct cue keys."""

    if cue_cloud is None:
        return {}
    routes = cue_keys_per_unit(cue_cloud)
    return {unit_id: keys for unit_id, keys in routes.items() if len(keys) >= min_keys}


def competing_cue_keys(cue_cloud: CueCloud | None) -> dict[str, list[str]]:
    """Sorted cue-key lists for units with plural routes."""

    return {unit_id: sorted(keys) for unit_id, keys in competing_units(cue_cloud).items()}


def has_competing_routes(
    cue_cloud: CueCloud | None,
    unit_ids: set[str],
) -> bool:
    """Whether any unit in ``unit_ids`` carries plural cue routes."""

    if cue_cloud is None or not unit_ids:
        return False
    competing = competing_units(cue_cloud)
    return any(unit_id in competing for unit_id in unit_ids)
