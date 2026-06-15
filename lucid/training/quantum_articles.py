"""Train a small source-backed quantum computing concept checkpoint.

This importer stores paraphrased concept facts with source refs. It does not
copy article bodies into the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from lucid.runtime.paths import DEFAULT_TRAINING_CHECKPOINT, resolve_checkpoint
from lucid.training.checkpoint.metadata import (
    apply_runtime_promotion_fields,
    ensure_metadata,
    record_support,
    source_backed_shadow_promotion,
)
from lucid.training.checkpoint.store import CheckpointState, load_checkpoint, save_checkpoint

_TOKEN_RE = re.compile(r"[^a-z0-9_]+")

ARTICLE_SOURCES = [
    {
        "source_id": "ibm_quantum_computing",
        "title": "IBM: What Is Quantum Computing?",
        "url": "https://www.ibm.com/think/topics/quantum-computing",
    },
    {
        "source_id": "microsoft_quantum_overview",
        "title": "Microsoft Learn: What Is Quantum Computing?",
        "url": "https://learn.microsoft.com/en-us/azure/quantum/overview-understanding-quantum-computing",
    },
    {
        "source_id": "nist_quantum_explained",
        "title": "NIST: Quantum Computing Explained",
        "url": "https://www.nist.gov/quantum-information-science/quantum-computing-explained",
    },
]

CONCEPT_FACTS = [
    {
        "concept_id": "quantum_computing",
        "terms": ["quantum computing", "quantum computer", "quantum computers"],
        "relations": [
            {
                "relation": "uses",
                "target": "quantum mechanics",
                "confidence": 0.9,
                "source_refs": ["ibm_quantum_computing", "microsoft_quantum_overview"],
            },
            {
                "relation": "uses",
                "target": "qubits",
                "confidence": 0.9,
                "source_refs": ["microsoft_quantum_overview"],
            },
            {
                "relation": "uses",
                "target": "quantum entanglement and quantum interference",
                "confidence": 0.82,
                "source_refs": ["microsoft_quantum_overview"],
            },
        ],
    },
    {
        "concept_id": "qubit",
        "terms": ["qubit", "qubits", "quantum bit", "quantum bits"],
        "relations": [
            {
                "relation": "type_of",
                "target": "unit of quantum information",
                "confidence": 0.92,
                "source_refs": ["ibm_quantum_computing", "microsoft_quantum_overview"],
            },
            {
                "relation": "property",
                "target": "can be prepared in superposition",
                "confidence": 0.86,
                "source_refs": ["ibm_quantum_computing", "nist_quantum_explained"],
            },
            {
                "relation": "challenge",
                "target": "noise and environmental disturbance",
                "confidence": 0.84,
                "source_refs": ["nist_quantum_explained"],
            },
        ],
    },
    {
        "concept_id": "superposition",
        "terms": ["superposition"],
        "relations": [
            {
                "relation": "type_of",
                "target": "quantum principle",
                "confidence": 0.88,
                "source_refs": ["ibm_quantum_computing", "nist_quantum_explained"],
            },
            {
                "relation": "property",
                "target": "can be disrupted by measurement or disturbance",
                "confidence": 0.78,
                "source_refs": ["nist_quantum_explained"],
            },
        ],
    },
    {
        "concept_id": "entanglement",
        "terms": ["entanglement", "quantum entanglement"],
        "relations": [
            {
                "relation": "type_of",
                "target": "quantum principle",
                "confidence": 0.88,
                "source_refs": ["ibm_quantum_computing", "microsoft_quantum_overview"],
            },
            {
                "relation": "uses",
                "target": "linked quantum systems",
                "confidence": 0.78,
                "source_refs": ["microsoft_quantum_overview"],
            },
        ],
    },
    {
        "concept_id": "decoherence",
        "terms": ["decoherence"],
        "relations": [
            {
                "relation": "challenge",
                "target": "loss of useful quantum behavior through disturbance",
                "confidence": 0.82,
                "source_refs": ["ibm_quantum_computing", "nist_quantum_explained"],
            }
        ],
    },
]

RELATION_ALIASES = [
    {"alias_id": "alias_qubit", "surface_pattern": "qubit", "relation_candidates": ["concept"], "confidence": 0.8},
    {"alias_id": "alias_qubits", "surface_pattern": "qubits", "relation_candidates": ["concept"], "confidence": 0.8},
    {"alias_id": "alias_superposition", "surface_pattern": "superposition", "relation_candidates": ["property"], "confidence": 0.75},
    {"alias_id": "alias_entanglement", "surface_pattern": "entanglement", "relation_candidates": ["property"], "confidence": 0.75},
    {"alias_id": "alias_decoherence", "surface_pattern": "decoherence", "relation_candidates": ["challenge"], "confidence": 0.75},
]

BOOTSTRAP_OPERATORS = [
    {
        "operator_id": "concept_relations_support_definition",
        "family": "concept",
        "pattern": [
            ["relation", "type_of", "X", "Y"],
            ["relation", "property", "X", "Z"],
        ],
        "effects": [["supports", "definition_support", "X", "Y"]],
        "default_confidence": 0.72,
    },
    {
        "operator_id": "motion_propagates_through_coupling",
        "family": "physical",
        "pattern": [
            ["relation", "coupled", "X", "Y"],
            ["event", "moves", "Y", "L"],
        ],
        "effects": [["state", "at", "X", "L"]],
        "default_confidence": 0.75,
    },
    {
        "operator_id": "greater_than_transitive",
        "family": "constraint",
        "pattern": [
            ["constraint", "greater_than", "A", "B"],
            ["constraint", "greater_than", "B", "C"],
        ],
        "effects": [["constraint", "greater_than", "A", "C"]],
        "default_confidence": 0.9,
    },
]


def _upsert_by_key(
    rows: list[dict[str, Any]],
    key: str,
    record: dict[str, Any],
    *,
    merge_lists: bool = True,
) -> None:
    value = record.get(key)
    for index, row in enumerate(rows):
        if row.get(key) == value:
            merged = dict(row)
            for field, item in record.items():
                if isinstance(item, list) and merge_lists:
                    seen = {
                        json.dumps(existing, sort_keys=True, default=str)
                        for existing in merged.get(field, [])
                    }
                    merged.setdefault(field, [])
                    for candidate in item:
                        token = json.dumps(candidate, sort_keys=True, default=str)
                        if token not in seen:
                            merged[field].append(candidate)
                            seen.add(token)
                else:
                    merged[field] = item
            rows[index] = merged
            return
    rows.append(dict(record))


def _normalize_key(value: object) -> str:
    clean = _TOKEN_RE.sub("_", str(value or "").strip().lower())
    return "_".join(part for part in clean.split("_") if part)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        rows.append(item)
        seen.add(item)
    return rows


def _cue_keys(value: object) -> list[str]:
    key = _normalize_key(value)
    if not key:
        return []
    parts = [part for part in key.split("_") if len(part) > 2]
    return _unique([key, *parts])


def _concept_source_refs(concept: dict[str, Any]) -> list[str]:
    return _unique(
        [
            str(source_ref)
            for relation in concept.get("relations", [])
            for source_ref in relation.get("source_refs", [])
            if str(source_ref)
        ]
    )


def _confidence_average(concept: dict[str, Any]) -> float:
    values = [
        float(relation.get("confidence", 0.0) or 0.0)
        for relation in concept.get("relations", [])
        if isinstance(relation, dict)
    ]
    if not values:
        return 0.5
    return round(sum(values) / len(values), 4)


def _weighted_signature(pairs: list[tuple[str, float]]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for raw, weight in pairs:
        keys = _cue_keys(raw)
        for index, key in enumerate(keys):
            scale = 1.0 if index == 0 else 0.35
            weights[key] = max(weights.get(key, 0.0), round(float(weight) * scale, 4))
    return dict(sorted(weights.items()))


def _relation_handle(concept_id: str, index: int, relation: dict[str, Any]) -> str:
    return (
        f"relation:{concept_id}:{index}:"
        f"{_normalize_key(relation.get('relation'))}:{_normalize_key(relation.get('target'))}"
    )


def _claim_handle(concept_id: str, index: int) -> str:
    return f"claim:{concept_id}:{index}"


def _target_concept_ids(target: str, concepts: list[dict[str, Any]]) -> list[str]:
    target_key = _normalize_key(target)
    matches: list[str] = []
    for concept in concepts:
        concept_id = str(concept.get("concept_id") or "")
        keys = _unique(
            [
                _normalize_key(concept_id),
                *[
                    _normalize_key(term)
                    for term in concept.get("terms", [])
                    if str(term).strip()
                ],
            ]
        )
        if any(key and (key in target_key or target_key in key) for key in keys):
            matches.append(concept_id)
    return matches


def _trace_record_for_concept(concept: dict[str, Any]) -> dict[str, Any]:
    concept_id = str(concept["concept_id"])
    source_refs = _concept_source_refs(concept)
    cue_pairs: list[tuple[str, float]] = [
        (concept_id, 0.95),
        (f"t_{concept_id}", 0.9),
        *[(term, 0.88) for term in concept.get("terms", [])],
    ]
    for relation in concept.get("relations", []):
        cue_pairs.append((relation.get("target", ""), 0.42))
    return {
        "trace_id": f"t_{concept_id}",
        "trace_family": concept_id,
        "alias": concept_id,
        "cue_affinities": _weighted_signature(cue_pairs),
        "cluster_id": concept_id,
        "heat_tier": "quarantine",
        "maturity_state": "provisional",
        "activation_bias": 0.04,
        "activation_count": 1,
        "success_count": 0,
        "failure_count": 0,
        "created_from_cues": _unique([concept_id, *concept.get("terms", [])]),
        "created_from_examples": source_refs,
        "source_refs": source_refs,
        "description": f"Source-backed trace for {concept_id}",
        "last_update_summary": "quantum_article_import",
    }


def _basin_record_for_concept(
    concept: dict[str, Any],
    concepts: list[dict[str, Any]],
) -> dict[str, Any]:
    concept_id = str(concept["concept_id"])
    source_refs = _concept_source_refs(concept)
    relations = [
        {
            "relation": str(relation.get("relation") or ""),
            "target": str(relation.get("target") or ""),
            "confidence": float(relation.get("confidence", 0.0) or 0.0),
            "source_refs": [
                str(source_ref)
                for source_ref in relation.get("source_refs", [])
                if str(source_ref)
            ],
        }
        for relation in concept.get("relations", [])
        if isinstance(relation, dict)
    ]
    relation_handles = [
        _relation_handle(concept_id, index, relation)
        for index, relation in enumerate(relations)
    ]
    evidence_handles = [
        f"concept:{concept_id}",
        *[_claim_handle(concept_id, index) for index, _ in enumerate(relations)],
    ]
    cooperation_links: dict[str, float] = {}
    for relation in relations:
        for target_concept_id in _target_concept_ids(relation.get("target", ""), concepts):
            if target_concept_id == concept_id:
                continue
            cooperation_links[f"b_{target_concept_id}"] = max(
                cooperation_links.get(f"b_{target_concept_id}", 0.0),
                round(0.35 + float(relation.get("confidence", 0.0)) * 0.35, 4),
            )
    activation_pairs: list[tuple[str, float]] = [
        (concept_id, 0.95),
        (f"t_{concept_id}", 0.95),
        *[(term, 0.88) for term in concept.get("terms", [])],
    ]
    semantic_pairs: list[tuple[str, float]] = [
        (concept_id, 0.95),
        *[(term, 0.75) for term in concept.get("terms", [])],
    ]
    for relation in relations:
        activation_pairs.append((relation.get("target", ""), 0.38))
        semantic_pairs.append((relation.get("relation", ""), 0.5))
        semantic_pairs.append((relation.get("target", ""), 0.45))
    return {
        "basin_id": f"b_{concept_id}",
        "family_hint": concept_id,
        "frame_affinities": {
            "frame_active": 0.72,
            "concept": 0.62,
            "event": 0.38,
        },
        "activation_signature": _weighted_signature(activation_pairs),
        "semantic_signature": _weighted_signature(semantic_pairs),
        "evidence_handles": evidence_handles,
        "relation_handles": relation_handles,
        "source_refs": source_refs,
        "trust_score": _confidence_average(concept),
        "heat_tier": "quarantine",
        "cooperation_links": dict(sorted(cooperation_links.items())),
        "suppression_links": {},
        "support_examples": source_refs,
        "quantized_payload": {
            "precision": "uint8_sparse",
            "canonical_label": concept_id,
            "concept_id": concept_id,
            "terms": [_normalize_key(term) for term in concept.get("terms", [])],
            "relations": relations,
            "source_count": len(source_refs),
        },
    }


def train_quantum_articles(
    checkpoint: str | Path = DEFAULT_TRAINING_CHECKPOINT,
) -> dict[str, Any]:
    root = resolve_checkpoint(checkpoint)
    state: CheckpointState = load_checkpoint(root, create=True)

    source_store = state.ensure_store("concept_bank").setdefault("sources", [])
    concept_store = state.ensure_store("concept_bank").setdefault("concepts", [])
    alias_store = state.ensure_store("relation_aliases").setdefault("aliases", [])
    operator_store = state.ensure_store("operator_bank").setdefault("operators", [])
    trace_store = state.ensure_store("tracebank").setdefault("records", [])
    basin_store = state.ensure_store("basin_bank").setdefault("records", [])

    for source in ARTICLE_SOURCES:
        _upsert_by_key(source_store, "source_id", source)
        ensure_metadata(
            state,
            f"source:{source['source_id']}",
            "source",
            source="quantum_article_import",
            source_refs=[{"ref_id": source["url"], "title": source["title"]}],
        )

    for concept in CONCEPT_FACTS:
        source_refs = _concept_source_refs(concept)
        metadata = source_backed_shadow_promotion(
            state,
            f"concept:{concept['concept_id']}",
            "concept",
            source="quantum_article_import",
            source_refs=[
                {"ref_id": source_ref} for source_ref in source_refs
            ],
            support_count=len(concept.get("relations", [])),
            trust_score=_confidence_average(concept),
        )
        concept_record = dict(concept)
        concept_record["heat_tier"] = metadata["heat_tier"]
        concept_record["commit_permission"] = metadata["commit_permission"]
        _upsert_by_key(concept_store, "concept_id", concept_record, merge_lists=False)

        trace = _trace_record_for_concept(concept)
        trace_metadata = source_backed_shadow_promotion(
            state,
            f"trace:{trace['trace_id']}",
            "trace",
            source="quantum_article_import",
            precision_tier="uint8_sparse",
            source_refs=[{"ref_id": source_ref} for source_ref in trace["source_refs"]],
            support_count=len(trace.get("source_refs", [])),
            trust_score=_confidence_average(concept),
        )
        trace_metadata["quantization_candidate"] = True
        apply_runtime_promotion_fields(trace, trace_metadata, has_maturity=True)
        _upsert_by_key(trace_store, "trace_id", trace, merge_lists=False)

        basin = _basin_record_for_concept(concept, CONCEPT_FACTS)
        basin_metadata = source_backed_shadow_promotion(
            state,
            f"basin:{basin['basin_id']}",
            "basin",
            source="quantum_article_import",
            precision_tier="uint8_sparse",
            source_refs=[{"ref_id": source_ref} for source_ref in basin["source_refs"]],
            support_count=len(basin.get("relation_handles", [])),
            trust_score=float(basin.get("trust_score", 0.0) or 0.0),
        )
        basin_metadata["quantization_candidate"] = True
        apply_runtime_promotion_fields(basin, basin_metadata)
        _upsert_by_key(basin_store, "basin_id", basin, merge_lists=False)

    for alias in RELATION_ALIASES:
        _upsert_by_key(alias_store, "alias_id", alias, merge_lists=False)
        record_support(state, f"alias:{alias['alias_id']}", "relation_alias")

    for operator in BOOTSTRAP_OPERATORS:
        _upsert_by_key(operator_store, "operator_id", operator, merge_lists=False)
        ensure_metadata(
            state,
            f"operator:{operator['operator_id']}",
            "operator",
            source="universal_bootstrap",
        )

    save_checkpoint(state, root, force=True, step_delta=1)
    return {
        "checkpoint": str(root),
        "sources": len(ARTICLE_SOURCES),
        "concepts": len(concept_store),
        "relation_aliases": len(alias_store),
        "operators": len(operator_store),
        "traces": len(trace_store),
        "basins": len(basin_store),
        "metadata_objects": len(state.ensure_store("learned_metadata").get("objects", {})),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train source-backed quantum concept checkpoint")
    parser.add_argument("--checkpoint", default=DEFAULT_TRAINING_CHECKPOINT)
    args = parser.parse_args(argv)
    print(json.dumps(train_quantum_articles(args.checkpoint), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
