"""Train a small source-backed quantum computing concept checkpoint.

This importer stores paraphrased concept facts with source refs. It does not
copy article bodies into the checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lucid.runtime.paths import DEFAULT_TRAINING_CHECKPOINT, resolve_checkpoint
from lucid.training.checkpoint.metadata import ensure_metadata, record_support
from lucid.training.checkpoint.store import CheckpointState, load_checkpoint, save_checkpoint

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


def train_quantum_articles(
    checkpoint: str | Path = DEFAULT_TRAINING_CHECKPOINT,
) -> dict[str, Any]:
    root = resolve_checkpoint(checkpoint)
    state: CheckpointState = load_checkpoint(root, create=True)

    source_store = state.ensure_store("concept_bank").setdefault("sources", [])
    concept_store = state.ensure_store("concept_bank").setdefault("concepts", [])
    alias_store = state.ensure_store("relation_aliases").setdefault("aliases", [])
    operator_store = state.ensure_store("operator_bank").setdefault("operators", [])

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
        _upsert_by_key(concept_store, "concept_id", concept, merge_lists=False)
        metadata = ensure_metadata(
            state,
            f"concept:{concept['concept_id']}",
            "concept",
            source="quantum_article_import",
            source_refs=[
                {"ref_id": source_ref}
                for relation in concept.get("relations", [])
                for source_ref in relation.get("source_refs", [])
            ],
        )
        metadata["support_count"] = max(int(metadata.get("support_count", 0)), len(concept.get("relations", [])))

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
