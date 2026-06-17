from __future__ import annotations

from lucid.training.ingest_quality import (
    filter_concepts,
    is_valid_candidate_term,
    is_valid_subject_term,
    reject_concept_id,
    reject_relation,
)


def test_reject_fragment_subjects() -> None:
    assert reject_concept_id("based") == "fragment_subject"
    assert reject_concept_id("another") == "fragment_subject"
    assert reject_concept_id("best_quantum") == "vendor_artifact"
    assert reject_concept_id("decoherence_decoherence") == "repeated_token_parts"


def test_reject_junk_relation_targets() -> None:
    assert reject_relation(
        {"relation": "type_of", "target": "no longer supported"}
    ) == "junk_target_marker"
    assert reject_relation({"relation": "capability", "target": "run"}) == "target_too_short"
    assert reject_relation(
        {
            "relation": "type_of",
            "target": "a multidisciplinary field that uses quantum mechanics",
        }
    ) is None


def test_valid_core_concepts_pass() -> None:
    assert reject_concept_id("quantum_computing") is None
    assert reject_concept_id("photosynthesis") is None
    assert reject_concept_id("chloroplast") is None
    assert is_valid_candidate_term("quantum_computing")
    assert is_valid_subject_term("photosynthesis")


def test_filter_concepts_drops_junk_keeps_core() -> None:
    concepts = [
        {
            "concept_id": "quantum_computing",
            "terms": ["quantum computing"],
            "relations": [
                {
                    "relation": "type_of",
                    "target": "a multidisciplinary field that uses quantum mechanics to solve problems",
                    "source_refs": ["a"],
                }
            ],
            "source_refs": ["a"],
        },
        {
            "concept_id": "based",
            "terms": ["based"],
            "relations": [
                {
                    "relation": "type_of",
                    "target": "a device that takes input data and transforms it",
                    "source_refs": ["a"],
                }
            ],
            "source_refs": ["a"],
        },
        {
            "concept_id": "content",
            "terms": ["content"],
            "relations": [
                {
                    "relation": "type_of",
                    "target": "no longer supported",
                    "source_refs": ["a"],
                }
            ],
            "source_refs": ["a"],
        },
    ]
    kept, stats = filter_concepts(concepts, corpus_terms=frozenset({"quantum", "computing"}))
    ids = {row["concept_id"] for row in kept}
    assert ids == {"quantum_computing"}
    assert stats["concepts_after_quality_filter"] == 1
    assert stats["concept_rejections"].get("fragment_subject", 0) >= 1
