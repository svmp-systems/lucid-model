from __future__ import annotations

from lucid.training.ingest_quality import reject_concept_retention, retain_concepts


def test_reject_concept_retention_drops_branches_and_discourse() -> None:
    assert reject_concept_retention({"concept_id": "attention__reading_abc", "relations": [{}]}) == "contradiction_branch"
    assert reject_concept_retention({"concept_id": "although", "relations": [{"relation": "type_of"}]}) == "discourse_subject"
    assert reject_concept_retention({"concept_id": "accurate", "relations": [{"relation": "type_of"}], "source_refs": ["a"]}) == "discourse_subject"


def test_reject_concept_retention_keeps_supported_domain_terms() -> None:
    assert (
        reject_concept_retention(
            {
                "concept_id": "transformer",
                "relations": [{}, {}, {}],
                "source_refs": ["a", "b"],
            }
        )
        is None
    )
    assert (
        reject_concept_retention(
            {
                "concept_id": "boltzmann_machine",
                "relations": [{}, {}],
                "source_refs": ["a"],
            }
        )
        is None
    )


def test_retain_concepts_summary() -> None:
    concepts = [
        {"concept_id": "although", "relations": [{}]},
        {"concept_id": "attention_mechanism", "relations": [{}, {}], "source_refs": ["a"]},
        {"concept_id": "hopfield_network", "relations": [{}, {}, {}], "source_refs": ["a", "b"]},
    ]
    kept, stats = retain_concepts(concepts)
    assert [concept["concept_id"] for concept in kept] == ["attention_mechanism", "hopfield_network"]
    assert stats["concepts_before_retention"] == 3
    assert stats["concepts_after_retention"] == 2
