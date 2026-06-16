from __future__ import annotations

from lucid.training.ingest_learning import (
    apply_contradiction_branches,
    cap_relations_by_facet,
    consolidate_trace_records,
    consolidate_vendor_artifact_concepts,
    evaluate_crosstalk,
    normalize_relation_target,
    relation_targets_conflict,
)
from lucid.training.scale_ingest import Article, extract_concepts_with_learning, run_ingest_learning_pipeline


def _article(source_id: str, sentences: list[str]) -> Article:
    return Article(
        source_id=source_id,
        title=f"Test {source_id}",
        url=f"https://example.test/{source_id}",
        text=" ".join(sentences),
        sentences=sentences,
    )


def test_relation_targets_conflict_detects_incompatible_type_of() -> None:
    assert relation_targets_conflict(
        "type_of",
        "a multidisciplinary field using quantum mechanics",
        "merely a theoretical curiosity with no practical use",
    )
    assert not relation_targets_conflict(
        "type_of",
        "a multidisciplinary field using quantum mechanics",
        "a multidisciplinary field that uses quantum mechanics",
    )


def test_facet_caps_keep_more_than_global_four() -> None:
    relations = []
    for index in range(6):
        relations.append(
            {
                "relation": "type_of",
                "target": f"definition claim number {index} with enough words to pass filters",
                "confidence": 0.9 - index * 0.01,
                "source_refs": ["article_a"],
            }
        )
    for index in range(4):
        relations.append(
            {
                "relation": "uses",
                "target": f"mechanism claim number {index} with enough words to pass filters",
                "confidence": 0.85 - index * 0.01,
                "source_refs": ["article_a"],
            }
        )
    capped = cap_relations_by_facet(relations, max_per_facet=8, max_total=24)
    assert len(capped) == 10


def test_crosstalk_splits_conflicting_article_readings() -> None:
    article_one = _article(
        "article_one",
        [
            "Quantum computing is a multidisciplinary field that uses quantum mechanics to solve complex problems.",
            "Quantum computing relies on qubits and superposition to perform useful computation.",
        ],
    )
    article_two = _article(
        "article_two",
        [
            "Quantum computing is merely a theoretical curiosity with no practical use in industry today.",
            "Quantum computing hardware remains too noisy for reliable results across most benchmarks.",
        ],
    )

    concepts, report = extract_concepts_with_learning([article_one, article_two])
    branched = [concept for concept in concepts if "__reading_" in str(concept.get("concept_id") or "")]
    assert branched, "expected a contradiction branch for conflicting readings"
    assert report.contradiction_splits >= 1
    assert evaluate_crosstalk(concepts, base_concept_id="quantum_computing")


def test_agreement_keeps_single_branch_for_shared_target() -> None:
    article_one = _article(
        "article_one",
        [
            "Quantum computing is a multidisciplinary field that uses quantum mechanics to solve complex problems.",
            "Quantum computing relies on qubits and superposition to perform useful computation.",
        ],
    )
    article_two = _article(
        "article_two",
        [
            "Quantum computing is a multidisciplinary field that uses quantum mechanics and qubits.",
            "Quantum computing hardware uses superposition and entanglement during computation.",
        ],
    )

    concepts, report = extract_concepts_with_learning([article_one, article_two])
    branches = [concept for concept in concepts if str(concept.get("concept_id") or "").startswith("quantum_computing")]
    assert len(branches) == 1
    assert report.contradiction_splits == 0


def test_consolidate_trace_records_deduplicates_claim_traces() -> None:
    traces = [
        {
            "trace_id": "t_term_quantum_computing",
            "trace_family": "quantum_computing",
            "alias": "quantum computing",
            "cue_affinities": {"quantum_computing": 0.95},
        },
        {
            "trace_id": "t_claim_quantum_computing_aaa",
            "trace_family": "quantum_computing",
            "alias": "quantum_computing_type_of",
            "cue_affinities": {
                "quantum_computing": 0.88,
                "type_of": 0.64,
                "multidisciplinary field": 0.56,
            },
            "source_refs": ["article_one"],
            "activation_count": 1,
            "trust_score": 0.82,
        },
        {
            "trace_id": "t_claim_quantum_computing_bbb",
            "trace_family": "quantum_computing",
            "alias": "quantum_computing_type_of",
            "cue_affinities": {
                "quantum_computing": 0.88,
                "type_of": 0.64,
                "multidisciplinary field": 0.56,
            },
            "source_refs": ["article_two"],
            "activation_count": 1,
            "trust_score": 0.84,
        },
    ]
    consolidated, deduped = consolidate_trace_records(traces)
    assert deduped == 1
    assert len(consolidated) == 2
    claim = next(row for row in consolidated if row["trace_id"].startswith("t_claim_"))
    assert claim["source_refs"] == ["article_one", "article_two"]


def test_apply_contradiction_branches_splits_manual_fixture() -> None:
    concept = {
        "concept_id": "quantum_computing",
        "terms": ["quantum computing"],
        "relations": [
            {
                "relation": "type_of",
                "target": "a multidisciplinary field using quantum mechanics",
                "confidence": 0.86,
                "source_refs": ["article_one"],
                "source_sentence": "Quantum computing is a multidisciplinary field using quantum mechanics.",
            },
            {
                "relation": "type_of",
                "target": "merely a theoretical curiosity with no practical use",
                "confidence": 0.84,
                "source_refs": ["article_two"],
                "source_sentence": "Quantum computing is merely a theoretical curiosity with no practical use.",
            },
        ],
        "source_refs": ["article_one", "article_two"],
    }
    branched, events = apply_contradiction_branches(
        [concept],
        branch_hash=lambda relation: "abc123",
    )
    assert len(branched) == 2
    assert events
    assert any("__reading_" in str(row.get("concept_id") or "") for row in branched)


def test_multi_property_relations_do_not_split_without_contradiction() -> None:
    concept = {
        "concept_id": "qubit",
        "terms": ["qubit"],
        "relations": [
            {
                "relation": "property",
                "target": "represented by quantum particles in superposition states",
                "confidence": 0.86,
                "source_refs": ["article_one"],
            },
            {
                "relation": "capability",
                "target": "be in a superposition of both the zero and one states at the same time",
                "confidence": 0.84,
                "source_refs": ["article_two"],
            },
            {
                "relation": "challenge",
                "target": "highly sensitive to external environments and stray particles of light",
                "confidence": 0.82,
                "source_refs": ["article_three"],
            },
        ],
        "source_refs": ["article_one", "article_two", "article_three"],
    }
    branched, events = apply_contradiction_branches(
        [concept],
        branch_hash=lambda relation: "abc123",
    )
    assert len(branched) == 1
    assert not events


def test_normalize_relation_target_remaps_gerund_type_of() -> None:
    relation, target = normalize_relation_target(
        "type_of",
        "exploring potential applications for cleaner fertilization",
    )
    assert relation == "capability"
    assert "exploring" in target


def test_consolidate_vendor_artifact_redirects_google_quantum() -> None:
    concepts = [
        {
            "concept_id": "google_quantum",
            "terms": ["google quantum"],
            "relations": [
                {
                    "relation": "capability",
                    "target": "exploring potential applications for cleaner fertilization",
                    "confidence": 0.82,
                    "source_refs": ["google_quantum_ai_intro"],
                    "source_sentence": "Google Quantum AI is exploring potential applications for cleaner fertilization.",
                }
            ],
            "source_refs": ["google_quantum_ai_intro"],
        },
        {
            "concept_id": "quantum_computer",
            "terms": ["quantum computer"],
            "relations": [
                {
                    "relation": "uses",
                    "target": "quantum physics to access different computational abilities than classical computers",
                    "confidence": 0.84,
                    "source_refs": ["google_quantum_ai_intro"],
                }
            ],
            "source_refs": ["google_quantum_ai_intro"],
        },
    ]
    cleaned = consolidate_vendor_artifact_concepts(concepts)
    ids = {concept["concept_id"] for concept in cleaned}
    assert "google_quantum" not in ids
    quantum_computer = next(row for row in cleaned if row["concept_id"] == "quantum_computer")
    relations = quantum_computer.get("relations") or []
    assert any(relation.get("relation") == "capability" for relation in relations)


def test_benign_not_phrases_do_not_force_type_of_split() -> None:
    assert not relation_targets_conflict(
        "type_of",
        "a multidisciplinary field using quantum mechanics to solve complex problems",
        "a wafer not much bigger than the silicon chips found in a laptop",
    )


def test_run_ingest_learning_pipeline_reports_coverage() -> None:
    articles = [
        _article(
            "article_one",
            [
                "Quantum computing is a multidisciplinary field that uses quantum mechanics to solve complex problems.",
                "Qubits are the unit of quantum information used by quantum computers.",
            ],
        )
    ]
    concepts, traces, report = run_ingest_learning_pipeline(articles)
    assert concepts
    assert traces
    assert report.sentence_audit.eligible == 2
    assert report.sentence_audit.extracted >= 1
