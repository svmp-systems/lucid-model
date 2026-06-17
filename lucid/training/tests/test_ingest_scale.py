from __future__ import annotations

import json
from pathlib import Path

from lucid.training.ingest_config import (
    IngestConfig,
    build_corpus_context,
    discover_corpus_terms,
    infer_source_entity,
    load_sources_from_path,
)
from lucid.training.scale_ingest import (
    Article,
    extract_concepts_with_learning,
    run_ingest_learning_pipeline,
    split_sentences,
)


def _article(source_id: str, sentences: list[str], *, title: str = "") -> Article:
    return Article(
        source_id=source_id,
        title=title or f"Test {source_id}",
        url=f"https://example.test/{source_id}",
        text=" ".join(sentences),
        sentences=sentences,
    )


def test_cross_domain_biology_sentences_are_not_gated() -> None:
    biology = (
        "Photosynthesis is a biochemical process that uses sunlight to convert carbon dioxide into glucose."
    )
    cfg = IngestConfig.scale_default()
    rows = split_sentences(biology, config=cfg)
    assert rows == [biology]


def test_cross_domain_biology_extracts_concepts() -> None:
    articles = [
        _article(
            "biology_intro",
            [
                "Photosynthesis is a biochemical process that uses sunlight to convert carbon dioxide into glucose.",
                "Chloroplasts are organelles that enable photosynthesis in plant cells through light reactions.",
            ],
            title="Intro Biology",
        ),
        _article(
            "biology_followup",
            [
                "Photosynthesis relies on chlorophyll pigments to capture light energy from the sun.",
                "Mitochondria are organelles that produce energy for cells through cellular respiration.",
            ],
            title="Followup Biology",
        ),
    ]
    concepts, report = extract_concepts_with_learning(articles, config=IngestConfig.scale_default())
    concept_ids = {str(concept["concept_id"]) for concept in concepts}
    assert "photosynthesis" in concept_ids
    assert report.sentence_audit.extracted >= 2


def test_cross_domain_mixed_corpus_builds_corpus_terms() -> None:
    articles = [
        _article(
            "quantum",
            [
                "Quantum computing is a multidisciplinary field that uses quantum mechanics to solve complex problems.",
            ],
        ),
        _article(
            "biology",
            [
                "Photosynthesis is a biochemical process that uses sunlight to convert carbon dioxide into glucose.",
            ],
        ),
    ]
    ctx = build_corpus_context(articles, config=IngestConfig.scale_default())
    assert "photosynthesis" in ctx.corpus_terms
    assert "mechanics" in ctx.corpus_terms


def test_load_sources_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "sources.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_id": "wiki_photosynthesis",
                        "title": "Wikipedia: Photosynthesis",
                        "url": "https://example.test/photosynthesis",
                    }
                ),
                json.dumps(
                    {
                        "source_id": "wiki_mitochondria",
                        "title": "Wikipedia: Mitochondria",
                        "url": "https://example.test/mitochondria",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    rows = load_sources_from_path(path)
    assert len(rows) == 2
    assert rows[0]["source_id"] == "wiki_photosynthesis"


def test_infer_source_entity_from_title() -> None:
    assert infer_source_entity({"title": "IBM: What Is Quantum Computing?"}) == "IBM"
    assert infer_source_entity({"title": "Nature", "url": "https://www.nature.com/articles/example"}) == "Nature"


def test_scale_pipeline_reports_corpus_terms() -> None:
    articles = [
        _article(
            "article_one",
            [
                "Quantum computing is a multidisciplinary field that uses quantum mechanics to solve complex problems.",
                "Photosynthesis is a biochemical process that uses sunlight to convert carbon dioxide into glucose.",
            ],
        )
    ]
    _concepts, _traces, report, ctx = run_ingest_learning_pipeline(
        articles,
        config=IngestConfig(max_candidate_terms=500),
    )
    assert len(ctx.corpus_terms) >= 5
    assert report.sentence_audit.eligible == 2


def test_scale_pipeline_filters_junk_concepts() -> None:
    articles = [
        _article(
            "article_one",
            [
                "Quantum computing is a multidisciplinary field that uses quantum mechanics to solve complex problems.",
                "Based on recent advances, researchers continue exploring quantum hardware architectures.",
            ],
        )
    ]
    concepts, report = extract_concepts_with_learning(articles, config=IngestConfig.scale_default())
    concept_ids = {str(concept["concept_id"]) for concept in concepts}
    assert "quantum_computing" in concept_ids
    assert "based" not in concept_ids
    assert report.concepts_after_quality_filter == len(concepts)
    assert report.concepts_before_quality_filter >= report.concepts_after_quality_filter


def test_discover_corpus_terms_promotes_repeated_terms() -> None:
    sentences = [
        "Photosynthesis is a biochemical process that uses sunlight to convert carbon dioxide into glucose.",
        "Photosynthesis relies on chlorophyll pigments to capture light energy from the sun.",
    ]
    terms = discover_corpus_terms(sentences, config=IngestConfig.scale_default())
    assert "photosynthesis" in terms
