"""Configuration and corpus utilities for large-scale cross-domain source ingest."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")

# Seed vocabulary for the built-in quantum article pack (optional boost, not a gate).
QUANTUM_DOMAIN_SEED = {
    "algorithm",
    "algorithms",
    "annealing",
    "bit",
    "bits",
    "circuit",
    "circuits",
    "classical",
    "coherence",
    "computer",
    "computers",
    "computing",
    "decoherence",
    "entanglement",
    "error",
    "errors",
    "gate",
    "gates",
    "hardware",
    "information",
    "interference",
    "measurement",
    "mechanics",
    "noise",
    "processor",
    "processors",
    "quantum",
    "qubit",
    "qubits",
    "simulation",
    "superposition",
}

GENERAL_BROAD_SINGLE_TERMS = {
    "algorithm",
    "algorithms",
    "application",
    "applications",
    "approach",
    "approaches",
    "area",
    "areas",
    "case",
    "cases",
    "circuit",
    "circuits",
    "classical",
    "computer",
    "computers",
    "computing",
    "data",
    "development",
    "example",
    "examples",
    "form",
    "forms",
    "gate",
    "gates",
    "hardware",
    "information",
    "level",
    "method",
    "methods",
    "model",
    "models",
    "number",
    "part",
    "parts",
    "problem",
    "problems",
    "process",
    "processor",
    "processors",
    "quantum",
    "research",
    "result",
    "results",
    "state",
    "states",
    "study",
    "studies",
    "system",
    "systems",
    "technology",
    "type",
    "types",
    "use",
    "uses",
    "way",
    "ways",
    "work",
}

INGEST_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "may",
    "more",
    "most",
    "not",
    "of",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "same",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "use",
    "used",
    "using",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "will",
    "with",
    "you",
    "your",
}


@dataclass(slots=True)
class IngestConfig:
    """Tunable ingest profile for cross-domain scale runs."""

    cross_domain: bool = True
    max_candidate_terms: int = 2000
    max_relations_per_facet: int = 32
    max_relations_per_concept: int = 48
    min_sentence_length: int = 38
    max_sentence_length: int = 360
    min_term_count: int = 2
    min_term_doc_count: int = 1
    promote_long_unigram_len: int = 6
    corpus_term_limit: int = 400
    domain_terms_seed: frozenset[str] = frozenset()
    broad_single_terms: frozenset[str] = field(default_factory=lambda: frozenset(GENERAL_BROAD_SINGLE_TERMS))
    skip_article_errors: bool = True
    fetch_timeout: int = 30
    progress_logging: bool = True
    progress_interval: int = 5
    checkpoint_label: str = "scale cross-domain source ingest"

    @classmethod
    def scale_default(cls) -> IngestConfig:
        return cls(
            cross_domain=True,
            max_candidate_terms=2000,
            max_relations_per_facet=32,
            max_relations_per_concept=48,
            skip_article_errors=True,
            checkpoint_label="scale cross-domain source ingest",
        )

    @classmethod
    def quantum_default(cls) -> IngestConfig:
        return cls(
            cross_domain=False,
            max_candidate_terms=120,
            max_relations_per_facet=8,
            max_relations_per_concept=24,
            domain_terms_seed=frozenset(QUANTUM_DOMAIN_SEED),
            skip_article_errors=False,
            checkpoint_label="scale-style 5 quantum articles + basic language",
        )


@dataclass(slots=True)
class CorpusContext:
    config: IngestConfig
    corpus_terms: frozenset[str]
    source_entities: dict[str, str] = field(default_factory=dict)


def word_tokens(text: str) -> list[str]:
    return [token.lower().strip("-'") for token in WORD_RE.findall(text) if token.strip("-'")]


def singularize(term: str) -> str:
    if term.endswith("ies") and len(term) > 4:
        return term[:-3] + "y"
    if term.endswith(("sis", "ics", "ss", "us")):
        return term
    if term.endswith("s") and len(term) > 3:
        return term[:-1]
    return term


def infer_source_entity(source: dict[str, Any]) -> str:
    explicit = str(source.get("source_entity") or "").strip()
    if explicit:
        return explicit
    title = str(source.get("title") or "").strip()
    if ":" in title:
        return title.split(":", 1)[0].strip()
    url = str(source.get("url") or "").strip()
    if url:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        label = host.split(".")[0]
        if label:
            return label.replace("-", " ").title()
    source_id = str(source.get("source_id") or "").strip()
    if source_id:
        return source_id.replace("_", " ").title()
    return ""


def build_source_entity_map(sources: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for source in sources:
        source_id = str(source.get("source_id") or "").strip()
        if not source_id:
            continue
        entity = infer_source_entity(source)
        if entity:
            mapping[source_id] = entity
    return mapping


def load_sources_from_path(path: str | Path) -> list[dict[str, str]]:
    file_path = Path(path)
    raw = file_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"sources file is empty: {file_path}")
    if file_path.suffix.lower() == ".jsonl":
        rows: list[dict[str, str]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("each JSONL row must be an object")
            rows.append(_normalize_source_row(item))
        return rows
    payload = json.loads(raw)
    if isinstance(payload, list):
        return [_normalize_source_row(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("sources"), list):
        return [_normalize_source_row(item) for item in payload["sources"] if isinstance(item, dict)]
    raise ValueError(f"unsupported sources file format: {file_path}")


def _normalize_source_row(row: dict[str, Any]) -> dict[str, str]:
    source_id = str(row.get("source_id") or "").strip()
    url = str(row.get("url") or "").strip()
    title = str(row.get("title") or source_id or url).strip()
    if not source_id:
        source_id = _source_id_from_url(url) if url else title.lower().replace(" ", "_")
    if not url and not title:
        raise ValueError("each source needs at least a url or title")
    normalized: dict[str, str] = {
        "source_id": source_id,
        "title": title,
        "url": url,
    }
    entity = str(row.get("source_entity") or "").strip()
    if entity:
        normalized["source_entity"] = entity
    return normalized


def _source_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_") or "root"
    host = parsed.netloc.replace(".", "_").replace("-", "_")
    return f"{host}_{path}"[:96]


def discover_corpus_terms(
    sentences: list[str],
    *,
    config: IngestConfig,
    doc_count_hint: int = 1,
) -> frozenset[str]:
    """Derive domain vocabulary from sentence text (cross-domain, corpus-driven)."""

    unigram_counts: Counter[str] = Counter()
    bigram_counts: Counter[str] = Counter()
    doc_unigrams: Counter[str] = Counter()
    doc_bigrams: Counter[str] = Counter()

    for sentence in sentences:
        tokens = [
            singularize(token)
            for token in word_tokens(sentence)
            if len(token) > 2 and token not in INGEST_STOPWORDS
        ]
        seen_uni: set[str] = set()
        seen_bi: set[str] = set()
        for token in tokens:
            if len(token) >= 4 or token in config.domain_terms_seed:
                unigram_counts[token] += 1
                seen_uni.add(token)
        for index in range(max(0, len(tokens) - 1)):
            bigram = f"{tokens[index]} {tokens[index + 1]}"
            if len(bigram) >= 7:
                bigram_counts[bigram] += 1
                seen_bi.add(bigram)
        for token in seen_uni:
            doc_unigrams[token] += 1
        for bigram in seen_bi:
            doc_bigrams[bigram] += 1

    selected: set[str] = set(config.domain_terms_seed)
    min_count = max(1, config.min_term_count)
    min_docs = max(1, config.min_term_doc_count)

    for token, count in unigram_counts.items():
        if token in config.broad_single_terms:
            continue
        if count >= min_count or doc_unigrams[token] >= min_docs or len(token) >= config.promote_long_unigram_len:
            selected.add(token)

    for bigram, count in bigram_counts.items():
        if count >= min_count or doc_bigrams[bigram] >= min_docs:
            selected.add(bigram)

    ranked = sorted(
        selected,
        key=lambda term: (
            -(unigram_counts.get(term, 0) + bigram_counts.get(term, 0)),
            -(doc_unigrams.get(term, 0) + doc_bigrams.get(term, 0)),
            term,
        ),
    )
    if config.corpus_term_limit > 0:
        ranked = ranked[: config.corpus_term_limit]
    return frozenset(ranked)


def build_corpus_context(
    articles: list[Any],
    *,
    config: IngestConfig,
    sources: list[dict[str, Any]] | None = None,
) -> CorpusContext:
    sentences = [sentence for article in articles for sentence in article.sentences]
    corpus_terms = discover_corpus_terms(sentences, config=config, doc_count_hint=max(1, len(articles)))
    source_entities = build_source_entity_map(sources or [])
    for article in articles:
        source_id = str(getattr(article, "source_id", "") or "")
        if source_id and source_id not in source_entities:
            entity = infer_source_entity(
                {
                    "source_id": source_id,
                    "title": getattr(article, "title", ""),
                    "url": getattr(article, "url", ""),
                }
            )
            if entity:
                source_entities[source_id] = entity
    return CorpusContext(config=config, corpus_terms=corpus_terms, source_entities=source_entities)
