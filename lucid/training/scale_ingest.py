"""Scale-style source ingestion for article memory and basic language.

This module intentionally avoids hand-authored concept facts. The only
hand-selected data are source URLs and a small basic-language phrase corpus.
Concepts, aliases, traces, claims, and facet basins are derived from source text.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from lucid.runtime.paths import resolve_checkpoint, resolve_train_path
from lucid.training.checkpoint.metadata import (
    apply_runtime_promotion_fields,
    ensure_metadata,
    promote_operator_from_evidence,
    record_contradiction,
    record_support,
    source_backed_shadow_promotion,
)
from lucid.training.ingest_progress import ingest_log, should_log_progress
from lucid.training.checkpoint.registry import register_checkpoint
from lucid.training.checkpoint.slots import promote_to_loaded
from lucid.training.checkpoint.store import (
    CheckpointState,
    checkpoint_summary,
    empty_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from lucid.training.ingest_config import (
    IngestConfig,
    CorpusContext,
    QUANTUM_DOMAIN_SEED,
    build_corpus_context,
    build_source_entity_map,
    infer_source_entity,
    load_sources_from_path,
)
from lucid.training.ingest_learning import (
    DEFAULT_MAX_CANDIDATE_TERMS,
    DEFAULT_MAX_RELATIONS_PER_CONCEPT,
    DEFAULT_MAX_RELATIONS_PER_FACET,
    IngestLearningReport,
    SentenceAudit,
    apply_contradiction_branches,
    build_mechanism_relation_aliases,
    cap_relations_by_facet,
    consolidate_vendor_artifact_concepts,
    consolidate_trace_records,
    count_concept_heat_tiers,
    evaluate_crosstalk,
    format_ingest_audit_text,
    normalize_relation_target,
    write_ingest_audit_report,
)
from lucid.training.ingest_quality import (
    filter_concepts,
    is_valid_candidate_term,
    is_valid_subject_term,
    reject_relation,
)
from lucid.training.quantum_articles import BOOTSTRAP_OPERATORS
from lucid.training.source_context import (
    VENDOR_ARTIFACT_RE,
    clear_source_entities,
    register_source_entities,
    source_entity_for_article,
)

TOKEN_RE = re.compile(r"[^a-z0-9_]+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

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
    {
        "source_id": "aws_quantum_computing",
        "title": "AWS: What Is Quantum Computing?",
        "url": "https://aws.amazon.com/what-is/quantum-computing/",
    },
    {
        "source_id": "google_quantum_ai_intro",
        "title": "Google Quantum AI: What Is Quantum Computing?",
        "url": "https://quantumai.google/whatisqc",
    },
]

BASIC_LANGUAGE_PHRASES = [
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
    "how are you",
    "what can you do",
]

PARAPHRASE_QUERY_PREFIXES = [
    "what is",
    "what is a",
    "what are",
    "tell me about",
    "explain",
    "can you explain",
    "how does",
    "how do",
    "describe",
    "what does",
    "give me an overview of",
    "i want to know about",
]

STOPWORDS = {
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
SIGNATURE_PART_STOPWORDS = {
    "algorithm",
    "algorithms",
    "bit",
    "bits",
    "circuit",
    "circuits",
    "classical",
    "computer",
    "computers",
    "computing",
    "gate",
    "gates",
    "hardware",
    "mechanic",
    "mechanics",
    "particle",
    "particles",
    "processor",
    "processors",
    "quantum",
    "state",
    "states",
    "system",
    "systems",
}

DOMAIN_TERMS = set(QUANTUM_DOMAIN_SEED)

FACET_BY_RELATION = {
    "type_of": "definition",
    "property": "definition",
    "related_to": "definition",
    "uses": "mechanism",
    "enables": "mechanism",
    "capability": "mechanism",
    "measurement": "mechanism",
    "contrast": "contrast",
    "challenge": "challenge",
}

BAD_TARGET_STARTS = {
    "also",
    "analogous to",
    "best at",
    "both true",
    "combined",
    "created by",
    "designing",
    "encouraging",
    "expected to",
    "however",
    "much easier",
    "probably",
    "similar to",
    "some researchers",
    "worth little",
    "not ideal",
    "key to",
    "breaking new ground",
    "estimated to",
}
RHETORICAL_SENTENCE_MARKERS = {
    "as an analogy",
    "for example, if",
    "imagine that",
    "imagine you",
    "let's say",
    "similar to",
    "suppose that",
    "you can think of",
}
MARKETING_SENTENCE_MARKERS = {
    "our comprehensive",
    "our focused",
    "our roadmap",
    "our services",
    "our solution",
    "superpower",
}
WEB_CHROME_SENTENCE_MARKERS = {
    "open_in_new",
    "podcast",
    "play the",
    "read the report",
    "subscribe",
    "table of contents",
}
MODIFIER_FOLLOWERS = {
    "approach",
    "approaches",
    "chip",
    "chips",
    "component",
    "components",
    "device",
    "devices",
    "hardware",
    "processor",
    "processors",
    "system",
    "systems",
    "technology",
    "technologies",
    "type",
    "types",
}
BROAD_SINGLE_TERMS = {
    "algorithm",
    "algorithms",
    "circuit",
    "circuits",
    "classical",
    "computer",
    "computers",
    "computing",
    "gate",
    "gates",
    "hardware",
    "processor",
    "processors",
    "quantum",
    "system",
    "systems",
    "technology",
}
RELATION_PRIORITY = {
    "type_of": 0,
    "capability": 1,
    "property": 2,
    "uses": 3,
    "enables": 4,
    "measurement": 5,
    "challenge": 6,
    "contrast": 7,
    "related_to": 8,
}


@dataclass(slots=True)
class Article:
    source_id: str
    title: str
    url: str
    text: str
    sentences: list[str]


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignore_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        _ = attrs
        if tag.lower() in {"script", "style", "svg", "noscript"}:
            self._ignore_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "svg", "noscript"} and self._ignore_depth:
            self._ignore_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignore_depth:
            return
        text = " ".join(html.unescape(data).split())
        if text:
            self._chunks.append(text)

    def text(self) -> str:
        return " ".join(self._chunks)


def normalize_key(value: object) -> str:
    clean = TOKEN_RE.sub("_", str(value or "").strip().lower())
    return "_".join(part for part in clean.split("_") if part)


def humanize_key(value: str) -> str:
    return normalize_key(value).replace("_", " ")


def short_hash(value: str, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        rows.append(item)
        seen.add(item)
    return rows


def upsert_by_key(rows: list[dict[str, Any]], key: str, record: dict[str, Any]) -> None:
    value = record.get(key)
    for index, row in enumerate(rows):
        if row.get(key) == value:
            rows[index] = {**row, **record}
            return
    rows.append(dict(record))


def weighted_signature(pairs: list[tuple[str, float]], *, limit: int = 96) -> dict[str, float]:
    weights: dict[str, float] = {}
    for raw, weight in pairs:
        key = normalize_key(raw)
        if not key:
            continue
        weights[key] = max(weights.get(key, 0.0), round(float(weight), 4))
        for part in key.split("_"):
            if len(part) <= 2 or part in SIGNATURE_PART_STOPWORDS:
                continue
            weights[part] = max(weights.get(part, 0.0), round(float(weight) * 0.35, 4))
    ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return dict(sorted(ranked))


def fetch_url(url: str, *, timeout: int = 30) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "LucidScaleIngest/0.3 (+source-backed training)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - selected article URLs only.
        raw = response.read()
    return raw.decode("utf-8", errors="ignore")


def extract_text(raw_html: str) -> str:
    parser = TextExtractor()
    parser.feed(raw_html)
    return re.sub(r"\s+", " ", parser.text()).strip()


def split_sentences(
    text: str,
    *,
    config: IngestConfig | None = None,
    corpus_terms: frozenset[str] | None = None,
) -> list[str]:
    cfg = config or IngestConfig.scale_default()
    gate_terms = corpus_terms if corpus_terms is not None else frozenset(cfg.domain_terms_seed)
    rows: list[str] = []
    seen: set[str] = set()
    for raw in SENTENCE_RE.split(text):
        sentence = " ".join(raw.strip().split())
        if not cfg.min_sentence_length <= len(sentence) <= cfg.max_sentence_length:
            continue
        lowered = sentence.lower()
        if not cfg.cross_domain and gate_terms and not any(term in lowered for term in gate_terms):
            continue
        if not usable_sentence(lowered):
            continue
        if lowered in seen:
            continue
        rows.append(sentence)
        seen.add(lowered)
    return rows


def usable_sentence(lowered: str) -> bool:
    if any(marker in lowered for marker in RHETORICAL_SENTENCE_MARKERS):
        return False
    if any(marker in lowered for marker in MARKETING_SENTENCE_MARKERS):
        return False
    if any(marker in lowered for marker in WEB_CHROME_SENTENCE_MARKERS):
        return False
    if lowered.startswith(("copyright ", "get started", "learn more", "sign up", "try ")):
        return False
    return True


def load_articles(
    sources: list[dict[str, str]] | None = None,
    *,
    config: IngestConfig | None = None,
) -> tuple[list[Article], list[dict[str, str]]]:
    cfg = config or IngestConfig.scale_default()
    rows = sources if sources is not None else ARTICLE_SOURCES
    articles: list[Article] = []
    errors: list[dict[str, str]] = []
    total = len(rows)
    ingest_log(f"fetching {total} article source(s)...", cfg)
    for index, source in enumerate(rows, start=1):
        source_id = str(source.get("source_id") or "")
        url = str(source.get("url") or "")
        title = str(source.get("title") or source_id or url)
        try:
            if not url:
                raise RuntimeError("source row missing url")
            raw = fetch_url(url, timeout=cfg.fetch_timeout)
            text = extract_text(raw)
            sentences = split_sentences(text, config=cfg)
            if not sentences:
                raise RuntimeError(f"no usable sentences extracted from {url}")
            articles.append(
                Article(
                    source_id=source_id,
                    title=title,
                    url=url,
                    text=text,
                    sentences=sentences,
                )
            )
            if should_log_progress(index, total, cfg.progress_interval):
                ingest_log(
                    f"fetch {index}/{total} ok {source_id} ({len(sentences)} sentences)",
                    cfg,
                )
        except Exception as exc:  # noqa: BLE001 - collect per-source failures for scale runs
            error = {
                "source_id": source_id,
                "url": url,
                "error": str(exc),
            }
            if cfg.skip_article_errors:
                errors.append(error)
                if should_log_progress(index, total, cfg.progress_interval):
                    ingest_log(f"fetch {index}/{total} skip {source_id}: {exc}", cfg)
                continue
            raise RuntimeError(f"failed to ingest {url or source_id}: {exc}") from exc
    if not articles:
        raise RuntimeError("no articles loaded successfully")
    ingest_log(
        f"fetch complete: {len(articles)} loaded, {len(errors)} failed",
        cfg,
    )
    return articles, errors


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


def term_variants(term: str) -> list[str]:
    canonical = humanize_key(term)
    key = normalize_key(canonical)
    variants = {canonical, key.replace("_", " ")}
    if " " not in canonical:
        variants.add(singularize(canonical))
        variants.add(canonical + "s")
    else:
        parts = canonical.split()
        last = parts[-1]
        if not last.endswith("s"):
            variants.add(" ".join([*parts[:-1], last + "s"]))
        singular_last = singularize(last)
        if singular_last != last:
            variants.add(" ".join([*parts[:-1], singular_last]))
    return dedupe([variant for variant in variants if variant])


def extract_candidate_terms(
    articles: list[Article],
    *,
    corpus_context: CorpusContext | None = None,
    limit: int | None = None,
) -> list[str]:
    ctx = corpus_context or build_corpus_context(articles, config=IngestConfig.scale_default())
    cfg = ctx.config
    term_limit = limit if limit is not None else cfg.max_candidate_terms
    corpus_terms = ctx.corpus_terms
    broad_terms = set(cfg.broad_single_terms)
    counts: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()
    for article in articles:
        article_terms: set[str] = set()
        for sentence in article.sentences:
            tokens = [
                singularize(token)
                for token in word_tokens(sentence)
                if len(token) > 2 and token not in STOPWORDS
            ]
            for n in range(1, 5):
                for index in range(0, max(0, len(tokens) - n + 1)):
                    gram = tokens[index : index + n]
                    if not gram or all(token in STOPWORDS for token in gram):
                        continue
                    if n > 1 and gram[0] in STOPWORDS and gram[-1] in STOPWORDS:
                        continue
                    phrase = " ".join(gram)
                    if VENDOR_ARTIFACT_RE.match(normalize_key(phrase)):
                        continue
                    if len(phrase) < 4:
                        continue
                    gram_set = set(gram)
                    in_corpus = bool(gram_set & corpus_terms) or phrase in corpus_terms
                    if n == 1:
                        if phrase in broad_terms:
                            continue
                        if not in_corpus and len(phrase) < cfg.promote_long_unigram_len:
                            continue
                    elif not in_corpus and not cfg.cross_domain:
                        continue
                    counts[phrase] += 1
                    article_terms.add(phrase)
        for term in article_terms:
            doc_counts[term] += 1

    scored: list[tuple[float, str]] = []
    for term, count in counts.items():
        tokens = term.split()
        if len(tokens) == 1 and tokens[0] in broad_terms:
            continue
        if (
            count < cfg.min_term_count
            and doc_counts[term] < cfg.min_term_doc_count
            and len(tokens[0]) < cfg.promote_long_unigram_len
        ):
            continue
        doc_boost = 1.0 + 0.3 * doc_counts[term]
        domain_boost = 1.5 if set(tokens) & corpus_terms else 1.0
        multi_boost = 1.0 + 0.15 * (len(tokens) - 1)
        score = count * doc_boost * domain_boost * multi_boost
        scored.append((score, term))
    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked = [term for _score, term in scored[: term_limit * 2]]
    return [
        term
        for term in ranked[:term_limit]
        if is_valid_candidate_term(term, broad_terms=cfg.broad_single_terms)
    ]


def subject_anchor(sentence: str, subject: str) -> int | None:
    span = subject_span(sentence, subject)
    if span is not None:
        return span[1]
    lowered = sentence.lower()
    for variant in sorted(term_variants(subject), key=lambda value: (-len(value), value)):
        match = re.search(rf"\b{re.escape(variant.lower())}\b", lowered)
        if match:
            return match.end()
    return None


def classify_relation(sentence: str, subject: str) -> tuple[str, str]:
    lowered = sentence.lower()
    stop = subject_anchor(lowered, subject)
    if stop is None:
        return "", ""
    if len(lowered[:stop].split()) > 16:
        return "", ""

    after = lowered[stop:]
    after = re.sub(r"^[\s,;:()\-]+", "", after)
    after = re.sub(r"^(?:which|that|they|it)\s+", "", after)
    if any(after.startswith(marker) for marker in RHETORICAL_SENTENCE_MARKERS):
        return "", ""
    contrast_markers = (
        "although ",
        "though ",
        "while ",
        "whereas ",
        "instead of",
        "unlike",
        "different from",
        "compared with",
        "rather than",
    )
    if lowered.strip().startswith(contrast_markers) or any(
        marker in lowered for marker in contrast_markers[4:]
    ):
        target = clean_target(sentence, max_words=32)
        return ("contrast", target) if usable_target("contrast", target) else ("", "")
    near = r"^(?:[a-z0-9'-]+\s+){0,8}?"
    patterns: list[tuple[str, str]] = [
        ("type_of", near + r"\b(?:is|are|refers to|is called|are called)\b\s+(?P<tail>.+)"),
        ("type_of", near + r"\b(?:defined as|known as|described as|understood as)\b\s+(?P<tail>.+)"),
        ("uses", near + r"\b(?:uses|use|using|relies on|leverage|leverages|utilize|utilizes|utilizing)\b\s+(?P<tail>.+)"),
        ("uses", near + r"\b(?:is used to|are used to)\b\s+(?P<tail>.+)"),
        ("uses", near + r"\b(?:involves|involve|employs|employ)\b\s+(?P<tail>.+)"),
        ("uses", near + r"\b(?:depends on|depend on|requires|require|needs|need)\b\s+(?P<tail>.+)"),
        ("uses", near + r"\b(?:works by|work by|operates by|operate by|functions by|function by)\b\s+(?P<tail>.+)"),
        ("enables", near + r"\b(?:enables|enable|allows|allow|supports|support)\b\s+(?P<tail>.+)"),
        ("enables", near + r"\b(?:makes it possible|make it possible|makes possible|make possible)\b\s+(?P<tail>.+)"),
        ("capability", near + r"\b(?:can|could|may|might)\b\s+(?P<tail>.+)"),
        ("capability", near + r"\b(?:designed to|built to|intended to|aimed to|meant to)\b\s+(?P<tail>.+)"),
        ("property", near + r"\b(?:has|have|include|includes|contain|contains|store|stores|represent|represents|take on|takes on)\b\s+(?P<tail>.+)"),
        ("property", near + r"\b(?:characterized by|consists of|consist of|made up of|composed of|formed from)\b\s+(?P<tail>.+)"),
        ("property", near + r"\b(?:provides|provide|offers|offer|delivers|deliver)\b\s+(?P<tail>.+)"),
        ("related_to", near + r"\b(?:related to|associated with|connected to|linked to)\b\s+(?P<tail>.+)"),
    ]
    for relation, pattern in patterns:
        match = re.search(pattern, after, re.I)
        if match:
            target = clean_target(match.group("tail"))
            normalized_target = normalize_key(target).replace("_", " ")
            if relation == "type_of" and normalized_target.startswith(("able to ", "can ", "could ", "may ", "might ")):
                relation = "capability"
            elif relation == "type_of" and normalized_target.startswith("used to "):
                relation = "capability"
                target = f"be {target}"
            elif relation == "type_of" and (
                normalized_target.startswith(("error prone", "highly sensitive"))
                or any(word in normalized_target.split() for word in {"decoherence", "error", "errors", "noise", "noisy"})
            ):
                relation = "challenge"
            elif relation == "type_of" and normalized_target.startswith(
                ("based on", "built with", "composed of", "entangled", "placed into", "represented by")
            ):
                relation = "property"
            relation, target = normalize_relation_target(relation, target)
            if usable_target(relation, target):
                return relation, target
            return "", ""

    if any(word in lowered for word in ("noise", "noisy", "error", "errors", "decoherence", "challenge", "fragile", "difficult")):
        target = clean_target(sentence)
        return ("challenge", target) if usable_target("challenge", target) else ("", "")
    if any(word in lowered for word in ("measure", "measured", "measurement")):
        target = clean_target(sentence)
        return ("measurement", target) if usable_target("measurement", target) else ("", "")
    if any(word in lowered for word in ("instead of", "unlike", "different from", "compared with")):
        target = clean_target(sentence)
        return ("contrast", target) if usable_target("contrast", target) else ("", "")
    return "", ""


def subject_span(sentence: str, subject: str) -> tuple[int, int] | None:
    variants = sorted(term_variants(subject), key=lambda value: (-len(value), value))
    for variant in variants:
        escaped = re.escape(variant.lower())
        for match in re.finditer(rf"\b{escaped}\b", sentence, re.I):
            previous_words = word_tokens(sentence[: match.start()])
            previous_word = previous_words[-1] if previous_words else ""
            previous_previous_word = previous_words[-2] if len(previous_words) >= 2 else ""
            if previous_word in {"about", "for", "in", "of", "through", "to", "toward", "towards", "using", "via", "with"}:
                continue
            if (
                previous_word in {"a", "an", "the"}
                and previous_previous_word
                in {"about", "after", "before", "by", "for", "from", "in", "of", "through", "to", "toward", "towards", "using", "via", "with"}
            ):
                continue
            if (
                previous_word in {"a", "an", "the"}
                and previous_previous_word
                in {"building", "constructing", "creating", "designing", "developing", "running"}
            ):
                continue
            if previous_word and previous_word not in STOPWORDS and " " in variant.strip():
                continue
            tail = sentence[match.end() :]
            next_word = next(iter(word_tokens(tail)), "")
            if next_word in MODIFIER_FOLLOWERS:
                continue
            return match.start(), match.end()
    return None


def clean_target(text: str, *, max_words: int = 24) -> str:
    cleaned = re.sub(r"\[[^\]]+\]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;")
    cleaned = re.sub(r"^(?:generally speaking|in other words|that is),?\s+", "", cleaned, flags=re.I)
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words]).rstrip(",;:") + "..."
    return cleaned


def usable_target(relation: str, target: str, *, corpus_terms: frozenset[str] | None = None) -> bool:
    lowered = normalize_key(target).replace("_", " ")
    if not lowered or lowered in BAD_TARGET_STARTS:
        return False
    if any(lowered.startswith(prefix) for prefix in BAD_TARGET_STARTS):
        return False
    words = lowered.split()
    if len(words) < 3 and relation in {"type_of", "property", "uses", "enables"}:
        return False
    if len(words) <= 2 and corpus_terms and not (set(words) & set(corpus_terms)):
        return False
    if relation == "type_of" and lowered.startswith(("worth ", "both ", "not ")):
        return False
    return True


def fallback_subject(sentence: str, terms: list[str]) -> str:
    """Pick a term whose relation pattern matches even when strict span heuristics fail."""

    lowered = sentence.lower()
    matches = [
        term
        for term in terms
        if any(re.search(rf"\b{re.escape(variant.lower())}\b", lowered) for variant in term_variants(term))
    ]
    scored: list[tuple[int, int, str]] = []
    for term in matches:
        relation, _target = classify_relation(sentence, humanize_key(term))
        if not relation:
            continue
        anchor = subject_anchor(lowered, humanize_key(term))
        position = anchor if anchor is not None else 999
        scored.append((RELATION_PRIORITY.get(relation, 99), position, term))
    if not scored:
        return ""
    scored.sort(key=lambda row: (row[0], row[1], -len(row[2].split()), row[2]))
    return scored[0][2]


def best_subject(sentence: str, terms: list[str], *, broad_terms: frozenset[str] | None = None) -> str:
    valid_terms = [term for term in terms if is_valid_subject_term(term, broad_terms=broad_terms)]
    if not valid_terms:
        return ""
    normalized = " " + humanize_key(sentence) + " "
    matches = [
        term
        for term in valid_terms
        if any(f" {humanize_key(variant)} " in normalized for variant in term_variants(term))
    ]
    scored: list[tuple[int, int, int, str]] = []
    lowered = sentence.lower()
    for term in matches:
        relation, _target = classify_relation(sentence, humanize_key(term))
        if not relation:
            continue
        span = subject_span(lowered, humanize_key(term))
        if span is None:
            continue
        scored.append(
            (
                RELATION_PRIORITY.get(relation, 99),
                span[0],
                -len(term.split()),
                term,
            )
        )
    if scored:
        scored.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
        return scored[0][3]
    fallback = fallback_subject(sentence, valid_terms)
    if fallback:
        return fallback
    matches.sort(key=lambda term: (-len(term.split()), -len(term), term))
    return matches[0] if matches else ""


def source_ref_list(source_refs: list[str]) -> list[dict[str, str]]:
    return [{"ref_id": ref_id} for ref_id in dedupe(source_refs)]


def relation_handle(concept_id: str, relation: dict[str, Any]) -> str:
    base = f"{concept_id}:{relation.get('relation')}:{relation.get('target')}:{relation.get('source_sentence')}"
    return f"relation:{concept_id}:{short_hash(base)}"


def claim_handle(concept_id: str, relation: dict[str, Any]) -> str:
    base = f"{concept_id}:{relation.get('source_sentence')}:{relation.get('source_refs')}"
    return f"claim:{concept_id}:{short_hash(base)}"


def _term_spans(sentence: str, terms: list[str]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for term in terms:
        concept_id = normalize_key(term)
        for variant in term_variants(term):
            surface = humanize_key(variant)
            if not surface:
                continue
            for match in re.finditer(rf"\b{re.escape(surface)}\b", sentence, re.I):
                key = (match.start(), match.end(), concept_id)
                if key in seen:
                    continue
                spans.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "concept_id": concept_id,
                        "surface": match.group(0),
                    }
                )
                seen.add(key)
    spans.sort(key=lambda row: (int(row["start"]), -(int(row["end"]) - int(row["start"]))))
    filtered: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        if any(start >= left and end <= right for left, right in occupied):
            continue
        filtered.append(span)
        occupied.append((start, end))
    return filtered


def _alias_connector_between(text: str) -> str:
    normalized = " ".join(text.lower().strip(" ,;:()[]{}.-").split())
    if normalized in {"or", "also called", "also known as", "known as", "referred to as"}:
        return "alias_of"
    if normalized in {"short for", "stands for"}:
        return "abbreviation_of"
    return ""


def _concept_support_score(
    concept_id: str,
    relations_by_concept: dict[str, list[dict[str, Any]]],
    term_source_refs: dict[str, set[str]],
) -> tuple[int, int, int, str]:
    relation_count = len(relations_by_concept.get(concept_id, []))
    source_count = len(term_source_refs.get(concept_id, set()))
    # Prefer the concept with more evidence; ties prefer compact canonical terms.
    return (relation_count, source_count, -len(concept_id), concept_id)


def _canonical_for_equivalence(
    left_id: str,
    right_id: str,
    relations_by_concept: dict[str, list[dict[str, Any]]],
    term_source_refs: dict[str, set[str]],
) -> str:
    left_score = _concept_support_score(left_id, relations_by_concept, term_source_refs)
    right_score = _concept_support_score(right_id, relations_by_concept, term_source_refs)
    return left_id if left_score >= right_score else right_id


def extract_equivalence_edges(
    articles: list[Article],
    terms: list[str],
    relations_by_concept: dict[str, list[dict[str, Any]]],
    term_source_refs: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Learn source-backed concept identity edges from general alias patterns."""

    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    known_ids = {normalize_key(term) for term in terms}
    for article in articles:
        for sentence in article.sentences:
            spans = _term_spans(sentence, terms)
            for left, right in zip(spans, spans[1:]):
                left_id = str(left["concept_id"])
                right_id = str(right["concept_id"])
                if left_id == right_id or left_id not in known_ids or right_id not in known_ids:
                    continue
                connector = _alias_connector_between(sentence[int(left["end"]) : int(right["start"])])
                if not connector:
                    continue
                canonical = _canonical_for_equivalence(
                    left_id,
                    right_id,
                    relations_by_concept,
                    term_source_refs,
                )
                alias = right_id if canonical == left_id else left_id
                key = (alias, canonical, connector)
                row = edges.setdefault(
                    key,
                    {
                        "alias_concept_id": alias,
                        "canonical_concept_id": canonical,
                        "relation": connector,
                        "confidence": 0.88 if connector == "alias_of" else 0.82,
                        "source_refs": [],
                        "source_sentences": [],
                    },
                )
                row["source_refs"] = dedupe([*row["source_refs"], article.source_id])
                row["source_sentences"] = dedupe([*row["source_sentences"], sentence])
    return list(edges.values())


def _canonical_groups(
    concept_ids: list[str],
    equivalence_edges: list[dict[str, Any]],
    relations_by_concept: dict[str, list[dict[str, Any]]],
    term_source_refs: dict[str, set[str]],
) -> dict[str, str]:
    parent = {concept_id: concept_id for concept_id in concept_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        if left not in parent or right not in parent:
            return
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        canonical = _canonical_for_equivalence(root_left, root_right, relations_by_concept, term_source_refs)
        other = root_right if canonical == root_left else root_left
        parent[other] = canonical

    for edge in equivalence_edges:
        union(str(edge.get("alias_concept_id") or ""), str(edge.get("canonical_concept_id") or ""))

    return {concept_id: find(concept_id) for concept_id in concept_ids}


def merge_equivalent_concepts(
    concepts: list[dict[str, Any]],
    equivalence_edges: list[dict[str, Any]],
    relations_by_concept: dict[str, list[dict[str, Any]]],
    term_source_refs: dict[str, set[str]],
    *,
    max_per_facet: int = DEFAULT_MAX_RELATIONS_PER_FACET,
    max_total: int = DEFAULT_MAX_RELATIONS_PER_CONCEPT,
) -> list[dict[str, Any]]:
    if not equivalence_edges:
        return concepts

    concept_ids = [str(concept["concept_id"]) for concept in concepts]
    canonical_by_id = _canonical_groups(
        concept_ids,
        equivalence_edges,
        relations_by_concept,
        term_source_refs,
    )
    by_id = {str(concept["concept_id"]): concept for concept in concepts}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for concept_id, concept in by_id.items():
        grouped[canonical_by_id.get(concept_id, concept_id)].append(concept)

    merged: list[dict[str, Any]] = []
    for canonical_id, group in grouped.items():
        terms: list[str] = []
        relations: list[dict[str, Any]] = []
        source_refs: list[str] = []
        merged_ids: list[str] = []
        for concept in group:
            concept_id = str(concept["concept_id"])
            merged_ids.append(concept_id)
            terms.extend([humanize_key(concept_id), *[str(term) for term in concept.get("terms", [])]])
            relations.extend([dict(relation) for relation in concept.get("relations", [])])
            source_refs.extend([str(ref) for ref in concept.get("source_refs", []) if str(ref)])
        group_edges = [
            edge
            for edge in equivalence_edges
            if str(edge.get("alias_concept_id")) in merged_ids
            and str(edge.get("canonical_concept_id")) in merged_ids
        ]
        canonical = dict(by_id.get(canonical_id) or group[0])
        canonical["concept_id"] = canonical_id
        canonical["terms"] = dedupe(terms)
        canonical["relations"] = cap_relations_by_facet(
            merge_relations(relations),
            max_per_facet=max_per_facet,
            max_total=max_total,
        )
        canonical["source_refs"] = sorted(set(source_refs))
        extraction = dict(canonical.get("extraction") or {})
        extraction["canonicalization"] = {
            "method": "source_alias_equivalence_patterns",
            "merged_concept_ids": sorted(set(merged_ids)),
            "equivalence_edge_count": len(group_edges),
        }
        canonical["extraction"] = extraction
        if group_edges:
            canonical["equivalence_edges"] = group_edges
        merged.append(canonical)
    merged.sort(key=lambda concept: str(concept.get("concept_id", "")))
    return merged


def extract_concepts(
    articles: list[Article],
    *,
    audit: SentenceAudit | None = None,
    config: IngestConfig | None = None,
    corpus_context: CorpusContext | None = None,
) -> list[dict[str, Any]]:
    concepts, _report = extract_concepts_with_learning(
        articles,
        audit=audit,
        config=config,
        corpus_context=corpus_context,
    )
    return concepts


def extract_concepts_with_learning(
    articles: list[Article],
    *,
    audit: SentenceAudit | None = None,
    config: IngestConfig | None = None,
    corpus_context: CorpusContext | None = None,
) -> tuple[list[dict[str, Any]], IngestLearningReport]:
    cfg = config or IngestConfig.scale_default()
    ctx = corpus_context or build_corpus_context(articles, config=cfg)
    sentence_audit = audit or SentenceAudit()
    terms = extract_candidate_terms(articles, corpus_context=ctx)
    ingest_log(f"claims pass: {len(articles)} articles, {len(terms)} candidate terms", cfg)
    relations_by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    term_source_refs: dict[str, set[str]] = defaultdict(set)

    for article_index, article in enumerate(articles, start=1):
        sentence_audit.total_raw += len(article.sentences)
        for sentence in article.sentences:
            sentence_audit.eligible += 1
            subject = best_subject(sentence, terms, broad_terms=cfg.broad_single_terms)
            if not subject:
                sentence_audit.skipped_no_subject += 1
                sentence_audit.skip_reasons["no_subject"] += 1
                continue
            concept_id = normalize_key(subject)
            relation, target = classify_relation(sentence, humanize_key(subject))
            if not relation:
                sentence_audit.skipped_no_relation += 1
                sentence_audit.skip_reasons["no_relation_pattern"] += 1
                continue
            if not target or normalize_key(target) == concept_id:
                sentence_audit.skipped_self_target += 1
                sentence_audit.skip_reasons["self_target"] += 1
                continue
            record = {
                "relation": relation,
                "target": target,
                "confidence": relation_confidence(relation, sentence),
                "source_refs": [article.source_id],
                "source_sentence": sentence,
            }
            reject_reason = reject_relation(record)
            if reject_reason:
                sentence_audit.skipped_no_relation += 1
                sentence_audit.skip_reasons[f"quality_{reject_reason}"] += 1
                continue
            entity = ctx.source_entities.get(article.source_id) or source_entity_for_article(article.source_id)
            if entity:
                record["source_entity"] = entity
            relations_by_concept[concept_id].append(record)
            term_source_refs[concept_id].add(article.source_id)
            sentence_audit.extracted += 1
        if should_log_progress(article_index, len(articles), cfg.progress_interval):
            ingest_log(
                f"claims pass {article_index}/{len(articles)} {article.source_id} "
                f"({sentence_audit.extracted} claims so far)",
                cfg,
            )

    equivalence_edges = extract_equivalence_edges(
        articles,
        terms,
        relations_by_concept,
        term_source_refs,
    )
    ingest_log(f"equivalence edges: {len(equivalence_edges)}", cfg)

    concepts: list[dict[str, Any]] = []
    for term in terms:
        concept_id = normalize_key(term)
        relations = cap_relations_by_facet(
            merge_relations(relations_by_concept.get(concept_id, [])),
            max_per_facet=cfg.max_relations_per_facet,
            max_total=cfg.max_relations_per_concept,
        )
        if not relations:
            continue
        concepts.append(
            {
                "concept_id": concept_id,
                "terms": term_variants(term),
                "relations": relations,
                "source_refs": sorted(term_source_refs.get(concept_id, [])),
                "extraction": {
                    "method": "scale_ingest_ngram_sentence_claims",
                    "relation_count": len(relations),
                },
            }
        )

    merged = merge_equivalent_concepts(
        concepts,
        equivalence_edges,
        relations_by_concept,
        term_source_refs,
        max_per_facet=cfg.max_relations_per_facet,
        max_total=cfg.max_relations_per_concept,
    )
    ingest_log(f"merged concepts: {len(merged)}", cfg)
    before_split = len(merged)
    branched, conflict_events = apply_contradiction_branches(
        merged,
        branch_hash=lambda relation: short_hash(
            f"{relation.get('relation')}:{relation.get('target')}:{relation.get('source_sentence')}"
        ),
    )
    branched = consolidate_vendor_artifact_concepts(branched)
    filtered, quality_stats = filter_concepts(
        branched,
        broad_terms=cfg.broad_single_terms,
        corpus_terms=ctx.corpus_terms,
    )
    ingest_log(
        f"quality filter: {quality_stats['concepts_before_quality_filter']} -> "
        f"{quality_stats['concepts_after_quality_filter']} concepts",
        cfg,
    )
    report = IngestLearningReport(
        sentence_audit=sentence_audit,
        concepts_before_split=before_split,
        concepts_after_split=len(filtered),
        contradiction_splits=len(conflict_events),
        contradiction_events=conflict_events,
        article_sentence_counts={article.source_id: len(article.sentences) for article in articles},
        concepts_before_quality_filter=int(quality_stats["concepts_before_quality_filter"]),
        concepts_after_quality_filter=int(quality_stats["concepts_after_quality_filter"]),
        quality_rejections=dict(quality_stats["concept_rejections"]),
        relation_quality_rejections=dict(quality_stats["relation_rejections"]),
    )
    return filtered, report


def relation_confidence(relation: str, sentence: str) -> float:
    base = {
        "type_of": 0.82,
        "property": 0.76,
        "uses": 0.78,
        "enables": 0.76,
        "capability": 0.72,
        "challenge": 0.7,
        "measurement": 0.68,
        "contrast": 0.68,
        "related_to": 0.6,
    }.get(relation, 0.6)
    if len(sentence.split()) <= 28:
        base += 0.04
    return round(min(0.92, base), 4)


def merge_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for relation in relations:
        key = (str(relation.get("relation")), normalize_key(relation.get("target")))
        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(relation)
            continue
        existing["confidence"] = max(float(existing.get("confidence", 0.0)), float(relation.get("confidence", 0.0)))
        existing["source_refs"] = dedupe(list(existing.get("source_refs", [])) + list(relation.get("source_refs", [])))
    rows = list(merged.values())
    rows.sort(
        key=lambda item: (
            RELATION_PRIORITY.get(str(item.get("relation")), 99),
            -float(item.get("confidence", 0.0)),
            str(item.get("target")),
        )
    )
    return rows


def concept_source_refs(concept: dict[str, Any]) -> list[str]:
    return dedupe(
        list(concept.get("source_refs", []))
        + [
            str(ref)
            for relation in concept.get("relations", [])
            for ref in relation.get("source_refs", [])
            if str(ref)
        ]
    )


def build_trace_records(concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for concept in concepts:
        concept_id = str(concept["concept_id"])
        source_refs = concept_source_refs(concept)
        terms = [str(term) for term in concept.get("terms", []) if str(term)]
        relation_confidences = [
            float(relation.get("confidence", 0.0) or 0.0)
            for relation in concept.get("relations", [])
            if isinstance(relation, dict)
        ]
        trust = round(sum(relation_confidences) / max(1, len(relation_confidences)), 4)
        records.append(
            {
                "trace_id": f"t_term_{concept_id}",
                "trace_family": concept_id,
                "alias": terms[0] if terms else concept_id,
                "cue_affinities": weighted_signature(
                    [(concept_id, 0.95), *[(term, 0.9) for term in terms]],
                ),
                "cluster_id": concept_id,
                "heat_tier": "quarantine",
                "maturity_state": "provisional",
                "activation_bias": 0.04,
                "activation_count": len(source_refs),
                "success_count": 0,
                "failure_count": 0,
                "created_from_cues": terms,
                "created_from_examples": source_refs,
                "source_refs": source_refs,
                "trust_score": trust,
                "description": f"Auto-extracted term trace for {concept_id}",
                "last_update_summary": "scale_ingest",
            }
        )
        for index, relation in enumerate(concept.get("relations", [])):
            rid = short_hash(f"{concept_id}:{index}:{relation.get('source_sentence')}")
            records.append(
                {
                    "trace_id": f"t_claim_{concept_id}_{rid}",
                    "trace_family": concept_id,
                    "alias": f"{concept_id}_{relation.get('relation')}",
                    "cue_affinities": weighted_signature(
                        [
                            (concept_id, 0.88),
                            (relation.get("relation", ""), 0.64),
                            (relation.get("target", ""), 0.56),
                            *[(term, 0.78) for term in terms],
                            *(
                                [(str(relation.get("source_entity", "")), 0.72)]
                                if relation.get("source_entity")
                                else []
                            ),
                        ],
                    ),
                    "cluster_id": concept_id,
                    "heat_tier": "quarantine",
                    "maturity_state": "provisional",
                    "activation_bias": 0.02,
                    "activation_count": len(relation.get("source_refs", [])),
                    "success_count": 0,
                    "failure_count": 0,
                    "created_from_cues": [concept_id, str(relation.get("target", ""))],
                    "created_from_examples": list(relation.get("source_refs", [])),
                    "source_refs": list(relation.get("source_refs", [])),
                    "trust_score": float(relation.get("confidence", 0.0) or 0.0),
                    "description": f"Auto-extracted claim trace for {concept_id}",
                    "last_update_summary": "scale_ingest",
                }
            )
    return records


def build_alias_records(concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aliases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for concept in concepts:
        concept_id = str(concept["concept_id"])
        for term in concept.get("terms", []):
            for variant in term_variants(str(term)):
                key = normalize_key(variant)
                if not key or key in seen:
                    continue
                seen.add(key)
                aliases.append(
                    {
                        "alias_id": f"alias_{key}",
                        "surface_pattern": variant,
                        "relation_candidates": ["concept", concept_id],
                        "confidence": 0.7 if "_" in key else 0.62,
                        "source": "scale_ingest",
                    }
                )
        for edge in concept.get("equivalence_edges") or []:
            if not isinstance(edge, dict):
                continue
            alias_concept_id = str(edge.get("alias_concept_id") or "").strip()
            canonical_id = str(edge.get("canonical_concept_id") or concept_id).strip()
            if not alias_concept_id or canonical_id != concept_id:
                continue
            for variant in term_variants(humanize_key(alias_concept_id)):
                key = normalize_key(variant)
                if not key:
                    continue
                aliases.append(
                    {
                        "alias_id": f"alias_equiv_{key}_to_{concept_id}",
                        "surface_pattern": variant,
                        "relation_candidates": ["concept", concept_id],
                        "canonical_concept_id": concept_id,
                        "alias_concept_id": alias_concept_id,
                        "alias_kind": str(edge.get("relation") or "alias_of"),
                        "confidence": float(edge.get("confidence", 0.86) or 0.86),
                        "source": "scale_ingest_equivalence",
                        "source_refs": list(edge.get("source_refs") or []),
                        "source_sentences": list(edge.get("source_sentences") or []),
                    }
                )
    return aliases


def target_concept_ids(target: str, concepts: list[dict[str, Any]]) -> list[str]:
    key = normalize_key(target)
    matches: list[str] = []
    for concept in concepts:
        concept_id = str(concept["concept_id"])
        candidates = [concept_id, *[normalize_key(term) for term in concept.get("terms", [])]]
        if any(candidate and candidate != concept_id and candidate in key for candidate in candidates):
            matches.append(concept_id)
        elif concept_id and concept_id in key:
            matches.append(concept_id)
    return dedupe(matches)


def build_basin_records(concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    basins: list[dict[str, Any]] = []
    for concept in concepts:
        concept_id = str(concept["concept_id"])
        terms = [str(term) for term in concept.get("terms", []) if str(term)]
        by_facet: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for relation in concept.get("relations", []):
            facet = FACET_BY_RELATION.get(str(relation.get("relation")), "definition")
            by_facet[facet].append(relation)
        for facet, relations in sorted(by_facet.items()):
            source_refs = dedupe([ref for relation in relations for ref in relation.get("source_refs", [])])
            relation_handles = [relation_handle(concept_id, relation) for relation in relations]
            evidence_handles = [f"concept:{concept_id}", *[claim_handle(concept_id, relation) for relation in relations]]
            relation_trace_ids = [
                f"t_claim_{concept_id}_{short_hash(f'{concept_id}:{index}:{relation.get('source_sentence')}')}"
                for index, relation in enumerate(concept.get("relations", [])[:8])
                if relation in relations
            ]
            cooperation_links: dict[str, float] = {}
            for other_facet in by_facet:
                if other_facet != facet:
                    cooperation_links[f"b_{concept_id}_{other_facet}"] = 0.55
            for relation in relations:
                for target_id in target_concept_ids(str(relation.get("target", "")), concepts):
                    if target_id != concept_id:
                        cooperation_links[f"b_{target_id}_definition"] = max(
                            cooperation_links.get(f"b_{target_id}_definition", 0.0),
                            0.48,
                        )
            trust = round(sum(float(r.get("confidence", 0.0)) for r in relations) / max(1, len(relations)), 4)
            activation_pairs = [
                (concept_id, 0.95),
                (f"t_term_{concept_id}", 0.96),
                (facet, 0.66),
                *[(term, 0.86) for term in terms],
                *[(trace_id, 0.88) for trace_id in relation_trace_ids],
                *[(relation.get("target", ""), 0.46) for relation in relations],
            ]
            semantic_pairs = [
                (concept_id, 0.9),
                (facet, 0.75),
                *[(term, 0.72) for term in terms],
                *[(relation.get("relation", ""), 0.58) for relation in relations],
                *[(relation.get("target", ""), 0.5) for relation in relations],
            ]
            basins.append(
                {
                    "basin_id": f"b_{concept_id}_{facet}",
                    "family_hint": concept_id,
                    "frame_affinities": {"frame_active": 0.7, "concept": 0.66, "event": 0.34},
                    "activation_signature": weighted_signature(activation_pairs),
                    "semantic_signature": weighted_signature(semantic_pairs),
                    "evidence_handles": evidence_handles,
                    "relation_handles": relation_handles,
                    "source_refs": source_refs,
                    "trust_score": trust,
                    "heat_tier": "quarantine",
                    "cooperation_links": dict(sorted(cooperation_links.items())),
                    "suppression_links": {},
                    "support_examples": source_refs,
                    "quantized_payload": {
                        "precision": "uint8_sparse",
                        "canonical_label": humanize_key(concept_id),
                        "concept_id": concept_id,
                        "facet": facet,
                        "terms": [normalize_key(term) for term in terms],
                        "relations": [
                            {
                                "relation": relation.get("relation"),
                                "target": relation.get("target"),
                                "confidence": relation.get("confidence"),
                                "source_refs": relation.get("source_refs", []),
                            }
                            for relation in relations
                        ],
                        "source_count": len(source_refs),
                    },
                }
            )
    return basins


def build_basic_language_records() -> dict[str, list[dict[str, Any]]]:
    aliases: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    basins: list[dict[str, Any]] = []
    decoder_targets: list[dict[str, Any]] = []
    by_kind: dict[str, list[str]] = defaultdict(list)
    for phrase in BASIC_LANGUAGE_PHRASES:
        kind, response = basic_language_response(phrase)
        by_kind[kind].append(phrase)
        key = normalize_key(phrase)
        aliases.append(
            {
                "alias_id": f"alias_basic_{key}",
                "surface_pattern": phrase,
                "relation_candidates": ["social", kind],
                "confidence": 0.9,
                "source": "basic_language_phrase_corpus",
            }
        )
        traces.append(
            {
                "trace_id": f"t_basic_{key}",
                "trace_family": f"basic_{kind}",
                "alias": phrase,
                "cue_affinities": weighted_signature([(phrase, 0.95), (f"social:{kind}", 0.9), (kind, 0.82)]),
                "cluster_id": f"basic_{kind}",
                "heat_tier": "warm",
                "maturity_state": "active",
                "activation_bias": 0.06,
                "activation_count": 1,
                "success_count": 1,
                "failure_count": 0,
                "created_from_cues": [phrase],
                "created_from_examples": ["basic_language_phrase_corpus"],
                "source_refs": ["basic_language_phrase_corpus"],
                "description": f"Basic language trace for {phrase}",
                "last_update_summary": "scale_ingest_basic_language",
            }
        )
        decoder_targets.append(
            {
                "template_id": f"basic_{key}",
                "episode_id": f"basic-language-{key}",
                "expected_answer": response,
                "lucidity_target": "COMMIT",
                "validator": "exact_social",
                "source": "basic_language_phrase_corpus",
            }
        )

    for kind, phrases in by_kind.items():
        response = basic_language_response(phrases[0])[1]
        basins.append(
            {
                "basin_id": f"b_basic_{kind}",
                "family_hint": f"basic_{kind}",
                "frame_affinities": {"frame_active": 0.72, "event": 0.45},
                "activation_signature": weighted_signature(
                    [(f"t_basic_{normalize_key(phrase)}", 0.9) for phrase in phrases]
                    + [(phrase, 0.85) for phrase in phrases]
                    + [(kind, 0.8)]
                ),
                "semantic_signature": weighted_signature([(kind, 0.9), *[(phrase, 0.75) for phrase in phrases]]),
                "evidence_handles": [f"basic_phrase:{normalize_key(phrase)}" for phrase in phrases],
                "relation_handles": [f"basic_response:{kind}"],
                "source_refs": ["basic_language_phrase_corpus"],
                "trust_score": 0.9,
                "heat_tier": "warm",
                "cooperation_links": {},
                "suppression_links": {},
                "support_examples": phrases,
                "quantized_payload": {
                    "precision": "uint8_sparse",
                    "canonical_label": f"basic {kind}",
                    "facet": "speech",
                    "relations": [
                        {
                            "relation": "speech_response",
                            "target": response,
                            "confidence": 0.9,
                            "source_refs": ["basic_language_phrase_corpus"],
                        }
                    ],
                },
            }
        )
    return {
        "aliases": aliases,
        "traces": traces,
        "basins": basins,
        "decoder_targets": decoder_targets,
    }


def build_paraphrase_query_records() -> dict[str, list[dict[str, Any]]]:
    """Aliases and traces so paraphrased concept questions bind to query frames."""
    aliases: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for prefix in PARAPHRASE_QUERY_PREFIXES:
        key = normalize_key(prefix)
        aliases.append(
            {
                "alias_id": f"alias_query_{key}",
                "surface_pattern": prefix,
                "relation_candidates": ["concept_query", "definition_query"],
                "confidence": 0.88,
                "source": "paraphrase_query_corpus",
            }
        )
        traces.append(
            {
                "trace_id": f"t_query_{key}",
                "trace_family": "concept_query_like",
                "alias": prefix,
                "cue_affinities": weighted_signature(
                    [(prefix, 0.92), ("concept_query", 0.9), ("definition_query", 0.86)]
                ),
                "cluster_id": "concept_query",
                "heat_tier": "warm",
                "maturity_state": "active",
                "activation_bias": 0.05,
                "activation_count": 1,
                "success_count": 1,
                "failure_count": 0,
                "created_from_cues": [prefix],
                "created_from_examples": ["paraphrase_query_corpus"],
                "source_refs": ["paraphrase_query_corpus"],
                "description": f"Paraphrase query prefix: {prefix}",
                "last_update_summary": "scale_ingest_paraphrase_query",
            }
        )
    return {"aliases": aliases, "traces": traces, "basins": [], "decoder_targets": []}


def basic_language_response(phrase: str) -> tuple[str, str]:
    key = normalize_key(phrase)
    if key in {"hi", "hello", "hey", "good_morning", "good_afternoon", "good_evening"}:
        return "greeting", "Hello."
    if key in {"thanks", "thank_you"}:
        return "thanks", "You're welcome."
    if key in {"bye", "goodbye"}:
        return "farewell", "Goodbye."
    if key == "how_are_you":
        return "how_are_you", "I'm here and ready to help."
    return "capability", "I'm Lucid. I answer from audited pipeline state."


def run_ingest_learning_pipeline(
    articles: list[Article],
    *,
    audit: SentenceAudit | None = None,
    config: IngestConfig | None = None,
    corpus_context: CorpusContext | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], IngestLearningReport, CorpusContext]:
    cfg = config or IngestConfig.scale_default()
    ctx = corpus_context or build_corpus_context(articles, config=cfg)
    concepts, report = extract_concepts_with_learning(
        articles,
        audit=audit,
        config=cfg,
        corpus_context=ctx,
    )
    traces = build_trace_records(concepts)
    report.traces_before_consolidation = len(traces)
    ingest_log(f"building traces: {len(traces)} raw", cfg)
    traces, deduped = consolidate_trace_records(traces)
    report.traces_after_consolidation = len(traces)
    report.traces_deduplicated = deduped
    return concepts, traces, report, ctx


def resolve_ingest_sources(
    *,
    sources: list[dict[str, str]] | None = None,
    sources_path: str | Path | None = None,
) -> list[dict[str, str]]:
    if sources_path:
        return load_sources_from_path(sources_path)
    if sources is not None:
        return sources
    return ARTICLE_SOURCES


def train_scale_ingest(
    checkpoint: str | Path = "checkpoints/saves/v.0.2",
    *,
    pin_loaded: bool = False,
    write_audit: bool = True,
    config: IngestConfig | None = None,
    sources: list[dict[str, str]] | None = None,
    sources_path: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or IngestConfig.scale_default()
    root = resolve_checkpoint(checkpoint)
    state = empty_checkpoint(root.name or "v.0.2")
    state.checkpoint_id = root.name or "v.0.2"

    source_rows = resolve_ingest_sources(sources=sources, sources_path=sources_path)
    ingest_log(f"starting checkpoint {root.name} from {len(source_rows)} source(s)", cfg)
    clear_source_entities()
    register_source_entities(build_source_entity_map(source_rows))

    articles, load_errors = load_articles(source_rows, config=cfg)
    ingest_log("building corpus context...", cfg)
    corpus_ctx = build_corpus_context(articles, config=cfg, sources=source_rows)
    register_source_entities(corpus_ctx.source_entities)
    ingest_log(f"corpus terms: {len(corpus_ctx.corpus_terms)}", cfg)

    concepts, traces, learning_report, corpus_ctx = run_ingest_learning_pipeline(
        articles,
        config=cfg,
        corpus_context=corpus_ctx,
    )
    ingest_log(f"assembling stores ({len(concepts)} concepts, {len(traces)} traces)...", cfg)
    aliases = [
        *build_alias_records(concepts),
        *build_mechanism_relation_aliases(corpus_ctx.source_entities),
    ]
    basins = build_basin_records(concepts)
    basic = build_basic_language_records()
    paraphrase = build_paraphrase_query_records()

    concept_bank = state.ensure_store("concept_bank")
    concept_store = concept_bank.setdefault("concepts", [])
    source_store = concept_bank.setdefault("sources", [])
    trace_store = state.ensure_store("tracebank").setdefault("records", [])
    basin_store = state.ensure_store("basin_bank").setdefault("records", [])
    alias_store = state.ensure_store("relation_aliases").setdefault("aliases", [])
    operator_store = state.ensure_store("operator_bank").setdefault("operators", [])
    decoder_store = state.ensure_store("decoder_adapter").setdefault("render_targets", [])

    upsert_by_key(
        source_store,
        "source_id",
        {
            "source_id": "basic_language_phrase_corpus",
            "title": "Basic language phrase corpus",
            "url": "",
            "source_type": "phrase_corpus",
        },
    )
    ensure_metadata(
        state,
        "source:basic_language_phrase_corpus",
        "source",
        source="scale_ingest",
        source_refs=[{"ref_id": "basic_language_phrase_corpus", "title": "Basic language phrase corpus"}],
    )

    for article in articles:
        upsert_by_key(
            source_store,
            "source_id",
            {
                "source_id": article.source_id,
                "title": article.title,
                "url": article.url,
                "source_type": "article",
                "sentence_count": len(article.sentences),
            },
        )
        ensure_metadata(
            state,
            f"source:{article.source_id}",
            "source",
            source="scale_ingest",
            source_refs=[{"ref_id": article.url, "title": article.title}],
        )

    for concept in concepts:
        relation_confidences = [
            float(relation.get("confidence", 0.0) or 0.0)
            for relation in concept.get("relations", [])
            if isinstance(relation, dict)
        ]
        trust = sum(relation_confidences) / max(1, len(relation_confidences))
        metadata = source_backed_shadow_promotion(
            state,
            f"concept:{concept['concept_id']}",
            "concept",
            source="scale_ingest",
            source_refs=source_ref_list(concept_source_refs(concept)),
            support_count=len(concept.get("relations", [])),
            trust_score=trust,
        )
        concept_record = dict(concept)
        concept_record["heat_tier"] = metadata["heat_tier"]
        concept_record["commit_permission"] = metadata["commit_permission"]
        upsert_by_key(concept_store, "concept_id", concept_record)

    for trace in [*traces, *basic["traces"], *paraphrase["traces"]]:
        if str(trace.get("heat_tier") or "") == "warm":
            metadata = ensure_metadata(
                state,
                f"trace:{trace['trace_id']}",
                "trace",
                source="scale_ingest",
                precision_tier="uint8_sparse",
                source_refs=source_ref_list(list(trace.get("source_refs", []))),
            )
            metadata["support_count"] = max(int(metadata.get("support_count", 0)), 1)
            metadata["shadow_pass_count"] = max(int(metadata.get("shadow_pass_count", 0)), 1)
            metadata["heat_tier"] = "warm"
            metadata["commit_permission"] = "normal_support"
        else:
            metadata = source_backed_shadow_promotion(
                state,
                f"trace:{trace['trace_id']}",
                "trace",
                source="scale_ingest",
                precision_tier="uint8_sparse",
                source_refs=source_ref_list(list(trace.get("source_refs", []))),
                support_count=max(
                    len(trace.get("source_refs", [])),
                    int(trace.get("activation_count", 0) or 0),
                ),
                trust_score=float(trace.get("trust_score", 0.0) or 0.0),
            )
        metadata["quantization_candidate"] = True
        apply_runtime_promotion_fields(trace, metadata, has_maturity=True)
        upsert_by_key(trace_store, "trace_id", trace)

    for basin in [*basins, *basic["basins"]]:
        if str(basin.get("heat_tier") or "") == "warm":
            metadata = ensure_metadata(
                state,
                f"basin:{basin['basin_id']}",
                "basin",
                source="scale_ingest",
                precision_tier="uint8_sparse",
                source_refs=source_ref_list(list(basin.get("source_refs", []))),
            )
            metadata["support_count"] = max(int(metadata.get("support_count", 0)), 1)
            metadata["shadow_pass_count"] = max(int(metadata.get("shadow_pass_count", 0)), 1)
            metadata["heat_tier"] = "warm"
            metadata["commit_permission"] = "normal_support"
        else:
            metadata = source_backed_shadow_promotion(
                state,
                f"basin:{basin['basin_id']}",
                "basin",
                source="scale_ingest",
                precision_tier="uint8_sparse",
                source_refs=source_ref_list(list(basin.get("source_refs", []))),
                support_count=len(basin.get("relation_handles", [])),
                trust_score=float(basin.get("trust_score", 0.0) or 0.0),
            )
        metadata["quantization_candidate"] = True
        apply_runtime_promotion_fields(basin, metadata)
        upsert_by_key(basin_store, "basin_id", basin)

    for alias in [*aliases, *basic["aliases"], *paraphrase["aliases"]]:
        upsert_by_key(alias_store, "alias_id", alias)
        record_support(state, f"alias:{alias['alias_id']}", "relation_alias")

    for operator in BOOTSTRAP_OPERATORS:
        operator_record = dict(operator)
        promote_operator_from_evidence(
            state,
            operator_record,
            source="scale_ingest",
            source_refs=["scale_ingest_bootstrap"],
            support_count=3,
            shadow_pass_count=1,
            trust_score=float(operator.get("default_confidence", 0.0) or 0.0),
        )
        upsert_by_key(operator_store, "operator_id", operator_record)

    for event in learning_report.contradiction_events:
        base_id = str(event.get("base_concept_id") or "")
        if base_id:
            record_contradiction(state, f"concept:{base_id}", "concept")

    for target in basic["decoder_targets"]:
        upsert_by_key(decoder_store, "template_id", target)

    metadata_objects = state.ensure_store("learned_metadata").get("objects", {})
    warm, probation, quarantine = count_concept_heat_tiers(concepts, metadata_objects)
    learning_report.warm_concepts = warm
    learning_report.probation_concepts = probation
    learning_report.quarantine_concepts = quarantine

    audit_payload = learning_report.to_dict()
    audit_payload["checkpoint"] = str(root)
    audit_payload["concepts"] = len(concepts)
    audit_payload["traces"] = len(trace_store)
    audit_payload["basins"] = len(basin_store)
    audit_payload["corpus_terms"] = len(corpus_ctx.corpus_terms)
    audit_payload["load_errors"] = load_errors
    audit_payload["ingest_config"] = {
        "cross_domain": cfg.cross_domain,
        "max_candidate_terms": cfg.max_candidate_terms,
        "max_relations_per_concept": cfg.max_relations_per_concept,
        "max_relations_per_facet": cfg.max_relations_per_facet,
    }
    audit_path: str | None = None
    if write_audit:
        audit_dir = resolve_train_path("audit/runs/ingest")
        audit_file = audit_dir / f"{root.name or 'ingest'}-report.json"
        write_ingest_audit_report(audit_file, audit_payload)
        audit_path = str(audit_file)
        text_report = audit_dir / f"{root.name or 'ingest'}-report.txt"
        text_report.write_text(format_ingest_audit_text(learning_report), encoding="utf-8")
        ingest_log(f"audit report -> {audit_file}", cfg)

    ingest_log(f"saving checkpoint -> {root}", cfg)
    save_checkpoint(state, root, force=True, step_delta=1)
    summary = checkpoint_summary(load_checkpoint(root, create=False))
    registered = register_checkpoint(
        name=root.name,
        path=root,
        label=cfg.checkpoint_label,
        command="lucid.training.scale_ingest",
        summary=summary,
    )
    loaded: str | None = None
    if pin_loaded:
        loaded = str(promote_to_loaded(root, label=cfg.checkpoint_label))
        ingest_log(f"pinned loaded checkpoint -> {loaded}", cfg)
    ingest_log(
        f"done: {len(concepts)} concepts, {len(trace_store)} traces, {len(basin_store)} basins",
        cfg,
    )

    return {
        "checkpoint": str(root),
        "registered": registered,
        "loaded": loaded,
        "articles": len(articles),
        "article_sentence_counts": learning_report.article_sentence_counts,
        "concepts": len(concepts),
        "traces": len(trace_store),
        "basins": len(basin_store),
        "relation_aliases": len(alias_store),
        "decoder_targets": len(decoder_store),
        "metadata_objects": len(state.ensure_store("learned_metadata").get("objects", {})),
        "store_counts": summary["store_counts"],
        "ingest_learning": learning_report.to_dict(),
        "ingest_audit_path": audit_path,
        "ingest_coverage_ratio": learning_report.sentence_audit.to_dict()["coverage_ratio"],
        "corpus_terms": len(corpus_ctx.corpus_terms),
        "load_errors": load_errors,
        "ingest_config": {
            "cross_domain": cfg.cross_domain,
            "max_candidate_terms": cfg.max_candidate_terms,
            "max_relations_per_concept": cfg.max_relations_per_concept,
            "max_relations_per_facet": cfg.max_relations_per_facet,
        },
    }


def run_crosstalk_smoke() -> dict[str, Any]:
    """Synthetic two-article crosstalk check from constitution/test/learning.md."""

    articles = [
        Article(
            source_id="crosstalk_article_one",
            title="Crosstalk Article 1",
            url="https://example.test/crosstalk/one",
            text="",
            sentences=[
                "Quantum computing is a multidisciplinary field that uses quantum mechanics to solve complex problems.",
                "Quantum computing relies on qubits and superposition to perform useful computation.",
            ],
        ),
        Article(
            source_id="crosstalk_article_two",
            title="Crosstalk Article 2",
            url="https://example.test/crosstalk/two",
            text="",
            sentences=[
                "Quantum computing is merely a theoretical curiosity with no practical use in industry today.",
                "Quantum computing hardware remains too noisy for reliable results across most benchmarks.",
            ],
        ),
    ]
    concepts, report = extract_concepts_with_learning(articles)
    report.crosstalk_pass = evaluate_crosstalk(concepts, base_concept_id="quantum_computing")
    return {
        "crosstalk_pass": report.crosstalk_pass,
        "concepts_after_split": report.concepts_after_split,
        "contradiction_splits": report.contradiction_splits,
        "contradiction_events": report.contradiction_events,
        "ingest_learning": report.to_dict(),
    }


def audit_ingest_from_articles(
    articles: list[Article] | None = None,
    *,
    config: IngestConfig | None = None,
    sources_path: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or IngestConfig.scale_default()
    load_errors: list[dict[str, str]] = []
    if articles is not None:
        rows = articles
    else:
        source_rows = resolve_ingest_sources(sources_path=sources_path)
        rows, load_errors = load_articles(source_rows, config=cfg)
    concepts, traces, report, ctx = run_ingest_learning_pipeline(rows, config=cfg)
    payload = report.to_dict()
    payload.update(
        {
            "concepts": len(concepts),
            "traces": len(traces),
            "corpus_terms": len(ctx.corpus_terms),
            "load_errors": load_errors,
            "ingest_audit_text": format_ingest_audit_text(report),
        }
    )
    return payload


def _ingest_config_from_args(args: argparse.Namespace) -> IngestConfig:
    if getattr(args, "quantum_preset", False):
        return IngestConfig.quantum_default()
    return IngestConfig(
        cross_domain=not getattr(args, "domain_gate", False),
        max_candidate_terms=getattr(args, "max_candidate_terms", 2000),
        max_relations_per_facet=getattr(args, "max_relations_per_facet", 32),
        max_relations_per_concept=getattr(args, "max_relations_per_concept", 48),
        skip_article_errors=not getattr(args, "fail_fast", False),
        progress_logging=not getattr(args, "no_progress", False),
        progress_interval=max(1, int(getattr(args, "progress_interval", 5))),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train scale-style source-backed checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/saves/v.0.2")
    parser.add_argument("--sources", help="JSON or JSONL file listing article sources")
    parser.add_argument("--pin-loaded", action="store_true")
    parser.add_argument("--no-audit", action="store_true", help="Skip writing ingest audit report files")
    parser.add_argument("--quantum-preset", action="store_true", help="Use legacy quantum-only ingest profile")
    parser.add_argument("--domain-gate", action="store_true", help="Require seed/domain terms in each sentence")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first article load failure")
    parser.add_argument("--max-candidate-terms", type=int, default=2000)
    parser.add_argument("--max-relations-per-facet", type=int, default=32)
    parser.add_argument("--max-relations-per-concept", type=int, default=48)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=5,
        help="Log fetch/claims progress every N articles (1 = every article)",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable stderr progress logging")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            train_scale_ingest(
                args.checkpoint,
                pin_loaded=args.pin_loaded,
                write_audit=not args.no_audit,
                config=_ingest_config_from_args(args),
                sources_path=args.sources,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
