"""Cross-domain quality gates for source ingest concepts and relations."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from lucid.training.ingest_config import GENERAL_BROAD_SINGLE_TERMS
from lucid.training.source_context import (
    DEFINITION_JUNK_MARKERS,
    VENDOR_ARTIFACT_RE,
    is_renderable_definition_target,
)

TOKEN_RE = re.compile(r"[^a-z0-9_]+")

# Sentence-fragment / discourse subjects that are never concept nodes.
FRAGMENT_SUBJECTS = frozenset(
    {
        "advance",
        "advanced",
        "advantage",
        "another",
        "based",
        "behavior",
        "best",
        "beyond",
        "built",
        "capable",
        "complex",
        "content",
        "control",
        "current",
        "currently",
        "different",
        "electronic",
        "energy",
        "environment",
        "even",
        "fast",
        "field",
        "however",
        "important",
        "instead",
        "like",
        "outside",
        "possible",
        "process",
        "researcher",
        "result",
        "results",
        "today",
        "use",
        "used",
        "using",
        "work",
        "world",
        "demonstrated",
        "developed",
        "developer",
        "discover",
        "difference",
        "enough",
        "explore",
        "exploring",
        "first",
        "four",
        "critical",
        "electric",
    }
)

ORDINAL_OR_NUMBER_SUBJECTS = frozenset(
    {
        "first",
        "second",
        "third",
        "fourth",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    }
)

VENDOR_ORG_QUANTUM_RE = re.compile(
    r"^(?:azure|google|ibm|microsoft|aws|nist|best|current|important|control|like|quantum_utility)_quantum$"
)

# Generic academic / page-flow terms that should not become concept nodes.
GENERIC_SUBJECT_UNIGRAMS = frozenset(
    {
        "benchmark",
        "breakthrough",
        "calculation",
        "capability",
        "challenge",
        "chemistry",
        "collapse",
        "combined",
        "component",
        "computation",
        "computational",
        "concept",
        "contain",
        "contrast",
        "create",
    }
)

GENERIC_NGRAM_PARTS = frozenset(
    {
        "above",
        "below",
        "between",
        "characteristic",
        "critical",
        "during",
        "general",
        "high",
        "large",
        "level",
        "lower",
        "main",
        "many",
        "most",
        "much",
        "multiple",
        "new",
        "other",
        "same",
        "several",
        "small",
        "specific",
        "such",
        "temperature",
        "various",
        "very",
        "well",
        "remain",
        "remains",
    }
)

SUBJECT_PREFIX_BLOCK = frozenset(
    {
        "another",
        "based",
        "best",
        "control",
        "current",
        "important",
        "like",
        "possible",
        "quantum_utility",
    }
)

JUNK_TARGET_MARKERS = (
    *DEFINITION_JUNK_MARKERS,
    "no longer supported",
    "click here",
    "sign up",
    "learn more",
    "get started",
    "subscribe",
    "open_in_new",
    "table of contents",
    "read the report",
    "copyright",
)

JUNK_TARGET_WORDS = frozenset(
    {
        "try",
    }
)


def _target_has_junk_marker(lowered: str) -> bool:
    if any(marker in lowered for marker in JUNK_TARGET_MARKERS):
        return True
    return any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in JUNK_TARGET_WORDS)

MIN_TARGET_WORDS = {
    "type_of": 4,
    "property": 4,
    "uses": 4,
    "enables": 4,
    "capability": 4,
    "measurement": 3,
    "contrast": 3,
    "challenge": 3,
    "related_to": 4,
}


def normalize_key(value: object) -> str:
    clean = TOKEN_RE.sub("_", str(value or "").strip().lower())
    return "_".join(part for part in clean.split("_") if part)


def _concept_tokens(concept_id: str) -> list[str]:
    return [token for token in concept_id.split("_") if token]


def concept_id_has_repeated_parts(concept_id: str) -> bool:
    parts = _concept_tokens(concept_id)
    if len(parts) < 2:
        return False
    seen: set[str] = set()
    for part in parts:
        if len(part) <= 3:
            continue
        if part in seen:
            return True
        seen.add(part)
    return False


def concept_is_generic_ngram(concept_id: str) -> bool:
    parts = _concept_tokens(concept_id)
    if len(parts) < 3:
        return False
    generic_pool = (
        FRAGMENT_SUBJECTS
        | GENERIC_SUBJECT_UNIGRAMS
        | GENERIC_NGRAM_PARTS
        | GENERAL_BROAD_SINGLE_TERMS
    )
    if all(part in generic_pool for part in parts):
        return True
    return parts[0] in {"characteristic", "general", "specific", "various"}


def looks_like_verb_fragment(word: str) -> bool:
    if word in ORDINAL_OR_NUMBER_SUBJECTS:
        return True
    if word.endswith("ing") and len(word) < 12:
        return True
    if word.endswith("ed") and len(word) < 12:
        return True
    return False


def reject_concept_id(concept_id: str, *, broad_terms: frozenset[str] | None = None) -> str | None:
    key = normalize_key(concept_id)
    if not key:
        return "empty_concept_id"
    if VENDOR_ARTIFACT_RE.match(key) or VENDOR_ORG_QUANTUM_RE.match(key):
        return "vendor_artifact"
    if concept_id_has_repeated_parts(key):
        return "repeated_token_parts"
    if concept_is_generic_ngram(key):
        return "generic_ngram"
    tokens = _concept_tokens(key)
    if not tokens:
        return "empty_concept_id"
    if len(tokens) == 1:
        word = tokens[0]
        broad = broad_terms or frozenset(GENERAL_BROAD_SINGLE_TERMS)
        if word in FRAGMENT_SUBJECTS or word in GENERIC_SUBJECT_UNIGRAMS:
            return "fragment_subject"
        if looks_like_verb_fragment(word):
            return "verb_fragment"
        if word in broad:
            return "broad_single_term"
        if len(word) < 4:
            return "too_short_unigram"
    first = tokens[0]
    if first in FRAGMENT_SUBJECTS or first in SUBJECT_PREFIX_BLOCK:
        return "fragment_prefix"
    if len(tokens) >= 2 and tokens[-1] in FRAGMENT_SUBJECTS and tokens[0] in {"best", "current", "important", "control"}:
        return "vendor_ngram_tail"
    return None


def reject_relation(relation: dict[str, Any]) -> str | None:
    relation_type = str(relation.get("relation") or "")
    target = str(relation.get("target") or "").strip()
    if not target:
        return "empty_target"
    lowered = target.lower()
    if _target_has_junk_marker(lowered):
        return "junk_target_marker"
    if "..." in target:
        return "truncated_target"
    words = lowered.split()
    min_words = MIN_TARGET_WORDS.get(relation_type, 4)
    if len(words) < min_words:
        return "target_too_short"
    if relation_type in {"type_of", "property", "related_to"} and not is_renderable_definition_target(
        target,
        relation=relation_type,
    ):
        return "unrenderable_definition"
    return None


def is_valid_candidate_term(term: str, *, broad_terms: frozenset[str] | None = None) -> bool:
    return reject_concept_id(normalize_key(term), broad_terms=broad_terms) is None


def is_valid_subject_term(term: str, *, broad_terms: frozenset[str] | None = None) -> bool:
    return is_valid_candidate_term(term, broad_terms=broad_terms)


def passes_concept_support(concept: dict[str, Any], *, corpus_terms: frozenset[str] | None = None) -> str | None:
    concept_id = str(concept.get("concept_id") or "")
    if str(concept.get("branch_reason") or "") == "contradiction_split" or "__reading_" in concept_id:
        return None
    relations = [rel for rel in concept.get("relations") or [] if isinstance(rel, dict)]
    source_refs = [str(ref) for ref in concept.get("source_refs") or [] if str(ref)]
    tokens = _concept_tokens(concept_id)
    if len(tokens) >= 2:
        return None
    word = tokens[0] if tokens else ""
    if len(word) >= 8:
        return None
    if len(source_refs) >= 2:
        return None
    if len(relations) >= 2:
        return None
    if corpus_terms and (word in corpus_terms or concept_id.replace("_", " ") in corpus_terms):
        if len(relations) >= 1 and not reject_relation(relations[0]):
            return None
    return "insufficient_support"


def filter_concept_relations(concept: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter[str]]:
    kept: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for relation in concept.get("relations") or []:
        if not isinstance(relation, dict):
            continue
        reason = reject_relation(relation)
        if reason:
            reasons[reason] += 1
            continue
        kept.append(relation)
    return kept, reasons


def filter_concepts(
    concepts: list[dict[str, Any]],
    *,
    broad_terms: frozenset[str] | None = None,
    corpus_terms: frozenset[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    broad = broad_terms or frozenset(GENERAL_BROAD_SINGLE_TERMS)
    kept: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    relation_rejections: Counter[str] = Counter()

    for concept in concepts:
        concept_id = str(concept.get("concept_id") or "")
        id_reason = reject_concept_id(concept_id, broad_terms=broad)
        if id_reason:
            rejections[id_reason] += 1
            continue

        relations, rel_reasons = filter_concept_relations(concept)
        relation_rejections.update(rel_reasons)
        if not relations:
            rejections["no_usable_relations"] += 1
            continue

        candidate = dict(concept)
        candidate["relations"] = relations
        support_reason = passes_concept_support(candidate, corpus_terms=corpus_terms)
        if support_reason:
            rejections[support_reason] += 1
            continue

        extraction = dict(candidate.get("extraction") or {})
        extraction["quality_pass"] = True
        candidate["extraction"] = extraction
        kept.append(candidate)

    return kept, {
        "concepts_before_quality_filter": len(concepts),
        "concepts_after_quality_filter": len(kept),
        "concept_rejections": dict(rejections),
        "relation_rejections": dict(relation_rejections),
    }


# Discourse / page-flow subjects that passed first-pass filter but are not topic concepts.
RETENTION_DISCOURSE_SUBJECTS = frozenset(
    {
        "ability",
        "academic",
        "access",
        "accuracy",
        "accurate",
        "action",
        "actually",
        "alternative",
        "although",
        "apply",
        "article",
        "aspect",
        "assume",
        "assumption",
        "book",
        "certain",
        "cluster",
        "connection",
        "convergence",
        "correctly",
        "dataset",
        "dimension",
        "distribution",
        "dropout",
        "ensemble",
        "expectation",
        "forward",
        "function",
        "fundamental",
        "game",
        "general",
        "goal",
        "happen",
        "input",
        "interpretability",
        "intelligence",
        "language",
        "larger",
        "limitation",
        "linear",
        "long",
        "many",
        "mean",
        "monte",
        "network",
        "openai",
        "optimal",
        "order",
        "parameter",
        "player",
        "procedure",
        "randomly",
        "release",
        "require",
        "search",
        "significant",
        "since",
        "single",
        "sometime",
        "sparse",
        "special",
        "text",
        "though",
        "time",
        "transition",
        "vision",
        "vocabulary",
        "weight",
        "well",
        "whether",
        "word",
    }
)

_DOMAIN_TERM_PARTS = frozenset(
    {
        "learning",
        "network",
        "model",
        "attention",
        "transformer",
        "gradient",
        "boltzmann",
        "hopfield",
        "energy",
        "encoder",
        "decoder",
        "embedding",
        "neural",
        "machine",
        "loss",
        "layer",
        "training",
        "inference",
        "algorithm",
        "activation",
        "convolutional",
        "recurrent",
        "reinforcement",
        "generative",
        "language",
        "autoencoder",
        "diffusion",
        "markov",
        "monte",
        "carlo",
        "ising",
        "softmax",
        "backpropagation",
        "classification",
        "regression",
        "optimization",
    }
)


def reject_concept_retention(concept: dict[str, Any]) -> str | None:
    """Stricter gate for keeping concepts in a trained checkpoint."""
    concept_id = str(concept.get("concept_id") or "")
    if not concept_id:
        return "empty_concept_id"
    if "__reading_" in concept_id or str(concept.get("branch_reason") or "") == "contradiction_split":
        return "contradiction_branch"
    id_reason = reject_concept_id(concept_id)
    if id_reason:
        return id_reason
    if concept_id in RETENTION_DISCOURSE_SUBJECTS:
        return "discourse_subject"
    relations = [rel for rel in concept.get("relations") or [] if isinstance(rel, dict)]
    if not relations:
        return "no_usable_relations"
    source_refs = [str(ref) for ref in concept.get("source_refs") or [] if str(ref)]
    relation_count = len(relations)
    source_count = len(source_refs)
    if relation_count >= 3 and source_count >= 2:
        return None
    if relation_count >= 2 and source_count >= 1 and len(concept_id) >= 8:
        if "_" in concept_id:
            return None
        if any(part in _DOMAIN_TERM_PARTS for part in _concept_tokens(concept_id)):
            return None
    if relation_count >= 3:
        return None
    if relation_count >= 2 and len(concept_id) >= 12:
        return None
    return "insufficient_support"


def retain_concepts(concepts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    for concept in concepts:
        reason = reject_concept_retention(concept)
        if reason:
            rejections[reason] += 1
            continue
        kept.append(concept)
    return kept, {
        "concepts_before_retention": len(concepts),
        "concepts_after_retention": len(kept),
        "retention_rejections": dict(rejections),
    }
