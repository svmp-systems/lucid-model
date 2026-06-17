"""Shared source-entity and vendor-cue maps for ingest, DMF, and binding."""

from __future__ import annotations

import re

SOURCE_ENTITY_BY_ARTICLE: dict[str, str] = {
    "google_quantum_ai_intro": "Google Quantum AI",
    "ibm_quantum_computing": "IBM Quantum",
    "microsoft_quantum_overview": "Microsoft Azure Quantum",
    "aws_quantum_computing": "AWS Quantum",
    "nist_quantum_explained": "NIST",
}

_RUNTIME_SOURCE_ENTITIES: dict[str, str] = {}


def register_source_entities(mapping: dict[str, str]) -> None:
    """Register per-run source entities discovered during ingest."""

    for source_id, entity in mapping.items():
        key = str(source_id or "").strip()
        value = str(entity or "").strip()
        if key and value:
            _RUNTIME_SOURCE_ENTITIES[key] = value


def clear_source_entities() -> None:
    _RUNTIME_SOURCE_ENTITIES.clear()


def source_entity_for_article(source_id: str) -> str:
    key = str(source_id or "").strip()
    if not key:
        return ""
    return _RUNTIME_SOURCE_ENTITIES.get(key) or SOURCE_ENTITY_BY_ARTICLE.get(key, "")

VENDOR_CUE_TO_SOURCE: dict[str, str] = {
    "google": "google_quantum_ai_intro",
    "ibm": "ibm_quantum_computing",
    "microsoft": "microsoft_quantum_overview",
    "aws": "aws_quantum_computing",
    "nist": "nist_quantum_explained",
}

VENDOR_ARTIFACT_RE = re.compile(
    r"^(?:google|ibm|microsoft|aws|nist)_quantum$"
    r"|^(?:make|computer|superconducting|ion|topological)_quantum$"
    r"|^computer_would$"
    r"|^classical_computer_need$"
)

MECHANISM_VERB_SURFACES = frozenset(
    {
        "utilizing",
        "utilize",
        "utilizes",
        "using",
        "uses",
        "use",
        "employing",
        "employs",
        "leveraging",
        "leverages",
    }
)

GERUND_TARGET_RE = re.compile(
    r"^(?:exploring|using|building|developing|creating|designing|"
    r"running|performing|achieving|enabling|allowing|supporting|"
    r"investigating|researching|working)\b",
    re.I,
)

VENDOR_DEFINITION_SENSE_SLOTS = frozenset({"quantum_sense", "google_sense"})

MECHANISM_RELATIONS = frozenset({"uses", "capability", "enables"})
DEFINITION_RELATIONS = frozenset({"type_of", "capability", "property"})
VENDOR_DEFINITION_RELATIONS = frozenset({"capability", "type_of", "property", "uses"})

DEFINITION_JUNK_MARKERS = (
    "join now",
    "case studies",
    "case study",
    "explore how",
    "history of",
    "click here",
    "sign up",
    "mercedes-benz",
    "mercedes benz",
    "boeing",
    "exxonmobil",
)

DEFINITION_FRAGMENT_MARKERS = (
    "black box",
    "some explaining to do",
    "maybe not for long",
    "experts warn",
    "not for long",
    "hallucinate",
    "evolve to solve",
    "extinction, experts",
)

MIN_DEFINITION_RENDER_SCORE = 0.35
MIN_DEFINITION_COMMIT_SCORE = 0.58

_BROKEN_LEAD_RE = re.compile(r"^(?:a|an|the)\s+['\"]")
_TRAILING_QUOTE_RE = re.compile(r"['\"]\s*$")


def is_renderable_definition_target(
    target: str,
    *,
    relation: str = "",
    concept_id: str = "",
) -> bool:
    cleaned = " ".join(str(target or "").strip().split())
    if len(cleaned) < 12:
        return False
    if GERUND_TARGET_RE.match(cleaned):
        return False
    lowered = cleaned.lower()
    if any(marker in lowered for marker in DEFINITION_JUNK_MARKERS):
        return False
    if any(marker in lowered for marker in DEFINITION_FRAGMENT_MARKERS):
        return False
    if _BROKEN_LEAD_RE.match(lowered) or _TRAILING_QUOTE_RE.search(cleaned):
        return False
    if cleaned.count('"') % 2 == 1:
        return False
    if "..." in cleaned or cleaned.endswith(" even"):
        return False
    if lowered.count(". ") >= 1 and relation in {"type_of", "is_a", "kind_of"}:
        return False
    words = lowered.split()
    if relation in {"type_of", "is_a", "kind_of"} and len(words) < 6:
        return False
    if concept_id and score_definition_target_for_concept(cleaned, concept_id) < MIN_DEFINITION_RENDER_SCORE:
        return False
    return True


def vendor_source_from_surfaces(surfaces: set[str]) -> str:
    for cue, source in VENDOR_CUE_TO_SOURCE.items():
        if cue in surfaces:
            return source
    return ""


def is_mechanism_query_surfaces(surfaces: set[str]) -> bool:
    if "quantum" not in surfaces:
        return False
    if surfaces & MECHANISM_VERB_SURFACES:
        return True
    return "how" in surfaces


def is_vendor_definition_query_surfaces(surfaces: set[str]) -> bool:
    if is_mechanism_query_surfaces(surfaces):
        return False
    if "quantum" not in surfaces:
        return False
    return any(cue in surfaces for cue in VENDOR_CUE_TO_SOURCE)


CONCEPT_TOPIC_ALIASES: dict[str, str] = {
    "ai": "artificial_intelligence",
    "a_i": "artificial_intelligence",
    "ml": "machine_learning",
    "llm": "large_language_model",
    "llms": "large_language_model",
    "nlp": "natural_language_processing",
    "dl": "deep_learning",
    "nn": "neural_network",
    "nns": "neural_network",
    "rl": "reinforcement_learning",
    "ebm": "energy_based_model",
    "ebms": "energy_based_model",
    "gpt": "generative_pre_trained_transformer",
}

DEFINITION_CONCEPT_PREFERENCES: dict[str, str] = {
    "transformer": "transformer_architecture",
}

FOLLOWUP_TOPIC_PRONOUNS = frozenset({"it", "this", "that", "these", "those", "they", "them"})

_BASIN_FACETS = ("definition", "mechanism", "challenge", "contrast")

_CROSS_SENSE_TARGET_MARKERS: dict[str, frozenset[str]] = {
    "transformer": frozenset({"bayesian", "causal", "markov", "belief network", "directed acyclic"}),
    "transformer_architecture": frozenset(
        {
            "bayesian",
            "hugging face",
            "library produced",
            "pretrained model",
            "positional encoding methods",
            "generally the same",
            "sublayers per",
            "partially masked",
            "reverse information flow",
        }
    ),
}

_ML_DEFINITION_HINTS = frozenset(
    {
        "attention",
        "neural",
        "network",
        "encoder",
        "decoder",
        "token",
        "embedding",
        "architecture",
        "foundation",
        "model",
        "layer",
        "parallel",
        "sequence",
        "hugging",
    }
)

_DEFINITION_QUERY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\s*what\s+is\s+(?:an?\s+)?(?P<topic>.+?)\s*$", re.I), "definition_query"),
    (re.compile(r"^\s*what\s+are\s+(?P<topic>.+?)\s*$", re.I), "definition_query"),
    (re.compile(r"^\s*(?:tell\s+me\s+about|describe|explain)\s+(?P<topic>.+?)\s*$", re.I), "definition_query"),
    (re.compile(r"^\s*(?:can\s+you\s+)?explain\s+(?P<topic>.+?)\s*$", re.I), "definition_query"),
    (re.compile(r"^\s*what\s+does\s+(?P<topic>.+?)\s+mean\s*$", re.I), "definition_query"),
    (re.compile(r"^\s*give\s+me\s+an?\s+overview\s+of\s+(?P<topic>.+?)\s*$", re.I), "definition_query"),
    (re.compile(r"^\s*i\s+want\s+to\s+know\s+about\s+(?P<topic>.+?)\s*$", re.I), "definition_query"),
    (re.compile(r"^\s*how\s+does\s+(?P<topic>.+?)\s+work\s*$", re.I), "mechanism_query"),
    (re.compile(r"^\s*how\s+do\s+(?P<topic>.+?)\s+work\s*$", re.I), "mechanism_query"),
    (re.compile(r"^\s*how\s+does\s+(?:it|this|that)\s+work\s*$", re.I), "mechanism_query"),
    (re.compile(r"^\s*how\s+do\s+(?:they|these|those)\s+work\s*$", re.I), "mechanism_query"),
)


def _normalize_topic_key(value: str) -> str:
    cleaned = re.sub(r"[^\w\s-]+", " ", str(value or "").strip().lower())
    return "_".join(cleaned.replace("-", " ").split())


def clean_query_topic(topic_surface: str) -> str:
    """Strip leading articles so 'a transformer' resolves like 'transformer'."""
    return _ARTICLE_PREFIX_RE.sub("", str(topic_surface or "").strip()).strip()


def resolve_concept_topic(value: str) -> str:
    """Map surface cue or alias to canonical concept id when known."""

    key = _normalize_topic_key(value)
    if not key:
        return ""
    if key in CONCEPT_TOPIC_ALIASES:
        return CONCEPT_TOPIC_ALIASES[key]
    return key


def preferred_definition_concept(concept_id: str) -> str:
    key = _normalize_topic_key(concept_id)
    return DEFINITION_CONCEPT_PREFERENCES.get(key, key)


def extract_session_concept_topics(session_context: dict[str, object] | None) -> list[str]:
    if not isinstance(session_context, dict):
        return []
    topics: list[str] = []
    seen: set[str] = set()
    for turn in session_context.get("recent_turns") or []:
        if not isinstance(turn, dict):
            continue
        user = str(turn.get("user_input") or "").strip()
        if not user:
            continue
        parsed = parse_concept_query(user)
        if parsed is None:
            continue
        concept_id = parsed[1]
        if concept_id in seen:
            continue
        seen.add(concept_id)
        topics.append(concept_id)
    return topics


def parse_concept_basin_id(basin_id: str) -> tuple[str, str] | None:
    bid = str(basin_id or "").strip().lower()
    if not bid.startswith("b_"):
        return None
    body = bid[2:]
    for facet in _BASIN_FACETS:
        suffix = f"_{facet}"
        if body.endswith(suffix):
            return body[: -len(suffix)], facet
    return None


_DEFINITION_NEGATIVE_MARKERS = frozenset(
    {
        "ethical implications",
        "societal",
        "protests",
        "legal actions",
        "pause ai",
        "governments",
        "black box",
        "some explaining to do",
    }
)
_DEFINITION_USAGE_MARKERS = frozenset(
    {
        "de facto",
        "now used",
        "used alongside",
        "across a range",
        "ongoing ai boom",
        "choice for building",
        "increasingly focused",
        "had success in other applications",
        "constructed to calculate",
        "calculate output tokens",
        "such as: disaster",
        "evaluating chess",
        "in the context of",
        "often discussed",
    }
)
_DEFINITION_KIND_MARKERS = frozenset(
    {
        "architecture",
        "neural network",
        "attention mechanism",
        "attention to",
        "deep learning model",
        "sequence",
        "encoder",
        "decoder",
        "parallelizable",
        "recurrent",
        "language model",
    }
)
_ARTICLE_PREFIX_RE = re.compile(r"^(?:a|an|the)\s+", re.I)


def is_cross_sense_target(target: str, concept_id: str) -> bool:
    cleaned = " ".join(str(target or "").strip().split()).lower()
    if not cleaned:
        return False
    markers = _CROSS_SENSE_TARGET_MARKERS.get(_normalize_topic_key(concept_id), frozenset())
    return any(marker in cleaned for marker in markers)


def is_mechanism_like_target(target: str) -> bool:
    cleaned = " ".join(str(target or "").strip().split()).lower()
    if len(cleaned.split()) < 6:
        return False
    hints = (
        "attention",
        "recurrent",
        "token",
        "layer",
        "encoder",
        "decoder",
        "self-attention",
        "positional",
        "parallel",
        "sequence",
        "matrix",
        "vector",
    )
    return any(hint in cleaned for hint in hints)


def score_definition_target_for_concept(target: str, concept_id: str) -> float:
    cleaned = " ".join(str(target or "").strip().split()).lower()
    if not cleaned:
        return -1.0
    score = min(len(cleaned) / 120.0, 0.4)
    if any(marker in cleaned for marker in _DEFINITION_NEGATIVE_MARKERS):
        score -= 1.5
    if any(marker in cleaned for marker in _DEFINITION_USAGE_MARKERS):
        score -= 1.2
    if any(marker in cleaned for marker in _DEFINITION_KIND_MARKERS):
        score += 0.55
    if any(token in cleaned for token in ("field", "discipline", "technology", "science", "engineering", "system")):
        score += 0.6
    if cleaned.startswith(("a ", "an ", "the ")):
        score += 0.15
    if cleaned.startswith(("still ", "of particular", "an emergent", "an implementation")):
        score -= 0.8
    if cleaned.startswith("can "):
        score -= 0.5

    concept_key = _normalize_topic_key(concept_id)
    concept_tokens = {token for token in concept_key.split("_") if len(token) > 3}
    blocked = _CROSS_SENSE_TARGET_MARKERS.get(concept_key, frozenset())
    if blocked and any(marker in cleaned for marker in blocked):
        score -= 1.5
    if concept_tokens and any(token in cleaned for token in concept_tokens):
        score += 0.35
    if any(term in cleaned for term in _ML_DEFINITION_HINTS):
        score += 0.25
    if cleaned.count('"') >= 2 or cleaned.count("'") >= 2:
        score -= 0.4
    return score


def parse_concept_query(text: str) -> tuple[str, str, str] | None:
    """Return (topic_surface, concept_id, frame_type) for general concept questions."""

    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return None
    for pattern, frame_type in _DEFINITION_QUERY_PATTERNS:
        match = pattern.match(cleaned)
        if not match:
            continue
        topic_surface = ""
        if "topic" in match.groupdict() and match.group("topic") is not None:
            topic_surface = clean_query_topic(" ".join(match.group("topic").strip().split()))
        concept_id = resolve_concept_topic(topic_surface) if topic_surface else ""
        if frame_type == "mechanism_query" and concept_id:
            concept_id = preferred_definition_concept(concept_id)
        if topic_surface and _normalize_topic_key(topic_surface) in FOLLOWUP_TOPIC_PRONOUNS:
            return None
        if not concept_id and frame_type == "mechanism_query" and topic_surface == "":
            return None
        if not concept_id:
            return None
        return topic_surface or concept_id.replace("_", " "), concept_id, frame_type
    return None


def parse_concept_query_with_context(
    text: str,
    session_context: dict[str, object] | None,
) -> tuple[str, str, str] | None:
    parsed = parse_concept_query(text)
    if parsed is not None:
        return parsed
    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return None
    for pattern, frame_type in _DEFINITION_QUERY_PATTERNS:
        match = pattern.match(cleaned)
        if not match:
            continue
        topic_surface = ""
        if "topic" in match.groupdict() and match.group("topic") is not None:
            topic_surface = " ".join(str(match.group("topic") or "").strip().split())
        needs_context = (
            not topic_surface
            or _normalize_topic_key(topic_surface) in FOLLOWUP_TOPIC_PRONOUNS
        )
        if not needs_context:
            continue
        topics = extract_session_concept_topics(session_context)
        if not topics:
            return None
        concept_id = topics[-1]
        return concept_id.replace("_", " "), concept_id, frame_type
    return None


def is_concept_definition_query(*, raw_text: str = "", surfaces: set[str] | None = None) -> bool:
    if parse_concept_query(raw_text):
        return True
    if not surfaces:
        return False
    for surface in surfaces:
        resolved = resolve_concept_topic(surface)
        if resolved and resolved != surface:
            return True
    return False


def concept_definition_primary_basin(
    candidate_basin_states: list,
    concept_id: str,
    *,
    frame_type: str = "definition_query",
) -> str:
    concept_key = _normalize_topic_key(concept_id)
    if not concept_key or concept_key in FOLLOWUP_TOPIC_PRONOUNS:
        return ""
    facet = "mechanism" if frame_type == "mechanism_query" else "definition"
    preferred = preferred_definition_concept(concept_key) if facet == "definition" else concept_key
    lookup_keys = [preferred]
    if preferred != concept_key:
        lookup_keys.append(concept_key)
    ranked: list[tuple[int, float, str]] = []
    for state in candidate_basin_states:
        basin_id = str(getattr(state, "basin_id", "") or "")
        if not basin_id or is_speech_basin(basin_id, state):
            continue
        parsed = parse_concept_basin_id(basin_id)
        if parsed is None:
            continue
        basin_concept, basin_facet = parsed
        if basin_facet != facet:
            continue
        priority = 0
        if basin_concept == lookup_keys[0]:
            priority = 3
        elif len(lookup_keys) > 1 and basin_concept == lookup_keys[1]:
            priority = 2
        else:
            continue
        energy = float(getattr(state, "energy", 0.0) or 0.0)
        ranked.append((priority, energy, basin_id))
    if not ranked:
        return ""
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return ranked[0][2]


def is_speech_basin(basin_id: str, state: object | None = None) -> bool:
    bid = str(basin_id or "").strip()
    if bid.startswith("b_basic_") or bid in {"social_speech"}:
        return True
    if state is not None:
        payload = getattr(state, "quantized_payload", None)
        if isinstance(payload, dict) and str(payload.get("facet") or "").strip().lower() == "speech":
            return True
    return False


def is_term_definition_query_surfaces(surfaces: set[str]) -> bool:
    if is_mechanism_query_surfaces(surfaces):
        return False
    if is_vendor_definition_query_surfaces(surfaces):
        return True
    if {"quantum", "computing"} <= surfaces or "quantum_computing" in surfaces:
        return True
    if {"quantum", "computer"} <= surfaces or "quantum_computer" in surfaces:
        return True
    resolved = {resolve_concept_topic(surface) for surface in surfaces}
    if resolved - surfaces:
        return True
    return False


def vendor_frame_sense_unresolved_ok(unresolved_slot_names: list[str] | tuple[str, ...]) -> bool:
    unresolved = {str(name) for name in unresolved_slot_names if str(name).strip()}
    return not unresolved or unresolved <= VENDOR_DEFINITION_SENSE_SLOTS

VENDOR_REDIRECT_TARGETS: dict[str, list[str]] = {
    "google_quantum_ai_intro": ["quantum_computer", "quantum_computing", "quantum_algorithm"],
    "ibm_quantum_computing": ["quantum_computer", "quantum_computing"],
    "microsoft_quantum_overview": ["quantum_computer", "quantum_computing"],
    "aws_quantum_computing": ["quantum_computer", "quantum_computing", "quantum_circuit", "superconducting_qubit"],
    "nist_quantum_explained": ["quantum_computer", "quantum_computing", "qubit", "superconducting_qubit"],
}
