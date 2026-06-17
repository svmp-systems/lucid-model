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


def is_renderable_definition_target(target: str, *, relation: str = "") -> bool:
    cleaned = " ".join(str(target or "").strip().split())
    if len(cleaned) < 12:
        return False
    if GERUND_TARGET_RE.match(cleaned):
        return False
    lowered = cleaned.lower()
    if any(marker in lowered for marker in DEFINITION_JUNK_MARKERS):
        return False
    if "..." in cleaned or cleaned.endswith(" even"):
        return False
    if lowered.count(". ") >= 1 and relation in {"type_of", "is_a", "kind_of"}:
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


def is_term_definition_query_surfaces(surfaces: set[str]) -> bool:
    if is_mechanism_query_surfaces(surfaces):
        return False
    if is_vendor_definition_query_surfaces(surfaces):
        return True
    if {"quantum", "computing"} <= surfaces or "quantum_computing" in surfaces:
        return True
    if {"quantum", "computer"} <= surfaces or "quantum_computer" in surfaces:
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
