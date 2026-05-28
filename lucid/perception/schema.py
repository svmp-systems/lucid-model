"""PerceptualEvidenceGraph JSON Schema — used on API *output*, not in the prompt."""

from __future__ import annotations

from typing import Any

from lucid.ir.perception import PerceptionInput

# Model output only (provenance is attached after parse).
_DEFINITIONS: dict[str, Any] = {
    "CandidateUnit": {
        "type": "object",
        "properties": {
            "unit_id": {"type": "string"},
            "surface": {"type": "string"},
            "kind_hint": {"type": "string"},
            "type_hints": {"type": "array", "items": {"type": "string"}},
            "feature_signature": {"type": "string"},
            "position_or_time": {"type": "string"},
            "confidence": {"type": "number"},
            "salience": {"type": "number"},
            "uncertainty": {"type": ["string", "null"]},
        },
        "required": ["unit_id"],
        "additionalProperties": False,
    },
    "CandidateRegion": {
        "type": "object",
        "properties": {
            "region_id": {"type": "string"},
            "role_hint": {"type": "string"},
            "member_unit_ids": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "uncertainty": {"type": ["string", "null"]},
        },
        "required": ["region_id"],
        "additionalProperties": False,
    },
    "CandidateContainer": {
        "type": "object",
        "properties": {
            "container_id": {"type": "string"},
            "kind_hint": {"type": "string"},
            "border_signature": {"type": "string"},
            "interior_region_id": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["container_id"],
        "additionalProperties": False,
    },
    "CandidateMarker": {
        "type": "object",
        "properties": {
            "marker_id": {"type": "string"},
            "surface": {"type": "string"},
            "marker_type_hints": {"type": "array", "items": {"type": "string"}},
            "possible_target_unit_ids": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
        },
        "required": ["marker_id"],
        "additionalProperties": False,
    },
    "ArrangementHint": {
        "type": "object",
        "properties": {
            "hint_type": {"type": "string"},
            "source_unit_id": {"type": "string"},
            "target_unit_id": {"type": "string"},
            "weight": {"type": "number"},
        },
        "required": ["hint_type", "source_unit_id", "target_unit_id"],
        "additionalProperties": False,
    },
    "ChangeHint": {
        "type": "object",
        "properties": {
            "change_type": {"type": "string"},
            "before_unit_id": {"type": "string"},
            "after_unit_id": {"type": "string"},
            "weight": {"type": "number"},
            "details": {"type": "object", "additionalProperties": True},
        },
        "required": ["change_type"],
        "additionalProperties": False,
    },
    "GroupingHint": {
        "type": "object",
        "properties": {
            "group_id": {"type": "string"},
            "member_unit_ids": {"type": "array", "items": {"type": "string"}},
            "grouping_reason": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["group_id"],
        "additionalProperties": False,
    },
    "ReferenceHint": {
        "type": "object",
        "properties": {
            "source_unit_id": {"type": "string"},
            "target_unit_id": {"type": "string"},
            "reference_type": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["source_unit_id", "target_unit_id"],
        "additionalProperties": False,
    },
    "UncertaintyFlag": {
        "type": "object",
        "properties": {
            "target_id": {"type": "string"},
            "uncertainty_type": {"type": "string"},
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["target_id", "uncertainty_type"],
        "additionalProperties": False,
    },
}

_LIST_KEYS = (
    "candidate_units",
    "candidate_regions",
    "candidate_containers",
    "candidate_markers",
    "arrangement_hints",
    "change_hints",
    "grouping_hints",
    "reference_hints",
    "uncertainty_flags",
)

PERCEPTUAL_EVIDENCE_GRAPH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidate_units": {"type": "array", "items": {"$ref": "#/$defs/CandidateUnit"}},
        "candidate_regions": {"type": "array", "items": {"$ref": "#/$defs/CandidateRegion"}},
        "candidate_containers": {"type": "array", "items": {"$ref": "#/$defs/CandidateContainer"}},
        "candidate_markers": {"type": "array", "items": {"$ref": "#/$defs/CandidateMarker"}},
        "arrangement_hints": {"type": "array", "items": {"$ref": "#/$defs/ArrangementHint"}},
        "change_hints": {"type": "array", "items": {"$ref": "#/$defs/ChangeHint"}},
        "grouping_hints": {"type": "array", "items": {"$ref": "#/$defs/GroupingHint"}},
        "reference_hints": {"type": "array", "items": {"$ref": "#/$defs/ReferenceHint"}},
        "uncertainty_flags": {"type": "array", "items": {"$ref": "#/$defs/UncertaintyFlag"}},
    },
    "required": list(_LIST_KEYS),
    "additionalProperties": False,
    "$defs": _DEFINITIONS,
}

_SYSTEM_PROMPT = (
    "Perception stage: extract surface evidence from the user message. "
    "For text payloads you MUST populate candidate_units: one entry per substantive word "
    "(nouns, verbs, adjectives, adverbs) and multi-word phrases when useful. "
    "Add candidate_markers for function words that structure the sentence (in, while, which, and). "
    "Add uncertainty_flags for ambiguous terms (e.g. bank). "
    "Other list fields stay [] unless the input clearly supports them. "
    "An all-empty graph (every list []) is invalid when payload text is non-empty. "
    "unit_id like u_bank; never trace ids or resolved senses. Do not interpret or answer."
)

_EMPTY_GRAPH_RETRY = (
    "Your last response had empty candidate_units and candidate_markers but the payload "
    "contains text. Re-analyze the payload and return candidate_units for each substantive "
    "word (e.g. for 'go to the bank' include u_go, u_bank at minimum). Keep other lists [] "
    "unless supported. Flag polysemy on bank."
)


def empty_graph_template() -> dict[str, Any]:
    return {key: [] for key in _LIST_KEYS}


def _slug(surface: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", surface.lower()).strip("_")[:32] or "span"


def _coerce_list_items(key: str, items: list[Any]) -> list[dict[str, Any]]:
    """Coerce model output: each list element must be an object (strings → minimal objects)."""
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
            continue
        if isinstance(item, str) and item.strip():
            text = item.strip()
            low = _slug(text)
            if key == "candidate_units":
                out.append({"unit_id": f"u_{low}", "surface": text, "kind_hint": "span"})
            elif key == "candidate_markers":
                out.append({"marker_id": f"m_{low}", "surface": text})
            elif key == "uncertainty_flags":
                out.append(
                    {
                        "target_id": f"u_{low}",
                        "uncertainty_type": "polysemy",
                        "severity": "medium",
                    }
                )
            else:
                out.append({"id": low, "surface": text})
            continue
        raise ValueError(f"{key} items must be objects or strings, got {type(item).__name__}")
    return out


def normalize_graph_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Post-output: ensure all list fields exist; coerce sloppy model shapes."""
    if not isinstance(data, dict):
        raise ValueError("graph must be a JSON object")
    out = empty_graph_template()
    for key in _LIST_KEYS:
        value = data.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            raise ValueError(f"{key} must be an array")
        out[key] = _coerce_list_items(key, value)
    return out


def build_system_prompt() -> str:
    """Short task instructions only — schema is enforced via response_format."""
    return _SYSTEM_PROMPT


def graph_has_text_evidence(graph: object) -> bool:
    """True if graph contains at least one unit or marker."""
    if isinstance(graph, dict):
        units = graph.get("candidate_units") or []
        markers = graph.get("candidate_markers") or []
    else:
        units = getattr(graph, "candidate_units", None) or []
        markers = getattr(graph, "candidate_markers", None) or []
    return bool(units or markers)


def build_user_message(inp: PerceptionInput) -> str:
    import json

    modality = inp.modality.value if hasattr(inp.modality, "value") else str(inp.modality)
    body: dict[str, Any] = {
        "modality": modality,
        "payload": inp.raw_payload,
        "task": "emit PerceptualEvidenceGraph JSON; candidate_units must not be empty for non-empty text",
    }
    if modality == "text" and isinstance(inp.raw_payload, str):
        body["text_to_analyze"] = inp.raw_payload
    if inp.prior_context:
        body["prior_context"] = inp.prior_context
    if inp.task_intent_hint is not None:
        body["task_intent_hint"] = (
            inp.task_intent_hint.value
            if hasattr(inp.task_intent_hint, "value")
            else str(inp.task_intent_hint)
        )
    return json.dumps(body, ensure_ascii=False)


def empty_graph_retry_message() -> str:
    return _EMPTY_GRAPH_RETRY


def build_messages(inp: PerceptionInput) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_message(inp)},
    ]


def structured_response_format() -> dict[str, Any]:
    """API output constraint: PerceptualEvidenceGraph JSON Schema."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "perceptual_evidence_graph",
            "strict": True,
            "schema": PERCEPTUAL_EVIDENCE_GRAPH_SCHEMA,
        },
    }


def json_object_response_format() -> dict[str, Any]:
    """Weaker fallback when the API rejects json_schema."""
    return {"type": "json_object"}
