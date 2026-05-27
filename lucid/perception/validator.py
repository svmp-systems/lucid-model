"""Validate and sanitize model output into a legal PerceptualEvidenceGraph."""

from __future__ import annotations

import re
from typing import Any

from lucid.ir.common import Modality, UncertaintySeverity
from lucid.ir.perception import PerceptualEvidenceGraph
from lucid.ir.serde import from_dict, to_dict

# Keys that imply meaning commitment or downstream semantics — never allowed.
_FORBIDDEN_KEYS = frozenset(
    {
        "trace_id",
        "trace_ids",
        "basin_id",
        "basin_ids",
        "interpretation",
        "meaning",
        "final_answer",
        "task_type",
        "rule_family",
        "bank_sense",
        "sense",
        "committed_state",
        "lucidity_target",
    }
)

# unit_id values like t_money are learned trace IDs, not surface units.
_TRACE_ID_UNIT = re.compile(r"^t[_-]", re.I)

_POLYSEMY_SURFACES = frozenset({"bank", "bark", "match", "spring", "file", "lead"})


def _walk_forbidden(obj: Any, path: str = "") -> list[str]:
    errors: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_l = str(key).lower()
            if key_l in _FORBIDDEN_KEYS or key_l.endswith("_sense"):
                errors.append(f"forbidden key at {path}.{key}")
            errors.extend(_walk_forbidden(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            errors.extend(_walk_forbidden(item, f"{path}[{i}]"))
    return errors


def sanitize_graph_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown top-level keys; keep only PerceptualEvidenceGraph fields."""

    allowed = {
        "candidate_units",
        "candidate_regions",
        "candidate_containers",
        "candidate_markers",
        "arrangement_hints",
        "change_hints",
        "grouping_hints",
        "reference_hints",
        "uncertainty_flags",
        "provenance",
    }
    out: dict[str, Any] = {}
    for key in allowed:
        if key in data:
            out[key] = data[key]
    if "provenance" in out and isinstance(out["provenance"], dict):
        prov = dict(out["provenance"])
        prov.pop("raw_model_response", None)
        out["provenance"] = prov
    return out


def validate_graph_dict(data: dict[str, Any]) -> list[str]:
    errors = _walk_forbidden(data)
    units = data.get("candidate_units") or []
    if not isinstance(units, list):
        errors.append("candidate_units must be a list")
        return errors
    for unit in units:
        if not isinstance(unit, dict):
            continue
        uid = str(unit.get("unit_id", ""))
        if _TRACE_ID_UNIT.match(uid):
            errors.append(f"unit_id looks like trace id: {uid}")
        surface = str(unit.get("surface", "")).lower()
        hints = unit.get("type_hints") or []
        if isinstance(hints, list):
            for hint in hints:
                h = str(hint).lower()
                if "financial" in h or "river" in h or "semantic" in h:
                    errors.append(f"type_hints commit meaning on {uid}: {hint}")
        if surface in _POLYSEMY_SURFACES:
            flags = data.get("uncertainty_flags") or []
            flagged = any(
                isinstance(f, dict)
                and str(f.get("target_id", "")) == uid
                or str(f.get("target_id", "")) == surface
                for f in flags
            )
            if not flagged:
                errors.append(f"polysemy surface '{surface}' missing uncertainty_flag")
    return errors


def parse_graph_dict(data: dict[str, Any], *, modality: Modality) -> PerceptualEvidenceGraph:
    cleaned = sanitize_graph_dict(data)
    errors = validate_graph_dict(cleaned)
    if errors:
        raise ValueError("; ".join(errors[:8]))
    graph: PerceptualEvidenceGraph = from_dict(cleaned, PerceptualEvidenceGraph)
    if graph.provenance.modality is None:
        graph.provenance.modality = modality
    return graph


def merge_provenance(
    graph: PerceptualEvidenceGraph,
    *,
    adapter_version: str,
    segmentation_pass_id: str,
    extra: dict[str, Any] | None = None,
) -> None:
    graph.provenance.adapter_version = adapter_version
    graph.provenance.segmentation_pass_id = segmentation_pass_id
    if extra:
        graph.provenance.extra.update(extra)
