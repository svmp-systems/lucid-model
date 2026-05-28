"""Parse and sanity-check LLM JSON → PerceptualEvidenceGraph."""

from __future__ import annotations

import json
import re
from typing import Any

from lucid.ir.common import Modality
from lucid.ir.perception import PerceptualEvidenceGraph
from lucid.ir.serde import from_dict

from lucid.cognition.input.perception.schema import normalize_graph_dict

_FORBIDDEN = ("bank_sense", "trace_id", "task_type", "interpretation", "final_answer")


def graph_from_dict(data: dict[str, Any], *, modality: Modality) -> PerceptualEvidenceGraph:
    normalized = normalize_graph_dict(data)
    lower = json.dumps(normalized).lower()
    for token in _FORBIDDEN:
        if token in lower:
            raise ValueError(f"forbidden: {token}")
    for unit in normalized.get("candidate_units") or []:
        uid = unit.get("unit_id", "") if isinstance(unit, dict) else ""
        if re.match(r"^t[_-]", str(uid), re.I):
            raise ValueError("unit_id must not look like a trace id")
    graph: PerceptualEvidenceGraph = from_dict(normalized, PerceptualEvidenceGraph)
    if graph.provenance.modality is None:
        graph.provenance.modality = modality
    graph.provenance.extra["backend"] = "llm"
    return graph
