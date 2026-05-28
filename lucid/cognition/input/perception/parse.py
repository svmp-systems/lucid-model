"""Parse and sanity-check LLM JSON → PerceptualEvidenceGraph."""

from __future__ import annotations

import json
import re
from typing import Any

from lucid.ir.common import Modality
from lucid.ir.perception import PerceptualEvidenceGraph
from lucid.ir.serde import from_dict

_FORBIDDEN = ("bank_sense", "trace_id", "task_type", "interpretation", "final_answer")


def graph_from_dict(data: dict[str, Any], *, modality: Modality) -> PerceptualEvidenceGraph:
    lower = json.dumps(data).lower()
    for token in _FORBIDDEN:
        if token in lower:
            raise ValueError(f"forbidden: {token}")
    for unit in data.get("candidate_units") or []:
        if re.match(r"^t[_-]", str(unit.get("unit_id", "")), re.I):
            raise ValueError("unit_id must not look like a trace id")
    graph: PerceptualEvidenceGraph = from_dict(data, PerceptualEvidenceGraph)
    if graph.provenance.modality is None:
        graph.provenance.modality = modality
    graph.provenance.extra["backend"] = "llm"
    return graph
