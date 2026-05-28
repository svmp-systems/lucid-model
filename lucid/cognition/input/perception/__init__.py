"""Perception: raw input → PerceptualEvidenceGraph (LLM by default)."""

from __future__ import annotations

from typing import Any

from lucid.ir.common import Modality
from lucid.ir.perception import PerceptionInput, PerceptualEvidenceGraph

from lucid.cognition.input.perception.config import PerceptionConfig
from lucid.cognition.input.perception.llm import perceive_llm
from lucid.cognition.input.perception.rule import perceive_grid, perceive_text
from lucid.cognition.input.perception.schema import (
    PERCEPTUAL_EVIDENCE_GRAPH_SCHEMA,
    compact_graph,
    empty_graph_template,
    graph_from_dict,
    normalize_graph_dict,
    structured_response_format,
    to_compact_json,
)


def perceive(
    inp: PerceptionInput,
    *,
    config: PerceptionConfig | None = None,
    context: Any = None,
) -> PerceptualEvidenceGraph:
    cfg = config or PerceptionConfig.from_env()
    modality = inp.modality if isinstance(inp.modality, Modality) else Modality(str(inp.modality))
    if cfg.backend == "llm":
        return perceive_llm(inp, cfg, context=context)
    if modality == Modality.GRID:
        return perceive_grid(inp)
    return perceive_text(inp)

__all__ = [
    "PerceptionConfig",
    "perceive",
    "graph_from_dict",
    "compact_graph",
    "to_compact_json",
    "PERCEPTUAL_EVIDENCE_GRAPH_SCHEMA",
    "empty_graph_template",
    "normalize_graph_dict",
    "structured_response_format",
]
