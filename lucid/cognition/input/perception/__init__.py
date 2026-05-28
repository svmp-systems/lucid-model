"""Perception: raw input → PerceptualEvidenceGraph (evidence only, no meaning)."""

from __future__ import annotations

from typing import Any

from lucid.ir.common import Modality
from lucid.ir.perception import PerceptionInput, PerceptualEvidenceGraph

from lucid.perception.config import PerceptionConfig
from lucid.perception.llm import perceive_llm
from lucid.perception.parse import graph_from_dict
from lucid.perception.rule import perceive_grid, perceive_text


def perceive(
    inp: PerceptionInput,
    *,
    config: PerceptionConfig | None = None,
    context: Any = None,
) -> PerceptualEvidenceGraph:
    cfg = config or PerceptionConfig.from_env()
    modality = inp.modality if isinstance(inp.modality, Modality) else Modality(str(inp.modality))
    if cfg.backend == "llm":
        return perceive_llm(inp, cfg)
    if modality == Modality.GRID:
        return perceive_grid(inp)
    return perceive_text(inp)


__all__ = ["PerceptionConfig", "perceive", "graph_from_dict"]
