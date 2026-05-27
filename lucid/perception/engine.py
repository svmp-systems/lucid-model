"""Route perception by modality and configured backend."""

from __future__ import annotations

from typing import Any

from lucid.ir.common import Modality
from lucid.ir.perception import PerceptionInput, PerceptualEvidenceGraph

from lucid.perception.config import PerceptionConfig
from lucid.perception.llm import LlmPerceptionAdapter
from lucid.perception.protocol import PerceptionAdapter
from lucid.perception.rule_grid import RuleGridPerceptionAdapter
from lucid.perception.rule_text import RuleTextPerceptionAdapter

_text_rule = RuleTextPerceptionAdapter()
_grid_rule = RuleGridPerceptionAdapter()


def build_adapter(config: PerceptionConfig | None = None) -> PerceptionAdapter:
    cfg = config or PerceptionConfig.from_env()
    if cfg.backend == "llm":
        return LlmPerceptionAdapter(cfg)
    if cfg.backend != "rule":
        raise ValueError(f"unknown perception backend: {cfg.backend!r} (use rule or llm)")
    return _RouterRuleAdapter()


class _RouterRuleAdapter:
    adapter_id = "rule_router_v1"

    def perceive(self, inp: PerceptionInput, *, context: object = None) -> PerceptualEvidenceGraph:
        modality = inp.modality if isinstance(inp.modality, Modality) else Modality(str(inp.modality))
        if modality == Modality.GRID:
            return _grid_rule.perceive(inp, context=context)
        return _text_rule.perceive(inp, context=context)


def perceive(inp: PerceptionInput, *, context: Any = None, config: PerceptionConfig | None = None) -> PerceptualEvidenceGraph:
    adapter = build_adapter(config)
    return adapter.perceive(inp, context=context)
