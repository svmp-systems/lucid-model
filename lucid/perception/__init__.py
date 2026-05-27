"""Perception stage — model or rule adapters emit PerceptualEvidenceGraph only."""

from lucid.perception.config import PerceptionConfig
from lucid.perception.engine import build_adapter, perceive
from lucid.perception.validator import parse_graph_dict, sanitize_graph_dict, validate_graph_dict

__all__ = [
    "PerceptionConfig",
    "build_adapter",
    "perceive",
    "parse_graph_dict",
    "sanitize_graph_dict",
    "validate_graph_dict",
]
