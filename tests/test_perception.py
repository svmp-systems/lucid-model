from __future__ import annotations

import pytest

from lucid.ir.common import Modality
from lucid.ir.perception import PerceptionInput
from lucid.perception.config import PerceptionConfig
from lucid.perception.engine import perceive
from lucid.perception.validator import parse_graph_dict, validate_graph_dict


def test_rule_text_extracts_spans_and_polysemy_flag() -> None:
    text = "Alex found money while kayaking and later put it in the bank."
    inp = PerceptionInput(raw_payload=text, modality=Modality.TEXT)
    graph = perceive(inp, config=PerceptionConfig(backend="rule"))

    surfaces = {u.surface.lower() for u in graph.candidate_units}
    assert "bank" in surfaces or any("bank" in u.surface for u in graph.candidate_units)
    assert graph.uncertainty_flags
    assert any("bank" in f.target_id or "bank" in str(f.uncertainty_type) for f in graph.uncertainty_flags)


def test_rule_grid_emits_change_hints() -> None:
    inp = PerceptionInput(
        raw_payload={"input": [[0, 1, 0], [0, 0, 0]], "output": [[0, 0, 1], [0, 0, 0]]},
        modality=Modality.GRID,
    )
    graph = perceive(inp, config=PerceptionConfig(backend="rule"))
    assert graph.candidate_units
    assert graph.change_hints


def test_validator_rejects_trace_ids() -> None:
    bad = {
        "candidate_units": [{"unit_id": "t_money", "surface": "money", "kind_hint": "noun"}],
        "uncertainty_flags": [],
    }
    errors = validate_graph_dict(bad)
    assert any("trace" in e for e in errors)


def test_validator_rejects_semantic_type_hints() -> None:
    bad = {
        "candidate_units": [
            {
                "unit_id": "u_bank",
                "surface": "bank",
                "kind_hint": "noun",
                "type_hints": ["financial_institution"],
            }
        ],
        "uncertainty_flags": [{"target_id": "u_bank", "uncertainty_type": "polysemy", "severity": "medium"}],
    }
    errors = validate_graph_dict(bad)
    assert any("meaning" in e or "financial" in e for e in errors)


def test_parse_graph_dict_roundtrip() -> None:
    data = {
        "candidate_units": [
            {"unit_id": "u_bank", "surface": "bank", "kind_hint": "noun_span", "confidence": 0.9}
        ],
        "uncertainty_flags": [
            {"target_id": "u_bank", "uncertainty_type": "polysemy_surface_form", "severity": "medium"}
        ],
    }
    graph = parse_graph_dict(data, modality=Modality.TEXT)
    assert graph.candidate_units[0].surface == "bank"


def test_llm_backend_requires_api_key() -> None:
    inp = PerceptionInput(raw_payload="hello", modality=Modality.TEXT)
    cfg = PerceptionConfig(backend="llm", api_key="")
    with pytest.raises(ValueError, match="API_KEY"):
        perceive(inp, config=cfg)
