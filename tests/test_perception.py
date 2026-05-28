from __future__ import annotations

import json

import pytest

from lucid.ir.common import Modality
from lucid.ir.perception import PerceptionInput
from lucid.perception import PerceptionConfig, empty_graph_template, graph_from_dict, perceive
from lucid.perception.schema import (
    PERCEPTUAL_EVIDENCE_GRAPH_SCHEMA,
    build_system_prompt,
    normalize_graph_dict,
    structured_response_format,
)


def test_rule_text_spans_and_polysemy() -> None:
    inp = PerceptionInput(
        raw_payload="Alex found money and put it in the bank.",
        modality=Modality.TEXT,
    )
    graph = perceive(inp, config=PerceptionConfig(backend="rule"))
    assert any(u.surface.lower() == "bank" for u in graph.candidate_units)
    assert graph.uncertainty_flags


def test_rule_grid_change_hints() -> None:
    inp = PerceptionInput(
        raw_payload={"input": [[0, 1, 0], [0, 0, 0]], "output": [[0, 0, 1], [0, 0, 0]]},
        modality=Modality.GRID,
    )
    graph = perceive(inp, config=PerceptionConfig(backend="rule"))
    assert graph.candidate_units
    assert graph.change_hints


def test_graph_from_dict_rejects_trace_ids() -> None:
    with pytest.raises(ValueError, match="trace"):
        graph_from_dict(
            {"candidate_units": [{"unit_id": "t_money", "surface": "money"}]},
            modality=Modality.TEXT,
        )


def test_normalize_fills_missing_list_keys() -> None:
    graph = normalize_graph_dict({"candidate_units": [{"unit_id": "u_hi", "surface": "hi"}]})
    assert graph == {**empty_graph_template(), "candidate_units": [{"unit_id": "u_hi", "surface": "hi"}]}


def test_normalize_coerces_string_units() -> None:
    graph = normalize_graph_dict({"candidate_units": ["go", "bank"]})
    assert graph["candidate_units"][0]["surface"] == "go"
    assert graph["candidate_units"][1]["unit_id"] == "u_bank"


def test_system_prompt_does_not_embed_schema() -> None:
    prompt = build_system_prompt()
    assert "JSON SCHEMA" not in prompt
    assert "$defs" not in prompt
    assert len(prompt) < 700


def test_structured_response_format_carries_schema() -> None:
    rf = structured_response_format()
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] is PERCEPTUAL_EVIDENCE_GRAPH_SCHEMA


def test_graph_from_dict_accepts_sparse_llm_shape() -> None:
    graph = graph_from_dict(
        {
            "candidate_units": [{"unit_id": "u_bank", "surface": "bank", "kind_hint": "span"}],
            "uncertainty_flags": [
                {"target_id": "u_bank", "uncertainty_type": "polysemy", "severity": "medium"}
            ],
        },
        modality=Modality.TEXT,
    )
    assert graph.candidate_units[0].surface == "bank"
    assert graph.candidate_regions == []


def test_load_dotenv_sets_key(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=test-key-from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LUCID_PERCEPTION_API_KEY", raising=False)
    from lucid.perception.config import PerceptionConfig

    cfg = PerceptionConfig.from_env()
    assert cfg.api_key == "test-key-from-dotenv"


def test_require_text_evidence_rejects_empty_graph() -> None:
    from lucid.ir.perception import PerceptualEvidenceGraph
    from lucid.perception.llm import _require_text_evidence

    inp = PerceptionInput(raw_payload="go to the bank", modality=Modality.TEXT)
    with pytest.raises(ValueError, match="empty evidence graph"):
        _require_text_evidence(inp, PerceptualEvidenceGraph())


def test_empty_graph_retry_message_mentions_units() -> None:
    from lucid.perception.schema import empty_graph_retry_message

    msg = empty_graph_retry_message()
    assert "candidate_units" in msg
    assert "bank" in msg


def test_build_user_message_includes_text_to_analyze() -> None:
    from lucid.perception.schema import build_user_message

    body = build_user_message(PerceptionInput(raw_payload="go to the bank", modality=Modality.TEXT))
    assert "text_to_analyze" in body
    assert "go to the bank" in body


def test_system_prompt_requires_non_empty_units() -> None:
    prompt = build_system_prompt()
    assert "MUST" in prompt or "must" in prompt.lower()
    assert "candidate_units" in prompt
    assert "all-empty" in prompt.lower() or "empty" in prompt.lower()


def test_llm_retries_on_empty_graph(monkeypatch) -> None:
    from lucid.perception import llm as llm_mod

    calls: list[int] = []

    def fake_chat(_cfg, _messages):
        calls.append(1)
        if len(calls) == 1:
            return json.dumps({key: [] for key in empty_graph_template()})
        return json.dumps(
            {
                "candidate_units": [{"unit_id": "u_bank", "surface": "bank", "kind_hint": "span"}],
                "candidate_markers": [],
                "candidate_regions": [],
                "candidate_containers": [],
                "arrangement_hints": [],
                "change_hints": [],
                "grouping_hints": [],
                "reference_hints": [],
                "uncertainty_flags": [],
            }
        )

    monkeypatch.setattr(llm_mod, "_chat", fake_chat)
    inp = PerceptionInput(raw_payload="go to the bank", modality=Modality.TEXT)
    graph = llm_mod.perceive_llm(inp, PerceptionConfig(backend="llm", api_key="test-key", use_json_schema=False))
    assert len(calls) == 2
    assert graph.candidate_units[0].surface == "bank"


def test_llm_requires_api_key() -> None:
    with pytest.raises(ValueError, match="API_KEY"):
        perceive(
            PerceptionInput(raw_payload="hi", modality=Modality.TEXT),
            config=PerceptionConfig(backend="llm", api_key=""),
        )
