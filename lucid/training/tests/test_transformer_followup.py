"""Regression tests for transformer definition + follow-up mechanism queries."""

from __future__ import annotations

from lucid.chat import run_chat_turn, start_session


def test_transformer_definition_and_followup_mechanism() -> None:
    sid = start_session()
    first = run_chat_turn(
        "what is a transformer",
        session_id=sid,
        checkpoint="checkpoints/saves/v.0.3-ai-ml",
        perception_backend="rule",
    )
    assert "bayesian network" not in first.assistant_output.lower()
    lowered = first.assistant_output.lower()
    assert (
        "transformer" in lowered
        or "attention" in lowered
        or "not confident" in lowered
    )

    second = run_chat_turn(
        "how does it work",
        session_id=sid,
        checkpoint="checkpoints/saves/v.0.3-ai-ml",
        perception_backend="rule",
    )
    lowered = second.assistant_output.lower()
    assert "student" not in lowered
    assert "bayesian network" not in lowered or "transformer" in lowered
    assert len(second.assistant_output.split(".")) <= 3
    assert (
        "attention" in lowered
        or "uses" in lowered
        or "not confident" in lowered
    )
