"""When checkpoint definitions are junk fragments, chat should not wordsoup."""

from __future__ import annotations

from lucid.chat import run_chat_turn, start_session


def test_what_is_ai_does_not_emit_black_box_fragment() -> None:
    sid = start_session()
    result = run_chat_turn(
        "what is AI",
        session_id=sid,
        checkpoint="checkpoints/saves/v.0.3-ai-ml",
        perception_backend="rule",
    )
    lowered = result.assistant_output.lower()
    assert "black box" not in lowered
    assert "some explaining to do" not in lowered
    assert "i'm lucid" not in lowered
    assert (
        "not confident" in lowered
        or "does not force a single reading" in lowered
        or "field of computer science" in lowered
        or "machine learning" in lowered
    )
