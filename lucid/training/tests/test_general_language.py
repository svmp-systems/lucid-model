"""General conversational language speech."""

from __future__ import annotations

from lucid.chat import run_chat_turn
from lucid.cognition.output.lucidity.chat_speech import classify_social_utterance
from lucid.ir.common import TaskIntent
from lucid.training.corpus.engine import generate
from lucid.training.general_language import train_general_language


def test_classify_greeting():
    assert classify_social_utterance("hi") == ("greeting", "Hello.")
    assert classify_social_utterance("Hello there") == ("greeting", "Hello.")
    assert classify_social_utterance("thank you") == ("thanks", "You're welcome.")


def test_chat_social_episodes_validate():
    episodes = generate("chat_social", 10, seed=1)
    assert episodes
    assert all(ep.task_intent == TaskIntent.CHAT for ep in episodes)
    assert all(ep.gold.lucidity_target == "COMMIT" for ep in episodes)


def test_chat_hi_replies_with_hello(tmp_path):
    audit_dir = tmp_path / "chat"
    result = run_chat_turn(
        "hi",
        session_id="general-language-hi",
        audit_dir=audit_dir,
        perception_backend="rule",
    )
    assert "hello" in result.assistant_output.lower()


def test_chat_how_are_you_with_rule_perception(tmp_path):
    audit_dir = tmp_path / "chat"
    session_id = "general-language-how-are-you"
    run_chat_turn("hi", session_id=session_id, audit_dir=audit_dir, perception_backend="rule")
    result = run_chat_turn(
        "how are you",
        session_id=session_id,
        audit_dir=audit_dir,
        perception_backend="rule",
    )
    assert "help" in result.assistant_output.lower()
    assert "how you" not in result.assistant_output.lower()


def test_general_language_bootstrap(tmp_path):
    checkpoint = tmp_path / "v0.0.1"
    from lucid.training.quantum_articles import train_quantum_articles

    train_quantum_articles(checkpoint)
    summary = train_general_language(
        checkpoint,
        episode_count=12,
        seed=7,
        run_module_train=False,
        registry_name="v0.0.1",
    )
    assert summary["episodes"] == 12
    assert summary["archived"]["name"] == "v0.0.1"
