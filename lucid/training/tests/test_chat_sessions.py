"""Audited chat sessions through the universal CLI."""

from __future__ import annotations

import json
from pathlib import Path

from lucid.chat import run_chat_turn
from lucid.cli import main
from lucid.cognition.pipe_orchestrator.runner import OrchestratorRunner


def test_chat_memory_is_session_local_and_recallable(tmp_path: Path, capsys):
    audit_dir = tmp_path / "chat"

    assert (
        main(
            [
                "chat",
                "send",
                "remember the colour blue",
                "--session-id",
                "blue-session",
                "--audit-dir",
                str(audit_dir),
                "--perception",
                "rule",
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "chat",
                "send",
                "what colour did I tell you?",
                "--session-id",
                "blue-session",
                "--audit-dir",
                str(audit_dir),
                "--perception",
                "rule",
                "--json",
            ]
        )
        == 0
    )
    recalled = json.loads(capsys.readouterr().out)
    assert recalled["assistant_output"].lower() == "blue"

    assert (
        main(
            [
                "chat",
                "send",
                "what colour did I tell you?",
                "--session-id",
                "fresh-session",
                "--audit-dir",
                str(audit_dir),
                "--perception",
                "rule",
                "--json",
            ]
        )
        == 0
    )
    fresh = json.loads(capsys.readouterr().out)
    assert "blue" not in fresh["assistant_output"].lower()

    blue_memory = json.loads((audit_dir / "blue-session" / "memory.json").read_text(encoding="utf-8"))
    fresh_memory = json.loads((audit_dir / "fresh-session" / "memory.json").read_text(encoding="utf-8"))
    assert blue_memory["memories"]
    assert fresh_memory["memories"] == []


def test_chat_memory_rebinds_within_session(tmp_path: Path, capsys):
    audit_dir = tmp_path / "chat"
    for message in ["remember the colour blue", "actually make it green"]:
        assert (
            main(
                [
                    "chat",
                    "send",
                    message,
                    "--session-id",
                    "rebind-session",
                    "--audit-dir",
                    str(audit_dir),
                    "--perception",
                    "rule",
                    "--json",
                ]
            )
            == 0
        )
        capsys.readouterr()

    assert (
        main(
            [
                "chat",
                "send",
                "what colour did I tell you?",
                "--session-id",
                "rebind-session",
                "--audit-dir",
                str(audit_dir),
                "--perception",
                "rule",
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["assistant_output"].lower() == "green"

    memory = json.loads((audit_dir / "rebind-session" / "memory.json").read_text(encoding="utf-8"))
    assert any(event["event_type"] == "binding_rebound" for event in memory["events"])


def test_chat_memory_cli_smoke(tmp_path: Path, capsys):
    audit_dir = tmp_path / "chat"
    main(["chat", "start", "--session-id", "memory-cli", "--audit-dir", str(audit_dir)])
    capsys.readouterr()

    assert main(["chat", "memory", "--session-id", "memory-cli", "--audit-dir", str(audit_dir)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["session_id"] == "memory-cli"
    assert data["memories"] == []


def test_chat_control_turns_do_not_leak_prior_domain_answers(tmp_path: Path):
    audit_dir = tmp_path / "chat"
    session_id = "control-turns"

    ack = run_chat_turn(
        "oh okay",
        session_id=session_id,
        audit_dir=audit_dir,
        perception_backend="llm",
    )
    confused = run_chat_turn(
        "huh?",
        session_id=session_id,
        audit_dir=audit_dir,
        perception_backend="llm",
    )
    meta = run_chat_turn(
        "why did it answer",
        session_id=session_id,
        audit_dir=audit_dir,
        perception_backend="llm",
    )

    assert ack.assistant_output == "Okay."
    assert "clarify" in confused.assistant_output.lower()
    assert "treated the last short turn" in meta.assistant_output.lower()
    assert not ack.run_audit_dir
    assert not confused.run_audit_dir
    assert not meta.run_audit_dir

    session = json.loads((audit_dir / session_id / "session.json").read_text(encoding="utf-8"))
    assert [turn["response_source"] for turn in session["turns"]] == [
        "chat_control",
        "chat_control",
        "chat_control",
    ]


def test_chat_passes_bounded_session_context_into_runner(monkeypatch, tmp_path: Path):
    audit_dir = tmp_path / "chat"
    for index in range(10):
        main(
            [
                "chat",
                "send",
                f"remember item {index}",
                "--session-id",
                "bounded-smoke",
                "--audit-dir",
                str(audit_dir),
                "--perception",
                "rule",
            ]
        )

    original_run_episode = OrchestratorRunner.run_episode
    seen = {}

    def spy_run_episode(self, episode, **kwargs):
        session_context = episode.context["session_context"]
        seen["selected_prior_turns"] = session_context["history_policy"]["selected_prior_turns"]
        seen["omitted_prior_turns"] = session_context["history_policy"]["omitted_prior_turns"]
        state = kwargs.get("session_state")
        seen["state_turns"] = len(state.turns)
        seen["active_memories"] = len(state.active_memories)
        return original_run_episode(self, episode, **kwargs)

    monkeypatch.setattr(OrchestratorRunner, "run_episode", spy_run_episode)
    run_chat_turn(
        "what did I tell you?",
        session_id="bounded-smoke",
        audit_dir=audit_dir,
        perception_backend="rule",
    )

    assert seen == {
        "selected_prior_turns": 8,
        "omitted_prior_turns": 2,
        "state_turns": 8,
        "active_memories": 10,
    }


def test_chat_resolves_explicit_loaded_checkpoint_before_runtime(monkeypatch, tmp_path: Path):
    audit_dir = tmp_path / "chat"
    seen = {}

    monkeypatch.setattr(
        "lucid.chat.resolve_inference_checkpoint",
        lambda checkpoint: "checkpoints/loaded" if checkpoint == "loaded" else None,
    )
    original_run_episode = OrchestratorRunner.run_episode

    def spy_run_episode(self, episode, **kwargs):
        seen["checkpoint"] = self.config.checkpoint
        return original_run_episode(self, episode, **kwargs)

    monkeypatch.setattr(OrchestratorRunner, "run_episode", spy_run_episode)
    run_chat_turn(
        "What is a qubit?",
        session_id="checkpoint-alias-smoke",
        audit_dir=audit_dir,
        perception_backend="rule",
        checkpoint="loaded",
    )

    assert seen["checkpoint"] == "checkpoints/loaded"
