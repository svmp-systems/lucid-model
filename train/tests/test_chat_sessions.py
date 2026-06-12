"""Audited chat sessions through the universal CLI."""

from __future__ import annotations

import json
from pathlib import Path

from lucid.cli import main
from lucid.chat import run_chat_turn
from lucid.cognition.orchestrator.runner import OrchestratorRunner


def test_chat_session_two_turns(tmp_path: Path, capsys):
    audit_dir = tmp_path / "chat"
    assert main(["chat", "start", "--session-id", "session-smoke", "--audit-dir", str(audit_dir)]) == 0
    assert capsys.readouterr().out.strip() == "session-smoke"

    assert (
        main(
            [
                "chat",
                "send",
                "remember hello model",
                "--session-id",
                "session-smoke",
                "--audit-dir",
                str(audit_dir),
                "--perception",
                "rule",
                "--json",
            ]
        )
        == 0
    )
    first = json.loads(capsys.readouterr().out)
    assert first["session_id"] == "session-smoke"
    assert first["turn_index"] == 1
    assert Path(first["run_audit_dir"]).exists()

    assert (
        main(
            [
                "chat",
                "send",
                "what did I just say?",
                "--session-id",
                "session-smoke",
                "--audit-dir",
                str(audit_dir),
                "--perception",
                "rule",
                "--json",
            ]
        )
        == 0
    )
    second = json.loads(capsys.readouterr().out)
    assert second["turn_index"] == 2

    session_json = audit_dir / "session-smoke" / "session.json"
    data = json.loads(session_json.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["summary"]["headline"] == "session-smoke - 2 turns"
    assert data["full_history"] == data["turns"]
    assert [turn["user_input"] for turn in data["turns"]] == [
        "remember hello model",
        "what did I just say?",
    ]
    assert [item["kind"] for item in data["working_memory"]] == ["fact"]
    assert data["working_memory"][0]["source_turn_index"] == 1
    assert [item["turn_index"] for item in data["unclear_items"]] == [1, 2]

    transcript = (audit_dir / "session-smoke" / "transcript.txt").read_text(encoding="utf-8")
    assert "Turn 1" in transcript
    assert "remember hello model" in transcript
    history_log = (audit_dir / "session-smoke" / "history.jsonl").read_text(encoding="utf-8")
    assert len(history_log.splitlines()) == 2
    memory = json.loads((audit_dir / "session-smoke" / "memory.json").read_text(encoding="utf-8"))
    assert memory["session_id"] == "session-smoke"
    assert memory["memories"][0]["scope"] == "session"
    assert first["run_id"] in json.loads(
        (Path(first["run_audit_dir"]) / "manifest.json").read_text(encoding="utf-8")
    )["run_id"]


def test_chat_list_sessions(tmp_path: Path, capsys):
    audit_dir = tmp_path / "chat"
    main(["chat", "start", "--session-id", "b", "--audit-dir", str(audit_dir)])
    capsys.readouterr()
    main(["chat", "start", "--session-id", "a", "--audit-dir", str(audit_dir)])
    capsys.readouterr()

    assert main(["chat", "list", "--audit-dir", str(audit_dir)]) == 0
    assert capsys.readouterr().out.splitlines() == ["a", "b"]


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
    assert fresh_memory["session_id"] == "fresh-session"
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
    active = [item for item in memory["memories"] if item["status"] == "active"]
    assert active[-1]["content"]["salient"] == "green"


def test_chat_memory_cli_smoke(tmp_path: Path, capsys):
    audit_dir = tmp_path / "chat"
    main(["chat", "start", "--session-id", "memory-cli", "--audit-dir", str(audit_dir)])
    capsys.readouterr()

    assert main(["chat", "memory", "--session-id", "memory-cli", "--audit-dir", str(audit_dir)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["session_id"] == "memory-cli"
    assert data["memories"] == []


def test_chat_turn_can_update_dmf_checkpoint(tmp_path: Path, capsys):
    audit_dir = tmp_path / "chat"
    checkpoint = tmp_path / "checkpoint"

    assert (
        main(
            [
                "chat",
                "send",
                "remember bright cyan square",
                "--session-id",
                "memory-smoke",
                "--audit-dir",
                str(audit_dir),
                "--perception",
                "rule",
                "--checkpoint",
                str(checkpoint),
                "--learn-to-dmf",
                "--json",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    learning = result["dmf_learning"]
    assert learning["action"] == "update"
    assert learning["checkpoint"] == str(checkpoint)
    assert learning["updated_trace_indices"]

    tracebank = json.loads((checkpoint / "tracebank.json").read_text(encoding="utf-8"))
    assert tracebank["records"]
    assert any(record["cue_affinities"] for record in tracebank["records"])

    session = json.loads((audit_dir / "memory-smoke" / "session.json").read_text(encoding="utf-8"))
    assert session["turns"][0]["dmf_learning"]["tracebank_hash_after"]
    assert session["working_memory"][0]["kind"] == "fact"
    assert session["working_memory"][0]["text"] == "remember bright cyan square"
    assert session["unclear_items"][0]["run_id"] == result["run_id"]

    dmf_audit_files = list((audit_dir / "memory-smoke" / "dmf_learning").glob("*/*.json"))
    assert dmf_audit_files


def test_chat_dmf_learning_requires_checkpoint(tmp_path: Path, capsys):
    exit_code = main(
        [
            "chat",
            "send",
            "remember this",
            "--session-id",
            "memory-smoke",
            "--audit-dir",
            str(tmp_path / "chat"),
            "--learn-to-dmf",
        ]
    )

    assert exit_code == 2
    assert "requires --checkpoint" in capsys.readouterr().err


def test_chat_rejects_path_like_session_ids(tmp_path: Path, capsys):
    exit_code = main(
        [
            "chat",
            "start",
            "--session-id",
            "../escape",
            "--audit-dir",
            str(tmp_path / "chat"),
        ]
    )

    assert exit_code == 2
    assert "session_id must" in capsys.readouterr().err
    assert not (tmp_path / "escape").exists()


def test_chat_passes_session_state_into_runner(monkeypatch, tmp_path: Path):
    audit_dir = tmp_path / "chat"
    main(["chat", "start", "--session-id", "state-smoke", "--audit-dir", str(audit_dir)])
    main(
        [
            "chat",
            "send",
            "remember first turn",
            "--session-id",
            "state-smoke",
            "--audit-dir",
            str(audit_dir),
            "--perception",
            "rule",
        ]
    )

    original_run_episode = OrchestratorRunner.run_episode
    seen = {}

    def spy_run_episode(self, episode, **kwargs):
        seen["turn_index"] = kwargs.get("turn_index")
        state = kwargs.get("session_state")
        seen["session_state_turns"] = len(state.turns) if state is not None else -1
        seen["session_state_first_input"] = state.turns[0].user_input if state and state.turns else ""
        return original_run_episode(self, episode, **kwargs)

    monkeypatch.setattr(OrchestratorRunner, "run_episode", spy_run_episode)

    run_chat_turn(
        "second turn",
        session_id="state-smoke",
        audit_dir=audit_dir,
        perception_backend="rule",
    )

    assert seen == {
        "turn_index": 2,
        "session_state_turns": 1,
        "session_state_first_input": "remember first turn",
    }


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
