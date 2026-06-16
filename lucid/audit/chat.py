"""Session audit and lightweight memory helpers for chat turns."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lucid.ir.pipeline import SessionState, TurnRecord
from lucid.ir.serde import to_dict
from lucid.runtime.paths import resolve_train_path

_MAX_CONTEXT_TURNS = 8
_SAFE_SESSION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(slots=True)
class ChatAuditTurn:
    turn_index: int
    run_id: str
    user_input: str
    assistant_output: str
    run_audit_dir: str = ""
    lucidity_decision: str = ""
    dmf_learning: dict[str, object] = field(default_factory=dict)
    response_source: str = "pipeline"
    memory_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ChatRecord:
    session_id: str
    turns: list[ChatAuditTurn] = field(default_factory=list)
    working_memory: list[dict[str, Any]] = field(default_factory=list)
    summaries: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class MemoryEvent:
    event_type: str
    key: str
    value: str = ""
    previous_value: str = ""
    turn_index: int = 0
    source: str = ""


@dataclass(slots=True)
class SessionMemory:
    session_id: str
    memories: list[dict[str, Any]] = field(default_factory=list)
    events: list[MemoryEvent] = field(default_factory=list)


def _validate_session_id(session_id: str) -> str:
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("session id cannot be empty")
    if not _SAFE_SESSION_RE.match(sid):
        raise ValueError("session id may contain only letters, numbers, dot, dash, or underscore")
    return sid


def _audit_root(audit_dir: str | Path) -> Path:
    return resolve_train_path(audit_dir, mkdir=True)


def session_file(audit_dir: str | Path, session_id: str) -> Path:
    sid = _validate_session_id(session_id)
    return _audit_root(audit_dir) / sid / "session.json"


def memory_file(audit_dir: str | Path, session_id: str) -> Path:
    sid = _validate_session_id(session_id)
    return _audit_root(audit_dir) / sid / "memory.json"


def new_chat_record(session_id: str) -> ChatRecord:
    return ChatRecord(session_id=_validate_session_id(session_id))


def new_session_memory(session_id: str) -> SessionMemory:
    return SessionMemory(session_id=_validate_session_id(session_id))


def load_chat_record(audit_dir: str | Path, session_id: str) -> ChatRecord | None:
    path = session_file(audit_dir, session_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    turns = [
        ChatAuditTurn(
            turn_index=int(item.get("turn_index", index + 1)),
            run_id=str(item.get("run_id") or ""),
            user_input=str(item.get("user_input") or ""),
            assistant_output=str(item.get("assistant_output") or ""),
            run_audit_dir=str(item.get("run_audit_dir") or ""),
            lucidity_decision=str(item.get("lucidity_decision") or ""),
            dmf_learning=dict(item.get("dmf_learning") or {}),
            response_source=str(item.get("response_source") or "pipeline"),
            memory_events=list(item.get("memory_events") or []),
        )
        for index, item in enumerate(data.get("turns") or [])
        if isinstance(item, dict)
    ]
    return ChatRecord(
        session_id=str(data.get("session_id") or session_id),
        turns=turns,
        working_memory=[
            dict(item) for item in data.get("working_memory", []) if isinstance(item, dict)
        ],
        summaries=[dict(item) for item in data.get("summaries", []) if isinstance(item, dict)],
    )


def save_chat_record(audit_dir: str | Path, record: ChatRecord) -> Path:
    path = session_file(audit_dir, record.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(record), indent=2, sort_keys=True), encoding="utf-8")
    return path


def delete_chat_session(audit_dir: str | Path, session_id: str) -> None:
    """Remove a session directory and all of its on-disk audit artifacts."""
    sid = _validate_session_id(session_id)
    session_dir = _audit_root(audit_dir) / sid
    if not session_dir.is_dir():
        raise ValueError(f"session not found: {sid}")
    shutil.rmtree(session_dir)


def append_chat_turn(
    audit_dir: str | Path,
    *,
    session_id: str,
    turn: ChatAuditTurn,
) -> ChatRecord:
    record = load_chat_record(audit_dir, session_id) or new_chat_record(session_id)
    record.turns.append(turn)
    refresh_history_views(record)
    save_chat_record(audit_dir, record)
    return record


def load_session_memory(audit_dir: str | Path, session_id: str) -> SessionMemory | None:
    path = memory_file(audit_dir, session_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    events = [
        MemoryEvent(
            event_type=str(item.get("event_type") or ""),
            key=str(item.get("key") or ""),
            value=str(item.get("value") or ""),
            previous_value=str(item.get("previous_value") or ""),
            turn_index=int(item.get("turn_index") or 0),
            source=str(item.get("source") or ""),
        )
        for item in data.get("events") or []
        if isinstance(item, dict)
    ]
    return SessionMemory(
        session_id=str(data.get("session_id") or session_id),
        memories=[dict(item) for item in data.get("memories", []) if isinstance(item, dict)],
        events=events,
    )


def save_session_memory(audit_dir: str | Path, memory: SessionMemory) -> Path:
    path = memory_file(audit_dir, memory.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(memory), indent=2, sort_keys=True), encoding="utf-8")
    return path


def refresh_history_views(record: ChatRecord) -> None:
    record.working_memory = [
        {
            "turn_index": turn.turn_index,
            "user_input": turn.user_input,
            "assistant_output": turn.assistant_output,
            "lucidity_decision": turn.lucidity_decision,
        }
        for turn in record.turns[-_MAX_CONTEXT_TURNS:]
    ]
    omitted = max(0, len(record.turns) - _MAX_CONTEXT_TURNS)
    record.summaries = [{"omitted_prior_turns": omitted}] if omitted else []


def build_session_context(
    record: ChatRecord,
    memory: SessionMemory,
    *,
    current_turn_index: int,
) -> dict[str, Any]:
    refresh_history_views(record)
    omitted = max(0, len(record.turns) - _MAX_CONTEXT_TURNS)
    recent = [
        {
            "turn_index": turn.turn_index,
            "user_input": turn.user_input,
            "assistant_output": turn.assistant_output,
        }
        for turn in record.turns[-_MAX_CONTEXT_TURNS:]
    ]
    return {
        "session_id": record.session_id,
        "current_turn_index": current_turn_index,
        "recent_turns": recent,
        "active_memories": list(memory.memories),
        "active_bindings": _active_bindings(memory),
        "unresolved_items": [],
        "summaries": list(record.summaries),
        "history_policy": {
            "max_prior_turns": _MAX_CONTEXT_TURNS,
            "selected_prior_turns": len(recent),
            "omitted_prior_turns": omitted,
        },
    }


def to_session_state(record: ChatRecord, memory: SessionMemory) -> SessionState:
    refresh_history_views(record)
    return SessionState(
        session_id=record.session_id,
        turns=[
            TurnRecord(
                turn_index=turn.turn_index,
                user_input=turn.user_input,
                run_id=turn.run_id,
                lucidity_decision=turn.lucidity_decision,
                decoder_surface=turn.assistant_output,
            )
            for turn in record.turns[-_MAX_CONTEXT_TURNS:]
        ],
        active_memories=list(memory.memories),
        active_bindings=_active_bindings(memory),
        unresolved_items=[],
        summaries=list(record.summaries),
    )


def memory_reply_for_text(text: str, memory: SessionMemory) -> str:
    lowered = text.lower()
    if "colour" not in lowered and "color" not in lowered:
        return ""
    if "what" not in lowered and "tell" not in lowered:
        return ""
    item = _find_memory(memory, "colour") or _find_memory(memory, "color")
    return str(item.get("value") or "") if item else ""


def update_session_memory_from_turn(
    memory: SessionMemory,
    turn: ChatAuditTurn,
    *,
    lucidity_decision: str = "",
) -> list[MemoryEvent]:
    _ = lucidity_decision
    text = turn.user_input.strip()
    updates = _extract_memory_updates(text, memory)
    events: list[MemoryEvent] = []
    for key, value in updates:
        existing = _find_memory(memory, key)
        if existing is None:
            memory.memories.append(
                {
                    "key": key,
                    "value": value,
                    "source_turn_index": turn.turn_index,
                    "updated_turn_index": turn.turn_index,
                }
            )
            event = MemoryEvent(
                event_type="binding_added",
                key=key,
                value=value,
                turn_index=turn.turn_index,
                source=turn.run_id,
            )
        else:
            previous = str(existing.get("value") or "")
            existing["value"] = value
            existing["updated_turn_index"] = turn.turn_index
            event = MemoryEvent(
                event_type="binding_rebound" if previous != value else "binding_refreshed",
                key=key,
                value=value,
                previous_value=previous,
                turn_index=turn.turn_index,
                source=turn.run_id,
            )
        memory.events.append(event)
        events.append(event)
    return events


def _active_bindings(memory: SessionMemory) -> list[dict[str, Any]]:
    return [
        {
            "binding_type": "session_memory",
            "key": item.get("key", ""),
            "value": item.get("value", ""),
            "updated_turn_index": item.get("updated_turn_index", 0),
        }
        for item in memory.memories
    ]


def _find_memory(memory: SessionMemory, key: str) -> dict[str, Any] | None:
    for item in reversed(memory.memories):
        if str(item.get("key") or "") == key:
            return item
    return None


def _extract_memory_updates(text: str, memory: SessionMemory) -> list[tuple[str, str]]:
    lowered = text.lower()
    colour = re.search(
        r"\b(?:remember\s+(?:the\s+)?colou?r|(?:the\s+)?colou?r\s+is)\s+([a-z0-9_-]+)\b",
        lowered,
    )
    if colour:
        return [("colour", colour.group(1))]

    remember_item = re.search(r"\bremember\s+(.+?)\s*$", text, flags=re.IGNORECASE)
    if remember_item:
        value = remember_item.group(1).strip()
        if value:
            key = f"memory:{len(memory.memories) + 1:04d}"
            return [(key, value)]

    rebind = re.search(r"\b(?:actually\s+)?make\s+it\s+([a-z0-9_-]+)\b", lowered)
    if rebind:
        if _find_memory(memory, "colour") is not None:
            return [("colour", rebind.group(1))]
        if memory.memories:
            key = str(memory.memories[-1].get("key") or "")
            if key:
                return [(key, rebind.group(1))]
    return []
