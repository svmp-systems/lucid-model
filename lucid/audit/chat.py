"""Auditable chat session receipts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

from lucid.ir.pipeline import SessionState, TurnRecord
from lucid.ir.serde import from_dict, to_dict, to_json

CHAT_SCHEMA_VERSION = 1
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(slots=True)
class ChatAuditTurn:
    turn_index: int
    run_id: str
    user_input: str
    assistant_output: str
    run_audit_dir: str
    created_at: str = ""
    lucidity_decision: str = ""
    dmf_learning: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatWorkingMemoryItem:
    item_id: str
    kind: str
    text: str
    source_turn_index: int
    evidence: str = ""


@dataclass(slots=True)
class ChatUnclearItem:
    turn_index: int
    question: str
    reason: str
    run_id: str = ""


@dataclass(slots=True)
class ChatAuditRecord:
    session_id: str
    created_at: str
    updated_at: str
    turns: list[ChatAuditTurn] = field(default_factory=list)
    working_memory: list[ChatWorkingMemoryItem] = field(default_factory=list)
    unclear_items: list[ChatUnclearItem] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def session_dir(base_dir: Path | str, session_id: str) -> Path:
    validate_session_id(session_id)
    return Path(base_dir) / session_id


def session_file(base_dir: Path | str, session_id: str) -> Path:
    return session_dir(base_dir, session_id) / "session.json"


def transcript_file(base_dir: Path | str, session_id: str) -> Path:
    return session_dir(base_dir, session_id) / "transcript.txt"


def history_log_file(base_dir: Path | str, session_id: str) -> Path:
    return session_dir(base_dir, session_id) / "history.jsonl"


def summarize_chat(record: ChatAuditRecord) -> dict[str, Any]:
    last = record.turns[-1].assistant_output if record.turns else ""
    preview = last[:96] + ("..." if len(last) > 96 else "")
    return {
        "headline": f"{record.session_id} - {len(record.turns)} turns",
        "lines": [
            f"session_id: {record.session_id}",
            f"turns: {len(record.turns)}",
            f"working_memory_items: {len(record.working_memory)}",
            f"unclear_items: {len(record.unclear_items)}",
            f"created_at: {record.created_at}",
            f"updated_at: {record.updated_at}",
            f"last_assistant_output: {preview or '-'}",
        ],
    }


def new_chat_record(session_id: str) -> ChatAuditRecord:
    validate_session_id(session_id)
    now = utc_now_iso()
    record = ChatAuditRecord(session_id=session_id, created_at=now, updated_at=now)
    record.summary = summarize_chat(record)
    return record


def validate_session_id(session_id: str) -> None:
    if not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError(
            "session_id must start with a letter or number and contain only letters, "
            "numbers, dot, underscore, or hyphen"
        )


def load_chat_record(base_dir: Path | str, session_id: str) -> ChatAuditRecord | None:
    path = session_file(base_dir, session_id)
    if not path.exists():
        return None
    raw = _read_json(path)
    if "working_memory" not in raw and "working_history" in raw:
        raw["working_memory"] = [
            {
                "item_id": f"legacy_working_{item.get('turn_index', 0)}",
                "kind": "context",
                "text": " | ".join(
                    part
                    for part in [
                        str(item.get("user_input") or "").strip(),
                        str(item.get("assistant_output") or "").strip(),
                    ]
                    if part
                ),
                "source_turn_index": int(item.get("turn_index") or 0),
                "evidence": "legacy working_history",
            }
            for item in raw.get("working_history") or []
            if isinstance(item, dict)
        ]
    raw.pop("working_history", None)
    raw.pop("full_history", None)
    data = to_dict(from_dict(raw, ChatAuditRecord))
    data.pop("schema_version", None)
    return from_dict(data, ChatAuditRecord)


def save_chat_record(base_dir: Path | str, record: ChatAuditRecord) -> Path:
    target_dir = session_dir(base_dir, record.session_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    refresh_history_views(record)
    record.updated_at = utc_now_iso()
    record.summary = summarize_chat(record)
    data = {"schema_version": CHAT_SCHEMA_VERSION, **to_dict(record)}
    data["full_history"] = data["turns"]
    session_path = target_dir / "session.json"
    session_path.write_text(to_json(data), encoding="utf-8")
    _write_transcript(target_dir / "transcript.txt", record)
    return session_path


def append_chat_turn(
    base_dir: Path | str,
    *,
    session_id: str,
    turn: ChatAuditTurn,
) -> ChatAuditRecord:
    record = load_chat_record(base_dir, session_id) or new_chat_record(session_id)
    if not turn.created_at:
        turn.created_at = utc_now_iso()
    record.turns.append(turn)
    save_chat_record(base_dir, record)
    _append_history_event(base_dir, record.session_id, turn)
    return record


def refresh_history_views(record: ChatAuditRecord, *, working_limit: int = 24) -> None:
    items: list[ChatWorkingMemoryItem] = []
    for turn in record.turns:
        items.extend(_extract_working_memory(turn))
    record.working_memory = items[-working_limit:]

    last_commit = max(
        (turn.turn_index for turn in record.turns if turn.lucidity_decision == "commit"),
        default=0,
    )
    record.unclear_items = [
        ChatUnclearItem(
            turn_index=turn.turn_index,
            question=_compact_text(turn.user_input, limit=180),
            reason=f"lucidity_decision={turn.lucidity_decision or 'unknown'}",
            run_id=turn.run_id,
        )
        for turn in record.turns
        if turn.turn_index > last_commit and _is_unclear(turn)
    ]


def to_session_state(record: ChatAuditRecord) -> SessionState:
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
            for turn in record.turns
        ],
    )


def _is_unclear(turn: ChatAuditTurn) -> bool:
    if turn.assistant_output.strip() in {"", "(no decoder output)", "(holding: ambiguity)"}:
        return True
    return turn.lucidity_decision not in {"", "commit"}


def _compact_text(text: str, *, limit: int = 240) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _extract_working_memory(turn: ChatAuditTurn) -> list[ChatWorkingMemoryItem]:
    text = _compact_text(turn.user_input, limit=360)
    lower = text.lower()
    kinds: list[str] = []
    if any(token in lower for token in ("we need", "need to", "goal", "want to", "let's")):
        kinds.append("goal")
    if any(token in lower for token in ("remember", "fact", " is ", " are ", " means ")):
        kinds.append("fact")
    if any(token in lower for token in ("if ", "unless", "when ", "must", "requires", "only")):
        kinds.append("condition")
    if any(token in lower for token in ("suggest", "should", "could", "maybe", "consider")):
        kinds.append("suggestion")
    if any(token in lower for token in ("prefer", "don't", "dont", "do not", "never", "always")):
        kinds.append("preference")
    return [
        ChatWorkingMemoryItem(
            item_id=f"turn_{turn.turn_index:04d}_{kind}",
            kind=kind,
            text=text,
            source_turn_index=turn.turn_index,
            evidence="user_input",
        )
        for kind in dict.fromkeys(kinds)
    ]


def _append_history_event(base_dir: Path | str, session_id: str, turn: ChatAuditTurn) -> None:
    path = history_log_file(base_dir, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "schema_version": CHAT_SCHEMA_VERSION,
        "event_type": "chat_turn",
        "session_id": session_id,
        "turn": to_dict(turn),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(to_json(event, indent=None) + "\n")


def _write_transcript(path: Path, record: ChatAuditRecord) -> None:
    lines = [
        record.summary.get("headline", record.session_id),
        "=" * len(record.summary.get("headline", record.session_id)),
        "",
    ]
    for turn in record.turns:
        lines.extend(
            [
                f"Turn {turn.turn_index}",
                f"user: {turn.user_input}",
                f"assistant: {turn.assistant_output}",
                f"run_audit_dir: {turn.run_audit_dir}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
