"""Auditable, session-local chat memory."""

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
    response_source: str = "pipeline"
    memory_events: list[dict[str, Any]] = field(default_factory=list)


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
class ChatMemoryRecord:
    memory_id: str
    kind: str
    content: dict[str, Any]
    source_turn_index: int
    source_run_id: str = ""
    scope: str = "session"
    confidence: float = 0.0
    status: str = "active"
    refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatBindingRecord:
    binding_id: str
    surface: str
    target_ref: str
    target_type: str = "memory"
    source_turn_index: int = 0
    status: str = "active"
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatMemoryEvent:
    event_type: str
    turn_index: int
    run_id: str = ""
    memory_id: str = ""
    binding_id: str = ""
    operation: str = ""
    old_target_ref: str = ""
    new_target_ref: str = ""
    reason: str = ""
    lucidity_decision: str = ""
    created_at: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatSessionMemory:
    session_id: str
    created_at: str
    updated_at: str
    memories: list[ChatMemoryRecord] = field(default_factory=list)
    bindings: list[ChatBindingRecord] = field(default_factory=list)
    summaries: list[ChatMemoryRecord] = field(default_factory=list)
    unresolved_items: list[ChatUnclearItem] = field(default_factory=list)
    events: list[ChatMemoryEvent] = field(default_factory=list)


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


def validate_session_id(session_id: str) -> None:
    if not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError(
            "session_id must start with a letter or number and contain only letters, "
            "numbers, dot, underscore, or hyphen"
        )


def session_dir(base_dir: Path | str, session_id: str) -> Path:
    validate_session_id(session_id)
    return Path(base_dir) / session_id


def session_file(base_dir: Path | str, session_id: str) -> Path:
    return session_dir(base_dir, session_id) / "session.json"


def memory_file(base_dir: Path | str, session_id: str) -> Path:
    return session_dir(base_dir, session_id) / "memory.json"


def history_log_file(base_dir: Path | str, session_id: str) -> Path:
    return session_dir(base_dir, session_id) / "history.jsonl"


def new_chat_record(session_id: str) -> ChatAuditRecord:
    validate_session_id(session_id)
    now = utc_now_iso()
    record = ChatAuditRecord(session_id=session_id, created_at=now, updated_at=now)
    record.summary = summarize_chat(record)
    return record


def new_session_memory(session_id: str) -> ChatSessionMemory:
    validate_session_id(session_id)
    now = utc_now_iso()
    return ChatSessionMemory(session_id=session_id, created_at=now, updated_at=now)


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


def load_chat_record(base_dir: Path | str, session_id: str) -> ChatAuditRecord | None:
    path = session_file(base_dir, session_id)
    if not path.exists():
        return None
    raw = _read_json(path)
    raw.pop("full_history", None)
    data = to_dict(from_dict(raw, ChatAuditRecord))
    data.pop("schema_version", None)
    return from_dict(data, ChatAuditRecord)


def load_session_memory(base_dir: Path | str, session_id: str) -> ChatSessionMemory | None:
    path = memory_file(base_dir, session_id)
    if not path.exists():
        return None
    raw = _read_json(path)
    data = to_dict(from_dict(raw, ChatSessionMemory))
    data.pop("schema_version", None)
    return from_dict(data, ChatSessionMemory)


def save_chat_record(base_dir: Path | str, record: ChatAuditRecord) -> Path:
    target = session_dir(base_dir, record.session_id)
    target.mkdir(parents=True, exist_ok=True)
    refresh_history_views(record)
    record.updated_at = utc_now_iso()
    record.summary = summarize_chat(record)
    data = {"schema_version": CHAT_SCHEMA_VERSION, **to_dict(record)}
    data["full_history"] = data["turns"]
    path = target / "session.json"
    path.write_text(to_json(data), encoding="utf-8")
    _write_transcript(target / "transcript.txt", record)
    return path


def save_session_memory(base_dir: Path | str, memory: ChatSessionMemory) -> Path:
    target = session_dir(base_dir, memory.session_id)
    target.mkdir(parents=True, exist_ok=True)
    memory.updated_at = utc_now_iso()
    path = target / "memory.json"
    path.write_text(to_json({"schema_version": CHAT_SCHEMA_VERSION, **to_dict(memory)}), encoding="utf-8")
    return path


def append_chat_turn(base_dir: Path | str, *, session_id: str, turn: ChatAuditTurn) -> ChatAuditRecord:
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
    last_commit = max((turn.turn_index for turn in record.turns if turn.lucidity_decision == "commit"), default=0)
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


def build_session_context(
    record: ChatAuditRecord,
    memory: ChatSessionMemory,
    *,
    current_turn_index: int,
    recent_turn_limit: int = 8,
    memory_limit: int = 32,
    binding_limit: int = 64,
) -> dict[str, Any]:
    refresh_history_views(record)
    recent_turns = record.turns[-recent_turn_limit:]
    active_memories = [item for item in memory.memories if item.status == "active"][-memory_limit:]
    active_bindings = [item for item in memory.bindings if item.status == "active"][-binding_limit:]
    return {
        "schema_version": CHAT_SCHEMA_VERSION,
        "session_id": record.session_id,
        "current_turn_index": current_turn_index,
        "history_policy": {
            "recent_turn_limit": recent_turn_limit,
            "memory_limit": memory_limit,
            "binding_limit": binding_limit,
            "total_prior_turns": len(record.turns),
            "selected_prior_turns": len(recent_turns),
            "omitted_prior_turns": max(0, len(record.turns) - len(recent_turns)),
        },
        "recent_turns": [to_dict(turn) for turn in recent_turns],
        "active_memories": [to_dict(item) for item in active_memories],
        "active_bindings": [to_dict(item) for item in active_bindings],
        "unresolved_items": [to_dict(item) for item in memory.unresolved_items or record.unclear_items],
        "summaries": [to_dict(item) for item in memory.summaries[-4:]],
    }


def update_session_memory_from_turn(memory: ChatSessionMemory, turn: ChatAuditTurn, *, lucidity_decision: str = "") -> list[ChatMemoryEvent]:
    events: list[ChatMemoryEvent] = []
    directive = _extract_memory_directive(turn.user_input)
    if not directive:
        memory.unresolved_items = [
            ChatUnclearItem(turn.turn_index, _compact_text(turn.user_input, limit=180), f"lucidity_decision={lucidity_decision or 'unknown'}", turn.run_id)
        ] if _is_unclear(turn) else []
        return events

    memory_id = f"m_{turn.turn_index:04d}_{len(memory.memories) + 1:04d}"
    tokens = _significant_tokens(directive)
    salient = tokens[-1] if tokens else directive
    memory.memories.append(
        ChatMemoryRecord(
            memory_id=memory_id,
            kind="fact",
            content={"text": directive, "source_text": turn.user_input, "salient": salient},
            source_turn_index=turn.turn_index,
            source_run_id=turn.run_id,
            confidence=0.72,
            metadata={"source": "explicit_user_directive", "lucidity_decision": lucidity_decision or "unknown"},
        )
    )
    event = ChatMemoryEvent("memory_upserted", turn.turn_index, run_id=turn.run_id, memory_id=memory_id, operation="upsert", reason="explicit_user_directive", lucidity_decision=lucidity_decision, created_at=utc_now_iso())
    memory.events.append(event)
    events.append(event)

    surfaces = _binding_surfaces(directive)
    if _looks_like_rebind(turn.user_input) and memory.bindings:
        rebound = _rebind_latest(memory, surfaces, memory_id, turn, lucidity_decision)
        memory.events.append(rebound)
        events.append(rebound)
    else:
        for surface in surfaces:
            binding = ChatBindingRecord(f"b_{turn.turn_index:04d}_{len(memory.bindings) + 1:04d}", surface, f"memory:{memory_id}", source_turn_index=turn.turn_index, confidence=0.7)
            memory.bindings.append(binding)
            added = ChatMemoryEvent("binding_added", turn.turn_index, run_id=turn.run_id, memory_id=memory_id, binding_id=binding.binding_id, operation="add", new_target_ref=binding.target_ref, reason="memory_surface", lucidity_decision=lucidity_decision, created_at=utc_now_iso(), details={"surface": surface})
            memory.events.append(added)
            events.append(added)
    memory.unresolved_items = []
    return events


def memory_reply_for_text(text: str, memory: ChatSessionMemory) -> str:
    query_tokens = set(_significant_tokens(text))
    active = [item for item in memory.memories if item.status == "active"]
    if not active or ("?" not in text and not (query_tokens & {"what", "which", "tell", "remind"})):
        return ""
    best: tuple[int, ChatMemoryRecord] | None = None
    for item in active:
        item_tokens = set(_significant_tokens(str(item.content.get("text") or "")))
        binding_tokens: set[str] = set()
        for binding in memory.bindings:
            if binding.status == "active" and binding.target_ref == f"memory:{item.memory_id}":
                binding_tokens.update(_significant_tokens(binding.surface))
        score = len(query_tokens & (item_tokens | binding_tokens))
        if best is None or score > best[0] or (score == best[0] and item.source_turn_index > best[1].source_turn_index):
            best = (score, item)
    if best is None:
        return ""
    score, item = best
    if score <= 0 and "what did i" not in text.lower():
        return ""
    answer = str(item.content.get("salient") or item.content.get("text") or "").strip()
    return answer[:1].upper() + answer[1:] if answer else ""


def to_session_state(record: ChatAuditRecord, memory: ChatSessionMemory | None = None, *, recent_turn_limit: int = 8) -> SessionState:
    return SessionState(
        session_id=record.session_id,
        turns=[
            TurnRecord(turn.turn_index, turn.user_input, turn.run_id, turn.lucidity_decision, turn.assistant_output)
            for turn in record.turns[-recent_turn_limit:]
        ],
        active_memories=[to_dict(item) for item in (memory.memories if memory else []) if item.status == "active"],
        active_bindings=[to_dict(item) for item in (memory.bindings if memory else []) if item.status == "active"],
        unresolved_items=[to_dict(item) for item in (memory.unresolved_items if memory else record.unclear_items)],
        summaries=[to_dict(item) for item in (memory.summaries if memory else [])],
    )


def _is_unclear(turn: ChatAuditTurn) -> bool:
    if turn.assistant_output.strip() in {"", "(no decoder output)", "(holding: ambiguity)"}:
        return True
    return turn.lucidity_decision not in {"", "commit"}


def _compact_text(text: str, *, limit: int = 240) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 3].rstrip() + "..."


def _extract_working_memory(turn: ChatAuditTurn) -> list[ChatWorkingMemoryItem]:
    directive = _extract_memory_directive(turn.user_input)
    if not directive:
        return []
    return [ChatWorkingMemoryItem(f"turn_{turn.turn_index:04d}_fact", "fact", directive, turn.turn_index, "user_input")]


def _extract_memory_directive(text: str) -> str:
    clean = _compact_text(text, limit=480)
    lowered = clean.lower()
    for prefix in ("remember that ", "remember ", "keep in mind that ", "keep in mind ", "note that ", "note "):
        if lowered.startswith(prefix):
            return clean[len(prefix) :].strip(" .")
    if _looks_like_rebind(clean):
        for marker in ("actually make it ", "make it ", "actually ", "instead ", "change ", "make ", "use "):
            idx = lowered.find(marker)
            if idx >= 0:
                return clean[idx + len(marker) :].strip(" .")
    return ""


def _looks_like_rebind(text: str) -> bool:
    lowered = f" {text.lower()} "
    return any(marker in lowered for marker in (" actually ", " instead ", " change ", " make it ", " use the other "))


def _significant_tokens(text: str) -> list[str]:
    stop = {"a", "an", "and", "are", "as", "be", "did", "do", "for", "i", "in", "is", "it", "me", "my", "of", "or", "that", "the", "this", "to", "was", "you"}
    return [token.lower() for token in re.findall(r"\b[A-Za-z0-9_'-]+\b", text) if token.lower() not in stop]


def _binding_surfaces(text: str, *, limit: int = 8) -> list[str]:
    tokens = _significant_tokens(text)
    surfaces = tokens[:-1] if len(tokens) > 1 else tokens
    return list(dict.fromkeys(surfaces or tokens))[:limit]


def _rebind_latest(memory: ChatSessionMemory, surfaces: list[str], memory_id: str, turn: ChatAuditTurn, lucidity_decision: str) -> ChatMemoryEvent:
    binding = next((item for item in reversed(memory.bindings) if item.status == "active"), memory.bindings[-1])
    old_target = binding.target_ref
    binding.status = "superseded"
    if old_target.startswith("memory:"):
        old_id = old_target.split(":", 1)[1]
        for record in memory.memories:
            if record.memory_id == old_id:
                record.status = "superseded"
                break
    surface = binding.surface if len(surfaces) <= 1 else surfaces[0]
    new_binding = ChatBindingRecord(f"b_{turn.turn_index:04d}_{len(memory.bindings) + 1:04d}", surface, f"memory:{memory_id}", source_turn_index=turn.turn_index, confidence=0.7)
    memory.bindings.append(new_binding)
    return ChatMemoryEvent("binding_rebound", turn.turn_index, run_id=turn.run_id, memory_id=memory_id, binding_id=new_binding.binding_id, operation="rebind", old_target_ref=old_target, new_target_ref=new_binding.target_ref, reason="latest_active_binding", lucidity_decision=lucidity_decision, created_at=utc_now_iso(), details={"surface": surface})


def _append_history_event(base_dir: Path | str, session_id: str, turn: ChatAuditTurn) -> None:
    path = history_log_file(base_dir, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"schema_version": CHAT_SCHEMA_VERSION, "event_type": "chat_turn", "session_id": session_id, "turn": to_dict(turn)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(to_json(event, indent=None) + "\n")


def _write_transcript(path: Path, record: ChatAuditRecord) -> None:
    lines = [record.summary.get("headline", record.session_id), "=" * len(record.summary.get("headline", record.session_id)), ""]
    for turn in record.turns:
        lines.extend([f"Turn {turn.turn_index}", f"user: {turn.user_input}", f"assistant: {turn.assistant_output}", f"run_audit_dir: {turn.run_audit_dir}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
