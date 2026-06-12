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


def session_dir(base_dir: Path | str, session_id: str) -> Path:
    validate_session_id(session_id)
    return Path(base_dir) / session_id


def session_file(base_dir: Path | str, session_id: str) -> Path:
    return session_dir(base_dir, session_id) / "session.json"


def transcript_file(base_dir: Path | str, session_id: str) -> Path:
    return session_dir(base_dir, session_id) / "transcript.txt"


def history_log_file(base_dir: Path | str, session_id: str) -> Path:
    return session_dir(base_dir, session_id) / "history.jsonl"


def memory_file(base_dir: Path | str, session_id: str) -> Path:
    return session_dir(base_dir, session_id) / "memory.json"


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


def new_session_memory(session_id: str) -> ChatSessionMemory:
    validate_session_id(session_id)
    now = utc_now_iso()
    return ChatSessionMemory(session_id=session_id, created_at=now, updated_at=now)


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


def load_session_memory(base_dir: Path | str, session_id: str) -> ChatSessionMemory | None:
    path = memory_file(base_dir, session_id)
    if not path.exists():
        return None
    raw = _read_json(path)
    data = to_dict(from_dict(raw, ChatSessionMemory))
    data.pop("schema_version", None)
    return from_dict(data, ChatSessionMemory)


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


def save_session_memory(base_dir: Path | str, memory: ChatSessionMemory) -> Path:
    target_dir = session_dir(base_dir, memory.session_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    memory.updated_at = utc_now_iso()
    path = target_dir / "memory.json"
    path.write_text(
        to_json({"schema_version": CHAT_SCHEMA_VERSION, **to_dict(memory)}),
        encoding="utf-8",
    )
    return path


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


def build_session_context(
    record: ChatAuditRecord,
    memory: ChatSessionMemory,
    *,
    current_turn_index: int,
    recent_turn_limit: int = 8,
    memory_limit: int = 32,
    binding_limit: int = 64,
) -> dict[str, Any]:
    """Select bounded, session-local context for the next pipeline run."""

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
        "recent_turns": [
            {
                "turn_index": turn.turn_index,
                "user_input": turn.user_input,
                "assistant_output": turn.assistant_output,
                "lucidity_decision": turn.lucidity_decision,
                "run_id": turn.run_id,
            }
            for turn in recent_turns
        ],
        "active_memories": [to_dict(item) for item in active_memories],
        "active_bindings": [to_dict(item) for item in active_bindings],
        "unresolved_items": [to_dict(item) for item in memory.unresolved_items or record.unclear_items],
        "summaries": [to_dict(item) for item in memory.summaries[-4:]],
    }


def update_session_memory_from_turn(
    memory: ChatSessionMemory,
    turn: ChatAuditTurn,
    *,
    lucidity_decision: str = "",
) -> list[ChatMemoryEvent]:
    """Apply generic session-local memory updates from one completed turn.

    This is intentionally domain-neutral. It stores explicit user-provided
    session directives as generic records and lets the pipeline interpret the
    record content in later turns.
    """

    events: list[ChatMemoryEvent] = []
    directive = _extract_memory_directive(turn.user_input)
    if not directive:
        memory.unresolved_items = [
            ChatUnclearItem(
                turn_index=turn.turn_index,
                question=_compact_text(turn.user_input, limit=180),
                reason=f"lucidity_decision={lucidity_decision or 'unknown'}",
                run_id=turn.run_id,
            )
        ] if _is_unclear(turn) else []
        return events

    memory_id = f"m_{turn.turn_index:04d}_{len(memory.memories) + 1:04d}"
    tokens = _significant_tokens(directive)
    salient = tokens[-1] if tokens else directive
    record = ChatMemoryRecord(
        memory_id=memory_id,
        kind="fact",
        content={
            "text": directive,
            "source_text": turn.user_input,
            "salient": salient,
        },
        source_turn_index=turn.turn_index,
        source_run_id=turn.run_id,
        confidence=0.72,
        metadata={
            "source": "explicit_user_directive",
            "lucidity_decision": lucidity_decision or "unknown",
        },
    )
    memory.memories.append(record)
    memory_event = ChatMemoryEvent(
        event_type="memory_upserted",
        turn_index=turn.turn_index,
        run_id=turn.run_id,
        memory_id=memory_id,
        operation="upsert",
        reason="explicit_user_directive",
        lucidity_decision=lucidity_decision,
        created_at=utc_now_iso(),
    )
    memory.events.append(memory_event)
    events.append(memory_event)

    surfaces = _binding_surfaces(directive)
    if _looks_like_rebind(turn.user_input) and memory.bindings:
        rebound = _rebind_latest(memory, surfaces, memory_id, turn, lucidity_decision)
        memory.events.append(rebound)
        events.append(rebound)
    else:
        for surface in surfaces:
            existing = _active_binding_for_surface(memory, surface)
            if existing is not None:
                existing.status = "superseded"
                event = _binding_event(
                    "binding_rebound",
                    existing,
                    new_target_ref=f"memory:{memory_id}",
                    turn=turn,
                    lucidity_decision=lucidity_decision,
                    reason="surface_reused",
                )
                memory.events.append(event)
                events.append(event)
            binding = ChatBindingRecord(
                binding_id=f"b_{turn.turn_index:04d}_{len(memory.bindings) + 1:04d}",
                surface=surface,
                target_ref=f"memory:{memory_id}",
                source_turn_index=turn.turn_index,
                confidence=0.7,
                metadata={"source": "memory_surface"},
            )
            memory.bindings.append(binding)
            event = ChatMemoryEvent(
                event_type="binding_added",
                turn_index=turn.turn_index,
                run_id=turn.run_id,
                memory_id=memory_id,
                binding_id=binding.binding_id,
                operation="add",
                new_target_ref=binding.target_ref,
                reason="memory_surface",
                lucidity_decision=lucidity_decision,
                created_at=utc_now_iso(),
                details={"surface": surface},
            )
            memory.events.append(event)
            events.append(event)

    memory.unresolved_items = []
    return events


def memory_reply_for_text(text: str, memory: ChatSessionMemory) -> str:
    """Return a concise session-memory answer when the current text asks for it."""

    query_tokens = set(_significant_tokens(text))
    active = [item for item in memory.memories if item.status == "active"]
    if not active:
        return ""

    question_like = "?" in text or bool(query_tokens & {"what", "which", "who", "where", "when", "tell", "remind"})
    if not question_like:
        return ""

    best: tuple[int, ChatMemoryRecord] | None = None
    for item in active:
        content_tokens = set(_significant_tokens(str(item.content.get("text") or "")))
        binding_tokens: set[str] = set()
        for binding in memory.bindings:
            if binding.status == "active" and binding.target_ref == f"memory:{item.memory_id}":
                binding_tokens.update(_significant_tokens(binding.surface))
        score = len(query_tokens & (content_tokens | binding_tokens))
        if best is None or score > best[0] or (score == best[0] and item.source_turn_index > best[1].source_turn_index):
            best = (score, item)

    if best is None:
        return ""
    score, item = best
    if score <= 0 and not _asks_for_recent_session_memory(text):
        return ""
    answer = str(item.content.get("salient") or item.content.get("text") or "").strip()
    return answer[:1].upper() + answer[1:] if answer else ""


def to_session_state(
    record: ChatAuditRecord,
    memory: ChatSessionMemory | None = None,
    *,
    recent_turn_limit: int = 8,
) -> SessionState:
    selected_turns = record.turns[-recent_turn_limit:]
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
            for turn in selected_turns
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


def _extract_memory_directive(text: str) -> str:
    clean = _compact_text(text, limit=480)
    lowered = clean.lower()
    prefixes = (
        "remember that ",
        "remember ",
        "keep in mind that ",
        "keep in mind ",
        "note that ",
        "note ",
    )
    for prefix in prefixes:
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
    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "did",
        "do",
        "for",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "you",
    }
    return [
        token.lower()
        for token in re.findall(r"\b[A-Za-z0-9_'-]+\b", text)
        if token.lower() not in stop
    ]


def _binding_surfaces(text: str, *, limit: int = 8) -> list[str]:
    tokens = _significant_tokens(text)
    if not tokens:
        return []
    surfaces = tokens[:-1] if len(tokens) > 1 else tokens
    if not surfaces:
        surfaces = tokens
    return list(dict.fromkeys(surfaces))[:limit]


def _active_binding_for_surface(memory: ChatSessionMemory, surface: str) -> ChatBindingRecord | None:
    for binding in reversed(memory.bindings):
        if binding.status == "active" and binding.surface == surface:
            return binding
    return None


def _rebind_latest(
    memory: ChatSessionMemory,
    surfaces: list[str],
    memory_id: str,
    turn: ChatAuditTurn,
    lucidity_decision: str,
) -> ChatMemoryEvent:
    binding = memory.bindings[-1]
    for candidate in reversed(memory.bindings):
        if candidate.status == "active":
            binding = candidate
            break
    old_target = binding.target_ref
    binding.status = "superseded"
    if old_target.startswith("memory:"):
        old_id = old_target.split(":", 1)[1]
        for record in memory.memories:
            if record.memory_id == old_id:
                record.status = "superseded"
                break
    surface = binding.surface if len(surfaces) <= 1 else surfaces[0]
    new_binding = ChatBindingRecord(
        binding_id=f"b_{turn.turn_index:04d}_{len(memory.bindings) + 1:04d}",
        surface=surface,
        target_ref=f"memory:{memory_id}",
        source_turn_index=turn.turn_index,
        confidence=0.7,
        metadata={"source": "rebind_latest", "previous_binding_id": binding.binding_id},
    )
    memory.bindings.append(new_binding)
    return ChatMemoryEvent(
        event_type="binding_rebound",
        turn_index=turn.turn_index,
        run_id=turn.run_id,
        memory_id=memory_id,
        binding_id=new_binding.binding_id,
        operation="rebind",
        old_target_ref=old_target,
        new_target_ref=new_binding.target_ref,
        reason="latest_active_binding",
        lucidity_decision=lucidity_decision,
        created_at=utc_now_iso(),
        details={"surface": surface, "previous_binding_id": binding.binding_id},
    )


def _binding_event(
    event_type: str,
    binding: ChatBindingRecord,
    *,
    new_target_ref: str,
    turn: ChatAuditTurn,
    lucidity_decision: str,
    reason: str,
) -> ChatMemoryEvent:
    return ChatMemoryEvent(
        event_type=event_type,
        turn_index=turn.turn_index,
        run_id=turn.run_id,
        binding_id=binding.binding_id,
        operation="rebind",
        old_target_ref=binding.target_ref,
        new_target_ref=new_target_ref,
        reason=reason,
        lucidity_decision=lucidity_decision,
        created_at=utc_now_iso(),
        details={"surface": binding.surface},
    )


def _asks_for_recent_session_memory(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "what did i tell",
            "what did i say",
            "what value did i give",
            "remind me",
        )
    )


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
