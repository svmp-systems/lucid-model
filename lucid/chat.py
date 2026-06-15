"""Session-by-session chat runner for the Lucid pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from lucid.audit.chat import (
    ChatAuditTurn,
    append_chat_turn,
    build_session_context,
    load_chat_record,
    load_session_memory,
    memory_reply_for_text,
    new_session_memory,
    refresh_history_views,
    save_chat_record,
    save_session_memory,
    session_file,
    to_session_state,
    update_session_memory_from_turn,
)
from lucid.audit.logger import content_hash
from lucid.cognition.input.perception import PerceptionConfig
from lucid.cognition.memory.dmf import DynamicMemoryField, trace_record_from_store
from lucid.cognition.pipe_orchestrator.runner import OrchestratorConfig, OrchestratorRunner
from lucid.ir.common import Modality, TaskIntent
from lucid.ir.serde import to_dict
from lucid.ir.training import Episode
from lucid.training.checkpoint.slots import resolve_inference_checkpoint
from lucid.training.checkpoint.store import load_checkpoint, save_checkpoint
from lucid.training.learn.dmf import apply_lucidity_trace_feedback, learn_from_episode
from lucid.runtime.paths import resolve_train_path


@dataclass(slots=True)
class ChatTurnResult:
    session_id: str
    turn_index: int
    run_id: str
    assistant_output: str
    session_audit_path: str
    run_audit_dir: str
    dmf_learning: dict[str, object] | None = None


def start_session(*, session_id: str | None = None, audit_dir: str | Path = "audit/chat") -> str:
    sid = session_id or f"chat-{uuid4()}"
    record = load_chat_record(audit_dir, sid)
    if record is None:
        from lucid.audit.chat import new_chat_record

        save_chat_record(audit_dir, new_chat_record(sid))
    if load_session_memory(audit_dir, sid) is None:
        save_session_memory(audit_dir, new_session_memory(sid))
    return sid


def list_sessions(*, audit_dir: str | Path = "audit/chat") -> list[str]:
    root = resolve_train_path(audit_dir)
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if (path / "session.json").exists())


def run_chat_turn(
    text: str,
    *,
    session_id: str,
    audit_dir: str | Path = "audit/chat",
    perception_backend: str = "",
    checkpoint: str = "",
    learn_to_dmf: bool = False,
    learning_rate: float = 0.2,
) -> ChatTurnResult:
    if not text.strip():
        raise ValueError("chat input cannot be empty")
    if learn_to_dmf and not checkpoint:
        raise ValueError("chat DMF learning requires --checkpoint")
    runtime_checkpoint = _resolve_chat_runtime_checkpoint(checkpoint)

    record = load_chat_record(audit_dir, session_id)
    if record is None:
        start_session(session_id=session_id, audit_dir=audit_dir)
        record = load_chat_record(audit_dir, session_id)
    assert record is not None
    memory = load_session_memory(audit_dir, session_id) or new_session_memory(session_id)

    turn_index = len(record.turns) + 1
    refresh_history_views(record)
    session_context = build_session_context(record, memory, current_turn_index=turn_index)

    control_reply = _chat_control_reply(text, record)
    if control_reply:
        turn = ChatAuditTurn(
            turn_index=turn_index,
            run_id=f"{session_id}-turn-{turn_index:04d}-chat-control",
            user_input=text,
            assistant_output=control_reply,
            run_audit_dir="",
            lucidity_decision="commit",
            dmf_learning={},
            response_source="chat_control",
        )
        events = update_session_memory_from_turn(memory, turn, lucidity_decision="commit")
        turn.memory_events = [to_dict(event) for event in events]
        save_session_memory(audit_dir, memory)
        updated = append_chat_turn(audit_dir, session_id=session_id, turn=turn)
        return ChatTurnResult(
            session_id=session_id,
            turn_index=turn_index,
            run_id=turn.run_id,
            assistant_output=control_reply,
            session_audit_path=str(session_file(audit_dir, updated.session_id)),
            run_audit_dir="",
            dmf_learning=None,
        )

    episode = Episode(
        episode_id=f"{session_id}-turn-{turn_index:04d}",
        modality=Modality.TEXT,
        raw_input=text,
        task_intent=TaskIntent.CHAT,
        context={
            "session_id": session_id,
            "turn_index": turn_index,
            "session_context": session_context,
            "previous_turns": session_context["recent_turns"],
            "working_memory": [to_dict(item) for item in record.working_memory],
            "active_memories": session_context["active_memories"],
            "active_bindings": session_context["active_bindings"],
            "unclear_items": session_context["unresolved_items"],
        },
        meta={"source": "lucid.chat"},
    )

    perception_cfg = PerceptionConfig.from_env()
    if perception_backend:
        perception_cfg.backend = perception_backend
    runner = OrchestratorRunner(
        config=OrchestratorConfig(
            audit_base_dir=str(audit_dir),
            perception=perception_cfg,
            checkpoint=runtime_checkpoint,
        )
    )
    run = runner.run_episode(
        episode,
        session_id=session_id,
        turn_index=turn_index,
        session_state=to_session_state(record, memory),
    )
    assistant_output = run.decoder_output.surface_text if run.decoder_output is not None else ""
    if not assistant_output:
        assistant_output = "(no decoder output)"

    response_source = "pipeline"
    memory_reply = memory_reply_for_text(text, memory)
    if (
        response_source == "pipeline"
        and memory_reply
        and _pipeline_output_needs_memory_fallback(assistant_output, user_input=text)
    ):
        assistant_output = memory_reply
        response_source = "session_memory"

    lucidity_decision = run.lucidity_output.decision.value if run.lucidity_output is not None else ""

    dmf_learning: dict[str, object] | None = None
    if learn_to_dmf:
        dmf_learning = _learn_turn_into_dmf(
            checkpoint=runtime_checkpoint or checkpoint,
            audit_dir=audit_dir,
            session_id=session_id,
            turn_index=turn_index,
            run_id=run.context.run_id,
            cue_cloud=run.cue_cloud,
            passed_lucidity=lucidity_decision == "commit",
            learning_rate=learning_rate,
        )

    turn = ChatAuditTurn(
        turn_index=turn_index,
        run_id=run.context.run_id,
        user_input=text,
        assistant_output=assistant_output,
        run_audit_dir=run.context.audit_dir,
        lucidity_decision=lucidity_decision,
        dmf_learning=dmf_learning or {},
        response_source=response_source,
    )
    events = update_session_memory_from_turn(memory, turn, lucidity_decision=lucidity_decision)
    turn.memory_events = [to_dict(event) for event in events]
    save_session_memory(audit_dir, memory)

    updated = append_chat_turn(audit_dir, session_id=session_id, turn=turn)
    return ChatTurnResult(
        session_id=session_id,
        turn_index=turn_index,
        run_id=run.context.run_id,
        assistant_output=assistant_output,
        session_audit_path=str(session_file(audit_dir, updated.session_id)),
        run_audit_dir=run.context.audit_dir,
        dmf_learning=dmf_learning,
    )


def _resolve_chat_runtime_checkpoint(checkpoint: str) -> str:
    text = (checkpoint or "").strip()
    if not text:
        return ""
    return resolve_inference_checkpoint(text) or text


def _learn_turn_into_dmf(
    *,
    checkpoint: str,
    audit_dir: str | Path,
    session_id: str,
    turn_index: int,
    run_id: str,
    cue_cloud: object,
    passed_lucidity: bool,
    learning_rate: float,
) -> dict[str, object]:
    if cue_cloud is None:
        return {"action": "defer", "reason": "missing_cue_cloud", "checkpoint": checkpoint}

    state = load_checkpoint(checkpoint, create=True)
    tracebank_store = state.ensure_store("tracebank")
    records = tracebank_store.get("records", [])
    if not isinstance(records, list):
        records = []
        tracebank_store["records"] = records

    dmf = DynamicMemoryField(
        tracebank=[trace_record_from_store(record) for record in records if isinstance(record, dict)],
        audit_base_dir=Path(audit_dir) / session_id / "dmf_learning",
    )
    before_hash = content_hash(tracebank_store)
    updated = learn_from_episode(dmf, cue_cloud, learning_rate=learning_rate, spawn_if_novel=True)  # type: ignore[arg-type]
    promoted = apply_lucidity_trace_feedback(dmf, updated, passed_lucidity=passed_lucidity)

    tracebank_store["records"] = [_trace_record_to_store(trace) for trace in dmf.tracebank]
    tracebank_store["next_id"] = _next_store_id(tracebank_store["records"])
    save_checkpoint(state, checkpoint, step_delta=1)
    after_hash = content_hash(tracebank_store)
    return {
        "action": "update",
        "checkpoint": checkpoint,
        "session_id": session_id,
        "turn_index": turn_index,
        "run_id": run_id,
        "updated_trace_indices": updated,
        "promoted_trace_indices": promoted,
        "passed_lucidity": passed_lucidity,
        "learning_rate": learning_rate,
        "tracebank_hash_before": before_hash,
        "tracebank_hash_after": after_hash,
        "tracebank_size": len(tracebank_store["records"]),
    }


def _trace_record_to_store(trace: object) -> dict[str, object]:
    data = to_dict(trace)
    if not isinstance(data, dict):
        return {}
    alias = str(data.get("alias") or "").strip()
    if alias and not data.get("trace_family"):
        data["trace_family"] = alias
    return data


def _next_store_id(records: list[object]) -> int:
    max_seen = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        raw = str(record.get("trace_id") or "").strip()
        if len(raw) > 1 and raw[0] == "t" and raw[1:].isdigit():
            max_seen = max(max_seen, int(raw[1:]))
    return max_seen + 1


def _pipeline_output_needs_memory_fallback(text: str, *, user_input: str = "") -> bool:
    stripped = text.strip()
    if not stripped or stripped == "(no decoder output)":
        return True
    lowered = stripped.lower()
    if (
        "no approved text content was available" in lowered
        or "not confident enough to answer" in lowered
        or lowered.startswith("(holding:")
    ):
        return True
    if user_input:
        user_words = _word_tokens(user_input)
        out_words = _word_tokens(lowered)
        if user_words and out_words:
            overlap = len(set(user_words) & set(out_words)) / max(len(set(out_words)), 1)
            if overlap >= 0.6 and len(out_words) <= len(user_words) + 3:
                return True
    return False


def _word_tokens(text: str) -> list[str]:
    normalized = re.sub(r"[^\w\s]", " ", text.lower())
    return [token for token in normalized.split() if token]


_ACK_RE = re.compile(
    r"^(oh\s+)?(ok|okay|cool|great|nice|got it|understood|makes sense|alright|fine)$",
    re.I,
)
_CONFUSION_RE = re.compile(
    r"^(huh|uhh|wait what|what|what do you mean|i do not get it|i dont get it|confused)$",
    re.I,
)
_WHY_ANSWER_RE = re.compile(
    r"\bwhy\s+(did|do|are)\s+(it|you)\s+(answer|respond|say)\b|\bwhy did it answer\b",
    re.I,
)


def _normalize_control_text(text: str) -> str:
    normalized = re.sub(r"[^\w\s']", " ", text.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _chat_control_reply(text: str, record: object) -> str:
    _ = record
    normalized = _normalize_control_text(text)
    if not normalized:
        return ""
    if _WHY_ANSWER_RE.search(normalized):
        return (
            "I treated the last short turn as a new request. That should have been a "
            "clarification or defer instead."
        )
    if _CONFUSION_RE.match(normalized):
        return "I may have answered too eagerly. What part should I clarify?"
    if _ACK_RE.match(normalized):
        return "Okay."
    return ""
