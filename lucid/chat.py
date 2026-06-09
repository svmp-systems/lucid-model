"""Session-by-session chat runner for the Lucid pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from lucid.audit.logger import content_hash
from lucid.audit.chat import (
    ChatAuditTurn,
    append_chat_turn,
    load_chat_record,
    refresh_history_views,
    save_chat_record,
    session_file,
    to_session_state,
)
from lucid.cognition.input.perception import PerceptionConfig
from lucid.cognition.orchestrator.runner import OrchestratorConfig, OrchestratorRunner
from lucid.ir.common import Modality, TaskIntent
from lucid.ir.serde import to_dict
from lucid.ir.training import Episode
from lucid.memory.dmf import DynamicMemoryField, trace_record_from_store
from lucid.training.checkpoints import load_checkpoint, save_checkpoint
from lucid.training.dmf import apply_lucidity_trace_feedback, learn_from_episode


@dataclass(slots=True)
class ChatTurnResult:
    session_id: str
    turn_index: int
    run_id: str
    assistant_output: str
    session_audit_path: str
    run_audit_dir: str
    dmf_learning: dict[str, object] | None = None


def start_session(
    *,
    session_id: str | None = None,
    audit_dir: str | Path = "audit/chat",
) -> str:
    sid = session_id or f"chat-{uuid4()}"
    record = load_chat_record(audit_dir, sid)
    if record is None:
        from lucid.audit.chat import new_chat_record

        save_chat_record(audit_dir, new_chat_record(sid))
    return sid


def list_sessions(*, audit_dir: str | Path = "audit/chat") -> list[str]:
    root = Path(audit_dir)
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

    record = load_chat_record(audit_dir, session_id)
    if record is None:
        start_session(session_id=session_id, audit_dir=audit_dir)
        record = load_chat_record(audit_dir, session_id)
    assert record is not None

    turn_index = len(record.turns) + 1
    refresh_history_views(record)
    episode = Episode(
        episode_id=f"{session_id}-turn-{turn_index:04d}",
        modality=Modality.TEXT,
        raw_input=text,
        task_intent=TaskIntent.ANSWER,
        context={
            "session_id": session_id,
            "turn_index": turn_index,
            "previous_turns": [
                {
                    "turn_index": turn.turn_index,
                    "user_input": turn.user_input,
                    "assistant_output": turn.assistant_output,
                }
                for turn in record.turns
            ],
            "full_history": [
                {
                    "turn_index": turn.turn_index,
                    "user_input": turn.user_input,
                    "assistant_output": turn.assistant_output,
                    "lucidity_decision": turn.lucidity_decision,
                }
                for turn in record.turns
            ],
            "working_memory": [to_dict(item) for item in record.working_memory],
            "unclear_items": [to_dict(item) for item in record.unclear_items],
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
            checkpoint=checkpoint,
        )
    )
    run = runner.run_episode(
        episode,
        session_id=session_id,
        turn_index=turn_index,
        session_state=to_session_state(record),
    )
    assistant_output = ""
    if run.decoder_output is not None:
        assistant_output = run.decoder_output.surface_text
    if not assistant_output:
        assistant_output = "(no decoder output)"

    lucidity_decision = ""
    if run.lucidity_output is not None:
        lucidity_decision = run.lucidity_output.decision.value

    dmf_learning: dict[str, object] | None = None
    if learn_to_dmf:
        dmf_learning = _learn_turn_into_dmf(
            checkpoint=checkpoint,
            audit_dir=audit_dir,
            session_id=session_id,
            turn_index=turn_index,
            run_id=run.context.run_id,
            cue_cloud=run.cue_cloud,
            passed_lucidity=lucidity_decision == "commit",
            learning_rate=learning_rate,
        )

    updated = append_chat_turn(
        audit_dir,
        session_id=session_id,
        turn=ChatAuditTurn(
            turn_index=turn_index,
            run_id=run.context.run_id,
            user_input=text,
            assistant_output=assistant_output,
            run_audit_dir=run.context.audit_dir,
            lucidity_decision=lucidity_decision,
            dmf_learning=dmf_learning or {},
        ),
    )
    return ChatTurnResult(
        session_id=session_id,
        turn_index=turn_index,
        run_id=run.context.run_id,
        assistant_output=assistant_output,
        session_audit_path=str(session_file(audit_dir, updated.session_id)),
        run_audit_dir=run.context.audit_dir,
        dmf_learning=dmf_learning,
    )


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
        return {
            "action": "defer",
            "reason": "missing_cue_cloud",
            "checkpoint": checkpoint,
            "updated_trace_indices": [],
            "promoted_trace_indices": [],
        }

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
    updated = learn_from_episode(
        dmf,
        cue_cloud,  # type: ignore[arg-type]
        learning_rate=learning_rate,
        spawn_if_novel=True,
    )
    promoted = apply_lucidity_trace_feedback(
        dmf,
        updated,
        passed_lucidity=passed_lucidity,
    )

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
