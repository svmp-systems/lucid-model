"""Durable audit records for DMF trace updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lucid.audit.direct_run import safe_token, utc_now_iso
from lucid.audit.logger import content_hash
from lucid.audit.sanitize import sanitize_audit_value
from lucid.ir.serde import to_dict, to_json
from lucid.cognition.memory.dmf import DmfAuditEvent
from lucid.runtime.paths import DEFAULT_AUDIT_DMF, resolve_train_path

SCHEMA_VERSION = 2


@dataclass(slots=True)
class DmfTraceUpdateRecord:
    event: DmfAuditEvent
    trace_before: dict[str, Any] | None = None
    trace_after: dict[str, Any] | None = None
    tracebank_snapshot_before: str = ""
    tracebank_snapshot_after: str = ""
    summary: dict[str, Any] = field(default_factory=dict)


class DmfUpdateAuditLogger:
    """Write one readable JSON file per DMF learning or lifecycle event."""

    def __init__(self, base_dir: Path | str = DEFAULT_AUDIT_DMF) -> None:
        self.base_dir = resolve_train_path(base_dir, mkdir=True)

    def write_event(
        self,
        event: DmfAuditEvent,
        *,
        trace_before: Any = None,
        trace_after: Any = None,
        tracebank_snapshot_before: str = "",
        tracebank_snapshot_after: str = "",
    ) -> Path:
        before = to_dict(trace_before)
        after = to_dict(trace_after)
        event.before_hash = content_hash(before) if before is not None else ""
        event.after_hash = content_hash(after) if after is not None else ""

        record = DmfTraceUpdateRecord(
            event=event,
            trace_before=before,
            trace_after=after,
            tracebank_snapshot_before=tracebank_snapshot_before,
            tracebank_snapshot_after=tracebank_snapshot_after,
            summary={
                "headline": event.summary,
                "lines": [
                    f"event_type: {event.event_type}",
                    f"trace_index: {event.trace_index}",
                    f"trace_id_before: {event.trace_id_before or '-'}",
                    f"trace_id_after: {event.trace_id_after or '-'}",
                    f"cue_keys: {', '.join(event.cue_keys) or '-'}",
                    f"before_hash: {event.before_hash or '-'}",
                    f"after_hash: {event.after_hash or '-'}",
                ],
            },
        )

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        event_token = safe_token(event.event_type, max_len=40)
        file_name = f"{stamp}_{event_token}.json"
        event_dir = self.base_dir / event_token
        event_dir.mkdir(parents=True, exist_ok=True)
        path = event_dir / file_name
        event.audit_path = str(path)
        payload = sanitize_audit_value(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "smoke",
                "created_at": utc_now_iso(),
                "run_id": file_name.removesuffix(".json"),
                "module": "dmf",
                "event_type": event.event_type,
                **to_dict(record),
            }
        )
        path.write_text(to_json(payload), encoding="utf-8")
        return path
