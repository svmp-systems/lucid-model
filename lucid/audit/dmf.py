"""Durable audit records for DMF trace updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4
from typing import Any

from lucid.audit.logger import content_hash
from lucid.ir.serde import to_dict, to_json
from lucid.memory.dmf import DmfAuditEvent

SCHEMA_VERSION = 1


@dataclass(slots=True)
class DmfTraceUpdateRecord:
    event: DmfAuditEvent
    trace_before: dict[str, Any] | None = None
    trace_after: dict[str, Any] | None = None
    tracebank_snapshot_before: str = ""
    tracebank_snapshot_after: str = ""
    summary: dict[str, Any] = field(default_factory=dict)


def _safe_path_part(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return (clean.strip("_") or "dmf")[:80]


class DmfUpdateAuditLogger:
    """Write one readable JSON file per DMF learning or lifecycle event."""

    def __init__(self, base_dir: Path | str = "audit/dmf") -> None:
        self.base_dir = Path(base_dir)

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

        event_dir = self.base_dir / _safe_path_part(event.event_type)
        event_dir.mkdir(parents=True, exist_ok=True)
        path = event_dir / f"{uuid4()}.json"
        event.audit_path = str(path)
        payload = {"schema_version": SCHEMA_VERSION, **to_dict(record)}
        path.write_text(to_json(payload), encoding="utf-8")
        return path
