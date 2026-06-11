"""Edit tracebank and basin_bank rows inside a checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lucid.audit.logger import content_hash
from lucid.runtime.paths import DEFAULT_AUDIT_MEMORY, resolve_train_path
from lucid.training.checkpoint.store import STORE_FILES, load_checkpoint, save_checkpoint


@dataclass(slots=True)
class MemoryEditResult:
    store: str
    record_id: str
    action: str
    before_hash: str
    after_hash: str
    audit_path: str


def _write_edit_audit(
    audit_dir: Path,
    *,
    store: str,
    record_id: str,
    before: Any,
    after: Any,
) -> Path:
    audit_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "store": store,
        "record_id": record_id,
        "before_hash": content_hash(before),
        "after_hash": content_hash(after),
        "before": before,
        "after": after,
    }
    path = audit_dir / f"edit_{store}_{record_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    readme = audit_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "memory edit audit",
                f"store: {store}",
                f"record_id: {record_id}",
                f"file: {path.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def edit_trace(
    checkpoint: str | Path,
    trace_id: str,
    *,
    patch: dict[str, Any],
    audit_dir: str | Path | None = None,
) -> MemoryEditResult:
    state = load_checkpoint(checkpoint, create=True)
    store = state.ensure_store("tracebank")
    before_rows = json.loads(json.dumps(store.get("records", [])))
    record = next((row for row in store["records"] if str(row.get("trace_id")) == trace_id), None)
    if record is None:
        raise KeyError(f"trace not found: {trace_id}")
    before = json.loads(json.dumps(record))
    record.update(patch)
    after = json.loads(json.dumps(record))
    save_checkpoint(state, checkpoint, force=True)
    audit_path = _write_edit_audit(
        resolve_train_path(audit_dir or DEFAULT_AUDIT_MEMORY),
        store="tracebank",
        record_id=trace_id,
        before=before,
        after=after,
    )
    return MemoryEditResult(
        store="tracebank",
        record_id=trace_id,
        action="update",
        before_hash=content_hash(before),
        after_hash=content_hash(after),
        audit_path=str(audit_path),
    )


def edit_basin(
    checkpoint: str | Path,
    basin_id: str,
    *,
    patch: dict[str, Any],
    audit_dir: str | Path | None = None,
) -> MemoryEditResult:
    state = load_checkpoint(checkpoint, create=True)
    store = state.ensure_store("basin_bank")
    record = next((row for row in store["records"] if str(row.get("basin_id")) == basin_id), None)
    if record is None:
        raise KeyError(f"basin not found: {basin_id}")
    before = json.loads(json.dumps(record))
    record.update(patch)
    after = json.loads(json.dumps(record))
    save_checkpoint(state, checkpoint, force=True)
    audit_path = _write_edit_audit(
        resolve_train_path(audit_dir or DEFAULT_AUDIT_MEMORY),
        store="basin_bank",
        record_id=basin_id,
        before=before,
        after=after,
    )
    return MemoryEditResult(
        store="basin_bank",
        record_id=basin_id,
        action="update",
        before_hash=content_hash(before),
        after_hash=content_hash(after),
        audit_path=str(audit_path),
    )


def list_traces(checkpoint: str | Path) -> list[dict[str, Any]]:
    root = resolve_train_path(checkpoint)
    path = root / STORE_FILES["tracebank"]
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("records", [])


def list_basins(checkpoint: str | Path) -> list[dict[str, Any]]:
    root = resolve_train_path(checkpoint)
    path = root / STORE_FILES["basin_bank"]
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("records", [])
