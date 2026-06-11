"""Human-readable checkpoint registry (cp_001, cp_002, …)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lucid.runtime.paths import resolve_train_path

REGISTRY_FILE = "checkpoints/saves/registry.json"
_CP_NAME = re.compile(r"^cp_\d{3,}$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def registry_path() -> Path:
    return resolve_train_path(REGISTRY_FILE)


def _empty_registry() -> dict[str, Any]:
    return {"schema_version": 1, "next_index": 1, "checkpoints": []}


def load_registry() -> dict[str, Any]:
    path = registry_path()
    if not path.is_file():
        return _empty_registry()
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("schema_version", 1)
    data.setdefault("next_index", 1)
    data.setdefault("checkpoints", [])
    return data


def save_registry(data: dict[str, Any]) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sanitize_checkpoint_name(name: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name.strip())
    return clean.strip("_") or "cp_unnamed"


def is_standard_checkpoint_name(name: str) -> bool:
    return bool(_CP_NAME.match(name))


def allocate_standard_name() -> str:
    registry = load_registry()
    index = max(1, int(registry.get("next_index", 1)))
    name = f"cp_{index:03d}"
    registry["next_index"] = index + 1
    save_registry(registry)
    return name


def register_checkpoint(
    *,
    name: str,
    path: Path,
    label: str = "",
    command: str = "",
    summary: dict[str, Any],
) -> dict[str, Any]:
    registry = load_registry()
    record = {
        "name": name,
        "path": str(path),
        "created_at": _utc_now_iso(),
        "label": label.strip(),
        "command": command.strip(),
        "checkpoint_id": summary.get("checkpoint_id", name),
        "training_steps": summary.get("training_steps", 0),
        "store_counts": summary.get("store_counts", {}),
    }
    checkpoints: list[dict[str, Any]] = [
        row for row in registry.get("checkpoints", []) if row.get("name") != name
    ]
    checkpoints.append(record)
    checkpoints.sort(key=lambda row: row.get("name", ""))
    registry["checkpoints"] = checkpoints
    save_registry(registry)
    return record


def list_registry() -> list[dict[str, Any]]:
    return list(load_registry().get("checkpoints", []))


def lookup_registry(name: str) -> dict[str, Any] | None:
    clean = sanitize_checkpoint_name(name)
    for row in list_registry():
        if row.get("name") == clean:
            return row
    return None
