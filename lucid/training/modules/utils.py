"""Small helpers for checkpoint-backed trainers."""

from __future__ import annotations

import copy
from typing import Any


def snapshot(value: Any) -> Any:
    return copy.deepcopy(value)


def increment_counter(mapping: dict[str, int], key: str, amount: int = 1) -> None:
    mapping[key] = int(mapping.get(key, 0)) + amount


def find_record(records: list[dict], key: str, value: str) -> dict | None:
    for record in records:
        if str(record.get(key, "")) == value:
            return record
    return None


def next_id(store: dict, prefix: str) -> str:
    raw = int(store.get("next_id", 1))
    store["next_id"] = raw + 1
    return f"{prefix}{raw:04d}"


def upsert_pattern(records: list[dict], identity: dict, payload: dict) -> tuple[dict, bool]:
    for record in records:
        if all(record.get(key) == value for key, value in identity.items()):
            record.update(payload)
            record["seen_count"] = int(record.get("seen_count", 0)) + 1
            return record, False
    record = {**identity, **payload, "seen_count": 1}
    records.append(record)
    return record, True
