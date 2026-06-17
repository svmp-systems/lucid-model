"""Checkpoint sharding and keyed-store compaction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lucid.audit.logger import content_hash
from lucid.runtime.paths import resolve_train_path
from lucid.training.checkpoint.store import STORE_FILES, load_checkpoint, save_checkpoint

DEFAULT_MAX_ITEMS_PER_SHARD = 256
SHARD_ROOT = "_shards"

SHARDABLE_STORES: dict[str, tuple[str, str]] = {
    "tracebank": ("records", "trace_id"),
    "basin_bank": ("records", "basin_id"),
    "operator_bank": ("operators", "operator_id"),
    "relation_aliases": ("aliases", "alias_id"),
    "concept_bank": ("concepts", "concept_id"),
    "perception_examples": ("examples", "episode_id"),
    "projector_examples": ("examples", "episode_id"),
}


def compact_store_payload(
    store_name: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    item_field, id_field = SHARDABLE_STORES[store_name]
    rows = payload.get(item_field)
    if not isinstance(rows, list):
        return dict(payload), {
            "item_field": item_field,
            "id_field": id_field,
            "item_count": 0,
            "deduped_item_count": 0,
            "anonymous_item_count": 0,
        }

    positions: dict[str, int] = {}
    compacted_rows: list[dict[str, Any]] = []
    anonymous_count = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        row_id = str(row.get(id_field) or "").strip()
        if row_id:
            key = f"id:{row_id}"
        else:
            key = f"anonymous:{content_hash(row)}:{index}"
            anonymous_count += 1
        if key in positions:
            existing = compacted_rows[positions[key]]
            compacted_rows[positions[key]] = {**existing, **row}
        else:
            positions[key] = len(compacted_rows)
            compacted_rows.append(dict(row))

    compacted = dict(payload)
    compacted[item_field] = compacted_rows
    return compacted, {
        "item_field": item_field,
        "id_field": id_field,
        "item_count": len(rows),
        "compacted_item_count": len(compacted_rows),
        "deduped_item_count": len(rows) - len(compacted_rows),
        "anonymous_item_count": anonymous_count,
    }


def compact_checkpoint(
    path: str | Path,
    *,
    max_items_per_shard: int = DEFAULT_MAX_ITEMS_PER_SHARD,
    stores: list[str] | None = None,
) -> dict[str, Any]:
    """Compact keyed stores and write manifest-indexed shard sidecars."""

    root = resolve_train_path(path)
    state = load_checkpoint(root, create=False)
    selected = [store for store in (stores or list(SHARDABLE_STORES)) if store in SHARDABLE_STORES]
    if not selected:
        raise ValueError("no shardable checkpoint stores selected")

    shard_manifest: dict[str, Any] = dict(state.manifest.get("checkpoint_shards", {}))
    store_summaries: dict[str, Any] = {}
    for store_name in selected:
        payload = state.ensure_store(store_name)
        if not isinstance(payload, dict):
            continue
        compacted, compact_metrics = compact_store_payload(store_name, payload)
        state.stores[store_name] = compacted
        shard_index = write_store_shards(
            root,
            store_name,
            compacted,
            max_items_per_shard=max_items_per_shard,
        )
        shard_manifest[store_name] = {
            "store_name": store_name,
            "store_file": STORE_FILES.get(store_name, f"{store_name}.json"),
            "index_file": f"{SHARD_ROOT}/{store_name}/index.json",
            "item_field": shard_index["item_field"],
            "id_field": shard_index["id_field"],
            "item_count": shard_index["item_count"],
            "shard_count": shard_index["shard_count"],
            "max_items_per_shard": shard_index["max_items_per_shard"],
            "payload_hash": shard_index["payload_hash"],
            "compaction_policy": "keyed_last_write_wins_sidecar_shards",
        }
        store_summaries[store_name] = {**compact_metrics, **shard_manifest[store_name]}

    state.manifest["checkpoint_shards"] = shard_manifest
    save_checkpoint(state, root, force=True)
    return {
        "checkpoint": str(root),
        "max_items_per_shard": max(1, int(max_items_per_shard)),
        "stores": store_summaries,
    }


def write_store_shards(
    checkpoint: str | Path,
    store_name: str,
    payload: dict[str, Any],
    *,
    max_items_per_shard: int = DEFAULT_MAX_ITEMS_PER_SHARD,
) -> dict[str, Any]:
    if store_name not in SHARDABLE_STORES:
        raise ValueError(f"store is not shardable: {store_name}")
    item_field, id_field = SHARDABLE_STORES[store_name]
    rows = payload.get(item_field)
    if not isinstance(rows, list):
        rows = []

    shard_size = max(1, int(max_items_per_shard))
    root = resolve_train_path(checkpoint, mkdir=True)
    shard_dir = root / SHARD_ROOT / store_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    for old_shard in shard_dir.glob("shard_*.json"):
        old_shard.unlink()

    base_payload = {key: value for key, value in payload.items() if key != item_field}
    base_path = shard_dir / "base.json"
    base_path.write_text(json.dumps(base_payload, indent=2, sort_keys=True), encoding="utf-8")

    shard_rows: list[dict[str, Any]] = []
    for index in range(0, len(rows), shard_size):
        chunk = [row for row in rows[index : index + shard_size] if isinstance(row, dict)]
        shard_name = f"shard_{len(shard_rows) + 1:05d}.json"
        shard_payload = {
            "schema_version": 1,
            "store_name": store_name,
            "item_field": item_field,
            "items": chunk,
        }
        shard_path = shard_dir / shard_name
        shard_path.write_text(json.dumps(shard_payload, indent=2, sort_keys=True), encoding="utf-8")
        shard_rows.append(
            {
                "file": shard_name,
                "item_count": len(chunk),
                "first_id": _row_id(chunk[0], id_field) if chunk else "",
                "last_id": _row_id(chunk[-1], id_field) if chunk else "",
                "content_hash": content_hash(shard_payload),
            }
        )

    index_payload = {
        "schema_version": 1,
        "store_name": store_name,
        "store_file": STORE_FILES.get(store_name, f"{store_name}.json"),
        "base_file": "base.json",
        "base_hash": content_hash(base_payload),
        "item_field": item_field,
        "id_field": id_field,
        "item_count": len([row for row in rows if isinstance(row, dict)]),
        "shard_count": len(shard_rows),
        "max_items_per_shard": shard_size,
        "payload_hash": content_hash(payload),
        "shards": shard_rows,
    }
    (shard_dir / "index.json").write_text(
        json.dumps(index_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return index_payload


def load_store_from_shards(checkpoint: str | Path, store_name: str) -> dict[str, Any]:
    if store_name not in SHARDABLE_STORES:
        raise ValueError(f"store is not shardable: {store_name}")
    item_field, _id_field = SHARDABLE_STORES[store_name]
    root = resolve_train_path(checkpoint)
    index_path = root / SHARD_ROOT / store_name / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing shard index: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    base_file = str(index.get("base_file") or "base.json")
    base_path = index_path.parent / base_file
    base = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}
    if not isinstance(base, dict):
        base = {}

    items: list[dict[str, Any]] = []
    for shard in index.get("shards", []):
        if not isinstance(shard, dict):
            continue
        shard_file = str(shard.get("file") or "")
        if not shard_file:
            continue
        shard_path = index_path.parent / shard_file
        shard_payload = json.loads(shard_path.read_text(encoding="utf-8"))
        shard_items = shard_payload.get("items") if isinstance(shard_payload, dict) else None
        if isinstance(shard_items, list):
            items.extend(row for row in shard_items if isinstance(row, dict))

    payload = dict(base)
    payload[item_field] = items
    return payload


def _row_id(row: dict[str, Any], id_field: str) -> str:
    return str(row.get(id_field) or "").strip()
