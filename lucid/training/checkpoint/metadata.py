"""Shared metadata for learned checkpoint objects."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from lucid.training.checkpoint.store import CheckpointState

QUARANTINE = "quarantine"
PROBATION = "probation"
WARM = "warm"
HOT = "hot"
COLD = "cold"
ARCHIVED = "archived"

SUPPORT_ONLY = "support_only"
NORMAL_SUPPORT = "normal_support"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_metadata(
    object_id: str,
    object_type: str,
    *,
    source: str = "",
    heat_tier: str = QUARANTINE,
    precision_tier: str = "fp32",
    commit_permission: str = SUPPORT_ONLY,
    source_refs: list[dict[str, Any]] | None = None,
    audit_refs: list[str] | None = None,
) -> dict[str, Any]:
    now = utc_now_iso()
    return {
        "object_id": object_id,
        "object_type": object_type,
        "source": source,
        "created_at": now,
        "updated_at": now,
        "heat_tier": heat_tier,
        "precision_tier": precision_tier,
        "commit_permission": commit_permission,
        "support_count": 0,
        "contradiction_count": 0,
        "target_fix_count": 0,
        "shadow_pass_count": 0,
        "canary_pass_rate": 0.0,
        "last_failed_replay": "",
        "source_refs": list(source_refs or []),
        "audit_refs": list(audit_refs or []),
        "quantization_candidate": False,
    }


def metadata_store(state: CheckpointState) -> dict[str, Any]:
    store = state.ensure_store("learned_metadata")
    store.setdefault("objects", {})
    return store


def ensure_metadata(
    state: CheckpointState,
    object_id: str,
    object_type: str,
    **kwargs: Any,
) -> dict[str, Any]:
    objects = metadata_store(state)["objects"]
    if object_id not in objects:
        objects[object_id] = default_metadata(object_id, object_type, **kwargs)
    else:
        objects[object_id].setdefault("object_id", object_id)
        objects[object_id].setdefault("object_type", object_type)
        objects[object_id]["updated_at"] = utc_now_iso()
    return objects[object_id]


def record_support(state: CheckpointState, object_id: str, object_type: str) -> dict[str, Any]:
    record = ensure_metadata(state, object_id, object_type)
    record["support_count"] = int(record.get("support_count", 0)) + 1
    record["updated_at"] = utc_now_iso()
    return record


def record_contradiction(
    state: CheckpointState,
    object_id: str,
    object_type: str,
    *,
    replay_id: str = "",
) -> dict[str, Any]:
    record = ensure_metadata(state, object_id, object_type)
    record["contradiction_count"] = int(record.get("contradiction_count", 0)) + 1
    if replay_id:
        record["last_failed_replay"] = replay_id
    record["heat_tier"] = QUARANTINE
    record["commit_permission"] = SUPPORT_ONLY
    record["updated_at"] = utc_now_iso()
    return record


def promote_heat_tier(state: CheckpointState, object_id: str, object_type: str) -> dict[str, Any]:
    record = ensure_metadata(state, object_id, object_type)
    tier = str(record.get("heat_tier") or QUARANTINE)
    support = int(record.get("support_count", 0))
    contradictions = int(record.get("contradiction_count", 0))
    shadow = int(record.get("shadow_pass_count", 0))

    if contradictions:
        next_tier = QUARANTINE
        permission = SUPPORT_ONLY
    elif tier == QUARANTINE and shadow >= 1:
        next_tier = PROBATION
        permission = SUPPORT_ONLY
    elif tier in {QUARANTINE, PROBATION} and support >= 3:
        next_tier = WARM
        permission = NORMAL_SUPPORT
    elif tier == WARM and support >= 8:
        next_tier = HOT
        permission = NORMAL_SUPPORT
    elif tier == HOT and support >= 16:
        next_tier = COLD
        permission = NORMAL_SUPPORT
        record["quantization_candidate"] = True
    else:
        next_tier = tier
        permission = str(record.get("commit_permission") or SUPPORT_ONLY)

    record["heat_tier"] = next_tier
    record["commit_permission"] = permission
    record["updated_at"] = utc_now_iso()
    return record
