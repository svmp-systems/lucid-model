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

ACTIVE = "active"
PROVISIONAL = "provisional"
STABILIZED = "stabilized"


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


def _source_ref_count(source_refs: list[dict[str, Any]] | list[str] | None) -> int:
    refs: set[str] = set()
    for ref in source_refs or []:
        if isinstance(ref, dict):
            token = str(ref.get("ref_id") or ref.get("url") or ref.get("title") or "").strip()
        else:
            token = str(ref).strip()
        if token:
            refs.add(token)
    return len(refs)


def source_backed_shadow_promotion(
    state: CheckpointState,
    object_id: str,
    object_type: str,
    *,
    source_refs: list[dict[str, Any]] | list[str] | None = None,
    support_count: int = 0,
    trust_score: float = 0.0,
    source: str = "source_backed_replay",
    precision_tier: str = "uint8_sparse",
) -> dict[str, Any]:
    """Promote source-backed imports after deterministic replay evidence.

    The object still enters through quarantine/probation first. A warm runtime
    tier requires repeated support or multiple trusted sources, so one-off
    scraped claims do not become directly committable.
    """

    record = ensure_metadata(
        state,
        object_id,
        object_type,
        source=source,
        precision_tier=precision_tier,
        source_refs=[
            ref if isinstance(ref, dict) else {"ref_id": str(ref)}
            for ref in source_refs or []
            if str(ref)
        ],
    )
    source_count = _source_ref_count(source_refs)
    support = max(int(record.get("support_count", 0)), int(support_count), source_count)
    record["support_count"] = support
    if source_count:
        record["shadow_pass_count"] = max(int(record.get("shadow_pass_count", 0)), 1)

    contradictions = int(record.get("contradiction_count", 0))
    if contradictions:
        record["heat_tier"] = QUARANTINE
        record["commit_permission"] = SUPPORT_ONLY
    elif support >= 3 or (source_count >= 2 and trust_score >= 0.78) or (
        source_count >= 1 and trust_score >= 0.8
    ):
        record["heat_tier"] = WARM
        record["commit_permission"] = NORMAL_SUPPORT
    elif source_count > 0:
        record["heat_tier"] = PROBATION
        record["commit_permission"] = SUPPORT_ONLY
    else:
        record["heat_tier"] = QUARANTINE
        record["commit_permission"] = SUPPORT_ONLY

    record["trust_score"] = max(float(record.get("trust_score", 0.0) or 0.0), float(trust_score or 0.0))
    record["promotion_reason"] = (
        f"source_backed_shadow_replay:sources={source_count}:support={support}:trust={trust_score:.3f}"
    )
    record["updated_at"] = utc_now_iso()
    return record


def runtime_heat_tier(metadata: dict[str, Any]) -> str:
    return str(metadata.get("heat_tier") or QUARANTINE)


def runtime_maturity_state(metadata: dict[str, Any]) -> str:
    tier = runtime_heat_tier(metadata)
    if tier in {WARM, HOT, COLD}:
        return ACTIVE
    if tier == ARCHIVED:
        return STABILIZED
    return PROVISIONAL


def apply_runtime_promotion_fields(
    record: dict[str, Any],
    metadata: dict[str, Any],
    *,
    has_maturity: bool = False,
) -> dict[str, Any]:
    record["heat_tier"] = runtime_heat_tier(metadata)
    if has_maturity:
        record["maturity_state"] = runtime_maturity_state(metadata)
    record["commit_permission"] = str(metadata.get("commit_permission") or SUPPORT_ONLY)
    record["promotion_reason"] = str(metadata.get("promotion_reason") or "")
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
