"""Checkpoint persistence for module and global training."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lucid.audit.logger import content_hash
from lucid.runtime.paths import DEFAULT_CHECKPOINT, resolve_train_path


STORE_FILES: dict[str, str] = {
    "perception_examples": "perception_examples.json",
    "cue_encoder_map": "cue_encoder_map.json",
    "tracebank": "tracebank.json",
    "basin_bank": "basin_bank.json",
    "interference_graph": "interference_graph.json",
    "binding_affordances": "binding_affordances.json",
    "learned_metadata": "learned_metadata.json",
    "operator_bank": "operator_bank.json",
    "relation_aliases": "relation_aliases.json",
    "concept_bank": "concept_bank.json",
    "context_policy": "context_policy.json",
    "lucidity_policy": "lucidity_policy.json",
    "projector_examples": "projector_examples.json",
    "decoder_adapter": "decoder_adapter.json",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _empty_store(name: str) -> Any:
    if name in {"tracebank", "basin_bank"}:
        return {"records": [], "next_id": 1}
    if name == "interference_graph":
        return {"gates": [], "edges": []}
    if name == "binding_affordances":
        return {
            "patterns": [],
            "region_frame_hints": {
                "main_clause": "event_one",
                "relative_clause": "event_two",
            },
        }
    if name == "learned_metadata":
        return {"objects": {}}
    if name == "operator_bank":
        return {"operators": []}
    if name == "relation_aliases":
        return {"aliases": []}
    if name == "concept_bank":
        return {"concepts": [], "sources": []}
    if name == "context_policy":
        return {"scope_patterns": [], "gate_patterns": []}
    if name == "lucidity_policy":
        return {"decision_counts": {}, "template_decisions": {}}
    if name == "decoder_adapter":
        return {"correction_pairs": [], "render_targets": []}
    if name == "perception_examples":
        return {"examples": []}
    if name == "cue_encoder_map":
        return {"cue_targets": [], "feature_index": {}, "relation_index": {}}
    if name == "projector_examples":
        return {"examples": []}
    return {}


@dataclass(slots=True)
class CheckpointState:
    checkpoint_id: str
    stores: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)

    def ensure_store(self, name: str) -> Any:
        if name not in self.stores:
            self.stores[name] = _empty_store(name)
        return self.stores[name]


def empty_checkpoint(checkpoint_id: str = "local") -> CheckpointState:
    return CheckpointState(
        checkpoint_id=checkpoint_id,
        stores={name: _empty_store(name) for name in STORE_FILES},
        manifest={
            "schema_version": 1,
            "checkpoint_id": checkpoint_id,
            "created_at": _utc_now_iso(),
            "updated_at": "",
            "store_files": STORE_FILES,
            "store_hashes": {},
            "training_steps": 0,
        },
    )


def load_checkpoint(path: str | Path = DEFAULT_CHECKPOINT, *, create: bool = True) -> CheckpointState:
    root = resolve_train_path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        if not create:
            raise FileNotFoundError(f"missing checkpoint manifest: {manifest_path}")
        return empty_checkpoint(root.name or "local")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stores: dict[str, Any] = {}
    for name, file_name in STORE_FILES.items():
        store_path = root / file_name
        if store_path.exists():
            stores[name] = json.loads(store_path.read_text(encoding="utf-8"))
        else:
            stores[name] = _empty_store(name)
    return CheckpointState(
        checkpoint_id=str(manifest.get("checkpoint_id") or root.name or "local"),
        stores=stores,
        manifest=manifest,
    )


def save_checkpoint(
    state: CheckpointState,
    path: str | Path,
    *,
    force: bool = False,
    step_delta: int = 0,
) -> Path:
    root = resolve_train_path(path, mkdir=True)
    manifest_path = root / "manifest.json"
    if root.exists() and any(root.iterdir()) and not manifest_path.exists() and not force:
        raise FileExistsError(
            f"{root} is not a Lucid checkpoint; pass --force to overwrite/create here"
        )
    root.mkdir(parents=True, exist_ok=True)

    for name in STORE_FILES:
        state.ensure_store(name)

    previous_hashes = dict(state.manifest.get("store_hashes", {}))
    store_hashes: dict[str, str] = {}
    store_bytes: dict[str, int] = {}
    changed_stores: list[str] = []
    written_store_bytes = 0
    for name, file_name in STORE_FILES.items():
        payload = state.stores[name]
        payload_text = json.dumps(payload, indent=2, sort_keys=True)
        payload_bytes = len(payload_text.encode("utf-8"))
        store_bytes[name] = payload_bytes
        store_hash = content_hash(payload)
        store_hashes[name] = store_hash
        store_path = root / file_name
        if previous_hashes.get(name) != store_hash or not store_path.exists():
            store_path.write_text(payload_text, encoding="utf-8")
            changed_stores.append(name)
            written_store_bytes += payload_bytes

    manifest = dict(state.manifest)
    manifest.setdefault("schema_version", 1)
    manifest.setdefault("checkpoint_id", state.checkpoint_id)
    manifest.setdefault("created_at", _utc_now_iso())
    manifest["updated_at"] = _utc_now_iso()
    manifest["store_files"] = STORE_FILES
    manifest["store_hashes"] = store_hashes
    manifest["checkpoint_scale_metrics"] = {
        "changed_store_count": len(changed_stores),
        "unchanged_store_count": len(STORE_FILES) - len(changed_stores),
        "changed_stores": changed_stores,
        "store_bytes": store_bytes,
        "total_store_bytes": sum(store_bytes.values()),
        "written_store_bytes": written_store_bytes,
        "rewrite_policy": "skip_unchanged_json_stores",
    }
    manifest["training_steps"] = int(manifest.get("training_steps", 0)) + max(0, step_delta)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    state.manifest = manifest
    return manifest_path


def _metadata_lifecycle_counts(state: CheckpointState) -> dict[str, Any]:
    objects = state.ensure_store("learned_metadata").get("objects", {})
    heat_tiers: dict[str, int] = {}
    object_types: dict[str, int] = {}
    quantization_candidates = 0
    if not isinstance(objects, dict):
        return {
            "objects": 0,
            "heat_tiers": {},
            "object_types": {},
            "quantization_candidates": 0,
        }
    for record in objects.values():
        if not isinstance(record, dict):
            continue
        tier = str(record.get("heat_tier") or "quarantine")
        object_type = str(record.get("object_type") or "unknown")
        heat_tiers[tier] = heat_tiers.get(tier, 0) + 1
        object_types[object_type] = object_types.get(object_type, 0) + 1
        if bool(record.get("quantization_candidate")):
            quantization_candidates += 1
    return {
        "objects": len(objects),
        "heat_tiers": dict(sorted(heat_tiers.items())),
        "object_types": dict(sorted(object_types.items())),
        "quantization_candidates": quantization_candidates,
    }


def checkpoint_summary(state: CheckpointState) -> dict[str, Any]:
    return {
        "checkpoint_id": state.checkpoint_id,
        "store_hashes": {
            name: content_hash(state.ensure_store(name)) for name in STORE_FILES
        },
        "store_counts": {
            "tracebank": len(state.ensure_store("tracebank").get("records", [])),
            "basin_bank": len(state.ensure_store("basin_bank").get("records", [])),
            "interference_gates": len(state.ensure_store("interference_graph").get("gates", [])),
            "binding_patterns": len(state.ensure_store("binding_affordances").get("patterns", [])),
            "learned_metadata_objects": len(
                state.ensure_store("learned_metadata").get("objects", {})
            ),
            "operators": len(state.ensure_store("operator_bank").get("operators", [])),
            "relation_aliases": len(state.ensure_store("relation_aliases").get("aliases", [])),
            "concepts": len(state.ensure_store("concept_bank").get("concepts", [])),
            "context_scope_patterns": len(
                state.ensure_store("context_policy").get("scope_patterns", [])
            ),
            "decoder_targets": len(state.ensure_store("decoder_adapter").get("render_targets", [])),
            "perception_examples": len(state.ensure_store("perception_examples").get("examples", [])),
            "cue_targets": len(state.ensure_store("cue_encoder_map").get("cue_targets", [])),
            "cue_feature_keys": len(state.ensure_store("cue_encoder_map").get("feature_index", {})),
            "projector_examples": len(state.ensure_store("projector_examples").get("examples", [])),
        },
        "metadata_lifecycle": _metadata_lifecycle_counts(state),
        "checkpoint_scale_metrics": dict(state.manifest.get("checkpoint_scale_metrics", {})),
        "checkpoint_shards": dict(state.manifest.get("checkpoint_shards", {})),
        "training_steps": int(state.manifest.get("training_steps", 0)),
    }
