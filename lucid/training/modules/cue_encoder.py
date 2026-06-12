"""Cue encoder trainer with seed and calibrate modes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lucid.cognition.input.cue import CueEncoderConfig, encode_cues, measure_cue_recall, normalize_cue_key
from lucid.ir.training import Episode
from lucid.training.corpus import adapters
from lucid.training.checkpoint.store import CheckpointState
from lucid.training.modules.base import ModuleTrainer, TrainingResult, write_module_audit
from lucid.training.modules.utils import snapshot, upsert_pattern

_RELATION_FAMILIES = frozenset(
    {
        "object_carryover",
        "temporal_subordinate",
        "coordination",
        "contrast",
        "locative_relation_like",
        "coreference_like",
    }
)


def _upsert_index_entry(
    index: dict[str, list[dict[str, Any]]],
    feature_key: str,
    entry: dict[str, Any],
) -> None:
    records = index.setdefault(feature_key, [])
    for record in records:
        if record.get("cue_key") == entry.get("cue_key"):
            record["weight"] = max(float(record.get("weight", 0.0)), float(entry["weight"]))
            record["seen_count"] = int(record.get("seen_count", 0)) + 1
            pattern = entry.get("feature_pattern")
            if isinstance(pattern, list) and pattern:
                merged = sorted(set(record.get("feature_pattern") or []) | set(pattern))
                record["feature_pattern"] = merged
            examples = record.setdefault("episode_ids", [])
            if entry["episode_id"] not in examples:
                examples.append(entry["episode_id"])
            return
    records.append({**entry, "seen_count": 1, "episode_ids": [entry["episode_id"]]})


def _feature_keys_for_target(episode: Episode, target: dict[str, Any]) -> list[str]:
    evidence_ref = str(target.get("evidence_ref") or "")
    span_by_id = {span.span_id: span for span in episode.gold.spans}
    span = span_by_id.get(evidence_ref)
    keys: list[str] = []
    if span is not None:
        surface = normalize_cue_key(span.surface)
        kind = normalize_cue_key(span.kind_hint)
        if surface:
            keys.append(f"surface:{surface}")
            for flag in episode.gold.uncertainty_flags:
                if flag.target_id == span.span_id:
                    keys.append(
                        f"uncertainty:{normalize_cue_key(flag.uncertainty_type)}:surface:{surface}"
                    )
        if kind:
            keys.append(f"kind:{kind}")

    trace_family = str(target.get("trace_family") or "")
    inferred = {
        "position_shift_like": "change:position_shift",
        "shape_preserved_like": "grid:shape_preserved",
        "color_preserved_like": "grid:color_preserved",
    }.get(trace_family)
    if inferred:
        keys.append(inferred)
    return sorted(set(keys))


def _relation_feature_keys(episode: Episode, target: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    trace_family = normalize_cue_key(str(target.get("trace_family") or ""))
    if trace_family in _RELATION_FAMILIES:
        keys.append(f"reference:{trace_family}")
    for marker in episode.gold.markers:
        surface = normalize_cue_key(marker.surface)
        if surface:
            keys.append(f"marker_surface:{surface}")
        for hint in marker.marker_type_hints:
            normalized = normalize_cue_key(hint)
            if normalized:
                keys.append(f"marker_type:{normalized}")
    return sorted(set(keys))


def _route_entry(
    episode: Episode,
    target: dict[str, Any],
    *,
    feature_pattern: list[str],
    source: str,
) -> dict[str, Any]:
    return {
        "cue_key": str(target["trace_family"]),
        "weight": float(target["weight"]),
        "preserve_as_alternative": bool(target["keep_alive"]),
        "feature_pattern": sorted(set(feature_pattern)),
        "source": source,
        "episode_id": episode.episode_id,
        "template_id": episode.template_id,
    }


def _store_target_routes(
    store: dict[str, Any],
    episode: Episode,
    targets: list[dict[str, Any]],
    *,
    source: str,
) -> tuple[list[str], list[str]]:
    feature_keys: list[str] = []
    relation_keys: list[str] = []
    store.setdefault("feature_index", {})
    store.setdefault("relation_index", {})

    for target in targets:
        family = str(target["trace_family"])
        primitive_keys = _feature_keys_for_target(episode, target)
        relation_route_keys = _relation_feature_keys(episode, target)
        entry = _route_entry(
            episode,
            target,
            feature_pattern=primitive_keys or [f"trace:{normalize_cue_key(family)}"],
            source=source,
        )
        for feature_key in primitive_keys:
            _upsert_index_entry(store["feature_index"], feature_key, entry)
            feature_keys.append(feature_key)
        if family in _RELATION_FAMILIES or relation_route_keys:
            rel_entry = _route_entry(
                episode,
                target,
                feature_pattern=relation_route_keys or primitive_keys,
                source=source,
            )
            for feature_key in relation_route_keys or [f"reference:{normalize_cue_key(family)}"]:
                _upsert_index_entry(store["relation_index"], feature_key, rel_entry)
                relation_keys.append(feature_key)
    return feature_keys, relation_keys


class CueEncoderTrainer(ModuleTrainer):
    name = "cue_encoder"
    store_name = "cue_encoder_map"

    def train(
        self,
        episode: Episode,
        state: CheckpointState,
        audit_dir: Path,
    ) -> TrainingResult:
        store = state.ensure_store(self.store_name)
        before = snapshot(store)
        targets = adapters.cue_encoder_targets(episode)
        trace_targets = targets["trace_targets"]
        if not trace_targets:
            return write_module_audit(
                audit_dir=audit_dir,
                module=self.name,
                episode=episode,
                action="DEFER",
                reason="episode_has_no_cue_targets",
                before=before,
                after=store,
                updated_objects=[],
                metrics={"trace_target_count": 0},
            )

        mode = str(episode.meta.get("train_mode") or "calibrate")
        gold_families = {str(target["trace_family"]) for target in trace_targets}
        recall_metrics: dict[str, Any] = {}
        missing: set[str] = set()

        if mode == "calibrate":
            cue_input = adapters.episode_to_cue_encoder_input(episode)
            cloud = encode_cues(cue_input, config=CueEncoderConfig(cue_map=store))
            recall_metrics = measure_cue_recall(cloud, gold_families)
            missing = set(recall_metrics.get("missing") or [])
            if not missing and float(recall_metrics.get("recall", 0.0)) >= 0.999:
                record, _created = upsert_pattern(
                    store["cue_targets"],
                    {"template_id": episode.template_id, "episode_id": episode.episode_id},
                    targets,
                )
                return write_module_audit(
                    audit_dir=audit_dir,
                    module=self.name,
                    episode=episode,
                    action="NO_UPDATE",
                    reason="cue_recall_already_sufficient",
                    before=before,
                    after=store,
                    updated_objects=[str(record["episode_id"])],
                    metrics={
                        "train_mode": mode,
                        "trace_target_count": len(trace_targets),
                        **recall_metrics,
                    },
                )

        record, _created = upsert_pattern(
            store["cue_targets"],
            {"template_id": episode.template_id, "episode_id": episode.episode_id},
            targets,
        )
        patch_targets = trace_targets
        source = "episode_gold"
        if mode == "calibrate" and missing:
            patch_targets = [target for target in trace_targets if str(target["trace_family"]) in missing]
            source = "calibrate_missing_route"

        feature_keys, relation_keys = _store_target_routes(
            store,
            episode,
            patch_targets,
            source=source,
        )

        after = snapshot(store)
        return write_module_audit(
            audit_dir=audit_dir,
            module=self.name,
            episode=episode,
            action="UPDATE",
            reason="stored_high_recall_cue_targets" if mode == "seed" else "patched_missing_cue_routes",
            before=before,
            after=after,
            updated_objects=[str(record["episode_id"])],
            metrics={
                "train_mode": mode,
                "trace_target_count": len(trace_targets),
                "patched_target_count": len(patch_targets),
                "indexed_feature_count": len(set(feature_keys)),
                "indexed_relation_count": len(set(relation_keys)),
                **recall_metrics,
            },
        )
