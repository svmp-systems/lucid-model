"""Binding trainer from frame targets and scope assignments."""

from __future__ import annotations

from pathlib import Path

from lucid.cognition.input.cue.encoder import normalize_cue_key
from lucid.ir.training import Episode
from lucid.training.corpus import adapters
from lucid.training.checkpoint.store import CheckpointState
from lucid.training.modules.base import ModuleTrainer, TrainingResult, write_module_audit
from lucid.training.modules.utils import snapshot, upsert_pattern


class BindingTrainer(ModuleTrainer):
    name = "binding"
    store_name = "binding_affordances"

    def train(
        self,
        episode: Episode,
        state: CheckpointState,
        audit_dir: Path,
    ) -> TrainingResult:
        store = state.ensure_store(self.store_name)
        before = snapshot(store)
        frame_targets = adapters.binding_frame_targets(episode)
        scope_targets = adapters.binding_targets(episode)
        if not frame_targets and not scope_targets:
            return write_module_audit(
                audit_dir=audit_dir,
                module=self.name,
                episode=episode,
                action="DEFER",
                reason="episode_has_no_binding_targets",
                before=before,
                after=store,
                updated_objects=[],
                metrics={"frame_target_count": 0, "scope_assignment_count": 0},
            )

        updated: list[str] = []
        span_by_id = {span.span_id: span for span in episode.gold.spans}

        for target in frame_targets:
            slot_targets = target.get("slot_targets") or []
            if slot_targets:
                for slot_target in slot_targets:
                    trace_family = str(slot_target.get("trace_family") or "")
                    hints = dict(slot_target.get("affinity_hints") or {})
                    if not hints:
                        hints = {"slot_member_like": 1.0}
                    for span_id in slot_target.get("member_span_ids") or []:
                        span = span_by_id.get(span_id)
                        if span is None:
                            continue
                        feature_key = _span_feature_key(span)
                        for hint, hint_weight in hints.items():
                            hint_name = normalize_cue_key(str(hint))
                            if not hint_name:
                                continue
                            weight = max(
                                float(slot_target.get("confidence") or 0.0),
                                float(hint_weight),
                            )
                            identity = {
                                "pattern_type": "slot_affinity",
                                "feature_key": feature_key,
                                "frame_type": target["frame_type"],
                                "slot_hint": hint_name,
                            }
                            existing = _find_pattern(store["patterns"], identity)
                            if existing is not None:
                                weight = max(float(existing.get("weight", 0.0)), weight)
                            record, _created = upsert_pattern(
                                store["patterns"],
                                identity,
                                {
                                    "weight": weight,
                                    "trace_family": trace_family,
                                    "template_id": episode.template_id,
                                },
                            )
                            updated.append(
                                f"{record['feature_key']}~{record['slot_hint']}"
                            )
            else:
                for span_id in target["member_span_ids"]:
                    span = span_by_id.get(span_id)
                    if span is None:
                        continue
                    feature_key = _span_feature_key(span)
                    identity = {
                        "pattern_type": "frame_member",
                        "feature_key": feature_key,
                        "frame_type": target["frame_type"],
                    }
                    record, _created = upsert_pattern(
                        store["patterns"],
                        identity,
                        {
                            "weight": float(target["confidence"]),
                            "template_id": episode.template_id,
                        },
                    )
                    updated.append(f"{record['feature_key']}~frame_member")

            region_hint = _region_hint_for_frame(target["frame_id"])
            if region_hint:
                hints = store.setdefault("region_frame_hints", {})
                hints[region_hint] = target["frame_id"]
                updated.append(f"region:{region_hint}->{target['frame_id']}")

        for target in scope_targets:
            identity = {
                "template_id": episode.template_id,
                "span_id": target["span_id"],
                "primary_frame": target["primary_frame"],
            }
            record, _created = upsert_pattern(store["patterns"], identity, target)
            updated.append(f"{record['span_id']}->{record['primary_frame']}")

        after = snapshot(store)
        return write_module_audit(
            audit_dir=audit_dir,
            module=self.name,
            episode=episode,
            action="UPDATE",
            reason="stored_binding_affordances_and_scope_patterns",
            before=before,
            after=after,
            updated_objects=sorted(set(updated)),
            metrics={
                "frame_target_count": len(frame_targets),
                "scope_assignment_count": len(scope_targets),
                "pattern_count": len(store.get("patterns", [])),
            },
        )


def _find_pattern(patterns: list[dict], identity: dict) -> dict | None:
    for record in patterns:
        if all(record.get(key) == value for key, value in identity.items()):
            return record
    return None


def _span_feature_key(span: object) -> str:
    kind = normalize_cue_key(getattr(span, "kind_hint", "") or "span")
    surface = normalize_cue_key(getattr(span, "surface", ""))
    return f"unit:{kind}:{surface}"


def _region_hint_for_frame(frame_id: str) -> str:
    if frame_id == "event_one":
        return "main_clause"
    if frame_id == "event_two":
        return "relative_clause"
    return ""
