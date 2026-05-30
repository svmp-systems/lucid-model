"""Binding trainer from generated frame/scope targets."""

from __future__ import annotations

from pathlib import Path

from lucid.ir.training import Episode
from lucid.training import adapters
from lucid.training.checkpoints import CheckpointState
from lucid.training.trainers.base import ModuleTrainer, TrainingResult, write_module_audit
from lucid.training.trainers.utils import snapshot, upsert_pattern


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
        targets = adapters.binding_targets(episode)
        if not targets:
            return write_module_audit(
                audit_dir=audit_dir,
                module=self.name,
                episode=episode,
                action="DEFER",
                reason="episode_has_no_binding_scope_targets",
                before=before,
                after=store,
                updated_objects=[],
                metrics={"assignment_count": 0},
            )

        updated: list[str] = []
        for target in targets:
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
            reason="stored_trace_role_scope_affordances",
            before=before,
            after=after,
            updated_objects=updated,
            metrics={"assignment_count": len(targets)},
        )
