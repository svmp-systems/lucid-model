"""Context-op trainer from scope and gate labels."""

from __future__ import annotations

from pathlib import Path

from lucid.ir.training import Episode
from lucid.training.corpus import adapters
from lucid.training.checkpoint.store import CheckpointState
from lucid.training.modules.base import ModuleTrainer, TrainingResult, write_module_audit
from lucid.training.modules.utils import snapshot, upsert_pattern


class ContextOpTrainer(ModuleTrainer):
    name = "context-op"
    store_name = "context_policy"

    def train(
        self,
        episode: Episode,
        state: CheckpointState,
        audit_dir: Path,
    ) -> TrainingResult:
        store = state.ensure_store(self.store_name)
        before = snapshot(store)
        targets = adapters.context_targets(episode)
        scope_targets = targets["scope_assignments"]
        gate_targets = targets["interference_gates"]
        if not scope_targets and not gate_targets:
            return write_module_audit(
                audit_dir=audit_dir,
                module=self.name,
                episode=episode,
                action="DEFER",
                reason="episode_has_no_context_targets",
                before=before,
                after=store,
                updated_objects=[],
                metrics={"scope_assignment_count": 0, "gate_count": 0},
            )

        updated: list[str] = []
        for target in scope_targets:
            record, _created = upsert_pattern(
                store["scope_patterns"],
                {
                    "template_id": episode.template_id,
                    "span_id": target["span_id"],
                    "primary_frame": target["primary_frame"],
                },
                target,
            )
            updated.append(f"scope:{record['span_id']}->{record['primary_frame']}")
        for target in gate_targets:
            record, _created = upsert_pattern(
                store["gate_patterns"],
                {
                    "template_id": episode.template_id,
                    "gate_id": target["gate_id"],
                    "scope_frame_id": target["scope_frame_id"],
                },
                target,
            )
            updated.append(f"gate:{record['gate_id']}")

        after = snapshot(store)
        return write_module_audit(
            audit_dir=audit_dir,
            module=self.name,
            episode=episode,
            action="UPDATE",
            reason="stored_scope_and_gate_policy_targets",
            before=before,
            after=after,
            updated_objects=updated,
            metrics={
                "scope_assignment_count": len(scope_targets),
                "gate_count": len(gate_targets),
            },
        )
