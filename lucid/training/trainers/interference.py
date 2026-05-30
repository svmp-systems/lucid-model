"""Interference trainer from generated support/block gates."""

from __future__ import annotations

from pathlib import Path

from lucid.ir.training import Episode
from lucid.training import adapters
from lucid.training.checkpoints import CheckpointState
from lucid.training.trainers.base import ModuleTrainer, TrainingResult, write_module_audit
from lucid.training.trainers.utils import snapshot, upsert_pattern


class InterferenceTrainer(ModuleTrainer):
    name = "interference"
    store_name = "interference_graph"

    def train(
        self,
        episode: Episode,
        state: CheckpointState,
        audit_dir: Path,
    ) -> TrainingResult:
        store = state.ensure_store(self.store_name)
        before = snapshot(store)
        gates = adapters.interference_targets(episode)
        if not gates:
            return write_module_audit(
                audit_dir=audit_dir,
                module=self.name,
                episode=episode,
                action="DEFER",
                reason="episode_has_no_interference_gates",
                before=before,
                after=store,
                updated_objects=[],
                metrics={"gate_count": 0},
            )

        updated: list[str] = []
        for gate in gates:
            record, _created = upsert_pattern(
                store["gates"],
                {
                    "template_id": episode.template_id,
                    "gate_id": gate["gate_id"],
                    "scope_frame_id": gate["scope_frame_id"],
                },
                gate,
            )
            updated.append(str(record["gate_id"]))
        after = snapshot(store)
        return write_module_audit(
            audit_dir=audit_dir,
            module=self.name,
            episode=episode,
            action="UPDATE",
            reason="stored_support_block_gate_targets",
            before=before,
            after=after,
            updated_objects=updated,
            metrics={"gate_count": len(gates)},
        )
