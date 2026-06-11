"""Projector trainer from verifiable projection examples."""

from __future__ import annotations

from pathlib import Path

from lucid.ir.training import Episode
from lucid.training.corpus import adapters
from lucid.training.checkpoint.store import CheckpointState
from lucid.training.modules.base import ModuleTrainer, TrainingResult, write_module_audit
from lucid.training.modules.utils import snapshot, upsert_pattern


class ProjectorTrainer(ModuleTrainer):
    name = "projector"
    store_name = "projector_examples"

    def train(
        self,
        episode: Episode,
        state: CheckpointState,
        audit_dir: Path,
    ) -> TrainingResult:
        store = state.ensure_store(self.store_name)
        before = snapshot(store)
        target = adapters.projector_target(episode)
        if not target:
            return write_module_audit(
                audit_dir=audit_dir,
                module=self.name,
                episode=episode,
                action="DEFER",
                reason="episode_has_no_projector_target",
                before=before,
                after=store,
                updated_objects=[],
                metrics={"example_count": 0},
            )

        record, _created = upsert_pattern(
            store["examples"],
            {"template_id": episode.template_id, "episode_id": episode.episode_id},
            target,
        )
        after = snapshot(store)
        return write_module_audit(
            audit_dir=audit_dir,
            module=self.name,
            episode=episode,
            action="UPDATE",
            reason="stored_projection_training_example",
            before=before,
            after=after,
            updated_objects=[str(record["episode_id"])],
            metrics={"example_count": len(store["examples"])},
        )
