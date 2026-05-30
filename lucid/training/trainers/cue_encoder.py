"""Cue encoder trainer from high-recall trace targets."""

from __future__ import annotations

from pathlib import Path

from lucid.ir.training import Episode
from lucid.training import adapters
from lucid.training.checkpoints import CheckpointState
from lucid.training.trainers.base import ModuleTrainer, TrainingResult, write_module_audit
from lucid.training.trainers.utils import snapshot, upsert_pattern


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

        record, _created = upsert_pattern(
            store["cue_targets"],
            {"template_id": episode.template_id, "episode_id": episode.episode_id},
            targets,
        )
        after = snapshot(store)
        return write_module_audit(
            audit_dir=audit_dir,
            module=self.name,
            episode=episode,
            action="UPDATE",
            reason="stored_high_recall_cue_targets",
            before=before,
            after=after,
            updated_objects=[str(record["episode_id"])],
            metrics={"trace_target_count": len(trace_targets)},
        )
