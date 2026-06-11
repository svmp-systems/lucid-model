"""Perception trainer from generated evidence labels."""

from __future__ import annotations

from pathlib import Path

from lucid.ir.training import Episode
from lucid.training.corpus import adapters
from lucid.training.checkpoint.store import CheckpointState
from lucid.training.modules.base import ModuleTrainer, TrainingResult, write_module_audit
from lucid.training.modules.utils import snapshot, upsert_pattern


class PerceptionTrainer(ModuleTrainer):
    name = "perception"
    store_name = "perception_examples"

    def train(
        self,
        episode: Episode,
        state: CheckpointState,
        audit_dir: Path,
    ) -> TrainingResult:
        store = state.ensure_store(self.store_name)
        before = snapshot(store)
        targets = adapters.perception_targets(episode)
        if not any(targets.values()):
            return write_module_audit(
                audit_dir=audit_dir,
                module=self.name,
                episode=episode,
                action="DEFER",
                reason="episode_has_no_perception_gold",
                before=before,
                after=store,
                updated_objects=[],
                metrics={"span_count": 0, "marker_count": 0, "region_count": 0},
            )

        record, _created = upsert_pattern(
            store["examples"],
            {"template_id": episode.template_id, "episode_id": episode.episode_id},
            {"raw_input": episode.raw_input, "targets": targets},
        )
        after = snapshot(store)
        return write_module_audit(
            audit_dir=audit_dir,
            module=self.name,
            episode=episode,
            action="UPDATE",
            reason="stored_evidence_extraction_targets",
            before=before,
            after=after,
            updated_objects=[str(record["episode_id"])],
            metrics={
                "span_count": len(targets["spans"]),
                "marker_count": len(targets["markers"]),
                "region_count": len(targets["regions"]),
            },
        )
