"""Basin trainer from generated basin-family targets."""

from __future__ import annotations

from pathlib import Path

from lucid.ir.training import Episode
from lucid.training.corpus import adapters
from lucid.training.checkpoint.store import CheckpointState
from lucid.training.modules.base import ModuleTrainer, TrainingResult, write_module_audit
from lucid.training.modules.utils import find_record, next_id, snapshot


class BasinsTrainer(ModuleTrainer):
    name = "basins"
    store_name = "basin_bank"

    def train(
        self,
        episode: Episode,
        state: CheckpointState,
        audit_dir: Path,
    ) -> TrainingResult:
        store = state.ensure_store(self.store_name)
        before = snapshot(store)
        targets = adapters.basin_targets(episode)
        if not targets:
            return write_module_audit(
                audit_dir=audit_dir,
                module=self.name,
                episode=episode,
                action="DEFER",
                reason="episode_has_no_basin_targets",
                before=before,
                after=store,
                updated_objects=[],
                metrics={"basin_target_count": 0},
            )

        updated: list[str] = []
        for target in targets:
            family = target["family_hint"]
            record = find_record(store["records"], "family_hint", family)
            if record is None:
                record = {
                    "basin_id": next_id(store, "b"),
                    "family_hint": family,
                    "frame_affinities": {},
                    "support_examples": [],
                }
                store["records"].append(record)
            frame_id = target["frame_id"] or "unscoped"
            old = float(record["frame_affinities"].get(frame_id, 0.0))
            record["frame_affinities"][frame_id] = min(1.0, max(old, target["confidence"]))
            if episode.episode_id not in record["support_examples"]:
                record["support_examples"].append(episode.episode_id)
            updated.append(str(record["basin_id"]))

        after = snapshot(store)
        return write_module_audit(
            audit_dir=audit_dir,
            module=self.name,
            episode=episode,
            action="UPDATE",
            reason="stored_basin_family_targets",
            before=before,
            after=after,
            updated_objects=sorted(set(updated)),
            metrics={"basin_target_count": len(targets), "basin_count": len(store["records"])},
        )
