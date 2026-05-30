"""DMF tracebank trainer."""

from __future__ import annotations

from pathlib import Path

from lucid.ir.training import Episode
from lucid.training import adapters
from lucid.training.checkpoints import CheckpointState
from lucid.training.trainers.base import ModuleTrainer, TrainingResult, write_module_audit
from lucid.training.trainers.utils import find_record, next_id, snapshot


class DmfTrainer(ModuleTrainer):
    name = "dmf"
    store_name = "tracebank"

    def train(
        self,
        episode: Episode,
        state: CheckpointState,
        audit_dir: Path,
    ) -> TrainingResult:
        store = state.ensure_store(self.store_name)
        before = snapshot(store)
        targets = adapters.dmf_targets(episode)
        if not targets:
            return write_module_audit(
                audit_dir=audit_dir,
                module=self.name,
                episode=episode,
                action="DEFER",
                reason="episode_has_no_trace_targets",
                before=before,
                after=store,
                updated_objects=[],
                metrics={"target_count": 0},
            )

        updated: list[str] = []
        for target in targets:
            family = target["trace_family"]
            record = find_record(store["records"], "trace_family", family)
            if record is None:
                record = {
                    "trace_id": next_id(store, "t"),
                    "trace_family": family,
                    "alias": family,
                    "cue_affinities": {},
                    "created_from_episodes": [],
                    "activation_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "maturity_state": "provisional",
                    "heat_tier": "hot",
                }
                store["records"].append(record)
            old = float(record["cue_affinities"].get(family, 0.0))
            weight = float(target["weight"])
            record["cue_affinities"][family] = min(1.0, max(old, old + 0.2 * weight))
            record["activation_count"] = int(record.get("activation_count", 0)) + 1
            if episode.episode_id not in record["created_from_episodes"]:
                record["created_from_episodes"].append(episode.episode_id)
            updated.append(str(record["trace_id"]))

        after = snapshot(store)
        return write_module_audit(
            audit_dir=audit_dir,
            module=self.name,
            episode=episode,
            action="UPDATE",
            reason="reinforced_trace_targets_from_episode_gold",
            before=before,
            after=after,
            updated_objects=sorted(set(updated)),
            metrics={
                "target_count": len(targets),
                "trace_count": len(store["records"]),
            },
        )
