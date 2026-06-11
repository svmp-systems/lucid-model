"""Decoder trainer from committed-state rendering targets."""

from __future__ import annotations

from pathlib import Path

from lucid.ir.training import Episode
from lucid.training.corpus import adapters
from lucid.training.checkpoint.store import CheckpointState
from lucid.training.modules.base import ModuleTrainer, TrainingResult, write_module_audit
from lucid.training.modules.utils import snapshot, upsert_pattern


class DecoderTrainer(ModuleTrainer):
    name = "decoder"
    store_name = "decoder_adapter"

    def train(
        self,
        episode: Episode,
        state: CheckpointState,
        audit_dir: Path,
    ) -> TrainingResult:
        store = state.ensure_store(self.store_name)
        before = snapshot(store)
        target = adapters.decoder_target(episode)
        expected = target["expected_answer"]
        if expected is None:
            return write_module_audit(
                audit_dir=audit_dir,
                module=self.name,
                episode=episode,
                action="DEFER",
                reason="episode_has_no_decoder_expected_answer",
                before=before,
                after=store,
                updated_objects=[],
                metrics={"has_expected_answer": False},
            )

        record, _created = upsert_pattern(
            store["render_targets"],
            {"template_id": episode.template_id, "episode_id": episode.episode_id},
            target,
        )
        pair = {
            "episode_id": episode.episode_id,
            "committed_state_hint": {
                "template_id": episode.template_id,
                "lucidity_target": target["lucidity_target"],
            },
            "corrected_output": expected,
            "update_scope": "decoder_only",
        }
        if pair not in store["correction_pairs"]:
            store["correction_pairs"].append(pair)

        after = snapshot(store)
        return write_module_audit(
            audit_dir=audit_dir,
            module=self.name,
            episode=episode,
            action="UPDATE",
            reason="stored_decoder_render_target",
            before=before,
            after=after,
            updated_objects=[str(record["episode_id"])],
            metrics={
                "has_expected_answer": True,
                "correction_pair_count": len(store["correction_pairs"]),
            },
        )
