"""Lucidity/governor policy trainer from generated commit targets."""

from __future__ import annotations

from pathlib import Path

from lucid.ir.training import Episode
from lucid.training.corpus import adapters
from lucid.training.checkpoint.store import CheckpointState
from lucid.training.modules.base import ModuleTrainer, TrainingResult, write_module_audit
from lucid.training.modules.utils import increment_counter, snapshot


class LucidityTrainer(ModuleTrainer):
    name = "lucidity"
    store_name = "lucidity_policy"

    def train(
        self,
        episode: Episode,
        state: CheckpointState,
        audit_dir: Path,
    ) -> TrainingResult:
        store = state.ensure_store(self.store_name)
        before = snapshot(store)
        target = adapters.lucidity_target(episode)
        decision = str(target.get("decision") or "").strip()
        if not decision:
            return write_module_audit(
                audit_dir=audit_dir,
                module=self.name,
                episode=episode,
                action="DEFER",
                reason="episode_has_no_lucidity_target",
                before=before,
                after=store,
                updated_objects=[],
                metrics={"decision": ""},
            )

        increment_counter(store["decision_counts"], decision)
        template = episode.template_id or "unknown"
        template_decisions = store["template_decisions"].setdefault(template, {})
        increment_counter(template_decisions, decision)
        store.setdefault("rationales", {})[episode.episode_id] = target["rationale"]

        after = snapshot(store)
        return write_module_audit(
            audit_dir=audit_dir,
            module=self.name,
            episode=episode,
            action="UPDATE",
            reason="updated_lucidity_decision_statistics",
            before=before,
            after=after,
            updated_objects=[f"{template}:{decision}"],
            metrics={
                "decision": decision,
                "decision_count": store["decision_counts"][decision],
            },
        )
