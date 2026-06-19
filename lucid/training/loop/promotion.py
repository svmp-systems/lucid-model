"""Apply promoted training patches to a Lucid checkpoint."""

from __future__ import annotations

from pathlib import Path

from lucid.ir.serde import to_dict
from lucid.ir.training import Episode
from lucid.runtime.paths import DEFAULT_AUDIT_TRAINING_RUNS, resolve_checkpoint, resolve_train_path
from lucid.training.checkpoint.store import CheckpointState, load_checkpoint, save_checkpoint
from lucid.training.loop.orchestrator import Patch
from lucid.training.modules import get_trainer

PATCH_TYPE_TO_TRAINER: dict[str, str] = {
    "PerceptionPatch": "perception",
    "CueEncoderPatch": "cue_encoder",
    "TracePatch": "dmf",
    "BindingPatch": "binding",
    "ContextPatch": "context-op",
    "InterferencePatch": "interference",
    "BasinPatch": "basins",
    "LucidityPatch": "lucidity",
    "ProjectorPatch": "projector",
    "DecoderPatch": "decoder",
}


class CheckpointPromotionHook:
    """Persist promoted orchestrator patches through module trainers."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        audit_dir: str | Path = DEFAULT_AUDIT_TRAINING_RUNS,
    ) -> None:
        self.checkpoint_path = resolve_checkpoint(checkpoint_path)
        self.audit_dir = resolve_train_path(audit_dir)
        self.state: CheckpointState = load_checkpoint(self.checkpoint_path, create=True)

    def on_promote(self, patch: Patch, episode: Episode) -> None:
        trainer_name = PATCH_TYPE_TO_TRAINER.get(patch.patch_type, "")
        if not trainer_name:
            return
        trainer = get_trainer(trainer_name)
        step_dir = self.audit_dir / f"promote_{patch.patch_id[:8]}"
        trainer.train(episode, self.state, step_dir)
        save_checkpoint(self.state, self.checkpoint_path, force=True, step_delta=1)

    def summary(self) -> dict:
        from lucid.training.checkpoint.store import checkpoint_summary

        return checkpoint_summary(self.state)
