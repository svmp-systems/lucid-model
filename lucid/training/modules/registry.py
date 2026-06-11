"""Registry for checkpoint-backed module trainers."""

from __future__ import annotations

from lucid.training.modules.base import ModuleTrainer
from lucid.training.modules.basins import BasinsTrainer
from lucid.training.modules.binding import BindingTrainer
from lucid.training.modules.context_op import ContextOpTrainer
from lucid.training.modules.cue_encoder import CueEncoderTrainer
from lucid.training.modules.decoder import DecoderTrainer
from lucid.training.modules.dmf import DmfTrainer
from lucid.training.modules.interference import InterferenceTrainer
from lucid.training.modules.lucidity import LucidityTrainer
from lucid.training.modules.perception import PerceptionTrainer
from lucid.training.modules.projector import ProjectorTrainer


def trainer_registry() -> dict[str, ModuleTrainer]:
    trainers: list[ModuleTrainer] = [
        PerceptionTrainer(),
        CueEncoderTrainer(),
        DmfTrainer(),
        BindingTrainer(),
        ContextOpTrainer(),
        InterferenceTrainer(),
        BasinsTrainer(),
        LucidityTrainer(),
        ProjectorTrainer(),
        DecoderTrainer(),
    ]
    return {trainer.name: trainer for trainer in trainers}


def get_trainer(name: str) -> ModuleTrainer:
    registry = trainer_registry()
    key = name.strip()
    if key not in registry:
        known = ", ".join(sorted(registry))
        raise KeyError(f"unknown trainer {name!r}; known: {known}")
    return registry[key]


def trainer_names() -> list[str]:
    return sorted(trainer_registry())
