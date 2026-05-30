"""Registry for checkpoint-backed module trainers."""

from __future__ import annotations

from lucid.training.trainers.base import ModuleTrainer
from lucid.training.trainers.basins import BasinsTrainer
from lucid.training.trainers.binding import BindingTrainer
from lucid.training.trainers.context_op import ContextOpTrainer
from lucid.training.trainers.cue_encoder import CueEncoderTrainer
from lucid.training.trainers.decoder import DecoderTrainer
from lucid.training.trainers.dmf import DmfTrainer
from lucid.training.trainers.interference import InterferenceTrainer
from lucid.training.trainers.lucidity import LucidityTrainer
from lucid.training.trainers.perception import PerceptionTrainer
from lucid.training.trainers.projector import ProjectorTrainer


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
