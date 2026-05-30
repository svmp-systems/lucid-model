"""Checkpoint-backed module trainers."""

from lucid.training.trainers.base import ModuleTrainer, TrainingResult
from lucid.training.trainers.registry import get_trainer, trainer_names, trainer_registry

__all__ = [
    "ModuleTrainer",
    "TrainingResult",
    "get_trainer",
    "trainer_names",
    "trainer_registry",
]
