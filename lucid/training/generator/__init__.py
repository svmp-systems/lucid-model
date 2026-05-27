"""Synthetic training episodes — engine + output + recipes."""

from lucid.training.generator.engine import AmbiguityKnob, generate, get_recipe, list_recipes
from lucid.training.generator.output import build_phase1_pack, read_episodes, write_episodes

__all__ = [
    "AmbiguityKnob",
    "build_phase1_pack",
    "generate",
    "get_recipe",
    "list_recipes",
    "read_episodes",
    "write_episodes",
]
