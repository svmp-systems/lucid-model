"""Small grid: one cell changes color, position stays."""

from __future__ import annotations

import uuid
from random import Random

from lucid.ir.common import Modality, TaskIntent
from lucid.ir.training import Episode, GoldLabels, TraceTarget
from lucid.training.generator.engine import AmbiguityKnob, require_valid

NAME = "grid_recolor"
MODALITY = "grid"

SIZE = 6
COLORS = [1, 2, 3, 4, 5]


def _blank(rows: int, cols: int) -> list[list[int]]:
    return [[0] * cols for _ in range(rows)]


def make(rng: Random, knob: AmbiguityKnob) -> Episode:
    color_before = rng.choice(COLORS)
    color_after = rng.choice([c for c in COLORS if c != color_before])
    row, col = rng.randint(0, SIZE - 1), rng.randint(0, SIZE - 1)

    input_grid = _blank(SIZE, SIZE)
    output_grid = _blank(SIZE, SIZE)
    input_grid[row][col] = color_before
    output_grid[row][col] = color_after

    gold = GoldLabels(
        trace_activations=[
            TraceTarget("recolor_like", 0.9),
            TraceTarget("position_preserved_like", 0.92),
        ],
        lucidity_target="COMMIT",
        lucidity_rationale="color changed at fixed cell",
        expected_answer=output_grid,
        validator_result=True,
    )

    episode = Episode(
        episode_id=str(uuid.uuid4()),
        modality=Modality.GRID,
        template_id=NAME,
        raw_input={"input": input_grid, "output": output_grid},
        gold=gold,
        validator="exact_grid",
        meta={
            "recipe": NAME,
            "cell": (row, col),
            "color_before": color_before,
            "color_after": color_after,
            "ambiguity_level": knob.level,
        },
        task_intent=TaskIntent.SOLVE_GRID,
    )
    require_valid(episode)
    return episode
