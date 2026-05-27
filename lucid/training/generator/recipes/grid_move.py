"""Small grid: one colored cell moves."""

from __future__ import annotations

import uuid
from random import Random

from lucid.ir.common import Modality, TaskIntent
from lucid.ir.training import Episode, GoldLabels, TraceTarget
from lucid.training.generator.engine import AmbiguityKnob, require_valid

NAME = "grid_move"
MODALITY = "grid"

SIZE = 6
COLORS = [1, 2, 3, 4, 5]
DIRECTIONS = ["left", "right", "up", "down"]


def _blank(rows: int, cols: int) -> list[list[int]]:
    return [[0] * cols for _ in range(rows)]


def _move(pos: tuple[int, int], direction: str, steps: int, rows: int, cols: int) -> tuple[int, int]:
    row, col = pos
    if direction == "left":
        col = max(0, col - steps)
    elif direction == "right":
        col = min(cols - 1, col + steps)
    elif direction == "up":
        row = max(0, row - steps)
    else:
        row = min(rows - 1, row + steps)
    return row, col


def make(rng: Random, knob: AmbiguityKnob) -> Episode:
    color = rng.choice(COLORS)
    direction = rng.choice(DIRECTIONS)
    steps = rng.randint(1, 3)
    before = (rng.randint(0, SIZE - 1), rng.randint(0, SIZE - 1))
    after = _move(before, direction, steps, SIZE, SIZE)
    if before == after:
        after = _move(before, direction, min(steps + 1, 3), SIZE, SIZE)

    input_grid = _blank(SIZE, SIZE)
    output_grid = _blank(SIZE, SIZE)
    input_grid[before[0]][before[1]] = color
    output_grid[after[0]][after[1]] = color

    gold = GoldLabels(
        trace_activations=[
            TraceTarget("position_shift_like", 0.88),
            TraceTarget("shape_preserved_like", 0.93),
            TraceTarget("color_preserved_like", 0.91),
        ],
        lucidity_target="COMMIT",
        lucidity_rationale="cell moved, color unchanged",
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
            "direction": direction,
            "steps": steps,
            "before": before,
            "after": after,
            "ambiguity_level": knob.level,
        },
        task_intent=TaskIntent.SOLVE_GRID,
    )
    require_valid(episode)
    return episode
