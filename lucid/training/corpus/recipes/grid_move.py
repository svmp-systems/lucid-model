"""Small grid: one colored cell moves."""

from __future__ import annotations

import uuid
from random import Random

from lucid.ir.common import Modality, TaskIntent
from lucid.ir.training import Episode, GoldLabels
from lucid.training.corpus.engine import AmbiguityKnob, require_valid
from lucid.training.corpus.recipes import grid_common

NAME = "grid_move"
MODALITY = "grid"

SIZE = 6
COLORS = [1, 2, 3, 4, 5]
DIRECTIONS = ["left", "right", "up", "down"]
FRAME_ID = "frame_position_shift"


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
    after = before
    for candidate_steps in range(1, 4):
        candidate = _move(before, direction, candidate_steps, SIZE, SIZE)
        if candidate != before:
            after = candidate
            steps = candidate_steps
            break
    if after == before:
        for alt_direction in DIRECTIONS:
            candidate = _move(before, alt_direction, 1, SIZE, SIZE)
            if candidate != before:
                after = candidate
                direction = alt_direction
                steps = 1
                break

    input_grid = _blank(SIZE, SIZE)
    output_grid = _blank(SIZE, SIZE)
    input_grid[before[0]][before[1]] = color
    output_grid[after[0]][after[1]] = color

    primary_weight, decoy_weight, lucidity = grid_common.trace_weights_for_knob(
        knob,
        primary="position_shift_like",
        decoy="recolor_like",
    )
    confidence = round(0.78 + knob.level * 0.12, 3)
    cell_in, cell_out = grid_common.grid_cell_spans(before, after)

    gold = GoldLabels(
        spans=[cell_in, cell_out],
        frame_targets=[
            grid_common.move_frame_target(
                FRAME_ID,
                before=before,
                after=after,
                confidence=confidence,
            )
        ],
        scope_assignments=grid_common.grid_scope_assignments(
            FRAME_ID,
            same_cell=False,
        ),
        interference_gates=grid_common.grid_interference_gates(
            FRAME_ID,
            primary_trace="position_shift_like",
            blocked_traces=("recolor_like",),
        ),
        trace_activations=grid_common.move_trace_activations(
            primary_weight,
            decoy_weight,
            ambiguous=lucidity == "PRESERVE_AMBIGUITY",
        ),
        basin_families=grid_common.grid_basin("grid_position_shift", FRAME_ID, primary_weight),
        lucidity_target=lucidity,
        lucidity_rationale=(
            "single-cell move is unambiguous"
            if lucidity == "COMMIT"
            else "move vs recolor traces still competing"
        ),
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
