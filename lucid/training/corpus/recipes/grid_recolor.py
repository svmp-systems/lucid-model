"""Small grid: one cell changes color, position stays."""

from __future__ import annotations

import uuid
from random import Random

from lucid.ir.common import Modality, TaskIntent
from lucid.ir.training import Episode, GoldLabels
from lucid.training.corpus.engine import AmbiguityKnob, require_valid
from lucid.training.corpus.recipes import grid_common

NAME = "grid_recolor"
MODALITY = "grid"

SIZE = 6
COLORS = [1, 2, 3, 4, 5]
FRAME_ID = "frame_recolor"


def _blank(rows: int, cols: int) -> list[list[int]]:
    return [[0] * cols for _ in range(rows)]


def make(rng: Random, knob: AmbiguityKnob) -> Episode:
    color_before = rng.choice(COLORS)
    color_after = rng.choice([c for c in COLORS if c != color_before])
    row, col = rng.randint(0, SIZE - 1), rng.randint(0, SIZE - 1)
    cell = (row, col)

    input_grid = _blank(SIZE, SIZE)
    output_grid = _blank(SIZE, SIZE)
    input_grid[row][col] = color_before
    output_grid[row][col] = color_after

    primary_weight, decoy_weight, lucidity = grid_common.trace_weights_for_knob(
        knob,
        primary="recolor_like",
        decoy="position_shift_like",
    )
    confidence = round(0.78 + knob.level * 0.12, 3)
    cell_in, cell_out = grid_common.grid_cell_spans(cell, cell)

    gold = GoldLabels(
        spans=[cell_in, cell_out],
        frame_targets=[
            grid_common.recolor_frame_target(
                FRAME_ID,
                cell=cell,
                confidence=confidence,
            )
        ],
        scope_assignments=grid_common.grid_scope_assignments(
            FRAME_ID,
            same_cell=True,
        ),
        interference_gates=grid_common.grid_interference_gates(
            FRAME_ID,
            primary_trace="recolor_like",
            blocked_traces=("position_shift_like",),
        ),
        trace_activations=grid_common.recolor_trace_activations(
            primary_weight,
            decoy_weight,
            ambiguous=lucidity == "PRESERVE_AMBIGUITY",
        ),
        basin_families=grid_common.grid_basin("grid_recolor", FRAME_ID, primary_weight),
        lucidity_target=lucidity,
        lucidity_rationale=(
            "recolor at fixed cell is unambiguous"
            if lucidity == "COMMIT"
            else "recolor vs move traces still competing"
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
            "cell": cell,
            "color_before": color_before,
            "color_after": color_after,
            "ambiguity_level": knob.level,
        },
        task_intent=TaskIntent.SOLVE_GRID,
    )
    require_valid(episode)
    return episode
