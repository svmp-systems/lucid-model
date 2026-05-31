"""Universal projector runtime.

The projector is consequence testing, not truth selection. It evaluates candidate
programs against train pairs, produces implied artifacts for test inputs, and
returns a recommendation for lucidity to judge.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import uuid4

from lucid.ir.projector import (
    ProjectionConstraints,
    ProjectionGridPair,
    ProjectionOp,
    ProjectionProgram,
    ProjectorInput,
    ProjectorOutput,
    ProjectorRollout,
    RolloutFitScores,
)
from lucid.ir.serde import to_dict

Grid = list[list[int]]

_COMMIT_FIT_THRESHOLD = 0.999
_PRESERVE_FIT_THRESHOLD = 0.5


def run_projector(inp: ProjectorInput, context: object | None = None) -> ProjectorOutput:
    return UniversalProjector().run(inp, context=context)


class UniversalProjector:
    """Apply auditable universal programs and score their consequences."""

    def run(self, inp: ProjectorInput, *, context: object | None = None) -> ProjectorOutput:
        _ = context
        constraints = inp.constraints
        target_ids = _projection_targets(inp)
        requested_programs = _programs_from_request(inp)
        audit_notes = [
            "projector tests implications only; lucidity decides whether to commit",
            f"target_count: {len(target_ids)}",
            f"train_pairs: {len(constraints.train_pairs)}",
            f"test_inputs: {len(constraints.test_inputs)}",
            f"candidate_frames: {len(inp.candidate_frames)}",
            f"context_frames: {len(inp.context_frames)}",
        ]

        if not constraints.train_pairs and not requested_programs:
            return ProjectorOutput(
                recommendation="preserve_ambiguity",
                recommendation_to_lucidity="preserve_ambiguity",
                audit_notes=[
                    *audit_notes,
                    "no train pairs supplied; projector cannot score consequence fit",
                ],
            )

        if requested_programs and not constraints.train_pairs:
            rollouts = [
                _generic_rollout(program, inp)
                for program in requested_programs[: max(1, constraints.max_rollouts)]
            ]
            return ProjectorOutput(
                rollouts=rollouts,
                best_rollout_id=rollouts[0].rollout_id if rollouts else "",
                recommendation="preserve_ambiguity",
                recommendation_to_lucidity="preserve_ambiguity",
                audit_notes=[
                    *audit_notes,
                    f"request_programs: {len(requested_programs)}",
                    "no train pairs supplied; generated implications remain unvalidated",
                ],
            )

        candidates = requested_programs
        if not candidates:
            candidates = _candidate_programs(constraints, target_ids)
        if not candidates:
            return ProjectorOutput(
                recommendation="search_wider",
                recommendation_to_lucidity="search_wider",
                audit_notes=[
                    *audit_notes,
                    "no universal program fit could be inferred from train pairs",
                ],
            )

        rollouts: list[ProjectorRollout] = []
        for program in candidates[: max(1, constraints.max_rollouts)]:
            rollouts.append(_score_program(program, constraints))

        best = max(rollouts, key=lambda rollout: rollout.fit_scores.aggregate_fit)
        recommendation = _recommend(best.fit_scores.aggregate_fit)
        return ProjectorOutput(
            rollouts=rollouts,
            best_rollout_id=best.rollout_id,
            recommendation=recommendation,
            recommendation_to_lucidity=recommendation,
            audit_notes=[
                *audit_notes,
                f"candidate_programs: {len(candidates)}",
                f"rollouts_scored: {len(rollouts)}",
                f"best_fit: {best.fit_scores.aggregate_fit:.3f}",
                f"recommendation_to_lucidity: {recommendation}",
            ],
        )


def _projection_targets(inp: ProjectorInput) -> list[str]:
    targets = list(inp.target_assembly_ids) + list(inp.target_basin_ids)
    targets.extend(inp.projection_request.projector_targets)
    return list(dict.fromkeys(targets))


def _candidate_programs(
    constraints: ProjectionConstraints,
    target_ids: list[str],
) -> list[ProjectionProgram]:
    first = constraints.train_pairs[0]
    candidates = _programs_for_pair(first, target_ids, constraints.output_shape_rules)
    ranked: list[tuple[float, ProjectionProgram]] = []
    for program in candidates:
        scores = [_fit_score(_apply_program(pair.input_grid, program), pair.output_grid)[0]
                  for pair in constraints.train_pairs]
        ranked.append((sum(scores) / len(scores), program))
    ranked.sort(key=lambda item: (-item[0], item[1].description))
    return [program for _score, program in ranked]


def _generic_rollout(program: ProjectionProgram, inp: ProjectorInput) -> ProjectorRollout:
    return ProjectorRollout(
        rollout_id=f"rollout-{uuid4()}",
        assembly_id=program.assembly_id,
        target_basin_ids=list(program.target_basin_ids),
        implied_artifact={
            "artifact_type": "generic",
            "program": to_dict(program),
            "candidate_frames": to_dict(inp.candidate_frames),
            "context_frames": to_dict(inp.context_frames),
            "projection_request": to_dict(inp.projection_request),
        },
        fit_scores=RolloutFitScores(
            aggregate_fit=0.0,
            unexplained_cells=0,
            consistency_score=0.0,
        ),
        program=program,
        program_ref=program.program_id,
    )


def _programs_from_request(inp: ProjectorInput) -> list[ProjectionProgram]:
    raw_programs = inp.projection_request.extra.get("programs") or []
    if not isinstance(raw_programs, list):
        return []
    programs: list[ProjectionProgram] = []
    for index, raw in enumerate(raw_programs):
        if not isinstance(raw, dict):
            continue
        ops = _ops_from_payload(raw.get("ops") or [])
        if not ops:
            continue
        target_basin_ids = raw.get("target_basin_ids") or inp.target_basin_ids
        if not isinstance(target_basin_ids, list):
            target_basin_ids = []
        programs.append(
            ProjectionProgram(
                program_id=str(raw.get("program_id") or f"program_request_{index}"),
                ops=ops,
                target_basin_ids=[str(target) for target in target_basin_ids],
                assembly_id=str(raw.get("assembly_id") or ""),
                description=str(raw.get("description") or "request_program"),
            )
        )
    return programs


def _ops_from_payload(items: list[Any]) -> list[ProjectionOp]:
    ops: list[ProjectionOp] = []
    for item in items:
        if isinstance(item, ProjectionOp):
            ops.append(item)
            continue
        if not isinstance(item, dict):
            continue
        op_type = item.get("op_type") or item.get("type")
        if not op_type:
            continue
        params = item.get("params") or {}
        source_refs = item.get("source_refs") or []
        ops.append(
            ProjectionOp(
                op_type=str(op_type),
                params=params if isinstance(params, dict) else {},
                source_refs=[str(ref) for ref in source_refs]
                if isinstance(source_refs, list)
                else [],
            )
        )
    return ops


def _programs_for_pair(
    pair: ProjectionGridPair,
    target_ids: list[str],
    output_shape_rules: dict[str, Any],
) -> list[ProjectionProgram]:
    candidates: list[ProjectionProgram] = []

    shape_ops = _shape_ops(pair.input_grid, pair.output_grid, output_shape_rules)
    shaped_input = _apply_ops(pair.input_grid, shape_ops) if shape_ops else pair.input_grid

    if shaped_input == pair.output_grid:
        candidates.append(_program("Copy", shape_ops + [ProjectionOp("Copy")], target_ids))

    move = _infer_move(shaped_input, pair.output_grid)
    if move is not None:
        candidates.append(
            _program(
                "Move",
                shape_ops + [ProjectionOp("Move", {"dx": move[0], "dy": move[1]})],
                target_ids,
            )
        )

    color_map = _infer_color_map(shaped_input, pair.output_grid)
    if color_map:
        if len(color_map) == 1:
            c_from, c_to = next(iter(color_map.items()))
            candidates.append(
                _program(
                    "Recolor",
                    shape_ops + [ProjectionOp("Recolor", {"c_from": c_from, "c_to": c_to})],
                    target_ids,
                )
            )
        candidates.append(
            _program(
                "MapSymbol",
                shape_ops + [ProjectionOp("MapSymbol", {"map": color_map})],
                target_ids,
            )
        )

    fill = _infer_fill(shaped_input, pair.output_grid)
    if fill is not None:
        candidates.append(
            _program(
                "Fill",
                shape_ops
                + [ProjectionOp("Fill", {"cells": fill["cells"], "color": fill["color"]})],
                target_ids,
            )
        )

    return _dedupe_programs(candidates)


def _program(
    description: str,
    ops: list[ProjectionOp],
    target_ids: list[str],
) -> ProjectionProgram:
    assembly_id = next((target for target in target_ids if target.startswith("asy")), "")
    basin_ids = [target for target in target_ids if target != assembly_id]
    return ProjectionProgram(
        program_id=f"prog-{uuid4()}",
        ops=ops,
        target_basin_ids=basin_ids,
        assembly_id=assembly_id,
        description=description,
    )


def _dedupe_programs(programs: list[ProjectionProgram]) -> list[ProjectionProgram]:
    seen: set[str] = set()
    out: list[ProjectionProgram] = []
    for program in programs:
        key = repr([to_dict(op) for op in program.ops])
        if key in seen:
            continue
        seen.add(key)
        out.append(program)
    return out


def _shape_ops(
    input_grid: Grid,
    output_grid: Grid,
    output_shape_rules: dict[str, Any],
) -> list[ProjectionOp]:
    rows = output_shape_rules.get("rows")
    cols = output_shape_rules.get("cols")
    if rows is None:
        rows = len(output_grid)
    if cols is None:
        cols = len(output_grid[0]) if output_grid else 0
    if (len(input_grid), len(input_grid[0]) if input_grid else 0) == (rows, cols):
        return []
    return [ProjectionOp("PredictOutputShape", {"rows": int(rows), "cols": int(cols)})]


def _score_program(
    program: ProjectionProgram,
    constraints: ProjectionConstraints,
) -> ProjectorRollout:
    per_pair: dict[str, float] = {}
    total_unexplained = 0
    failure_point = ""
    train_outputs: dict[str, Grid] = {}

    for pair in constraints.train_pairs:
        predicted = _apply_program(pair.input_grid, program)
        fit, unexplained = _fit_score(predicted, pair.output_grid)
        per_pair[pair.pair_id] = fit
        total_unexplained += unexplained
        train_outputs[pair.pair_id] = predicted
        if not failure_point and fit < _COMMIT_FIT_THRESHOLD:
            failure_point = pair.pair_id

    aggregate = sum(per_pair.values()) / len(per_pair)
    test_outputs = [
        _apply_program(test_input, program) for test_input in constraints.test_inputs
    ]
    return ProjectorRollout(
        rollout_id=f"rollout-{uuid4()}",
        assembly_id=program.assembly_id,
        target_basin_ids=list(program.target_basin_ids),
        implied_artifact={
            "artifact_type": "grid",
            "train_outputs": train_outputs,
            "test_outputs": test_outputs,
            "program": to_dict(program),
            "program_ops": [op.op_type for op in program.ops],
        },
        fit_scores=RolloutFitScores(
            per_train_pair=per_pair,
            aggregate_fit=aggregate,
            unexplained_cells=total_unexplained,
            consistency_score=aggregate,
        ),
        program=program,
        program_ref=program.program_id,
        failure_point=failure_point,
    )


def _apply_program(grid: Grid, program: ProjectionProgram) -> Grid:
    return _apply_ops(grid, program.ops)


def _apply_ops(grid: Grid, ops: list[ProjectionOp]) -> Grid:
    out = _copy_grid(grid)
    for op in ops:
        out = _apply_op(out, op)
    return out


def _apply_op(grid: Grid, op: ProjectionOp) -> Grid:
    if op.op_type == "Copy":
        return _copy_grid(grid)
    if op.op_type == "PredictOutputShape":
        return _resize_grid(grid, int(op.params["rows"]), int(op.params["cols"]))
    if op.op_type == "Move":
        return _move_grid(grid, int(op.params["dx"]), int(op.params["dy"]))
    if op.op_type == "Recolor":
        return _map_colors(grid, {int(op.params["c_from"]): int(op.params["c_to"])})
    if op.op_type == "MapSymbol":
        return _map_colors(grid, {int(k): int(v) for k, v in op.params["map"].items()})
    if op.op_type == "Fill":
        out = _copy_grid(grid)
        color = int(op.params["color"])
        for row, col in op.params["cells"]:
            if 0 <= row < len(out) and 0 <= col < len(out[row]):
                out[row][col] = color
        return out
    raise ValueError(f"unknown projection op: {op.op_type}")


def _infer_move(input_grid: Grid, output_grid: Grid) -> tuple[int, int] | None:
    before = _nonzero_cells(input_grid)
    after = _nonzero_cells(output_grid)
    if not before or len(before) != len(after):
        return None
    before_values = sorted(value for _row, _col, value in before)
    after_values = sorted(value for _row, _col, value in after)
    if before_values != after_values:
        return None
    dx = after[0][1] - before[0][1]
    dy = after[0][0] - before[0][0]
    moved = sorted((row + dy, col + dx, value) for row, col, value in before)
    if moved == sorted(after):
        return dx, dy
    return None


def _infer_color_map(input_grid: Grid, output_grid: Grid) -> dict[int, int]:
    if _shape(input_grid) != _shape(output_grid):
        return {}
    mapping: dict[int, int] = {}
    changed = False
    for row_index, row in enumerate(input_grid):
        for col_index, value in enumerate(row):
            out_value = output_grid[row_index][col_index]
            if value == 0 and out_value == 0:
                continue
            if value == 0 or out_value == 0:
                return {}
            existing = mapping.get(value)
            if existing is not None and existing != out_value:
                return {}
            mapping[value] = out_value
            changed = changed or value != out_value
    return mapping if changed else {}


def _infer_fill(input_grid: Grid, output_grid: Grid) -> dict[str, Any] | None:
    if _shape(input_grid) != _shape(output_grid):
        return None
    cells: list[tuple[int, int]] = []
    color: int | None = None
    for row_index, row in enumerate(input_grid):
        for col_index, value in enumerate(row):
            out_value = output_grid[row_index][col_index]
            if value == out_value:
                continue
            if value != 0:
                return None
            if color is None:
                color = out_value
            if out_value != color:
                return None
            cells.append((row_index, col_index))
    if color is None or not cells:
        return None
    return {"cells": cells, "color": color}


def _fit_score(predicted: Grid, expected: Grid) -> tuple[float, int]:
    rows = max(len(predicted), len(expected))
    cols = max(
        max((len(row) for row in predicted), default=0),
        max((len(row) for row in expected), default=0),
    )
    if rows == 0 or cols == 0:
        return (1.0, 0) if predicted == expected else (0.0, 0)
    total = rows * cols
    mismatches = 0
    for row in range(rows):
        for col in range(cols):
            if _cell(predicted, row, col) != _cell(expected, row, col):
                mismatches += 1
    return (total - mismatches) / total, mismatches


def _recommend(fit: float) -> str:
    if fit >= _COMMIT_FIT_THRESHOLD:
        return "suggest_commit"
    if fit >= _PRESERVE_FIT_THRESHOLD:
        return "preserve_ambiguity"
    return "search_wider"


def _copy_grid(grid: Grid) -> Grid:
    return [list(row) for row in grid]


def _resize_grid(grid: Grid, rows: int, cols: int) -> Grid:
    out = [[0 for _ in range(cols)] for _ in range(rows)]
    for row_index, row in enumerate(grid[:rows]):
        for col_index, value in enumerate(row[:cols]):
            out[row_index][col_index] = value
    return out


def _move_grid(grid: Grid, dx: int, dy: int) -> Grid:
    out = [[0 for _ in row] for row in grid]
    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            if value == 0:
                continue
            new_row = row_index + dy
            new_col = col_index + dx
            if 0 <= new_row < len(out) and 0 <= new_col < len(out[new_row]):
                out[new_row][new_col] = value
    return out


def _map_colors(grid: Grid, mapping: dict[int, int]) -> Grid:
    return [[mapping.get(value, value) for value in row] for row in grid]


def _nonzero_cells(grid: Grid) -> list[tuple[int, int, int]]:
    return [
        (row, col, value)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value
    ]


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def _cell(grid: Grid, row: int, col: int) -> int:
    if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[row]):
        return 0
    return grid[row][col]


def with_constraints(inp: ProjectorInput, constraints: ProjectionConstraints) -> ProjectorInput:
    """Return a copy of ``inp`` with resolved constraints attached."""
    return replace(inp, constraints=constraints)
