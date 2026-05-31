from __future__ import annotations

import json
from pathlib import Path

from lucid.audit.inspect import print_run
from lucid.cognition.input.perception import PerceptionConfig
from lucid.cognition.orchestrator.runner import OrchestratorConfig, OrchestratorRunner
from lucid.cognition.orchestrator.stages import FunctionStage
from lucid.cognition.orchestrator.stub_stages import build_default_stage_fns
from lucid.cognition.projector import run_projector
from lucid.ir.common import DecoderMode, LucidityDecision, Modality, TaskIntent
from lucid.ir.lucidity import DecoderPolicy, LucidityInput, LucidityOutput, SearchDirectives
from lucid.ir.projector import ProjectionConstraints, ProjectionGridPair, ProjectorInput
from lucid.ir.training import Episode, GoldLabels


def test_projector_scores_move_rollout_and_recommends_commit() -> None:
    inp = ProjectorInput(
        projection_request=SearchDirectives(projector_targets=["asy_move"], max_rollouts=2),
        constraints=ProjectionConstraints(
            train_pairs=[
                ProjectionGridPair(
                    pair_id="train_0",
                    input_grid=[[0, 1, 0], [0, 0, 0]],
                    output_grid=[[0, 0, 1], [0, 0, 0]],
                )
            ],
            test_inputs=[[[2, 0, 0], [0, 0, 0]]],
            max_rollouts=2,
        ),
        task_intent=TaskIntent.SOLVE_GRID.value,
    )

    out = run_projector(inp)

    assert out.recommendation_to_lucidity == "suggest_commit"
    assert out.best_rollout_id
    best = next(rollout for rollout in out.rollouts if rollout.rollout_id == out.best_rollout_id)
    assert best.fit_scores.aggregate_fit == 1.0
    assert best.program is not None
    assert [op.op_type for op in best.program.ops] == ["Move"]
    assert best.implied_artifact["test_outputs"] == [[[0, 2, 0], [0, 0, 0]]]


def test_projector_scores_recolor_rollout() -> None:
    inp = ProjectorInput(
        projection_request=SearchDirectives(projector_targets=["b_recolor"], max_rollouts=2),
        constraints=ProjectionConstraints(
            train_pairs=[
                ProjectionGridPair(
                    pair_id="train_0",
                    input_grid=[[0, 3], [0, 0]],
                    output_grid=[[0, 5], [0, 0]],
                )
            ],
            test_inputs=[[[3, 0], [0, 3]]],
            max_rollouts=2,
        ),
        task_intent=TaskIntent.SOLVE_GRID.value,
    )

    out = run_projector(inp)
    best = next(rollout for rollout in out.rollouts if rollout.rollout_id == out.best_rollout_id)

    assert out.recommendation == "suggest_commit"
    assert best.implied_artifact["test_outputs"] == [[[5, 0], [0, 5]]]
    assert best.program is not None
    assert best.program.ops[-1].op_type in {"Recolor", "MapSymbol"}


def test_projector_scores_basin_parameterized_program() -> None:
    inp = ProjectorInput(
        projection_request=SearchDirectives(
            projector_targets=["b_copy"],
            extra={
                "programs": [
                    {
                        "program_id": "p_copy_from_basin",
                        "target_basin_ids": ["b_copy"],
                        "ops": [{"op_type": "Copy"}],
                    }
                ]
            },
        ),
        constraints=ProjectionConstraints(
            train_pairs=[
                ProjectionGridPair(
                    pair_id="train_0",
                    input_grid=[[0, 4], [0, 0]],
                    output_grid=[[0, 4], [0, 0]],
                )
            ],
            test_inputs=[[[1, 0], [0, 0]]],
        ),
        task_intent=TaskIntent.SOLVE_GRID.value,
    )

    out = run_projector(inp)
    best = next(rollout for rollout in out.rollouts if rollout.rollout_id == out.best_rollout_id)

    assert out.recommendation_to_lucidity == "suggest_commit"
    assert best.program_ref == "p_copy_from_basin"
    assert best.target_basin_ids == ["b_copy"]
    assert best.implied_artifact["test_outputs"] == [[[1, 0], [0, 0]]]


def test_projector_without_train_pairs_preserves_ambiguity() -> None:
    out = run_projector(
        ProjectorInput(
            projection_request=SearchDirectives(projector_targets=["b_unknown"]),
            task_intent=TaskIntent.SOLVE_GRID.value,
        )
    )

    assert out.rollouts == []
    assert out.recommendation_to_lucidity == "preserve_ambiguity"
    assert any("no train pairs" in note for note in out.audit_notes)


def test_projector_emits_unvalidated_generic_rollout_from_basin_program() -> None:
    inp = ProjectorInput(
        projection_request=SearchDirectives(
            projector_targets=["b_plan"],
            extra={
                "programs": [
                    {
                        "program_id": "p_plan",
                        "target_basin_ids": ["b_plan"],
                        "ops": [
                            {
                                "op_type": "PlanStep",
                                "params": {"action_ref": "t_next_action"},
                            }
                        ],
                    }
                ]
            },
        ),
        task_intent=TaskIntent.ACT.value,
    )

    out = run_projector(inp)

    assert out.recommendation_to_lucidity == "preserve_ambiguity"
    assert out.best_rollout_id
    assert out.rollouts[0].implied_artifact["artifact_type"] == "generic"
    assert out.rollouts[0].program_ref == "p_plan"


def test_default_grid_pipeline_requests_projection_and_decodes_grid(tmp_path: Path) -> None:
    runner = OrchestratorRunner(
        config=OrchestratorConfig(
            audit_base_dir=str(tmp_path),
            perception=PerceptionConfig(backend="rule"),
        )
    )

    run = runner.run_episode(
        Episode(
            episode_id="ep-default-grid-projector",
            modality=Modality.GRID,
            raw_input={
                "input": [[0, 1, 0], [0, 0, 0]],
                "output": [[0, 0, 1], [0, 0, 0]],
            },
            gold=GoldLabels(expected_answer=[[0, 0, 1], [0, 0, 0]]),
            task_intent=TaskIntent.SOLVE_GRID,
        )
    )

    assert run.projector_output is not None
    assert run.projector_output.recommendation_to_lucidity == "suggest_commit"
    assert run.lucidity_output.decision == LucidityDecision.COMMIT
    assert run.decoder_output is not None
    assert run.decoder_output.surface_grid == [[0, 0, 1], [0, 0, 0]]

    run_dir = Path(run.context.audit_dir)
    assert (run_dir / "projector.json").exists()
    assert (run_dir / "lucidity_02.json").exists()


def test_inspect_prints_repeated_lucidity_occurrence(tmp_path: Path, capsys) -> None:
    runner = OrchestratorRunner(
        config=OrchestratorConfig(
            audit_base_dir=str(tmp_path),
            perception=PerceptionConfig(backend="rule"),
        )
    )
    run = runner.run_episode(
        Episode(
            episode_id="ep-inspect-repeated-lucidity",
            modality=Modality.GRID,
            raw_input={
                "input": [[0, 1, 0], [0, 0, 0]],
                "output": [[0, 0, 1], [0, 0, 0]],
            },
            gold=GoldLabels(expected_answer=[[0, 0, 1], [0, 0, 0]]),
            task_intent=TaskIntent.SOLVE_GRID,
        )
    )

    print_run(Path(run.context.audit_dir))
    out = capsys.readouterr().out

    assert "--- lucidity: decision: request_projection ---" in out
    assert "--- lucidity#2: decision: commit ---" in out


def test_orchestrator_projection_stage_writes_auditable_program(tmp_path: Path) -> None:
    def lucidity_with_projection(inp: LucidityInput, _ctx: object) -> LucidityOutput:
        if inp.pass_kind == "pre_check":
            return LucidityOutput(
                decision=LucidityDecision.REQUEST_PROJECTION,
                decoder_policy=DecoderPolicy(mode=DecoderMode.HOLD.value),
                search_directives=SearchDirectives(projector_targets=["asy_move"], max_rollouts=2),
            )
        assert inp.projection_output is not None
        assert inp.projection_output.recommendation_to_lucidity == "suggest_commit"
        return LucidityOutput(
            decision=LucidityDecision.COMMIT,
            decoder_policy=DecoderPolicy(mode=DecoderMode.EXPRESS_COMMITTED.value),
        )

    fns = build_default_stage_fns()
    fns["lucidity"] = lucidity_with_projection
    stages = {name: FunctionStage(stage_name=name, fn=fn) for name, fn in fns.items()}
    runner = OrchestratorRunner(
        config=OrchestratorConfig(
            audit_base_dir=str(tmp_path),
            perception=PerceptionConfig(backend="rule"),
        ),
        stages=stages,
    )

    run = runner.run_episode(
        Episode(
            episode_id="ep-projector-move",
            modality=Modality.GRID,
            raw_input={
                "input": [[0, 1, 0], [0, 0, 0]],
                "output": [[0, 0, 1], [0, 0, 0]],
            },
            gold=GoldLabels(expected_answer=[[0, 0, 1], [0, 0, 0]]),
            task_intent=TaskIntent.SOLVE_GRID,
        )
    )

    run_dir = Path(run.context.audit_dir)
    record = json.loads((run_dir / "projector.json").read_text(encoding="utf-8"))
    rollout = record["output"]["rollouts"][0]

    assert record["summary"]["headline"] == "1 rollouts"
    assert record["output"]["recommendation_to_lucidity"] == "suggest_commit"
    assert rollout["program"]["ops"][0]["op_type"] == "Move"
    assert rollout["implied_artifact"]["test_outputs"] == [[[0, 0, 1], [0, 0, 0]]]
    assert run.cost_metrics.projector_rollout_count == 1
