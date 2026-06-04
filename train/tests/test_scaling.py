from __future__ import annotations

from pathlib import Path

import pytest

from lucid.audit.scaling import (
    ScalingConfig,
    ScalingRecorder,
    build_scale_id,
    config_hash_from_mapping,
    export_summary_csv,
    format_summary,
    iter_points,
    point_from_pipeline_run,
    point_from_trainer_step,
    summarize_points,
)
from lucid.ir.common import DecoderMode, LucidityDecision, Modality, TaskIntent
from lucid.ir.lucidity import DecoderPolicy, LucidityOutput
from lucid.ir.pipeline import PipelineRun, RunContext
from lucid.ir.training import Episode, GoldLabels


@pytest.fixture
def scaling_dir(tmp_path: Path) -> Path:
    base = tmp_path / "audit" / "scaling"
    base.mkdir(parents=True)
    return base


def test_build_scale_id_stable() -> None:
    args = dict(
        module_under_test="cue_encoder",
        training_mode="calibrate",
        corpus_slice="bank_destination",
        config_hash="abc123",
    )
    assert build_scale_id(**args) == build_scale_id(**args)


def test_config_hash_changes() -> None:
    assert config_hash_from_mapping({"k": 1}) != config_hash_from_mapping({"k": 2})


def test_record_and_summarize(scaling_dir: Path) -> None:
    cfg = ScalingConfig(enabled=True, data_dir=scaling_dir)
    ScalingRecorder(cfg).record(
        point_from_trainer_step(
            module_under_test="cue_encoder",
            corpus_slice="bank_destination",
            module_gold_score=0.9,
            wall_time_ms=12.5,
            config=cfg,
        )
    )
    rows = list(iter_points(cfg.points_path))
    assert len(rows) == 1
    text = format_summary(summarize_points(rows, scale_id=rows[0]["scale_id"]))
    assert "module_gold_mean: 0.900" in text


def test_export_csv(scaling_dir: Path) -> None:
    cfg = ScalingConfig(enabled=True, data_dir=scaling_dir)
    rec = ScalingRecorder(cfg)
    for score in (0.5, 0.7, 0.9):
        rec.record(
            point_from_trainer_step(
                module_under_test="perception",
                corpus_slice="phase1",
                module_gold_score=score,
                config=cfg,
            )
        )
    out = cfg.exports_dir / "test.csv"
    export_summary_csv(list(iter_points(cfg.points_path)), out)
    assert "module_gold_mean" in out.read_text(encoding="utf-8")


def test_pipeline_gold_match() -> None:
    episode = Episode(
        episode_id="ep-1",
        modality=Modality.TEXT,
        template_id="bank_destination",
        raw_input="test",
        gold=GoldLabels(lucidity_target="PRESERVE_AMBIGUITY"),
        validator="exact_sense",
        task_intent=TaskIntent.ANSWER,
    )
    run = PipelineRun(context=RunContext(run_id="run-1", episode=episode))
    run.lucidity_output = LucidityOutput(
        decision=LucidityDecision.PRESERVE_AMBIGUITY,
        decoder_policy=DecoderPolicy(mode=DecoderMode.EXPRESS_UNCERTAINTY),
    )
    point = point_from_pipeline_run(run, config=ScalingConfig(enabled=False))
    assert point.module_gold_score == 1.0
    assert point.validator_success is True


def test_disabled_writes_nothing(scaling_dir: Path) -> None:
    cfg = ScalingConfig(enabled=False, data_dir=scaling_dir)
    ScalingRecorder(cfg).record(
        point_from_trainer_step(
            module_under_test="cue_encoder",
            corpus_slice="x",
            module_gold_score=1.0,
            config=cfg,
        )
    )
    assert not cfg.points_path.exists()
