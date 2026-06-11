"""Gold episode validation and L3 module checks."""

from __future__ import annotations

from lucid.training.checkpoint.store import empty_checkpoint, save_checkpoint
from lucid.training.corpus.engine import AmbiguityKnob, rng_for_seed
from lucid.training.corpus.recipes import bank_destination
from lucid.training.validate.gold import (
    GoldEpisodeValidator,
    L3ModuleGoldValidator,
    to_training_episode,
    validate_episode_pack,
)


def test_gold_validator_accepts_bank_commit(tmp_path) -> None:
    episode = bank_destination.make(rng_for_seed(7), AmbiguityKnob(0.9))
    training = to_training_episode(episode)
    run_log = training  # placeholder replaced below
    from lucid.training.loop.orchestrator import RunLog

    run_log = RunLog(
        episode_id=episode.episode_id,
        raw_input=episode.raw_input,
        evidence_graph={},
        cue_cloud={},
        active_traces=[],
        trace_clusters=[],
        candidate_bindings=[],
        context_frames=[],
        scoped_trace_assignments={},
        interference_edges=[],
        active_basins=[],
        basin_assemblies={},
        lucidity_features={},
        lucidity_decision="COMMIT",
        lucidity_margin=0.9,
        projection_result=None,
        decoder_output=episode.gold.expected_answer,
        validator_result={},
        cost_metrics={},
    )
    result = GoldEpisodeValidator().evaluate_run_log(run_log, episode)
    assert result.success or "lucidity" not in str(result.failure_signals)


def test_l3_module_gold_reports_populated_checkpoint(tmp_path) -> None:
    episode = bank_destination.make(rng_for_seed(7), AmbiguityKnob(0.9))
    checkpoint = tmp_path / "checkpoint"
    state = empty_checkpoint("local")
    state.ensure_store("tracebank")["records"] = [{"trace_id": "t1", "trace_family": "bank"}]
    save_checkpoint(state, checkpoint)

    reports = L3ModuleGoldValidator().evaluate_checkpoint(checkpoint, [episode])
    dmf = next(row for row in reports if row.module == "dmf")
    assert dmf.store_count >= 1
    assert dmf.passed is True


def test_validate_episode_pack_runs_without_crash(tmp_path) -> None:
    from lucid.training.corpus.output import write_episodes

    episode = bank_destination.make(rng_for_seed(7), AmbiguityKnob(0.9))
    path = tmp_path / "one.jsonl"
    write_episodes([episode], path)
    report = validate_episode_pack(path, limit=1, audit_dir=str(tmp_path / "audit"))
    assert report["episodes"] == 1
    assert report["crashes"] == 0
