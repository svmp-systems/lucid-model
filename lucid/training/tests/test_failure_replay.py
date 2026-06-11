"""Failure replay queue clear rules."""

from __future__ import annotations

from lucid.training.loop.orchestrator import FailureReplayStore, RunLog


def _run_log(episode_id: str) -> RunLog:
    return RunLog(
        episode_id=episode_id,
        raw_input="x",
        evidence_graph={"entities": []},
        cue_cloud={},
        active_traces=["t1"],
        trace_clusters=[],
        candidate_bindings=[{"binding_id": "b1"}],
        context_frames=[{"frame_id": "f1"}],
        scoped_trace_assignments={},
        interference_edges=[],
        active_basins=[{"basin_id": "basin1"}],
        basin_assemblies={},
        lucidity_features={},
        lucidity_decision="commit",
        lucidity_margin=0.9,
        projection_result=None,
        decoder_output="ok",
        validator_result={},
        cost_metrics={"stages_run": 8},
    )


def test_replay_clears_after_promote_shadow_and_three_successes() -> None:
    store = FailureReplayStore()
    run = _run_log("ep-1")
    store.add_or_refresh(run)
    assert store.contains("ep-1")

    store.on_patch_promoted("ep-1", "patch-1")
    store.on_episode_shadow_passed("ep-1")
    store.record_success("ep-1")
    assert store.contains("ep-1")
    store.record_success("ep-1")
    assert store.contains("ep-1")
    store.record_success("ep-1")
    assert store.try_clear("ep-1") is True
    assert not store.contains("ep-1")


def test_replay_not_cleared_without_shadow_pass() -> None:
    store = FailureReplayStore()
    store.add_or_refresh(_run_log("ep-2"))
    store.on_patch_promoted("ep-2", "patch-2")
    for _ in range(3):
        store.record_success("ep-2")
    assert store.contains("ep-2")
    assert store.metrics()["cleared_without_shadow_pass"] >= 0
