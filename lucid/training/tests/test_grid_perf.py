"""Phase-1 grid latency gate (<2s on CPU for micro fixture)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from lucid.cognition.input.perception import PerceptionConfig
from lucid.cognition.pipe_orchestrator.runner import OrchestratorConfig, OrchestratorRunner
from lucid.training.corpus.engine import AmbiguityKnob, rng_for_seed
from lucid.training.corpus.recipes import grid_move


def test_grid_micro_fixture_under_two_seconds(tmp_path: Path) -> None:
    episode = grid_move.make(rng_for_seed(9), AmbiguityKnob(0.95))
    runner = OrchestratorRunner(
        config=OrchestratorConfig(
            audit_base_dir=str(tmp_path / "audit"),
            perception=PerceptionConfig(backend="rule", write_audit=False),
        )
    )
    start = time.perf_counter()
    runner.run_episode(episode)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"grid pipeline took {elapsed:.2f}s, expected <2s"
