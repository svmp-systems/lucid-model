from __future__ import annotations

import json
from pathlib import Path

from lucid.cli import main as lucid_main
from lucid.training.corpus.engine import AmbiguityKnob, rng_for_seed
from lucid.training.corpus.recipes import bank_destination
from lucid.training.source_policy import (
    GENERATOR_BLOCK_REASON,
    training_source_policy,
)
from lucid.training.validate.gold import to_training_episode


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_generated_episode_gold_is_validation_only_by_default() -> None:
    episode = bank_destination.make(rng_for_seed(7), AmbiguityKnob(0.9))

    policy = training_source_policy(episode)
    training = to_training_episode(episode)

    assert policy.source_kind == "generator"
    assert policy.source_role == "validation_canary"
    assert policy.promotion_eligible is False
    assert training.metadata["training_policy"]["block_reason"] == GENERATOR_BLOCK_REASON


def test_direct_training_defers_generated_gold_without_legacy_flag(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    audit_dir = tmp_path / "audit"

    exit_code = lucid_main(
        [
            "train",
            "dmf",
            "--fixture",
            "bank",
            "--checkpoint",
            str(checkpoint),
            "--audit-dir",
            str(audit_dir),
            "--steps",
            "1",
        ]
    )

    assert exit_code == 0
    assert not (checkpoint / "manifest.json").exists()
    step_update = next(audit_dir.glob("dmf_*/step_*/module_update.json"))
    payload = _read_json(step_update)
    assert payload["action"] == "DEFER"
    assert payload["reason"] == GENERATOR_BLOCK_REASON
    assert payload["metrics"]["training_policy"]["source_role"] == "validation_canary"


def test_global_training_defers_generated_gold_without_legacy_flag(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    audit_dir = tmp_path / "audit"

    exit_code = lucid_main(
        [
            "train",
            "global",
            "--fixture",
            "phase1-mini",
            "--checkpoint",
            str(checkpoint),
            "--audit-dir",
            str(audit_dir),
            "--steps",
            "2",
        ]
    )

    assert exit_code == 0
    assert not (checkpoint / "manifest.json").exists()
    run_dir = next(audit_dir.glob("global_*"))
    metrics = _read_json(run_dir / "metrics.json")
    assert metrics["defer_count"] == 2
    decision = _read_json(next(run_dir.glob("step_*/governor_decision.json")))
    assert decision["decision"]["reason"] == GENERATOR_BLOCK_REASON
