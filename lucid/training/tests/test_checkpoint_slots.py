"""Training vs loaded checkpoint slots."""

from __future__ import annotations

import json

import pytest

from lucid.runtime.paths import (
    DEFAULT_LOADED_CHECKPOINT,
    DEFAULT_TRAINING_CHECKPOINT,
    resolve_checkpoint,
    train_root,
)
from lucid.training.checkpoint.slots import (
    clear_loaded_checkpoint,
    loaded_checkpoint_ready,
    promote_to_loaded,
    resolve_inference_checkpoint,
    resolve_training_checkpoint,
    save_training_snapshot,
)
from lucid.training.checkpoint.store import empty_checkpoint, save_checkpoint


@pytest.fixture(autouse=True)
def _isolated_train_root(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("LUCID_TRAIN_ROOT", str(tmp_path))
    train_root.cache_clear()
    clear_loaded_checkpoint()
    yield
    train_root.cache_clear()


def _seed_checkpoint(path, *, steps: int, marker: str) -> None:
    state = empty_checkpoint(path.name or "slot")
    state.ensure_store("tracebank")["records"] = [{"trace_id": marker, "alias": marker}]
    state.manifest["training_steps"] = steps
    save_checkpoint(state, path)


def test_training_and_loaded_are_separate_trees(tmp_path) -> None:
    training = resolve_checkpoint(DEFAULT_TRAINING_CHECKPOINT)
    _seed_checkpoint(training, steps=1, marker="t-v1")
    promote_to_loaded(training, label="v1")

    _seed_checkpoint(training, steps=99, marker="t-v2")

    loaded = resolve_checkpoint(DEFAULT_LOADED_CHECKPOINT)
    loaded_data = json.loads((loaded / "tracebank.json").read_text(encoding="utf-8"))
    training_data = json.loads((training / "tracebank.json").read_text(encoding="utf-8"))

    assert loaded_data["records"][0]["trace_id"] == "t-v1"
    assert training_data["records"][0]["trace_id"] == "t-v2"


def test_inference_uses_loaded_when_set() -> None:
    training = resolve_checkpoint(DEFAULT_TRAINING_CHECKPOINT)
    _seed_checkpoint(training, steps=1, marker="loaded-only")
    assert resolve_inference_checkpoint("") is None

    promote_to_loaded(training)
    assert loaded_checkpoint_ready()
    assert resolve_inference_checkpoint("") == DEFAULT_LOADED_CHECKPOINT
    assert resolve_inference_checkpoint("", cold=True) is None
    assert resolve_inference_checkpoint("checkpoints/training") == "checkpoints/training"


def test_save_named_snapshot_without_loading() -> None:
    training = resolve_checkpoint(DEFAULT_TRAINING_CHECKPOINT)
    _seed_checkpoint(training, steps=5, marker="save-me")
    save_training_snapshot("bank-v1")

    save_path = resolve_checkpoint("checkpoints/saves/bank-v1")
    data = json.loads((save_path / "tracebank.json").read_text(encoding="utf-8"))
    assert data["records"][0]["trace_id"] == "save-me"
    assert not loaded_checkpoint_ready()

    promote_to_loaded("bank-v1")
    assert loaded_checkpoint_ready()


def test_clear_loaded() -> None:
    training = resolve_checkpoint(DEFAULT_TRAINING_CHECKPOINT)
    _seed_checkpoint(training, steps=1, marker="x")
    promote_to_loaded(training)
    clear_loaded_checkpoint()
    assert not loaded_checkpoint_ready()
    assert resolve_inference_checkpoint("") is None


def test_resolve_training_checkpoint_default() -> None:
    assert resolve_training_checkpoint("") == DEFAULT_TRAINING_CHECKPOINT
    assert resolve_training_checkpoint("checkpoints/local") == "checkpoints/local"
