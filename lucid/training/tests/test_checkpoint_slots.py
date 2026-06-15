"""Training vs loaded checkpoint slots."""

from __future__ import annotations

import json

import pytest

from lucid.cognition.memory.basin_bank import basin_bank_from_checkpoint, load_basin_bank
from lucid.memory.dmf import load_dynamic_memory_field, tracebank_from_checkpoint
from lucid.runtime.paths import (
    DEFAULT_LOADED_CHECKPOINT,
    DEFAULT_TRAINING_CHECKPOINT,
    resolve_checkpoint,
    train_root,
)
from lucid.training.checkpoint.slots import (
    archive_training_checkpoint,
    clear_loaded_checkpoint,
    loaded_checkpoint_ready,
    promote_to_loaded,
    read_loaded_pointer,
    resolve_checkpoint_ref,
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


def test_runtime_memory_loaders_resolve_loaded_alias() -> None:
    training = resolve_checkpoint(DEFAULT_TRAINING_CHECKPOINT)
    state = empty_checkpoint("runtime-alias")
    state.ensure_store("tracebank")["records"] = [
        {
            "trace_id": "trace_qubit",
            "alias": "qubit",
            "cue_affinities": {"qubit": 1.0},
        }
    ]
    state.ensure_store("basin_bank")["records"] = [
        {
            "basin_id": "basin_qubit",
            "family_hint": "quantum",
            "activation_signature": {"qubit": 1.0},
            "trust_score": 1.0,
        }
    ]
    save_checkpoint(state, training)
    promote_to_loaded(training)

    assert [trace.trace_id for trace in tracebank_from_checkpoint("loaded")] == ["trace_qubit"]
    assert [trace.trace_id for trace in load_dynamic_memory_field("loaded").tracebank] == [
        "trace_qubit"
    ]
    assert [record.basin_id for record in basin_bank_from_checkpoint("loaded")] == ["basin_qubit"]
    assert [record.basin_id for record in load_basin_bank("loaded").records] == ["basin_qubit"]


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


def test_auto_archive_assigns_cp_names_in_order() -> None:
    training = resolve_checkpoint(DEFAULT_TRAINING_CHECKPOINT)
    _seed_checkpoint(training, steps=1, marker="a")
    first = archive_training_checkpoint(command="train binding")
    _seed_checkpoint(training, steps=2, marker="b")
    second = archive_training_checkpoint(command="train dmf")

    assert first["name"] == "cp_001"
    assert second["name"] == "cp_002"
    assert (resolve_checkpoint("checkpoints/saves/cp_001") / "manifest.json").is_file()


def test_inference_resolves_cp_shorthand() -> None:
    training = resolve_checkpoint(DEFAULT_TRAINING_CHECKPOINT)
    _seed_checkpoint(training, steps=4, marker="cp-shorthand")
    archive_training_checkpoint(name="cp_003", command="test")

    assert resolve_checkpoint_ref("cp_003") == "checkpoints/saves/cp_003"
    assert resolve_inference_checkpoint("cp_003") == "checkpoints/saves/cp_003"
    assert resolve_inference_checkpoint("", cold=True) is None


def test_pin_after_archive() -> None:
    training = resolve_checkpoint(DEFAULT_TRAINING_CHECKPOINT)
    _seed_checkpoint(training, steps=2, marker="pin-me")
    record = archive_training_checkpoint(name="cp_010", label="bank pass", command="train binding")
    promote_to_loaded(record["name"], label="bank pass")

    assert loaded_checkpoint_ready()
    pointer = read_loaded_pointer()
    assert pointer is not None
    assert pointer.get("save_name") == "cp_010"
    assert resolve_inference_checkpoint("") == DEFAULT_LOADED_CHECKPOINT
