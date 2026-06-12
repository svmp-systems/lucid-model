"""Guard against stray repo-root artifact trees (train/, audit/, tests/, checkpoints/)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lucid.audit.logger import AuditLogger
from lucid.audit.direct_run import smoke_run_dir
from lucid.runtime import paths as lucid_paths
from lucid.runtime.paths import (
    DEFAULT_AUDIT_RUNS,
    DEFAULT_AUDIT_TRAINING_RUNS,
    DEFAULT_CHECKPOINT,
    DEFAULT_INTERFERENCE_LEARNING_AUDIT,
    DEFAULT_INTERFERENCE_STORE,
    assert_no_stray_repo_artifact_trees,
    repo_root,
    resolve_checkpoint,
    resolve_train_path,
    train_root,
)
from lucid.training.checkpoint.store import load_checkpoint


def _stray(name: str) -> Path:
    return repo_root() / name


@pytest.fixture(autouse=True)
def _remove_stray_repo_trees() -> None:
    """Drop mistaken repo-root trees before each test (they break path resolution)."""
    import shutil

    for name in ("train", "tests", "audit", "checkpoints"):
        path = _stray(name)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    yield
    for name in ("train", "tests", "audit", "checkpoints"):
        path = _stray(name)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def test_canonical_train_and_tests_roots_under_lucid_training() -> None:
    root = train_root()
    assert root.name == "tree"
    assert root.parent.name == "training"
    assert root.parent.parent.name == "lucid"
    assert lucid_paths.tests_root() == root.parent / "tests"


def test_legacy_train_prefix_resolves_to_canonical_tree() -> None:
    legacy = resolve_train_path("train/audit/runs/pipeline")
    modern = resolve_train_path(DEFAULT_AUDIT_RUNS)
    assert legacy == modern
    assert legacy.is_relative_to(train_root())


def test_checkpoint_and_audit_paths_stay_under_train_root(tmp_path: Path) -> None:
    cp = resolve_checkpoint(DEFAULT_CHECKPOINT, mkdir=True)
    assert cp.is_relative_to(train_root())
    audit = resolve_train_path(DEFAULT_AUDIT_TRAINING_RUNS, mkdir=True)
    assert audit.is_relative_to(train_root())
    store = resolve_train_path(DEFAULT_INTERFERENCE_STORE, mkdir=True)
    assert store.is_relative_to(train_root())
    learn = resolve_train_path(DEFAULT_INTERFERENCE_LEARNING_AUDIT, mkdir=True)
    assert learn.is_relative_to(train_root())


def test_audit_logger_does_not_create_repo_root_train() -> None:
    logger = AuditLogger(base_dir=DEFAULT_AUDIT_RUNS)
    assert logger.base_dir.is_relative_to(train_root())
    assert not _stray("train").exists()
    assert not _stray("audit").exists()


def test_smoke_audit_writer_does_not_create_repo_root_train() -> None:
    run_dir, _run_id = smoke_run_dir(
        "paths_guard",
        label="guard",
        audit_base_dir="audit/runs/smoke/paths_guard",
    )
    assert run_dir.is_relative_to(train_root())
    assert not _stray("train").exists()
    assert not _stray("audit").exists()


def test_load_checkpoint_uses_canonical_tree(tmp_path: Path) -> None:
    cp = resolve_checkpoint(DEFAULT_CHECKPOINT, mkdir=True)
    state = load_checkpoint(cp, create=True)
    assert state.checkpoint_id
    assert not _stray("train").exists()
    assert not _stray("audit").exists()


def test_repo_root_train_remapped_if_passed_explicitly() -> None:
    stray = _stray("train")
    stray.mkdir(parents=True, exist_ok=True)
    (stray / "audit").mkdir(parents=True, exist_ok=True)
    resolved = resolve_train_path(stray / "audit/runs/pipeline")
    assert resolved == resolve_train_path(DEFAULT_AUDIT_RUNS)
    assert resolved.is_relative_to(train_root())


def test_repo_root_audit_remapped_if_passed_explicitly() -> None:
    stray = _stray("audit")
    stray.mkdir(parents=True, exist_ok=True)
    (stray / "training").mkdir(parents=True, exist_ok=True)
    resolved = resolve_train_path(stray / "training")
    assert resolved == resolve_train_path(DEFAULT_AUDIT_TRAINING_RUNS)
    assert resolved.is_relative_to(train_root())
    assert not (stray / "training" / "new_run").exists()


def test_legacy_repo_root_audit_training_string_remapped() -> None:
    stray = _stray("audit")
    stray.mkdir(parents=True, exist_ok=True)
    (stray / "training" / "binding_checkpoint_deadbeef").mkdir(parents=True)
    resolved = resolve_train_path("audit/training")
    assert resolved == resolve_train_path(DEFAULT_AUDIT_TRAINING_RUNS)
    assert resolved.is_relative_to(train_root())


def test_assert_no_stray_repo_artifact_trees_raises_on_root_audit() -> None:
    stray = _stray("audit")
    stray.mkdir(parents=True, exist_ok=True)
    (stray / "training").mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError, match="stray repo-root audit/"):
        assert_no_stray_repo_artifact_trees()
