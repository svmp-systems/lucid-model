"""Pytest entrypoint: ``py -m pytest lucid/training/tests`` from repo root."""

from __future__ import annotations

import shutil

import pytest

from lucid.runtime.paths import assert_no_stray_repo_artifact_trees, repo_root

_STRAY_REPO_ARTIFACT_DIRS = ("train", "tests", "audit", "checkpoints")


@pytest.fixture(scope="session", autouse=True)
def _purge_stray_repo_root_artifact_trees() -> None:
    """Remove legacy repo-root artifact trees so path guards stay honest."""
    for name in _STRAY_REPO_ARTIFACT_DIRS:
        path = repo_root() / name
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    yield
    for name in _STRAY_REPO_ARTIFACT_DIRS:
        path = repo_root() / name
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    assert_no_stray_repo_artifact_trees()
