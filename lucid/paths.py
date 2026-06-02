"""Repo layout helpers — training artifacts live under ``train/`` (gitignored)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


@lru_cache(maxsize=1)
def train_root() -> Path:
    override = os.environ.get("LUCID_TRAIN_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "train"


def smoke_audit_dir(module: str) -> str:
    return f"train/audit/runs/smoke/{module}"


DEFAULT_CHECKPOINT = "train/checkpoints/local"
DEFAULT_AUDIT_RUNS = "train/audit/runs/pipeline"
DEFAULT_AUDIT_SCALING = "train/audit/scaling"
