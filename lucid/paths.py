"""Repo layout: library code in ``lucid/``, training artifacts in ``train/`` (gitignored)."""

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


def audit_dir(*parts: str) -> Path:
    return train_root().joinpath("audit", *parts)


def checkpoints_dir(name: str = "local") -> Path:
    return train_root() / "checkpoints" / name


def resolve_train_path(path: str | Path, *, mkdir: bool = False) -> Path:
    """Resolve CLI paths: cwd-relative, then under ``train/``."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        resolved = candidate
    elif candidate.exists():
        resolved = candidate.resolve()
    else:
        under_train = train_root() / candidate
        resolved = under_train if under_train.exists() or not candidate.exists() else candidate.resolve()

    if mkdir:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


DEFAULT_CHECKPOINT = "train/checkpoints/local"
DEFAULT_AUDIT_RUNS = "train/audit/runs/pipeline"
DEFAULT_AUDIT_TRAINING_RUNS = "train/audit/runs/training"
DEFAULT_AUDIT_SCALING = "train/audit/scaling"
DEFAULT_AUDIT_SMOKE = "train/audit/runs/smoke"

DEFAULT_AUDIT_CUE_ENCODER = smoke_audit_dir("cue_encoder")
DEFAULT_AUDIT_BINDING = smoke_audit_dir("binding")
DEFAULT_AUDIT_DMF = smoke_audit_dir("dmf")
DEFAULT_AUDIT_PERCEPTION = smoke_audit_dir("perception")
DEFAULT_AUDIT_BASINS = smoke_audit_dir("basins")
DEFAULT_AUDIT_LUCIDITY = smoke_audit_dir("lucidity")
