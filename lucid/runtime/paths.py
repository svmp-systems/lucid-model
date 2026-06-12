"""Resolve repo vs train-tree paths for CLI and runtime artifacts.

Library code lives under ``lucid/``. Local checkpoints, episodes, and audits live under
``lucid/training/tree/`` (gitignored). Pytest sources live under ``lucid/training/tests/``.

All runtime writes must go through :func:`resolve_train_path` (or :func:`resolve_checkpoint`).
Never use bare ``Path("train/...")`` or ``Path("audit/...")`` — that can recreate stray repo-root trees.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Relative to :func:`train_root` (no leading ``train/`` — that prefix is legacy CLI only).
_AUDIT_RUNS = ("audit", "runs")
_AUDIT_SCALING = ("audit", "scaling")


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
    return repo_root() / "lucid" / "training" / "tree"


def tests_root() -> Path:
    return repo_root() / "lucid" / "training" / "tests"


def _join_relative(*parts: str) -> str:
    return "/".join(parts)


def smoke_audit_dir(module: str) -> str:
    return _join_relative(*_AUDIT_RUNS, "smoke", module)


def audit_dir(*parts: str) -> Path:
    return train_root().joinpath("audit", *parts)


def checkpoints_dir(name: str = "local") -> Path:
    return train_root() / "checkpoints" / name


def _normalize_train_path_text(path: str | Path) -> str:
    return str(path).replace("\\", "/").lstrip("./")


def _strip_legacy_train_prefix(text: str) -> str:
    for prefix in ("lucid/training/tree/", "lucid/training/tree/"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    if text in {"lucid/training/tree", "lucid/training/tree"}:
        return ""
    if text.startswith("train/"):
        return text[len("train/") :]
    if text == "train":
        return ""
    return text


def _normalize_legacy_audit_layout(text: str) -> str:
    """Map mistaken ``audit/training/...`` trees to ``audit/runs/training/...``."""
    if text == "audit/training":
        return "audit/runs/training"
    if text.startswith("audit/training/"):
        return "audit/runs/training/" + text[len("audit/training/") :]
    return text


def _is_train_tree_relative(text: str) -> bool:
    if not text:
        return True
    first = text.split("/", 1)[0]
    return first in {"audit", "checkpoints", "data"}


def _remap_stray_repo_tree(path: Path, *, stray_name: str) -> Path:
    """Remap ``repo_root/{stray_name}/...`` to ``train_root/{stray_name}/...`` when they differ."""
    stray = repo_root() / stray_name
    canonical = train_root() / stray_name
    if stray.resolve() == canonical.resolve():
        return path
    try:
        rel = path.resolve().relative_to(stray.resolve())
    except ValueError:
        return path
    return canonical / rel


def _remap_stray_repo_train(path: Path) -> Path:
    """If resolution landed under repo-root ``train/``, remap to canonical train_root."""
    stray = repo_root() / "train"
    canonical = train_root()
    if stray.resolve() == canonical.resolve():
        return path
    try:
        rel = path.resolve().relative_to(stray.resolve())
    except ValueError:
        return path
    return canonical / rel


def _remap_stray_repo_artifacts(path: Path) -> Path:
    """Never leave runtime artifacts under repo-root ``train/``, ``audit/``, or ``checkpoints/``."""
    resolved = _remap_stray_repo_train(path)
    for name in ("audit", "checkpoints"):
        resolved = _remap_stray_repo_tree(resolved, stray_name=name)
    text = _normalize_train_path_text(resolved)
    if resolved.is_relative_to(train_root()):
        try:
            rel = str(resolved.relative_to(train_root())).replace("\\", "/")
        except ValueError:
            rel = text
        normalized = _normalize_legacy_audit_layout(rel)
        if normalized != rel:
            resolved = train_root() / normalized
    return resolved


def _repo_root_artifact(name: str) -> Path:
    return repo_root() / name


def resolve_train_path(path: str | Path, *, mkdir: bool = False) -> Path:
    """Resolve a path into ``lucid/training/tree/`` (never repo-root artifact trees)."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        text = _normalize_train_path_text(candidate)
        inner = _normalize_legacy_audit_layout(_strip_legacy_train_prefix(text))
        if inner != text or not text:
            resolved = train_root() / inner if inner else train_root()
        elif _is_train_tree_relative(text):
            resolved = train_root() / _normalize_legacy_audit_layout(text)
        else:
            under_train = train_root() / candidate
            if under_train.exists() or not candidate.parts:
                resolved = under_train
            elif candidate.exists():
                cwd_hit = candidate.resolve()
                if _is_train_tree_relative(text) or any(
                    _repo_root_artifact(name) in cwd_hit.parents or cwd_hit == _repo_root_artifact(name)
                    for name in ("train", "audit", "checkpoints")
                ):
                    resolved = under_train
                else:
                    resolved = cwd_hit
            else:
                resolved = under_train

    resolved = _remap_stray_repo_artifacts(resolved)

    if mkdir:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_checkpoint(path: str | Path, *, mkdir: bool = False) -> Path:
    """Resolve a checkpoint directory (alias for :func:`resolve_train_path`)."""
    return resolve_train_path(path, mkdir=mkdir)


def assert_no_stray_repo_artifact_trees() -> None:
    """Fail fast when repo-root artifact trees exist outside ``lucid/training/tree/``."""
    canonical = train_root()
    stray_train = repo_root() / "train"
    if stray_train.exists() and stray_train.resolve() != canonical.resolve():
        raise RuntimeError(
            f"stray repo-root train/ at {stray_train} — use {canonical} via resolve_train_path()"
        )

    stray_audit = repo_root() / "audit"
    canonical_audit = canonical / "audit"
    if stray_audit.exists() and stray_audit.resolve() != canonical_audit.resolve():
        raise RuntimeError(
            f"stray repo-root audit/ at {stray_audit} — use {canonical_audit} via resolve_train_path()"
        )

    stray_checkpoints = repo_root() / "checkpoints"
    canonical_checkpoints = canonical / "checkpoints"
    if stray_checkpoints.exists() and stray_checkpoints.resolve() != canonical_checkpoints.resolve():
        raise RuntimeError(
            "stray repo-root checkpoints/ at "
            f"{stray_checkpoints} — use {canonical_checkpoints} via resolve_checkpoint()"
        )

    stray_tests = repo_root() / "tests"
    if stray_tests.exists() and not (stray_tests / ".gitkeep").exists():
        try:
            next(stray_tests.iterdir())
        except StopIteration:
            pass
        else:
            raise RuntimeError(
                f"stray repo-root tests/ at {stray_tests} — use {tests_root()} for pytest"
            )


def assert_no_stray_repo_train_tree() -> None:
    """Backward-compatible alias for :func:`assert_no_stray_repo_artifact_trees`."""
    assert_no_stray_repo_artifact_trees()


# CLI-facing defaults (legacy ``train/`` prefix accepted by :func:`resolve_train_path`).
DEFAULT_TRAINING_CHECKPOINT = "checkpoints/training"
DEFAULT_LOADED_CHECKPOINT = "checkpoints/loaded"
LOADED_CHECKPOINT_POINTER = "checkpoints/loaded.json"
# Backward-compatible alias — prefer :data:`DEFAULT_TRAINING_CHECKPOINT` for new code.
DEFAULT_CHECKPOINT = DEFAULT_TRAINING_CHECKPOINT
DEFAULT_AUDIT_RUNS = _join_relative(*_AUDIT_RUNS, "pipeline")
DEFAULT_AUDIT_TRAINING_RUNS = _join_relative(*_AUDIT_RUNS, "training")
DEFAULT_AUDIT_SCALING = _join_relative(*_AUDIT_SCALING)
DEFAULT_AUDIT_SMOKE = _join_relative(*_AUDIT_RUNS, "smoke")
DEFAULT_AUDIT_VALIDATION = _join_relative(*_AUDIT_RUNS, "validation")
DEFAULT_AUDIT_MEMORY = _join_relative(*_AUDIT_RUNS, "memory")
DEFAULT_ASK_LATEST = _join_relative(*_AUDIT_RUNS, "ask", "latest.txt")
DEFAULT_INTERFERENCE_STORE = _join_relative("audit", "interference_learning", "interference_links.json")
DEFAULT_INTERFERENCE_LEARNING_AUDIT = _join_relative("audit", "interference_learning", "runs")

DEFAULT_AUDIT_CUE_ENCODER = smoke_audit_dir("cue_encoder")
DEFAULT_AUDIT_BINDING = smoke_audit_dir("binding")
DEFAULT_AUDIT_DMF = smoke_audit_dir("dmf")
DEFAULT_AUDIT_PERCEPTION = smoke_audit_dir("perception")
DEFAULT_AUDIT_BASINS = smoke_audit_dir("basins")
DEFAULT_AUDIT_LUCIDITY = smoke_audit_dir("lucidity")
