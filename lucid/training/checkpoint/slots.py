"""Training vs loaded checkpoint slots — training writes one tree, inference reads another."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lucid.runtime.paths import (
    DEFAULT_LOADED_CHECKPOINT,
    DEFAULT_TRAINING_CHECKPOINT,
    LOADED_CHECKPOINT_POINTER,
    resolve_checkpoint,
    resolve_train_path,
)
from lucid.training.checkpoint.store import STORE_FILES, checkpoint_summary, load_checkpoint


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_slot_path(path: str | Path) -> Path:
    text = str(path).replace("\\", "/").strip()
    if text in {"training", "train"}:
        return resolve_checkpoint(DEFAULT_TRAINING_CHECKPOINT)
    if text in {"loaded", "active", "savepoint"}:
        return resolve_checkpoint(DEFAULT_LOADED_CHECKPOINT)
    if text.startswith("saves/"):
        return resolve_checkpoint(f"checkpoints/{text}")
    if "/" not in text and text not in {"local", "training", "loaded"}:
        candidate = resolve_checkpoint(f"checkpoints/saves/{text}")
        if (candidate / "manifest.json").is_file():
            return candidate
    return resolve_checkpoint(path)


def loaded_pointer_path() -> Path:
    return resolve_train_path(LOADED_CHECKPOINT_POINTER)


def loaded_checkpoint_ready() -> bool:
    loaded = resolve_checkpoint(DEFAULT_LOADED_CHECKPOINT)
    return (loaded / "manifest.json").is_file()


def read_loaded_pointer() -> dict[str, Any] | None:
    path = loaded_pointer_path()
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_loaded_pointer(
    *,
    source_path: Path,
    label: str = "",
) -> dict[str, Any]:
    source = resolve_checkpoint(source_path)
    if not (source / "manifest.json").is_file():
        raise FileNotFoundError(f"no checkpoint manifest at {source}")

    summary = checkpoint_summary(load_checkpoint(source, create=False))
    record = {
        "schema_version": 1,
        "loaded_at": _utc_now_iso(),
        "source_path": str(source),
        "source_label": label.strip(),
        "checkpoint_id": summary["checkpoint_id"],
        "training_steps": summary["training_steps"],
        "store_counts": summary["store_counts"],
    }
    pointer = loaded_pointer_path()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def clear_loaded_pointer() -> None:
    path = loaded_pointer_path()
    if path.is_file():
        path.unlink()


def copy_checkpoint_tree(source: str | Path, dest: str | Path) -> Path:
    """Replace ``dest`` with a full copy of ``source`` checkpoint stores."""
    src = _normalize_slot_path(source)
    dst = resolve_checkpoint(dest)
    if not (src / "manifest.json").is_file():
        raise FileNotFoundError(f"no checkpoint manifest at {src}")

    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return dst


def promote_to_loaded(source: str | Path = DEFAULT_TRAINING_CHECKPOINT, *, label: str = "") -> Path:
    """Copy a checkpoint tree into the loaded slot (inference save point)."""
    src = _normalize_slot_path(source)
    dst = copy_checkpoint_tree(src, DEFAULT_LOADED_CHECKPOINT)
    write_loaded_pointer(source_path=src, label=label)
    return dst


def save_training_snapshot(name: str) -> Path:
    """Archive the current training checkpoint under ``checkpoints/saves/<name>``."""
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name.strip())
    clean = clean.strip("_") or "snapshot"
    dest = f"checkpoints/saves/{clean}"
    return copy_checkpoint_tree(DEFAULT_TRAINING_CHECKPOINT, dest)


def clear_loaded_checkpoint() -> None:
    """Remove loaded snapshot and pointer (inference runs cold until next load)."""
    loaded = resolve_checkpoint(DEFAULT_LOADED_CHECKPOINT)
    if loaded.exists():
        shutil.rmtree(loaded)
    clear_loaded_pointer()


def resolve_training_checkpoint(explicit: str | None = None) -> str:
    text = (explicit or "").strip()
    if text:
        return text
    return DEFAULT_TRAINING_CHECKPOINT


def resolve_inference_checkpoint(
    explicit: str | None = None,
    *,
    cold: bool = False,
) -> str | None:
    """Checkpoint path for ``lucid ask`` / ``lucid run`` (loaded save point unless overridden)."""
    text = (explicit or "").strip()
    if text:
        return text
    if cold:
        return None
    if loaded_checkpoint_ready():
        return DEFAULT_LOADED_CHECKPOINT
    return None


def list_named_saves() -> list[tuple[str, Path]]:
    root = resolve_train_path("checkpoints/saves")
    if not root.is_dir():
        return []
    return [
        (child.name, child)
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / "manifest.json").is_file()
    ]


def format_checkpoint_status() -> str:
    training = resolve_checkpoint(DEFAULT_TRAINING_CHECKPOINT)
    loaded = resolve_checkpoint(DEFAULT_LOADED_CHECKPOINT)
    lines = [
        "checkpoint slots",
        "=" * 14,
        "",
        "training (mutable — lucid train writes here)",
        f"  path: {training}",
    ]
    if (training / "manifest.json").is_file():
        summary = checkpoint_summary(load_checkpoint(training, create=False))
        lines.append(f"  steps: {summary['training_steps']}")
        lines.append(f"  tracebank: {summary['store_counts']['tracebank']}")
        lines.append(f"  basin_bank: {summary['store_counts']['basin_bank']}")
    else:
        lines.append("  (empty — run lucid train …)")

    legacy = resolve_checkpoint("checkpoints/local")
    if (legacy / "manifest.json").is_file() and legacy != training:
        lines.append(f"  legacy: {legacy} (use lucid checkpoint load checkpoints/local to adopt)")

    lines.extend(["", "loaded (inference save point — lucid ask / run use this when set)", f"  path: {loaded}"])
    pointer = read_loaded_pointer()
    if loaded_checkpoint_ready() and pointer:
        lines.append(f"  from: {pointer.get('source_path', '?')}")
        if pointer.get("source_label"):
            lines.append(f"  label: {pointer['source_label']}")
        lines.append(f"  loaded_at: {pointer.get('loaded_at', '?')}")
        lines.append(f"  steps: {pointer.get('training_steps', '?')}")
    else:
        lines.append("  (not loaded — lucid ask runs without memory until lucid checkpoint load)")

    saves = list_named_saves()
    lines.extend(["", f"named saves ({len(saves)})"])
    if saves:
        for name, path in saves:
            lines.append(f"  {name}: {path}")
    else:
        lines.append("  (none — lucid checkpoint save <name>)")

    lines.extend(
        [
            "",
            "commands",
            "  lucid checkpoint load              copy training → loaded",
            "  lucid checkpoint load <save>       copy saves/<save> → loaded",
            "  lucid checkpoint save <name>       archive training → saves/<name>",
            "  lucid checkpoint clear             drop loaded save point",
            "  lucid ask \"…\" --checkpoint training   one-off use training weights",
        ]
    )
    return "\n".join(lines)
