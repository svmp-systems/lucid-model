"""Training vs loaded checkpoint slots — training writes one tree, inference reads another."""

from __future__ import annotations

import json
import re
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
    train_root,
)
from lucid.training.checkpoint.registry import (
    allocate_standard_name,
    list_registry,
    lookup_registry,
    register_checkpoint,
    sanitize_checkpoint_name,
)
from lucid.training.checkpoint.store import checkpoint_summary, load_checkpoint

_CP_REF = re.compile(r"^cp_\d{3,}$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def checkpoint_ref(path: str | Path) -> str:
    """Train-tree-relative checkpoint path (e.g. ``checkpoints/saves/cp_001``)."""
    resolved = resolve_checkpoint(path)
    try:
        rel = resolved.relative_to(train_root())
    except ValueError:
        return str(path).replace("\\", "/")
    return str(rel).replace("\\", "/")


def resolve_checkpoint_ref(text: str | Path) -> str:
    """Resolve shorthand (``cp_001``, ``training``, ``loaded``) to a train-tree path."""
    return checkpoint_ref(_normalize_slot_path(text))


def _normalize_slot_path(path: str | Path) -> Path:
    text = str(path).replace("\\", "/").strip()
    if text in {"training", "train"}:
        return resolve_checkpoint(DEFAULT_TRAINING_CHECKPOINT)
    if text in {"loaded", "active", "savepoint", "pinned"}:
        return resolve_checkpoint(DEFAULT_LOADED_CHECKPOINT)
    if text.startswith("saves/"):
        return resolve_checkpoint(f"checkpoints/{text}")
    if text.startswith("checkpoints/saves/"):
        return resolve_checkpoint(text)
    if _CP_REF.match(text):
        return resolve_checkpoint(f"checkpoints/saves/{text}")
    if "/" not in text and text not in {"local", "training", "loaded"}:
        by_save = resolve_checkpoint(f"checkpoints/saves/{text}")
        if (by_save / "manifest.json").is_file():
            return by_save
        row = lookup_registry(text)
        if row and Path(str(row.get("path", ""))).is_dir():
            candidate = Path(str(row["path"]))
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
    save_name: str = "",
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
        "save_name": save_name.strip(),
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
    save_name = src.name if src.parent.name == "saves" else ""
    write_loaded_pointer(source_path=src, label=label, save_name=save_name)
    return dst


def save_training_snapshot(name: str, *, source: str | Path = DEFAULT_TRAINING_CHECKPOINT) -> Path:
    """Archive a checkpoint tree under ``checkpoints/saves/<name>``."""
    clean = sanitize_checkpoint_name(name)
    dest = f"checkpoints/saves/{clean}"
    return copy_checkpoint_tree(source, dest)


def archive_training_checkpoint(
    *,
    source: str | Path = DEFAULT_TRAINING_CHECKPOINT,
    name: str | None = None,
    label: str = "",
    command: str = "",
) -> dict[str, Any]:
    """Archive a checkpoint to ``checkpoints/saves/cp_NNN`` (or explicit name) and register it."""
    src = resolve_checkpoint(source)
    if not (src / "manifest.json").is_file():
        raise FileNotFoundError("checkpoint is empty — run training before archiving")

    save_name = sanitize_checkpoint_name(name) if name else allocate_standard_name()
    dest = save_training_snapshot(save_name, source=src)
    summary = checkpoint_summary(load_checkpoint(dest, create=False))
    return register_checkpoint(
        name=save_name,
        path=dest,
        label=label,
        command=command,
        summary=summary,
    )


def clear_loaded_checkpoint() -> None:
    """Remove loaded snapshot and pointer (inference runs cold until next load)."""
    loaded = resolve_checkpoint(DEFAULT_LOADED_CHECKPOINT)
    if loaded.exists():
        shutil.rmtree(loaded)
    clear_loaded_pointer()


def resolve_training_checkpoint(explicit: str | None = None) -> str:
    text = (explicit or "").strip()
    if text:
        return resolve_checkpoint_ref(text)
    return DEFAULT_TRAINING_CHECKPOINT


def resolve_inference_checkpoint(
    explicit: str | None = None,
    *,
    cold: bool = False,
) -> str | None:
    """Checkpoint path for ``lucid ask`` / ``lucid run`` (loaded save point unless overridden)."""
    text = (explicit or "").strip()
    if text:
        return resolve_checkpoint_ref(text)
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
        if pointer.get("save_name"):
            lines.append(f"  save: {pointer['save_name']}")
        if pointer.get("source_label"):
            lines.append(f"  label: {pointer['source_label']}")
        lines.append(f"  loaded_at: {pointer.get('loaded_at', '?')}")
        lines.append(f"  steps: {pointer.get('training_steps', '?')}")
    else:
        lines.append("  (not loaded — lucid ask runs without memory until lucid checkpoint load)")

    registry = list_registry()
    lines.extend(["", f"standard saves ({len(registry)})"])
    if registry:
        for row in registry:
            tag = " [loaded]" if pointer and row.get("name") == pointer.get("save_name") else ""
            label = f" — {row['label']}" if row.get("label") else ""
            cmd = f" ({row['command']})" if row.get("command") else ""
            lines.append(
                f"  {row['name']}: steps={row.get('training_steps', '?')}{label}{cmd}{tag}"
            )
    else:
        lines.append("  (none — lucid train creates cp_001 after each run)")

    orphan_saves = [
        (name, path)
        for name, path in list_named_saves()
        if not any(row.get("name") == name for row in registry)
    ]
    if orphan_saves:
        lines.extend(["", f"other saves ({len(orphan_saves)})"])
        for name, path in orphan_saves:
            lines.append(f"  {name}: {path}")

    lines.extend(
        [
            "",
            "commands",
            "  lucid checkpoint load              copy training → loaded (pin for inference)",
            "  lucid checkpoint load cp_001       pin a standard save for inference",
            "  lucid checkpoint save <name>       manual archive training → saves/<name>",
            "  lucid checkpoint clear             drop loaded save point",
            "  lucid train … --pin              archive cp_NNN and pin it for inference",
            "  lucid ask \"…\" --checkpoint cp_001   one-off use a specific save",
        ]
    )
    return "\n".join(lines)
