"""Shared contracts and audit writer for module trainers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from lucid.audit.logger import content_hash
from lucid.ir.training import Episode
from lucid.training.checkpoint.store import CheckpointState


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class TrainingResult:
    module: str
    action: str
    episode_id: str
    updated_objects: list[str] = field(default_factory=list)
    before_hash: str = ""
    after_hash: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    audit_path: str = ""
    reason: str = ""


class ModuleTrainer(ABC):
    name: str
    store_name: str

    @abstractmethod
    def train(
        self,
        episode: Episode,
        state: CheckpointState,
        audit_dir: Path,
    ) -> TrainingResult:
        """Apply one local module update to checkpoint state."""


def write_module_audit(
    *,
    audit_dir: Path,
    module: str,
    episode: Episode,
    action: str,
    reason: str,
    before: Any,
    after: Any,
    updated_objects: list[str],
    metrics: dict[str, Any],
) -> TrainingResult:
    audit_dir.mkdir(parents=True, exist_ok=True)
    before_hash = content_hash(before)
    after_hash = content_hash(after)
    result = TrainingResult(
        module=module,
        action=action,
        episode_id=episode.episode_id,
        updated_objects=updated_objects,
        before_hash=before_hash,
        after_hash=after_hash,
        metrics=metrics,
        reason=reason,
    )

    files = {
        "before": "before.json",
        "after": "after.json",
        "module_update": "module_update.json",
        "metrics": "metrics.json",
    }
    (audit_dir / files["before"]).write_text(
        json.dumps(before, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (audit_dir / files["after"]).write_text(
        json.dumps(after, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (audit_dir / files["metrics"]).write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "event_id": str(uuid4()),
        "created_at": utc_now_iso(),
        "module": module,
        "episode_id": episode.episode_id,
        "action": action,
        "reason": reason,
        "updated_objects": updated_objects,
        "before_hash": before_hash,
        "after_hash": after_hash,
        "metrics": metrics,
    }
    (audit_dir / files["module_update"]).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "module": module,
        "episode_id": episode.episode_id,
        "action": action,
        "reason": reason,
        "before_hash": before_hash,
        "after_hash": after_hash,
        "files": files,
    }
    (audit_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (audit_dir / "README.txt").write_text(
        "\n".join(
            [
                f"{module} training - {episode.episode_id}",
                "=" * (len(module) + len(episode.episode_id) + 12),
                "",
                f"action: {action}",
                f"reason: {reason}",
                f"updated_objects: {', '.join(updated_objects) if updated_objects else '-'}",
                f"before_hash: {before_hash}",
                f"after_hash: {after_hash}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    result.audit_path = str(audit_dir)
    return result
