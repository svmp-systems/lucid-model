"""Train-tree layout helpers and run listing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lucid.runtime.paths import (
    DEFAULT_AUDIT_RUNS,
    DEFAULT_AUDIT_SCALING,
    DEFAULT_AUDIT_SMOKE,
    DEFAULT_AUDIT_TRAINING_RUNS,
    resolve_train_path,
    train_root,
)

AuditKind = Literal["smoke", "training", "pipeline", "scaling"]

PIPELINE_MODULES: tuple[str, ...] = (
    "perception",
    "cue_encoder",
    "dmf",
    "binding",
    "context_op",
    "interference",
    "basins",
    "lucidity",
    "projector",
    "decoder",
)


@dataclass(slots=True)
class AuditRunRef:
    kind: AuditKind
    path: Path
    run_id: str
    module: str
    headline: str


def audit_kind_root(kind: AuditKind, module: str = "") -> Path:
    if kind == "smoke":
        base = resolve_train_path(DEFAULT_AUDIT_SMOKE)
        return base / module if module else base
    if kind == "training":
        return resolve_train_path(DEFAULT_AUDIT_TRAINING_RUNS)
    if kind == "pipeline":
        return resolve_train_path(DEFAULT_AUDIT_RUNS)
    return resolve_train_path(DEFAULT_AUDIT_SCALING)


def _headline_from_manifest(manifest_path: Path) -> str:
    if not manifest_path.is_file():
        return ""
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    summary = data.get("summary") or {}
    if isinstance(summary, dict) and summary.get("headline"):
        return str(summary["headline"])
    module = data.get("module") or data.get("stage_name") or ""
    metrics = data.get("metrics") or {}
    if isinstance(metrics, dict) and metrics.get("update_count") is not None:
        return (
            f"{module} training · {metrics.get('update_count', 0)} updates / "
            f"{metrics.get('step_count', 0)} steps"
        )
    prim = data.get("primitive_activation_count")
    if prim is not None:
        return f"{module} · {prim} primitive activations"
    frames = data.get("candidate_frame_count")
    if frames is not None:
        return f"{module} · {frames} frames"
    return str(data.get("run_id") or manifest_path.parent.name)


def list_runs(
    kind: AuditKind,
    *,
    module: str = "",
    limit: int = 30,
) -> list[AuditRunRef]:
    if kind == "scaling":
        points = resolve_train_path(DEFAULT_AUDIT_SCALING) / "points.jsonl"
        if points.is_file():
            lines = [ln for ln in points.read_text(encoding="utf-8").splitlines() if ln.strip()]
            return [
                AuditRunRef(
                    kind=kind,
                    path=points,
                    run_id="points.jsonl",
                    module="scaling",
                    headline=f"{len(lines)} scaling points",
                )
            ]
        return []

    if kind == "smoke" and module:
        roots = [audit_kind_root(kind, module)]
    elif kind == "smoke":
        smoke_root = audit_kind_root("smoke")
        roots = [smoke_root / name for name in PIPELINE_MODULES if (smoke_root / name).is_dir()]
    else:
        roots = [audit_kind_root(kind)]

    refs: list[AuditRunRef] = []
    for root in roots:
        if not root.is_dir():
            continue
        mod_name = module or (root.name if kind == "smoke" else "")
        for child in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
            if not child.is_dir():
                continue
            manifest = child / "manifest.json"
            if not manifest.is_file():
                continue
            refs.append(
                AuditRunRef(
                    kind=kind,
                    path=child,
                    run_id=child.name,
                    module=mod_name or str(
                        json.loads(manifest.read_text(encoding="utf-8")).get("module", "")
                    ),
                    headline=_headline_from_manifest(manifest),
                )
            )
            if len(refs) >= limit:
                return refs
    return refs


def format_run_list(refs: list[AuditRunRef]) -> str:
    if not refs:
        return "(no runs)"
    lines = ["kind          module           run_id                         headline", "-" * 90]
    for ref in refs:
        lines.append(
            f"{ref.kind:<13} {ref.module:<16} {ref.run_id:<30} {ref.headline[:40]}"
        )
    return "\n".join(lines)


def list_checkpoints() -> list[tuple[str, Path]]:
    root = train_root() / "checkpoints"
    if not root.is_dir():
        return []
    return [
        (child.name, child)
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / "manifest.json").is_file()
    ]


def format_checkpoint_list(entries: list[tuple[str, Path]]) -> str:
    if not entries:
        return f"(no checkpoints under {train_root() / 'checkpoints'})"
    lines = ["checkpoint_id    path", "-" * 60]
    for cid, path in entries:
        lines.append(f"{cid:<16} {path}")
    return "\n".join(lines)


def write_train_readmes() -> None:
    root = train_root()
    (root / "README.txt").write_text(
        "\n".join(
            [
                "Lucid training artifacts (local only — not committed except this file)",
                "",
                "checkpoints/training/   mutable training workspace (lucid train)",
                "checkpoints/loaded/     inference save point (lucid checkpoint load)",
                "checkpoints/saves/      named archives (lucid checkpoint save)",
                "audit/runs/smoke/      CLI smoke audits (readable folder names)",
                "audit/runs/training/   lucid train module runs",
                "audit/runs/pipeline/   lucid run full pipeline",
                "audit/scaling/         points.jsonl (secrets redacted on write)",
                "data/generated/        lucid-gen output",
                "",
            ]
        ),
        encoding="utf-8",
    )
