"""Readable smoke audits: ``<base>/<module>/<YYYYMMDDTHHMMSSZ>_<label>/``."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from lucid.audit.logger import content_hash
from lucid.audit.sanitize import sanitize_audit_value
from lucid.ir.serde import to_dict


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_token(value: str, *, max_len: int = 48) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return (clean.strip("_") or "run")[:max_len]


def new_run_id(*, label: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{safe_token(label)}"


def smoke_run_dir(
    module: str,
    *,
    label: str = "run",
    audit_base_dir: str | Path,
) -> tuple[Path, str]:
    root = Path(audit_base_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id(label=label)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


def write_smoke_run(
    *,
    module: str,
    label: str,
    stage_input: Any,
    stage_output: Any,
    audit_base_dir: str | Path,
    build_manifest_extra: Callable[[Any, Any], dict[str, Any]],
    build_readme_lines: Callable[[str, str, Any, Any], list[str]],
    details: dict[str, Any] | None = None,
) -> Path:
    run_dir, run_id = smoke_run_dir(module, label=label, audit_base_dir=audit_base_dir)
    files = {
        "input": "input.json",
        "output": "output.json",
        "manifest": "manifest.json",
        "readme": "README.txt",
    }
    safe_in = sanitize_audit_value(stage_input)
    safe_out = sanitize_audit_value(stage_output)
    safe_details = sanitize_audit_value(details or {})

    (run_dir / files["input"]).write_text(
        json.dumps(to_dict(safe_in), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / files["output"]).write_text(
        json.dumps(to_dict(safe_out), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "kind": "smoke",
        "created_at": utc_now_iso(),
        "run_id": run_id,
        "module": module,
        "stage_name": module,
        "label": label,
        "input_hash": content_hash(stage_input),
        "output_hash": content_hash(stage_output),
        "files": files,
        "details": safe_details,
    }
    manifest.update(sanitize_audit_value(build_manifest_extra(stage_input, stage_output)))
    (run_dir / files["manifest"]).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = build_readme_lines(module, label, stage_input, stage_output)
    (run_dir / files["readme"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir
