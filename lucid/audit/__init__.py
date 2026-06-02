"""Audit logger — readable + machine-parseable JSON per stage."""

from pathlib import Path

from lucid.audit.dmf import DmfTraceUpdateRecord, DmfUpdateAuditLogger
from lucid.audit.logger import (
    AuditLogger,
    RunAuditManifest,
    StageAuditRef,
    canonical_json,
    content_hash,
    resolve_run_dir,
    summarize_stage_output,
)
from lucid.audit.scaling import (
    ScalingConfig,
    ScalingPoint,
    record_pipeline_run,
    record_point,
    summarize_file,
)


def format_manifest(manifest: RunAuditManifest) -> str:
    from lucid.audit.inspect import format_manifest as _format_manifest

    return _format_manifest(manifest)


def print_run(run_dir: Path | str, *, stage: str | None = None) -> None:
    from lucid.audit.inspect import print_run as _print_run

    _print_run(run_dir, stage=stage)


__all__ = [
    "AuditLogger",
    "DmfTraceUpdateRecord",
    "DmfUpdateAuditLogger",
    "RunAuditManifest",
    "ScalingConfig",
    "ScalingPoint",
    "StageAuditRef",
    "canonical_json",
    "content_hash",
    "format_manifest",
    "print_run",
    "record_pipeline_run",
    "record_point",
    "resolve_run_dir",
    "summarize_file",
    "summarize_stage_output",
]
