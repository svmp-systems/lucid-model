"""Audit logger — readable + machine-parseable JSON per stage."""

from lucid.audit.inspect import format_manifest, print_run
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
