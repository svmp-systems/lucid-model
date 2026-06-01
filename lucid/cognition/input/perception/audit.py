"""LLM perception audit detail written through ``lucid.audit.logger``."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lucid.audit.logger import AuditLogger, content_hash
from lucid.ir.common import Provenance
from lucid.ir.perception import PerceptionInput, PerceptualEvidenceGraph
from lucid.ir.pipeline import RunContext
from lucid.ir.serde import to_dict

_AUDIT_VERSION = "perception-llm-detail-v1"
_STAGE_NAME = "perception_llm"
_ADAPTER_VERSION = "llm-perception-v1"


def payload_hash(payload: Any) -> str:
    return content_hash(payload)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def perception_run_id(inp: PerceptionInput) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"perception-{stamp}-{payload_hash(inp.raw_payload)[:12]}"


def perception_audit_path(base_dir: str | Path, run_id: str) -> Path:
    return Path(base_dir) / f"{run_id}.json"


def input_summary(inp: PerceptionInput) -> dict[str, Any]:
    modality = inp.modality.value if hasattr(inp.modality, "value") else str(inp.modality)
    payload = inp.raw_payload
    if isinstance(payload, str):
        preview = payload[:500]
        payload_type = "str"
        payload_size = len(payload)
    else:
        preview = repr(payload)[:500]
        payload_type = type(payload).__name__
        payload_size = len(json.dumps(payload, default=str))
    return {
        "modality": modality,
        "payload_type": payload_type,
        "payload_size": payload_size,
        "payload_preview": preview,
        "input_hash": payload_hash(payload),
        "task_intent_hint": getattr(inp.task_intent_hint, "value", inp.task_intent_hint),
        "prior_context_keys": sorted(inp.prior_context.keys()),
        "provenance_seed": inp.provenance_seed,
    }


def graph_summary(graph: PerceptualEvidenceGraph | None) -> dict[str, Any]:
    if graph is None:
        return {}
    return {
        "candidate_units": len(graph.candidate_units),
        "candidate_regions": len(graph.candidate_regions),
        "candidate_containers": len(graph.candidate_containers),
        "candidate_markers": len(graph.candidate_markers),
        "arrangement_hints": len(graph.arrangement_hints),
        "change_hints": len(graph.change_hints),
        "grouping_hints": len(graph.grouping_hints),
        "reference_hints": len(graph.reference_hints),
        "uncertainty_flags": len(graph.uncertainty_flags),
    }


def write_perception_audit(
    *,
    base_dir: str | Path,
    run_id: str,
    inp: PerceptionInput,
    config: Any,
    messages: list[dict[str, str]],
    attempts: list[dict[str, Any]],
    graph: PerceptualEvidenceGraph | None,
    error: str | None = None,
) -> Path:
    """Write LLM-specific perception detail using the shared audit logger."""
    base = Path(base_dir)
    path = perception_audit_path(base, run_id)
    success = graph is not None and error is None
    provenance = graph.provenance if graph is not None else Provenance(modality=inp.modality)

    AuditLogger(base_dir=base, adapter_version=_ADAPTER_VERSION).write_stage(
        run_dir=base,
        context=RunContext(run_id=run_id),
        stage_name=_STAGE_NAME,
        stage_input=input_summary(inp),
        stage_output={
            "audit_version": _AUDIT_VERSION,
            "success": success,
            "human_summary": _human_summary(inp, config, attempts, graph, error),
            "config": _config_summary(config),
            "messages": messages,
            "attempts": attempts,
            "graph_summary": graph_summary(graph),
            "graph": to_dict(graph) if graph is not None else None,
            "error": error,
        },
        file_name=path.name,
        success=success,
        error_message=error or "",
        provenance=provenance,
    )
    return path


def _config_summary(config: Any) -> dict[str, Any]:
    return {
        "backend": getattr(config, "backend", ""),
        "model": getattr(config, "model", ""),
        "base_url": getattr(config, "base_url", ""),
        "timeout_s": getattr(config, "timeout_s", None),
        "use_json_schema": getattr(config, "use_json_schema", None),
        "min_text_units": getattr(config, "min_text_units", None),
        "api_key_present": bool(getattr(config, "api_key", "")),
    }


def _human_summary(
    inp: PerceptionInput,
    config: Any,
    attempts: list[dict[str, Any]],
    graph: PerceptualEvidenceGraph | None,
    error: str | None,
) -> str:
    model = getattr(config, "model", "unknown")
    counts = graph_summary(graph)
    if error:
        return (
            f"LLM perception failed after {len(attempts)} attempt(s) "
            f"on {inp.modality}: {error}"
        )
    return (
        f"LLM perception using {model} produced "
        f"{counts.get('candidate_units', 0)} unit(s), "
        f"{counts.get('candidate_markers', 0)} marker(s), "
        f"{counts.get('uncertainty_flags', 0)} uncertainty flag(s) "
        f"after {len(attempts)} attempt(s)."
    )
