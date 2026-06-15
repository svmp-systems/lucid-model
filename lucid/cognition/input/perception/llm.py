"""LLM perception — schema on API output via response_format."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lucid.audit.logger import AuditLogger, content_hash
from lucid.runtime.paths import resolve_train_path
from lucid.ir.common import Modality, Provenance
from lucid.ir.perception import PerceptionInput, PerceptualEvidenceGraph
from lucid.ir.pipeline import RunContext
from lucid.ir.serde import to_dict

from lucid.cognition.input.perception.config import PerceptionConfig
from lucid.cognition.input.perception.schema import (
    build_messages,
    empty_graph_retry_message,
    graph_from_dict,
    graph_has_text_evidence,
    json_object_response_format,
    structured_response_format,
)

_AUDIT_VERSION = "perception-llm-detail-v1"
_STAGE_NAME = "perception_llm"
_ADAPTER_VERSION = "llm-perception-v1"
_MAX_ATTEMPTS = 3


def payload_hash(payload: Any) -> str:
    return content_hash(payload)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def perception_run_id(inp: PerceptionInput) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"perception-{stamp}-{payload_hash(inp.raw_payload)[:12]}"


def perception_audit_path(base_dir: str | Path, run_id: str) -> Path:
    return resolve_train_path(base_dir) / f"{run_id}.json"


def _input_summary(inp: PerceptionInput) -> dict[str, Any]:
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


def _graph_summary(graph: PerceptualEvidenceGraph | None) -> dict[str, Any]:
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


def _config_summary(config: PerceptionConfig) -> dict[str, Any]:
    return {
        "backend": config.backend,
        "model": config.model,
        "base_url": config.base_url,
        "timeout_s": config.timeout_s,
        "use_json_schema": config.use_json_schema,
        "min_text_units": config.min_text_units,
        "api_key_present": bool(config.api_key),
    }


def _human_summary(
    inp: PerceptionInput,
    config: PerceptionConfig,
    attempts: list[dict[str, Any]],
    graph: PerceptualEvidenceGraph | None,
    error: str | None,
) -> str:
    counts = _graph_summary(graph)
    if error:
        return (
            f"LLM perception failed after {len(attempts)} attempt(s) "
            f"on {inp.modality}: {error}"
        )
    return (
        f"LLM perception using {config.model} produced "
        f"{counts.get('candidate_units', 0)} unit(s), "
        f"{counts.get('candidate_markers', 0)} marker(s), "
        f"{counts.get('uncertainty_flags', 0)} uncertainty flag(s) "
        f"after {len(attempts)} attempt(s)."
    )


def perceive_llm(
    inp: PerceptionInput,
    cfg: PerceptionConfig,
    *,
    context: Any = None,
) -> PerceptualEvidenceGraph:
    if not cfg.api_key:
        raise ValueError("LUCID_PERCEPTION_API_KEY or OPENAI_API_KEY required for llm backend")

    modality = inp.modality if isinstance(inp.modality, Modality) else Modality(str(inp.modality))
    messages = build_messages(inp, context=context)
    run_id = perception_run_id(inp)
    attempts: list[dict[str, Any]] = []
    last_err: Exception | None = None

    for attempt in range(_MAX_ATTEMPTS):
        raw = ""
        data: dict[str, Any] | None = None
        response_format = "unknown"
        try:
            chat_result = _chat(cfg, messages)
            if isinstance(chat_result, tuple):
                raw, response_format = chat_result
            else:
                raw, response_format = chat_result, "unknown"
            data = _parse_json(raw)
            graph = graph_from_dict(data, modality=modality)
            if isinstance(inp.raw_payload, str):
                graph.provenance.extra["raw_text"] = inp.raw_payload.strip()
            graph.provenance.extra["model"] = cfg.model
            graph.provenance.extra["llm_attempt"] = attempt + 1
            graph.provenance.extra["input_hash"] = payload_hash(inp.raw_payload)
            graph.provenance.extra["perception_run_id"] = run_id
            graph.provenance.extra["perception_audit_path"] = str(
                perception_audit_path(cfg.audit_dir, run_id)
            )
            graph.provenance.adapter_version = "llm-perception-v1"
            graph.provenance.segmentation_pass_id = f"llm:{cfg.model}:attempt-{attempt + 1}"
            _require_text_evidence(inp, graph, min_units=cfg.min_text_units)
            attempts.append(_attempt_record(attempt + 1, raw, data, response_format, graph=graph))
            _write_audit(cfg, run_id, inp, messages, attempts, graph)
            return graph
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            last_err = exc
            attempts.append(
                _attempt_record(
                    attempt + 1,
                    raw,
                    data,
                    response_format,
                    error=str(exc),
                )
            )
            messages.append({"role": "user", "content": _retry_message(exc, inp)})

    error = f"llm perception failed after {_MAX_ATTEMPTS} attempts: {last_err}"
    _write_audit(cfg, run_id, inp, messages, attempts, None, error=error)
    raise RuntimeError(error) from last_err


def _retry_message(exc: Exception, inp: PerceptionInput) -> str:
    if "empty evidence graph" in str(exc):
        return empty_graph_retry_message()
    payload = inp.raw_payload if isinstance(inp.raw_payload, str) else str(inp.raw_payload)
    return (
        f"Invalid output: {exc}. Return valid JSON matching the response schema. "
        f"Payload to analyze: {payload[:500]}"
    )


def _require_text_evidence(
    inp: PerceptionInput,
    graph: PerceptualEvidenceGraph,
    *,
    min_units: int = 1,
) -> None:
    modality = inp.modality if isinstance(inp.modality, Modality) else Modality(str(inp.modality))
    if modality != Modality.TEXT:
        return
    payload = inp.raw_payload if isinstance(inp.raw_payload, str) else str(inp.raw_payload)
    if not payload.strip():
        return
    if not graph_has_text_evidence(graph):
        raise ValueError(
            "model returned empty evidence graph for non-empty text "
            "(candidate_units and candidate_markers are both empty)"
        )
    if len(graph.candidate_units) < min_units:
        raise ValueError(
            f"model returned too few candidate_units for non-empty text "
            f"({len(graph.candidate_units)} < {min_units})"
        )


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in response")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("root must be a JSON object")
    return parsed


def _chat(cfg: PerceptionConfig, messages: list[dict[str, str]]) -> tuple[str, str]:
    url = f"{cfg.base_url}/chat/completions"
    base_body: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": 0,
    }

    formats: list[dict[str, Any] | None] = [None]
    if cfg.use_json_schema:
        formats = [structured_response_format(), json_object_response_format(), None]

    last_error: Exception | None = None
    for response_format in formats:
        body = dict(base_body)
        response_format_name = "none"
        if response_format is not None:
            body["response_format"] = response_format
            response_format_name = response_format.get("type", "unknown")
        try:
            payload = _post(cfg, url, body)
        except RuntimeError as exc:
            if response_format is not None and _is_format_rejected(str(exc)):
                last_error = exc
                continue
            raise
        content = (payload.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        if not content.strip():
            raise RuntimeError("empty model response")
        choice = (payload.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            raise RuntimeError("model hit max tokens before completing JSON")
        return content, response_format_name

    raise RuntimeError(f"perception API failed for all response formats: {last_error}")


def _attempt_record(
    attempt: int,
    raw: str,
    parsed: dict[str, Any] | None,
    response_format: str,
    *,
    graph: PerceptualEvidenceGraph | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "timestamp": utc_now(),
        "response_format": response_format,
        "raw_response": raw,
        "parsed": parsed,
        "graph_summary": {
            "candidate_units": len(graph.candidate_units) if graph else 0,
            "candidate_markers": len(graph.candidate_markers) if graph else 0,
            "uncertainty_flags": len(graph.uncertainty_flags) if graph else 0,
        },
        "error": error,
    }


def _write_audit(
    cfg: PerceptionConfig,
    run_id: str,
    inp: PerceptionInput,
    messages: list[dict[str, str]],
    attempts: list[dict[str, Any]],
    graph: PerceptualEvidenceGraph | None,
    *,
    error: str | None = None,
) -> None:
    if not cfg.write_audit:
        return
    base = resolve_train_path(cfg.audit_dir, mkdir=True)
    path = perception_audit_path(base, run_id)
    success = graph is not None and error is None
    provenance = graph.provenance if graph is not None else Provenance(modality=inp.modality)

    AuditLogger(base_dir=base, adapter_version=_ADAPTER_VERSION).write_stage(
        run_dir=base,
        context=RunContext(run_id=run_id),
        stage_name=_STAGE_NAME,
        stage_input=_input_summary(inp),
        stage_output={
            "audit_version": _AUDIT_VERSION,
            "success": success,
            "human_summary": _human_summary(inp, cfg, attempts, graph, error),
            "config": _config_summary(cfg),
            "messages": messages,
            "attempts": attempts,
            "graph_summary": _graph_summary(graph),
            "graph": to_dict(graph) if graph is not None else None,
            "error": error,
        },
        file_name=path.name,
        success=success,
        error_message=error or "",
        provenance=provenance,
    )


def _is_format_rejected(msg: str) -> bool:
    return any(token in msg for token in ("400", "422", "response_format", "json_schema"))


def _post(cfg: PerceptionConfig, url: str, body: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"perception API {exc.code}: {detail}") from exc
