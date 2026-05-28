"""LLM perception — schema on API output via response_format."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from lucid.ir.common import Modality
from lucid.ir.perception import PerceptionInput, PerceptualEvidenceGraph

from lucid.perception.config import PerceptionConfig
from lucid.perception.parse import graph_from_dict
from lucid.perception.schema import (
    build_messages,
    empty_graph_retry_message,
    graph_has_text_evidence,
    json_object_response_format,
    structured_response_format,
)

_MAX_ATTEMPTS = 3


def perceive_llm(inp: PerceptionInput, cfg: PerceptionConfig) -> PerceptualEvidenceGraph:
    if not cfg.api_key:
        raise ValueError("LUCID_PERCEPTION_API_KEY or OPENAI_API_KEY required for llm backend")

    modality = inp.modality if isinstance(inp.modality, Modality) else Modality(str(inp.modality))
    messages = build_messages(inp)
    last_err: Exception | None = None

    for attempt in range(_MAX_ATTEMPTS):
        try:
            raw = _chat(cfg, messages)
            data = _parse_json(raw)
            graph = graph_from_dict(data, modality=modality)
            graph.provenance.extra["model"] = cfg.model
            graph.provenance.extra["llm_attempt"] = attempt + 1
            _require_text_evidence(inp, graph)
            return graph
        except (ValueError, json.JSONDecodeError) as exc:
            last_err = exc
            messages.append({"role": "user", "content": _retry_message(exc, inp)})

    raise RuntimeError(f"llm perception failed after {_MAX_ATTEMPTS} attempts: {last_err}") from last_err


def _retry_message(exc: Exception, inp: PerceptionInput) -> str:
    if "empty evidence graph" in str(exc):
        return empty_graph_retry_message()
    payload = inp.raw_payload if isinstance(inp.raw_payload, str) else str(inp.raw_payload)
    return (
        f"Invalid output: {exc}. Return valid JSON matching the response schema. "
        f"Payload to analyze: {payload[:500]}"
    )


def _require_text_evidence(inp: PerceptionInput, graph: PerceptualEvidenceGraph) -> None:
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


def _chat(cfg: PerceptionConfig, messages: list[dict[str, str]]) -> str:
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
        if response_format is not None:
            body["response_format"] = response_format
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
        return content

    raise RuntimeError(f"perception API failed for all response formats: {last_error}")


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
