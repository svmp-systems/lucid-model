"""OpenAI-compatible chat API — perception JSON only."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from lucid.ir.common import Modality
from lucid.ir.perception import PerceptionInput, PerceptualEvidenceGraph

from lucid.perception.config import PerceptionConfig
from lucid.perception.prompt import build_messages
from lucid.perception.validator import merge_provenance, parse_graph_dict


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model response")
    return json.loads(text[start : end + 1])


class LlmPerceptionAdapter:
    adapter_id = "llm_v1"

    def __init__(self, config: PerceptionConfig) -> None:
        self.config = config
        if not config.api_key:
            raise ValueError(
                "LUCID_PERCEPTION_API_KEY or OPENAI_API_KEY required for llm perception backend"
            )

    def perceive(self, inp: PerceptionInput, *, context: object = None) -> PerceptualEvidenceGraph:
        modality = inp.modality if isinstance(inp.modality, Modality) else Modality(str(inp.modality))
        last_err: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                raw = self._chat(build_messages(inp))
                data = _extract_json(raw)
                graph = parse_graph_dict(data, modality=modality)
                merge_provenance(
                    graph,
                    adapter_version=self.config.adapter_version,
                    segmentation_pass_id=f"{self.adapter_id}_attempt_{attempt}",
                    extra={"backend": "llm", "model": self.config.model},
                )
                graph.provenance.extra["raw_model_response"] = raw[:4000]
                return graph
            except (ValueError, json.JSONDecodeError, KeyError) as exc:
                last_err = exc
                messages = build_messages(inp)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Previous output was invalid: "
                            f"{exc}. Return corrected JSON only, following all rules."
                        ),
                    }
                )
        raise RuntimeError(f"llm perception failed after retries: {last_err}") from last_err

    def _chat(self, messages: list[dict[str, str]]) -> str:
        url = f"{self.config.base_url}/chat/completions"
        payload_body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self.config.base_url.rstrip("/").endswith("openai.com/v1") or "json" in self.config.model:
            payload_body["response_format"] = {"type": "json_object"}

        try:
            payload = self._post_chat(url, payload_body)
        except urllib.error.HTTPError as exc:
            if exc.code == 400 and "response_format" in payload_body:
                payload_body.pop("response_format", None)
                payload = self._post_chat(url, payload_body)
            else:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"perception API HTTP {exc.code}: {detail}") from exc

        return self._content_from_payload(payload)

    def _post_chat(self, url: str, payload_body: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload_body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _content_from_payload(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("empty choices from perception API")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if not content.strip():
            raise RuntimeError("empty content from perception API")
        return content
