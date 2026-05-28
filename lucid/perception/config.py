"""Perception backend settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from lucid.perception.env import load_dotenv


@dataclass(slots=True)
class PerceptionConfig:
    backend: str = "llm"  # llm (default) | rule
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    timeout_s: float = 90.0
    use_json_schema: bool = True  # OpenAI-style structured output when supported

    @classmethod
    def from_env(cls) -> PerceptionConfig:
        load_dotenv()
        return cls(
            backend=os.environ.get("LUCID_PERCEPTION_BACKEND", "llm").strip().lower(),
            model=os.environ.get("LUCID_PERCEPTION_MODEL", "gpt-4o-mini"),
            base_url=os.environ.get("LUCID_PERCEPTION_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            api_key=(os.environ.get("LUCID_PERCEPTION_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""),
            timeout_s=float(os.environ.get("LUCID_PERCEPTION_TIMEOUT_S", "90")),
            use_json_schema=os.environ.get("LUCID_PERCEPTION_USE_JSON_SCHEMA", "1").strip()
            not in ("0", "false", "no"),
        )
