"""Perception backend configuration (env + dataclass)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(slots=True)
class PerceptionConfig:
    """How perception runs. Default ``rule`` works offline; ``llm`` calls an API."""

    backend: str = "rule"  # rule | llm
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    temperature: float = 0.0
    timeout_s: float = 90.0
    max_retries: int = 1
    adapter_version: str = "perception-0.1.0"

    @classmethod
    def from_env(cls) -> PerceptionConfig:
        backend = _env("LUCID_PERCEPTION_BACKEND", "rule").lower()
        return cls(
            backend=backend,
            model=_env("LUCID_PERCEPTION_MODEL", "gpt-4o-mini"),
            base_url=_env("LUCID_PERCEPTION_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            api_key=_env("LUCID_PERCEPTION_API_KEY") or _env("OPENAI_API_KEY"),
            temperature=float(_env("LUCID_PERCEPTION_TEMPERATURE", "0")),
            timeout_s=float(_env("LUCID_PERCEPTION_TIMEOUT_S", "90")),
            max_retries=int(_env("LUCID_PERCEPTION_MAX_RETRIES", "1")),
            adapter_version=_env("LUCID_PERCEPTION_ADAPTER_VERSION", "perception-0.1.0"),
        )
