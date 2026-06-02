"""Perception backend settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from lucid.paths import DEFAULT_AUDIT_PERCEPTION


def _parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[7:].strip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return (key, value) if key else None


def _find_dotenv(start: Path | None = None) -> Path | None:
    here = (start or Path.cwd()).resolve()
    for directory in [here, *here.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
        if (directory / "pyproject.toml").is_file():
            return None
    return None


def load_dotenv(path: Path | None = None, *, override: bool = False) -> bool:
    dotenv = path or _find_dotenv()
    if dotenv is None or not dotenv.is_file():
        return False
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value
    return True


@dataclass(slots=True)
class PerceptionConfig:
    backend: str = "llm"  # llm (default) | rule
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    timeout_s: float = 90.0
    use_json_schema: bool = True  # OpenAI-style structured output when supported
    audit_dir: str = DEFAULT_AUDIT_PERCEPTION
    write_audit: bool = True
    min_text_units: int = 1

    @classmethod
    def from_env(cls) -> PerceptionConfig:
        load_dotenv()
        return cls(
            backend=os.environ.get("LUCID_PERCEPTION_BACKEND", "llm").strip().lower(),
            model=os.environ.get("LUCID_PERCEPTION_MODEL", "gpt-4o-mini"),
            base_url=os.environ.get(
                "LUCID_PERCEPTION_BASE_URL",
                "https://api.openai.com/v1",
            ).rstrip("/"),
            api_key=(
                os.environ.get("LUCID_PERCEPTION_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or ""
            ),
            timeout_s=float(os.environ.get("LUCID_PERCEPTION_TIMEOUT_S", "90")),
            use_json_schema=os.environ.get("LUCID_PERCEPTION_USE_JSON_SCHEMA", "1").strip()
            not in ("0", "false", "no"),
            audit_dir=os.environ.get("LUCID_PERCEPTION_AUDIT_DIR", DEFAULT_AUDIT_PERCEPTION),
            write_audit=os.environ.get("LUCID_PERCEPTION_WRITE_AUDIT", "1").strip()
            not in ("0", "false", "no"),
            min_text_units=int(os.environ.get("LUCID_PERCEPTION_MIN_TEXT_UNITS", "1")),
        )
