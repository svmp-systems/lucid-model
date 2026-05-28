"""Load ``.env`` into ``os.environ`` (stdlib only)."""

from __future__ import annotations

import os
from pathlib import Path


def _parse_line(line: str) -> tuple[str, str] | None:
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
    if not key:
        return None
    return key, value


def find_dotenv(start: Path | None = None) -> Path | None:
    """Find ``.env`` in cwd or parents (stop at repo root with pyproject.toml)."""
    here = (start or Path.cwd()).resolve()
    for directory in [here, *here.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
        if (directory / "pyproject.toml").is_file():
            return candidate if candidate.is_file() else None
    return None


def load_dotenv(path: Path | None = None, *, override: bool = False) -> bool:
    """Load variables from ``.env``. Returns True if a file was loaded."""
    dotenv = path or find_dotenv()
    if dotenv is None or not dotenv.is_file():
        return False
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        parsed = _parse_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value
    return True
