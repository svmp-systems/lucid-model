"""Redact secrets and local paths before writing audit or scaling artifacts."""

from __future__ import annotations

import re
from typing import Any

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|secret|token|password|authorization|credential|private[_-]?key)",
    re.IGNORECASE,
)
_OPENAI_KEY_RE = re.compile(r"\bsk-[a-zA-Z0-9_-]{20,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+[a-zA-Z0-9._-]{20,}\b", re.IGNORECASE)
_LOCAL_PATH_RE = re.compile(
    r"[A-Za-z]:\\Users\\[^\\]+(?:\\.*)?|/Users/[^/]+(?:/.*)?|/home/[^/]+(?:/.*)?",
    re.IGNORECASE,
)

_REDACTED = "[redacted]"


def redact_string(value: str) -> str:
    if not value:
        return value
    out = _OPENAI_KEY_RE.sub(_REDACTED, value)
    out = _BEARER_RE.sub("Bearer [redacted]", out)
    out = _LOCAL_PATH_RE.sub("[local-path-redacted]", out)
    return out


def sanitize_audit_value(value: Any) -> Any:
    """Deep-copy sanitize for JSON-serializable audit/scaling payloads."""

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if _SECRET_KEY_RE.search(str(key)):
                if isinstance(item, str) and item:
                    out[key] = _REDACTED
                elif item:
                    out[key] = True
                else:
                    out[key] = item
                continue
            out[key] = sanitize_audit_value(item)
        return out
    if isinstance(value, list):
        return [sanitize_audit_value(item) for item in value]
    if isinstance(value, str):
        return redact_string(value)
    return value


def perception_config_summary(config: Any) -> dict[str, Any]:
    """Safe perception settings for provenance (never includes api_key value)."""

    return sanitize_audit_value(
        {
            "backend": getattr(config, "backend", ""),
            "model": getattr(config, "model", ""),
            "base_url": getattr(config, "base_url", ""),
            "timeout_s": getattr(config, "timeout_s", None),
            "use_json_schema": getattr(config, "use_json_schema", None),
            "min_text_units": getattr(config, "min_text_units", None),
            "api_key_present": bool(getattr(config, "api_key", "") or ""),
            "write_audit": getattr(config, "write_audit", None),
        }
    )
