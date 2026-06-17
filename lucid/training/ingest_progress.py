"""Line-oriented progress logging for long ingest runs."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid.training.ingest_config import IngestConfig


def should_log_progress(step: int, total: int, interval: int) -> bool:
    """Return True when *step* (1-based) should emit a progress line."""
    if total <= 0 or step <= 0:
        return False
    if interval <= 1:
        return True
    if step == 1 or step == total:
        return True
    return step % interval == 0


def ingest_log(message: str, config: IngestConfig | None = None) -> None:
    """Write a timestamped progress line to stderr (unbuffered in most shells)."""
    if config is not None and not config.progress_logging:
        return
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[ingest {ts}] {message}", file=sys.stderr, flush=True)
