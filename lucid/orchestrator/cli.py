"""Compatibility wrapper for ``lucid.cognition.orchestrator.cli``."""

from lucid.cognition.orchestrator.cli import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
