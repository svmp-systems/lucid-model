"""Orchestrator: run the Lucid pipeline step-by-step with audits.

This module is intentionally small in Phase 1: it provides a concrete driver
that executes stages in order and records stage results. Real stage logic can
be swapped in via the stage registry.
"""

from lucid.orchestrator.runner import OrchestratorRunner

__all__ = ["OrchestratorRunner"]

