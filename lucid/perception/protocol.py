"""Perception adapter protocol."""

from __future__ import annotations

from typing import Any, Protocol

from lucid.ir.perception import PerceptionInput, PerceptualEvidenceGraph


class PerceptionAdapter(Protocol):
    adapter_id: str

    def perceive(self, inp: PerceptionInput, *, context: Any = None) -> PerceptualEvidenceGraph: ...
