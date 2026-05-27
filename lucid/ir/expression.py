"""Layer 5 — decoder: express lucidity-approved state only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lucid.ir.lucidity import CommittedState, DecoderPolicy, LucidityOutput


@dataclass(slots=True)
class DecoderInput:
    lucidity_output: LucidityOutput
    committed_state: CommittedState | None = None
    decoder_policy: DecoderPolicy | None = None
    user_facing_context: str = ""


@dataclass(slots=True)
class DecoderOutput:
    surface_text: str = ""
    surface_grid: list[list[int]] | None = None
    surface_action: dict[str, Any] | None = None
    cited_trace_ids: list[str] = field(default_factory=list)
    uncertainty_presentation: dict[str, Any] = field(default_factory=dict)
    refused: bool = False
    refusal_reason: str = ""
