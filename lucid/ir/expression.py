"""Layer 5 — decoder: express lucidity-approved state only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lucid.ir.lucidity import (
    CommittedState,
    DecoderPolicy,
    LucidityOutput,
    LucidityRenderPacket,
    SourceRef,
)


@dataclass(slots=True)
class SentenceRef:
    sentence_id: str
    source_refs: list[SourceRef] = field(default_factory=list)
    unit_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FaithfulnessReport:
    passed: bool = True
    unsupported_sentence_count: int = 0
    omitted_required_units: list[str] = field(default_factory=list)
    policy_violations: list[str] = field(default_factory=list)
    reparse_match_score: float = 1.0


@dataclass(slots=True)
class DecoderInput:
    lucidity_output: LucidityOutput
    render_packet: LucidityRenderPacket | None = None
    committed_state: CommittedState | None = None
    decoder_policy: DecoderPolicy | None = None
    user_facing_context: str = ""
    output_channel: str = ""


@dataclass(slots=True)
class DecoderOutput:
    surface_text: str = ""
    surface_grid: list[list[int]] | None = None
    surface_action: dict[str, Any] | None = None
    structured_payload: dict[str, Any] | None = None
    render_mode: str = ""
    cited_refs: list[SourceRef] = field(default_factory=list)
    sentence_refs: list[SentenceRef] = field(default_factory=list)
    cited_trace_ids: list[str] = field(default_factory=list)
    uncertainty_presentation: dict[str, Any] = field(default_factory=dict)
    refused: bool = False
    refusal_reason: str = ""
    faithfulness_report: FaithfulnessReport = field(default_factory=FaithfulnessReport)
    audit_notes: list[str] = field(default_factory=list)
