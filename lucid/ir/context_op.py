"""Layer 4 — context-op: scoped frames and interference gates."""

from __future__ import annotations

from dataclasses import dataclass, field

from lucid.ir.binding import CandidateFrame
from lucid.ir.common import AmbiguityPolicy, ComputePolicy
from lucid.ir.dmf import DmfOutput
from lucid.ir.perception import PerceptualEvidenceGraph


@dataclass(slots=True)
class ContextFrame:
    context_frame_id: str
    member_frame_ids: list[str] = field(default_factory=list)
    scope_notes: str = ""
    heat_policy: str = "active"


@dataclass(slots=True)
class ScopedTraceAssignment:
    trace_id: str
    primary_context_frame_id: str
    secondary_context_frame_ids: list[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass(slots=True)
class FrameLink:
    source_frame_id: str
    target_frame_id: str
    link_type: str = ""
    weight: float = 0.0


@dataclass(slots=True)
class InterferenceGate:
    gate_id: str
    scope_frame_id: str
    allowed_trace_ids: list[str] = field(default_factory=list)
    blocked_trace_ids: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass(slots=True)
class LocalBasinPressure:
    context_frame_id: str
    basin_family_hints: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ContextOpInput:
    binding_candidate_frames: list[CandidateFrame]
    dmf_output: DmfOutput
    perceptual_evidence_graph: PerceptualEvidenceGraph
    prior_context_frames: list[ContextFrame] = field(default_factory=list)
    lucidity_feedback: list[str] = field(default_factory=list)
    compute_policy: ComputePolicy = field(default_factory=ComputePolicy)


@dataclass(slots=True)
class ContextOpOutput:
    context_frames: list[ContextFrame] = field(default_factory=list)
    scoped_trace_assignments: list[ScopedTraceAssignment] = field(default_factory=list)
    frame_links: list[FrameLink] = field(default_factory=list)
    interference_gates: list[InterferenceGate] = field(default_factory=list)
    local_basin_pressures: list[LocalBasinPressure] = field(default_factory=list)
    ambiguity_policy: AmbiguityPolicy = AmbiguityPolicy.PRESERVE_PLURAL
    compute_policy: ComputePolicy = field(default_factory=ComputePolicy)
