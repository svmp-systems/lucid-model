"""Layer 4 — interference: local support and conflict."""

from __future__ import annotations

from dataclasses import dataclass, field

from lucid.ir.binding import CandidateFrame
from lucid.ir.common import ComputePolicy
from lucid.ir.context_op import (
    ContextFrame,
    FrameLink,
    InterferenceGate,
    LocalBasinPressure,
    ScopedTraceAssignment,
)
from lucid.ir.dmf import DmfOutput


@dataclass(slots=True)
class TraceTraceEdge:
    trace_id_a: str
    trace_id_b: str
    delta: float
    scope_frame_id: str = ""


@dataclass(slots=True)
class TraceFrameEdge:
    trace_id: str
    frame_id: str
    delta: float


@dataclass(slots=True)
class FrameBasinEdge:
    frame_id: str
    basin_id: str
    delta: float


@dataclass(slots=True)
class BasinBasinEdge:
    basin_id_a: str
    basin_id_b: str
    delta: float
    relation: str = "compete"  # compete | cooperate


@dataclass(slots=True)
class BasinEnergyDelta:
    scope_frame_id: str
    basin_id: str
    delta: float
    reason_refs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ConflictReport:
    scope_frame_id: str
    conflict_type: str
    members: list[str] = field(default_factory=list)
    severity: float = 0.0


@dataclass(slots=True)
class LearnedInterferenceLink:
    source_id: str
    target_id: str
    weight: float
    scope_hint: str = ""


@dataclass(slots=True)
class InterferenceInput:
    context_frames: list[ContextFrame]
    candidate_frames: list[CandidateFrame]
    dmf_output: DmfOutput
    interference_gates: list[InterferenceGate] = field(default_factory=list)
    scoped_trace_assignments: list[ScopedTraceAssignment] = field(default_factory=list)
    frame_links: list[FrameLink] = field(default_factory=list)
    local_basin_pressures: list[LocalBasinPressure] = field(default_factory=list)
    learned_interference_links: list[LearnedInterferenceLink] = field(default_factory=list)
    compute_policy: ComputePolicy = field(default_factory=ComputePolicy)


@dataclass(slots=True)
class InterferenceOutput:
    trace_trace_edges: list[TraceTraceEdge] = field(default_factory=list)
    trace_frame_edges: list[TraceFrameEdge] = field(default_factory=list)
    frame_basin_edges: list[FrameBasinEdge] = field(default_factory=list)
    basin_basin_edges: list[BasinBasinEdge] = field(default_factory=list)
    basin_energy_deltas: dict[str, float] = field(default_factory=dict)
    scoped_basin_energy_deltas: list[BasinEnergyDelta] = field(default_factory=list)
    cooperation_maps: dict[str, list[str]] = field(default_factory=dict)
    competition_maps: dict[str, list[str]] = field(default_factory=dict)
    conflict_reports: list[ConflictReport] = field(default_factory=list)
    audit_notes: list[str] = field(default_factory=list)
