"""Layer 4 — interference: local support and conflict."""

from __future__ import annotations

from dataclasses import dataclass, field

from lucid.ir.binding import CandidateFrame
from lucid.ir.common import ComputePolicy
from lucid.ir.context_op import ContextFrame, InterferenceGate
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
class InterferenceConflictReport:
    report_id: str
    conflict_type: str
    severity: float
    scope_frame_id: str = ""
    trace_ids: list[str] = field(default_factory=list)
    basin_ids: list[str] = field(default_factory=list)
    source_edge_ids: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass(slots=True)
class InterferenceInput:
    context_frames: list[ContextFrame]
    candidate_frames: list[CandidateFrame]
    dmf_output: DmfOutput
    interference_gates: list[InterferenceGate] = field(default_factory=list)
    compute_policy: ComputePolicy = field(default_factory=ComputePolicy)


@dataclass(slots=True)
class InterferenceOutput:
    trace_trace_edges: list[TraceTraceEdge] = field(default_factory=list)
    trace_frame_edges: list[TraceFrameEdge] = field(default_factory=list)
    frame_basin_edges: list[FrameBasinEdge] = field(default_factory=list)
    basin_basin_edges: list[BasinBasinEdge] = field(default_factory=list)
    conflict_reports: list[InterferenceConflictReport] = field(default_factory=list)
    basin_energy_deltas: dict[str, float] = field(default_factory=dict)
    cooperation_maps: dict[str, list[str]] = field(default_factory=dict)
    competition_maps: dict[str, list[str]] = field(default_factory=dict)
