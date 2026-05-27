"""Layer 4 — basins: hypothesis attractors."""

from __future__ import annotations

from dataclasses import dataclass, field

from lucid.ir.binding import CandidateFrame
from lucid.ir.common import ComputePolicy
from lucid.ir.context_op import ContextFrame, LocalBasinPressure
from lucid.ir.interference import InterferenceOutput


@dataclass(slots=True)
class CandidateBasinState:
    basin_id: str
    energy: float = 0.0
    assembly_id: str = ""
    member_basin_ids: list[str] = field(default_factory=list)
    supporting_trace_ids: list[str] = field(default_factory=list)
    supporting_frame_ids: list[str] = field(default_factory=list)
    scope_frame_ids: list[str] = field(default_factory=list)
    margin_vs_next: float = 0.0
    coherence_score: float = 0.0
    heat_tier: str = "hot"


@dataclass(slots=True)
class BasinAssembly:
    assembly_id: str
    member_basin_ids: list[str] = field(default_factory=list)
    combined_energy: float = 0.0
    assembly_coherence: float = 0.0
    scope_frame_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BasinConflict:
    scope_frame_id: str
    conflict_type: str
    basin_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CompetitionSummary:
    top_basin_id: str = ""
    second_basin_id: str = ""
    top_margin: float = 0.0
    active_basin_count: int = 0


@dataclass(slots=True)
class BasinInput:
    interference_output: InterferenceOutput
    candidate_frames: list[CandidateFrame]
    context_frames: list[ContextFrame]
    local_basin_pressures: list[LocalBasinPressure] = field(default_factory=list)
    basin_field_snapshot_id: str = ""
    heat_policy: str = "standard"
    prior_basin_state: list[CandidateBasinState] = field(default_factory=list)
    compute_policy: ComputePolicy = field(default_factory=ComputePolicy)


@dataclass(slots=True)
class BasinOutput:
    candidate_basin_states: list[CandidateBasinState] = field(default_factory=list)
    basin_assemblies: list[BasinAssembly] = field(default_factory=list)
    competition_summary: CompetitionSummary = field(default_factory=CompetitionSummary)
    unresolved_conflicts: list[BasinConflict] = field(default_factory=list)
    binding_stability_hint: float = 0.0
    audit_notes: list[str] = field(default_factory=list)
