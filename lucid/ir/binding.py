"""Layer 4 — binding: candidate frames and competition."""

from __future__ import annotations

from dataclasses import dataclass, field

from lucid.ir.common import ComputePolicy
from lucid.ir.cue import CueCloud
from lucid.ir.dmf import DmfOutput
from lucid.ir.perception import PerceptualEvidenceGraph


@dataclass(slots=True)
class CandidateFrame:
    frame_id: str
    frame_type: str  # word_sense | event | relation | rule | transform | ...
    # Anonymous slot_id -> trace_id bindings. Human role names belong in audit hints,
    # not in the binding contract.
    role_assignments: dict[str, str] = field(default_factory=dict)
    relation_assignments: dict[str, str] = field(default_factory=dict)
    slot_evidence_refs: dict[str, list[str]] = field(default_factory=dict)
    slot_affinity_hints: dict[str, dict[str, float]] = field(default_factory=dict)
    member_evidence_refs: list[str] = field(default_factory=list)
    confidence: float = 0.0
    unresolved_slot_names: list[str] = field(default_factory=list)
    supporting_trace_ids: list[str] = field(default_factory=list)
    conflicting_trace_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FrameCompetitionEdge:
    frame_id_a: str
    frame_id_b: str
    relation: str = "compete"  # compete | cooperate
    weight: float = 0.0


@dataclass(slots=True)
class BindingInput:
    dmf_output: DmfOutput
    perceptual_evidence_graph: PerceptualEvidenceGraph
    cue_cloud: CueCloud | None = None
    affordance_hints: dict[str, float] = field(default_factory=dict)
    prior_candidate_frames: list[CandidateFrame] = field(default_factory=list)
    compute_policy: ComputePolicy = field(default_factory=ComputePolicy)


@dataclass(slots=True)
class BindingOutput:
    candidate_frames: list[CandidateFrame] = field(default_factory=list)
    frame_competition_edges: list[FrameCompetitionEdge] = field(default_factory=list)
    binding_stability_score: float = 0.0
    audit_notes: list[str] = field(default_factory=list)
