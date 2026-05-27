"""Layer 5 — optional projector for consequence testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lucid.ir.basins import BasinOutput
from lucid.ir.binding import CandidateFrame
from lucid.ir.common import ComputePolicy
from lucid.ir.context_op import ContextFrame
from lucid.ir.lucidity import CommittedState, SearchDirectives
from lucid.ir.perception import PerceptualEvidenceGraph


@dataclass(slots=True)
class ProjectionConstraints:
    output_shape_rules: dict[str, Any] = field(default_factory=dict)
    train_pair_refs: list[str] = field(default_factory=list)
    test_input_refs: list[str] = field(default_factory=list)
    max_rollouts: int = 4


@dataclass(slots=True)
class RolloutFitScores:
    per_train_pair: dict[str, float] = field(default_factory=dict)
    aggregate_fit: float = 0.0
    unexplained_cells: int = 0
    consistency_score: float = 0.0


@dataclass(slots=True)
class ProjectorRollout:
    rollout_id: str
    assembly_id: str = ""
    target_basin_ids: list[str] = field(default_factory=list)
    implied_artifact: dict[str, Any] = field(default_factory=dict)
    fit_scores: RolloutFitScores = field(default_factory=RolloutFitScores)
    program_ref: str = ""
    failure_point: str = ""


@dataclass(slots=True)
class ProjectorInput:
    projection_request: SearchDirectives
    target_basin_ids: list[str] = field(default_factory=list)
    target_assembly_ids: list[str] = field(default_factory=list)
    candidate_frames: list[CandidateFrame] = field(default_factory=list)
    context_frames: list[ContextFrame] = field(default_factory=list)
    perceptual_evidence_graph: PerceptualEvidenceGraph | None = None
    partial_committed_state: CommittedState | None = None
    constraints: ProjectionConstraints = field(default_factory=ProjectionConstraints)
    task_intent: str = ""
    compute_policy: ComputePolicy = field(default_factory=ComputePolicy)


@dataclass(slots=True)
class ProjectorOutput:
    rollouts: list[ProjectorRollout] = field(default_factory=list)
    best_rollout_id: str = ""
    recommendation: str = ""  # suggest_commit | search_wider | preserve_ambiguity
    audit_notes: list[str] = field(default_factory=list)
