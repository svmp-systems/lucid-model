"""Layer 5 — lucidity gate, committed state, checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lucid.ir.basins import BasinOutput
from lucid.ir.binding import BindingOutput
from lucid.ir.common import (
    CommitShape,
    ComputePolicy,
    LucidityDecision,
    Provenance,
    SearchTarget,
)
from lucid.ir.context_op import ContextOpOutput
from lucid.ir.dmf import DmfOutput
from lucid.ir.interference import InterferenceOutput
from lucid.ir.perception import PerceptualEvidenceGraph
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid.ir.projector import ProjectorOutput


@dataclass(slots=True)
class CheckResult:
    passed: bool
    score: float
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LucidityCheckResults:
    margin_check: CheckResult | None = None
    coverage_check: CheckResult | None = None
    coherence_check: CheckResult | None = None
    binding_stability_check: CheckResult | None = None
    scope_check: CheckResult | None = None
    projection_fit_check: CheckResult | None = None
    contradiction_check: CheckResult | None = None
    maturity_check: CheckResult | None = None
    risk_check: CheckResult | None = None


@dataclass(slots=True)
class ConfidenceSummary:
    overall_confidence: float = 0.0
    margin: float = 0.0
    coverage: float = 0.0
    coherence: float = 0.0
    projection_fit: float | None = None


@dataclass(slots=True)
class FrameCommit:
    context_frame_id: str
    frame_type: str
    basin_id: str
    role_map: dict[str, str] = field(default_factory=dict)
    scope_notes: str = ""


@dataclass(slots=True)
class StructuredClaim:
    claim_type: str
    subject_ref: str
    predicate_ref: str
    confidence: float = 0.0
    scope_frame_id: str = ""


@dataclass(slots=True)
class RolloutStep:
    step_index: int
    action_ref: str
    predicted_state_basin_id: str
    fit_score: float = 0.0


@dataclass(slots=True)
class CommittedState:
    commit_id: str
    commit_shape: CommitShape = CommitShape.SINGLE
    primary_basin_id: str = ""
    assembly_ids: list[str] = field(default_factory=list)
    member_basin_ids: list[str] = field(default_factory=list)
    frame_commits: list[FrameCommit] = field(default_factory=list)
    rollout_steps: list[RolloutStep] = field(default_factory=list)
    claims: list[StructuredClaim] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    projection_artifact: dict[str, Any] = field(default_factory=dict)
    provenance_chain: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PreservedHypothesis:
    hypothesis_id: str
    frame_id: str = ""
    basin_id: str = ""
    narrative_hint: str = ""
    confidence: float = 0.0


@dataclass(slots=True)
class SearchDirectives:
    search_target: SearchTarget = SearchTarget.ALL
    cue_budget_multiplier: float = 1.0
    allow_new_frames: bool = False
    projector_targets: list[str] = field(default_factory=list)
    max_rollouts: int = 0
    rollout_mode: str = "none"  # none | single_step | multi_step
    rollout_depth: int = 0
    rebind_frame_ids: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DecoderPolicy:
    mode: str  # DecoderMode value
    forbid_single_answer: bool = False
    forbid_invented_facts: bool = True
    require_cite_traces: bool = False
    max_detail_level: str = "medium"
    output_format: str = "text"  # text | grid | action | plan
    show_alternatives: bool = False
    show_confidence: bool = False
    show_scope: bool = False
    refusal_reason: str = ""


@dataclass(slots=True)
class LucidityInput:
    basin_output: BasinOutput
    binding_output: BindingOutput
    context_op_output: ContextOpOutput
    interference_output: InterferenceOutput
    dmf_output: DmfOutput
    perceptual_evidence_graph: PerceptualEvidenceGraph
    task_intent: str = "answer"
    risk_level: str = "medium"
    stakes_policy: str = "standard"
    projection_output: ProjectorOutput | None = None  # type: ignore[name-defined]
    pass_kind: str = "pre_check"  # pre_check | final_check | recheck
    iteration_count: int = 0
    prior_decisions: list[LucidityDecision] = field(default_factory=list)
    compute_policy: ComputePolicy = field(default_factory=ComputePolicy)


@dataclass(slots=True)
class LucidityOutput:
    decision: LucidityDecision
    decoder_policy: DecoderPolicy
    check_results: LucidityCheckResults = field(default_factory=LucidityCheckResults)
    confidence_summary: ConfidenceSummary = field(default_factory=ConfidenceSummary)
    committed_state: CommittedState | None = None
    preserved_hypotheses: list[PreservedHypothesis] = field(default_factory=list)
    search_directives: SearchDirectives | None = None
    secondary_decisions: list[LucidityDecision] = field(default_factory=list)
    audit_notes: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)
