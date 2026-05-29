"""Layer 6 — episodes, gold labels, run logs, patches, failure replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lucid.ir.basins import BasinOutput
from lucid.ir.binding import BindingOutput
from lucid.ir.common import Modality, TaskIntent
from lucid.ir.context_op import ContextOpOutput
from lucid.ir.cue import CueCloud
from lucid.ir.dmf import DmfOutput
from lucid.ir.expression import DecoderOutput
from lucid.ir.interference import InterferenceOutput
from lucid.ir.lucidity import LucidityOutput
from lucid.ir.perception import PerceptualEvidenceGraph, UncertaintyFlag
from lucid.ir.projector import ProjectorOutput


# --- Gold label building blocks (generator targets) ---


@dataclass(slots=True)
class GoldSpan:
    span_id: str
    surface: str = ""
    kind_hint: str = ""
    position: str = ""


@dataclass(slots=True)
class GoldMarker:
    marker_id: str
    surface: str = ""
    marker_type_hints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GoldRegion:
    region_id: str
    role_hint: str = ""
    member_span_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TraceTarget:
    trace_family: str
    weight: float
    evidence_ref: str = ""
    keep_alive: bool = True


@dataclass(slots=True)
class ScopeAssignment:
    span_id: str
    primary_frame: str
    secondary_frames: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GateDirective:
    gate_id: str
    scope_frame_id: str
    allowed_trace_ids: list[str] = field(default_factory=list)
    blocked_trace_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BasinTarget:
    family_hint: str
    frame_id: str = ""
    confidence: float = 0.0


@dataclass(slots=True)
class GoldLabels:
    spans: list[GoldSpan] = field(default_factory=list)
    markers: list[GoldMarker] = field(default_factory=list)
    regions: list[GoldRegion] = field(default_factory=list)
    uncertainty_flags: list[UncertaintyFlag] = field(default_factory=list)
    trace_activations: list[TraceTarget] = field(default_factory=list)
    ambiguity_policy: str = "preserve_plural"
    scope_assignments: list[ScopeAssignment] = field(default_factory=list)
    interference_gates: list[GateDirective] = field(default_factory=list)
    basin_families: list[BasinTarget] = field(default_factory=list)
    lucidity_target: str = ""
    lucidity_rationale: str = ""
    expected_answer: str | dict[str, Any] | None = None
    validator_result: bool | None = None


@dataclass(slots=True)
class Episode:
    episode_id: str
    modality: Modality | str
    template_id: str = ""
    raw_input: str | dict[str, Any] = ""
    gold: GoldLabels = field(default_factory=GoldLabels)
    validator: str = ""
    seed: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    task_intent: TaskIntent | str = TaskIntent.ANSWER
    context: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingEpisode:
    episode_id: str
    raw_input: str | dict[str, Any]
    modality: Modality | str
    task_intent: TaskIntent | str = TaskIntent.ANSWER
    context: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    expected_output: str | dict[str, Any] | None = None
    validator: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CostMetrics:
    wall_time_ms: float = 0.0
    stage_times_ms: dict[str, float] = field(default_factory=dict)
    active_trace_count: int = 0
    candidate_frame_count: int = 0
    active_basin_count: int = 0
    projector_rollout_count: int = 0


@dataclass(slots=True)
class ValidationResult:
    success: bool
    failure_type: str = ""
    score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunLog:
    episode_id: str
    run_id: str = ""
    evidence_graph: PerceptualEvidenceGraph | None = None
    cue_cloud: CueCloud | None = None
    dmf_output: DmfOutput | None = None
    binding_output: BindingOutput | None = None
    context_op_output: ContextOpOutput | None = None
    interference_output: InterferenceOutput | None = None
    basin_output: BasinOutput | None = None
    lucidity_output: LucidityOutput | None = None
    projection_output: ProjectorOutput | None = None
    decoder_output: DecoderOutput | None = None
    validator_result: ValidationResult | None = None
    cost_metrics: CostMetrics = field(default_factory=CostMetrics)
    lucidity_features: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class FailureDiagnosis:
    primary_module: str
    failure_type: str
    confidence: float = 0.0
    responsible_objects: list[str] = field(default_factory=list)
    suggested_update_level: str = "local"


@dataclass(slots=True)
class ShadowTestBundle:
    episode_ids: list[str] = field(default_factory=list)
    retention_suite_version: str = ""


@dataclass(slots=True)
class UpdateProposal:
    patch_type: str
    target_objects: list[str] = field(default_factory=list)
    update_level: str = "local"
    expected_fix: str = ""
    risk_level: str = "low"
    shadow_test_bundle: ShadowTestBundle = field(default_factory=ShadowTestBundle)


@dataclass(slots=True)
class PatchResult:
    patch_id: str
    fixed_target: str = ""
    retention_passed: bool = False
    cost_delta: float = 0.0
    quality_delta: float = 0.0
    promoted: bool = False
    episode_shadow_passed: bool = False
    retention_suite_version: str = ""
    notes: str = ""


@dataclass(slots=True)
class FailureReplayEntry:
    episode_id: str
    run_log_id_last_failure: str = ""
    patch_ids_applied: list[str] = field(default_factory=list)
    shadow_passed: bool = False
    consecutive_successes: int = 0
    entered_at: str = ""
    last_attempt_at: str = ""


@dataclass(slots=True)
class RetentionSuiteSnapshot:
    suite_version: str
    canary_count: int = 0
    family_coverage: dict[str, int] = field(default_factory=dict)
    episode_ids: list[str] = field(default_factory=list)
    last_audit_at: str = ""
