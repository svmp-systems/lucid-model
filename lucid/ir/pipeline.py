"""Layer 7 — run context, session state, stage protocol, full pipeline run."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from lucid.ir.basins import BasinInput, BasinOutput
from lucid.ir.binding import BindingInput, BindingOutput
from lucid.ir.common import AuditEnvelope, ComputePolicy, Provenance, TaskIntent
from lucid.ir.context_op import ContextOpInput, ContextOpOutput
from lucid.ir.cue import CueEncoderInput, CueCloud
from lucid.ir.dmf import DmfInput, DmfOutput
from lucid.ir.expression import DecoderInput, DecoderOutput
from lucid.ir.interference import InterferenceInput, InterferenceOutput
from lucid.ir.lucidity import LucidityInput, LucidityOutput
from lucid.ir.perception import PerceptionInput, PerceptualEvidenceGraph
from lucid.ir.projector import ProjectorInput, ProjectorOutput
from lucid.ir.training import CostMetrics, RunLog, TrainingEpisode


class StageName(str, Enum):
    PERCEPTION = "perception"
    CUE_ENCODER = "cue_encoder"
    DMF = "dmf"
    BINDING = "binding"
    CONTEXT_OP = "context_op"
    INTERFERENCE = "interference"
    BASINS = "basins"
    LUCIDITY = "lucidity"
    PROJECTOR = "projector"
    DECODER = "decoder"


@dataclass(slots=True)
class TurnRecord:
    turn_index: int
    user_input: str | dict[str, Any]
    run_id: str = ""
    lucidity_decision: str = ""
    decoder_surface: str = ""


@dataclass(slots=True)
class SessionState:
    session_id: str
    turns: list[TurnRecord] = field(default_factory=list)
    carryover_evidence_refs: list[str] = field(default_factory=list)
    carryover_trace_ids: list[str] = field(default_factory=list)
    carryover_frame_ids: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)


@dataclass(slots=True)
class RunContext:
    run_id: str
    session_id: str = ""
    turn_index: int = 0
    mode: str = "inference"  # inference | training_observation | shadow
    task_intent: TaskIntent = TaskIntent.ANSWER
    compute_policy: ComputePolicy = field(default_factory=ComputePolicy)
    session_state: SessionState | None = None
    episode: TrainingEpisode | None = None
    audit_dir: str = ""
    iteration_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StageResult:
    stage_name: StageName | str
    success: bool
    duration_ms: float = 0.0
    output_type: str = ""
    audit: AuditEnvelope | None = None
    error_message: str = ""


@runtime_checkable
class PipelineStage(Protocol):
    """Each stage implements run(input, context) -> output."""

    stage_name: StageName

    def run(self, stage_input: Any, context: RunContext) -> Any: ...


@dataclass(slots=True)
class PipelineRun:
    context: RunContext
    perception_input: PerceptionInput | None = None
    evidence_graph: PerceptualEvidenceGraph | None = None
    cue_encoder_input: CueEncoderInput | None = None
    cue_cloud: CueCloud | None = None
    dmf_input: DmfInput | None = None
    dmf_output: DmfOutput | None = None
    binding_input: BindingInput | None = None
    binding_output: BindingOutput | None = None
    context_op_input: ContextOpInput | None = None
    context_op_output: ContextOpOutput | None = None
    interference_input: InterferenceInput | None = None
    interference_output: InterferenceOutput | None = None
    basin_input: BasinInput | None = None
    basin_output: BasinOutput | None = None
    lucidity_input: LucidityInput | None = None
    lucidity_output: LucidityOutput | None = None
    projector_input: ProjectorInput | None = None
    projector_output: ProjectorOutput | None = None
    decoder_input: DecoderInput | None = None
    decoder_output: DecoderOutput | None = None
    stage_results: list[StageResult] = field(default_factory=list)
    cost_metrics: CostMetrics = field(default_factory=CostMetrics)

    def to_run_log(self) -> RunLog:
        episode_id = self.context.episode.episode_id if self.context.episode else ""
        lucidity_decision = ""
        if self.lucidity_output:
            lucidity_decision = self.lucidity_output.decision.value
        return RunLog(
            episode_id=episode_id,
            run_id=self.context.run_id,
            evidence_graph=self.evidence_graph,
            cue_cloud=self.cue_cloud,
            dmf_output=self.dmf_output,
            binding_output=self.binding_output,
            context_op_output=self.context_op_output,
            interference_output=self.interference_output,
            basin_output=self.basin_output,
            lucidity_output=self.lucidity_output,
            projection_output=self.projector_output,
            decoder_output=self.decoder_output,
            cost_metrics=self.cost_metrics,
            lucidity_features={"decision": lucidity_decision} if lucidity_decision else {},
        )
