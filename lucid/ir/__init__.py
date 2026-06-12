"""Lucid IR — typed contracts for all pipeline stages."""

from lucid.ir import (
    basins,
    binding,
    common,
    context_op,
    cue,
    dmf,
    expression,
    interference,
    lucidity,
    memory,
    perception,
    pipeline,
    projector,
    serde,
    training,
)

from lucid.ir.basins import BasinInput, BasinOutput, CandidateBasinState
from lucid.ir.binding import BindingInput, BindingOutput, CandidateFrame
from lucid.ir.common import (
    AuditEnvelope,
    CommitShape,
    ComputePolicy,
    DecoderMode,
    LucidityDecision,
    Modality,
    Provenance,
    TaskIntent,
)
from lucid.ir.context_op import ContextOpInput, ContextOpOutput, ContextFrame
from lucid.ir.cue import CueCloud, CueEncoderInput
from lucid.ir.dmf import DmfInput, DmfOutput
from lucid.ir.expression import DecoderInput, DecoderOutput
from lucid.ir.interference import (
    InterferenceInput,
    InterferenceLearningPatch,
    InterferenceLearningResult,
    InterferenceOutput,
)
from lucid.ir.lucidity import (
    CommittedState,
    DecoderPolicy,
    LucidityInput,
    LucidityOutput,
    SearchDirectives,
)
from lucid.ir.memory import BasinRecord, TraceRecord
from lucid.ir.perception import PerceptionInput, PerceptualEvidenceGraph
from lucid.ir.pipeline import PipelineRun, RunContext, SessionState, StageName, StageResult
from lucid.ir.projector import ProjectorInput, ProjectorOutput
from lucid.ir.serde import from_dict, from_json, to_dict, to_json
from lucid.ir.training import Episode, FrameSlotTarget, FrameTarget, GoldLabels, RunLog, TrainingEpisode

__all__ = [
    "AuditEnvelope",
    "BasinInput",
    "BasinOutput",
    "BasinRecord",
    "BindingInput",
    "BindingOutput",
    "CandidateBasinState",
    "CandidateFrame",
    "CommitShape",
    "CommittedState",
    "ComputePolicy",
    "ContextFrame",
    "ContextOpInput",
    "ContextOpOutput",
    "CueCloud",
    "CueEncoderInput",
    "DecoderInput",
    "DecoderMode",
    "DecoderOutput",
    "DecoderPolicy",
    "DmfInput",
    "DmfOutput",
    "Episode",
    "FrameSlotTarget",
    "FrameTarget",
    "GoldLabels",
    "InterferenceInput",
    "InterferenceLearningPatch",
    "InterferenceLearningResult",
    "InterferenceOutput",
    "LucidityDecision",
    "LucidityInput",
    "LucidityOutput",
    "Modality",
    "PerceptionInput",
    "PerceptualEvidenceGraph",
    "PipelineRun",
    "ProjectorInput",
    "ProjectorOutput",
    "Provenance",
    "RunContext",
    "RunLog",
    "SearchDirectives",
    "SessionState",
    "StageName",
    "StageResult",
    "TaskIntent",
    "TraceRecord",
    "TrainingEpisode",
    "basins",
    "binding",
    "common",
    "context_op",
    "cue",
    "dmf",
    "expression",
    "from_dict",
    "from_json",
    "interference",
    "lucidity",
    "memory",
    "perception",
    "pipeline",
    "projector",
    "serde",
    "to_dict",
    "to_json",
    "training",
]
