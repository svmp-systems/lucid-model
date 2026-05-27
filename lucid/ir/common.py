"""Layer 1 — shared enums, provenance, compute limits, audit wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Modality(str, Enum):
    TEXT = "text"
    GRID = "grid"
    IMAGE = "image"
    AUDIO = "audio"
    INTERACTIVE = "interactive"


class TaskIntent(str, Enum):
    CHAT = "chat"
    ANSWER = "answer"
    SOLVE_GRID = "solve_grid"
    ACT = "act"
    OBSERVE = "observe"
    RETRIEVE = "retrieve"


class LucidityDecision(str, Enum):
    COMMIT = "commit"
    PRESERVE_AMBIGUITY = "preserve_ambiguity"
    REQUEST_PROJECTION = "request_projection"
    SEARCH_WIDER = "search_wider"
    RECHECK_BINDING = "recheck_binding"


class CommitShape(str, Enum):
    SINGLE = "single"
    PER_FRAME = "per_frame"
    ASSEMBLY = "assembly"
    ROLLOUT_PLAN = "rollout_plan"


class DecoderMode(str, Enum):
    EXPRESS_COMMITTED = "express_committed"
    EXPRESS_PLURAL = "express_plural"
    EXPRESS_UNCERTAINTY = "express_uncertainty"
    EXPRESS_REFUSAL = "express_refusal"
    HOLD = "hold"
    EXPRESS_PARTIAL_PROGRESS = "express_partial_progress"


class AmbiguityPolicy(str, Enum):
    PRESERVE_PLURAL = "preserve_plural"
    ALLOW_NARROW = "allow_narrow"
    FORCE_WIDEN = "force_widen"


class HeatTier(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    FROZEN = "frozen"


class UncertaintySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MaturityState(str, Enum):
    SEED = "seed"
    PROVISIONAL = "provisional"
    ACTIVE = "active"
    STABILIZED = "stabilized"
    CRYSTALLIZED = "crystallized"
    FROZEN = "frozen"


class SearchTarget(str, Enum):
    BINDING = "binding"
    CUE = "cue"
    DMF = "dmf"
    BASINS = "basins"
    ALL = "all"


@dataclass(slots=True)
class Provenance:
    source_id: str = ""
    modality: Modality | None = None
    timestamp: str = ""
    adapter_version: str = ""
    segmentation_pass_id: str = ""
    session_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ComputePolicy:
    max_active_traces: int = 128
    max_candidate_frames: int = 32
    max_active_basins: int = 16
    max_projector_rollouts: int = 4
    retrieval_budget_multiplier: float = 1.0
    mode: str = "standard"  # cheap | standard | deep


@dataclass(slots=True)
class AuditEnvelope:
    run_id: str
    stage_name: str
    timestamp: str
    input_hash: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    adapter_version: str = ""
    provenance: Provenance | None = None
