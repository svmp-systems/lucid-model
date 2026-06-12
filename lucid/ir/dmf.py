"""Layer 3 dynamic memory field input/output contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lucid.ir.common import ComputePolicy, Provenance
from lucid.ir.cue import CueCloud


@dataclass(slots=True)
class ActiveTrace:
    trace_id: str = ""
    activation: float = 0.0
    cluster_id: str = ""
    heat_tier: str = "hot"
    is_novel: bool = False


@dataclass(slots=True)
class TraceCluster:
    cluster_id: str
    member_trace_ids: list[str] = field(default_factory=list)
    cluster_strength: float = 0.0
    cluster_coherence: float = 0.0


@dataclass(slots=True)
class NoveltySignal:
    region_or_evidence_ref: str
    novelty_score: float
    suggested_action: str = "widen_search"


@dataclass(slots=True)
class ConflictSignal:
    trace_id_a: str
    trace_id_b: str
    severity: float = 0.0
    scope_frame_id: str = ""


@dataclass(slots=True)
class DmfInput:
    cue_cloud: CueCloud
    tracebank_snapshot_id: str = ""
    heat_policy: str = "standard"
    quarantine_filter: bool = True
    prior_active_trace_ids: list[str] = field(default_factory=list)
    compute_policy: ComputePolicy = field(default_factory=ComputePolicy)


@dataclass(slots=True)
class DmfOutput:
    active_traces: list[ActiveTrace] = field(default_factory=list)
    trace_clusters: list[TraceCluster] = field(default_factory=list)
    novelty_signals: list[NoveltySignal] = field(default_factory=list)
    conflict_signals: list[ConflictSignal] = field(default_factory=list)
    top_margin: float = 0.0
    second_margin: float = 0.0
    activation_entropy: float = 0.0
    coverage_score: float = 0.0
    adjusted_activations: dict[str, float] = field(default_factory=dict)
    activation_reasons: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    uncertainty_summary: str = ""
    tracebank_snapshot_id: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    audit_log: dict[str, str | int | float] = field(default_factory=dict)
