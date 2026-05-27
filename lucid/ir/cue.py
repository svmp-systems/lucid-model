"""Layer 3 — cue encoder input and cue cloud."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lucid.ir.common import AmbiguityPolicy, ComputePolicy, Provenance
from lucid.ir.perception import PerceptualEvidenceGraph


@dataclass(slots=True)
class TraceActivationRequest:
    trace_id: str
    weight: float
    evidence_refs: list[str] = field(default_factory=list)
    keep_alive: bool = True


@dataclass(slots=True)
class RelationalActivationRequest:
    trace_id: str
    weight: float
    relation_refs: list[str] = field(default_factory=list)
    endpoint_unit_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CueEncoderInput:
    perceptual_evidence_graph: PerceptualEvidenceGraph
    upstream_state: dict[str, Any] = field(default_factory=dict)
    task_intent_hint: str = ""
    retrieval_budget: int = 128
    ambiguity_policy_in: AmbiguityPolicy = AmbiguityPolicy.PRESERVE_PLURAL
    compute_policy: ComputePolicy = field(default_factory=ComputePolicy)
    provenance: Provenance = field(default_factory=Provenance)


@dataclass(slots=True)
class CueCloud:
    primitive_trace_activations: list[TraceActivationRequest] = field(default_factory=list)
    relational_trace_activations: list[RelationalActivationRequest] = field(default_factory=list)
    soft_context_priors: dict[str, float] = field(default_factory=dict)
    weak_structure_hints: list[str] = field(default_factory=list)
    ambiguity_policy: AmbiguityPolicy = AmbiguityPolicy.PRESERVE_PLURAL
    retrieval_budget_used: int = 0
    suppression_list: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)


@dataclass(slots=True)
class CuePacket:
    cloud_id: str
    top_k_trace_ids: list[str] = field(default_factory=list)
    activation_weights: dict[str, float] = field(default_factory=dict)
    policy_flags: list[str] = field(default_factory=list)
