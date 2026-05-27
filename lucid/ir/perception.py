"""Layer 2 — perception input and evidence graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lucid.ir.common import ComputePolicy, Modality, Provenance, TaskIntent, UncertaintySeverity


@dataclass(slots=True)
class PerceptionInput:
    raw_payload: str | dict[str, Any]
    modality: Modality
    task_intent_hint: TaskIntent | None = None
    prior_context: dict[str, Any] = field(default_factory=dict)
    compute_policy: ComputePolicy = field(default_factory=ComputePolicy)
    provenance_seed: str = ""


@dataclass(slots=True)
class CandidateUnit:
    unit_id: str
    surface: str = ""
    kind_hint: str = ""
    type_hints: list[str] = field(default_factory=list)
    feature_signature: str = ""
    position_or_time: str = ""
    confidence: float = 0.0
    salience: float = 0.0
    uncertainty: str | None = None


@dataclass(slots=True)
class CandidateRegion:
    region_id: str
    role_hint: str = ""
    member_unit_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    uncertainty: str | None = None


@dataclass(slots=True)
class CandidateContainer:
    container_id: str
    kind_hint: str = ""
    border_signature: str = ""
    interior_region_id: str = ""
    confidence: float = 0.0


@dataclass(slots=True)
class CandidateMarker:
    marker_id: str
    surface: str = ""
    marker_type_hints: list[str] = field(default_factory=list)
    possible_target_unit_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(slots=True)
class ArrangementHint:
    hint_type: str
    source_unit_id: str
    target_unit_id: str
    weight: float = 0.0


@dataclass(slots=True)
class ChangeHint:
    change_type: str
    before_unit_id: str = ""
    after_unit_id: str = ""
    weight: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GroupingHint:
    group_id: str
    member_unit_ids: list[str] = field(default_factory=list)
    grouping_reason: str = ""
    confidence: float = 0.0


@dataclass(slots=True)
class ReferenceHint:
    source_unit_id: str
    target_unit_id: str
    reference_type: str = ""
    confidence: float = 0.0


@dataclass(slots=True)
class UncertaintyFlag:
    target_id: str
    uncertainty_type: str
    severity: UncertaintySeverity = UncertaintySeverity.MEDIUM


@dataclass(slots=True)
class PerceptualEvidenceGraph:
    candidate_units: list[CandidateUnit] = field(default_factory=list)
    candidate_regions: list[CandidateRegion] = field(default_factory=list)
    candidate_containers: list[CandidateContainer] = field(default_factory=list)
    candidate_markers: list[CandidateMarker] = field(default_factory=list)
    arrangement_hints: list[ArrangementHint] = field(default_factory=list)
    change_hints: list[ChangeHint] = field(default_factory=list)
    grouping_hints: list[GroupingHint] = field(default_factory=list)
    reference_hints: list[ReferenceHint] = field(default_factory=list)
    uncertainty_flags: list[UncertaintyFlag] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)
