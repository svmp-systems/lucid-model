"""Persistent memory records — traces and basins (editable stores)."""

from __future__ import annotations

from dataclasses import dataclass, field

from lucid.ir.common import HeatTier, MaturityState


@dataclass(slots=True)
class WeightedLink:
    target_id: str
    weight: float


@dataclass(slots=True)
class TransitionLink:
    trigger_trace_id: str
    next_basin_id: str
    weight: float = 0.0


@dataclass(slots=True)
class TraceRecord:
    trace_id: str
    frame_affinities: dict[str, float] = field(default_factory=dict)
    support_links: list[WeightedLink] = field(default_factory=list)
    suppression_links: list[WeightedLink] = field(default_factory=list)
    maturity: MaturityState = MaturityState.ACTIVE
    heat_tier: HeatTier = HeatTier.HOT
    success_rate: float = 0.0
    failure_modes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BasinRecord:
    basin_id: str
    frame_affinities: dict[str, float] = field(default_factory=dict)
    support_links: list[WeightedLink] = field(default_factory=list)
    suppression_links: list[WeightedLink] = field(default_factory=list)
    cooperation_links: list[WeightedLink] = field(default_factory=list)
    transition_links: list[TransitionLink] = field(default_factory=list)
    maturity: MaturityState = MaturityState.ACTIVE
    heat_tier: HeatTier = HeatTier.HOT
    success_rate: float = 0.0
    failure_modes: list[str] = field(default_factory=list)
