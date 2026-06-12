"""Shared gold-label builders for grid recipes."""

from __future__ import annotations

from lucid.ir.training import (
    BasinTarget,
    FrameSlotTarget,
    FrameTarget,
    GateDirective,
    GoldSpan,
    ScopeAssignment,
    TraceTarget,
)
from lucid.training.corpus.engine import AmbiguityKnob


def cell_span(span_id: str, row: int, col: int) -> GoldSpan:
    return GoldSpan(
        span_id=span_id,
        surface=f"({row},{col})",
        kind_hint="cell",
        position=f"r{row}c{col}",
    )


def grid_cell_spans(
    before: tuple[int, int],
    after: tuple[int, int],
) -> tuple[GoldSpan, GoldSpan]:
    return cell_span("cell_in", before[0], before[1]), cell_span("cell_out", after[0], after[1])


def grid_scope_assignments(
    frame_id: str,
    *,
    same_cell: bool,
) -> list[ScopeAssignment]:
    if same_cell:
        return [
            ScopeAssignment("cell_in", frame_id),
            ScopeAssignment("cell_out", frame_id, [frame_id]),
        ]
    return [
        ScopeAssignment("cell_in", frame_id),
        ScopeAssignment("cell_out", frame_id),
    ]


def grid_interference_gates(
    frame_id: str,
    *,
    primary_trace: str,
    blocked_traces: tuple[str, ...],
) -> list[GateDirective]:
    if not blocked_traces:
        return []
    return [
        GateDirective(
            gate_id=f"isolate_{primary_trace}",
            scope_frame_id=frame_id,
            allowed_trace_ids=[primary_trace],
            blocked_trace_ids=list(blocked_traces),
        ),
    ]


def trace_weights_for_knob(
    knob: AmbiguityKnob,
    *,
    primary: str,
    decoy: str,
) -> tuple[float, float, str]:
    """Return primary weight, decoy weight, and lucidity target."""
    primary_weight = round(0.5 + knob.level * 0.4, 3)
    decoy_weight = round(max(0.05, 0.42 - knob.level * 0.37), 3)
    lucidity = "COMMIT" if primary_weight > 0.6 else "PRESERVE_AMBIGUITY"
    if lucidity == "PRESERVE_AMBIGUITY":
        primary_weight = min(primary_weight, 0.58)
        decoy_weight = min(decoy_weight, max(0.05, 0.58 - primary_weight))
    return primary_weight, decoy_weight, lucidity


def move_frame_target(
    frame_id: str,
    *,
    before: tuple[int, int],
    after: tuple[int, int],
    confidence: float,
) -> FrameTarget:
    return FrameTarget(
        frame_id=frame_id,
        frame_type="transform",
        slot_targets=[
            FrameSlotTarget(
                "slot_before",
                "position_shift_like",
                ["cell_in"],
                {"pre_change_state_like": 0.8},
                confidence,
            ),
            FrameSlotTarget(
                "slot_after",
                "position_shift_like",
                ["cell_out"],
                {"post_change_state_like": 0.8},
                confidence,
            ),
            FrameSlotTarget(
                "slot_object",
                "shape_preserved_like",
                ["cell_in", "cell_out"],
                {"object_like": 0.7},
                confidence - 0.02,
            ),
        ],
        member_span_ids=["cell_in", "cell_out"],
        confidence=confidence,
    )


def recolor_frame_target(
    frame_id: str,
    *,
    cell: tuple[int, int],
    confidence: float,
) -> FrameTarget:
    return FrameTarget(
        frame_id=frame_id,
        frame_type="transform",
        slot_targets=[
            FrameSlotTarget(
                "slot_cell",
                "recolor_like",
                ["cell_in", "cell_out"],
                {"attribute_change_like": 0.85},
                confidence,
            ),
        ],
        member_span_ids=["cell_in", "cell_out"],
        confidence=confidence,
    )


def move_trace_activations(
    move_weight: float,
    decoy_weight: float,
    *,
    ambiguous: bool,
) -> list[TraceTarget]:
    aux_cap = 0.58 if ambiguous else 0.98
    traces = [
        TraceTarget("position_shift_like", move_weight, "cell_in", True),
        TraceTarget(
            "shape_preserved_like",
            round(min(aux_cap, move_weight + (0.03 if ambiguous else 0.05)), 3),
        ),
        TraceTarget(
            "color_preserved_like",
            round(min(aux_cap, move_weight + (0.02 if ambiguous else 0.03)), 3),
        ),
    ]
    if decoy_weight > 0.08:
        traces.append(
            TraceTarget("recolor_like", decoy_weight, "cell_out", decoy_weight > 0.15)
        )
    return traces


def recolor_trace_activations(
    recolor_weight: float,
    decoy_weight: float,
    *,
    ambiguous: bool,
) -> list[TraceTarget]:
    aux_cap = 0.58 if ambiguous else 0.98
    traces = [
        TraceTarget("recolor_like", recolor_weight, "cell_in", True),
        TraceTarget(
            "position_preserved_like",
            round(min(aux_cap, recolor_weight + (0.02 if ambiguous else 0.02)), 3),
        ),
    ]
    if decoy_weight > 0.08:
        traces.append(
            TraceTarget(
                "position_shift_like",
                decoy_weight,
                "cell_out",
                decoy_weight > 0.15,
            )
        )
    return traces


def grid_basin(family_hint: str, frame_id: str, confidence: float) -> list[BasinTarget]:
    return [BasinTarget(family_hint, frame_id, confidence)]
