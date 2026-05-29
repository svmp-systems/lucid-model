from __future__ import annotations

from lucid.audit.logger import summarize_stage_output
from lucid.cognition.reasoning.context_op import run_context_op
from lucid.ir.binding import CandidateFrame
from lucid.ir.common import AmbiguityPolicy
from lucid.ir.context_op import ContextOpInput
from lucid.ir.dmf import ActiveTrace, ConflictSignal, DmfOutput
from lucid.ir.perception import CandidateRegion, CandidateUnit, PerceptualEvidenceGraph, ReferenceHint


def _bank_context_input(*, feedback: list[str] | None = None) -> ContextOpInput:
    graph = PerceptualEvidenceGraph(
        candidate_units=[
            CandidateUnit("u_found", "found"),
            CandidateUnit("u_money", "money"),
            CandidateUnit("u_kayaking", "kayaking"),
            CandidateUnit("u_placed", "placed"),
            CandidateUnit("u_bank", "bank"),
        ],
        reference_hints=[
            ReferenceHint(
                source_unit_id="u_placed",
                target_unit_id="u_money",
                reference_type="shared_theme",
                confidence=0.72,
            )
        ],
    )
    frames = [
        CandidateFrame(
            frame_id="event_one",
            frame_type="event",
            role_assignments={
                "ACTION": "t_found",
                "THEME": "t_money",
                "CONTEXT": "t_kayak",
            },
            member_evidence_refs=["u_found", "u_money", "u_kayaking"],
            confidence=0.76,
        ),
        CandidateFrame(
            frame_id="event_two",
            frame_type="event",
            role_assignments={
                "ACTION": "t_placed",
                "THEME": "t_money",
                "DESTINATION": "t_bank",
            },
            member_evidence_refs=["u_placed", "u_money", "u_bank"],
            confidence=0.74,
            unresolved_slot_names=["bank_sense"],
        ),
    ]
    dmf = DmfOutput(
        active_traces=[
            ActiveTrace("t_found", 0.82),
            ActiveTrace("t_money", 0.79),
            ActiveTrace("t_kayak", 0.76),
            ActiveTrace("t_placed", 0.74),
            ActiveTrace("t_bank", 0.58),
        ],
        conflict_signals=[
            ConflictSignal("t_kayak", "t_bank", severity=0.8),
        ],
        top_margin=0.04,
    )
    return ContextOpInput(
        binding_candidate_frames=frames,
        dmf_output=dmf,
        perceptual_evidence_graph=graph,
        lucidity_feedback=feedback or [],
    )


def test_context_op_scopes_shared_and_local_traces() -> None:
    out = run_context_op(_bank_context_input())

    assert [frame.context_frame_id for frame in out.context_frames] == [
        "cf_event_one",
        "cf_event_two",
    ]
    by_trace = {assignment.trace_id: assignment for assignment in out.scoped_trace_assignments}
    assert by_trace["t_kayak"].primary_context_frame_id == "cf_event_one"
    assert by_trace["t_bank"].primary_context_frame_id == "cf_event_two"
    assert by_trace["t_money"].primary_context_frame_id == "cf_event_one"
    assert "cf_event_two" in by_trace["t_money"].secondary_context_frame_ids


def test_context_op_links_frames_and_blocks_cross_scope_conflict() -> None:
    out = run_context_op(_bank_context_input())

    assert any(link.link_type == "shared_trace" for link in out.frame_links)
    assert any(link.link_type == "shared_theme" for link in out.frame_links)
    gate_two = next(
        gate for gate in out.interference_gates if gate.scope_frame_id == "cf_event_two"
    )
    assert "t_money" in gate_two.allowed_trace_ids
    assert "t_bank" in gate_two.allowed_trace_ids
    assert "t_kayak" in gate_two.blocked_trace_ids
    assert "block cross-scope" in gate_two.reason


def test_context_op_keeps_basin_pressure_soft_and_scoped() -> None:
    out = run_context_op(_bank_context_input())

    pressure_two = next(
        pressure
        for pressure in out.local_basin_pressures
        if pressure.context_frame_id == "cf_event_two"
    )
    assert "financial_destination_like" in pressure_two.basin_family_hints
    assert out.ambiguity_policy == AmbiguityPolicy.PRESERVE_PLURAL
    assert any("context_frames=2" in note for note in out.audit_notes)


def test_context_op_force_widen_feedback_updates_policy() -> None:
    out = run_context_op(_bank_context_input(feedback=["SEARCH_WIDER"]))

    assert out.ambiguity_policy == AmbiguityPolicy.FORCE_WIDEN
    assert out.compute_policy.mode == "deep_scope"
    assert out.compute_policy.retrieval_budget_multiplier >= 1.5
    assert all(frame.heat_policy == "widen" for frame in out.context_frames)


def test_context_op_region_fallback_when_binding_is_empty() -> None:
    graph = PerceptualEvidenceGraph(
        candidate_units=[CandidateUnit("u_cell", "(0,1)", kind_hint="cell")],
        candidate_regions=[
            CandidateRegion(
                region_id="legend_region",
                role_hint="legend_or_key_region",
                member_unit_ids=["u_cell"],
            )
        ],
    )
    out = run_context_op(
        ContextOpInput(
            binding_candidate_frames=[],
            dmf_output=DmfOutput(active_traces=[ActiveTrace("t_glyph", 0.7)]),
            perceptual_evidence_graph=graph,
        )
    )

    assert out.context_frames[0].context_frame_id == "cf_legend_region"
    assert out.scoped_trace_assignments[0].primary_context_frame_id == "cf_legend_region"
    assert out.local_basin_pressures[0].basin_family_hints["symbol_region_like"] > 0


def test_context_op_audit_summary_is_human_readable() -> None:
    out = run_context_op(_bank_context_input())
    summary = summarize_stage_output("context_op", out)

    assert "2 context frames" in summary["headline"]
    assert any("scoped_trace_assignments: 5" in line for line in summary["lines"])
    assert any("audit: context_frames=2" in line for line in summary["lines"])
