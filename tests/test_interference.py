from __future__ import annotations

from lucid.audit.logger import summarize_stage_output
from lucid.cli import main as lucid_cli
from lucid.cognition.reasoning.context_op import run_context_op
from lucid.cognition.reasoning.interference import run_interference
from lucid.ir.binding import CandidateFrame
from lucid.ir.context_op import ContextOpInput
from lucid.ir.dmf import ActiveTrace, ConflictSignal, DmfOutput
from lucid.ir.interference import InterferenceInput, LearnedInterferenceLink
from lucid.ir.perception import CandidateUnit, PerceptualEvidenceGraph, ReferenceHint


def _bank_interference_input() -> InterferenceInput:
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
    context_out = run_context_op(
        ContextOpInput(
            binding_candidate_frames=frames,
            dmf_output=dmf,
            perceptual_evidence_graph=graph,
        )
    )
    return InterferenceInput(
        context_frames=context_out.context_frames,
        candidate_frames=frames,
        dmf_output=dmf,
        interference_gates=context_out.interference_gates,
        scoped_trace_assignments=context_out.scoped_trace_assignments,
        frame_links=context_out.frame_links,
        local_basin_pressures=context_out.local_basin_pressures,
    )


def test_interference_honors_closed_gates_and_stays_scoped() -> None:
    out = run_interference(_bank_interference_input())

    assert not any(
        edge.scope_frame_id == "cf_event_two"
        and {edge.trace_id_a, edge.trace_id_b} == {"t_kayak", "t_bank"}
        for edge in out.trace_trace_edges
    )
    assert all("::" in key for key in out.basin_energy_deltas)
    assert any(delta.scope_frame_id == "cf_event_two" for delta in out.scoped_basin_energy_deltas)
    assert any("gates_honored=2" in note for note in out.audit_notes)


def test_interference_builds_local_support_and_conflict_pressure() -> None:
    inp = _bank_interference_input()
    inp.learned_interference_links.append(
        LearnedInterferenceLink("t_money", "t_bank", 0.6, scope_hint="cf_event_two")
    )

    out = run_interference(inp)

    assert any(edge.delta > 0 for edge in out.trace_frame_edges)
    assert any(edge.delta > 0 and edge.scope_frame_id == "cf_event_two" for edge in out.trace_trace_edges)
    assert any(edge.basin_id == "b_event_frame" for edge in out.frame_basin_edges)
    assert any(
        report.scope_frame_id == "cf_event_two"
        and report.conflict_type == "basin_family_competition"
        for report in out.conflict_reports
    )


def test_interference_audit_summary_is_human_readable() -> None:
    out = run_interference(_bank_interference_input())
    summary = summarize_stage_output("interference", out)

    assert "scoped basin deltas" in summary["headline"]
    assert any("conflict_reports:" in line for line in summary["lines"])
    assert any("audit: frames_processed=2" in line for line in summary["lines"])


def test_cli_runs_interference_component(capsys) -> None:
    exit_code = lucid_cli(["interference", "--fixture", "bank"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "scoped_basin_energy_deltas" in captured.out
    assert "cf_event_two" in captured.out
    assert "audit_notes" in captured.out
