from __future__ import annotations

import json
from pathlib import Path

from lucid.audit.logger import summarize_stage_output
from lucid.cli import main as lucid_cli
from lucid.cognition.reasoning.interference import (
    learn_interference,
    load_learned_interference_links,
)
from lucid.cognition.reasoning.context_op import run_context_op
from lucid.cognition.reasoning.interference import run_interference
from lucid.ir.binding import CandidateFrame
from lucid.ir.context_op import ContextFrame, ContextOpInput, InterferenceGate
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
    assert any(delta.scope_frame_id == "cf_event_two" for delta in out.scoped_basin_energy_deltas)


def test_interference_does_not_turn_supporting_trace_into_frame_conflict() -> None:
    frame = CandidateFrame(
        frame_id="local_quantum_computing",
        frame_type="local_reading",
        role_assignments={"slot_00": "t_term_quantum_computing"},
        supporting_trace_ids=["t_term_quantum_computing"],
        conflicting_trace_ids=["t_term_quantum_computing", "t_claim_quantum_circuit"],
        confidence=0.72,
    )
    out = run_interference(
        InterferenceInput(
            context_frames=[
                ContextFrame(
                    context_frame_id="cf_quantum_computing",
                    member_frame_ids=["local_quantum_computing"],
                )
            ],
            candidate_frames=[frame],
            dmf_output=DmfOutput(
                active_traces=[
                    ActiveTrace("t_term_quantum_computing", 0.69),
                    ActiveTrace("t_claim_quantum_circuit", 0.52),
                ]
            ),
            interference_gates=[
                InterferenceGate(
                    gate_id="gate_quantum_computing",
                    scope_frame_id="cf_quantum_computing",
                    allowed_trace_ids=["t_term_quantum_computing"],
                )
            ],
        )
    )

    assert any(
        edge.trace_id == "t_term_quantum_computing"
        and edge.frame_id == "local_quantum_computing"
        and edge.delta > 0
        for edge in out.trace_frame_edges
    )
    assert not any(
        report.conflict_type == "trace_frame_conflict"
        and "t_term_quantum_computing" in report.members
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


def test_interference_learning_persists_and_reloads_scoped_links(tmp_path: Path) -> None:
    inp = _bank_interference_input()
    out = run_interference(inp)
    store_path = tmp_path / "interference_links.json"
    audit_dir = tmp_path / "audit"

    result = learn_interference(
        inp,
        out,
        validation_success=True,
        store_path=store_path,
        audit_dir=audit_dir,
    )
    learned = load_learned_interference_links(store_path)

    assert result.patches
    assert store_path.exists()
    assert Path(result.audit_path, "interference_learning.json").exists()
    assert any(
        link.scope_hint == "cf_event_two"
        and {link.source_id, link.target_id} == {"t_money", "t_bank"}
        and link.weight > 0
        for link in learned
    )

    inp.learned_interference_links = learned
    learned_out = run_interference(inp)
    assert any(
        edge.scope_frame_id == "cf_event_two"
        and {edge.trace_id_a, edge.trace_id_b} == {"t_money", "t_bank"}
        and edge.delta > 0
        for edge in learned_out.trace_trace_edges
    )


def test_cli_runs_interference_learning_smoke(tmp_path: Path, capsys) -> None:
    store_path = tmp_path / "store.json"
    audit_dir = tmp_path / "audit"

    exit_code = lucid_cli(
        [
            "interference-learn",
            "--fixture",
            "bank",
            "--store",
            str(store_path),
            "--audit-dir",
            str(audit_dir),
        ]
    )

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["patches"]
    assert store_path.exists()
    assert Path(result["audit_path"], "README.txt").exists()

    exit_code = lucid_cli(
        [
            "interference",
            "--fixture",
            "bank",
            "--use-store",
            "--store",
            str(store_path),
        ]
    )
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert any(edge["scope_frame_id"] == "cf_event_two" for edge in out["trace_trace_edges"])


def test_interference_learning_failure_can_weaken_local_links(tmp_path: Path) -> None:
    inp = _bank_interference_input()
    out = run_interference(inp)
    store_path = tmp_path / "interference_links.json"

    result = learn_interference(
        inp,
        out,
        validation_success=False,
        failure_type="interference_or_basin",
        store_path=store_path,
        audit_dir=tmp_path / "audit",
    )
    learned = load_learned_interference_links(store_path)

    assert result.patches
    assert all(patch.delta < 0 for patch in result.patches)
    assert any(link.weight < 0 for link in learned)
