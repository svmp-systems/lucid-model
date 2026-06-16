"""Lucidity gate — checks, decisions, render packet."""

from __future__ import annotations

from lucid.cognition.output.lucidity import run_checks, run_lucidity
from lucid.cognition.output.lucidity.config import LucidityConfig
from lucid.cognition.output.projector import run_projector
from lucid.ir.basins import BasinConflict, BasinOutput, CandidateBasinState, CompetitionSummary
from lucid.ir.binding import BindingOutput, CandidateFrame
from lucid.ir.common import LucidityDecision, TaskIntent
from lucid.ir.context_op import ContextFrame, ContextOpOutput, InterferenceGate
from lucid.ir.dmf import ActiveTrace, ConflictSignal, DmfOutput
from lucid.ir.interference import ConflictReport, FrameBasinEdge, InterferenceOutput
from lucid.ir.lucidity import LucidityInput
from lucid.ir.perception import CandidateUnit, PerceptualEvidenceGraph
from lucid.ir.lucidity import SearchDirectives
from lucid.ir.projector import ProjectionConstraints, ProjectionGridPair, ProjectorInput
from lucid.ir.training import Episode, GoldLabels
from lucid.cognition.pipe_orchestrator.runner import OrchestratorConfig, OrchestratorRunner
from lucid.cognition.input.perception import PerceptionConfig


def _bank_lucidity_input(*, pass_kind: str = "pre_check") -> LucidityInput:
    graph = PerceptualEvidenceGraph(
        candidate_units=[
            CandidateUnit("u_found", "found", salience=0.8),
            CandidateUnit("u_money", "money", salience=0.85),
            CandidateUnit("u_kayaking", "kayaking", salience=0.7),
            CandidateUnit("u_placed", "placed", salience=0.8),
            CandidateUnit("u_bank", "bank", salience=0.9),
        ],
    )
    frames = [
        CandidateFrame(
            frame_id="event_one",
            frame_type="event",
            role_assignments={"ACTION": "t_found", "THEME": "t_money"},
            member_evidence_refs=["u_found", "u_money", "u_kayaking"],
            confidence=0.76,
        ),
        CandidateFrame(
            frame_id="event_two",
            frame_type="event",
            role_assignments={"ACTION": "t_placed", "DESTINATION": "t_bank"},
            member_evidence_refs=["u_placed", "u_bank"],
            confidence=0.74,
            unresolved_slot_names=["bank_sense"],
        ),
    ]
    dmf = DmfOutput(
        active_traces=[ActiveTrace("t_bank", 0.58, heat_tier="warm")],
        conflict_signals=[ConflictSignal("t_kayak", "t_bank", severity=0.8)],
        coverage_score=0.86,
        top_margin=0.04,
    )
    basins = BasinOutput(
        candidate_basin_states=[
            CandidateBasinState(basin_id="b_fin", energy=0.7, margin_vs_next=0.04),
            CandidateBasinState(basin_id="b_river", energy=0.5, margin_vs_next=0.0),
        ],
        competition_summary=CompetitionSummary(
            top_basin_id="b_fin",
            second_basin_id="b_river",
            top_margin=0.04,
            active_basin_count=2,
        ),
    )
    return LucidityInput(
        basin_output=basins,
        binding_output=BindingOutput(
            candidate_frames=frames,
            binding_stability_score=0.72,
        ),
        context_op_output=ContextOpOutput(
            context_frames=[
                ContextFrame("ctx_one", member_frame_ids=["event_one"]),
                ContextFrame("ctx_two", member_frame_ids=["event_two"]),
            ],
        ),
        interference_output=InterferenceOutput(),
        dmf_output=dmf,
        perceptual_evidence_graph=graph,
        task_intent="answer",
        pass_kind=pass_kind,
    )


def test_nine_checks_populated() -> None:
    checks, confidence = run_checks(_bank_lucidity_input(), LucidityConfig())
    assert checks.margin_check is not None
    assert checks.coverage_check is not None
    assert checks.coherence_check is not None
    assert checks.binding_stability_check is not None
    assert checks.scope_check is not None
    assert checks.projection_fit_check is not None
    assert checks.contradiction_check is not None
    assert checks.maturity_check is not None
    assert checks.risk_check is not None
    assert checks.margin_check.passed is False
    assert confidence.margin == 0.04


def test_maturity_uses_relevant_basin_traces_not_broad_recall() -> None:
    inp = _bank_lucidity_input()
    inp.dmf_output.active_traces = [
        ActiveTrace("t_bank", 0.58, heat_tier="warm"),
        ActiveTrace("t_background", 0.21, heat_tier="quarantine"),
    ]
    inp.basin_output.candidate_basin_states[0].supporting_trace_ids = ["t_bank"]

    checks, _ = run_checks(inp, LucidityConfig())

    assert checks.maturity_check is not None
    assert checks.maturity_check.passed is True
    assert checks.maturity_check.score == 1.0
    assert checks.maturity_check.details["active_trace_count"] == 2
    assert checks.maturity_check.details["relevant_trace_count"] == 1


def test_source_backed_mature_basin_can_commit_with_local_binding_noise() -> None:
    inp = LucidityInput(
        basin_output=BasinOutput(
            candidate_basin_states=[
                CandidateBasinState(
                    basin_id="b_qubit_definition",
                    energy=0.68,
                    supporting_trace_ids=["t_term_qubit"],
                    coherence_score=0.85,
                    source_refs=["ibm_quantum_computing"],
                    heat_tier="warm",
                    quantized_payload={
                        "canonical_label": "qubit",
                        "relations": [
                            {
                                "relation": "type_of",
                                "target": "unit of quantum information",
                                "confidence": 0.82,
                                "source_refs": ["ibm_quantum_computing"],
                            }
                        ],
                    },
                )
            ],
            competition_summary=CompetitionSummary(
                top_basin_id="b_qubit_definition",
                top_margin=0.02,
                active_basin_count=1,
            ),
            unresolved_conflicts=[
                BasinConflict(
                    scope_frame_id="cf_qubit",
                    conflict_type="low_margin_competition",
                    basin_ids=["b_qubit_definition", "b_qubit_mechanism"],
                ),
                BasinConflict(
                    scope_frame_id="cf_other",
                    conflict_type="low_margin_competition",
                    basin_ids=["b_other_definition", "b_other_mechanism"],
                ),
            ],
        ),
        binding_output=BindingOutput(
            candidate_frames=[
                CandidateFrame(
                    frame_id="local_u_qubit",
                    frame_type="local_reading",
                    confidence=0.1,
                    conflicting_trace_ids=["t_background"],
                )
            ],
            binding_stability_score=0.1,
        ),
        context_op_output=ContextOpOutput(),
        interference_output=InterferenceOutput(),
        dmf_output=DmfOutput(
            active_traces=[
                ActiveTrace("t_term_qubit", 0.64, heat_tier="warm"),
                ActiveTrace("t_background", 0.2, heat_tier="quarantine"),
            ],
            coverage_score=0.75,
        ),
        perceptual_evidence_graph=PerceptualEvidenceGraph(),
        task_intent="chat",
    )

    checks, _ = run_checks(inp, LucidityConfig())
    assert checks.coherence_check is not None
    assert checks.coherence_check.passed is False
    assert checks.contradiction_check is not None
    assert checks.contradiction_check.passed is True
    assert checks.contradiction_check.details["ignored_basin_conflict_count"] == 2

    out = run_lucidity(inp)

    assert out.decision == LucidityDecision.COMMIT
    assert out.committed_state is not None
    assert out.committed_state.primary_basin_id == "b_qubit_definition"


def test_source_backed_top_scope_ignores_peripheral_duplicate_basin_conflicts() -> None:
    inp = LucidityInput(
        basin_output=BasinOutput(
            candidate_basin_states=[
                CandidateBasinState(
                    basin_id="b_quantum_computing_definition",
                    energy=0.7,
                    supporting_trace_ids=["t_term_quantum_computing"],
                    scope_frame_ids=["cf_quantum_computing"],
                    coherence_score=0.82,
                    source_refs=["aws_quantum_computing"],
                    heat_tier="warm",
                    quantized_payload={
                        "canonical_label": "quantum computing",
                        "relations": [
                            {
                                "relation": "type_of",
                                "target": "multidisciplinary field using quantum mechanics",
                                "confidence": 0.86,
                                "source_refs": ["aws_quantum_computing"],
                            }
                        ],
                    },
                ),
                CandidateBasinState(
                    basin_id="b_quantum_computing_definition",
                    energy=0.66,
                    supporting_trace_ids=["t_claim_quantum_circuit"],
                    scope_frame_ids=["cf_quantum_circuit"],
                    coherence_score=0.82,
                    source_refs=["aws_quantum_computing"],
                    heat_tier="warm",
                ),
                CandidateBasinState(
                    basin_id="b_quantum_circuit_definition",
                    energy=0.64,
                    supporting_trace_ids=["t_claim_quantum_circuit"],
                    scope_frame_ids=["cf_quantum_circuit"],
                    coherence_score=0.82,
                    source_refs=["ibm_quantum_circuit"],
                    heat_tier="warm",
                ),
            ],
            competition_summary=CompetitionSummary(
                top_basin_id="b_quantum_computing_definition",
                second_basin_id="b_quantum_computing_mechanism",
                top_margin=0.05,
                active_basin_count=3,
            ),
            unresolved_conflicts=[
                BasinConflict(
                    scope_frame_id="cf_quantum_computing",
                    conflict_type="low_margin_competition",
                    basin_ids=[
                        "b_quantum_computing_definition",
                        "b_quantum_computing_mechanism",
                    ],
                ),
                BasinConflict(
                    scope_frame_id="cf_quantum_circuit",
                    conflict_type="low_margin_competition",
                    basin_ids=[
                        "b_quantum_computing_definition",
                        "b_quantum_circuit_definition",
                    ],
                ),
            ],
        ),
        binding_output=BindingOutput(
            candidate_frames=[
                CandidateFrame(
                    frame_id="local_quantum_computing",
                    frame_type="local_reading",
                    confidence=0.7,
                    role_assignments={"slot_00": "t_term_quantum_computing"},
                ),
                CandidateFrame(
                    frame_id="local_quantum_circuit",
                    frame_type="local_reading",
                    confidence=0.7,
                    role_assignments={"slot_00": "t_claim_quantum_circuit"},
                ),
            ],
            binding_stability_score=0.43,
        ),
        context_op_output=ContextOpOutput(),
        interference_output=InterferenceOutput(),
        dmf_output=DmfOutput(
            active_traces=[
                ActiveTrace("t_term_quantum_computing", 0.69, heat_tier="warm"),
                ActiveTrace("t_claim_quantum_circuit", 0.52, heat_tier="warm"),
            ],
            coverage_score=0.75,
        ),
        perceptual_evidence_graph=PerceptualEvidenceGraph(),
        task_intent="chat",
    )

    checks, _ = run_checks(inp, LucidityConfig())
    assert checks.contradiction_check is not None
    assert checks.contradiction_check.passed is True
    assert checks.contradiction_check.details["ignored_basin_conflict_count"] == 2

    out = run_lucidity(inp)

    assert out.decision == LucidityDecision.COMMIT
    assert out.committed_state is not None
    assert out.committed_state.primary_basin_id == "b_quantum_computing_definition"


def test_low_margin_preserves_ambiguity() -> None:
    out = run_lucidity(_bank_lucidity_input())
    assert out.decision == LucidityDecision.PRESERVE_AMBIGUITY
    assert out.render_packet is not None
    assert out.render_packet.render_mode == "plural"
    assert len(out.preserved_hypotheses) >= 1 or len(out.render_packet.preserved_alternatives) >= 1


def test_high_margin_commits() -> None:
    inp = _bank_lucidity_input()
    inp.basin_output.competition_summary.top_margin = 0.2
    inp.binding_output.binding_stability_score = 0.9
    out = run_lucidity(inp)
    assert out.decision == LucidityDecision.COMMIT
    assert out.committed_state is not None
    assert out.committed_state.primary_basin_id == "b_fin"
    assert out.committed_state.render_units
    assert out.render_packet is not None
    assert out.render_packet.approved_units
    assert any(unit.payload.get("roles") for unit in out.render_packet.approved_units)


def test_interference_conflict_report_blocks_commit() -> None:
    inp = _bank_lucidity_input()
    inp.basin_output.competition_summary.top_margin = 0.2
    inp.binding_output.binding_stability_score = 0.9
    inp.interference_output = InterferenceOutput(
        conflict_reports=[
            ConflictReport(
                scope_frame_id="ctx_two",
                conflict_type="scope_leak",
                members=["t_kayak", "t_bank"],
                severity=0.95,
            )
        ]
    )
    out = run_lucidity(inp)
    assert out.decision == LucidityDecision.PRESERVE_AMBIGUITY
    assert out.check_results.contradiction_check is not None
    assert out.check_results.contradiction_check.passed is False
    assert out.check_results.contradiction_check.details.get("interference_conflict_count", 0) >= 1


def test_high_risk_answer_requests_projection_before_commit() -> None:
    inp = _bank_lucidity_input()
    inp.basin_output.competition_summary.top_margin = 0.2
    inp.binding_output.binding_stability_score = 0.9
    inp.risk_level = "high"
    inp.stakes_policy = "strict"
    out = run_lucidity(inp)
    assert out.decision == LucidityDecision.REQUEST_PROJECTION
    assert out.search_directives is not None
    assert out.search_directives.projector_targets
    assert out.check_results.risk_check is not None
    assert out.check_results.risk_check.passed is False


def test_scope_check_catches_blocked_trace_in_committed_basin() -> None:
    inp = _bank_lucidity_input()
    inp.context_op_output.interference_gates = [
        InterferenceGate(
            gate_id="g_no_kayak_to_bank",
            scope_frame_id="ctx_two",
            blocked_trace_ids=["t_kayak"],
        )
    ]
    inp.basin_output.candidate_basin_states[0].scope_frame_ids = ["ctx_two"]
    inp.basin_output.candidate_basin_states[0].supporting_trace_ids = ["t_bank", "t_kayak"]
    inp.interference_output = InterferenceOutput(
        frame_basin_edges=[FrameBasinEdge(frame_id="event_two", basin_id="b_fin", delta=0.5)]
    )
    checks, _ = run_checks(inp, LucidityConfig())
    assert checks.scope_check is not None
    assert checks.scope_check.passed is False
    assert checks.scope_check.details["violations"]


def test_grid_pre_check_requests_projection() -> None:
    inp = _bank_lucidity_input(pass_kind="pre_check")
    inp.task_intent = TaskIntent.SOLVE_GRID.value
    out = run_lucidity(inp)
    assert out.decision == LucidityDecision.REQUEST_PROJECTION
    assert out.search_directives is not None
    assert out.decoder_policy.mode == "hold"


def test_grid_final_check_commits_after_projection(tmp_path) -> None:
    projection = run_projector(
        ProjectorInput(
            projection_request=SearchDirectives(projector_targets=["asy_grid_candidate"], max_rollouts=1),
            constraints=ProjectionConstraints(
                train_pairs=[
                    ProjectionGridPair(
                        pair_id="train_0",
                        input_grid=[[0, 1, 0], [0, 0, 0]],
                        output_grid=[[0, 0, 1], [0, 0, 0]],
                    )
                ],
                test_inputs=[[[2, 0, 0], [0, 0, 0]]],
                max_rollouts=1,
            ),
            task_intent="solve_grid",
        )
    )
    inp = _bank_lucidity_input(pass_kind="final_check")
    inp.task_intent = TaskIntent.SOLVE_GRID.value
    inp.projection_output = projection
    out = run_lucidity(inp)
    assert out.decision == LucidityDecision.COMMIT
    assert out.committed_state is not None
    assert out.committed_state.projection_artifact


def test_orchestrator_grid_pipeline_with_lucidity_gate(tmp_path) -> None:
    runner = OrchestratorRunner(
        config=OrchestratorConfig(
            audit_base_dir=str(tmp_path),
            perception=PerceptionConfig(backend="rule"),
        )
    )
    run = runner.run_episode(
        Episode(
            episode_id="ep-lucidity-grid",
            modality="grid",
            raw_input={
                "input": [[0, 1, 0], [0, 0, 0]],
                "output": [[0, 0, 1], [0, 0, 0]],
            },
            gold=GoldLabels(expected_answer=[[0, 0, 1], [0, 0, 0]]),
            task_intent=TaskIntent.SOLVE_GRID,
        )
    )
    assert run.lucidity_output.decision == LucidityDecision.COMMIT
    assert run.projector_output is not None
    checks = run.lucidity_output.check_results
    assert checks.projection_fit_check is not None
    assert checks.projection_fit_check.passed


def test_definition_basin_render_units_pick_one_source() -> None:
    from lucid.cognition.output.lucidity.commit import _definition_basin_render_units
    from lucid.ir.basins import CandidateBasinState

    state = CandidateBasinState(
        basin_id="b_quantum_computing_definition",
        energy=0.7,
        source_refs=["aws_quantum_computing", "ibm_quantum_computing"],
        quantized_payload={
            "concept_id": "quantum_computing",
            "canonical_label": "quantum computing",
            "relations": [
                {
                    "relation": "type_of",
                    "target": "multidisciplinary field using quantum mechanics",
                    "confidence": 0.9,
                    "source_refs": ["aws_quantum_computing"],
                },
                {
                    "relation": "type_of",
                    "target": "a wafer not much bigger than a laptop chip",
                    "confidence": 0.95,
                    "source_refs": ["ibm_quantum_computing"],
                },
                {
                    "relation": "property",
                    "target": "still under development",
                    "confidence": 0.8,
                    "source_refs": ["aws_quantum_computing"],
                },
                {
                    "relation": "type_of",
                    "target": "Join now Case studies IBM Quantum",
                    "confidence": 0.99,
                    "source_refs": ["ibm_quantum_computing"],
                },
            ],
        },
    )

    units = _definition_basin_render_units(state)
    targets = [str(unit.payload.get("target") or "") for unit in units]

    assert len(units) <= 3
    assert any("multidisciplinary field" in target for target in targets)
    assert not any("wafer" in target for target in targets)
    assert not any("join now" in target.lower() for target in targets)


def test_definition_basin_render_units_pick_one_source() -> None:
    from lucid.cognition.output.lucidity.commit import _definition_basin_render_units
    from lucid.ir.basins import CandidateBasinState

    state = CandidateBasinState(
        basin_id="b_quantum_computing_definition",
        energy=0.7,
        source_refs=["aws_quantum_computing", "ibm_quantum_computing"],
        quantized_payload={
            "concept_id": "quantum_computing",
            "canonical_label": "quantum computing",
            "relations": [
                {
                    "relation": "type_of",
                    "target": "multidisciplinary field using quantum mechanics",
                    "confidence": 0.9,
                    "source_refs": ["aws_quantum_computing"],
                },
                {
                    "relation": "type_of",
                    "target": "a wafer not much bigger than a laptop chip",
                    "confidence": 0.95,
                    "source_refs": ["ibm_quantum_computing"],
                },
                {
                    "relation": "property",
                    "target": "still under development",
                    "confidence": 0.8,
                    "source_refs": ["aws_quantum_computing"],
                },
                {
                    "relation": "type_of",
                    "target": "Join now Case studies IBM Quantum",
                    "confidence": 0.99,
                    "source_refs": ["ibm_quantum_computing"],
                },
            ],
        },
    )

    units = _definition_basin_render_units(state)
    targets = [str(unit.payload.get("target") or "") for unit in units]

    assert len(units) <= 3
    assert any("multidisciplinary field" in target for target in targets)
    assert not any("wafer" in target for target in targets)
    assert not any("join now" in target.lower() for target in targets)
