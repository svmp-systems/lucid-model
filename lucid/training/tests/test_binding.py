from __future__ import annotations

from pathlib import Path

from lucid.cli import main as lucid_main
from lucid.cognition.input.cue import encode_cues
from lucid.cognition.input.perception import PerceptionConfig, perceive
from lucid.cognition.pipe_orchestrator.runner import OrchestratorConfig, OrchestratorRunner
from lucid.cognition.reasoning.binding import BindingConfig, _apply_operators, run_binding
from lucid.ir.binding import BindingInput, GraphEdge, GraphNode, LocalGraph
from lucid.ir.common import ComputePolicy, Modality
from lucid.ir.cue import CueCloud, RelationalActivationRequest, TraceActivationRequest
from lucid.ir.dmf import ActiveTrace, ConflictSignal, DmfInput, DmfOutput
from lucid.ir.perception import (
    CandidateUnit,
    ChangeHint,
    PerceptionInput,
    PerceptualEvidenceGraph,
)
from lucid.cognition.memory.dmf import load_dynamic_memory_field
from lucid.training.corpus import adapters
from lucid.training.corpus.adapters import episode_to_cue_encoder_input, episode_to_dmf_cue_cloud
from lucid.training.corpus.engine import AmbiguityKnob, rng_for_seed
from lucid.training.corpus.output import write_episodes
from lucid.training.corpus.recipes import bank_destination
from lucid.training.quantum_articles import train_quantum_articles


def _bank_binding_input() -> BindingInput:
    graph = perceive(
        PerceptionInput(
            raw_payload="i found money while kayaking and deposited it at the bank",
            modality=Modality.TEXT,
        ),
        config=PerceptionConfig(backend="rule"),
    )
    cloud = CueCloud(
        primitive_trace_activations=[
            TraceActivationRequest("found", 0.9, ["u_found"]),
            TraceActivationRequest("money", 0.9, ["u_money"]),
            TraceActivationRequest("kayaking", 0.7, ["u_kayaking"]),
            TraceActivationRequest("deposited", 0.85, ["u_deposited"]),
            TraceActivationRequest("bank", 0.8, ["u_bank"]),
        ]
    )
    dmf = DmfOutput(
        active_traces=[
            ActiveTrace("t_found", 0.82),
            ActiveTrace("t_money", 0.79),
            ActiveTrace("t_kayak", 0.76),
            ActiveTrace("t_placed", 0.74),
            ActiveTrace("t_bank", 0.58),
        ],
        conflict_signals=[ConflictSignal("t_kayak", "t_bank", severity=0.8)],
        coverage_score=0.75,
    )
    return BindingInput(
        dmf_output=dmf,
        perceptual_evidence_graph=graph,
        cue_cloud=cloud,
    )


def test_binding_emits_plural_frames_from_regions() -> None:
    out = run_binding(_bank_binding_input())

    frame_ids = {frame.frame_id for frame in out.candidate_frames}
    assert "event_one" in frame_ids
    assert "event_two" in frame_ids
    event_two = next(frame for frame in out.candidate_frames if frame.frame_id == "event_two")
    assert event_two.role_assignments
    assert all(slot_id.startswith("slot_") for slot_id in event_two.role_assignments)
    assert event_two.slot_evidence_refs
    assert event_two.slot_affinity_hints
    assert out.binding_stability_score > 0.0
    assert out.frame_competition_edges


def test_binding_uses_unit_specific_trace_for_each_slot() -> None:
    graph = PerceptualEvidenceGraph(
        candidate_units=[
            CandidateUnit("u_act", "found", "verb"),
            CandidateUnit("u_theme", "money", "noun"),
        ],
    )
    out = run_binding(
        BindingInput(
            dmf_output=DmfOutput(
                active_traces=[
                    ActiveTrace("t_found", 1.0),
                    ActiveTrace("t_money", 1.0),
                ],
            ),
            perceptual_evidence_graph=graph,
            cue_cloud=CueCloud(
                primitive_trace_activations=[
                    TraceActivationRequest("t_found", 0.9, ["u_act"]),
                    TraceActivationRequest("t_money", 0.9, ["u_theme"]),
                ],
            ),
        )
    )

    frame = out.candidate_frames[0]
    slot_by_unit = {
        refs[0]: slot_id
        for slot_id, refs in frame.slot_evidence_refs.items()
        if refs
    }
    assert frame.role_assignments[slot_by_unit["u_act"]] == "t_found"
    assert frame.role_assignments[slot_by_unit["u_theme"]] == "t_money"
    assert not frame.conflicting_trace_ids


def test_binding_grid_change_hint_produces_transform_frame() -> None:
    graph = PerceptualEvidenceGraph(
        candidate_units=[
            CandidateUnit("u_in_0", "(0,1)"),
            CandidateUnit("u_out_0", "(1,2)"),
        ],
        change_hints=[
            ChangeHint(
                change_type="position_shift",
                before_unit_id="u_in_0",
                after_unit_id="u_out_0",
                weight=0.9,
            )
        ],
    )
    out = run_binding(
        BindingInput(
            dmf_output=DmfOutput(
                active_traces=[
                    ActiveTrace("t_shift", 0.9),
                    ActiveTrace("t_shape", 0.85),
                ]
            ),
            perceptual_evidence_graph=graph,
            cue_cloud=CueCloud(
                primitive_trace_activations=[
                    TraceActivationRequest("position_shift_like", 0.9, ["u_in_0", "u_out_0"]),
                ]
            ),
        )
    )
    assert any(frame.frame_type == "transform" for frame in out.candidate_frames)


def test_binding_trainer_and_checkpoint(tmp_path: Path) -> None:
    episode = bank_destination.make(rng_for_seed(3), AmbiguityKnob(0.8))
    checkpoint = tmp_path / "checkpoint"
    audit_dir = tmp_path / "audit"
    jsonl = tmp_path / "one.jsonl"
    write_episodes([episode], jsonl)

    assert (
        lucid_main(
            [
                "train",
                "dmf",
                "--episodes",
                str(jsonl),
                "--checkpoint",
                str(checkpoint),
                "--audit-dir",
                str(audit_dir),
                "--steps",
                "1",
                "--allow-generator-gold",
            ]
        )
        == 0
    )
    assert (
        lucid_main(
            [
                "train",
                "binding",
                "--episodes",
                str(jsonl),
                "--checkpoint",
                str(checkpoint),
                "--audit-dir",
                str(audit_dir),
                "--steps",
                "1",
                "--allow-generator-gold",
            ]
        )
        == 0
    )

    graph = perceive(
        PerceptionInput(raw_payload=episode.raw_input, modality=Modality.TEXT),
        config=PerceptionConfig(backend="rule"),
    )
    cue_input = episode_to_cue_encoder_input(episode)
    cloud = encode_cues(cue_input)
    dmf_cue = episode_to_dmf_cue_cloud(episode)
    dmf = load_dynamic_memory_field(checkpoint)
    dmf_out = dmf.run(
        DmfInput(cue_cloud=dmf_cue, compute_policy=ComputePolicy(max_active_traces=8))
    )
    out = run_binding(
        BindingInput(dmf_output=dmf_out, perceptual_evidence_graph=graph, cue_cloud=cloud or dmf_cue),
        config=BindingConfig(checkpoint=checkpoint),
    )
    gold_frames = {target["frame_id"] for target in adapters.binding_frame_targets(episode)}
    assert gold_frames.issubset({frame.frame_id for frame in out.candidate_frames})


def test_orchestrator_binding_frames_after_train(tmp_path: Path) -> None:
    episode = bank_destination.make(rng_for_seed(5), AmbiguityKnob(0.9))
    checkpoint = tmp_path / "checkpoint"
    jsonl = tmp_path / "ep.jsonl"
    write_episodes([episode], jsonl)
    lucid_main(
        [
            "train",
            "dmf",
            "--episodes",
            str(jsonl),
            "--checkpoint",
            str(checkpoint),
            "--steps",
            "1",
            "--allow-generator-gold",
        ]
    )
    lucid_main(
        [
            "train",
            "binding",
            "--episodes",
            str(jsonl),
            "--checkpoint",
            str(checkpoint),
            "--steps",
            "1",
            "--allow-generator-gold",
        ]
    )

    runner = OrchestratorRunner(
        config=OrchestratorConfig(
            audit_base_dir=str(tmp_path / "audit"),
            perception=PerceptionConfig(backend="rule"),
            checkpoint=str(checkpoint),
        )
    )
    run = runner.run_episode(episode)
    assert run.binding_output is not None
    assert len(run.binding_output.candidate_frames) >= 2


def test_binding_marks_competing_cue_routes_unresolved() -> None:
    graph = PerceptualEvidenceGraph(
        candidate_units=[
            CandidateUnit("u_went", "went", "verb"),
            CandidateUnit("u_bank", "bank", "noun"),
        ],
    )
    out = run_binding(
        BindingInput(
            dmf_output=DmfOutput(
                active_traces=[
                    ActiveTrace("t0001", 0.4, cluster_id="financial_action_like"),
                    ActiveTrace("t0002", 0.35, cluster_id="river_location_like"),
                ],
                uncertainty_summary="high",
            ),
            perceptual_evidence_graph=graph,
            cue_cloud=CueCloud(
                primitive_trace_activations=[
                    TraceActivationRequest("went", 0.62, ["u_went"]),
                    TraceActivationRequest("bank", 0.62, ["u_bank"]),
                ],
                relational_trace_activations=[
                    RelationalActivationRequest(
                        "river_location_like",
                        0.251,
                        endpoint_unit_ids=["u_bank"],
                    )
                ],
            ),
        )
    )
    frame = out.candidate_frames[0]
    assert "bank_sense" in frame.unresolved_slot_names
    assert "u_bank" not in {
        refs[0]
        for slot_id, refs in frame.slot_evidence_refs.items()
        if slot_id in frame.role_assignments
    }


def test_binding_prefers_compound_cue_frame_over_broad_token_frames() -> None:
    graph = PerceptualEvidenceGraph(
        candidate_units=[
            CandidateUnit("u_quantum", "quantum", "noun", position_or_time="0"),
            CandidateUnit("u_computing", "computing", "noun", position_or_time="8"),
        ],
    )
    out = run_binding(
        BindingInput(
            dmf_output=DmfOutput(
                active_traces=[
                    ActiveTrace("t_term_quantum_computing", 0.69, cluster_id="quantum_computing"),
                    ActiveTrace("t_claim_quantum_circuit", 0.52, cluster_id="quantum_circuit"),
                ],
                coverage_score=0.75,
            ),
            perceptual_evidence_graph=graph,
            cue_cloud=CueCloud(
                primitive_trace_activations=[
                    TraceActivationRequest("quantum", 0.62, ["u_quantum"]),
                    TraceActivationRequest("computing", 0.62, ["u_computing"]),
                    TraceActivationRequest(
                        "quantum_computing",
                        0.59,
                        ["u_computing", "u_quantum"],
                    ),
                ],
            ),
        )
    )

    compound = [
        frame
        for frame in out.candidate_frames
        if frame.frame_id.startswith("local_compound_quantum_computing")
    ]
    assert compound
    assert len(out.candidate_frames) == 1
    assert not any(frame.frame_id.startswith("local_u_quantum__") for frame in out.candidate_frames)
    assert compound[0].supporting_trace_ids == ["t_term_quantum_computing"]
    assert not compound[0].conflicting_trace_ids


def test_binding_loads_concept_graph_operators_from_quantum_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "quantum_checkpoint"
    train_quantum_articles(checkpoint)
    graph = perceive(
        PerceptionInput(raw_payload="what is a qubit", modality=Modality.TEXT),
        config=PerceptionConfig(backend="rule"),
    )

    out = run_binding(
        BindingInput(
            dmf_output=DmfOutput(),
            perceptual_evidence_graph=graph,
            cue_cloud=CueCloud(),
        ),
        config=BindingConfig(checkpoint=checkpoint),
    )

    concept_frames = [
        frame
        for frame in out.candidate_frames
        if any(local_graph.family == "concept" for local_graph in frame.local_graphs)
    ]
    assert concept_frames
    edge_labels = {
        edge.label
        for frame in concept_frames
        for local_graph in frame.local_graphs
        for edge in local_graph.edges
    }
    assert {"type_of", "property", "definition_support"}.issubset(edge_labels)
    assert any(
        receipt.operator_id == "concept_relations_support_definition"
        for frame in concept_frames
        for receipt in frame.operator_receipts
    )


def test_binding_skips_quarantined_operator_effects() -> None:
    graph = LocalGraph(
        graph_id="graph_operator_gate",
        nodes=[
            GraphNode("concept:qubit", "concept", "qubit"),
            GraphNode("value:unit", "value", "unit"),
            GraphNode("value:superposition", "value", "superposition"),
        ],
        edges=[
            GraphEdge("e_type", "relation", "concept:qubit", "value:unit", label="type_of"),
            GraphEdge(
                "e_property",
                "relation",
                "concept:qubit",
                "value:superposition",
                label="property",
            ),
        ],
    )
    operator = {
        "operator_id": "concept_relations_support_definition",
        "heat_tier": "quarantine",
        "commit_permission": "support_only",
        "pattern": [
            ["relation", "type_of", "X", "Y"],
            ["relation", "property", "X", "Z"],
        ],
        "effects": [["supports", "definition_support", "X", "Y"]],
    }

    _apply_operators(graph, [operator])

    assert "definition_support" not in {edge.label for edge in graph.edges}
    operator["heat_tier"] = "warm"
    operator["commit_permission"] = "normal_support"

    _apply_operators(graph, [operator])

    assert "definition_support" in {edge.label for edge in graph.edges}


def test_cli_bind_smoke(capsys) -> None:
    exit_code = lucid_main(["bind", "--fixture", "bank", "--backend", "rule"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "candidate_frames" in captured.out
    assert "event_one" in captured.out
