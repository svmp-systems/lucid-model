from __future__ import annotations

from pathlib import Path

from lucid.cli import main as lucid_main
from lucid.cognition.input.cue import encode_cues
from lucid.cognition.input.perception import PerceptionConfig, perceive
from lucid.cognition.orchestrator.runner import OrchestratorConfig, OrchestratorRunner
from lucid.cognition.reasoning.binding import BindingConfig, run_binding
from lucid.ir.binding import BindingInput
from lucid.ir.common import ComputePolicy, Modality
from lucid.ir.cue import CueCloud, TraceActivationRequest
from lucid.ir.dmf import ActiveTrace, ConflictSignal, DmfInput, DmfOutput
from lucid.ir.perception import (
    CandidateUnit,
    ChangeHint,
    PerceptionInput,
    PerceptualEvidenceGraph,
)
from lucid.memory.dmf import load_dynamic_memory_field
from lucid.training import adapters
from lucid.training.adapters import episode_to_cue_encoder_input, episode_to_dmf_cue_cloud
from lucid.training.generator.engine import AmbiguityKnob, rng_for_seed
from lucid.training.generator.output import write_episodes
from lucid.training.generator.recipes import bank_destination


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
    lucid_main(["train", "dmf", "--episodes", str(jsonl), "--checkpoint", str(checkpoint), "--steps", "1"])
    lucid_main(
        ["train", "binding", "--episodes", str(jsonl), "--checkpoint", str(checkpoint), "--steps", "1"]
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


def test_cli_bind_smoke(capsys) -> None:
    exit_code = lucid_main(["bind", "--fixture", "bank", "--backend", "rule"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "candidate_frames" in captured.out
    assert "event_one" in captured.out
