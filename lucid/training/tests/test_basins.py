from __future__ import annotations

import json
from pathlib import Path

import pytest

from lucid.cli import main as lucid_main
from lucid.cognition.input.perception import PerceptionConfig
from lucid.cognition.pipe_orchestrator.runner import OrchestratorConfig, OrchestratorRunner
from lucid.cognition.reasoning.basins import BasinsConfig, run_basins
from lucid.cognition.reasoning.context_op import run_context_op
from lucid.ir.basins import BasinInput, CandidateBasinState
from lucid.ir.binding import CandidateFrame
from lucid.ir.common import ComputePolicy
from lucid.ir.context_op import ContextFrame, LocalBasinPressure
from lucid.ir.interference import FrameBasinEdge, InterferenceOutput
from lucid.cognition.memory.basin_bank import normalize_family_hint
from lucid.training.corpus import adapters
from lucid.training.checkpoint.store import load_checkpoint
from lucid.training.tests.test_context_op import _bank_context_input


def _write_bank(path: Path, records: list[dict]) -> Path:
    path.mkdir()
    (path / "basin_bank.json").write_text(
        json.dumps({"records": records, "next_id": len(records) + 1}),
        encoding="utf-8",
    )
    return path


def _bank_basin_input() -> BasinInput:
    context_input = _bank_context_input()
    context_output = run_context_op(context_input)
    return BasinInput(
        interference_output=InterferenceOutput(),
        candidate_frames=context_input.binding_candidate_frames,
        context_frames=context_output.context_frames,
        local_basin_pressures=context_output.local_basin_pressures,
    )


def test_normalize_family_hint_strips_like_suffix() -> None:
    assert normalize_family_hint("financial_destination_like") == "financial_destination"


def test_basins_emit_plural_scoped_candidates(tmp_path: Path) -> None:
    checkpoint = _write_bank(
        tmp_path / "checkpoint",
        [
            {
                "basin_id": "b0001",
                "family_hint": "financial_destination",
                "frame_affinities": {"event_two": 0.82},
            },
            {
                "basin_id": "b0002",
                "family_hint": "river_destination",
                "frame_affinities": {"event_two": 0.78},
            },
            {
                "basin_id": "b0003",
                "family_hint": "outdoor_context",
                "frame_affinities": {"event_one": 0.8},
            },
        ],
    )

    out = run_basins(_bank_basin_input(), config=BasinsConfig(checkpoint=checkpoint))

    basin_ids = {state.basin_id for state in out.candidate_basin_states}
    assert {"b0001", "b0002"}.issubset(basin_ids)
    event_two = [
        state for state in out.candidate_basin_states if "cf_event_two" in state.scope_frame_ids
    ]
    assert len(event_two) >= 2
    assert out.competition_summary.top_basin_id
    assert out.competition_summary.active_basin_count >= 2
    assert event_two[0].margin_vs_next >= 0.0
    assert "basin_bank_size=3" in out.audit_notes


def test_basin_cap_preserves_scope_coverage(tmp_path: Path) -> None:
    checkpoint = _write_bank(
        tmp_path / "checkpoint",
        [
            {"basin_id": "b_a1", "frame_affinities": {"frame_a": 1.0}},
            {"basin_id": "b_a2", "frame_affinities": {"frame_a": 0.9}},
            {"basin_id": "b_a3", "frame_affinities": {"frame_a": 0.8}},
            {"basin_id": "b_b1", "frame_affinities": {"frame_b": 0.7}},
        ],
    )

    out = run_basins(
        BasinInput(
            interference_output=InterferenceOutput(),
            candidate_frames=[
                CandidateFrame(frame_id="frame_a", frame_type="event", confidence=1.0),
                CandidateFrame(frame_id="frame_b", frame_type="event", confidence=1.0),
            ],
            context_frames=[
                ContextFrame(context_frame_id="scope_a", member_frame_ids=["frame_a"]),
                ContextFrame(context_frame_id="scope_b", member_frame_ids=["frame_b"]),
            ],
            compute_policy=ComputePolicy(max_active_basins=2),
        ),
        config=BasinsConfig(checkpoint=checkpoint),
    )

    assert {
        (state.scope_frame_ids[0], state.basin_id) for state in out.candidate_basin_states
    } == {("scope_a", "b_a1"), ("scope_b", "b_b1")}


def test_activation_signature_wakes_source_backed_basin(tmp_path: Path) -> None:
    checkpoint = _write_bank(
        tmp_path / "checkpoint",
        [
            {
                "basin_id": "b_qubit",
                "family_hint": "qubit",
                "frame_affinities": {},
                "activation_signature": {"t_qubit": 1.0, "qubit": 0.8},
                "semantic_signature": {"qubit": 1.0},
                "evidence_handles": ["concept:qubit", "claim:qubit:0"],
                "relation_handles": ["relation:qubit:0:type_of"],
                "source_refs": ["ibm_quantum_computing"],
                "trust_score": 0.82,
                "heat_tier": "quarantine",
                "quantized_payload": {"precision": "uint8_sparse"},
            }
        ],
    )

    out = run_basins(
        BasinInput(
            interference_output=InterferenceOutput(),
            candidate_frames=[
                CandidateFrame(
                    frame_id="frame_question",
                    frame_type="concept",
                    confidence=0.9,
                    supporting_trace_ids=["t_qubit"],
                )
            ],
            context_frames=[
                ContextFrame(context_frame_id="scope_question", member_frame_ids=["frame_question"]),
            ],
        ),
        config=BasinsConfig(checkpoint=checkpoint),
    )

    assert [state.basin_id for state in out.candidate_basin_states] == ["b_qubit"]
    state = out.candidate_basin_states[0]
    assert state.evidence_handles == ["concept:qubit", "claim:qubit:0"]
    assert state.relation_handles == ["relation:qubit:0:type_of"]
    assert state.source_refs == ["ibm_quantum_computing"]
    assert state.quantized_payload["precision"] == "uint8_sparse"
    assert "basin_evidence_handles=2" in out.audit_notes


def test_prior_state_and_frame_edges_are_scope_local(tmp_path: Path) -> None:
    checkpoint = _write_bank(
        tmp_path / "checkpoint",
        [
            {
                "basin_id": "b_shared",
                "family_hint": "shared",
                "frame_affinities": {"frame_a": 0.8, "frame_b": 0.8},
            },
            {
                "basin_id": "b_peer",
                "family_hint": "shared",
                "frame_affinities": {"frame_a": 0.8},
            },
        ],
    )

    out = run_basins(
        BasinInput(
            interference_output=InterferenceOutput(
                frame_basin_edges=[
                    FrameBasinEdge(frame_id="frame_a", basin_id="b_shared", delta=0.2)
                ]
            ),
            candidate_frames=[
                CandidateFrame(frame_id="frame_a", frame_type="event", confidence=1.0),
                CandidateFrame(frame_id="frame_b", frame_type="event", confidence=1.0),
            ],
            context_frames=[
                ContextFrame(context_frame_id="scope_a", member_frame_ids=["frame_a"]),
                ContextFrame(context_frame_id="scope_b", member_frame_ids=["frame_b"]),
            ],
            local_basin_pressures=[
                LocalBasinPressure(context_frame_id="scope_a", basin_family_hints={"shared": 0.5}),
                LocalBasinPressure(context_frame_id="scope_b", basin_family_hints={"shared": 0.5}),
            ],
            prior_basin_state=[
                CandidateBasinState(basin_id="b_peer", energy=1.0, scope_frame_ids=["scope_a"])
            ],
        ),
        config=BasinsConfig(checkpoint=checkpoint),
    )

    energy = {
        (state.scope_frame_ids[0], state.basin_id): state.energy
        for state in out.candidate_basin_states
    }
    assert energy[("scope_a", "b_shared")] - energy[("scope_b", "b_shared")] == pytest.approx(0.2)
    assert energy[("scope_a", "b_peer")] - energy[("scope_a", "b_shared")] == pytest.approx(-0.08)


def test_basins_trainer_stores_quantized_payload(tmp_path: Path) -> None:
    from lucid.training.corpus.engine import AmbiguityKnob, rng_for_seed
    from lucid.training.corpus.output import write_episodes
    from lucid.training.corpus.recipes import bank_destination

    episode = bank_destination.make(rng_for_seed(3), AmbiguityKnob(0.8))
    checkpoint = tmp_path / "checkpoint"
    jsonl = tmp_path / "ep.jsonl"
    write_episodes([episode], jsonl)

    assert (
        lucid_main(
            [
                "train",
                "basins",
                "--episodes",
                str(jsonl),
                "--checkpoint",
                str(checkpoint),
                "--steps",
                "1",
            ]
        )
        == 0
    )

    store = load_checkpoint(checkpoint).ensure_store("basin_bank")
    gold_families = {target["family_hint"] for target in adapters.basin_targets(episode)}
    assert all(record.get("heat_tier") == "quarantine" for record in store["records"])
    assert all(
        record.get("quantized_payload", {}).get("precision") == "uint8_sparse"
        for record in store["records"]
    )
    assert gold_families.issubset(
        {normalize_family_hint(str(record["family_hint"])) for record in store["records"]}
    )


def test_orchestrator_basins_after_train(tmp_path: Path) -> None:
    from lucid.training.corpus.engine import AmbiguityKnob, rng_for_seed
    from lucid.training.corpus.output import write_episodes
    from lucid.training.corpus.recipes import bank_destination

    episode = bank_destination.make(rng_for_seed(5), AmbiguityKnob(0.9))
    checkpoint = tmp_path / "checkpoint"
    jsonl = tmp_path / "ep.jsonl"
    write_episodes([episode], jsonl)
    lucid_main(
        [
            "train",
            "basins",
            "--episodes",
            str(jsonl),
            "--checkpoint",
            str(checkpoint),
            "--steps",
            "1",
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
    assert run.basin_output is not None
    assert len(run.basin_output.candidate_basin_states) >= 1
    assert run.basin_output.competition_summary.active_basin_count >= 1


def test_universal_cli_basins_smoke_writes_audit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audit_dir = tmp_path / "audit"
    checkpoint = _write_bank(
        tmp_path / "checkpoint",
        [
            {
                "basin_id": "b0001",
                "family_hint": "financial_destination",
                "frame_affinities": {"event_two": 0.82},
            }
        ],
    )

    assert lucid_main(["basins", "--audit-dir", str(audit_dir), "--checkpoint", str(checkpoint)]) == 0

    output = json.loads(capsys.readouterr().out)
    run_dirs = list(audit_dir.iterdir())
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    stage = json.loads((run_dirs[0] / "output.json").read_text(encoding="utf-8"))
    assert output["candidate_basin_states"]
    assert len(run_dirs) == 1
    assert manifest["stage_name"] == "basins"
    assert len(stage["candidate_basin_states"]) == len(output["candidate_basin_states"])
