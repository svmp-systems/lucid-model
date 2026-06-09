from __future__ import annotations

import json
from pathlib import Path

import pytest

from lucid.cognition.orchestrator.runner import OrchestratorConfig, OrchestratorRunner
from lucid.cognition.reasoning.basins import BasinsConfig, run_basins
from lucid.cognition.reasoning.context_op import run_context_op
from lucid.ir.basins import BasinInput, CandidateBasinState
from lucid.ir.binding import CandidateFrame
from lucid.ir.common import ComputePolicy
from lucid.ir.context_op import ContextFrame, LocalBasinPressure
from lucid.ir.interference import FrameBasinEdge, InterferenceOutput
from lucid.memory.basin_bank import normalize_family_hint
from lucid.training import adapters
from lucid.training.checkpoints import load_checkpoint
from lucid.cli import main as lucid_main
from lucid.cognition.input.perception import PerceptionConfig
from test_context_op import _bank_context_input

def _bank_basin_input() -> BasinInput:
    context_out = run_context_op(_bank_context_input())
    return BasinInput(
        interference_output=InterferenceOutput(),
        candidate_frames=_bank_context_input().binding_candidate_frames,
        context_frames=context_out.context_frames,
        local_basin_pressures=context_out.local_basin_pressures,
    )


def _bank_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    records = [
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
    ]
    (checkpoint / "basin_bank.json").write_text(
        json.dumps({"records": records, "next_id": 4}),
        encoding="utf-8",
    )
    return checkpoint


def test_normalize_family_hint_strips_like_suffix() -> None:
    assert normalize_family_hint("financial_destination_like") == "financial_destination"


def test_basins_emit_plural_scoped_candidates(tmp_path: Path) -> None:
    checkpoint = _bank_checkpoint(tmp_path)
    out = run_basins(_bank_basin_input(), config=BasinsConfig(checkpoint=checkpoint))

    assert len(out.candidate_basin_states) >= 2
    basin_ids = {state.basin_id for state in out.candidate_basin_states}
    assert "b0001" in basin_ids
    assert "b0002" in basin_ids

    event_two = [
        state
        for state in out.candidate_basin_states
        if "cf_event_two" in state.scope_frame_ids
    ]
    assert len(event_two) >= 2
    assert out.competition_summary.top_basin_id
    assert out.competition_summary.active_basin_count >= 2
    assert event_two[0].margin_vs_next >= 0.0
    energies = sorted(state.energy for state in event_two)
    assert energies[-1] - energies[0] < 0.35


def test_basin_cap_preserves_scope_coverage(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "basin_bank.json").write_text(
        json.dumps(
            {
                "records": [
                    {"basin_id": "b_a1", "frame_affinities": {"frame_a": 1.0}},
                    {"basin_id": "b_a2", "frame_affinities": {"frame_a": 0.9}},
                    {"basin_id": "b_a3", "frame_affinities": {"frame_a": 0.8}},
                    {"basin_id": "b_b1", "frame_affinities": {"frame_b": 0.7}},
                ],
                "next_id": 5,
            }
        ),
        encoding="utf-8",
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
        (state.scope_frame_ids[0], state.basin_id)
        for state in out.candidate_basin_states
    } == {
        ("scope_a", "b_a1"),
        ("scope_b", "b_b1"),
    }


def test_competition_summary_uses_scope_local_margin(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "basin_bank.json").write_text(
        json.dumps(
            {
                "records": [
                    {"basin_id": "b_scope_a", "frame_affinities": {"frame_a": 1.0}},
                    {"basin_id": "b_scope_b", "frame_affinities": {"frame_b": 0.99}},
                ],
                "next_id": 3,
            }
        ),
        encoding="utf-8",
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
        ),
        config=BasinsConfig(checkpoint=checkpoint),
    )

    top = out.candidate_basin_states[0]
    assert top.basin_id == "b_scope_a"
    assert out.competition_summary.second_basin_id == ""
    assert out.competition_summary.top_margin == pytest.approx(top.energy)
    assert all(state.margin_vs_next == 0.0 for state in out.candidate_basin_states)


def test_assemblies_use_scoped_basin_instances(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "basin_bank.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "basin_id": "b_shared",
                        "frame_affinities": {"frame_a": 1.0, "frame_b": 0.5},
                        "cooperation_links": {"b_partner": 1.0},
                    },
                    {"basin_id": "b_partner", "frame_affinities": {"frame_b": 1.0}},
                ],
                "next_id": 3,
            }
        ),
        encoding="utf-8",
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
        ),
        config=BasinsConfig(checkpoint=checkpoint),
    )

    assert len(out.basin_assemblies) == 1
    assembly = out.basin_assemblies[0]
    energy = {
        (state.scope_frame_ids[0], state.basin_id): state.energy
        for state in out.candidate_basin_states
    }
    assert assembly.scope_frame_ids == ["scope_b"]
    assert assembly.combined_energy == pytest.approx(
        energy[("scope_b", "b_shared")] + energy[("scope_b", "b_partner")]
    )


def test_frame_basin_edges_apply_only_to_their_scope(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "basin_bank.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "basin_id": "b_shared",
                        "family_hint": "shared",
                        "frame_affinities": {"frame_a": 0.8, "frame_b": 0.8},
                    }
                ],
                "next_id": 2,
            }
        ),
        encoding="utf-8",
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
        ),
        config=BasinsConfig(checkpoint=checkpoint),
    )

    energy_by_scope = {
        state.scope_frame_ids[0]: state.energy
        for state in out.candidate_basin_states
        if state.basin_id == "b_shared"
    }
    assert energy_by_scope["scope_a"] - energy_by_scope["scope_b"] == pytest.approx(0.2)


def test_cooperation_links_skip_inactive_source_basins(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "basin_bank.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "basin_id": "b_active",
                        "family_hint": "shared",
                        "frame_affinities": {"frame_a": 0.8},
                    },
                    {
                        "basin_id": "b_inactive",
                        "family_hint": "inactive",
                        "frame_affinities": {},
                        "cooperation_links": {"b_active": 0.9},
                    },
                ],
                "next_id": 3,
            }
        ),
        encoding="utf-8",
    )

    out = run_basins(
        BasinInput(
            interference_output=InterferenceOutput(),
            candidate_frames=[
                CandidateFrame(frame_id="frame_a", frame_type="event", confidence=1.0),
            ],
            context_frames=[
                ContextFrame(context_frame_id="scope_a", member_frame_ids=["frame_a"]),
            ],
            local_basin_pressures=[
                LocalBasinPressure(context_frame_id="scope_a", basin_family_hints={"shared": 0.5}),
            ],
        ),
        config=BasinsConfig(checkpoint=checkpoint),
    )

    assert [state.basin_id for state in out.candidate_basin_states] == ["b_active"]
    assert out.basin_assemblies == []


def test_prior_basin_state_adds_continuity_bias(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "basin_bank.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "basin_id": "b_prior",
                        "family_hint": "shared",
                        "frame_affinities": {"frame_a": 0.8},
                    },
                    {
                        "basin_id": "b_peer",
                        "family_hint": "shared",
                        "frame_affinities": {"frame_a": 0.8},
                    },
                ],
                "next_id": 3,
            }
        ),
        encoding="utf-8",
    )

    out = run_basins(
        BasinInput(
            interference_output=InterferenceOutput(),
            candidate_frames=[
                CandidateFrame(frame_id="frame_a", frame_type="event", confidence=1.0),
            ],
            context_frames=[
                ContextFrame(context_frame_id="scope_a", member_frame_ids=["frame_a"]),
            ],
            prior_basin_state=[
                CandidateBasinState(
                    basin_id="b_prior",
                    energy=1.0,
                    scope_frame_ids=["scope_a"],
                )
            ],
        ),
        config=BasinsConfig(checkpoint=checkpoint),
    )

    energy_by_id = {state.basin_id: state.energy for state in out.candidate_basin_states}
    assert energy_by_id["b_prior"] - energy_by_id["b_peer"] == pytest.approx(0.12)


def test_suppression_links_penalize_target_in_same_scope(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "basin_bank.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "basin_id": "b_source",
                        "family_hint": "shared",
                        "frame_affinities": {"frame_a": 0.8},
                        "suppression_links": {"b_target": 1.0},
                    },
                    {
                        "basin_id": "b_target",
                        "family_hint": "shared",
                        "frame_affinities": {"frame_a": 0.8},
                    },
                ],
                "next_id": 3,
            }
        ),
        encoding="utf-8",
    )

    out = run_basins(
        BasinInput(
            interference_output=InterferenceOutput(),
            candidate_frames=[
                CandidateFrame(frame_id="frame_a", frame_type="event", confidence=1.0),
            ],
            context_frames=[
                ContextFrame(context_frame_id="scope_a", member_frame_ids=["frame_a"]),
            ],
        ),
        config=BasinsConfig(checkpoint=checkpoint),
    )

    energy_by_id = {state.basin_id: state.energy for state in out.candidate_basin_states}
    assert energy_by_id["b_source"] - energy_by_id["b_target"] == pytest.approx(0.25)


def test_basins_trainer_and_gold_families(tmp_path: Path) -> None:
    from lucid.training.generator.engine import AmbiguityKnob, rng_for_seed
    from lucid.training.generator.output import write_episodes
    from lucid.training.generator.recipes import bank_destination

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

    context_out = run_context_op(_bank_context_input())
    out = run_basins(
        BasinInput(
            interference_output=InterferenceOutput(),
            candidate_frames=_bank_context_input().binding_candidate_frames,
            context_frames=context_out.context_frames,
            local_basin_pressures=context_out.local_basin_pressures,
        ),
        config=BasinsConfig(checkpoint=checkpoint),
    )
    gold_families = {target["family_hint"] for target in adapters.basin_targets(episode)}
    store = load_checkpoint(checkpoint).ensure_store("basin_bank")
    family_by_id = {
        str(record["basin_id"]): normalize_family_hint(str(record.get("family_hint", "")))
        for record in store.get("records", [])
    }
    assert all("maturity_state" not in record and "heat_tier" not in record for record in store["records"])
    active_families = {
        family_by_id[state.basin_id]
        for state in out.candidate_basin_states
        if state.basin_id in family_by_id
    }
    assert gold_families.issubset(active_families)
    assert len(out.candidate_basin_states) >= 2


def test_orchestrator_basins_after_train(tmp_path: Path) -> None:
    from lucid.training.generator.engine import AmbiguityKnob, rng_for_seed
    from lucid.training.generator.output import write_episodes
    from lucid.training.generator.recipes import bank_destination

    episode = bank_destination.make(rng_for_seed(5), AmbiguityKnob(0.9))
    checkpoint = tmp_path / "checkpoint"
    jsonl = tmp_path / "ep.jsonl"
    write_episodes([episode], jsonl)
    lucid_main(["train", "basins", "--episodes", str(jsonl), "--checkpoint", str(checkpoint), "--steps", "1"])

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
    checkpoint = _bank_checkpoint(tmp_path)
    audit_dir = tmp_path / "audit"

    assert (
        lucid_main(
            [
                "basins",
                "--checkpoint",
                str(checkpoint),
                "--audit-dir",
                str(audit_dir),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    run_dirs = list(audit_dir.iterdir())
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert output["candidate_basin_states"]
    assert len(run_dirs) == 1
    assert manifest["stage_name"] == "basins"
    assert manifest["candidate_basin_count"] == len(output["candidate_basin_states"])
