from __future__ import annotations

import json
from pathlib import Path

import pytest

from lucid.ir.common import DecoderMode, LucidityDecision, Modality
from lucid.ir.lucidity import DecoderPolicy, LucidityOutput, SearchDirectives
from lucid.ir.training import Episode, GoldLabels
from lucid.cli import main as lucid_cli
from lucid.cognition.input.perception import PerceptionConfig
from lucid.cognition.pipe_orchestrator.runner import OrchestratorConfig, OrchestratorRunner
from lucid.cognition.pipe_orchestrator.stages import FunctionStage
from lucid.cognition.pipe_orchestrator.stub_stages import build_default_stage_fns


def test_orchestrator_dmf_activates_traces_from_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    train_exit = lucid_cli(
        [
            "train",
            "dmf",
            "--fixture",
            "bank",
            "--checkpoint",
            str(checkpoint),
            "--audit-dir",
            str(tmp_path / "train-audit"),
            "--steps",
            "1",
        ]
    )
    assert train_exit == 0

    episode = Episode(
        episode_id="ep-dmf",
        modality=Modality.TEXT,
        raw_input="I found money while kayaking and placed it in the bank.",
        seed=1,
    )
    runner = OrchestratorRunner(
        config=OrchestratorConfig(
            audit_base_dir=str(tmp_path / "audit"),
            perception=PerceptionConfig(backend="rule"),
            checkpoint=str(checkpoint),
        )
    )
    run = runner.run_episode(episode)

    assert run.dmf_output is not None
    assert run.dmf_output.active_traces or run.dmf_output.novelty_signals
    assert run.dmf_output.audit_log.get("tracebank_size", 0) >= 1


def test_orchestrator_runs_and_writes_audit(tmp_path: Path) -> None:
    episode = Episode(
        episode_id="ep-1",
        modality=Modality.TEXT,
        raw_input="go to the bank",
        gold=GoldLabels(lucidity_target="PRESERVE_AMBIGUITY", expected_answer="(test)"),
        seed=1,
    )
    runner = OrchestratorRunner(
        config=OrchestratorConfig(
            audit_base_dir=str(tmp_path),
            perception=PerceptionConfig(backend="rule"),
        )
    )
    run = runner.run_episode(episode)

    # Sanity: stage ordering exists and decoder produced output.
    stage_names = [r.stage_name for r in run.stage_results]
    assert stage_names[:3] == ["perception", "cue_encoder", "dmf"]
    assert run.decoder_output is not None

    # Audit run folder should exist with a manifest.
    audit_dir = Path(run.context.audit_dir)
    assert audit_dir.exists()
    assert (audit_dir / "manifest.json").exists()


def test_failed_stage_writes_partial_audit(tmp_path: Path) -> None:
    def fail_dmf(_inp: object, _ctx: object) -> object:
        raise ValueError("forced failure")

    fns = build_default_stage_fns()
    fns["dmf"] = fail_dmf
    stages = {name: FunctionStage(stage_name=name, fn=fn) for name, fn in fns.items()}
    runner = OrchestratorRunner(
        config=OrchestratorConfig(
            audit_base_dir=str(tmp_path),
            perception=PerceptionConfig(backend="rule"),
        ),
        stages=stages,
    )

    episode = Episode(episode_id="ep-fail", modality=Modality.TEXT, raw_input="go to the bank")
    with pytest.raises(RuntimeError, match="stage dmf failed"):
        runner.run_episode(episode)

    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir() and (p / "manifest.json").is_file()]
    assert len(run_dirs) == 1
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert [stage["stage_name"] for stage in manifest["stages"]] == [
        "perception",
        "cue_encoder",
        "dmf",
    ]

    dmf_record = json.loads((run_dirs[0] / "dmf.json").read_text(encoding="utf-8"))
    assert dmf_record["success"] is False
    assert "ValueError: forced failure" in dmf_record["error_message"]
    assert dmf_record["input"]["cue_cloud"] is not None


def test_projection_path_keeps_both_lucidity_audits(tmp_path: Path) -> None:
    def lucidity_with_projection(inp: object, _ctx: object) -> LucidityOutput:
        pass_kind = getattr(inp, "pass_kind")
        if pass_kind == "pre_check":
            return LucidityOutput(
                decision=LucidityDecision.REQUEST_PROJECTION,
                decoder_policy=DecoderPolicy(mode=DecoderMode.HOLD.value),
                search_directives=SearchDirectives(max_rollouts=1),
            )
        return LucidityOutput(
            decision=LucidityDecision.COMMIT,
            decoder_policy=DecoderPolicy(mode=DecoderMode.EXPRESS_COMMITTED.value),
        )

    fns = build_default_stage_fns()
    fns["lucidity"] = lucidity_with_projection
    stages = {name: FunctionStage(stage_name=name, fn=fn) for name, fn in fns.items()}
    runner = OrchestratorRunner(
        config=OrchestratorConfig(
            audit_base_dir=str(tmp_path),
            perception=PerceptionConfig(backend="rule"),
        ),
        stages=stages,
    )

    run = runner.run_episode(
        Episode(episode_id="ep-project", modality=Modality.TEXT, raw_input="project this")
    )
    audit_dir = Path(run.context.audit_dir)
    manifest = json.loads((audit_dir / "manifest.json").read_text(encoding="utf-8"))

    lucidity_refs = [
        stage for stage in manifest["stages"] if stage["stage_name"] == "lucidity"
    ]
    assert [stage["file_name"] for stage in lucidity_refs] == [
        "lucidity.json",
        "lucidity_02.json",
    ]

    first = json.loads((audit_dir / "lucidity.json").read_text(encoding="utf-8"))
    second = json.loads((audit_dir / "lucidity_02.json").read_text(encoding="utf-8"))
    assert first["input"]["pass_kind"] == "pre_check"
    assert first["output"]["decision"] == "request_projection"
    assert second["input"]["pass_kind"] == "final_check"
    assert second["output"]["decision"] == "commit"


def test_cli_accepts_pretty_json_with_bom(tmp_path: Path) -> None:
    payload = {
        "episode_id": "ep-cli",
        "modality": "text",
        "raw_input": "go to the bank",
        "gold": {
            "lucidity_target": "PRESERVE_AMBIGUITY",
            "expected_answer": "ok",
        },
    }
    episode_path = tmp_path / "episode.json"
    episode_path.write_text(json.dumps(payload, indent=2), encoding="utf-8-sig")

    exit_code = lucid_cli(
        ["run", str(episode_path), "--audit-dir", str(tmp_path / "audit"), "--perception", "rule"]
    )

    assert exit_code == 0
    audit_runs = [
        p
        for p in (tmp_path / "audit").iterdir()
        if p.is_dir() and (p / "manifest.json").is_file()
    ]
    assert len(audit_runs) == 1
    assert audit_runs[0].name.startswith("20")
    assert "ep-cli" in audit_runs[0].name

def test_cli_runs_perception_component(capsys) -> None:
    exit_code = lucid_cli(["perceive", "go to the bank", "--backend", "rule", "--compact"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "candidate_units" in captured.out
    assert "bank" in captured.out


def test_cli_runs_context_op_component(capsys) -> None:
    exit_code = lucid_cli(["context-op", "--fixture", "bank"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "context_frames" in captured.out
    assert "interference_gates" in captured.out
    assert "t_kayak" in captured.out


def test_llm_perception_audit_is_linked_from_stage_audit(monkeypatch, tmp_path: Path) -> None:
    from lucid.cognition.input.perception import llm as llm_mod

    def fake_chat(_cfg, _messages):
        return json.dumps(
            {
                "candidate_units": [{"unit_id": "u_bank", "surface": "bank"}],
                "candidate_regions": [],
                "candidate_containers": [],
                "candidate_markers": [],
                "arrangement_hints": [],
                "change_hints": [],
                "grouping_hints": [],
                "reference_hints": [],
                "uncertainty_flags": [],
            }
        )

    monkeypatch.setattr(llm_mod, "_chat", fake_chat)
    episode = Episode(
        episode_id="ep-llm-audit",
        modality=Modality.TEXT,
        raw_input="go to the bank",
    )
    runner = OrchestratorRunner(
        config=OrchestratorConfig(
            audit_base_dir=str(tmp_path),
            perception=PerceptionConfig(backend="llm", api_key="test-key"),
        )
    )

    run = runner.run_episode(episode)
    run_dir = Path(run.context.audit_dir)
    stage_audit = json.loads((run_dir / "perception.json").read_text(encoding="utf-8"))
    linked_path = Path(
        stage_audit["output"]["provenance"]["extra"]["perception_audit_path"]
    )

    assert linked_path.parent == run_dir / "perception"
    assert linked_path.exists()
    detail_audit = json.loads(linked_path.read_text(encoding="utf-8"))
    assert detail_audit["schema_version"] == 1
    assert detail_audit["stage_name"] == "perception_llm"
    assert detail_audit["output"]["attempts"][0]["raw_response"]
    assert detail_audit["output"]["graph"]["candidate_units"][0]["surface"] == "bank"
