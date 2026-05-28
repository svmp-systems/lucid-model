from __future__ import annotations

import json
from pathlib import Path

import pytest

from lucid.ir.common import DecoderMode, LucidityDecision, Modality
from lucid.ir.lucidity import DecoderPolicy, LucidityOutput, SearchDirectives
from lucid.ir.training import Episode, GoldLabels
from lucid.orchestrator.cli import main as run_cli
from lucid.perception import PerceptionConfig
from lucid.orchestrator.runner import OrchestratorConfig, OrchestratorRunner
from lucid.orchestrator.stages import FunctionStage
from lucid.orchestrator.stub_stages import build_default_stage_fns


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

    run_dirs = list((tmp_path / "runs").iterdir())
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

    exit_code = run_cli(
        [str(episode_path), "--audit-dir", str(tmp_path / "audit"), "--perception", "rule"]
    )

    assert exit_code == 0
    assert list((tmp_path / "audit" / "runs").iterdir())

