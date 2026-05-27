from __future__ import annotations

from pathlib import Path

from lucid.ir.common import Modality
from lucid.ir.training import Episode, GoldLabels
from lucid.orchestrator.runner import OrchestratorConfig, OrchestratorRunner


def test_orchestrator_runs_and_writes_audit(tmp_path: Path) -> None:
    episode = Episode(
        episode_id="ep-1",
        modality=Modality.TEXT,
        raw_input="go to the bank",
        gold=GoldLabels(lucidity_target="PRESERVE_AMBIGUITY", expected_answer="(test)"),
        seed=1,
    )
    runner = OrchestratorRunner(config=OrchestratorConfig(audit_base_dir=str(tmp_path)))
    run = runner.run_episode(episode)

    # Sanity: stage ordering exists and decoder produced output.
    stage_names = [r.stage_name for r in run.stage_results]
    assert stage_names[:3] == ["perception", "cue_encoder", "dmf"]
    assert run.decoder_output is not None

    # Audit run folder should exist with a manifest.
    audit_dir = Path(run.context.audit_dir)
    assert audit_dir.exists()
    assert (audit_dir / "manifest.json").exists()

