"""Training orchestrator loop on the real pipeline."""

from __future__ import annotations

from pathlib import Path

from lucid.cli import main as lucid_main


def test_train_loop_smoke(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    audit_dir = tmp_path / "audit"
    exit_code = lucid_main(
        [
            "train",
            "loop",
            "--fixture",
            "phase1-mini",
            "--checkpoint",
            str(checkpoint),
            "--audit-dir",
            str(audit_dir),
            "--steps",
            "2",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert not (checkpoint / "manifest.json").exists()
