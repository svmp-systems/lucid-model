from __future__ import annotations

import json
from pathlib import Path

import pytest

from lucid.cli import main as lucid_main


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_train_list_exposes_module_and_global_commands(capsys):
    exit_code = lucid_main(["train", "list"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "dmf" in captured.out
    assert "context-op" in captured.out
    assert "lucidity" in captured.out
    assert "decoder" in captured.out
    assert "global" in captured.out


def test_dmf_training_persists_checkpoint_and_audit(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    audit_dir = tmp_path / "audit"

    exit_code = lucid_main(
        [
            "train",
            "dmf",
            "--fixture",
            "bank",
            "--checkpoint",
            str(checkpoint),
            "--audit-dir",
            str(audit_dir),
            "--steps",
            "1",
        ]
    )

    assert exit_code == 0
    manifest = _read_json(checkpoint / "manifest.json")
    tracebank = _read_json(checkpoint / "tracebank.json")
    assert manifest["training_steps"] == 1
    assert manifest["store_hashes"]["tracebank"]
    assert tracebank["records"]

    run_dir = next(audit_dir.glob("dmf_*"))
    for file_name in [
        "manifest.json",
        "governor_decision.json",
        "module_update.json",
        "before.json",
        "after.json",
        "metrics.json",
        "README.txt",
    ]:
        assert (run_dir / file_name).exists()
    assert _read_json(run_dir / "governor_decision.json")["records"][0]["action"] == "NOT_APPLICABLE"

    step_dirs = sorted(run_dir.glob("step_*"))
    assert len(step_dirs) == 1
    module_update = _read_json(step_dirs[0] / "module_update.json")
    assert module_update["module"] == "dmf"
    assert module_update["action"] == "UPDATE"
    assert (step_dirs[0] / "before.json").exists()
    assert (step_dirs[0] / "after.json").exists()
    assert (step_dirs[0] / "README.txt").exists()


@pytest.mark.parametrize(
    "module_name",
    [
        "perception",
        "cue_encoder",
        "binding",
        "context-op",
        "interference",
        "basins",
        "lucidity",
        "projector",
        "decoder",
    ],
)
def test_each_module_training_command_writes_audit(module_name: str, tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    audit_dir = tmp_path / "audit"

    exit_code = lucid_main(
        [
            "train",
            module_name,
            "--fixture",
            "phase1-mini",
            "--checkpoint",
            str(checkpoint),
            "--audit-dir",
            str(audit_dir),
            "--steps",
            "3",
        ]
    )

    assert exit_code == 0
    assert (checkpoint / "manifest.json").exists()
    step_dirs = sorted(audit_dir.glob(f"{module_name}_*/step_*"))
    assert len(step_dirs) == 3
    assert all((step_dir / "module_update.json").exists() for step_dir in step_dirs)
    assert all((step_dir / "before.json").exists() for step_dir in step_dirs)
    assert all((step_dir / "after.json").exists() for step_dir in step_dirs)


def test_global_training_uses_governor_and_promotes_responsible_modules(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    audit_dir = tmp_path / "audit"

    exit_code = lucid_main(
        [
            "train",
            "global",
            "--fixture",
            "phase1-mini",
            "--checkpoint",
            str(checkpoint),
            "--audit-dir",
            str(audit_dir),
            "--steps",
            "4",
        ]
    )

    assert exit_code == 0
    manifest = _read_json(checkpoint / "manifest.json")
    assert manifest["training_steps"] == 4
    assert manifest["store_hashes"]["tracebank"]

    run_dir = next(audit_dir.glob("global_*"))
    for file_name in [
        "manifest.json",
        "governor_decision.json",
        "module_update.json",
        "before.json",
        "after.json",
        "metrics.json",
        "README.txt",
    ]:
        assert (run_dir / file_name).exists()
    assert _read_json(run_dir / "manifest.json")["command"] == "train global"
    assert len(_read_json(run_dir / "governor_decision.json")["records"]) == 4

    decisions = sorted(run_dir.glob("step_*/governor_decision.json"))
    assert len(decisions) == 4
    first_decision = _read_json(decisions[0])
    assert first_decision["decision"]["action"] == "UPDATE"

    live_updates = sorted(run_dir.glob("step_*/live/module_update.json"))
    assert live_updates
    assert _read_json(live_updates[0])["action"] == "UPDATE"


def test_checkpoint_refuses_accidental_overwrite_without_force(tmp_path: Path, capsys):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "unrelated.txt").write_text("not a checkpoint", encoding="utf-8")

    exit_code = lucid_main(
        [
            "train",
            "dmf",
            "--fixture",
            "bank",
            "--checkpoint",
            str(checkpoint),
            "--audit-dir",
            str(tmp_path / "audit"),
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "not a Lucid checkpoint" in captured.err
