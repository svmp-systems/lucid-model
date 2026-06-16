from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from lucid.training.checkpoint.cli import main as checkpoint_main
from lucid.training.checkpoint.metadata import (
    ARCHIVED,
    QUARANTINE,
    archive_stale_quarantine,
    ensure_metadata,
    summarize_metadata_lifecycle,
)
from lucid.training.checkpoint.store import empty_checkpoint, save_checkpoint


def test_archive_stale_quarantine_only_archives_unsupported_objects() -> None:
    state = empty_checkpoint("lifecycle")
    now = datetime(2026, 6, 16, tzinfo=timezone.utc)
    old = (now - timedelta(days=45)).isoformat()

    stale = ensure_metadata(state, "trace:stale", "trace")
    stale["created_at"] = old
    stale["updated_at"] = old

    supported = ensure_metadata(state, "trace:supported", "trace")
    supported["created_at"] = old
    supported["updated_at"] = old
    supported["support_count"] = 1

    archived = archive_stale_quarantine(state, max_age_days=30, now=now)

    assert [item["object_id"] for item in archived] == ["trace:stale"]
    objects = state.ensure_store("learned_metadata")["objects"]
    assert objects["trace:stale"]["heat_tier"] == ARCHIVED
    assert objects["trace:supported"]["heat_tier"] == QUARANTINE


def test_metadata_lifecycle_summary_counts_heat_and_stale_candidates() -> None:
    state = empty_checkpoint("lifecycle")
    now = datetime(2026, 6, 16, tzinfo=timezone.utc)
    old = (now - timedelta(days=45)).isoformat()
    record = ensure_metadata(state, "trace:stale", "trace")
    record["created_at"] = old
    record["updated_at"] = old
    record["quantization_candidate"] = True

    summary = summarize_metadata_lifecycle(state, stale_quarantine_days=30, now=now)

    assert summary["objects"] == 1
    assert summary["heat_tiers"] == {QUARANTINE: 1}
    assert summary["object_types"] == {"trace": 1}
    assert summary["stale_quarantine_candidates"] == 1
    assert summary["quantization_candidates"] == 1


def test_checkpoint_save_skips_unchanged_store_rewrites(tmp_path) -> None:
    state = empty_checkpoint("scale-metrics")
    save_checkpoint(state, tmp_path)
    first_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["checkpoint_scale_metrics"]["changed_store_count"] > 0

    save_checkpoint(state, tmp_path)
    second_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    metrics = second_manifest["checkpoint_scale_metrics"]
    assert metrics["changed_store_count"] == 0
    assert metrics["unchanged_store_count"] == len(second_manifest["store_files"])
    assert metrics["written_store_bytes"] == 0
    assert metrics["rewrite_policy"] == "skip_unchanged_json_stores"


def test_checkpoint_lifecycle_cli_archives_stale_quarantine(tmp_path, capsys) -> None:
    state = empty_checkpoint("lifecycle-cli")
    old = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
    stale = ensure_metadata(state, "trace:stale", "trace")
    stale["created_at"] = old
    stale["updated_at"] = old
    save_checkpoint(state, tmp_path)

    exit_code = checkpoint_main(
        [
            "lifecycle",
            "--checkpoint",
            str(tmp_path),
            "--archive-stale",
            "--max-age-days",
            "30",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["archived"][0]["object_id"] == "trace:stale"
    assert payload["after"]["heat_tiers"][ARCHIVED] == 1
