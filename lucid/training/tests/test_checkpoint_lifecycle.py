from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from lucid.training.checkpoint.cli import main as checkpoint_main
from lucid.training.checkpoint.metadata import (
    ARCHIVED,
    NORMAL_SUPPORT,
    QUARANTINE,
    SUPPORT_ONLY,
    WARM,
    archive_stale_quarantine,
    ensure_metadata,
    promote_operator_from_evidence,
    summarize_metadata_lifecycle,
)
from lucid.training.checkpoint.shards import compact_checkpoint, load_store_from_shards
from lucid.training.checkpoint.store import checkpoint_summary, empty_checkpoint, load_checkpoint, save_checkpoint


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


def test_checkpoint_compaction_writes_shard_sidecars_and_manifest(tmp_path) -> None:
    state = empty_checkpoint("compact")
    tracebank = state.ensure_store("tracebank")
    tracebank["records"] = [
        {"trace_id": f"t_{index}", "cue_affinities": {"x": 0.1}}
        for index in range(5)
    ]
    tracebank["records"].append({"trace_id": "t_3", "cue_affinities": {"x": 0.9}})
    save_checkpoint(state, tmp_path)

    summary = compact_checkpoint(tmp_path, max_items_per_shard=2, stores=["tracebank"])
    loaded = load_checkpoint(tmp_path, create=False)
    sharded = load_store_from_shards(tmp_path, "tracebank")

    assert summary["stores"]["tracebank"]["deduped_item_count"] == 1
    assert summary["stores"]["tracebank"]["shard_count"] == 3
    assert (tmp_path / "_shards" / "tracebank" / "index.json").exists()
    assert len(loaded.ensure_store("tracebank")["records"]) == 5
    assert len(sharded["records"]) == 5
    assert sharded["records"][3]["cue_affinities"]["x"] == 0.9
    assert checkpoint_summary(loaded)["checkpoint_shards"]["tracebank"]["shard_count"] == 3


def test_operator_promotion_requires_evidence_for_runtime_commit() -> None:
    state = empty_checkpoint("operator-promotion")
    operator = {"operator_id": "supporting_relation", "default_confidence": 0.84}

    metadata = promote_operator_from_evidence(
        state,
        operator,
        support_count=2,
        shadow_pass_count=1,
        trust_score=0.84,
        source_refs=["shadow_replay:001"],
    )

    assert metadata["heat_tier"] == WARM
    assert metadata["commit_permission"] == NORMAL_SUPPORT
    assert operator["heat_tier"] == WARM
    assert operator["commit_permission"] == NORMAL_SUPPORT
    assert metadata["promotion_policy"] == "learned_operator_evidence_v1"


def test_operator_promotion_keeps_contradicted_operator_in_quarantine() -> None:
    state = empty_checkpoint("operator-promotion")
    operator = {"operator_id": "overbroad_relation", "default_confidence": 0.95}

    metadata = promote_operator_from_evidence(
        state,
        operator,
        support_count=5,
        shadow_pass_count=2,
        contradiction_count=1,
        trust_score=0.95,
    )

    assert metadata["heat_tier"] == QUARANTINE
    assert metadata["commit_permission"] == SUPPORT_ONLY
    assert operator["heat_tier"] == QUARANTINE
    assert operator["commit_permission"] == SUPPORT_ONLY
