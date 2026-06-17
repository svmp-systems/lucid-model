"""Prune low-quality concepts from an existing checkpoint without re-ingesting."""

from __future__ import annotations

import argparse
import json
from typing import Any

from lucid.runtime.paths import resolve_checkpoint
from lucid.training.checkpoint.registry import register_checkpoint
from lucid.training.checkpoint.slots import promote_to_loaded
from lucid.training.checkpoint.store import (
    CheckpointState,
    checkpoint_summary,
    load_checkpoint,
    save_checkpoint,
)
from lucid.training.ingest_quality import filter_concept_relations, retain_concepts
from lucid.training.scale_ingest import (
    build_alias_records,
    build_basin_records,
    build_trace_records,
)
from lucid.training.ingest_learning import consolidate_trace_records

_SPEECH_ALIAS_PREFIXES = (
    "alias_basic_",
    "alias_query_",
    "alias_hi",
    "alias_hello",
    "alias_thanks",
    "alias_bye",
    "alias_what_is",
    "alias_explain",
    "alias_tell_me_about",
    "alias_how_does",
    "alias_mechanism_",
    "alias_source_",
)
_SPEECH_TRACE_PREFIXES = ("t_basic_", "t_query_")
_SPEECH_TRACE_IDS = frozenset({"t9001", "t9002"})
_SPEECH_BASIN_PREFIX = "b_basic_"
_SPEECH_ALIAS_SOURCES = frozenset(
    {
        "basic_language_phrase_corpus",
        "paraphrase_query_corpus",
        "scale_ingest_mechanism",
        "general_language_bootstrap",
    }
)


def _is_speech_alias(alias: dict[str, Any]) -> bool:
    alias_id = str(alias.get("alias_id") or "")
    if any(alias_id.startswith(prefix) for prefix in _SPEECH_ALIAS_PREFIXES):
        return True
    return str(alias.get("source") or "") in _SPEECH_ALIAS_SOURCES


def _is_speech_trace(trace: dict[str, Any]) -> bool:
    trace_id = str(trace.get("trace_id") or "")
    if trace_id in _SPEECH_TRACE_IDS:
        return True
    if any(trace_id.startswith(prefix) for prefix in _SPEECH_TRACE_PREFIXES):
        return True
    summary = str(trace.get("last_update_summary") or "")
    return summary in {"scale_ingest_basic_language", "scale_ingest_paraphrase_query"}


def _is_speech_basin(basin: dict[str, Any]) -> bool:
    return str(basin.get("basin_id") or "").startswith(_SPEECH_BASIN_PREFIX)


def _trace_belongs_to_concept(trace: dict[str, Any], kept_ids: set[str]) -> bool:
    trace_id = str(trace.get("trace_id") or "")
    family = str(trace.get("trace_family") or "")
    if family in kept_ids:
        return True
    for concept_id in kept_ids:
        if trace_id == f"t_term_{concept_id}":
            return True
        if trace_id.startswith(f"t_claim_{concept_id}_"):
            return True
    return False


def _basin_belongs_to_concept(basin: dict[str, Any], kept_ids: set[str]) -> bool:
    basin_id = str(basin.get("basin_id") or "")
    family = str(basin.get("family_hint") or "")
    if family in kept_ids:
        return True
    return any(basin_id.startswith(f"b_{concept_id}_") for concept_id in kept_ids)


def _alias_belongs_to_concept(alias: dict[str, Any], kept_ids: set[str]) -> bool:
    candidates = alias.get("relation_candidates") or []
    if len(candidates) >= 2 and str(candidates[0]) == "concept":
        return str(candidates[1]) in kept_ids
    canonical = str(alias.get("canonical_concept_id") or "")
    if canonical and canonical in kept_ids:
        return True
    return False


def _prune_metadata(state: CheckpointState, *, kept_ids: set[str]) -> int:
    metadata = state.ensure_store("learned_metadata").get("objects", {})
    if not isinstance(metadata, dict):
        return 0
    kept_trace_ids = {
        str(trace.get("trace_id"))
        for trace in state.ensure_store("tracebank").get("records", [])
        if trace.get("trace_id")
    }
    kept_basin_ids = {
        str(basin.get("basin_id"))
        for basin in state.ensure_store("basin_bank").get("records", [])
        if basin.get("basin_id")
    }
    removed = 0
    for key in list(metadata.keys()):
        if key.startswith("concept:") and key.split(":", 1)[1] not in kept_ids:
            del metadata[key]
            removed += 1
        elif key.startswith("trace:") and key.split(":", 1)[1] not in kept_trace_ids:
            del metadata[key]
            removed += 1
        elif key.startswith("basin:") and key.split(":", 1)[1] not in kept_basin_ids:
            del metadata[key]
            removed += 1
    return removed


def _sanitize_concept_relations(concepts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sanitized: list[dict[str, Any]] = []
    relation_rejections: dict[str, int] = {}
    dropped = 0
    for concept in concepts:
        relations, reasons = filter_concept_relations(concept)
        for reason, count in reasons.items():
            relation_rejections[reason] = relation_rejections.get(reason, 0) + count
        if not relations:
            dropped += 1
            continue
        candidate = dict(concept)
        candidate["relations"] = relations
        sanitized.append(candidate)
    return sanitized, {"concepts_dropped_no_relations": dropped, "relation_rejections": relation_rejections}


def prune_checkpoint_concepts(
    checkpoint: str,
    *,
    pin_loaded: bool = False,
    registry_name: str | None = None,
) -> dict[str, Any]:
    root = resolve_checkpoint(checkpoint)
    state = load_checkpoint(root, create=False)
    concept_bank = state.ensure_store("concept_bank")
    concepts = list(concept_bank.get("concepts") or [])
    kept, stats = retain_concepts(concepts)
    kept, relation_stats = _sanitize_concept_relations(kept)
    stats = {**stats, **relation_stats}
    kept_ids = {str(concept["concept_id"]) for concept in kept}

    speech_traces = [
        trace for trace in state.ensure_store("tracebank").get("records", []) if _is_speech_trace(trace)
    ]
    speech_basins = [
        basin for basin in state.ensure_store("basin_bank").get("records", []) if _is_speech_basin(basin)
    ]
    speech_aliases = [
        alias for alias in state.ensure_store("relation_aliases").get("aliases", []) if _is_speech_alias(alias)
    ]

    article_traces, _deduped = consolidate_trace_records(build_trace_records(kept))
    article_basins = build_basin_records(kept)
    article_aliases = build_alias_records(kept)

    concept_bank["concepts"] = kept
    state.ensure_store("tracebank")["records"] = [*speech_traces, *article_traces]
    state.ensure_store("basin_bank")["records"] = [*speech_basins, *article_basins]

    alias_by_id: dict[str, dict[str, Any]] = {}
    for alias in [*speech_aliases, *article_aliases]:
        alias_by_id[str(alias["alias_id"])] = alias
    state.ensure_store("relation_aliases")["aliases"] = list(alias_by_id.values())

    metadata_removed = _prune_metadata(state, kept_ids=kept_ids)

    save_checkpoint(state, root, force=True, step_delta=1)
    summary = checkpoint_summary(load_checkpoint(root, create=False))
    loaded: str | None = None
    if pin_loaded:
        loaded = str(promote_to_loaded(root, label=registry_name or root.name))

    archived = register_checkpoint(
        name=registry_name or root.name,
        path=root,
        label=f"pruned {stats['concepts_after_retention']} concepts",
        command="lucid.training.checkpoint_prune",
        summary=summary,
    )

    return {
        "checkpoint": str(root),
        "archived": archived,
        "loaded": loaded,
        "retention": stats,
        "metadata_objects_removed": metadata_removed,
        "store_counts": summary["store_counts"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prune weak/junk concepts from a checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/saves/v.0.3-ai-ml")
    parser.add_argument("--pin-loaded", action="store_true")
    parser.add_argument("--registry-name", default="")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prune_checkpoint_concepts(
                args.checkpoint,
                pin_loaded=args.pin_loaded,
                registry_name=args.registry_name or None,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
