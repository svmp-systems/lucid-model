"""Repair canonical definition facts in a checkpoint.

This is intentionally small and explicit: it only seeds high-confidence
definition relations for core concepts that the current ingest can miss.
"""

from __future__ import annotations

import json
from pathlib import Path

from lucid.runtime.paths import resolve_train_path


CORE_DEFINITIONS: dict[str, dict[str, object]] = {
    "artificial_intelligence": {
        "canonical_label": "artificial intelligence",
        "source_refs": ["wiki_artificial_intelligence"],
        "target": "a field of computer science focused on building intelligent systems",
        "terms": ["ai", "artificial_intelligence", "artificial intelligences"],
    },
}


def _load_json(path: Path) -> object:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _records(store: object) -> list[dict[str, object]]:
    if isinstance(store, dict):
        rows = store.setdefault("records", [])
        return rows if isinstance(rows, list) else []
    if isinstance(store, list):
        return store
    return []


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _definition_relation(seed: dict[str, object]) -> dict[str, object]:
    return {
        "confidence": 0.92,
        "relation": "type_of",
        "source_refs": list(seed["source_refs"]),
        "target": str(seed["target"]),
    }


def repair_basin_bank(checkpoint: Path) -> int:
    path = checkpoint / "basin_bank.json"
    store = _load_json(path)
    changed = 0
    rows = _records(store)

    for concept_id, seed in CORE_DEFINITIONS.items():
        basin_id = f"b_{concept_id}_definition"
        record = next((row for row in rows if row.get("basin_id") == basin_id), None)
        if record is None:
            record = {
                "basin_id": basin_id,
                "family_hint": concept_id,
                "commit_permission": "normal_support",
                "frame_affinities": {"concept": 0.78, "definition_query": 0.86, "frame_active": 0.76},
                "activation_signature": {
                    concept_id: 0.95,
                    "definition": 0.72,
                    f"t_term_{concept_id}": 0.96,
                    "type_of": 0.64,
                },
                "semantic_signature": {
                    concept_id: 0.9,
                    "definition": 0.8,
                    "type_of": 0.7,
                    "computer": 0.45,
                    "science": 0.45,
                    "system": 0.45,
                },
                "evidence_handles": [f"concept:{concept_id}"],
                "relation_handles": [f"relation:{concept_id}:core_definition"],
                "source_refs": list(seed["source_refs"]),
                "support_examples": list(seed["source_refs"]),
                "trust_score": 0.92,
                "heat_tier": "warm",
                "cooperation_links": {},
                "suppression_links": {},
                "quantized_payload": {},
            }
            rows.append(record)
            changed += 1

        payload = record.setdefault("quantized_payload", {})
        if not isinstance(payload, dict):
            payload = {}
            record["quantized_payload"] = payload
        payload.update(
            {
                "canonical_label": seed["canonical_label"],
                "concept_id": concept_id,
                "facet": "definition",
                "precision": "core_seed",
                "source_count": len(seed["source_refs"]),
                "terms": list(seed["terms"]),
            }
        )
        relations = payload.setdefault("relations", [])
        if not isinstance(relations, list):
            relations = []
            payload["relations"] = relations
        target = str(seed["target"])
        existing = next(
            (
                row
                for row in relations
                if isinstance(row, dict)
                and row.get("relation") == "type_of"
                and row.get("target") == target
            ),
            None,
        )
        if existing is None:
            relations.insert(0, _definition_relation(seed))
            changed += 1
        else:
            existing.update(_definition_relation(seed))

        record["trust_score"] = max(float(record.get("trust_score") or 0.0), 0.92)
        record["heat_tier"] = "warm"
        for key in ("source_refs", "support_examples"):
            values = record.setdefault(key, [])
            if isinstance(values, list):
                for ref in seed["source_refs"]:
                    if ref not in values:
                        values.append(ref)
        handles = record.setdefault("relation_handles", [])
        if isinstance(handles, list) and f"relation:{concept_id}:core_definition" not in handles:
            handles.insert(0, f"relation:{concept_id}:core_definition")

    if changed:
        _write_json(path, store)
    return changed


def repair_checkpoint(checkpoint: str | Path) -> int:
    root = resolve_train_path(checkpoint)
    return repair_basin_bank(root)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Checkpoint path, e.g. checkpoints/saves/v.0.3-ai-ml")
    args = parser.parse_args()
    changed = repair_checkpoint(args.checkpoint)
    print(json.dumps({"checkpoint": str(resolve_train_path(args.checkpoint)), "changes": changed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
