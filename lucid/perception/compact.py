"""Compact JSON view — drop empty lists and default field values."""

from __future__ import annotations

import json
from typing import Any

from lucid.ir.perception import PerceptualEvidenceGraph
from lucid.ir.serde import to_dict

from lucid.perception.schema import _LIST_KEYS


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if value == "" or value == 0 or value == 0.0:
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def compact_item(item: dict[str, Any]) -> dict[str, Any]:
    return {key: val for key, val in item.items() if not _is_empty_value(val)}


def compact_graph_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Non-empty list fields only; strip default-valued fields on each object."""
    out: dict[str, Any] = {}
    for key in _LIST_KEYS:
        raw_items = data.get(key) or []
        if not raw_items:
            continue
        items: list[Any] = []
        for item in raw_items:
            if isinstance(item, dict):
                compacted = compact_item(item)
                if compacted:
                    items.append(compacted)
            elif not _is_empty_value(item):
                items.append(item)
        if items:
            out[key] = items

    provenance = data.get("provenance")
    if isinstance(provenance, dict):
        compact_prov = compact_item(provenance)
        extra = provenance.get("extra")
        if isinstance(extra, dict):
            compact_extra = compact_item(extra)
            if compact_extra:
                compact_prov["extra"] = compact_extra
        if compact_prov:
            out["provenance"] = compact_prov

    return out


def compact_graph(graph: PerceptualEvidenceGraph) -> dict[str, Any]:
    return compact_graph_dict(to_dict(graph))


def to_compact_json(graph: PerceptualEvidenceGraph, *, indent: int = 2) -> str:
    return json.dumps(compact_graph(graph), indent=indent, ensure_ascii=False)
