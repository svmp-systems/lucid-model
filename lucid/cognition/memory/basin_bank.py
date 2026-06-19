"""Basin bank runtime — load and index checkpoint basin records."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from lucid.runtime.paths import resolve_checkpoint
from lucid.training.checkpoint.store import STORE_FILES

_TOKEN_RE = re.compile(r"[^a-z0-9_]+")


@dataclass(slots=True)
class BasinBankRecord:
    basin_id: str
    family_hint: str = ""
    frame_affinities: dict[str, float] = field(default_factory=dict)
    activation_signature: dict[str, float] = field(default_factory=dict)
    semantic_signature: dict[str, float] = field(default_factory=dict)
    evidence_handles: list[str] = field(default_factory=list)
    relation_handles: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    trust_score: float = 0.0
    heat_tier: str = ""
    quantized_payload: dict[str, object] = field(default_factory=dict)
    cooperation_links: dict[str, float] = field(default_factory=dict)
    suppression_links: dict[str, float] = field(default_factory=dict)


def normalize_family_hint(value: str) -> str:
    """Align trainer gold, context-op pressure keys, and runtime lookup."""

    clean = _TOKEN_RE.sub("_", str(value or "").strip().lower())
    clean = "_".join(part for part in clean.split("_") if part)
    if clean.endswith("_like"):
        clean = clean[: -len("_like")]
    return clean


def _float_map(raw: object, *, normalize_keys: bool = True) -> dict[str, float]:
    rows = raw if isinstance(raw, dict) else {}
    parsed: dict[str, float] = {}
    for key, value in rows.items():
        token = normalize_family_hint(str(key)) if normalize_keys else str(key)
        if not token:
            continue
        try:
            parsed[token] = float(value)
        except (TypeError, ValueError):
            continue
    return parsed


def _string_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item)]


def basin_bank_from_checkpoint(checkpoint: str | Path) -> list[BasinBankRecord]:
    root = resolve_checkpoint(checkpoint)
    path = root / STORE_FILES["basin_bank"]
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[BasinBankRecord] = []
    for row in payload.get("records", []):
        if not isinstance(row, dict):
            continue
        basin_id = str(row.get("basin_id", "")).strip()
        if not basin_id:
            continue
        quantized_payload = (
            row.get("quantized_payload")
            if isinstance(row.get("quantized_payload"), dict)
            else {}
        )
        records.append(
            BasinBankRecord(
                basin_id=basin_id,
                family_hint=str(row.get("family_hint", "")),
                frame_affinities=_float_map(row.get("frame_affinities"), normalize_keys=False),
                activation_signature=_float_map(row.get("activation_signature")),
                semantic_signature=_float_map(row.get("semantic_signature")),
                evidence_handles=_string_list(row.get("evidence_handles")),
                relation_handles=_string_list(row.get("relation_handles")),
                source_refs=_string_list(row.get("source_refs")),
                trust_score=float(row.get("trust_score", 0.0) or 0.0),
                heat_tier=str(row.get("heat_tier", "")),
                quantized_payload=dict(quantized_payload),
                cooperation_links=_float_map(row.get("cooperation_links"), normalize_keys=False),
                suppression_links=_float_map(row.get("suppression_links"), normalize_keys=False),
            )
        )
    return records


@dataclass
class BasinBank:
    records: list[BasinBankRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_id: dict[str, BasinBankRecord] = {
            record.basin_id: record for record in self.records if record.basin_id
        }
        self._by_family: dict[str, list[BasinBankRecord]] = {}
        for record in self.records:
            key = normalize_family_hint(record.family_hint)
            if key:
                self._by_family.setdefault(key, []).append(record)

    def snapshot_id(self) -> str:
        if not self.records:
            return "basin_bank:empty"
        ids = sorted(record.basin_id for record in self.records)
        return f"basin_bank:{len(ids)}:{ids[0]}:{ids[-1]}"

    def get(self, basin_id: str) -> BasinBankRecord | None:
        return self._by_id.get(basin_id)

    def lookup_family(self, family_hint: str) -> list[BasinBankRecord]:
        return list(self._by_family.get(normalize_family_hint(family_hint), []))


def load_basin_bank(checkpoint: str | Path | None = None) -> BasinBank:
    if not checkpoint:
        return BasinBank()
    root = resolve_checkpoint(checkpoint)
    if not root.exists():
        return BasinBank()
    return BasinBank(basin_bank_from_checkpoint(root))
