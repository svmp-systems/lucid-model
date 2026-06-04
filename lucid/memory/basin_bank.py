"""Basin bank runtime — load and index checkpoint basin records."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from lucid.training.checkpoints import STORE_FILES

_TOKEN_RE = re.compile(r"[^a-z0-9_]+")


@dataclass(slots=True)
class BasinBankRecord:
    basin_id: str
    family_hint: str = ""
    frame_affinities: dict[str, float] = field(default_factory=dict)
    cooperation_links: dict[str, float] = field(default_factory=dict)
    suppression_links: dict[str, float] = field(default_factory=dict)


def normalize_family_hint(value: str) -> str:
    """Align trainer gold, context-op pressure keys, and runtime lookup."""

    clean = _TOKEN_RE.sub("_", str(value or "").strip().lower())
    clean = "_".join(part for part in clean.split("_") if part)
    if clean.endswith("_like"):
        clean = clean[: -len("_like")]
    return clean


def basin_bank_from_checkpoint(checkpoint: str | Path) -> list[BasinBankRecord]:
    root = Path(checkpoint)
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
        affinities = {
            str(key): float(value)
            for key, value in dict(row.get("frame_affinities") or {}).items()
            if str(key)
        }
        cooperation = {
            str(key): float(value)
            for key, value in dict(row.get("cooperation_links") or {}).items()
            if str(key)
        }
        suppression = {
            str(key): float(value)
            for key, value in dict(row.get("suppression_links") or {}).items()
            if str(key)
        }
        records.append(
            BasinBankRecord(
                basin_id=basin_id,
                family_hint=str(row.get("family_hint", "")),
                frame_affinities=affinities,
                cooperation_links=cooperation,
                suppression_links=suppression,
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
    root = Path(checkpoint)
    if not root.exists():
        return BasinBank()
    return BasinBank(basin_bank_from_checkpoint(root))
