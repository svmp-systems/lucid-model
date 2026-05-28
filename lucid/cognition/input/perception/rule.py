"""Offline perception fallback — tokens, markers, regions, references."""

from __future__ import annotations

import re

from lucid.ir.common import Modality, UncertaintySeverity
from lucid.ir.perception import (
    ArrangementHint,
    CandidateMarker,
    CandidateRegion,
    CandidateUnit,
    ChangeHint,
    PerceptionInput,
    PerceptualEvidenceGraph,
    ReferenceHint,
    UncertaintyFlag,
)

_MARKERS = frozenset(
    {
        "a",
        "an",
        "the",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
        "am",
        "be",
        "been",
        "being",
        "in",
        "on",
        "at",
        "to",
        "of",
        "with",
        "for",
        "and",
        "or",
        "but",
        "while",
        "which",
        "that",
        "later",
        "after",
        "before",
    }
)
_DETERMINERS = frozenset(
    {"a", "an", "the", "my", "your", "his", "her", "its", "our", "their", "this", "that", "these", "those"}
)
_POLYSEMY = frozenset({"bank", "bark", "match", "spring", "safe", "vault"})
_DEPOSIT_VERBS = frozenset({"deposited", "placed", "put", "stored", "left"})
_MONEY_NOUNS = frozenset({"money", "cash", "coins", "funds", "bills", "savings"})


def perceive_text(inp: PerceptionInput) -> PerceptualEvidenceGraph:
    text = inp.raw_payload if isinstance(inp.raw_payload, str) else str(inp.raw_payload)
    text = text.strip()
    graph = PerceptualEvidenceGraph()
    graph.provenance.modality = Modality.TEXT
    graph.provenance.extra["backend"] = "rule"

    unit_ids: list[str] = []
    unit_by_low: dict[str, str] = {}
    tokens = [(m.group(0), m.start(), m.group(0).lower()) for m in re.finditer(r"\b[A-Za-z']+\b", text)]

    for index, (word, start, low) in enumerate(tokens):
        if low in _MARKERS:
            targets: list[str] = []
            if low in _DETERMINERS:
                for next_word, _, next_low in tokens[index + 1 :]:
                    if next_low not in _MARKERS:
                        targets.append(f"u_{next_low}")
                        break
            graph.candidate_markers.append(
                CandidateMarker(
                    marker_id=f"m_{low}",
                    surface=low,
                    possible_target_unit_ids=targets,
                    confidence=0.9,
                )
            )
            continue
        uid = f"u_{low}"
        if low not in unit_by_low:
            unit_by_low[low] = uid
            unit_ids.append(uid)
            graph.candidate_units.append(
                CandidateUnit(
                    unit_id=uid,
                    surface=word,
                    kind_hint="span",
                    position_or_time=str(start),
                    confidence=0.9,
                )
            )
        if low in _POLYSEMY:
            graph.uncertainty_flags.append(
                UncertaintyFlag(
                    target_id=uid,
                    uncertainty_type="polysemy",
                    severity=UncertaintySeverity.MEDIUM,
                )
            )

    if unit_ids:
        mid = max(1, len(unit_ids) // 2)
        graph.candidate_regions.append(
            CandidateRegion(
                region_id="r_main",
                role_hint="main_clause",
                member_unit_ids=unit_ids[:mid],
                confidence=0.75,
            )
        )
        if any(m.surface in ("which", "that", "while") for m in graph.candidate_markers):
            graph.candidate_regions.append(
                CandidateRegion(
                    region_id="r_sub",
                    role_hint="relative_clause",
                    member_unit_ids=unit_ids[mid:],
                    confidence=0.7,
                )
            )

    placed_uid = next((unit_by_low[v] for v in _DEPOSIT_VERBS if v in unit_by_low), None)
    money_uid = next((unit_by_low[v] for v in _MONEY_NOUNS if v in unit_by_low), None)
    if placed_uid and money_uid:
        graph.reference_hints.append(
            ReferenceHint(
                source_unit_id=placed_uid,
                target_unit_id=money_uid,
                reference_type="object_carryover",
                confidence=0.72,
            )
        )

    if "m_while" in {m.marker_id for m in graph.candidate_markers} and unit_ids:
        graph.arrangement_hints.append(
            ArrangementHint(
                hint_type="temporal_subordinate",
                source_unit_id="m_while",
                target_unit_id=unit_ids[0],
                weight=0.8,
            )
        )

    return graph


def _nonzero_cells(grid: list[list[int]]) -> list[tuple[int, int, int]]:
    return [(r, c, v) for r, row in enumerate(grid) for c, v in enumerate(row) if v != 0]


def perceive_grid(inp: PerceptionInput) -> PerceptualEvidenceGraph:
    graph = PerceptualEvidenceGraph()
    graph.provenance.modality = Modality.GRID
    graph.provenance.extra["backend"] = "rule"

    payload = inp.raw_payload
    if not isinstance(payload, dict):
        return graph

    in_grid = payload.get("input") or payload.get("input_grid") or []
    out_grid = payload.get("output") or payload.get("output_grid") or []
    in_cells = _nonzero_cells(in_grid) if in_grid else []
    out_cells = _nonzero_cells(out_grid) if out_grid else []

    for i, (r, c, color) in enumerate(in_cells):
        graph.candidate_units.append(
            CandidateUnit(
                unit_id=f"u_in_{i}",
                surface=f"({r},{c})",
                kind_hint="cell",
                feature_signature=str(color),
                confidence=0.95,
            )
        )
    for i, (r, c, color) in enumerate(out_cells):
        graph.candidate_units.append(
            CandidateUnit(
                unit_id=f"u_out_{i}",
                surface=f"({r},{c})",
                kind_hint="cell",
                feature_signature=str(color),
                confidence=0.95,
            )
        )

    if len(in_cells) == 1 and len(out_cells) == 1:
        moved = in_cells[0][:2] != out_cells[0][:2]
        graph.change_hints.append(
            ChangeHint(
                change_type="position_shift" if moved else "unchanged",
                before_unit_id="u_in_0",
                after_unit_id="u_out_0",
                weight=0.9,
            )
        )
    return graph
