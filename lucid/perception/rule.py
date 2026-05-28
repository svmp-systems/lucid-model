"""Offline perception fallback — tokens, markers, grid cells."""

from __future__ import annotations

import re

from lucid.ir.common import Modality, UncertaintySeverity
from lucid.ir.perception import (
    CandidateMarker,
    CandidateUnit,
    ChangeHint,
    PerceptionInput,
    PerceptualEvidenceGraph,
    UncertaintyFlag,
)

_MARKERS = frozenset({"while", "which", "that", "in", "and", "later", "after", "before"})
_POLYSEMY = frozenset({"bank", "bark", "match", "spring", "safe", "vault"})


def perceive_text(inp: PerceptionInput) -> PerceptualEvidenceGraph:
    text = inp.raw_payload if isinstance(inp.raw_payload, str) else str(inp.raw_payload)
    text = text.strip()
    graph = PerceptualEvidenceGraph()
    graph.provenance.modality = Modality.TEXT
    graph.provenance.extra["backend"] = "rule"

    for m in re.finditer(r"\b[A-Za-z']+\b", text):
        word = m.group(0)
        low = word.lower()
        if low in _MARKERS:
            graph.candidate_markers.append(
                CandidateMarker(marker_id=f"m_{low}", surface=low, confidence=0.9)
            )
            continue
        uid = f"u_{low}"
        graph.candidate_units.append(
            CandidateUnit(
                unit_id=uid,
                surface=word,
                kind_hint="span",
                position_or_time=str(m.start()),
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
