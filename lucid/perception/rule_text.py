"""Deterministic text perception — offline / CI, no model."""

from __future__ import annotations

import re
from typing import Any

from lucid.ir.common import Modality, UncertaintySeverity
from lucid.ir.perception import (
    ArrangementHint,
    CandidateMarker,
    CandidateRegion,
    CandidateUnit,
    PerceptionInput,
    PerceptualEvidenceGraph,
    ReferenceHint,
    UncertaintyFlag,
)

from lucid.perception.validator import merge_provenance

_MARKER_WORDS = {
    "while": ["temporal_subordinate"],
    "which": ["relative_pronoun"],
    "that": ["complementizer", "relative_pronoun"],
    "in": ["locative_preposition"],
    "on": ["locative_preposition"],
    "and": ["coordination"],
    "later": ["temporal_adverb"],
    "then": ["temporal_adverb"],
    "after": ["temporal_subordinate"],
    "before": ["temporal_subordinate"],
}

_POLYSEMY = frozenset({"bank", "bark", "match", "spring", "file", "lead", "safe", "vault"})


def _slug(surface: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", surface.lower()).strip("_")[:32] or "span"


class RuleTextPerceptionAdapter:
    adapter_id = "rule_text_v1"

    def perceive(self, inp: PerceptionInput, *, context: object = None) -> PerceptualEvidenceGraph:
        text = inp.raw_payload if isinstance(inp.raw_payload, str) else str(inp.raw_payload)
        text = text.strip()
        graph = PerceptualEvidenceGraph()
        if not text:
            merge_provenance(
                graph,
                adapter_version="rule_text_v1",
                segmentation_pass_id=self.adapter_id,
                extra={"backend": "rule"},
            )
            graph.provenance.modality = Modality.TEXT
            return graph

        units: list[CandidateUnit] = []
        markers: list[CandidateMarker] = []
        unit_by_surface: dict[str, str] = {}

        # Multi-word phrases first (longest match).
        phrase_patterns = [
            r"\bwhile\s+\w+ing\b",
            r"\bduring\s+a\s+\w+\b",
            r"\bafter\s+\w+ing\b",
            r"\bin\s+the\s+\w+\b",
            r"\bthe\s+\w+\b",
        ]
        covered: set[tuple[int, int]] = set()

        def _overlap(start: int, end: int) -> bool:
            return any(not (end <= s or start >= e) for s, e in covered)

        for pattern in phrase_patterns:
            for m in re.finditer(pattern, text, re.I):
                if _overlap(m.start(), m.end()):
                    continue
                surface = m.group(0)
                uid = f"u_{_slug(surface)}"
                units.append(
                    CandidateUnit(
                        unit_id=uid,
                        surface=surface,
                        kind_hint="phrase_span",
                        confidence=0.9,
                        salience=0.85,
                        position_or_time=str(m.start()),
                    )
                )
                unit_by_surface[surface.lower()] = uid
                covered.add((m.start(), m.end()))

        # Single tokens (content words + verbs).
        for m in re.finditer(r"\b[A-Za-z']+\b", text):
            if _overlap(m.start(), m.end()):
                continue
            surface = m.group(0)
            low = surface.lower()
            if low in _MARKER_WORDS:
                mid = f"m_{low}"
                markers.append(
                    CandidateMarker(
                        marker_id=mid,
                        surface=low,
                        marker_type_hints=_MARKER_WORDS[low],
                        confidence=0.88,
                    )
                )
                continue
            uid = f"u_{_slug(low)}"
            if low in unit_by_surface:
                continue
            kind = "verb_span" if low.endswith(("ed", "ing")) or low in {
                "found",
                "placed",
                "put",
                "deposited",
                "discovered",
            } else "noun_span"
            units.append(
                CandidateUnit(
                    unit_id=uid,
                    surface=surface,
                    kind_hint=kind,
                    confidence=0.92,
                    salience=0.8,
                    position_or_time=str(m.start()),
                )
            )
            unit_by_surface[low] = uid

        graph.candidate_units = units
        graph.candidate_markers = markers

        # Clause-ish regions from markers.
        main_members = [u.unit_id for u in units[: max(1, len(units) // 2)]]
        rel_members = [u.unit_id for u in units[len(units) // 2 :]]
        if main_members:
            graph.candidate_regions.append(
                CandidateRegion(
                    region_id="r_main",
                    role_hint="main_clause",
                    member_unit_ids=main_members,
                    confidence=0.75,
                )
            )
        if rel_members and any(m.surface in ("which", "that") for m in markers):
            graph.candidate_regions.append(
                CandidateRegion(
                    region_id="r_sub",
                    role_hint="relative_clause",
                    member_unit_ids=rel_members,
                    confidence=0.7,
                )
            )

        # Reference: placed/it → money-like noun if present.
        money_uids = [u.unit_id for u in units if u.surface.lower() in {"money", "cash", "coins", "funds", "bills", "savings"}]
        placed_uid = unit_by_surface.get("placed") or unit_by_surface.get("put") or unit_by_surface.get("deposited")
        if money_uids and placed_uid:
            graph.reference_hints.append(
                ReferenceHint(
                    source_unit_id=placed_uid,
                    target_unit_id=money_uids[0],
                    reference_type="object_carryover",
                    confidence=0.72,
                )
            )

        flagged_targets: set[str] = set()
        for u in units:
            low = u.surface.lower()
            if low in _POLYSEMY or any(word in low.split() for word in _POLYSEMY):
                graph.uncertainty_flags.append(
                    UncertaintyFlag(
                        target_id=u.unit_id,
                        uncertainty_type="polysemy_surface_form",
                        severity=UncertaintySeverity.MEDIUM,
                    )
                )
                flagged_targets.add(u.unit_id)

        for word in _POLYSEMY:
            if word not in text.lower():
                continue
            if any(word in u.surface.lower() for u in units):
                continue
            uid = f"u_{word}"
            units.append(
                CandidateUnit(
                    unit_id=uid,
                    surface=word,
                    kind_hint="noun_span",
                    confidence=0.85,
                    salience=0.75,
                )
            )
            graph.uncertainty_flags.append(
                UncertaintyFlag(
                    target_id=uid,
                    uncertainty_type="polysemy_surface_form",
                    severity=UncertaintySeverity.MEDIUM,
                )
            )
        graph.candidate_units = units

        merge_provenance(
            graph,
            adapter_version="rule_text_v1",
            segmentation_pass_id=self.adapter_id,
            extra={"backend": "rule", "input_length": len(text)},
        )
        graph.provenance.modality = Modality.TEXT
        return graph
