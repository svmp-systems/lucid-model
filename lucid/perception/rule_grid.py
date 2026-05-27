"""Deterministic grid perception — connected cells, change hints."""

from __future__ import annotations

from typing import Any

from lucid.ir.common import Modality
from lucid.ir.perception import (
    CandidateRegion,
    CandidateUnit,
    ChangeHint,
    GroupingHint,
    PerceptionInput,
    PerceptualEvidenceGraph,
)

from lucid.perception.validator import merge_provenance


def _cells(grid: list[list[int]]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val != 0:
                out.append((r, c, val))
    return out


class RuleGridPerceptionAdapter:
    adapter_id = "rule_grid_v1"

    def perceive(self, inp: PerceptionInput, *, context: object = None) -> PerceptualEvidenceGraph:
        payload = inp.raw_payload
        graph = PerceptualEvidenceGraph()
        graph.candidate_regions.append(
            CandidateRegion(region_id="r_canvas", role_hint="background", member_unit_ids=[], confidence=0.9)
        )

        if not isinstance(payload, dict):
            merge_provenance(
                graph,
                adapter_version="rule_grid_v1",
                segmentation_pass_id=self.adapter_id,
                extra={"backend": "rule", "note": "non-dict grid payload"},
            )
            graph.provenance.modality = Modality.GRID
            return graph

        inp_grid: list[list[int]] = payload.get("input") or payload.get("input_grid") or []
        out_grid: list[list[int]] = payload.get("output") or payload.get("output_grid") or []

        in_cells = _cells(inp_grid) if inp_grid else []
        out_cells = _cells(out_grid) if out_grid else []

        for i, (r, c, color) in enumerate(in_cells):
            uid = f"u_in_{i}"
            graph.candidate_units.append(
                CandidateUnit(
                    unit_id=uid,
                    surface=f"cell({r},{c})",
                    kind_hint="connected_component",
                    type_hints=["grid_cell"],
                    feature_signature=str(color),
                    position_or_time=f"{r},{c}",
                    confidence=0.96,
                    salience=0.9,
                )
            )

        for i, (r, c, color) in enumerate(out_cells):
            uid = f"u_out_{i}"
            graph.candidate_units.append(
                CandidateUnit(
                    unit_id=uid,
                    surface=f"cell({r},{c})",
                    kind_hint="connected_component",
                    type_hints=["grid_cell"],
                    feature_signature=str(color),
                    position_or_time=f"{r},{c}",
                    confidence=0.96,
                    salience=0.9,
                )
            )

        if in_cells and out_cells and len(in_cells) == 1 and len(out_cells) == 1:
            ir, ic, icol = in_cells[0]
            orow, oc, ocol = out_cells[0]
            graph.change_hints.append(
                ChangeHint(
                    change_type="position_shift" if (ir, ic) != (orow, oc) else "unchanged",
                    before_unit_id="u_in_0",
                    after_unit_id="u_out_0",
                    weight=0.88,
                    details={"color_preserved": icol == ocol},
                )
            )
            if icol == ocol:
                graph.change_hints.append(
                    ChangeHint(
                        change_type="color_preserved",
                        before_unit_id="u_in_0",
                        after_unit_id="u_out_0",
                        weight=0.93,
                    )
                )

        if inp_grid and out_grid:
            graph.grouping_hints.append(
                GroupingHint(
                    group_id="example_pair",
                    member_unit_ids=["u_in_0", "u_out_0"] if in_cells and out_cells else [],
                    grouping_reason="example_pair",
                    confidence=1.0,
                )
            )

        merge_provenance(
            graph,
            adapter_version="rule_grid_v1",
            segmentation_pass_id=self.adapter_id,
            extra={"backend": "rule", "in_cells": len(in_cells), "out_cells": len(out_cells)},
        )
        graph.provenance.modality = Modality.GRID
        return graph
