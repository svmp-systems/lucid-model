"""LLM prompts — perception only; no interpretation or answers."""

from __future__ import annotations

import json
from typing import Any

from lucid.ir.common import Modality
from lucid.ir.perception import PerceptionInput

_SYSTEM = """You are the perception stage of a cognitive pipeline.
Your ONLY job is to extract layered surface evidence from raw input.

RULES (strict):
- Output a single JSON object: PerceptualEvidenceGraph.
- List candidate_units (spans/objects), candidate_regions, candidate_markers.
- Add arrangement_hints, change_hints, grouping_hints, reference_hints when visible.
- Flag ambiguity with uncertainty_flags (polysemy, segmentation, identity) — do NOT resolve it.
- Use unit_id like u_found, u_bank — NEVER trace ids (t_money) or basin ids.
- type_hints are soft surface kinds only (noun_span, verb_span, connected_component) — no semantic senses.
- NEVER include: interpretation, meaning, task_type, rule_family, bank_sense, final_answer, trace_id.
- Prefer extra candidates over missing structure.
- confidence and salience in [0, 1].

JSON shape:
{
  "candidate_units": [{"unit_id","surface","kind_hint","type_hints","confidence","salience","position_or_time"}],
  "candidate_regions": [{"region_id","role_hint","member_unit_ids","confidence"}],
  "candidate_containers": [],
  "candidate_markers": [{"marker_id","surface","marker_type_hints","possible_target_unit_ids","confidence"}],
  "arrangement_hints": [{"hint_type","source_unit_id","target_unit_id","weight"}],
  "change_hints": [{"change_type","before_unit_id","after_unit_id","weight"}],
  "grouping_hints": [{"group_id","member_unit_ids","grouping_reason","confidence"}],
  "reference_hints": [{"source_unit_id","target_unit_id","reference_type","confidence"}],
  "uncertainty_flags": [{"target_id","uncertainty_type","severity"}],
  "provenance": {"segmentation_pass_id": "llm_v1"}
}

Respond with JSON only — no markdown fences, no commentary."""


def build_messages(inp: PerceptionInput) -> list[dict[str, str]]:
    payload: Any = inp.raw_payload
    if inp.modality == Modality.TEXT and isinstance(payload, str):
        user_body = {"modality": "text", "raw_text": payload}
    elif inp.modality == Modality.GRID:
        user_body = {"modality": "grid", "raw_grid": payload}
    else:
        user_body = {"modality": str(inp.modality.value), "raw_payload": payload}

    if inp.prior_context:
        user_body["prior_context"] = inp.prior_context
    if inp.task_intent_hint is not None:
        user_body["task_intent_hint"] = str(inp.task_intent_hint.value)

    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": json.dumps(user_body, ensure_ascii=False)},
    ]
