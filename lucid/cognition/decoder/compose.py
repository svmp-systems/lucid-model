"""Plan what to say once — merge duplicate script lines before wording."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from lucid.ir.lucidity import LucidityRenderPacket, RenderUnit


@dataclass(slots=True)
class ComposeBullet:
    unit_ids: list[str] = field(default_factory=list)
    unit_type: str = "claim"
    text_intent: str = "answer"
    payload: dict = field(default_factory=dict)
    required: bool = True


@dataclass(slots=True)
class ComposePlan:
    bullets: list[ComposeBullet] = field(default_factory=list)
    preserved_alternatives: list[dict] = field(default_factory=list)


def _payload_key(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def _unit_priority(unit: RenderUnit) -> tuple[int, int]:
    order = {
        "claim": 0,
        "frame_summary": 1,
        "alternative": 2,
        "caveat": 3,
        "artifact": 0,
        "action": 0,
    }
    intent_order = {"answer": 0, "reason": 1, "caveat": 2, "next_step": 3, "refusal": 0}
    return (order.get(unit.unit_type, 5), intent_order.get(unit.text_intent, 5))


def build_compose_plan(packet: LucidityRenderPacket) -> ComposePlan:
    """Merge units with identical payload; keep required over optional duplicates."""
    merged: dict[str, ComposeBullet] = {}
    for unit in sorted(packet.approved_units, key=_unit_priority):
        key = f"{unit.unit_type}:{_payload_key(unit.payload)}"
        if key in merged:
            bullet = merged[key]
            bullet.unit_ids.append(unit.unit_id)
            bullet.required = bullet.required or unit.required
            continue
        merged[key] = ComposeBullet(
            unit_ids=[unit.unit_id],
            unit_type=unit.unit_type,
            text_intent=unit.text_intent,
            payload=dict(unit.payload),
            required=unit.required,
        )

    bullets = list(merged.values())
    required = [bullet for bullet in bullets if bullet.required]
    optional = [bullet for bullet in bullets if not bullet.required]
    max_n = packet.render_constraints.max_sentences or 4
    if max_n > 0:
        bullets = required + optional[: max(0, max_n - len(required))]

    return ComposePlan(
        bullets=bullets,
        preserved_alternatives=list(packet.preserved_alternatives),
    )
