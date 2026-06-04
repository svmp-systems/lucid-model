"""Instruction applies to one part only — must not leak globally."""

from __future__ import annotations

import uuid
from random import Random

from lucid.ir.common import Modality, TaskIntent
from lucid.ir.training import (
    BasinTarget,
    Episode,
    FrameSlotTarget,
    FrameTarget,
    GateDirective,
    GoldLabels,
    GoldMarker,
    GoldRegion,
    GoldSpan,
    ScopeAssignment,
    TraceTarget,
)
from lucid.training.generator.engine import AmbiguityKnob, require_valid

NAME = "scoped_instruction"
MODALITY = "text"

EXAMPLES = [
    {
        "text": "Make the second paragraph shorter, but keep the first one detailed.",
        "target": "paragraph_two",
        "scope_word": "second",
        "action": "shorter",
    },
    {
        "text": "Apply that change only to the Python version, not the JavaScript one.",
        "target": "python_version",
        "scope_word": "only",
        "action": "apply_change",
    },
    {
        "text": "Actually, revert just the last edit.",
        "target": "last_edit",
        "scope_word": "just",
        "action": "revert",
    },
    {
        "text": "Bold the title in section two, but leave section one unchanged.",
        "target": "section_two",
        "scope_word": "section two",
        "action": "bold",
    },
]


def make(rng: Random, knob: AmbiguityKnob) -> Episode:
    example = rng.choice(EXAMPLES)
    target = example["target"]
    leak_weight = round(max(0.05, 0.4 - knob.level * 0.35), 3)
    scope_weight = round(0.55 + knob.level * 0.35, 3)
    lucidity = "COMMIT" if scope_weight > 0.6 else "PRESERVE_AMBIGUITY"

    gold = GoldLabels(
        spans=[
            GoldSpan("instruction", example["action"], "instruction"),
            GoldSpan("scope", example["scope_word"], "scope"),
        ],
        markers=[
            GoldMarker("scope_marker", example["scope_word"], ["scope_limiter"]),
        ],
        regions=[
            GoldRegion(target, "target", ["instruction", "scope"]),
            GoldRegion("everything_else", "protected", []),
        ],
        trace_activations=[
            TraceTarget("scoped_instruction", scope_weight, "instruction", True),
            TraceTarget("global_leak_risk", leak_weight, "", leak_weight > 0.15),
        ],
        frame_targets=[
            FrameTarget(
                frame_id=target,
                frame_type="instruction_scope",
                slot_targets=[
                    FrameSlotTarget(
                        "slot_instruction",
                        "scoped_instruction",
                        ["instruction"],
                        {"instruction_like": 0.9},
                        scope_weight,
                    ),
                    FrameSlotTarget(
                        "slot_scope",
                        "scoped_instruction",
                        ["scope"],
                        {"scope_limiter_like": 0.85},
                        scope_weight - 0.05,
                    ),
                ],
                member_span_ids=["instruction", "scope"],
                confidence=scope_weight,
            ),
        ],
        scope_assignments=[
            ScopeAssignment("instruction", target),
            ScopeAssignment("scope", target),
        ],
        interference_gates=[
            GateDirective(
                gate_id="no_scope_leak",
                scope_frame_id=target,
                allowed_trace_ids=["scoped_instruction"],
                blocked_trace_ids=["global_leak_risk"] if leak_weight > 0.08 else [],
            ),
        ],
        basin_families=[BasinTarget("scoped_edit", target, scope_weight)],
        lucidity_target=lucidity,
        lucidity_rationale=(
            "instruction bound to scoped region only"
            if lucidity == "COMMIT"
            else "global leak risk still elevated"
        ),
        expected_answer={"action": example["action"], "target": target},
        validator_result=True,
    )

    episode = Episode(
        episode_id=str(uuid.uuid4()),
        modality=Modality.TEXT,
        template_id=NAME,
        raw_input=example["text"],
        gold=gold,
        validator="scope_consistency",
        meta={"recipe": NAME, "target": target, "ambiguity_level": knob.level},
        task_intent=TaskIntent.ANSWER,
    )
    require_valid(episode)
    return episode
