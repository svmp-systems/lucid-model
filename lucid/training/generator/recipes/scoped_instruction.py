"""Instruction applies to one part only — must not leak globally."""

from __future__ import annotations

import uuid
from random import Random

from lucid.ir.common import Modality, TaskIntent
from lucid.ir.training import (
    Episode,
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

    gold = GoldLabels(
        spans=[
            GoldSpan("instruction", example["action"], "instruction"),
            GoldSpan("scope", example["scope_word"], "scope"),
        ],
        markers=[
            GoldMarker("scope_marker", example["scope_word"], ["scope_limiter"]),
        ],
        regions=[
            GoldRegion(example["target"], "target", ["instruction"]),
            GoldRegion("everything_else", "protected", []),
        ],
        trace_activations=[
            TraceTarget("scoped_instruction", 0.9, "instruction", True),
            TraceTarget("global_leak_risk", max(0.05, 0.4 - knob.level * 0.35), "", False),
        ],
        scope_assignments=[
            ScopeAssignment("instruction", example["target"]),
            ScopeAssignment("scope", example["target"]),
        ],
        interference_gates=[
            GateDirective(
                gate_id="no_scope_leak",
                scope_frame_id=example["target"],
                allowed_trace_ids=["scoped_instruction"],
                blocked_trace_ids=["global_leak_risk"],
            ),
        ],
        lucidity_target="COMMIT",
        lucidity_rationale="instruction bound to scoped region only",
        expected_answer={"action": example["action"], "target": example["target"]},
        validator_result=True,
    )

    episode = Episode(
        episode_id=str(uuid.uuid4()),
        modality=Modality.TEXT,
        template_id=NAME,
        raw_input=example["text"],
        gold=gold,
        validator="scope_consistency",
        meta={"recipe": NAME, "target": example["target"], "ambiguity_level": knob.level},
        task_intent=TaskIntent.ANSWER,
    )
    require_valid(episode)
    return episode
