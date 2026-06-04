"""Pronoun carries across two events without scope bleed."""

from __future__ import annotations

import uuid
from random import Random

from lucid.ir.common import Modality, TaskIntent, UncertaintySeverity
from lucid.ir.perception import UncertaintyFlag
from lucid.ir.training import (
    BasinTarget,
    Episode,
    FrameSlotTarget,
    FrameTarget,
    GateDirective,
    GoldLabels,
    GoldSpan,
    ScopeAssignment,
    TraceTarget,
)
from lucid.training.generator.engine import AmbiguityKnob, require_valid

NAME = "two_events"
MODALITY = "text"

STORY_PAIRS = [
    {
        "agent": "Alex",
        "e1": ("found", "it", "near the river"),
        "e2": ("sold", "it", "at the market"),
        "themes": ["an old coin", "a carved stone", "a glass bottle"],
    },
    {
        "agent": "Sam",
        "e1": ("met", "them", "in the park"),
        "e2": ("called", "them", "after the show"),
        "themes": ["the band", "the crew", "the team"],
    },
    {
        "agent": "they",
        "e1": ("picked up", "it", "by the dock"),
        "e2": ("wrapped", "it", "for shipping"),
        "themes": ["a small package", "a sealed box", "a gift"],
    },
]


def make(rng: Random, knob: AmbiguityKnob) -> Episode:
    story = rng.choice(STORY_PAIRS)
    agent = story["agent"]
    e1_verb, e1_pron, e1_place = story["e1"]
    e2_verb, e2_pron, e2_place = story["e2"]
    theme = rng.choice(story["themes"])

    text = f"{agent} {e1_verb} {theme} {e1_place}, then later {e2_verb} {e2_pron} {e2_place}."

    clear = knob.level > 0.5
    lucidity = "COMMIT" if clear else "PRESERVE_AMBIGUITY"
    ref_weight = round(0.65 + knob.level * 0.25, 3) if clear else round(0.35 + knob.level * 0.2, 3)
    event_sep = 0.85 if clear else 0.5

    gold = GoldLabels(
        spans=[
            GoldSpan("e1_verb", e1_verb, "verb"),
            GoldSpan("theme", theme, "noun"),
            GoldSpan("e1_place", e1_place, "modifier"),
            GoldSpan("e2_verb", e2_verb, "verb"),
            GoldSpan("pronoun_e1", e1_pron, "pronoun"),
            GoldSpan("pronoun_e2", e2_pron, "pronoun"),
            GoldSpan("e2_place", e2_place, "modifier"),
        ],
        uncertainty_flags=(
            []
            if clear
            else [
                UncertaintyFlag(
                    target_id="pronoun_e2",
                    uncertainty_type="reference",
                    severity=UncertaintySeverity.MEDIUM,
                )
            ]
        ),
        trace_activations=[
            TraceTarget("coreference_chain", ref_weight, "pronoun_e1", True),
            TraceTarget("event_separation", event_sep, "e2_verb", True),
        ],
        frame_targets=[
            FrameTarget(
                frame_id="event_one",
                frame_type="event",
                slot_targets=[
                    FrameSlotTarget(
                        "slot_event_anchor",
                        "event_separation",
                        ["e1_verb"],
                        {"event_anchor_like": 0.75},
                        0.78,
                    ),
                    FrameSlotTarget(
                        "slot_shared_object",
                        "coreference_chain",
                        ["theme", "pronoun_e1"],
                        {"object_like": 0.75, "shared_referent_like": 0.75},
                        0.78,
                    ),
                    FrameSlotTarget(
                        "slot_context",
                        "event_separation",
                        ["e1_place"],
                        {"context_like": 0.65},
                        0.7,
                    ),
                ],
                confidence=0.78,
            ),
            FrameTarget(
                frame_id="event_two",
                frame_type="event",
                slot_targets=[
                    FrameSlotTarget(
                        "slot_event_anchor",
                        "event_separation",
                        ["e2_verb"],
                        {"event_anchor_like": 0.75},
                        0.74,
                    ),
                    FrameSlotTarget(
                        "slot_shared_object",
                        "coreference_chain",
                        ["pronoun_e2"],
                        {"object_like": 0.6, "shared_referent_like": 0.75},
                        0.74,
                    ),
                    FrameSlotTarget(
                        "slot_context",
                        "event_separation",
                        ["e2_place"],
                        {"context_like": 0.65},
                        0.7,
                    ),
                ],
                unresolved_slot_names=[] if clear else ["pronoun_reference"],
                confidence=0.74,
            ),
        ],
        scope_assignments=[
            ScopeAssignment("e1_verb", "event_one"),
            ScopeAssignment("theme", "event_one"),
            ScopeAssignment("e1_place", "event_one"),
            ScopeAssignment("pronoun_e1", "event_one"),
            ScopeAssignment("e2_verb", "event_two"),
            ScopeAssignment("pronoun_e2", "event_two", ["event_one"]),
            ScopeAssignment("e2_place", "event_two"),
        ],
        interference_gates=[
            GateDirective(
                gate_id="keep_events_separate",
                scope_frame_id="event_two",
                allowed_trace_ids=["coreference_chain"],
                blocked_trace_ids=["e1_place"],
            ),
        ],
        basin_families=[BasinTarget("shared_referent", "event_one", ref_weight)],
        lucidity_target=lucidity,
        lucidity_rationale=(
            "pronoun picks up theme from first event" if clear else "referent still underspecified"
        ),
        expected_answer=theme if clear else None,
        validator_result=True,
    )

    episode = Episode(
        episode_id=str(uuid.uuid4()),
        modality=Modality.TEXT,
        template_id=NAME,
        raw_input=text,
        gold=gold,
        validator="reference_resolution",
        meta={"recipe": NAME, "ambiguity_level": knob.level},
        task_intent=TaskIntent.ANSWER,
    )
    require_valid(episode)
    return episode
