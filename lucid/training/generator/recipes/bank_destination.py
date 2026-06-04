"""Alex found cash while kayaking and later put it in the bank."""

from __future__ import annotations

import uuid
from random import Random

from lucid.ir.common import AmbiguityPolicy, Modality, TaskIntent, UncertaintySeverity
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

NAME = "bank_destination"
MODALITY = "text"

AGENTS = ["Alex", "Jamie", "she", "he", "they", "I", "Sam", "Jordan"]
FIND_VERBS = ["found", "discovered", "picked up", "came across"]
THEMES = ["the cash", "some money", "the coins", "her savings", "the funds", "the bills"]
DEPOSIT_VERBS = ["deposited", "placed", "put", "stored", "left"]
LOCATIONS = ["bank", "vault", "safe"]
OUTDOOR_CONTEXTS = [
    "while kayaking",
    "during a hike",
    "while fishing",
    "after swimming",
    "on a trail run",
    "while canoeing",
]


def make(rng: Random, knob: AmbiguityKnob) -> Episode:
    agent = rng.choice(AGENTS)
    theme = rng.choice(THEMES)
    find_verb = rng.choice(FIND_VERBS)
    deposit_verb = rng.choice(DEPOSIT_VERBS)
    location = rng.choice(LOCATIONS)
    context: str | None = None

    if knob.level > 0.3:
        context = rng.choice(OUTDOOR_CONTEXTS)
        text = (
            f"{agent} {find_verb} {theme} {context} and later "
            f"{deposit_verb} it in the {location}."
        )
        river_weight = max(0.05, 0.45 - knob.level * 0.4)
    else:
        text = f"{agent} {find_verb} {theme} and later {deposit_verb} it in the {location}."
        river_weight = 0.45

    if location in ("vault", "safe"):
        river_weight = min(river_weight, 0.15)

    money_weight = round(1.0 - river_weight, 3)
    river_weight = round(river_weight, 3)
    outdoor_weight = round(0.7 if context else 0.0, 3)

    should_commit = money_weight > 0.6
    lucidity = "COMMIT" if should_commit else "PRESERVE_AMBIGUITY"
    policy = (
        AmbiguityPolicy.ALLOW_NARROW.value
        if should_commit
        else AmbiguityPolicy.PRESERVE_PLURAL.value
    )

    gold = GoldLabels(
        spans=[
            GoldSpan("find", find_verb, "verb"),
            GoldSpan("theme", theme, "noun"),
            GoldSpan("location", location, "noun"),
            *([GoldSpan("context", context, "modifier")] if context else []),
            GoldSpan("deposit", deposit_verb, "verb"),
        ],
        uncertainty_flags=(
            [
                UncertaintyFlag(
                    target_id="location",
                    uncertainty_type="polysemy",
                    severity=UncertaintySeverity.HIGH,
                )
            ]
            if river_weight > 0.2 and location == "bank"
            else []
        ),
        trace_activations=[
            TraceTarget("financial_action_like", money_weight, "theme", True),
            TraceTarget("river_location_like", river_weight, "location", river_weight > 0.2),
            TraceTarget("outdoor_context_like", outdoor_weight, "context", False),
        ],
        ambiguity_policy=policy,
        frame_targets=[
            FrameTarget(
                frame_id="event_one",
                frame_type="event",
                slot_targets=[
                    FrameSlotTarget(
                        "slot_find",
                        "financial_action_like",
                        ["find"],
                        {"event_anchor_like": 0.8},
                        0.76,
                    ),
                    FrameSlotTarget(
                        "slot_theme",
                        "financial_action_like",
                        ["theme"],
                        {"object_like": 0.8},
                        0.76,
                    ),
                    *(
                        [
                            FrameSlotTarget(
                                "slot_context",
                                "outdoor_context_like",
                                ["context"],
                                {"context_like": 0.8},
                                0.72,
                            )
                        ]
                        if context
                        else []
                    ),
                ],
                unresolved_slot_names=[],
                confidence=0.76,
            ),
            FrameTarget(
                frame_id="event_two",
                frame_type="event",
                slot_targets=[
                    FrameSlotTarget(
                        "slot_deposit",
                        "financial_action_like",
                        ["deposit"],
                        {"event_anchor_like": 0.8},
                        0.74,
                    ),
                    FrameSlotTarget(
                        "slot_location",
                        "river_location_like",
                        ["location"],
                        {"location_like": 0.8},
                        0.74,
                    ),
                    FrameSlotTarget(
                        "slot_theme_carryover",
                        "financial_action_like",
                        ["theme"],
                        {"object_like": 0.65},
                        0.68,
                    ),
                ],
                unresolved_slot_names=["bank_sense"] if river_weight > 0.2 and location == "bank" else [],
                confidence=0.74,
            ),
        ],
        scope_assignments=[
            ScopeAssignment("find", "event_one"),
            *([ScopeAssignment("context", "event_one")] if context else []),
            ScopeAssignment("deposit", "event_two"),
            ScopeAssignment("location", "event_two"),
            ScopeAssignment("theme", "event_one", ["event_two"]),
        ],
        interference_gates=(
            [
                GateDirective(
                    gate_id="keep_outdoor_context_local",
                    scope_frame_id="event_two",
                    allowed_trace_ids=["financial_action_like", "river_location_like"],
                    blocked_trace_ids=["outdoor_context_like"],
                ),
            ]
            if context
            else [
                GateDirective(
                    gate_id="scope_bank_sense_to_deposit",
                    scope_frame_id="event_two",
                    allowed_trace_ids=["financial_action_like", "river_location_like"],
                    blocked_trace_ids=[],
                ),
            ]
        ),
        basin_families=[
            BasinTarget("financial_destination", "event_two", money_weight),
            BasinTarget("river_destination", "event_two", river_weight),
        ],
        lucidity_target=lucidity,
        lucidity_rationale=(
            "outdoor context favors money sense of location"
            if context and money_weight > 0.6
            else "bank/river sense still competing"
        ),
        expected_answer="financial" if should_commit else None,
        validator_result=True,
    )

    episode = Episode(
        episode_id=str(uuid.uuid4()),
        modality=Modality.TEXT,
        template_id=NAME,
        raw_input=text,
        gold=gold,
        validator="exact_sense",
        meta={
            "recipe": NAME,
            "agent": agent,
            "location": location,
            "has_outdoor_context": context is not None,
            "ambiguity_level": knob.level,
        },
        task_intent=TaskIntent.ANSWER,
    )
    require_valid(episode)
    return episode
