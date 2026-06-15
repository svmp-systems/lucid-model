"""Generator recipes for general conversational language."""

from __future__ import annotations

import uuid
from random import Random

from lucid.ir.common import AmbiguityPolicy, Modality, TaskIntent
from lucid.ir.training import (
    Episode,
    GoldLabels,
    GoldSpan,
    TraceTarget,
)
from lucid.training.corpus.engine import AmbiguityKnob, require_valid

NAME = "chat_social"
MODALITY = "text"

_GREETINGS = ["hi", "hello", "hey", "good morning", "good afternoon"]
_THANKS = ["thanks", "thank you", "thanks a lot"]
_FAREWELLS = ["bye", "goodbye", "see you"]
_HOW_ARE_YOU = ["how are you", "how's it going", "what's up"]
_CAPABILITY = ["what can you do", "who are you"]

_RESPONSES = {
    "greeting": "Hello.",
    "thanks": "You're welcome.",
    "farewell": "Goodbye.",
    "how_are_you": "I'm here and ready to help.",
    "capability": "I'm Lucid. I answer from audited pipeline state, not open-ended guessing.",
}


def _episode(text: str, *, template: str, kind: str) -> Episode:
    gold = GoldLabels(
        spans=[GoldSpan("utterance", text, "phrase")],
        trace_activations=[
            TraceTarget("social_speech_like", 0.95, "utterance", True),
        ],
        ambiguity_policy=AmbiguityPolicy.ALLOW_NARROW.value,
        lucidity_target="COMMIT",
        lucidity_rationale=f"general language {kind}",
        expected_answer=_RESPONSES[kind],
        validator_result=True,
    )
    episode = Episode(
        episode_id=str(uuid.uuid4()),
        modality=Modality.TEXT,
        template_id=template,
        raw_input=text,
        gold=gold,
        validator="exact_social",
        meta={"recipe": NAME, "speech_kind": kind},
        task_intent=TaskIntent.CHAT,
    )
    require_valid(episode)
    return episode


def make(rng: Random, knob: AmbiguityKnob) -> Episode:
    _ = knob
    kind = rng.choice(["greeting", "thanks", "farewell", "how_are_you", "capability"])
    if kind == "greeting":
        text = rng.choice(_GREETINGS)
        template = "chat_greeting"
    elif kind == "thanks":
        text = rng.choice(_THANKS)
        template = "chat_thanks"
    elif kind == "farewell":
        text = rng.choice(_FAREWELLS)
        template = "chat_farewell"
    elif kind == "how_are_you":
        text = rng.choice(_HOW_ARE_YOU)
        template = "chat_how_are_you"
    else:
        text = rng.choice(_CAPABILITY)
        template = "chat_capability"
    return _episode(text, template=template, kind=kind)
