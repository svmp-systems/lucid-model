"""Paraphrased definition/mechanism questions for concept binding."""

from __future__ import annotations

import uuid
from random import Random

from lucid.ir.common import AmbiguityPolicy, Modality, TaskIntent
from lucid.ir.training import (
    Episode,
    FrameSlotTarget,
    FrameTarget,
    GoldLabels,
    GoldSpan,
    ScopeAssignment,
    TraceTarget,
)
from lucid.training.corpus.engine import AmbiguityKnob, require_valid

NAME = "chat_qa_paraphrase"
MODALITY = "text"

_TEMPLATES = [
    "what is {topic}",
    "what is a {topic}",
    "what are {topic}s",
    "tell me about {topic}",
    "explain {topic}",
    "can you explain {topic}",
    "how does {topic} work",
    "how do {topic}s work",
    "describe {topic}",
    "what does {topic} mean",
    "give me an overview of {topic}",
    "i want to know about {topic}",
]

_TOPICS: list[tuple[str, str, str]] = [
    ("transformer", "transformer", "definition_query"),
    ("attention mechanism", "attention_mechanism", "definition_query"),
    ("neural network", "neural_network", "definition_query"),
    ("deep learning", "deep_learning", "definition_query"),
    ("machine learning", "machine_learning", "definition_query"),
    ("energy based model", "energy_based_model", "definition_query"),
    ("boltzmann machine", "boltzmann_machine", "definition_query"),
    ("hopfield network", "hopfield_network", "definition_query"),
    ("backpropagation", "backpropagation", "mechanism_query"),
    ("gradient descent", "gradient_descent", "mechanism_query"),
    ("reinforcement learning", "reinforcement_learning", "definition_query"),
    ("large language model", "large_language_model", "definition_query"),
    ("embedding", "embedding", "definition_query"),
    ("self attention", "self_attention", "mechanism_query"),
    ("encoder decoder model", "encoder_decoder", "definition_query"),
    ("generative ai", "generative_ai", "definition_query"),
    ("energy based machine learning", "ebm_ml", "definition_query"),
]


def make(rng: Random, knob: AmbiguityKnob) -> Episode:
    _ = knob
    topic_surface, trace_family, frame_type = rng.choice(_TOPICS)
    template = rng.choice(_TEMPLATES)
    text = template.format(topic=topic_surface)
    span_id = "topic"
    gold = GoldLabels(
        spans=[
            GoldSpan("utterance", text, "phrase"),
            GoldSpan(span_id, topic_surface, "noun"),
        ],
        trace_activations=[
            TraceTarget(trace_family, 0.92, span_id, True),
            TraceTarget("concept_query_like", 0.88, "utterance", True),
        ],
        scope_assignments=[
            ScopeAssignment(span_id=span_id, primary_frame="concept_query"),
        ],
        frame_targets=[
            FrameTarget(
                frame_id="concept_query",
                frame_type=frame_type,
                slot_targets=[
                    FrameSlotTarget(
                        "slot_topic",
                        trace_family,
                        [span_id],
                        {"concept_like": 0.9, "query_target_like": 0.86},
                        0.9,
                    ),
                ],
                member_span_ids=[span_id],
                confidence=0.9,
            )
        ],
        ambiguity_policy=AmbiguityPolicy.ALLOW_NARROW.value,
        lucidity_target="COMMIT",
        lucidity_rationale=f"paraphrase concept query for {trace_family}",
        expected_answer="",
        validator_result=True,
    )
    episode = Episode(
        episode_id=str(uuid.uuid4()),
        modality=Modality.TEXT,
        template_id=f"chat_qa_{trace_family}",
        raw_input=text,
        gold=gold,
        validator="concept_query",
        meta={"recipe": NAME, "trace_family": trace_family, "topic": topic_surface},
        task_intent=TaskIntent.CHAT,
    )
    require_valid(episode)
    return episode
