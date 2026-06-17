"""Social and general conversational speech for chat and short utterances."""

from __future__ import annotations

import re
from uuid import uuid4

from lucid.cognition.output.lucidity.config import LucidityConfig, normalize_task_intent
from lucid.ir.common import CommitShape, LucidityDecision
from lucid.ir.lucidity import (
    CommittedState,
    LucidityCheckResults,
    LucidityInput,
    LucidityOutput,
    RenderUnit,
    SourceRef,
)

_PUNCT_RE = re.compile(r"[^\w\s']+")
_SPACE_RE = re.compile(r"\s+")

_SOCIAL_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"^(hi|hello|hey|howdy|yo|sup)\b", re.I), "greeting", "Hello."),
    (re.compile(r"^(good morning|good afternoon|good evening)\b", re.I), "greeting", "Hello."),
    (re.compile(r"^(thanks|thank you|ty|thx)\b", re.I), "thanks", "You're welcome."),
    (re.compile(r"^(bye|goodbye|see you|see ya|later)\b", re.I), "farewell", "Goodbye."),
    (
        re.compile(
            r"\b(how are you|how're you|how you|hows it going|how's it going|whats up|what's up)\b",
            re.I,
        ),
        "how_are_you",
        "I'm here and ready to help.",
    ),
    (
        re.compile(r"\b(what can you do|who are you|what are you)\b", re.I),
        "capability",
        "I'm Lucid. I answer from audited pipeline state, not open-ended guessing.",
    ),
    (
        re.compile(
            r"^(oh\s+)?(ok|okay|cool|great|nice|got it|understood|makes sense|alright)\b",
            re.I,
        ),
        "acknowledge",
        "Okay.",
    ),
    (
        re.compile(r"^(huh|wait what|what do you mean)\b", re.I),
        "clarify",
        "What part should I clarify?",
    ),
)


def normalize_utterance(text: str) -> str:
    cleaned = _PUNCT_RE.sub(" ", text.strip().lower())
    return _SPACE_RE.sub(" ", cleaned).strip()


def utterance_from_input(inp: LucidityInput) -> str:
    raw_text = inp.perceptual_evidence_graph.provenance.extra.get("raw_text")
    if isinstance(raw_text, str) and raw_text.strip():
        return normalize_utterance(raw_text)
    surfaces = [
        unit.surface.strip()
        for unit in inp.perceptual_evidence_graph.candidate_units
        if unit.surface and unit.surface.strip()
    ]
    if surfaces:
        return normalize_utterance(" ".join(surfaces))
    return ""


_DOMAIN_HINT = re.compile(
    r"\b(colour|color|remember|quantum|entanglement|bank|episode|checkpoint|trace|basin|tell me)\b",
    re.I,
)
_MAX_SOCIAL_WORDS = 5


def classify_social_utterance(text: str) -> tuple[str, str] | None:
    normalized = normalize_utterance(text)
    if not normalized or _DOMAIN_HINT.search(normalized):
        return None
    from lucid.training.source_context import parse_concept_query

    if parse_concept_query(normalized):
        return None
    word_count = len(normalized.split())
    for pattern, kind, response in _SOCIAL_PATTERNS:
        if not pattern.search(normalized):
            continue
        if kind in {"how_are_you", "capability"} or word_count <= _MAX_SOCIAL_WORDS:
            return kind, response
    return None


def build_social_committed_state(*, kind: str, response: str, utterance: str) -> CommittedState:
    source_id = f"social_speech:{kind}"
    unit = RenderUnit(
        unit_id=f"social-{kind}",
        unit_type="claim",
        text_intent="answer",
        payload={
            "summary": response.rstrip("."),
            "speech_kind": kind,
            "utterance": utterance,
        },
        confidence=0.95,
        required=True,
        source_refs=[SourceRef(ref_type="source", ref_id=source_id, role="supports")],
    )
    return CommittedState(
        commit_id=str(uuid4()),
        commit_shape=CommitShape.SINGLE,
        primary_basin_id="social_speech",
        render_units=[unit],
        provenance_chain=[source_id],
    )


def try_social_speech_decision(inp: LucidityInput) -> LucidityOutput | None:
    task = normalize_task_intent(inp.task_intent)
    if task not in {"chat", "answer"}:
        return None

    utterance = utterance_from_input(inp)
    classified = classify_social_utterance(utterance)
    if classified is None:
        return None

    kind, response = classified
    decision = LucidityDecision.COMMIT
    checks = LucidityCheckResults()
    from lucid.cognition.output.lucidity.decide import decoder_policy_for

    policy = decoder_policy_for(decision, task_intent=inp.task_intent, checks=checks)
    policy.require_source_refs_per_sentence = True
    policy.max_sentences = 2

    return LucidityOutput(
        decision=decision,
        decoder_policy=policy,
        committed_state=build_social_committed_state(kind=kind, response=response, utterance=utterance),
        audit_notes=[
            f"lucidity:social_speech={kind}",
            f"lucidity:utterance={utterance or '(empty)'}",
            "lucidity:commit_social",
        ],
    )


def social_margin_threshold(config: LucidityConfig, task_intent: str) -> float:
    if normalize_task_intent(task_intent) == "chat":
        return 0.0
    return config.margin_threshold_answer
