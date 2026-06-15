"""Chat-quality pass — grammar and flow only, no new facts."""

from __future__ import annotations

import re

from lucid.ir.expression import SentenceRef
from lucid.ir.lucidity import LucidityRenderPacket


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _sentence_stems(text: str) -> str:
    lowered = text.lower().strip()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def dedupe_repetitive_sentences(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    seen: set[str] = set()
    kept: list[str] = []
    for sentence in sentences:
        stem = _sentence_stems(sentence)
        if not stem or stem in seen:
            continue
        seen.add(stem)
        kept.append(sentence.strip())
    return " ".join(kept)


def polish_for_chat(
    text: str,
    *,
    packet: LucidityRenderPacket,
    sentence_refs: list[SentenceRef],
) -> tuple[str, list[SentenceRef]]:
    """Light deterministic polish for chat channel."""
    if not text.strip():
        return text, sentence_refs

    cleaned = dedupe_repetitive_sentences(_normalize_space(text))
    max_sentences = packet.render_constraints.max_sentences or 0
    if max_sentences > 0:
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        if len(parts) > max_sentences:
            cleaned = " ".join(parts[:max_sentences])
            sentence_refs = sentence_refs[:max_sentences]

    lowered = cleaned.lower()
    if packet.render_constraints.tone == "careful" and "might" not in lowered and "not confident" not in lowered:
        if packet.render_mode in {"uncertainty", "plural"} and cleaned.endswith("."):
            cleaned = cleaned[:-1] + ", though the reading is not fully settled."

    return cleaned, sentence_refs
