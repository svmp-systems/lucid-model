"""Verify rendered text only states what the script allows."""

from __future__ import annotations

import re

from lucid.ir.expression import FaithfulnessReport, SentenceRef
from lucid.cognition.output.decoder.phrases import humanize
from lucid.ir.lucidity import DecoderPolicy, LucidityRenderPacket, RenderUnit, SourceRef

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "in",
        "on",
        "at",
        "to",
        "of",
        "for",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "this",
        "that",
        "it",
        "here",
        "means",
        "mean",
        "refers",
        "refer",
        "not",
        "also",
        "still",
        "possible",
        "reading",
        "readings",
        "part",
        "sentence",
        "text",
        "separate",
        "separately",
        "belongs",
        "could",
        "while",
        "with",
        "where",
        "as",
        "by",
        "earlier",
        "event",
        "describes",
        "describe",
        "merged",
        "merge",
        "should",
        "approved",
        "note",
        "scope",
        "live",
        "one",
        "another",
        "include",
        "includes",
        "force",
        "forced",
        "single",
        "fully",
        "settled",
        "though",
        "confidence",
        "limited",
        "cannot",
        "answer",
        "safely",
        "available",
        "multiple",
        "interpretations",
        "remain",
        "none",
        "committed",
    }
)


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _allowed_tokens(packet: LucidityRenderPacket) -> set[str]:
    tokens: set[str] = set()
    for unit in packet.approved_units:
        for ref in unit.source_refs:
            for word in re.findall(r"[a-zA-Z0-9_]+", ref.ref_id.lower()):
                if len(word) > 2:
                    tokens.add(word)
        for value in unit.payload.values():
            if isinstance(value, str):
                for word in re.findall(r"[a-zA-Z0-9_]+", value.lower()):
                    if len(word) > 2:
                        tokens.add(word)
                for word in re.findall(r"[a-zA-Z0-9_]+", humanize(value).lower()):
                    if len(word) > 2:
                        tokens.add(word)
            elif isinstance(value, (int, float)):
                tokens.add(str(value).lower())
    for alt in packet.preserved_alternatives:
        for key in ("narrative_hint", "basin_id", "hypothesis_id"):
            raw = alt.get(key)
            if isinstance(raw, str):
                for word in re.findall(r"[a-zA-Z0-9_]+", raw.lower()):
                    if len(word) > 2:
                        tokens.add(word)
    return tokens


def _unit_for_ids(packet: LucidityRenderPacket, unit_ids: list[str]) -> list[RenderUnit]:
    wanted = set(unit_ids)
    return [unit for unit in packet.approved_units if unit.unit_id in wanted]


def check_faithfulness(
    *,
    surface_text: str,
    sentence_refs: list[SentenceRef],
    packet: LucidityRenderPacket,
    policy: DecoderPolicy,
    structural_report: FaithfulnessReport | None = None,
) -> FaithfulnessReport:
    violations: list[str] = []
    unsupported = 0
    omitted: list[str] = []

    if re.search(r"\bis an?\s+(?:exploring|using|building|developing|creating)\b", surface_text, re.I):
        violations.append("gerund_definition_surface")

    required_ids = {unit.unit_id for unit in packet.approved_units if unit.required}
    covered_ids: set[str] = set()
    for ref in sentence_refs:
        covered_ids.update(ref.unit_ids)

    missing_required = required_ids - covered_ids
    if missing_required and packet.render_mode == "committed":
        omitted = sorted(missing_required)

    if packet.render_mode == "plural" and policy.forbid_single_answer:
        lowered = surface_text.lower()
        if "only" in lowered or "definitely" in lowered or "must be" in lowered:
            violations.append("collapsed_plural_reading")

    sentences = _split_sentences(surface_text)
    if packet.faithfulness_contract.require_source_refs_per_sentence and sentences:
        if len(sentence_refs) < len(sentences):
            violations.append("missing_sentence_refs")

    if packet.faithfulness_contract.forbid_new_entities and surface_text and packet.faithfulness_contract.require_reparse_check:
        allowed = _allowed_tokens(packet)
        for word in re.findall(r"[a-zA-Z]{4,}", surface_text.lower()):
            if word in _STOPWORDS:
                continue
            if word not in allowed and not any(word in token for token in allowed):
                # Allow basin/trace style ids embedded in text
                if re.search(rf"\b{re.escape(word)}\b", " ".join(allowed)):
                    continue
                unsupported += 1
                break

    for omission in packet.explicit_omissions:
        for forbidden in omission.forbidden_claim_refs:
            if forbidden and forbidden.lower() in surface_text.lower():
                violations.append(f"forbidden:{forbidden}")

    if structural_report is not None:
        violations.extend(structural_report.policy_violations)
        omitted.extend(structural_report.omitted_required_units)
        unsupported += structural_report.unsupported_sentence_count

    passed = unsupported == 0 and not violations and not omitted
    score = 1.0 if passed else max(0.0, 1.0 - 0.2 * (unsupported + len(violations) + len(omitted)))

    return FaithfulnessReport(
        passed=passed,
        unsupported_sentence_count=unsupported,
        omitted_required_units=omitted,
        policy_violations=violations,
        reparse_match_score=score,
    )


def collect_cited_refs(sentence_refs: list[SentenceRef]) -> list[SourceRef]:
    seen: set[tuple[str, str]] = set()
    out: list[SourceRef] = []
    for row in sentence_refs:
        for ref in row.source_refs:
            key = (ref.ref_type, ref.ref_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(ref)
    return out
