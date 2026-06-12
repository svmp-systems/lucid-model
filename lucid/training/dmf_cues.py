"""Cue-key expansion and co-frame learning helpers for DMF training."""

from __future__ import annotations

from lucid.cognition.input.cue.encoder import normalize_cue_key
from lucid.ir.training import Episode

_CONTENT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "he",
        "her",
        "his",
        "i",
        "in",
        "it",
        "later",
        "my",
        "of",
        "on",
        "she",
        "some",
        "the",
        "their",
        "them",
        "they",
        "to",
        "we",
        "while",
        "with",
        "your",
    }
)

COMPETING_TRACE_FAMILIES = (
    ("financial_action_like", "river_location_like"),
)


def surface_cue_keys(surface: str) -> set[str]:
    """Surface phrase plus salient content tokens (``some money`` → ``money``)."""

    normalized = normalize_cue_key(surface)
    keys: set[str] = set()
    if normalized:
        keys.add(normalized)
    for token in normalized.split("_"):
        if token and token not in _CONTENT_STOPWORDS:
            keys.add(token)
    return keys


def co_frame_span_ids(episode: Episode, evidence_ref: str) -> set[str]:
    """Span ids that share a scope frame with ``evidence_ref``."""

    if not evidence_ref:
        return set()

    frame_ids: set[str] = set()
    secondary_by_span: dict[str, set[str]] = {}
    for assignment in episode.gold.scope_assignments:
        secondary_by_span[assignment.span_id] = set(assignment.secondary_frames)
        if assignment.span_id == evidence_ref:
            frame_ids.add(assignment.primary_frame)
        if evidence_ref in assignment.secondary_frames:
            frame_ids.add(assignment.primary_frame)

    if not frame_ids:
        return {evidence_ref}

    co_refs: set[str] = {evidence_ref}
    for assignment in episode.gold.scope_assignments:
        if assignment.primary_frame in frame_ids:
            co_refs.add(assignment.span_id)
            co_refs.update(assignment.secondary_frames)
        if frame_ids & secondary_by_span.get(assignment.span_id, set()):
            co_refs.add(assignment.span_id)
    return co_refs


def training_cue_keys_for_target(episode: Episode, target: dict) -> set[str]:
    """All cue keys that should reinforce for one trace target."""

    span_by_id = {span.span_id: span for span in episode.gold.spans}
    family = str(target.get("trace_family") or "")
    evidence_ref = str(target.get("evidence_ref") or "").strip()
    keys: set[str] = set()
    if family:
        keys.add(normalize_cue_key(family))

    span_ids = co_frame_span_ids(episode, evidence_ref) if evidence_ref else set()
    if not span_ids and evidence_ref:
        span_ids = {evidence_ref}

    for span_id in span_ids:
        keys.add(normalize_cue_key(span_id))
        span = span_by_id.get(span_id)
        if span is not None:
            keys.update(surface_cue_keys(span.surface))

    return {key for key in keys if key}


def link_competing_trace_conflicts(
    records: list[dict],
    *,
    learning_rate: float = 0.35,
) -> int:
    """Write symmetric conflict links between known competing trace families."""

    family_to_idx = {
        str(record.get("trace_family") or ""): idx
        for idx, record in enumerate(records)
        if str(record.get("trace_family") or "")
    }
    updated = 0
    for left, right in COMPETING_TRACE_FAMILIES:
        left_idx = family_to_idx.get(left)
        right_idx = family_to_idx.get(right)
        if left_idx is None or right_idx is None:
            continue
        for idx, other_idx in ((left_idx, right_idx), (right_idx, left_idx)):
            record = records[idx]
            links = record.setdefault("conflict_links", {})
            key = str(other_idx)
            old = float(links.get(key, 0.0))
            new = min(1.0, max(old, old + learning_rate))
            if new != old:
                links[key] = new
                updated += 1
    return updated
