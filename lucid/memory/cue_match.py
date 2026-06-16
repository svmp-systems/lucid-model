"""Token-aware cue ↔ trace-affinity matching for DMF recall."""

from __future__ import annotations

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
        "my",
        "of",
        "on",
        "she",
        "some",
        "the",
        "their",
        "to",
        "we",
        "with",
        "your",
    }
)
_CONTEXTUAL_SINGLETONS = frozenset(
    {
        "algorithm",
        "bit",
        "circuit",
        "computer",
        "gate",
        "hardware",
        "mechanic",
        "particle",
        "processor",
        "quantum",
        "state",
        "system",
    }
)


def cue_match_tokens(key: str) -> set[str]:
    return {token for token in key.split("_") if token and token not in _CONTENT_STOPWORDS}


def best_affinity_for_cue(cue_key: str, cue_affinities: dict[str, float]) -> float:
    """Exact affinity match, else salient token overlap (``money`` ↔ ``some_money``)."""

    if not cue_key:
        return 0.0
    direct = float(cue_affinities.get(cue_key, 0.0))
    best = direct
    cue_tokens = cue_match_tokens(cue_key)
    block_contextual_singleton = len(cue_tokens) == 1 and next(iter(cue_tokens), "") in _CONTEXTUAL_SINGLETONS
    for aff_key, aff_val in cue_affinities.items():
        if aff_val <= 0:
            continue
        if cue_key == aff_key:
            best = max(best, aff_val)
            continue
        if block_contextual_singleton and len(cue_match_tokens(aff_key)) > 1:
            continue
        aff_tokens = cue_match_tokens(aff_key)
        if cue_key in aff_key or aff_key in cue_key:
            if len(cue_tokens) == 1 and len(aff_tokens) > 1:
                best = max(best, aff_val * 0.35)
            elif len(aff_tokens) == 1 and len(cue_tokens) > 1:
                best = max(best, aff_val * 0.65)
            else:
                best = max(best, aff_val * 0.95)
            continue
        shared = cue_tokens & aff_tokens
        if shared:
            best = max(best, aff_val * 0.9)
    return best


def trace_matches_cue_keys(cue_affinities: dict[str, float], cue_keys: set[str]) -> bool:
    return any(best_affinity_for_cue(key, cue_affinities) > 0.0 for key in cue_keys)
