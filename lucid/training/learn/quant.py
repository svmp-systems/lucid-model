"""Measurement helpers for safe memory-quantization experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RetrievalQualitySample:
    sample_id: str
    exact_top_ids: list[str]
    candidate_top_ids: list[str]
    exact_margin: float = 0.0
    candidate_margin: float = 0.0


@dataclass(frozen=True, slots=True)
class QuantizationMeasurementResult:
    sample_count: int
    k: int
    recall_at_k: float
    mean_margin_delta: float
    passed: bool
    failures: list[str] = field(default_factory=list)


def binary_signature(features: dict[str, float], *, threshold: float = 0.0) -> frozenset[str]:
    """Return inspectable binary feature bits for tier-0 retrieval experiments."""

    return frozenset(key for key, value in features.items() if value > threshold)


def rank_by_popcount(
    cue_bits: frozenset[str],
    records: dict[str, frozenset[str]],
    *,
    top_k: int,
) -> list[str]:
    """Rank candidate records by bit overlap without hiding the exact bits."""

    if top_k <= 0:
        return []
    ranked = sorted(
        records.items(),
        key=lambda item: (-len(cue_bits & item[1]), item[0]),
    )
    return [record_id for record_id, _bits in ranked[:top_k]]


def measure_candidate_quality(
    samples: list[RetrievalQualitySample],
    *,
    k: int,
    min_recall_at_k: float = 0.95,
    max_mean_margin_delta: float = 0.05,
) -> QuantizationMeasurementResult:
    """Measure whether an approximate candidate path preserves exact retrieval quality."""

    if k <= 0:
        raise ValueError("k must be positive")
    if not samples:
        return QuantizationMeasurementResult(
            sample_count=0,
            k=k,
            recall_at_k=0.0,
            mean_margin_delta=0.0,
            passed=False,
            failures=["no_samples"],
        )

    recall_total = 0.0
    margin_delta_total = 0.0
    failures: list[str] = []
    for sample in samples:
        exact = set(sample.exact_top_ids[:k])
        candidate = set(sample.candidate_top_ids[:k])
        recall = len(exact & candidate) / max(1, len(exact))
        margin_delta = abs(sample.exact_margin - sample.candidate_margin)
        recall_total += recall
        margin_delta_total += margin_delta
        if recall < min_recall_at_k:
            failures.append(f"{sample.sample_id}:recall={recall:.3f}")
        if margin_delta > max_mean_margin_delta:
            failures.append(f"{sample.sample_id}:margin_delta={margin_delta:.3f}")

    mean_recall = recall_total / len(samples)
    mean_margin_delta = margin_delta_total / len(samples)
    passed = (
        mean_recall >= min_recall_at_k
        and mean_margin_delta <= max_mean_margin_delta
        and not failures
    )
    return QuantizationMeasurementResult(
        sample_count=len(samples),
        k=k,
        recall_at_k=mean_recall,
        mean_margin_delta=mean_margin_delta,
        passed=passed,
        failures=failures,
    )
