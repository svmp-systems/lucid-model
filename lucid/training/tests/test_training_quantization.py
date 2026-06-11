from __future__ import annotations

import pytest

from lucid.cli import main as lucid_main
from lucid.training.learn.quant import (
    RetrievalQualitySample,
    binary_signature,
    measure_candidate_quality,
    rank_by_popcount,
)


def test_popcount_ranking_is_inspectable_and_deterministic():
    cue = binary_signature({"money": 1.0, "bank": 1.0, "kayaking": 0.0})
    records = {
        "t-finance": binary_signature({"money": 1.0, "bank": 1.0}),
        "t-outdoor": binary_signature({"kayaking": 1.0, "river": 1.0}),
        "t-bank": binary_signature({"bank": 1.0}),
    }

    assert rank_by_popcount(cue, records, top_k=2) == ["t-finance", "t-bank"]


def test_quantization_measurement_passes_only_when_recall_and_margin_hold():
    result = measure_candidate_quality(
        [
            RetrievalQualitySample(
                sample_id="s1",
                exact_top_ids=["t1", "t2", "t3"],
                candidate_top_ids=["t1", "t2", "t3"],
                exact_margin=0.12,
                candidate_margin=0.11,
            )
        ],
        k=3,
    )

    assert result.passed is True
    assert result.recall_at_k == 1.0
    assert result.mean_margin_delta == pytest.approx(0.01)


def test_quantization_measurement_fails_on_recall_loss():
    result = measure_candidate_quality(
        [
            RetrievalQualitySample(
                sample_id="s1",
                exact_top_ids=["t1", "t2", "t3"],
                candidate_top_ids=["t1", "wrong", "other"],
                exact_margin=0.2,
                candidate_margin=0.2,
            )
        ],
        k=3,
        min_recall_at_k=0.95,
    )

    assert result.passed is False
    assert result.failures == ["s1:recall=0.333"]


def test_quantization_measurement_rejects_empty_sample_set():
    result = measure_candidate_quality([], k=3)

    assert result.passed is False
    assert result.failures == ["no_samples"]


def test_lucid_quantization_smoke_command(capsys):
    exit_code = lucid_main(["quantization", "--fixture", "retrieval"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert '"passed": true' in captured.out
    assert '"ranked_ids"' in captured.out


def test_lucid_governor_smoke_command(capsys):
    exit_code = lucid_main(["governor", "--fixture", "failure"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert '"action": "UPDATE"' in captured.out
    assert '"update_regions"' in captured.out
