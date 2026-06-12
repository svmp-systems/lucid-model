from __future__ import annotations

from lucid.training.checkpoint.store import load_checkpoint
from lucid.training.quantum_articles import train_quantum_articles


def test_quantum_article_training_writes_cost_aware_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "quantum_checkpoint"

    summary = train_quantum_articles(checkpoint)
    state = load_checkpoint(checkpoint)

    assert summary["sources"] == 3
    assert summary["concepts"] >= 5
    assert summary["operators"] >= 3
    assert summary["metadata_objects"] >= 13
    assert state.ensure_store("concept_bank")["sources"]
    assert state.ensure_store("operator_bank")["operators"]
    qubit = state.ensure_store("learned_metadata")["objects"]["concept:qubit"]
    assert qubit["heat_tier"] == "quarantine"
    assert qubit["commit_permission"] == "support_only"
