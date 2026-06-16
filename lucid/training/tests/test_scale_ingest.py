from __future__ import annotations

from lucid.training.scale_ingest import classify_relation


def test_while_uses_sentence_is_contrast_not_mechanism() -> None:
    relation, target = classify_relation(
        "While quantum computing does use binary code, qubits process information differently from classical computers.",
        "quantum computing",
    )

    assert relation == "contrast"
    assert target == (
        "While quantum computing does use binary code, qubits process information "
        "differently from classical computers"
    )

