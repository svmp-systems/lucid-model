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
    assert summary["traces"] >= 5
    assert summary["basins"] >= 5
    assert summary["metadata_objects"] >= 23
    assert state.ensure_store("concept_bank")["sources"]
    assert state.ensure_store("operator_bank")["operators"]
    tracebank = state.ensure_store("tracebank")["records"]
    basin_bank = state.ensure_store("basin_bank")["records"]
    assert any(record["trace_id"] == "t_qubit" for record in tracebank)
    qubit_basin = next(record for record in basin_bank if record["basin_id"] == "b_qubit")
    assert qubit_basin["activation_signature"]["t_qubit"] >= 0.9
    assert qubit_basin["evidence_handles"]
    assert qubit_basin["relation_handles"]
    assert qubit_basin["source_refs"]
    assert qubit_basin["quantized_payload"]["precision"] == "uint8_sparse"
    assert qubit_basin["quantized_payload"]["relations"]
    qubit_trace = next(record for record in tracebank if record["trace_id"] == "t_qubit")
    assert qubit_trace["heat_tier"] == "warm"
    assert qubit_trace["maturity_state"] == "active"
    assert qubit_basin["heat_tier"] == "warm"
    qubit = state.ensure_store("learned_metadata")["objects"]["concept:qubit"]
    assert qubit["heat_tier"] == "warm"
    assert qubit["commit_permission"] == "normal_support"
    basin_metadata = state.ensure_store("learned_metadata")["objects"]["basin:b_qubit"]
    assert basin_metadata["precision_tier"] == "uint8_sparse"
    assert basin_metadata["quantization_candidate"] is True
    assert basin_metadata["commit_permission"] == "normal_support"
