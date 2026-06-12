"""Rich human-readable stage summary blocks."""

from __future__ import annotations

from lucid.audit.stage_summary import summarize_stage_output


def test_dmf_summary_explains_sparse_recall() -> None:
    summary = summarize_stage_output(
        "dmf",
        {
            "active_traces": [
                {"trace_id": "t0002", "activation": 0.35, "cluster_id": "river_location_like", "heat_tier": "hot"},
            ],
            "top_margin": 0.14,
            "coverage_score": 0.3,
            "audit_log": {
                "tracebank_size": 12,
                "selected_traces": 1,
                "candidate_traces": 1,
                "retrieval_mode": "indexed_top_k",
                "compute_limit": 128,
            },
        },
    )
    text = "\n".join(summary["narrative"])
    assert "Sparse recall" in text
    assert "12" in text
    assert "t0002" in text
    assert summary["lines"][0].startswith("active_traces:")


def test_binding_summary_lists_roles() -> None:
    summary = summarize_stage_output(
        "binding",
        {
            "candidate_frames": [
                {
                    "frame_id": "event_one",
                    "frame_type": "event",
                    "confidence": 0.84,
                    "role_assignments": {"slot_00": "found", "slot_01": "money"},
                    "slot_evidence_refs": {"slot_00": ["u_found"], "slot_01": ["u_money"]},
                    "supporting_trace_ids": ["found", "money"],
                    "unresolved_slot_names": [],
                }
            ],
            "binding_stability_score": 0.53,
        },
        stage_input={
            "perceptual_evidence_graph": {
                "candidate_units": [
                    {"unit_id": "u_found", "surface": "found"},
                    {"unit_id": "u_money", "surface": "money"},
                ]
            }
        },
    )
    text = "\n".join(summary["narrative"])
    assert "event_one" in text
    assert "found" in text
    assert "money" in text


def test_basins_summary_ranks_hypotheses() -> None:
    summary = summarize_stage_output(
        "basins",
        {
            "candidate_basin_states": [
                {
                    "basin_id": "b0001",
                    "energy": 0.76,
                    "margin_vs_next": 0.24,
                    "supporting_frame_ids": ["event_two"],
                    "supporting_trace_ids": ["kayaking", "placed"],
                    "scope_frame_ids": ["cf_event_two"],
                }
            ],
            "competition_summary": {"top_basin_id": "b0001", "top_margin": 0.24},
        },
    )
    text = "\n".join(summary["narrative"])
    assert "b0001" in text
    assert "leading" in text
    assert "kayaking" in text
