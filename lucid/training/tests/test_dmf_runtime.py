from __future__ import annotations

import json
from pathlib import Path

from lucid.cli import main as lucid_main
from lucid.ir.common import ComputePolicy, MaturityState
from lucid.ir.cue import CueCloud, TraceActivationRequest
from lucid.ir.dmf import DmfInput
from lucid.cognition.memory.dmf import DmfTraceRecord, DynamicMemoryField
from lucid.training.learn.dmf import apply_lucidity_trace_feedback, learn_from_episode


def test_dmf_returns_sparse_activation_and_clusters():
    dmf = DynamicMemoryField(tracebank=[DmfTraceRecord(), DmfTraceRecord(), DmfTraceRecord()])
    dmf.tracebank[0].cluster_id = "c_event"
    dmf.tracebank[1].cluster_id = "c_event"
    dmf.tracebank[2].cluster_id = "c_outdoor"
    learn_from_episode(
        dmf,
        CueCloud(
            primitive_trace_activations=[
                TraceActivationRequest(trace_id="money", weight=0.9),
                TraceActivationRequest(trace_id="find", weight=0.8),
            ]
        ),
        winning_trace_indices=[0],
    )
    learn_from_episode(
        dmf,
        CueCloud(
            primitive_trace_activations=[
                TraceActivationRequest(trace_id="bank", weight=0.9),
                TraceActivationRequest(trace_id="money", weight=0.7),
            ]
        ),
        winning_trace_indices=[1],
    )
    learn_from_episode(
        dmf,
        CueCloud(
            primitive_trace_activations=[
                TraceActivationRequest(trace_id="kayak", weight=0.9),
                TraceActivationRequest(trace_id="river", weight=0.8),
            ]
        ),
        winning_trace_indices=[2],
    )
    cue = CueCloud(
        primitive_trace_activations=[
            TraceActivationRequest(trace_id="money", weight=0.9),
            TraceActivationRequest(trace_id="bank", weight=0.8),
        ]
    )
    out = dmf.run(
        DmfInput(cue_cloud=cue, compute_policy=ComputePolicy(max_active_traces=2))
    )
    assert len(out.active_traces) == 2
    assert out.trace_clusters
    assert out.coverage_score > 0.0


def test_dmf_emits_novelty_when_cues_not_known():
    dmf = DynamicMemoryField(tracebank=[DmfTraceRecord(trace_id="")])
    learn_from_episode(
        dmf,
        CueCloud(
            primitive_trace_activations=[
                TraceActivationRequest(trace_id="known", weight=0.9),
            ]
        ),
        winning_trace_indices=[0],
    )
    cue = CueCloud(
        primitive_trace_activations=[
            TraceActivationRequest(trace_id="unknown_key", weight=0.9),
        ]
    )
    out = dmf.run(DmfInput(cue_cloud=cue))
    assert out.novelty_signals
    assert out.novelty_signals[0].suggested_action in {
        "spawn_provisional",
        "widen_search",
    }


def test_dmf_prior_active_traces_can_increase_score():
    dmf = DynamicMemoryField(
        tracebank=[
            DmfTraceRecord(trace_id="t-prev", activation_bias=0.01),
            DmfTraceRecord(trace_id="t-other", activation_bias=0.01),
        ]
    )
    learn_from_episode(
        dmf,
        CueCloud(
            primitive_trace_activations=[
                TraceActivationRequest(trace_id="weak", weight=0.8),
            ]
        ),
        winning_trace_indices=[0, 1],
    )
    cue = CueCloud(
        primitive_trace_activations=[
            TraceActivationRequest(trace_id="weak", weight=0.2),
        ]
    )
    out = dmf.run(
        DmfInput(
            cue_cloud=cue,
            prior_active_trace_ids=["t-prev"],
            compute_policy=ComputePolicy(max_active_traces=2),
        )
    )
    assert out.active_traces[0].trace_id == "t-prev"


def test_dmf_conflict_inhibition_reduces_adjusted_activation():
    dmf = DynamicMemoryField(
        tracebank=[
            DmfTraceRecord(trace_id="t-a", activation_bias=0.2, conflict_links={1: 0.9}),
            DmfTraceRecord(trace_id="t-b", activation_bias=0.8),
        ]
    )
    learn_from_episode(
        dmf,
        CueCloud(
            primitive_trace_activations=[
                TraceActivationRequest(trace_id="x", weight=0.9),
            ]
        ),
        winning_trace_indices=[0, 1],
    )
    cue = CueCloud(
        primitive_trace_activations=[
            TraceActivationRequest(trace_id="x", weight=0.9),
        ]
    )
    out = dmf.run(
        DmfInput(cue_cloud=cue, compute_policy=ComputePolicy(max_active_traces=2))
    )
    adj_a = out.adjusted_activations["t-a"]
    adj_b = out.adjusted_activations["t-b"]
    assert adj_b >= adj_a


def test_dmf_training_spawns_one_empty_id_trace_per_new_cue():
    dmf = DynamicMemoryField()
    updated = learn_from_episode(
        dmf,
        CueCloud(
            primitive_trace_activations=[
                TraceActivationRequest(trace_id="red", weight=0.8),
                TraceActivationRequest(trace_id="square", weight=0.6),
            ]
        ),
    )

    assert updated == [0, 1]
    assert [trace.trace_id for trace in dmf.tracebank] == ["", ""]
    assert dmf.tracebank[0].cue_affinities == {"red": 0.8}
    assert dmf.tracebank[1].cue_affinities == {"square": 0.6}
    assert any(event.event_type == "spawn_provisional_trace" for event in dmf.audit_events)


def test_dmf_training_reuses_recurring_cue_trace_instead_of_duplication():
    dmf = DynamicMemoryField()
    first = learn_from_episode(
        dmf,
        CueCloud(
            primitive_trace_activations=[
                TraceActivationRequest(trace_id="bank", weight=0.5),
            ]
        ),
    )
    second = learn_from_episode(
        dmf,
        CueCloud(
            primitive_trace_activations=[
                TraceActivationRequest(trace_id="bank", weight=0.5),
            ]
        ),
    )

    assert first == [0]
    assert second == [0]
    assert len(dmf.tracebank) == 1
    assert dmf.tracebank[0].trace_id == ""
    assert dmf.tracebank[0].cue_affinities["bank"] > 0.5


def test_dmf_coactivation_links_are_learned_between_episode_traces():
    dmf = DynamicMemoryField()
    learn_from_episode(
        dmf,
        CueCloud(
            primitive_trace_activations=[
                TraceActivationRequest(trace_id="money", weight=0.8),
                TraceActivationRequest(trace_id="bank", weight=0.7),
            ]
        ),
    )

    assert dmf.tracebank[0].coactivation_links[1] > 0.0
    assert dmf.tracebank[1].coactivation_links[0] > 0.0
    assert any(event.event_type == "link_coactivation" for event in dmf.audit_events)


def test_dmf_lucidity_feedback_promotes_anonymous_trace_to_learned_id():
    dmf = DynamicMemoryField()
    learn_from_episode(
        dmf,
        CueCloud(
            primitive_trace_activations=[
                TraceActivationRequest(trace_id="money", weight=0.8),
            ]
        ),
    )

    assert apply_lucidity_trace_feedback(
        dmf,
        [0],
        passed_lucidity=True,
        promotion_threshold=2,
    ) == []
    promoted = apply_lucidity_trace_feedback(
        dmf,
        [0],
        passed_lucidity=True,
        promotion_threshold=2,
    )

    assert promoted == [0]
    assert dmf.tracebank[0].trace_id == "t0001"
    assert dmf.tracebank[0].maturity_state == MaturityState.ACTIVE.value
    assert dmf.audit_events[-1].event_type == "promote_trace"


def test_dmf_failed_lucidity_keeps_trace_quarantined():
    dmf = DynamicMemoryField()
    learn_from_episode(
        dmf,
        CueCloud(
            primitive_trace_activations=[
                TraceActivationRequest(trace_id="noisy", weight=0.8),
            ]
        ),
    )
    apply_lucidity_trace_feedback(
        dmf,
        [0],
        passed_lucidity=False,
        failure_quarantine_threshold=1,
    )

    assert dmf.tracebank[0].trace_id == ""
    assert dmf.tracebank[0].maturity_state == MaturityState.PROVISIONAL.value
    assert dmf.audit_events[-1].event_type == "quarantine_trace"


def test_dmf_output_audit_counts_lifecycle_state():
    dmf = DynamicMemoryField(
        tracebank=[
            DmfTraceRecord(trace_id="", cue_affinities={"new": 1.0}),
            DmfTraceRecord(
                trace_id="t0001",
                cue_affinities={"known": 1.0},
                maturity_state=MaturityState.ACTIVE.value,
            ),
        ]
    )
    out = dmf.run(
        DmfInput(
            cue_cloud=CueCloud(
                primitive_trace_activations=[
                    TraceActivationRequest(trace_id="known", weight=0.9),
                ]
            )
        )
    )

    assert out.audit_log["provisional_traces"] == 1
    assert out.audit_log["active_or_better_traces"] == 1


def test_dmf_phrase_affinity_matches_head_word_cue():
    dmf = DynamicMemoryField(
        tracebank=[
            DmfTraceRecord(
                trace_id="t-finance",
                cue_affinities={"some_money": 0.9, "financial_action_like": 0.8},
            ),
            DmfTraceRecord(
                trace_id="t-outdoor",
                cue_affinities={"while_kayaking": 0.85, "outdoor_context_like": 0.7},
            ),
        ]
    )

    out = dmf.run(
        DmfInput(
            cue_cloud=CueCloud(
                primitive_trace_activations=[
                    TraceActivationRequest(trace_id="money", weight=0.9),
                    TraceActivationRequest(trace_id="kayaking", weight=0.8),
                ]
            ),
            compute_policy=ComputePolicy(max_active_traces=4),
        )
    )

    active = {trace.trace_id for trace in out.active_traces}
    assert active == {"t-finance", "t-outdoor"}
    assert out.coverage_score >= 1.0


def test_dmf_retrieval_uses_sparse_candidate_index():
    tracebank = [
        DmfTraceRecord(trace_id=f"t-noise-{idx}", activation_bias=1.0)
        for idx in range(20)
    ]
    tracebank.append(
        DmfTraceRecord(trace_id="t-match", cue_affinities={"target": 0.8})
    )
    dmf = DynamicMemoryField(tracebank=tracebank)

    out = dmf.run(
        DmfInput(
            cue_cloud=CueCloud(
                primitive_trace_activations=[
                    TraceActivationRequest(trace_id="target", weight=0.9),
                ]
            ),
            compute_policy=ComputePolicy(max_active_traces=3),
        )
    )

    assert [trace.trace_id for trace in out.active_traces] == ["t-match"]
    assert out.audit_log["candidate_traces"] == 1
    assert out.audit_log["retrieval_mode"] == "indexed_top_k"


def test_dmf_winner_learning_reinforces_only_relevant_existing_cues():
    dmf = DynamicMemoryField(
        tracebank=[
            DmfTraceRecord(trace_id="t-finance", cue_affinities={"money": 0.5}),
        ]
    )

    learn_from_episode(
        dmf,
        CueCloud(
            primitive_trace_activations=[
                TraceActivationRequest(trace_id="money", weight=0.8),
                TraceActivationRequest(trace_id="kayaking", weight=0.7),
            ]
        ),
        winning_trace_indices=[0],
    )

    assert dmf.tracebank[0].cue_affinities["money"] > 0.5
    assert "kayaking" not in dmf.tracebank[0].cue_affinities
    assert dmf.audit_events[-1].cue_keys == ["money"]


def test_dmf_learning_writes_durable_audit(tmp_path: Path):
    dmf = DynamicMemoryField(audit_base_dir=tmp_path / "dmf-audit")

    learn_from_episode(
        dmf,
        CueCloud(
            primitive_trace_activations=[
                TraceActivationRequest(trace_id="novel", weight=0.8),
            ]
        ),
    )

    audit_files = list((tmp_path / "dmf-audit" / "spawn_provisional_trace").glob("*.json"))
    assert len(audit_files) == 1
    record = json.loads(audit_files[0].read_text(encoding="utf-8"))
    assert record["event"]["event_type"] == "spawn_provisional_trace"
    assert record["event"]["after_hash"]
    assert record["trace_after"]["cue_affinities"] == {"novel": 0.8}
    assert record["summary"]["headline"]


def test_lucid_dmf_smoke_command(tmp_path: Path, capsys):
    exit_code = lucid_main(
        [
            "dmf",
            "--fixture",
            "bank",
            "--max-active",
            "2",
            "--audit-dir",
            str(tmp_path / "audit"),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["active_traces"]) == 2
    assert payload["audit_log"]["retrieval_mode"] == "indexed_top_k"
