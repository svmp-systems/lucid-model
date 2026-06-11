"""Round-trip JSON serialization for all IR layers."""

from __future__ import annotations

import json

from lucid.ir.basins import (
    BasinOutput,
    CandidateBasinState,
    CompetitionSummary,
)
from lucid.ir.binding import BindingOutput, CandidateFrame
from lucid.ir.common import (
    AuditEnvelope,
    CommitShape,
    ComputePolicy,
    DecoderMode,
    LucidityDecision,
    Modality,
    Provenance,
    TaskIntent,
)
from lucid.ir.context_op import ContextFrame, ContextOpOutput
from lucid.ir.cue import CueCloud, CueEncoderInput, TraceActivationRequest
from lucid.ir.dmf import ActiveTrace, DmfInput, DmfOutput
from lucid.ir.expression import DecoderOutput, FaithfulnessReport, SentenceRef
from lucid.ir.interference import InterferenceOutput, TraceTraceEdge
from lucid.ir.lucidity import (
    CommittedState,
    DecoderPolicy,
    LucidityOutput,
    LucidityRenderPacket,
    RenderUnit,
    SearchDirectives,
    SourceRef,
)
from lucid.ir.memory import BasinRecord, TraceRecord, WeightedLink
from lucid.ir.perception import (
    CandidateUnit,
    PerceptionInput,
    PerceptualEvidenceGraph,
)
from lucid.ir.pipeline import PipelineRun, RunContext, SessionState, StageName, StageResult
from lucid.ir.projector import ProjectorInput, ProjectorOutput, ProjectorRollout
from lucid.ir.serde import from_dict, from_json, to_dict, to_json
from lucid.ir.training import Episode, GoldLabels, RunLog, TrainingEpisode


def _roundtrip(obj, cls):
    restored = from_dict(to_dict(obj), cls)
    assert restored == obj
    text = to_json(obj)
    restored_json = from_json(text, cls)
    assert restored_json == obj
    return restored


def test_layer1_common():
    _roundtrip(
        AuditEnvelope(
            run_id="run-1",
            stage_name="perception",
            timestamp="2026-05-26T00:00:00Z",
            provenance=Provenance(source_id="src", modality=Modality.TEXT),
        ),
        AuditEnvelope,
    )
    _roundtrip(ComputePolicy(max_active_traces=64, mode="cheap"), ComputePolicy)


def test_layer2_perception():
    graph = PerceptualEvidenceGraph(
        candidate_units=[
            CandidateUnit(unit_id="u1", surface="bank", kind_hint="noun", confidence=0.9)
        ],
        provenance=Provenance(modality=Modality.TEXT),
    )
    _roundtrip(graph, PerceptualEvidenceGraph)
    _roundtrip(
        PerceptionInput(raw_payload="go to the bank", modality=Modality.TEXT),
        PerceptionInput,
    )


def test_layer3_cue_and_dmf():
    graph = PerceptualEvidenceGraph()
    cloud = CueCloud(
        primitive_trace_activations=[
            TraceActivationRequest(trace_id="t-financial", weight=0.7)
        ]
    )
    _roundtrip(cloud, CueCloud)
    _roundtrip(CueEncoderInput(perceptual_evidence_graph=graph), CueEncoderInput)
    _roundtrip(
        DmfOutput(active_traces=[ActiveTrace(trace_id="t1", activation=0.8)]),
        DmfOutput,
    )
    _roundtrip(DmfInput(cue_cloud=cloud), DmfInput)


def test_layer4_binding_context_interference_basins():
    frame = CandidateFrame(frame_id="f1", frame_type="word_sense", confidence=0.6)
    binding_out = BindingOutput(candidate_frames=[frame])
    _roundtrip(binding_out, BindingOutput)

    ctx_out = ContextOpOutput(
        context_frames=[ContextFrame(context_frame_id="cf1", member_frame_ids=["f1"])]
    )
    _roundtrip(ctx_out, ContextOpOutput)

    interference_out = InterferenceOutput(
        trace_trace_edges=[TraceTraceEdge(trace_id_a="t1", trace_id_b="t2", delta=-0.1)]
    )
    _roundtrip(interference_out, InterferenceOutput)

    basin_out = BasinOutput(
        candidate_basin_states=[
            CandidateBasinState(basin_id="b1", energy=1.2, margin_vs_next=0.3)
        ],
        competition_summary=CompetitionSummary(top_basin_id="b1", top_margin=0.3),
    )
    _roundtrip(basin_out, BasinOutput)


def test_layer5_lucidity_projector_decoder():
    policy = DecoderPolicy(mode=DecoderMode.EXPRESS_COMMITTED.value)
    committed = CommittedState(
        commit_id="c1",
        commit_shape=CommitShape.SINGLE,
        primary_basin_id="b1",
    )
    lucidity_out = LucidityOutput(
        decision=LucidityDecision.COMMIT,
        decoder_policy=policy,
        committed_state=committed,
    )
    _roundtrip(lucidity_out, LucidityOutput)

    directives = SearchDirectives(rollout_mode="multi_step", max_rollouts=3)
    _roundtrip(
        ProjectorInput(projection_request=directives, target_basin_ids=["b1"]),
        ProjectorInput,
    )
    _roundtrip(
        ProjectorOutput(
            rollouts=[ProjectorRollout(rollout_id="r1")],
            best_rollout_id="r1",
        ),
        ProjectorOutput,
    )
    packet = LucidityRenderPacket(
        packet_id="rp-1",
        decision=LucidityDecision.COMMIT,
        render_mode="committed",
        approved_units=[
            RenderUnit(
                unit_id="u1",
                unit_type="claim",
                payload={"summary": "river bank"},
                source_refs=[SourceRef(ref_type="trace", ref_id="t1")],
            )
        ],
    )
    _roundtrip(packet, LucidityRenderPacket)
    _roundtrip(
        DecoderOutput(
            surface_text="The river bank.",
            render_mode="committed",
            sentence_refs=[SentenceRef(sentence_id="s0", unit_ids=["u1"])],
            faithfulness_report=FaithfulnessReport(passed=True),
        ),
        DecoderOutput,
    )
    lucidity_with_packet = LucidityOutput(
        decision=LucidityDecision.COMMIT,
        decoder_policy=policy,
        committed_state=committed,
        render_packet=packet,
    )
    _roundtrip(lucidity_with_packet, LucidityOutput)


def test_layer6_memory_and_training():
    _roundtrip(
        TraceRecord(
            trace_id="t1",
            frame_affinities={"cf1": 0.8},
            support_links=[WeightedLink(target_id="t2", weight=0.5)],
        ),
        TraceRecord,
    )
    _roundtrip(BasinRecord(basin_id="b1", frame_affinities={"cf1": 0.9}), BasinRecord)

    gold = GoldLabels(
        lucidity_target="COMMIT",
        expected_answer="river bank",
    )
    episode = Episode(
        episode_id="ep-1",
        modality=Modality.TEXT,
        template_id="ambiguous_destination_v1",
        raw_input="go to the bank",
        gold=gold,
        validator="exact_sense",
        seed=42,
    )
    _roundtrip(episode, Episode)
    _roundtrip(
        TrainingEpisode(
            episode_id="tep-1",
            raw_input={"grid": [[0, 1], [1, 0]]},
            modality=Modality.GRID,
            task_intent=TaskIntent.SOLVE_GRID,
        ),
        TrainingEpisode,
    )


def test_layer7_pipeline():
    ctx = RunContext(
        run_id="run-abc",
        session_id="sess-1",
        task_intent=TaskIntent.CHAT,
        session_state=SessionState(session_id="sess-1"),
    )
    run = PipelineRun(
        context=ctx,
        stage_results=[
            StageResult(
                stage_name=StageName.PERCEPTION,
                success=True,
                duration_ms=12.5,
            )
        ],
    )
    restored = _roundtrip(run, PipelineRun)
    log = restored.to_run_log()
    assert log.run_id == "run-abc"


def test_run_log_from_pipeline():
    graph = PerceptualEvidenceGraph(
        candidate_units=[CandidateUnit(unit_id="u1", surface="hi")]
    )
    run = PipelineRun(
        context=RunContext(run_id="r1"),
        evidence_graph=graph,
        lucidity_output=LucidityOutput(
            decision=LucidityDecision.PRESERVE_AMBIGUITY,
            decoder_policy=DecoderPolicy(mode=DecoderMode.EXPRESS_UNCERTAINTY.value),
        ),
    )
    log = run.to_run_log()
    restored = _roundtrip(log, RunLog)
    assert restored.lucidity_output.decision == LucidityDecision.PRESERVE_AMBIGUITY


def test_json_loads_compatible():
    payload = {
        "episode_id": "e1",
        "modality": "text",
        "raw_input": "hello",
        "gold": {"lucidity_target": "COMMIT"},
        "validator": "v",
        "seed": 1,
        "meta": {},
        "task_intent": "chat",
        "context": {},
        "constraints": {},
    }
    ep = from_dict(payload, Episode)
    assert ep.modality == Modality.TEXT
    assert ep.task_intent == TaskIntent.CHAT
    assert json.loads(to_json(ep)) == to_dict(ep)
