"""Tests for general concept definition query routing."""

from __future__ import annotations

from lucid.cognition.input.cue.encoder import CueEncoderConfig, encode_cues
from lucid.cognition.input.perception import PerceptionConfig, perceive
from lucid.cognition.output.lucidity.checks import run_checks
from lucid.cognition.output.lucidity.commit import build_committed_state
from lucid.cognition.output.lucidity.config import LucidityConfig
from lucid.cognition.output.lucidity.decide import decide
from lucid.cognition.output.lucidity.chat_speech import classify_social_utterance
from lucid.cognition.reasoning.binding import BindingConfig, run_binding
from lucid.ir.basins import BasinOutput, CandidateBasinState, CompetitionSummary
from lucid.ir.binding import BindingInput, BindingOutput, CandidateFrame
from lucid.ir.common import Modality, TaskIntent
from lucid.ir.context_op import ContextFrame, ContextOpOutput
from lucid.ir.cue import CueEncoderInput
from lucid.ir.dmf import ActiveTrace, DmfInput, DmfOutput
from lucid.ir.interference import InterferenceOutput
from lucid.ir.lucidity import LucidityInput
from lucid.ir.perception import PerceptionInput
from lucid.memory.dmf import DynamicMemoryField, tracebank_from_checkpoint
from lucid.runtime.paths import resolve_train_path
from lucid.training.source_context import (
    concept_definition_primary_basin,
    is_renderable_definition_target,
    parse_concept_query,
    parse_concept_query_with_context,
    preferred_definition_concept,
    resolve_concept_topic,
    score_definition_target_for_concept,
)


def test_parse_concept_query_ai_alias() -> None:
    parsed = parse_concept_query("what is AI")
    assert parsed is not None
    topic, concept_id, frame_type = parsed
    assert topic.lower() == "ai"
    assert concept_id == "artificial_intelligence"
    assert frame_type == "definition_query"


def test_resolve_concept_topic_aliases() -> None:
    assert resolve_concept_topic("AI") == "artificial_intelligence"
    assert resolve_concept_topic("machine learning") == "machine_learning"


def test_parse_concept_query_rejects_bare_pronoun() -> None:
    assert parse_concept_query("how does it work") is None


def test_parse_concept_query_strips_article_on_mechanism() -> None:
    parsed = parse_concept_query("how does a transformer work")
    assert parsed is not None
    assert parsed[1] == "transformer_architecture"
    assert parsed[2] == "mechanism_query"


def test_parse_concept_query_with_context_resolves_followup() -> None:
    session = {
        "recent_turns": [
            {"turn_index": 1, "user_input": "what is a transformer", "assistant_output": "..."},
        ]
    }
    parsed = parse_concept_query_with_context("how does it work", session)
    assert parsed is not None
    assert parsed[1] == "transformer"
    assert parsed[2] == "mechanism_query"


def test_concept_basin_matching_does_not_use_substrings() -> None:
    states = [
        CandidateBasinState(basin_id="b_addition_mechanism", energy=0.9),
        CandidateBasinState(basin_id="b_transformer_mechanism", energy=0.4),
    ]
    assert concept_definition_primary_basin(states, "it", frame_type="mechanism_query") == ""
    assert concept_definition_primary_basin(states, "transformer", frame_type="mechanism_query") == "b_transformer_mechanism"


def test_transformer_definition_prefers_architecture_sense() -> None:
    assert preferred_definition_concept("transformer") == "transformer_architecture"
    bad = "a bayesian network with the requirement that the relationships be causal"
    good = "more parallelizable than earlier recurrent neural network models"
    assert score_definition_target_for_concept(bad, "transformer") < score_definition_target_for_concept(
        good, "transformer"
    )


def test_what_is_ai_is_not_social_speech() -> None:
    assert classify_social_utterance("what is AI") is None


def test_black_box_ai_fragment_is_not_renderable() -> None:
    bad = "a 'black box.' maybe not for long\""
    assert not is_renderable_definition_target(bad, relation="type_of", concept_id="artificial_intelligence")


def test_what_is_ai_pipeline_routing() -> None:
    checkpoint = resolve_train_path("checkpoints/saves/v.0.3-ai-ml")
    if not checkpoint.exists():
        return

    graph = perceive(
        PerceptionInput(raw_payload="what is AI", modality=Modality.TEXT, task_intent_hint="chat"),
        config=PerceptionConfig(backend="rule"),
    )
    assert graph.provenance.extra.get("concept_query", {}).get("concept_id") == "artificial_intelligence"

    cue = encode_cues(
        CueEncoderInput(
            perceptual_evidence_graph=graph,
            task_intent_hint=TaskIntent.CHAT,
            retrieval_budget=8,
        ),
        config=CueEncoderConfig(checkpoint=checkpoint),
    )
    cue_keys = {req.trace_id for req in cue.primitive_trace_activations}
    assert "artificial_intelligence" in cue_keys
    ai_weight = max(
        (req.weight for req in cue.primitive_trace_activations if req.trace_id == "artificial_intelligence"),
        default=0.0,
    )
    raw_ai_weight = max(
        (req.weight for req in cue.primitive_trace_activations if req.trace_id == "ai"),
        default=0.0,
    )
    assert ai_weight >= raw_ai_weight

    dmf = DynamicMemoryField(tracebank_from_checkpoint(checkpoint))
    dmf_out = dmf.run(
        DmfInput(
            cue_cloud=cue,
            quarantine_filter=True,
        )
    )
    active_clusters = {trace.cluster_id for trace in dmf_out.active_traces}
    assert "artificial_intelligence" in active_clusters

    binding = run_binding(
        BindingInput(
            dmf_output=dmf_out,
            perceptual_evidence_graph=graph,
            cue_cloud=cue,
        ),
        config=BindingConfig(checkpoint=checkpoint),
    )
    assert any(frame.frame_type == "definition_query" for frame in binding.candidate_frames)


def test_commit_prefers_artificial_intelligence_definition_over_speech() -> None:
    graph = perceive(
        PerceptionInput(raw_payload="what is AI", modality=Modality.TEXT, task_intent_hint="chat"),
        config=PerceptionConfig(backend="rule"),
    )
    ai_definition = CandidateBasinState(
        basin_id="b_artificial_intelligence_definition",
        energy=0.62,
        margin_vs_next=0.12,
        supporting_trace_ids=["t_term_artificial_intelligence"],
        source_refs=["wiki_artificial_intelligence"],
        coherence_score=0.82,
        quantized_payload={
            "concept_id": "artificial_intelligence",
            "canonical_label": "artificial intelligence",
            "relations": [
                {
                    "relation": "type_of",
                    "target": "a field of computer science focused on building intelligent systems",
                    "source_refs": ["wiki_artificial_intelligence"],
                }
            ],
        },
    )
    speech = CandidateBasinState(
        basin_id="b_basic_capability",
        energy=0.5,
        margin_vs_next=0.05,
        supporting_trace_ids=["t_basic_what_can_you_do"],
        source_refs=["basic_language_phrase_corpus"],
        quantized_payload={
            "facet": "speech",
            "relations": [
                {
                    "relation": "speech_response",
                    "target": "I'm Lucid. I answer from audited pipeline state.",
                }
            ],
        },
    )
    basins = BasinOutput(
        candidate_basin_states=[speech, ai_definition],
        competition_summary=CompetitionSummary(
            top_basin_id="b_artificial_intelligence_definition",
            second_basin_id="b_basic_capability",
            top_margin=0.12,
            active_basin_count=2,
        ),
    )
    binding = BindingOutput(
        candidate_frames=[
            CandidateFrame(
                frame_id="concept_query_artificial_intelligence",
                frame_type="definition_query",
                member_evidence_refs=["u_ai"],
                confidence=0.82,
                supporting_trace_ids=["t_term_artificial_intelligence"],
            )
        ],
        binding_stability_score=0.8,
    )
    lucidity_inp = LucidityInput(
        perceptual_evidence_graph=graph,
        basin_output=basins,
        binding_output=binding,
        context_op_output=ContextOpOutput(
            context_frames=[ContextFrame("cf_concept_query", member_frame_ids=["concept_query_artificial_intelligence"])]
        ),
        interference_output=InterferenceOutput(),
        dmf_output=DmfOutput(
            active_traces=[
                ActiveTrace(
                    "t_term_artificial_intelligence",
                    0.7,
                    heat_tier="warm",
                    cluster_id="artificial_intelligence",
                )
            ],
            coverage_score=0.8,
            top_margin=0.12,
        ),
        task_intent=TaskIntent.CHAT,
    )
    checks, confidence = run_checks(lucidity_inp, LucidityConfig())
    decision = decide(lucidity_inp, checks=checks, confidence=confidence, config=LucidityConfig())
    assert decision.decision.value == "commit"
    assert "lucidity:commit" in decision.audit_notes

    committed = build_committed_state(lucidity_inp)
    assert committed.primary_basin_id == "b_artificial_intelligence_definition"
    render_text = " ".join(
        str(unit.payload.get("target") or unit.payload.get("summary") or "")
        for unit in committed.render_units
    ).lower()
    assert "intelligent systems" in render_text
    assert "i'm lucid" not in render_text


def test_concept_definition_primary_basin_skips_speech() -> None:
    speech = CandidateBasinState(basin_id="b_basic_capability", energy=0.9)
    definition = CandidateBasinState(basin_id="b_artificial_intelligence_definition", energy=0.4)
    assert (
        concept_definition_primary_basin([speech, definition], "artificial_intelligence")
        == "b_artificial_intelligence_definition"
    )
