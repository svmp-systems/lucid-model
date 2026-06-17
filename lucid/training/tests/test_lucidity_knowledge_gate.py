"""Tests for unified lucidity gates on knowledge queries."""

from __future__ import annotations

from lucid.cognition.output.lucidity.checks import run_checks
from lucid.cognition.output.lucidity.config import LucidityConfig
from lucid.cognition.output.lucidity.decide import decide
from lucid.cognition.output.lucidity.evidence_quality import source_backed_renderable_basin
from lucid.ir.basins import BasinOutput, CandidateBasinState, CompetitionSummary
from lucid.ir.binding import BindingOutput, CandidateFrame
from lucid.ir.context_op import ContextOpOutput
from lucid.ir.dmf import ActiveTrace, DmfOutput
from lucid.ir.interference import InterferenceOutput
from lucid.ir.lucidity import LucidityInput
from lucid.ir.perception import PerceptualEvidenceGraph


def _knowledge_input(*, top_margin: float) -> LucidityInput:
    graph = PerceptualEvidenceGraph()
    graph.provenance.extra["raw_text"] = "what is a transformer"
    top_energy = 0.7
    second_energy = max(0.0, top_energy - top_margin)
    definition = CandidateBasinState(
        basin_id="b_transformer_architecture_definition",
        energy=top_energy,
        supporting_trace_ids=["t_term_transformer_architecture"],
        source_refs=["wiki_transformer"],
        coherence_score=0.82,
        quantized_payload={
            "concept_id": "transformer_architecture",
            "relations": [
                {
                    "relation": "type_of",
                    "target": "a neural network architecture based on attention mechanisms",
                    "source_refs": ["wiki_transformer"],
                }
            ],
        },
    )
    electrical = CandidateBasinState(
        basin_id="b_transformer_architecture_definition_alt",
        energy=second_energy,
        supporting_trace_ids=["t_term_transformer_architecture"],
        source_refs=["wiki_transformer_alt"],
        coherence_score=0.75,
        quantized_payload={
            "concept_id": "transformer_architecture",
            "relations": [
                {
                    "relation": "type_of",
                    "target": "a neural network architecture that uses attention for sequence modeling",
                    "source_refs": ["wiki_transformer_alt"],
                }
            ],
        },
    )
    return LucidityInput(
        perceptual_evidence_graph=graph,
        basin_output=BasinOutput(
            candidate_basin_states=[definition, electrical],
            competition_summary=CompetitionSummary(
                top_basin_id="b_transformer_architecture_definition",
                second_basin_id="b_transformer_architecture_definition_alt",
                top_margin=top_margin,
                active_basin_count=2,
            ),
        ),
        binding_output=BindingOutput(
            candidate_frames=[
                CandidateFrame(
                    frame_id="concept_query_transformer",
                    frame_type="definition_query",
                    member_evidence_refs=["u_transformer"],
                    confidence=0.82,
                    supporting_trace_ids=["t_term_transformer_architecture"],
                )
            ],
            binding_stability_score=0.8,
        ),
        context_op_output=ContextOpOutput(),
        interference_output=InterferenceOutput(),
        dmf_output=DmfOutput(
            active_traces=[
                ActiveTrace("t_term_transformer_architecture", 0.72, heat_tier="warm"),
                ActiveTrace("t_term_transformer", 0.55, heat_tier="warm"),
            ],
            coverage_score=0.8,
            top_margin=top_margin,
        ),
        task_intent="chat",
    )


def test_knowledge_query_uses_margin_threshold_in_chat() -> None:
    checks, _ = run_checks(_knowledge_input(top_margin=0.04), LucidityConfig())
    assert checks.margin_check is not None
    assert checks.margin_check.threshold == 0.08
    assert checks.margin_check.passed is False


def test_knowledge_query_low_margin_preserves_ambiguity() -> None:
    inp = _knowledge_input(top_margin=0.04)
    checks, confidence = run_checks(inp, LucidityConfig())
    out = decide(inp, checks, confidence, LucidityConfig())
    assert out.decision.value == "preserve_ambiguity"
    assert "lucidity:preserve_ambiguity_margin" in out.audit_notes


def test_knowledge_query_commits_when_checks_pass() -> None:
    inp = _knowledge_input(top_margin=0.12)
    checks, confidence = run_checks(inp, LucidityConfig())
    out = decide(inp, checks, confidence, LucidityConfig())
    assert out.decision.value == "commit"
    assert "lucidity:commit" in out.audit_notes


def test_knowledge_query_ignores_duplicate_target_basin_scopes() -> None:
    inp = _knowledge_input(top_margin=0.02)
    graph = inp.perceptual_evidence_graph
    graph.provenance.extra["raw_text"] = "what is AI"
    inp.basin_output.candidate_basin_states = [
        CandidateBasinState(
            basin_id="b_artificial_intelligence_challenge",
            energy=0.729,
            supporting_trace_ids=["t_term_artificial_intelligence"],
            source_refs=["wiki_artificial_intelligence"],
            coherence_score=0.61,
            quantized_payload={
                "concept_id": "artificial_intelligence",
                "relations": [
                    {
                        "relation": "property",
                        "target": "has practical and safety challenges",
                        "source_refs": ["wiki_artificial_intelligence"],
                    }
                ],
            },
        ),
        CandidateBasinState(
            basin_id="b_artificial_intelligence_definition",
            energy=0.704,
            supporting_trace_ids=["t_term_artificial_intelligence"],
            source_refs=["wiki_artificial_intelligence"],
            coherence_score=0.61,
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
        ),
        CandidateBasinState(
            basin_id="b_artificial_intelligence_definition",
            energy=0.646,
            supporting_trace_ids=["t_term_artificial_intelligence"],
            source_refs=["wiki_artificial_intelligence"],
            coherence_score=0.61,
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
        ),
    ]
    inp.basin_output.competition_summary = CompetitionSummary(
        top_basin_id="b_artificial_intelligence_challenge",
        second_basin_id="b_artificial_intelligence_definition",
        top_margin=0.025,
        active_basin_count=3,
    )
    inp.binding_output.candidate_frames[0].frame_id = "concept_query_artificial_intelligence"
    inp.binding_output.candidate_frames[0].supporting_trace_ids = ["t_term_artificial_intelligence"]
    inp.dmf_output.active_traces = [
        ActiveTrace("t_term_artificial_intelligence", 0.72, heat_tier="warm")
    ]

    assert source_backed_renderable_basin(inp)
    checks, confidence = run_checks(inp, LucidityConfig())
    assert checks.margin_check is not None
    assert checks.margin_check.passed
    assert checks.coherence_check is not None
    assert checks.coherence_check.threshold == LucidityConfig().coherence_threshold_source_backed

    out = decide(inp, checks, confidence, LucidityConfig())
    assert out.decision.value == "commit"
    assert out.committed_state is not None
    assert out.committed_state.primary_basin_id == "b_artificial_intelligence_definition"
