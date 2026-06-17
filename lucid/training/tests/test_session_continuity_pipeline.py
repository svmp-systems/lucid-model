"""End-to-end session continuity and binding-graph commit tests."""

from __future__ import annotations

from lucid.chat import run_chat_turn, start_session
from lucid.cognition.output.lucidity.commit import concept_query_render_units
from lucid.cognition.output.lucidity.evidence_quality import binding_graph_render_units
from lucid.cognition.input.perception import PerceptionConfig
from lucid.cognition.pipe_orchestrator.runner import OrchestratorConfig, OrchestratorRunner
from lucid.cognition.session_continuity import (
    context_frames_from_carryover,
    extract_pipeline_carryover,
    pipeline_carryover_from_session_context,
)
from lucid.ir.basins import BasinOutput, CandidateBasinState, CompetitionSummary
from lucid.ir.binding import BindingOutput, CandidateFrame, GraphEdge, GraphNode, LocalGraph
from lucid.ir.common import Modality, TaskIntent
from lucid.ir.context_op import ContextOpOutput
from lucid.ir.dmf import ActiveTrace, DmfOutput
from lucid.ir.interference import InterferenceOutput
from lucid.ir.lucidity import LucidityInput
from lucid.ir.perception import PerceptualEvidenceGraph
from lucid.ir.training import Episode


def test_pipeline_carryover_roundtrip() -> None:
    carryover = {
        "concept_topics": ["transformer_architecture"],
        "prior_context_frames": [
            {
                "context_frame_id": "cf_concept_query_transformer_architecture",
                "member_frame_ids": ["concept_query_transformer_architecture"],
                "scope_notes": "concept:transformer_architecture",
            }
        ],
        "carryover_trace_ids": ["t_term_transformer_architecture"],
        "unresolved_items": [],
    }
    session_context = {"pipeline_carryover": carryover}
    frames = context_frames_from_carryover(pipeline_carryover_from_session_context(session_context))
    assert len(frames) == 1
    assert frames[0].context_frame_id == "cf_concept_query_transformer_architecture"


def test_binding_graph_preferred_for_concept_render_units() -> None:
    graph = PerceptualEvidenceGraph()
    graph.provenance.extra["raw_text"] = "what is machine learning"
    graph.provenance.extra["session_context"] = {}
    frame = CandidateFrame(
        frame_id="concept_query_machine_learning__t_term_machine_learning",
        frame_type="definition_query",
        confidence=0.82,
        supporting_trace_ids=["t_term_machine_learning"],
        local_graphs=[
            LocalGraph(
                graph_id="graph_concept",
                family="concept",
                nodes=[
                    GraphNode(node_id="concept:machine_learning", node_kind="concept", label="machine_learning"),
                    GraphNode(
                        node_id="value:field",
                        node_kind="value",
                        label="a field of study focused on learning from data",
                    ),
                ],
                edges=[
                    GraphEdge(
                        edge_id="edge_ml_type",
                        edge_kind="relation",
                        source_id="concept:machine_learning",
                        target_id="value:field",
                        label="type_of",
                        confidence=0.86,
                        provenance_refs=["wiki_machine_learning"],
                    )
                ],
            )
        ],
    )
    inp = LucidityInput(
        perceptual_evidence_graph=graph,
        binding_output=BindingOutput(candidate_frames=[frame], binding_stability_score=0.8),
        basin_output=BasinOutput(
            candidate_basin_states=[
                CandidateBasinState(
                    basin_id="b_machine_learning_definition",
                    energy=0.6,
                    source_refs=["wiki_machine_learning"],
                    quantized_payload={
                        "concept_id": "machine_learning",
                        "relations": [
                            {
                                "relation": "type_of",
                                "target": "junk fragment black box",
                                "source_refs": ["wiki_machine_learning"],
                            }
                        ],
                    },
                )
            ],
            competition_summary=CompetitionSummary(
                top_basin_id="b_machine_learning_definition",
                top_margin=0.12,
            ),
        ),
        context_op_output=ContextOpOutput(),
        interference_output=InterferenceOutput(),
        dmf_output=DmfOutput(
            active_traces=[ActiveTrace("t_term_machine_learning", 0.7, heat_tier="warm")],
            coverage_score=0.8,
        ),
        task_intent="chat",
    )
    assert binding_graph_render_units(inp)
    units = concept_query_render_units(inp)
    assert units
    target = str(units[0].payload.get("target") or "")
    assert "learning from data" in target.lower()
    assert "black box" not in target.lower()


def test_session_followup_carries_concept_topic() -> None:
    sid = start_session()
    first = run_chat_turn(
        "what is machine learning",
        session_id=sid,
        checkpoint="checkpoints/saves/v.0.3-ai-ml",
        perception_backend="rule",
    )
    _ = first
    from lucid.audit.chat import load_session_memory

    memory = load_session_memory("audit/chat", sid)
    assert memory is not None
    topics = memory.pipeline_carryover.get("concept_topics") or []
    assert "machine_learning" in topics or any("machine" in str(topic) for topic in topics)


def test_orchestrator_passes_prior_context_frames() -> None:
    episode = Episode(
        episode_id="ep-session-test",
        modality=Modality.TEXT,
        raw_input="how does it work",
        task_intent=TaskIntent.CHAT,
        context={
            "session_context": {
                "recent_turns": [{"user_input": "what is a transformer", "assistant_output": "..."}],
                "pipeline_carryover": {
                    "concept_topics": ["transformer_architecture"],
                    "prior_context_frames": [
                        {
                            "context_frame_id": "cf_concept_query_transformer_architecture",
                            "member_frame_ids": ["concept_query_transformer_architecture"],
                            "scope_notes": "concept:transformer_architecture",
                        }
                    ],
                    "carryover_trace_ids": ["t_term_transformer_architecture"],
                    "unresolved_items": [],
                },
            }
        },
    )
    runner = OrchestratorRunner(
        config=OrchestratorConfig(
            audit_base_dir="audit/chat",
            checkpoint="checkpoints/saves/v.0.3-ai-ml",
            perception=PerceptionConfig(backend="rule"),
        )
    )
    run = runner.run_episode(episode, session_id="test-session", turn_index=2)
    assert run.context_op_input is not None
    assert run.context_op_input.prior_context_frames
    carryover = extract_pipeline_carryover(run)
    assert carryover.get("concept_topics")
