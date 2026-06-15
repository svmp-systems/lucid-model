"""Decoder: script → surface text, faithfulness, grid, plural."""

from __future__ import annotations

from uuid import uuid4

from lucid.cognition.output.decoder import run_decoder
from lucid.cognition.output.decoder.semantic_graph import build_semantic_graph
from lucid.cognition.output.decoder.discourse_plan import plan_discourse
from lucid.cognition.output.decoder.realization_ops import plan_realization
from lucid.cognition.output.decoder.structural_faithfulness import check_structural_faithfulness
from lucid.cognition.output.lucidity import attach_render_packet, build_render_packet
from lucid.ir.common import DecoderMode, LucidityDecision
from lucid.ir.expression import DecoderInput
from lucid.ir.lucidity import (
    CommittedState,
    DecoderPolicy,
    LucidityOutput,
    LucidityRenderPacket,
    RenderUnit,
    SourceRef,
)


def _packet_committed_bank() -> LucidityRenderPacket:
    return LucidityRenderPacket(
        packet_id=str(uuid4()),
        decision=LucidityDecision.COMMIT,
        render_mode="committed",
        output_format="text",
        approved_units=[
            RenderUnit(
                unit_id="u1",
                unit_type="claim",
                scope_frame_id="F2",
                payload={"bank_sense": "financial_storage"},
                required=True,
                source_refs=[
                    SourceRef(ref_type="trace", ref_id="t_bank"),
                    SourceRef(ref_type="basin", ref_id="b_fin"),
                ],
            ),
            RenderUnit(
                unit_id="u2",
                unit_type="caveat",
                payload={"kayaking_scope": "separate_event"},
                required=False,
                source_refs=[SourceRef(ref_type="frame", ref_id="F1")],
            ),
        ],
    )


def test_committed_bank_renders_natural_text() -> None:
    packet = _packet_committed_bank()
    policy = DecoderPolicy(mode=DecoderMode.EXPRESS_COMMITTED.value, output_channel="chat")
    out = run_decoder(
        DecoderInput(
            lucidity_output=LucidityOutput(
                decision=LucidityDecision.COMMIT,
                decoder_policy=policy,
                render_packet=packet,
            ),
            render_packet=packet,
            decoder_policy=policy,
            output_channel="chat",
        )
    )

    assert "bank" in out.surface_text.lower()
    assert "financial" in out.surface_text.lower()
    assert out.faithfulness_report.passed
    assert out.sentence_refs
    assert not out.refused


def test_plural_does_not_collapse_alternatives() -> None:
    packet = LucidityRenderPacket(
        packet_id=str(uuid4()),
        decision=LucidityDecision.PRESERVE_AMBIGUITY,
        render_mode="plural",
        preserved_alternatives=[
            {
                "basin_id": "b_fin",
                "narrative_hint": "financial bank",
                "source_refs": [SourceRef(ref_type="basin", ref_id="b_fin")],
            },
            {
                "basin_id": "b_river",
                "narrative_hint": "river bank",
                "source_refs": [SourceRef(ref_type="basin", ref_id="b_river")],
            },
        ],
    )
    policy = DecoderPolicy(
        mode=DecoderMode.EXPRESS_PLURAL.value,
        forbid_single_answer=True,
        output_channel="audit",
    )
    out = run_decoder(
        DecoderInput(
            lucidity_output=LucidityOutput(
                decision=LucidityDecision.PRESERVE_AMBIGUITY,
                decoder_policy=policy,
                render_packet=packet,
            ),
            render_packet=packet,
            decoder_policy=policy,
        )
    )

    lowered = out.surface_text.lower()
    assert "financial" in lowered or "b_fin" in lowered
    assert "river" in lowered or "b_river" in lowered
    assert "single reading" in lowered
    assert out.faithfulness_report.passed


def test_chat_plural_returns_uncertainty_not_audit_labels() -> None:
    packet = LucidityRenderPacket(
        packet_id=str(uuid4()),
        decision=LucidityDecision.PRESERVE_AMBIGUITY,
        render_mode="plural",
        preserved_alternatives=[
            {
                "basin_id": "b_fin",
                "narrative_hint": "financial bank",
                "source_refs": [SourceRef(ref_type="basin", ref_id="b_fin")],
            },
            {
                "basin_id": "b_river",
                "narrative_hint": "river bank",
                "source_refs": [SourceRef(ref_type="basin", ref_id="b_river")],
            },
        ],
    )
    policy = DecoderPolicy(
        mode=DecoderMode.EXPRESS_PLURAL.value,
        forbid_single_answer=True,
        output_channel="chat",
    )
    out = run_decoder(
        DecoderInput(
            lucidity_output=LucidityOutput(
                decision=LucidityDecision.PRESERVE_AMBIGUITY,
                decoder_policy=policy,
                render_packet=packet,
            ),
            render_packet=packet,
            decoder_policy=policy,
        )
    )

    lowered = out.surface_text.lower()
    assert "not confident" in lowered
    assert "b_fin" not in lowered
    assert "b_river" not in lowered
    assert out.faithfulness_report.passed


def test_grid_renderer_copies_artifact() -> None:
    grid = [[0, 1], [1, 0]]
    packet = LucidityRenderPacket(
        packet_id=str(uuid4()),
        decision=LucidityDecision.COMMIT,
        render_mode="committed",
        output_format="grid",
        approved_units=[
            RenderUnit(
                unit_id="g1",
                unit_type="artifact",
                payload={"grid_output": grid},
                required=True,
            )
        ],
    )
    policy = DecoderPolicy(
        mode=DecoderMode.EXPRESS_COMMITTED.value,
        output_format="grid",
    )
    out = run_decoder(
        DecoderInput(
            lucidity_output=LucidityOutput(
                decision=LucidityDecision.COMMIT,
                decoder_policy=policy,
                render_packet=packet,
            ),
            render_packet=packet,
            decoder_policy=policy,
        )
    )
    assert out.surface_grid == grid


def test_missing_packet_refuses() -> None:
    policy = DecoderPolicy(mode=DecoderMode.EXPRESS_COMMITTED.value)
    out = run_decoder(
        DecoderInput(
            lucidity_output=LucidityOutput(
                decision=LucidityDecision.COMMIT,
                decoder_policy=policy,
            ),
            decoder_policy=policy,
        )
    )
    assert out.refused
    assert "missing_render_packet" in out.refusal_reason


def test_build_render_packet_from_committed_state() -> None:
    out = LucidityOutput(
        decision=LucidityDecision.COMMIT,
        decoder_policy=DecoderPolicy(mode=DecoderMode.EXPRESS_COMMITTED.value),
        committed_state=CommittedState(
            commit_id="c1",
            primary_basin_id="b00491",
        ),
    )
    packet = build_render_packet(out)
    assert packet is not None
    assert packet.render_mode == "committed"
    assert packet.approved_units

    attached = attach_render_packet(out)
    assert attached.render_packet is not None


def test_compose_dedupes_duplicate_payloads() -> None:
    from lucid.cognition.output.decoder.compose import build_compose_plan

    packet = LucidityRenderPacket(
        packet_id="p",
        decision=LucidityDecision.COMMIT,
        render_mode="committed",
        approved_units=[
            RenderUnit(
                unit_id="a",
                unit_type="claim",
                payload={"bank_sense": "financial_storage"},
                required=True,
            ),
            RenderUnit(
                unit_id="b",
                unit_type="claim",
                payload={"bank_sense": "financial_storage"},
                required=False,
            ),
        ],
    )
    plan = build_compose_plan(packet)
    assert len(plan.bullets) == 1
    assert len(plan.bullets[0].unit_ids) == 2


def test_semantic_transducer_builds_faithful_realization_program() -> None:
    packet = _packet_committed_bank()

    graph = build_semantic_graph(packet)
    discourse = plan_discourse(graph, packet.render_constraints)
    program = plan_realization(discourse)
    report = check_structural_faithfulness(
        graph=graph,
        program=program,
        policy=DecoderPolicy(mode=DecoderMode.EXPRESS_COMMITTED.value),
    )

    assert [node.node_type for node in graph.nodes] == ["claim", "caveat"]
    assert any(step.function == "answer" for step in discourse.steps)
    assert any(op.op_type == "realize_claim" for op in program.ops)
    assert report.passed


def test_action_packet_renders_structured_action_language() -> None:
    packet = LucidityRenderPacket(
        packet_id="action",
        decision=LucidityDecision.COMMIT,
        render_mode="committed",
        output_format="action",
        approved_units=[
            RenderUnit(
                unit_id="act1",
                unit_type="action",
                payload={"action_type": "move_right", "target_ref": "red_square"},
                source_refs=[SourceRef(ref_type="trace", ref_id="t_move")],
            )
        ],
    )
    policy = DecoderPolicy(mode=DecoderMode.EXPRESS_COMMITTED.value, output_format="action")

    out = run_decoder(
        DecoderInput(
            lucidity_output=LucidityOutput(
                decision=LucidityDecision.COMMIT,
                decoder_policy=policy,
                render_packet=packet,
            ),
            render_packet=packet,
            decoder_policy=policy,
        )
    )

    assert "move right" in out.surface_text.lower()
    assert "red square" in out.surface_text.lower()
    assert out.faithfulness_report.passed


def test_canvas_decoder_renders_source_backed_graph_claim() -> None:
    packet = LucidityRenderPacket(
        packet_id="quantum",
        decision=LucidityDecision.COMMIT,
        render_mode="committed",
        output_format="text",
        approved_units=[
            RenderUnit(
                unit_id="graph-claim-0",
                unit_type="claim",
                payload={
                    "subject": "qubit",
                    "relation": "type_of",
                    "target": "unit of quantum information",
                },
                required=True,
                source_refs=[SourceRef(ref_type="source", ref_id="ibm_quantum_computing")],
            )
        ],
    )
    policy = DecoderPolicy(mode=DecoderMode.EXPRESS_COMMITTED.value, output_channel="chat")

    out = run_decoder(
        DecoderInput(
            lucidity_output=LucidityOutput(
                decision=LucidityDecision.COMMIT,
                decoder_policy=policy,
                render_packet=packet,
            ),
            render_packet=packet,
            decoder_policy=policy,
        )
    )

    assert "a qubit is a unit of quantum information" in out.surface_text.lower()
    assert "decoder:canvas_builder" in out.audit_notes
    assert "decoder:route=canvas" in out.audit_notes
    assert out.faithfulness_report.passed


def test_fluent_decoder_renders_contrast_relation_as_clause() -> None:
    packet = LucidityRenderPacket(
        packet_id="contrast",
        decision=LucidityDecision.COMMIT,
        render_mode="committed",
        output_format="text",
        approved_units=[
            RenderUnit(
                unit_id="graph-claim-0",
                unit_type="claim",
                payload={
                    "subject": "qubit",
                    "relation": "contrast",
                    "target": "But instead of regular classical bits, quantum computers use quantum bits.",
                },
                required=True,
                source_refs=[SourceRef(ref_type="source", ref_id="quantum_article")],
            )
        ],
    )
    policy = DecoderPolicy(mode=DecoderMode.EXPRESS_COMMITTED.value, output_channel="chat")

    out = run_decoder(
        DecoderInput(
            lucidity_output=LucidityOutput(
                decision=LucidityDecision.COMMIT,
                decoder_policy=policy,
                render_packet=packet,
            ),
            render_packet=packet,
            decoder_policy=policy,
        )
    )

    assert "qubit contrast" not in out.surface_text.lower()
    assert out.surface_text == "Instead of regular classical bits, quantum computers use quantum bits."
    assert out.faithfulness_report.passed


def test_fluent_decoder_composes_definition_answer() -> None:
    packet = LucidityRenderPacket(
        packet_id="quantum-definition",
        decision=LucidityDecision.COMMIT,
        render_mode="committed",
        output_format="text",
        approved_units=[
            RenderUnit(
                unit_id="graph-claim-0",
                unit_type="claim",
                payload={
                    "subject": "qubit",
                    "relation": "type_of",
                    "target": "unit of quantum information",
                },
                required=True,
                source_refs=[SourceRef(ref_type="source", ref_id="ibm_quantum_computing")],
            ),
            RenderUnit(
                unit_id="graph-claim-1",
                unit_type="claim",
                payload={
                    "subject": "qubit",
                    "relation": "property",
                    "target": "can be prepared in superposition",
                },
                required=True,
                source_refs=[SourceRef(ref_type="source", ref_id="nist_quantum_explained")],
            ),
            RenderUnit(
                unit_id="graph-claim-2",
                unit_type="claim",
                payload={
                    "subject": "qubit",
                    "relation": "challenge",
                    "target": "noise and environmental disturbance",
                },
                required=True,
                source_refs=[SourceRef(ref_type="source", ref_id="nist_quantum_explained")],
            ),
        ],
    )
    policy = DecoderPolicy(mode=DecoderMode.EXPRESS_COMMITTED.value, output_channel="chat")

    out = run_decoder(
        DecoderInput(
            lucidity_output=LucidityOutput(
                decision=LucidityDecision.COMMIT,
                decoder_policy=policy,
                render_packet=packet,
            ),
            render_packet=packet,
            decoder_policy=policy,
        )
    )

    assert (
        out.surface_text
        == "A qubit is a unit of quantum information. It can be prepared in superposition, "
        "but it is limited by noise and environmental disturbance."
    )
    assert "classical bit" not in out.surface_text.lower()
    assert out.faithfulness_report.passed
    assert len(out.sentence_refs) == 2
    assert set(out.sentence_refs[0].unit_ids) == {
        "graph-claim-0",
        "graph-claim-1",
        "graph-claim-2",
    }
    assert set(out.sentence_refs[1].unit_ids) == set(out.sentence_refs[0].unit_ids)


def test_fluent_decoder_merges_repeated_uses_claims() -> None:
    packet = LucidityRenderPacket(
        packet_id="quantum-fluent",
        decision=LucidityDecision.COMMIT,
        render_mode="committed",
        output_format="text",
        approved_units=[
            RenderUnit(
                unit_id="graph-claim-0",
                unit_type="claim",
                payload={
                    "subject": "quantum computing",
                    "relation": "uses",
                    "target": "quantum mechanics",
                },
                required=True,
                source_refs=[SourceRef(ref_type="source", ref_id="quantum_mechanics")],
            ),
            RenderUnit(
                unit_id="graph-claim-1",
                unit_type="claim",
                payload={
                    "subject": "quantum computing",
                    "relation": "uses",
                    "target": "qubits",
                },
                required=True,
                source_refs=[SourceRef(ref_type="source", ref_id="qubits")],
            ),
            RenderUnit(
                unit_id="graph-claim-2",
                unit_type="claim",
                payload={
                    "subject": "quantum computing",
                    "relation": "uses",
                    "target": "quantum entanglement and quantum interference",
                },
                required=True,
                source_refs=[SourceRef(ref_type="source", ref_id="entanglement")],
            ),
        ],
    )
    policy = DecoderPolicy(mode=DecoderMode.EXPRESS_COMMITTED.value, output_channel="chat", max_sentences=2)

    out = run_decoder(
        DecoderInput(
            lucidity_output=LucidityOutput(
                decision=LucidityDecision.COMMIT,
                decoder_policy=policy,
                render_packet=packet,
            ),
            render_packet=packet,
            decoder_policy=policy,
            output_channel="chat",
        )
    )

    lowered = out.surface_text.lower()
    assert lowered.count("quantum computing uses") == 1
    assert "quantum mechanics" in lowered
    assert "qubits" in lowered
    assert "entanglement" in lowered
    assert out.faithfulness_report.passed
    assert "decoder:fluent_realizer" in out.audit_notes


def test_committed_frame_summaries_compose_fluid_surface() -> None:
    packet = LucidityRenderPacket(
        packet_id=str(uuid4()),
        decision=LucidityDecision.COMMIT,
        render_mode="committed",
        output_format="text",
        approved_units=[
            RenderUnit(
                unit_id="frame-0",
                unit_type="frame_summary",
                text_intent="answer",
                payload={
                    "summary": "found money",
                    "member_evidence_surfaces": ["found", "money"],
                    "unresolved_slots": [],
                },
                required=True,
                source_refs=[SourceRef(ref_type="evidence", ref_id="u_found")],
            ),
            RenderUnit(
                unit_id="frame-1",
                unit_type="frame_summary",
                text_intent="answer",
                payload={
                    "summary": "kayaking",
                    "member_evidence_surfaces": ["kayaking"],
                    "unresolved_slots": [],
                },
                required=False,
                source_refs=[SourceRef(ref_type="evidence", ref_id="u_kayaking")],
            ),
            RenderUnit(
                unit_id="frame-2",
                unit_type="frame_summary",
                text_intent="answer",
                payload={
                    "summary": "placed bank",
                    "member_evidence_surfaces": ["placed", "bank"],
                    "unresolved_slots": ["bank_sense"],
                },
                required=False,
                source_refs=[SourceRef(ref_type="evidence", ref_id="u_bank")],
            ),
        ],
    )
    policy = DecoderPolicy(mode=DecoderMode.EXPRESS_COMMITTED.value, output_channel="chat", max_sentences=4)
    out = run_decoder(
        DecoderInput(
            lucidity_output=LucidityOutput(
                decision=LucidityDecision.COMMIT,
                decoder_policy=policy,
                render_packet=packet,
            ),
            render_packet=packet,
            decoder_policy=policy,
            output_channel="chat",
        )
    )

    assert "found money" in out.surface_text.lower()
    assert "kayaking" in out.surface_text.lower()
    assert "bank" in out.surface_text.lower()
    assert "basin id" not in out.surface_text.lower()
    assert "roles:" not in out.surface_text.lower()
    assert "not fully settled" in out.surface_text.lower()
    assert out.faithfulness_report.passed
