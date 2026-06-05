"""Decoder: script → surface text, faithfulness, grid, plural."""

from __future__ import annotations

from uuid import uuid4

from lucid.cognition.decoder import run_decoder
from lucid.cognition.decoder.semantic_graph import build_semantic_graph
from lucid.cognition.decoder.discourse_plan import plan_discourse
from lucid.cognition.decoder.realization_ops import plan_realization
from lucid.cognition.decoder.structural_faithfulness import check_structural_faithfulness
from lucid.cognition.lucidity import attach_render_packet, build_render_packet
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
    assert "financial" in lowered or "b_fin" in lowered
    assert "river" in lowered or "b_river" in lowered
    assert "single reading" in lowered
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
    from lucid.cognition.decoder.compose import build_compose_plan

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
