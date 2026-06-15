"""Decoder — render lucidity script into user-facing output."""

from __future__ import annotations

from lucid.cognition.output.decoder.faithfulness import check_faithfulness, collect_cited_refs
from lucid.cognition.output.decoder.polish import polish_for_chat
from lucid.cognition.output.decoder.renderers.text import render_literal_fallback
from lucid.cognition.output.decoder.router import route_render
from lucid.cognition.output.decoder.semantic_graph import build_semantic_graph
from lucid.cognition.output.decoder.discourse_plan import plan_discourse
from lucid.cognition.output.decoder.realization_ops import plan_realization
from lucid.cognition.output.decoder.structural_faithfulness import check_structural_faithfulness
from lucid.ir.common import DecoderMode
from lucid.ir.expression import DecoderInput, DecoderOutput, FaithfulnessReport, SentenceRef
from lucid.ir.lucidity import LucidityRenderPacket


def _render_chat_uncertainty(packet: LucidityRenderPacket, *, render_mode: str | None = None) -> DecoderOutput:
    refs = []
    for alt in packet.preserved_alternatives:
        for ref in alt.get("source_refs") or []:
            if hasattr(ref, "ref_type") and hasattr(ref, "ref_id"):
                refs.append(ref)
            elif isinstance(ref, dict) and ref.get("ref_id"):
                from lucid.ir.lucidity import SourceRef

                refs.append(
                    SourceRef(
                        ref_type=str(ref.get("ref_type") or "basin"),
                        ref_id=str(ref["ref_id"]),
                        scope_frame_id=str(ref.get("scope_frame_id") or ""),
                        role=str(ref.get("role") or ""),
                    )
                )
    sentence_refs = [
        SentenceRef(
            sentence_id="s0",
            source_refs=refs,
            unit_ids=[],
        )
    ]
    return DecoderOutput(
        surface_text="I'm not confident enough to answer from the current memory.",
        render_mode=render_mode or packet.render_mode,
        sentence_refs=sentence_refs,
        uncertainty_presentation={"alternatives": len(packet.preserved_alternatives)},
        audit_notes=["decoder:chat_uncertainty"],
    )


def run_decoder(inp: DecoderInput, ctx: object | None = None) -> DecoderOutput:
    """Express lucidity-approved state; refuse when script is missing (except hold)."""
    _ = ctx
    policy = inp.decoder_policy or inp.lucidity_output.decoder_policy
    packet = inp.render_packet or inp.lucidity_output.render_packet
    channel = (inp.output_channel or policy.output_channel or "chat").strip().lower()

    if policy.mode == DecoderMode.HOLD.value or (packet and packet.render_mode == "hold"):
        if channel == "chat":
            return _render_chat_uncertainty(
                packet
                or LucidityRenderPacket(
                    packet_id="hold",
                    decision=inp.lucidity_output.decision,
                    render_mode="hold",
                ),
                render_mode="hold",
            )
        out = route_render(
            packet
            or LucidityRenderPacket(
                packet_id="hold",
                decision=inp.lucidity_output.decision,
                render_mode="hold",
            ),
            policy,
        )
        out.audit_notes.append("decoder:contract=hold")
        return out

    if packet is None:
        return DecoderOutput(
            refused=True,
            refusal_reason="missing_render_packet",
            faithfulness_report=FaithfulnessReport(
                passed=False,
                policy_violations=["missing_render_packet"],
            ),
            audit_notes=["decoder:contract_error"],
        )

    graph = build_semantic_graph(packet)
    discourse = plan_discourse(graph, packet.render_constraints)
    program = plan_realization(discourse)
    structural_report = check_structural_faithfulness(
        graph=graph,
        program=program,
        policy=policy,
    )
    if channel == "chat" and packet.render_mode == "plural" and policy.forbid_single_answer:
        draft = _render_chat_uncertainty(packet)
    else:
        draft = route_render(packet, policy)

    report = check_faithfulness(
        surface_text=draft.surface_text,
        sentence_refs=draft.sentence_refs,
        packet=packet,
        policy=policy,
        structural_report=structural_report,
    )
    if not report.passed and draft.surface_text:
        fallback = render_literal_fallback(packet)
        fallback_report = check_faithfulness(
            surface_text=fallback.surface_text,
            sentence_refs=fallback.sentence_refs,
            packet=packet,
            policy=policy,
            structural_report=structural_report,
        )
        if fallback_report.passed or not draft.surface_text:
            draft = fallback
            report = fallback_report
            draft.audit_notes.append("decoder:fallback=literal")

    if channel == "chat" and draft.surface_text:
        polished, refs = polish_for_chat(
            draft.surface_text,
            packet=packet,
            sentence_refs=draft.sentence_refs,
        )
        polish_report = check_faithfulness(
            surface_text=polished,
            sentence_refs=refs,
            packet=packet,
            policy=policy,
            structural_report=structural_report,
        )
        if polish_report.passed:
            draft.surface_text = polished
            draft.sentence_refs = refs
            report = polish_report
            draft.audit_notes.append("decoder:polish=chat")
        else:
            draft.audit_notes.append("decoder:polish_skipped")

    draft.faithfulness_report = report
    draft.render_mode = draft.render_mode or packet.render_mode
    draft.cited_refs = collect_cited_refs(draft.sentence_refs)
    draft.cited_trace_ids = [
        ref.ref_id for ref in draft.cited_refs if ref.ref_type == "trace" and ref.ref_id
    ]
    if not report.passed and not draft.refused:
        draft.audit_notes.append("decoder:faithfulness_failed")

    return draft


__all__ = ["run_decoder"]
