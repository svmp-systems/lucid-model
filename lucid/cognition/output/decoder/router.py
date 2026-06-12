"""Pick renderer from packet format and render mode."""

from __future__ import annotations

from lucid.cognition.output.decoder.canvas import build_canvas, realize_canvas
from lucid.cognition.output.decoder.renderers.grid import render_grid
from lucid.cognition.output.decoder.renderers.hold import render_hold
from lucid.cognition.output.decoder.renderers.refusal import render_refusal
from lucid.cognition.output.decoder.semantic_graph import build_semantic_graph
from lucid.cognition.output.decoder.discourse_plan import plan_discourse
from lucid.cognition.output.decoder.realization_ops import plan_realization
from lucid.cognition.output.decoder.surface_realizer import realize_surface
from lucid.ir.expression import DecoderOutput
from lucid.ir.lucidity import DecoderPolicy, LucidityRenderPacket


def route_render(packet: LucidityRenderPacket, policy: DecoderPolicy) -> DecoderOutput:
    mode = (packet.render_mode or "").strip().lower()
    fmt = (packet.output_format or policy.output_format or "text").strip().lower()

    if mode == "hold":
        return render_hold()
    if mode == "refusal":
        if packet.approved_units:
            return _render_semantic(packet)
        return render_refusal(packet)
    if fmt == "grid":
        return render_grid(packet)
    if fmt in {"action", "plan", "tool_call", "structured_json"}:
        return _render_semantic(packet)
    if mode in {"plural", "uncertainty", "committed"}:
        return _render_semantic(packet)
    return _render_semantic(packet)


def _render_semantic(packet: LucidityRenderPacket) -> DecoderOutput:
    canvas = build_canvas(packet)
    if canvas.lines:
        out = realize_canvas(canvas, packet.render_constraints)
        out.audit_notes.append("decoder:route=canvas")
        return out

    graph = build_semantic_graph(packet)
    discourse = plan_discourse(graph, packet.render_constraints)
    program = plan_realization(discourse)
    out = realize_surface(program, packet.render_constraints)
    out.audit_notes.append("decoder:route=semantic")
    return out
