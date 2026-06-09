"""Default stub stage implementations.

These let the orchestrator execute the full pipeline "step-by-step" and write
audits even before real algorithmic stage implementations exist.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from lucid.audit.logger import resolve_run_dir
from lucid.cognition.input.cue import CueEncoderConfig, encode_cues
from lucid.ir.basins import BasinInput, BasinOutput
from lucid.ir.binding import BindingInput, BindingOutput, CandidateFrame
from lucid.ir.context_op import ContextOpInput, ContextOpOutput
from lucid.ir.cue import CueCloud, CueEncoderInput
from lucid.ir.dmf import DmfInput, DmfOutput
from lucid.ir.expression import DecoderInput, DecoderOutput
from lucid.ir.interference import InterferenceInput, InterferenceOutput
from lucid.cognition.lucidity import LucidityGateConfig, run_lucidity
from lucid.ir.lucidity import LucidityInput, LucidityOutput
from lucid.ir.perception import PerceptionInput, PerceptualEvidenceGraph
from lucid.ir.projector import ProjectorInput, ProjectorOutput
from lucid.cognition.input.perception import PerceptionConfig, perceive as run_perception
from lucid.cognition.decoder import run_decoder
from lucid.cognition.projector import run_projector
from lucid.cognition.reasoning.basins import BasinsConfig, run_basins
from lucid.cognition.reasoning.binding import BindingConfig, run_binding
from lucid.cognition.reasoning.context_op import run_context_op
from lucid.cognition.reasoning.interference import run_interference
from lucid.ir.pipeline import RunContext
from lucid.memory.dmf import DynamicMemoryField, load_dynamic_memory_field


def perception(inp: PerceptionInput, ctx: object) -> PerceptualEvidenceGraph:
    cfg = None
    if isinstance(ctx, dict):
        cfg = ctx.get("perception_config")
    else:
        extra = getattr(ctx, "extra", None)
        if isinstance(extra, dict):
            cfg = extra.get("perception_config")
    if cfg is None:
        cfg = PerceptionConfig.from_env()
    return run_perception(inp, context=ctx, config=cfg)


def _extra_from_context(ctx: object) -> dict:
    if isinstance(ctx, dict):
        return ctx
    extra = getattr(ctx, "extra", None)
    return extra if isinstance(extra, dict) else {}


def cue_encoder(inp: CueEncoderInput, ctx: object) -> CueCloud:
    extra = _extra_from_context(ctx)
    return encode_cues(
        inp,
        config=CueEncoderConfig(checkpoint=extra.get("checkpoint") or extra.get("cue_checkpoint")),
    )


def _dmf_runtime(ctx: object) -> DynamicMemoryField:
    extra = _extra_from_context(ctx)
    runtime = extra.get("dmf_runtime")
    if isinstance(runtime, DynamicMemoryField):
        return runtime

    audit_dir: Path | None = None
    if isinstance(ctx, RunContext) and ctx.run_id:
        base = str(extra.get("audit_base_dir") or "audit")
        audit_dir = resolve_run_dir(base, ctx) / "dmf"

    checkpoint = str(extra.get("checkpoint") or extra.get("dmf_checkpoint") or "").strip()
    runtime = load_dynamic_memory_field(
        checkpoint or None,
        audit_base_dir=audit_dir,
    )
    extra["dmf_runtime"] = runtime
    return runtime


def dmf(inp: DmfInput, ctx: object) -> DmfOutput:
    runtime = _dmf_runtime(ctx)
    extra = _extra_from_context(ctx)

    prior_ids = list(inp.prior_active_trace_ids)
    if not prior_ids:
        carryover = extra.get("prior_active_trace_ids")
        if isinstance(carryover, list):
            prior_ids = [str(item) for item in carryover if str(item)]

    run_input = inp
    if not inp.tracebank_snapshot_id:
        run_input = replace(inp, tracebank_snapshot_id=runtime.snapshot_id())
    if prior_ids and not inp.prior_active_trace_ids:
        run_input = replace(run_input, prior_active_trace_ids=prior_ids)

    out = runtime.run(run_input)
    extra["prior_active_trace_ids"] = [
        trace.trace_id for trace in out.active_traces if trace.trace_id
    ]
    return out


def binding(inp: BindingInput, ctx: object) -> BindingOutput:
    extra = _extra_from_context(ctx)
    prior = list(inp.prior_candidate_frames)
    if not prior:
        carried = extra.get("prior_candidate_frames")
        if isinstance(carried, list):
            prior = [frame for frame in carried if isinstance(frame, CandidateFrame)]

    feedback = extra.get("lucidity_feedback")
    widen = (
        isinstance(feedback, list)
        and any(str(item).strip().upper() == "RECHECK_BINDING" for item in feedback)
    )

    run_input = inp
    if prior and not inp.prior_candidate_frames:
        run_input = replace(inp, prior_candidate_frames=prior)

    return run_binding(
        run_input,
        config=BindingConfig(
            checkpoint=extra.get("checkpoint") or extra.get("binding_checkpoint"),
            widen_on_recheck=widen,
        ),
    )


def context_op(inp: ContextOpInput, _ctx: object) -> ContextOpOutput:
    return run_context_op(inp)


def interference(inp: InterferenceInput, ctx: object) -> InterferenceOutput:
    extra = _extra_from_context(ctx)
    learned = extra.get("learned_interference_links")
    if learned and not inp.learned_interference_links:
        inp = replace(inp, learned_interference_links=list(learned))
    return run_interference(inp)


def basins(inp: BasinInput, ctx: object) -> BasinOutput:
    extra = _extra_from_context(ctx)
    return run_basins(
        inp,
        config=BasinsConfig(
            checkpoint=extra.get("checkpoint") or extra.get("basins_checkpoint"),
        ),
    )


def projector(inp: ProjectorInput, _ctx: object) -> ProjectorOutput:
    return run_projector(inp, context=_ctx)


def lucidity(inp: LucidityInput, ctx: object) -> LucidityOutput:
    extra = _extra_from_context(ctx)
    checkpoint = extra.get("checkpoint") or extra.get("lucidity_checkpoint")
    return run_lucidity(
        inp,
        config=LucidityGateConfig(checkpoint=checkpoint),
        ctx=ctx,
    )


def decoder(inp: DecoderInput, ctx: object) -> DecoderOutput:
    if inp.render_packet is None and inp.lucidity_output.render_packet is not None:
        inp = replace(
            inp,
            render_packet=inp.lucidity_output.render_packet,
            committed_state=inp.committed_state or inp.lucidity_output.committed_state,
            decoder_policy=inp.decoder_policy or inp.lucidity_output.decoder_policy,
        )
    return run_decoder(inp, ctx)


def build_default_stage_fns() -> dict[str, object]:
    """Return mapping: stage_name -> function(stage_input, ctx)->output."""
    return {
        "perception": perception,
        "cue_encoder": cue_encoder,
        "dmf": dmf,
        "binding": binding,
        "context_op": context_op,
        "interference": interference,
        "basins": basins,
        "projector": projector,
        "lucidity": lucidity,
        "decoder": decoder,
    }
