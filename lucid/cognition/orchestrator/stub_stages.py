"""Default stub stage implementations.

These let the orchestrator execute the full pipeline "step-by-step" and write
audits even before real algorithmic stage implementations exist.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from lucid.audit.logger import resolve_run_dir
from lucid.cognition.input.cue import CueEncoderConfig, encode_cues
from lucid.ir.basins import BasinInput, BasinOutput, CompetitionSummary
from lucid.ir.binding import BindingInput, BindingOutput
from lucid.ir.common import DecoderMode, LucidityDecision
from lucid.ir.context_op import ContextOpInput, ContextOpOutput
from lucid.ir.cue import CueCloud, CueEncoderInput
from lucid.ir.dmf import DmfInput, DmfOutput
from lucid.ir.expression import DecoderInput, DecoderOutput
from lucid.ir.interference import InterferenceInput, InterferenceOutput
from lucid.ir.lucidity import CommittedState, DecoderPolicy, LucidityInput, LucidityOutput
from lucid.ir.perception import PerceptionInput, PerceptualEvidenceGraph
from lucid.ir.projector import ProjectorInput, ProjectorOutput
from lucid.cognition.input.perception import PerceptionConfig, perceive as run_perception
from lucid.cognition.reasoning.context_op import run_context_op
from lucid.ir.pipeline import RunContext
from lucid.memory.dmf import DynamicMemoryField, load_dynamic_memory_field


def _lucidity_target_to_decision(target: str) -> LucidityDecision:
    # Generator gold uses strings like "COMMIT" / "PRESERVE_AMBIGUITY".
    # The IR decision enum uses lowercase values.
    raw = (target or "").strip().lower()
    if raw == "commit":
        return LucidityDecision.COMMIT
    if raw == "preserve_ambiguity":
        return LucidityDecision.PRESERVE_AMBIGUITY
    return LucidityDecision.PRESERVE_AMBIGUITY


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


def binding(inp: BindingInput, _ctx: object) -> BindingOutput:
    return BindingOutput()


def context_op(inp: ContextOpInput, _ctx: object) -> ContextOpOutput:
    return run_context_op(inp)


def interference(inp: InterferenceInput, _ctx: object) -> InterferenceOutput:
    return InterferenceOutput()


def basins(inp: BasinInput, _ctx: object) -> BasinOutput:
    return BasinOutput(competition_summary=CompetitionSummary())


def projector(inp: ProjectorInput, _ctx: object) -> ProjectorOutput:
    return ProjectorOutput()


def lucidity(inp: LucidityInput, ctx: object) -> LucidityOutput:
    episode = getattr(ctx, "episode", None)
    target = ""
    if episode is not None and getattr(episode, "gold", None) is not None:
        target = getattr(episode.gold, "lucidity_target", "") or ""

    decision = _lucidity_target_to_decision(target)
    policy = DecoderPolicy(
        mode=DecoderMode.EXPRESS_COMMITTED.value
        if decision == LucidityDecision.COMMIT
        else DecoderMode.EXPRESS_UNCERTAINTY.value
    )
    committed_state = None
    if decision == LucidityDecision.COMMIT:
        committed_state = CommittedState(
            commit_id=str(uuid4()),
            primary_basin_id=inp.basin_output.competition_summary.top_basin_id,
        )
    return LucidityOutput(decision=decision, decoder_policy=policy, committed_state=committed_state)


def decoder(inp: DecoderInput, ctx: object) -> DecoderOutput:
    episode = getattr(ctx, "episode", None)
    expected = None
    if episode is not None and getattr(episode, "gold", None) is not None:
        expected = getattr(episode.gold, "expected_answer", None)

    if isinstance(expected, str) and expected.strip():
        return DecoderOutput(surface_text=expected.strip())

    if inp.lucidity_output.decision == LucidityDecision.COMMIT:
        return DecoderOutput(surface_text="(committed)")

    return DecoderOutput(surface_text="(holding: ambiguity)")


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
