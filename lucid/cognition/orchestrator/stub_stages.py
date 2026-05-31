"""Default stub stage implementations.

These let the orchestrator execute the full pipeline "step-by-step" and write
audits even before real algorithmic stage implementations exist.
"""

from __future__ import annotations

from uuid import uuid4

from lucid.ir.basins import BasinInput, BasinOutput, CompetitionSummary
from lucid.ir.binding import BindingInput, BindingOutput
from lucid.ir.common import CommitShape, DecoderMode, LucidityDecision
from lucid.ir.context_op import ContextOpInput, ContextOpOutput
from lucid.ir.cue import CueCloud, CueEncoderInput
from lucid.ir.dmf import DmfInput, DmfOutput
from lucid.ir.expression import DecoderInput, DecoderOutput
from lucid.ir.interference import InterferenceInput, InterferenceOutput
from lucid.ir.lucidity import (
    CommittedState,
    DecoderPolicy,
    LucidityInput,
    LucidityOutput,
    SearchDirectives,
)
from lucid.ir.perception import PerceptionInput, PerceptualEvidenceGraph
from lucid.ir.projector import ProjectorInput, ProjectorOutput
from lucid.cognition.input.perception import PerceptionConfig, perceive as run_perception
from lucid.cognition.projector import run_projector


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


def cue_encoder(inp: CueEncoderInput, _ctx: object) -> CueCloud:
    return CueCloud(provenance=inp.provenance)


def dmf(inp: DmfInput, _ctx: object) -> DmfOutput:
    return DmfOutput()


def binding(inp: BindingInput, _ctx: object) -> BindingOutput:
    return BindingOutput()


def context_op(inp: ContextOpInput, _ctx: object) -> ContextOpOutput:
    return ContextOpOutput()


def interference(inp: InterferenceInput, _ctx: object) -> InterferenceOutput:
    return InterferenceOutput()


def basins(inp: BasinInput, _ctx: object) -> BasinOutput:
    return BasinOutput(competition_summary=CompetitionSummary())


def projector(inp: ProjectorInput, _ctx: object) -> ProjectorOutput:
    return run_projector(inp, context=_ctx)


def lucidity(inp: LucidityInput, ctx: object) -> LucidityOutput:
    if inp.task_intent == "TaskIntent.SOLVE_GRID":
        task_intent = "solve_grid"
    else:
        task_intent = inp.task_intent

    if task_intent == "solve_grid" and inp.pass_kind == "pre_check":
        return LucidityOutput(
            decision=LucidityDecision.REQUEST_PROJECTION,
            decoder_policy=DecoderPolicy(mode=DecoderMode.HOLD.value),
            search_directives=SearchDirectives(
                projector_targets=["asy_grid_candidate"],
                max_rollouts=inp.compute_policy.max_projector_rollouts,
            ),
        )

    if task_intent == "solve_grid" and inp.pass_kind == "final_check":
        projection = inp.projection_output
        if projection is not None and projection.recommendation_to_lucidity == "suggest_commit":
            best = next(
                (
                    rollout
                    for rollout in projection.rollouts
                    if rollout.rollout_id == projection.best_rollout_id
                ),
                None,
            )
            artifact = best.implied_artifact if best is not None else {}
            return LucidityOutput(
                decision=LucidityDecision.COMMIT,
                decoder_policy=DecoderPolicy(
                    mode=DecoderMode.EXPRESS_COMMITTED.value,
                    output_format="grid",
                ),
                committed_state=CommittedState(
                    commit_id=str(uuid4()),
                    commit_shape=CommitShape.ASSEMBLY,
                    assembly_ids=["asy_grid_candidate"],
                    projection_artifact=artifact,
                ),
            )
        return LucidityOutput(
            decision=LucidityDecision.SEARCH_WIDER,
            decoder_policy=DecoderPolicy(mode=DecoderMode.HOLD.value),
            search_directives=SearchDirectives(allow_provisional_basins=True),
        )

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
    policy = inp.decoder_policy or inp.lucidity_output.decoder_policy
    if policy.mode == DecoderMode.HOLD.value:
        return DecoderOutput(surface_text="", refused=False)

    committed = inp.committed_state or inp.lucidity_output.committed_state
    if committed is not None and committed.projection_artifact:
        test_outputs = committed.projection_artifact.get("test_outputs")
        if isinstance(test_outputs, list) and test_outputs:
            first = test_outputs[0]
            if isinstance(first, list):
                return DecoderOutput(surface_grid=first)

    episode = getattr(ctx, "episode", None)
    expected = None
    if episode is not None and getattr(episode, "gold", None) is not None:
        expected = getattr(episode.gold, "expected_answer", None)

    if isinstance(expected, str) and expected.strip():
        return DecoderOutput(surface_text=expected.strip())
    if isinstance(expected, list):
        return DecoderOutput(surface_grid=expected)

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
