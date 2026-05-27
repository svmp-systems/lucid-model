"""Default stub stage implementations.

These let the orchestrator execute the full pipeline "step-by-step" and write
audits even before real algorithmic stage implementations exist.
"""

from __future__ import annotations

from uuid import uuid4

from lucid.ir.basins import BasinInput, BasinOutput, CompetitionSummary
from lucid.ir.binding import BindingInput, BindingOutput
from lucid.ir.common import DecoderMode, LucidityDecision, Modality, TaskIntent
from lucid.ir.context_op import ContextOpInput, ContextOpOutput
from lucid.ir.cue import CueCloud, CueEncoderInput
from lucid.ir.dmf import DmfInput, DmfOutput
from lucid.ir.expression import DecoderInput, DecoderOutput
from lucid.ir.interference import InterferenceInput, InterferenceOutput
from lucid.ir.lucidity import DecoderPolicy, LucidityInput, LucidityOutput
from lucid.ir.perception import PerceptionInput, PerceptualEvidenceGraph
from lucid.ir.projector import ProjectorInput, ProjectorOutput


def _as_modality(value: Modality | str) -> Modality:
    if isinstance(value, Modality):
        return value
    return Modality(str(value))


def _lucidity_target_to_decision(target: str) -> LucidityDecision:
    # Generator gold uses strings like "COMMIT" / "PRESERVE_AMBIGUITY".
    # The IR decision enum uses lowercase values.
    raw = (target or "").strip().lower()
    if raw == "commit":
        return LucidityDecision.COMMIT
    if raw == "preserve_ambiguity":
        return LucidityDecision.PRESERVE_AMBIGUITY
    return LucidityDecision.PRESERVE_AMBIGUITY


def perception(inp: PerceptionInput, _ctx: object) -> PerceptualEvidenceGraph:
    modality = inp.modality
    surface = ""
    if modality == Modality.TEXT and isinstance(inp.raw_payload, str):
        surface = inp.raw_payload.strip()
    graph = PerceptualEvidenceGraph()
    if surface:
        # Keep this extremely small; it's only to prove wiring.
        graph.candidate_units.append(
            # type: ignore[arg-type]
            __import__("lucid.ir.perception", fromlist=["CandidateUnit"]).CandidateUnit(
                unit_id="u0",
                surface=surface[:40],
                kind_hint="raw",
                confidence=0.1,
            )
        )
    graph.provenance.modality = modality
    return graph


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
        committed_state = __import__("lucid.ir.lucidity", fromlist=["CommittedState"]).CommittedState(
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

