"""Step-by-step orchestrator runner.

This runner executes each stage sequentially, records timing/success, and
writes audit artifacts using `lucid.audit.logger.AuditLogger`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import uuid4

from lucid.audit.logger import AuditLogger
from lucid.ir.basins import BasinInput, BasinOutput
from lucid.ir.binding import BindingInput, BindingOutput
from lucid.ir.common import Modality, TaskIntent
from lucid.ir.context_op import ContextOpInput, ContextOpOutput
from lucid.ir.cue import CueCloud, CueEncoderInput
from lucid.ir.common import AmbiguityPolicy
from lucid.ir.dmf import DmfInput, DmfOutput
from lucid.ir.expression import DecoderInput, DecoderOutput
from lucid.ir.interference import InterferenceInput, InterferenceOutput
from lucid.ir.lucidity import LucidityInput, LucidityOutput
from lucid.ir.perception import PerceptionInput, PerceptualEvidenceGraph
from lucid.ir.pipeline import PipelineRun, RunContext, StageName, StageResult
from lucid.ir.projector import ProjectorInput, ProjectorOutput
from lucid.ir.training import Episode
from lucid.orchestrator.stages import FunctionStage, Stage
from lucid.orchestrator.stub_stages import build_default_stage_fns


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


@dataclass(slots=True)
class OrchestratorConfig:
    audit_base_dir: str = "audit"
    adapter_version: str = "0.1.0"
    enable_projector: bool = True
    max_iterations: int = 2
    widen_retrieval_budget_multiplier: float = 1.5


class OrchestratorRunner:
    """Run all pipeline stages in a fixed order.

    In Phase 1 this provides "everything runs properly" wiring + auditing.
    Real stage implementations can be injected via `stages=...`.
    """

    def __init__(self, *, config: OrchestratorConfig | None = None, stages: dict[str, Stage] | None = None):
        self.config = config or OrchestratorConfig()
        if stages is None:
            stages = {
                name: FunctionStage(stage_name=name, fn=fn)  # type: ignore[arg-type]
                for name, fn in build_default_stage_fns().items()
            }
        self.stages = stages
        self.audit = AuditLogger(base_dir=self.config.audit_base_dir, adapter_version=self.config.adapter_version)

    def run_episode(self, episode: Episode, *, session_id: str = "", turn_index: int = 0) -> PipelineRun:
        ctx = RunContext(
            run_id=str(uuid4()),
            session_id=session_id,
            turn_index=turn_index,
            mode="inference",
            task_intent=episode.task_intent if isinstance(episode.task_intent, TaskIntent) else TaskIntent(str(episode.task_intent)),
            episode=None,
        )
        # Keep a reference for stub stages (e.g. lucidity/decoder).
        ctx.episode = episode  # type: ignore[assignment]

        run = PipelineRun(context=ctx)
        t0 = _now_ms()

        # 1) Perception (run once; downstream stages may iterate)
        modality = episode.modality if isinstance(episode.modality, Modality) else Modality(str(episode.modality))
        run.perception_input = PerceptionInput(raw_payload=episode.raw_input, modality=modality)
        run.evidence_graph = self._run_stage(StageName.PERCEPTION.value, run, run.perception_input)

        # 2–9) Iterative pipeline core (cue → ... → basins → lucidity → optional projector → lucidity)
        # The loop exists to support SEARCH_WIDER / RECHECK_BINDING and REQUEST_PROJECTION flows.
        retrieval_budget = 128
        ambiguity_policy = AmbiguityPolicy.PRESERVE_PLURAL
        lucidity_feedback: list[str] = []

        for iteration in range(max(1, int(self.config.max_iterations))):
            ctx.iteration_count = iteration

            # 2) Cue encoder
            run.cue_encoder_input = CueEncoderInput(
                perceptual_evidence_graph=run.evidence_graph,
                task_intent_hint=str(episode.task_intent),
                retrieval_budget=retrieval_budget,
                ambiguity_policy_in=ambiguity_policy,
            )
            run.cue_cloud = self._run_stage(StageName.CUE_ENCODER.value, run, run.cue_encoder_input)

            # 3) DMF
            run.dmf_input = DmfInput(cue_cloud=run.cue_cloud)
            run.dmf_output = self._run_stage(StageName.DMF.value, run, run.dmf_input)

            # 4) Binding
            run.binding_input = BindingInput(
                dmf_output=run.dmf_output,
                perceptual_evidence_graph=run.evidence_graph,
                cue_cloud=run.cue_cloud,
            )
            run.binding_output = self._run_stage(StageName.BINDING.value, run, run.binding_input)

            # 5) Context-op
            run.context_op_input = ContextOpInput(
                binding_candidate_frames=run.binding_output.candidate_frames,
                dmf_output=run.dmf_output,
                perceptual_evidence_graph=run.evidence_graph,
                lucidity_feedback=lucidity_feedback,
            )
            run.context_op_output = self._run_stage(StageName.CONTEXT_OP.value, run, run.context_op_input)

            # 6) Interference
            run.interference_input = InterferenceInput(
                context_frames=run.context_op_output.context_frames,
                candidate_frames=run.binding_output.candidate_frames,
                dmf_output=run.dmf_output,
                interference_gates=run.context_op_output.interference_gates,
            )
            run.interference_output = self._run_stage(StageName.INTERFERENCE.value, run, run.interference_input)

            # 7) Basins
            run.basin_input = BasinInput(
                interference_output=run.interference_output,
                candidate_frames=run.binding_output.candidate_frames,
                context_frames=run.context_op_output.context_frames,
                local_basin_pressures=run.context_op_output.local_basin_pressures,
            )
            run.basin_output = self._run_stage(StageName.BASINS.value, run, run.basin_input)

            # 8) Lucidity pre-check (no projection by default)
            run.projector_input = None
            run.projector_output = None
            run.lucidity_input = LucidityInput(
                basin_output=run.basin_output,
                binding_output=run.binding_output,
                context_op_output=run.context_op_output,
                interference_output=run.interference_output,
                dmf_output=run.dmf_output,
                perceptual_evidence_graph=run.evidence_graph,
                task_intent=str(episode.task_intent),
                projection_output=None,
                pass_kind="pre_check",
                iteration_count=ctx.iteration_count,
            )
            run.lucidity_output = self._run_stage(StageName.LUCIDITY.value, run, run.lucidity_input)

            # 9) Optional projector + lucidity final-check (only if requested)
            if (
                self.config.enable_projector
                and run.lucidity_output.decision == __import__("lucid.ir.common", fromlist=["LucidityDecision"]).LucidityDecision.REQUEST_PROJECTION
            ):
                directives = run.lucidity_output.search_directives or __import__(
                    "lucid.ir.lucidity", fromlist=["SearchDirectives"]
                ).SearchDirectives()
                run.projector_input = ProjectorInput(
                    projection_request=directives,
                    candidate_frames=run.binding_output.candidate_frames,
                    context_frames=run.context_op_output.context_frames,
                    perceptual_evidence_graph=run.evidence_graph,
                    task_intent=str(episode.task_intent),
                )
                run.projector_output = self._run_stage(StageName.PROJECTOR.value, run, run.projector_input)

                run.lucidity_input = LucidityInput(
                    basin_output=run.basin_output,
                    binding_output=run.binding_output,
                    context_op_output=run.context_op_output,
                    interference_output=run.interference_output,
                    dmf_output=run.dmf_output,
                    perceptual_evidence_graph=run.evidence_graph,
                    task_intent=str(episode.task_intent),
                    projection_output=run.projector_output,
                    pass_kind="final_check",
                    iteration_count=ctx.iteration_count,
                )
                run.lucidity_output = self._run_stage(StageName.LUCIDITY.value, run, run.lucidity_input)

            # If lucidity wants to loop, adjust policy and continue; otherwise stop.
            decision = run.lucidity_output.decision
            LucidityDecision = __import__("lucid.ir.common", fromlist=["LucidityDecision"]).LucidityDecision
            if decision == LucidityDecision.SEARCH_WIDER:
                lucidity_feedback = ["SEARCH_WIDER"]
                ambiguity_policy = AmbiguityPolicy.FORCE_WIDEN
                retrieval_budget = max(
                    retrieval_budget + 1,
                    int(retrieval_budget * float(self.config.widen_retrieval_budget_multiplier)),
                )
                continue
            if decision == LucidityDecision.RECHECK_BINDING:
                lucidity_feedback = ["RECHECK_BINDING"]
                # Keep retrieval stable; binding should change.
                ambiguity_policy = AmbiguityPolicy.PRESERVE_PLURAL
                continue

            break

        # 10) Decoder
        run.decoder_input = DecoderInput(
            lucidity_output=run.lucidity_output,
            committed_state=run.lucidity_output.committed_state,
            decoder_policy=run.lucidity_output.decoder_policy,
        )
        run.decoder_output = self._run_stage(StageName.DECODER.value, run, run.decoder_input)

        run.cost_metrics.wall_time_ms = max(0.0, _now_ms() - t0)

        self.audit.write_pipeline_run(run)
        return run

    def _run_stage(self, stage_name: str, run: PipelineRun, stage_input):
        stage = self.stages.get(stage_name)
        if stage is None:
            raise KeyError(f"missing stage implementation: {stage_name}")

        start = _now_ms()
        success = True
        error_message = ""
        output = None
        try:
            output = stage.run(stage_input, context=run.context)
        except Exception as exc:  # noqa: BLE001 - orchestrator must capture all failures
            success = False
            error_message = f"{type(exc).__name__}: {exc}"
        duration_ms = max(0.0, _now_ms() - start)

        run.stage_results.append(
            StageResult(
                stage_name=stage_name,
                success=success,
                duration_ms=duration_ms,
                output_type=type(output).__name__ if output is not None else "",
                error_message=error_message,
            )
        )
        run.cost_metrics.stage_times_ms[stage_name] = duration_ms

        if not success:
            raise RuntimeError(f"stage {stage_name} failed: {error_message}")
        return output

