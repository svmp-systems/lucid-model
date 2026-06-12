"""Step-by-step orchestrator runner.

This runner executes each stage sequentially, records timing/success, and
writes audit artifacts using ``lucid.audit.logger.AuditLogger``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from lucid.audit.logger import AuditLogger
from lucid.runtime.paths import DEFAULT_AUDIT_RUNS, resolve_train_path
from lucid.ir.basins import BasinInput
from lucid.ir.binding import BindingInput
from lucid.ir.common import AmbiguityPolicy, LucidityDecision, Modality, TaskIntent
from lucid.ir.context_op import ContextOpInput
from lucid.ir.cue import CueEncoderInput
from lucid.ir.dmf import DmfInput
from lucid.ir.expression import DecoderInput
from lucid.ir.interference import InterferenceInput
from lucid.ir.lucidity import LucidityInput, SearchDirectives
from lucid.ir.perception import PerceptionInput
from lucid.ir.pipeline import (
    PipelineRun,
    RunContext,
    SessionState,
    StageExecutionRecord,
    StageName,
    StageResult,
)
from lucid.ir.projector import ProjectionConstraints, ProjectionGridPair, ProjectorInput
from lucid.ir.serde import to_dict
from lucid.ir.training import Episode
from lucid.cognition.input.perception import PerceptionConfig

from .stages import FunctionStage, Stage
from .stub_stages import build_default_stage_fns


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _stage_key(stage_name: StageName | str) -> str:
    if isinstance(stage_name, StageName):
        return stage_name.value
    return str(stage_name)


def _pipeline_run_id(episode: Episode) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = (episode.episode_id or episode.template_id or "episode").strip()
    clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40].strip("_") or "episode"
    return f"{stamp}_{clean}_{uuid4().hex[:6]}"


def _is_grid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, list) for row in value)
        and all(isinstance(cell, int) for row in value for cell in row)
    )


def _grid_pairs_from_raw(raw_input: Any) -> list[ProjectionGridPair]:
    if not isinstance(raw_input, dict):
        return []
    pairs: list[ProjectionGridPair] = []
    raw_pairs = raw_input.get("train") or raw_input.get("train_pairs") or []
    if isinstance(raw_pairs, list):
        for index, item in enumerate(raw_pairs):
            if not isinstance(item, dict):
                continue
            input_grid = item.get("input") or item.get("input_grid")
            output_grid = item.get("output") or item.get("output_grid")
            if _is_grid(input_grid) and _is_grid(output_grid):
                pairs.append(
                    ProjectionGridPair(
                        pair_id=str(item.get("pair_id") or item.get("id") or f"pair_{index}"),
                        input_grid=input_grid,
                        output_grid=output_grid,
                    )
                )

    input_grid = raw_input.get("input") or raw_input.get("input_grid")
    output_grid = raw_input.get("output") or raw_input.get("output_grid")
    if _is_grid(input_grid) and _is_grid(output_grid):
        pairs.append(
            ProjectionGridPair(
                pair_id="pair_0" if not pairs else f"pair_{len(pairs)}",
                input_grid=input_grid,
                output_grid=output_grid,
            )
        )
    return pairs


def _test_inputs_from_raw(raw_input: Any) -> list[list[list[int]]]:
    if not isinstance(raw_input, dict):
        return []
    raw_tests = raw_input.get("test") or raw_input.get("test_inputs") or []
    tests: list[list[list[int]]] = []
    if _is_grid(raw_tests):
        tests.append(raw_tests)
    elif isinstance(raw_tests, list):
        for item in raw_tests:
            if _is_grid(item):
                tests.append(item)
            elif isinstance(item, dict):
                candidate = item.get("input") or item.get("input_grid")
                if _is_grid(candidate):
                    tests.append(candidate)
    input_grid = raw_input.get("input") or raw_input.get("input_grid")
    if not tests and _is_grid(input_grid):
        tests.append(input_grid)
    return tests


def _projection_constraints_from_episode(episode: Episode) -> ProjectionConstraints:
    raw_constraints = episode.constraints if isinstance(episode.constraints, dict) else {}
    projection = raw_constraints.get("projection")
    if not isinstance(projection, dict):
        projection = raw_constraints

    max_rollouts = projection.get("max_rollouts", 4)
    try:
        max_rollouts_int = int(max_rollouts)
    except (TypeError, ValueError):
        max_rollouts_int = 4

    train_pairs = _grid_pairs_from_raw(episode.raw_input)
    for index, item in enumerate(projection.get("train_pairs") or []):
        if not isinstance(item, dict):
            continue
        input_grid = item.get("input") or item.get("input_grid")
        output_grid = item.get("output") or item.get("output_grid")
        if _is_grid(input_grid) and _is_grid(output_grid):
            pair_id = str(
                item.get("pair_id") or item.get("id") or f"constraint_pair_{index}"
            )
            train_pairs.append(
                ProjectionGridPair(
                    pair_id=pair_id,
                    input_grid=input_grid,
                    output_grid=output_grid,
                )
            )

    test_inputs = _test_inputs_from_raw(episode.raw_input)
    for item in projection.get("test_inputs") or []:
        if _is_grid(item):
            test_inputs.append(item)

    output_shape_rules = projection.get("output_shape_rules") or {}
    return ProjectionConstraints(
        output_shape_rules=output_shape_rules if isinstance(output_shape_rules, dict) else {},
        train_pair_refs=[str(ref) for ref in projection.get("train_pair_refs") or []],
        test_input_refs=[str(ref) for ref in projection.get("test_input_refs") or []],
        train_pairs=train_pairs,
        test_inputs=test_inputs,
        max_rollouts=max_rollouts_int,
    )


@dataclass(slots=True)
class OrchestratorConfig:
    audit_base_dir: str = DEFAULT_AUDIT_RUNS
    adapter_version: str = "0.1.0"
    enable_projector: bool = True
    max_iterations: int = 2
    widen_retrieval_budget_multiplier: float = 1.5
    perception: PerceptionConfig | None = None
    checkpoint: str = ""


class OrchestratorRunner:
    """Run all pipeline stages in a fixed order.

    Real stage implementations can be injected via ``stages=...``. The default
    stages keep the pipeline executable offline and auditable end to end.
    """

    def __init__(
        self,
        *,
        config: OrchestratorConfig | None = None,
        stages: dict[str, Stage] | None = None,
    ) -> None:
        self.config = config or OrchestratorConfig()
        if stages is None:
            stages = {
                name: FunctionStage(stage_name=name, fn=fn)  # type: ignore[arg-type]
                for name, fn in build_default_stage_fns().items()
            }
        self.stages = stages
        self.audit = AuditLogger(
            base_dir=self.config.audit_base_dir,
            adapter_version=self.config.adapter_version,
        )

    def run_episode(
        self,
        episode: Episode,
        *,
        session_id: str = "",
        turn_index: int = 0,
        session_state: SessionState | None = None,
    ) -> PipelineRun:
        ctx = RunContext(
            run_id=_pipeline_run_id(episode),
            session_id=session_id,
            turn_index=turn_index,
            mode="inference",
            task_intent=(
                episode.task_intent
                if isinstance(episode.task_intent, TaskIntent)
                else TaskIntent(str(episode.task_intent))
            ),
            episode=None,
            session_state=session_state,
        )
        # Keep a reference for runtime stages that need training metadata.
        ctx.episode = episode  # type: ignore[assignment]
        perception_cfg = self.config.perception or PerceptionConfig.from_env()
        if perception_cfg.write_audit:
            perception_cfg = replace(
                perception_cfg,
                audit_dir=str(self.audit.run_directory(ctx) / "perception"),
            )
        ctx.extra["perception_config"] = perception_cfg
        if self.config.checkpoint:
            ctx.extra["checkpoint"] = self.config.checkpoint
        ctx.extra["template_id"] = str(episode.template_id or "")
        ctx.extra["episode_id"] = str(episode.episode_id or "")
        ctx.extra["audit_base_dir"] = str(resolve_train_path(self.config.audit_base_dir))
        if episode.context:
            ctx.extra["episode_context"] = episode.context
            session_context = episode.context.get("session_context")
            if isinstance(session_context, dict):
                ctx.extra["session_context"] = session_context

        run = PipelineRun(context=ctx)
        t0 = _now_ms()
        try:
            self._execute_episode(run, episode)
        except Exception as exc:
            run.cost_metrics.wall_time_ms = max(0.0, _now_ms() - t0)
            try:
                self.audit.write_pipeline_run(run)
            except Exception as audit_exc:  # noqa: BLE001 - keep original stage failure primary
                exc.add_note(f"audit write failed after pipeline failure: {audit_exc!r}")
            raise

        run.cost_metrics.wall_time_ms = max(0.0, _now_ms() - t0)
        self.audit.write_pipeline_run(run)
        try:
            from lucid.audit.scaling import record_pipeline_run

            record_pipeline_run(run)
        except Exception:  # noqa: BLE001 — scaling must not break pipeline runs
            pass
        return run

    def _execute_episode(self, run: PipelineRun, episode: Episode) -> None:
        # 1) Perception runs once; downstream stages may iterate.
        modality = (
            episode.modality
            if isinstance(episode.modality, Modality)
            else Modality(str(episode.modality))
        )
        run.perception_input = PerceptionInput(
            raw_payload=episode.raw_input,
            modality=modality,
            task_intent_hint=run.context.task_intent,
            prior_context=episode.context,
            provenance_seed=run.context.session_id,
        )
        run.evidence_graph = self._run_stage(StageName.PERCEPTION.value, run, run.perception_input)

        # 2-9) Iterative pipeline core.
        retrieval_budget = 128
        ambiguity_policy = AmbiguityPolicy.PRESERVE_PLURAL
        lucidity_feedback: list[str] = []
        prev_dmf_coverage: float | None = None

        prior_binding_frames: list = []
        for iteration in range(max(1, int(self.config.max_iterations))):
            run.context.iteration_count = iteration
            run.context.extra["lucidity_feedback"] = list(lucidity_feedback)
            if prior_binding_frames:
                run.context.extra["prior_candidate_frames"] = prior_binding_frames

            upstream_state: dict[str, Any] = {}
            if prev_dmf_coverage is not None:
                upstream_state["dmf_coverage_score"] = prev_dmf_coverage

            run.cue_encoder_input = CueEncoderInput(
                perceptual_evidence_graph=run.evidence_graph,
                upstream_state=upstream_state,
                task_intent_hint=str(episode.task_intent),
                retrieval_budget=retrieval_budget,
                ambiguity_policy_in=ambiguity_policy,
            )
            run.cue_cloud = self._run_stage(
                StageName.CUE_ENCODER.value,
                run,
                run.cue_encoder_input,
            )

            run.dmf_input = DmfInput(cue_cloud=run.cue_cloud)
            run.dmf_output = self._run_stage(StageName.DMF.value, run, run.dmf_input)
            prev_dmf_coverage = float(run.dmf_output.coverage_score)

            run.binding_input = BindingInput(
                dmf_output=run.dmf_output,
                perceptual_evidence_graph=run.evidence_graph,
                cue_cloud=run.cue_cloud,
                prior_candidate_frames=list(prior_binding_frames),
            )
            run.binding_output = self._run_stage(
                StageName.BINDING.value,
                run,
                run.binding_input,
            )
            prior_binding_frames = list(run.binding_output.candidate_frames)

            run.context_op_input = ContextOpInput(
                binding_candidate_frames=run.binding_output.candidate_frames,
                dmf_output=run.dmf_output,
                perceptual_evidence_graph=run.evidence_graph,
                cue_cloud=run.cue_cloud,
                lucidity_feedback=lucidity_feedback,
            )
            run.context_op_output = self._run_stage(
                StageName.CONTEXT_OP.value,
                run,
                run.context_op_input,
            )

            run.interference_input = InterferenceInput(
                context_frames=run.context_op_output.context_frames,
                candidate_frames=run.binding_output.candidate_frames,
                dmf_output=run.dmf_output,
                interference_gates=run.context_op_output.interference_gates,
                scoped_trace_assignments=run.context_op_output.scoped_trace_assignments,
                frame_links=run.context_op_output.frame_links,
                local_basin_pressures=run.context_op_output.local_basin_pressures,
            )
            run.interference_output = self._run_stage(
                StageName.INTERFERENCE.value,
                run,
                run.interference_input,
            )

            run.basin_input = BasinInput(
                interference_output=run.interference_output,
                candidate_frames=run.binding_output.candidate_frames,
                context_frames=run.context_op_output.context_frames,
                local_basin_pressures=run.context_op_output.local_basin_pressures,
            )
            run.basin_output = self._run_stage(StageName.BASINS.value, run, run.basin_input)

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
                iteration_count=run.context.iteration_count,
            )
            run.lucidity_output = self._run_stage(
                StageName.LUCIDITY.value,
                run,
                run.lucidity_input,
            )

            if (
                self.config.enable_projector
                and run.lucidity_output.decision == LucidityDecision.REQUEST_PROJECTION
            ):
                directives = run.lucidity_output.search_directives or SearchDirectives()
                run.projector_input = ProjectorInput(
                    projection_request=directives,
                    target_basin_ids=list(directives.projector_targets),
                    candidate_frames=run.binding_output.candidate_frames,
                    context_frames=run.context_op_output.context_frames,
                    perceptual_evidence_graph=run.evidence_graph,
                    constraints=_projection_constraints_from_episode(episode),
                    task_intent=str(episode.task_intent),
                )
                run.projector_output = self._run_stage(
                    StageName.PROJECTOR.value,
                    run,
                    run.projector_input,
                )
                run.cost_metrics.projector_rollout_count = len(run.projector_output.rollouts)

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
                    iteration_count=run.context.iteration_count,
                )
                run.lucidity_output = self._run_stage(
                    StageName.LUCIDITY.value,
                    run,
                    run.lucidity_input,
                )

            decision = run.lucidity_output.decision
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
                ambiguity_policy = AmbiguityPolicy.PRESERVE_PLURAL
                continue

            break

        run.decoder_input = DecoderInput(
            lucidity_output=run.lucidity_output,
            render_packet=run.lucidity_output.render_packet,
            committed_state=run.lucidity_output.committed_state,
            decoder_policy=run.lucidity_output.decoder_policy,
        )
        run.decoder_output = self._run_stage(StageName.DECODER.value, run, run.decoder_input)

    def _run_stage(self, stage_name: str, run: PipelineRun, stage_input: Any) -> Any:
        stage = self.stages.get(stage_name)
        if stage is None:
            error_message = f"KeyError: missing stage implementation: {stage_name}"
            self._record_stage(
                run=run,
                stage_name=stage_name,
                stage_input=stage_input,
                stage_output=None,
                success=False,
                duration_ms=0.0,
                error_message=error_message,
            )
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

        self._record_stage(
            run=run,
            stage_name=stage_name,
            stage_input=stage_input,
            stage_output=output,
            success=success,
            duration_ms=duration_ms,
            error_message=error_message,
        )

        if not success:
            raise RuntimeError(f"stage {stage_name} failed: {error_message}")
        return output

    def _record_stage(
        self,
        *,
        run: PipelineRun,
        stage_name: str,
        stage_input: Any,
        stage_output: Any,
        success: bool,
        duration_ms: float,
        error_message: str,
    ) -> None:
        stage_index = len(run.stage_results)
        occurrence = 1 + sum(
            1 for result in run.stage_results if _stage_key(result.stage_name) == stage_name
        )
        result = StageResult(
            stage_name=stage_name,
            success=success,
            duration_ms=duration_ms,
            output_type=type(stage_output).__name__ if stage_output is not None else "",
            error_message=error_message,
            stage_index=stage_index,
            occurrence=occurrence,
        )
        run.stage_results.append(result)
        run.stage_records.append(
            StageExecutionRecord(
                stage_name=stage_name,
                stage_index=stage_index,
                occurrence=occurrence,
                input_payload=to_dict(stage_input),
                output_payload=to_dict(stage_output),
            )
        )

        metric_key = stage_name if occurrence == 1 else f"{stage_name}_{occurrence:02d}"
        run.cost_metrics.stage_times_ms[metric_key] = duration_ms
