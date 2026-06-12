"""Run the real cognition pipeline inside the training orchestrator."""

from __future__ import annotations

from lucid.cognition.input.perception import PerceptionConfig
from lucid.cognition.pipe_orchestrator.runner import OrchestratorConfig, OrchestratorRunner
from lucid.cognition.pipe_orchestrator.stages import FunctionStage
from lucid.cognition.pipe_orchestrator.stub_stages import build_default_stage_fns
from lucid.ir.pipeline import PipelineRun
from lucid.ir.serde import to_dict
from lucid.ir.training import Episode
from lucid.training.loop.orchestrator import RunExecutor, RunLog, TrainingEpisode
from lucid.runtime.paths import DEFAULT_AUDIT_TRAINING_RUNS
from lucid.training.validate.gold import episode_from_training


class PipelineRunExecutor(RunExecutor):
    """Execute ``TrainingEpisode`` values through ``OrchestratorRunner``."""

    def __init__(
        self,
        *,
        checkpoint: str = "",
        audit_base_dir: str = DEFAULT_AUDIT_TRAINING_RUNS,
        perception_backend: str = "rule",
    ) -> None:
        super().__init__(
            perception=_PipelineStage("perception"),
            cue_encoder=_PipelineStage("cue_encoder"),
            dmf=_PipelineStage("dmf"),
            binding=_PipelineStage("binding"),
            context_op=_PipelineStage("context_op"),
            interference=_PipelineStage("interference"),
            lucidity=_PipelineStage("lucidity"),
            projector=_PipelineStage("projector"),
            decoder=_PipelineStage("decoder"),
        )
        stage_fns = build_default_stage_fns()
        stages = {
            name: FunctionStage(stage_name=name, fn=fn)  # type: ignore[arg-type]
            for name, fn in stage_fns.items()
        }
        self._runner = OrchestratorRunner(
            config=OrchestratorConfig(
                audit_base_dir=audit_base_dir,
                checkpoint=checkpoint or None,
                perception=PerceptionConfig(backend=perception_backend, write_audit=False),
            ),
            stages=stages,
        )
        self._checkpoint = checkpoint

    def run(self, episode: TrainingEpisode, mode: str = "training_observation") -> RunLog:
        _ = mode
        return self._run_pipeline_episode(episode)

    def _run_from_states(self, episode: TrainingEpisode, state: dict | None = None) -> RunLog:
        _ = state
        return self._run_pipeline_episode(episode)

    def _run_pipeline_episode(self, episode: TrainingEpisode) -> RunLog:
        ir_episode = episode_from_training(episode)
        pipeline_run = self._runner.run_episode(ir_episode)
        return pipeline_run_to_run_log(pipeline_run, ir_episode)


class _PipelineStage:
    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, stage_input: object) -> object:
        raise RuntimeError(f"pipeline stage {self.name} must run through OrchestratorRunner")


def pipeline_run_to_run_log(run: PipelineRun, episode: Episode) -> RunLog:
    active_traces = []
    if run.dmf_output is not None:
        active_traces = [trace.trace_id for trace in run.dmf_output.active_traces if trace.trace_id]

    candidate_bindings = []
    if run.binding_output is not None:
        candidate_bindings = [to_dict(frame) for frame in run.binding_output.candidate_frames]

    context_frames = []
    scoped: dict[str, str] = {}
    if run.context_op_output is not None:
        context_frames = [to_dict(frame) for frame in run.context_op_output.context_frames]
        for assignment in run.context_op_output.scoped_trace_assignments:
            scoped[assignment.trace_id] = assignment.primary_context_frame_id

    interference_edges = []
    active_basins = []
    if run.interference_output is not None:
        interference_edges = [to_dict(edge) for edge in run.interference_output.trace_trace_edges]
        if run.interference_output.scoped_basin_energy_deltas:
            active_basins = [
                {"basin_id": delta.basin_id, "energy": delta.delta, "scope_frame_id": delta.scope_frame_id}
                for delta in run.interference_output.scoped_basin_energy_deltas
            ]
        else:
            active_basins = [
                {"basin_id": basin_id, "energy": energy}
                for basin_id, energy in run.interference_output.basin_energy_deltas.items()
            ]

    lucidity_decision = ""
    lucidity_margin = 0.0
    if run.lucidity_output is not None:
        lucidity_decision = str(run.lucidity_output.decision.value if hasattr(run.lucidity_output.decision, "value") else run.lucidity_output.decision)
        if run.lucidity_output.confidence_summary is not None:
            lucidity_margin = float(run.lucidity_output.confidence_summary.overall_confidence)

    decoder_output = None
    if run.decoder_output is not None:
        if run.decoder_output.surface_grid is not None:
            decoder_output = run.decoder_output.surface_grid
        else:
            decoder_output = run.decoder_output.surface_text

    projection_result = None
    projector_called = run.projector_output is not None
    if run.projector_output is not None:
        projection_result = to_dict(run.projector_output)

    return RunLog(
        episode_id=episode.episode_id,
        raw_input=episode.raw_input,
        evidence_graph=to_dict(run.evidence_graph) if run.evidence_graph is not None else {},
        cue_cloud=to_dict(run.cue_cloud) if run.cue_cloud is not None else {},
        active_traces=active_traces,
        trace_clusters=[],
        candidate_bindings=candidate_bindings,
        context_frames=context_frames,
        scoped_trace_assignments=scoped,
        interference_edges=interference_edges,
        active_basins=active_basins,
        basin_assemblies={},
        lucidity_features={"decision": lucidity_decision},
        lucidity_decision=lucidity_decision,
        lucidity_margin=lucidity_margin,
        projection_result=projection_result,
        decoder_output=decoder_output,
        validator_result={},
        cost_metrics={
            "stages_run": 10 if projector_called else 9,
            "projector_called": projector_called,
            "wall_time_ms": float(run.cost_metrics.wall_time_ms),
        },
    )
