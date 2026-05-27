from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


class Perception:
    def run(self, stage_input):
        return {
            "entities": [{"id": "ent-1", "text": str(stage_input)}],
            "events": [{"id": "evt-1"}],
            "relations": [{"id": "rel-1"}],
        }


class FailingPerception:
    def run(self, stage_input):
        raise RuntimeError(f"cannot perceive {stage_input}")


class CueEncoder:
    def run(self, stage_input):
        return {"cue": "ok", "source": stage_input}


class DMF:
    def run(self, stage_input):
        return {"active_traces": ["trace-1"], "trace_clusters": ["cluster-1"]}


class Binding:
    def run(self, stage_input):
        return {"candidate_bindings": [{"binding_id": "bind-1"}]}


class ContextOp:
    def run(self, stage_input):
        return {
            "context_frames": [{"frame_id": "ctx-1"}],
            "scoped_trace_assignments": {"trace-1": "ctx-1"},
        }


class Interference:
    def run(self, stage_input):
        return {
            "interference_edges": ["edge-1"],
            "active_basins": [{"basin_id": "basin-1", "polluted": False}],
            "basin_assemblies": {"answer": "ok"},
        }


class Lucidity:
    def __init__(self, decision="commit", margin=0.9):
        self.decision = decision
        self.margin = margin

    def run(self, stage_input):
        return (self.decision, self.margin)


class Projector:
    def __init__(self):
        self.calls = 0

    def run(self, stage_input):
        self.calls += 1
        return {"projected_answer": "maybe"}


class Decoder:
    def run(self, stage_input):
        if stage_input.get("decision") == "test_consequence":
            projection = stage_input.get("projection_result") or {}
            return {"answer": projection.get("projected_answer", "unknown")}
        return stage_input.get("basin_assemblies", {})


def _load_orchestrator():
    sys.modules.pop("pipeline", None)
    if "orchestrator" in sys.modules:
        del sys.modules["orchestrator"]
    return importlib.import_module("orchestrator")


def _audit_dirs(base: Path) -> list[Path]:
    return sorted(path for path in base.iterdir() if path.is_dir())


def test_orchestrator_imports_without_pipeline_module():
    orchestrator_mod = _load_orchestrator()
    assert orchestrator_mod.TrainingEpisode.__name__ == "TrainingEpisode"


def test_end_to_end_success_path_no_patch(tmp_path: Path):
    orchestrator_mod = _load_orchestrator()
    episode = orchestrator_mod.TrainingEpisode(
        episode_id="ep-success",
        raw_input="hello",
        modality="text",
        task_intent="qa",
        context={},
        constraints={},
        expected_output={"answer": "ok"},
        validator_type="exact_match",
        metadata={"task_family": "chat"},
    )

    orch = orchestrator_mod.TrainingOrchestrator(
        perception=Perception(),
        cue_encoder=CueEncoder(),
        dmf=DMF(),
        binding=Binding(),
        context_op=ContextOp(),
        interference=Interference(),
        lucidity=Lucidity(decision="commit", margin=0.9),
        projector=Projector(),
        decoder=Decoder(),
        episodes=[episode],
        phase=1,
        debug=True,
        audit_base_dir=tmp_path / "audit",
    )
    orch.run(3)
    status = orch.get_status()
    assert status["patch_history_count"] == 0
    assert status["metrics"]["success_rate"] >= 0.9
    dirs = _audit_dirs(tmp_path / "audit")
    assert len(dirs) == 3
    assert (dirs[0] / "manifest.json").exists()
    assert (dirs[0] / "README.txt").exists()
    assert (dirs[0] / "run_log.json").exists()


def test_end_to_end_failure_goes_to_replay(tmp_path: Path):
    orchestrator_mod = _load_orchestrator()
    episode = orchestrator_mod.TrainingEpisode(
        episode_id="ep-fail",
        raw_input="input",
        modality="code",
        task_intent="repair",
        context={},
        constraints={},
        expected_output={"answer": "different"},
        validator_type="exact_match",
        metadata={"task_family": "unit"},
    )
    orch = orchestrator_mod.TrainingOrchestrator(
        perception=Perception(),
        cue_encoder=CueEncoder(),
        dmf=DMF(),
        binding=Binding(),
        context_op=ContextOp(),
        interference=Interference(),
        lucidity=Lucidity(decision="commit", margin=0.9),
        projector=Projector(),
        decoder=Decoder(),
        episodes=[episode],
        phase=1,
        debug=False,
        audit_base_dir=tmp_path / "audit",
    )
    orch.run_one_step()
    replay_metrics = orch.failure_replay_store.metrics()
    assert replay_metrics["failure_replay_queue_depth"] >= 1
    assert orch.get_status()["rejected_patch_count"] >= 1
    dirs = _audit_dirs(tmp_path / "audit")
    assert len(dirs) == 1
    assert (dirs[0] / "patch_result.json").exists()


def test_projector_runs_only_in_test_consequence_band(tmp_path: Path):
    orchestrator_mod = _load_orchestrator()
    projector = Projector()
    episode = orchestrator_mod.TrainingEpisode(
        episode_id="ep-mid-margin",
        raw_input="input",
        modality="multimodal",
        task_intent="predict",
        context={},
        constraints={},
        expected_output={"answer": "maybe"},
        validator_type="exact_match",
        metadata={"task_family": "vision"},
    )
    orch = orchestrator_mod.TrainingOrchestrator(
        perception=Perception(),
        cue_encoder=CueEncoder(),
        dmf=DMF(),
        binding=Binding(),
        context_op=ContextOp(),
        interference=Interference(),
        lucidity=Lucidity(decision="test_consequence", margin=0.5),
        projector=projector,
        decoder=Decoder(),
        episodes=[episode],
        phase=1,
        debug=False,
        audit_base_dir=tmp_path / "audit",
    )
    orch.run_one_step()
    assert projector.calls >= 1
    assert _audit_dirs(tmp_path / "audit")


def test_exception_path_writes_audit(tmp_path: Path):
    orchestrator_mod = _load_orchestrator()
    episode = orchestrator_mod.TrainingEpisode(
        episode_id="ep-exception",
        raw_input="bad input",
        modality="text",
        task_intent="qa",
        context={},
        constraints={},
        expected_output={"answer": "ok"},
        validator_type="exact_match",
        metadata={"task_family": "chat"},
    )
    orch = orchestrator_mod.TrainingOrchestrator(
        perception=FailingPerception(),
        cue_encoder=CueEncoder(),
        dmf=DMF(),
        binding=Binding(),
        context_op=ContextOp(),
        interference=Interference(),
        lucidity=Lucidity(decision="commit", margin=0.9),
        projector=Projector(),
        decoder=Decoder(),
        episodes=[episode],
        audit_base_dir=tmp_path / "audit",
    )

    with pytest.raises(RuntimeError, match="cannot perceive"):
        orch.run_one_step()

    dirs = _audit_dirs(tmp_path / "audit")
    assert len(dirs) == 1
    manifest = json.loads((dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["action"] == "exception"
    assert "RuntimeError: cannot perceive bad input" in manifest["error_message"]


def test_replay_clear_requires_patch_shadow_and_three_successes():
    orchestrator_mod = _load_orchestrator()
    store = orchestrator_mod.FailureReplayStore()
    run_log = orchestrator_mod.RunLog(
        episode_id="ep-replay",
        raw_input="x",
        evidence_graph={"entities": []},
        cue_cloud={},
        active_traces=[],
        trace_clusters=[],
        candidate_bindings=[],
        context_frames=[],
        scoped_trace_assignments={},
        interference_edges=[],
        active_basins=[],
        basin_assemblies={},
        lucidity_features={},
        lucidity_decision="commit",
        lucidity_margin=0.9,
        projection_result=None,
        decoder_output=None,
        validator_result={},
        cost_metrics={},
    )

    store.add_or_refresh(run_log)
    store.on_patch_promoted("ep-replay", "patch-1")
    for _ in range(3):
        store.record_success("ep-replay")

    assert store.try_clear("ep-replay") is False
    assert store.contains("ep-replay")

    store.on_episode_shadow_passed("ep-replay")
    assert store.try_clear("ep-replay") is True
    assert not store.contains("ep-replay")
