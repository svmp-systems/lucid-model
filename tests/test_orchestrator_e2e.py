from __future__ import annotations

import importlib
import sys
import types


def _install_stub_pipeline():
    module = types.ModuleType("pipeline")

    class Perception:
        def run(self, stage_input):
            return {
                "entities": [{"id": "ent-1", "text": str(stage_input)}],
                "events": [{"id": "evt-1"}],
                "relations": [{"id": "rel-1"}],
            }

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

    module.Perception = Perception
    module.CueEncoder = CueEncoder
    module.DMF = DMF
    module.Binding = Binding
    module.ContextOp = ContextOp
    module.Interference = Interference
    module.Lucidity = Lucidity
    module.Projector = Projector
    module.Decoder = Decoder
    sys.modules["pipeline"] = module
    return module


def _load_orchestrator():
    _install_stub_pipeline()
    if "orchestrator" in sys.modules:
        del sys.modules["orchestrator"]
    return importlib.import_module("orchestrator")


def test_end_to_end_success_path_no_patch():
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
        perception=orchestrator_mod.Perception(),
        cue_encoder=orchestrator_mod.CueEncoder(),
        dmf=orchestrator_mod.DMF(),
        binding=orchestrator_mod.Binding(),
        context_op=orchestrator_mod.ContextOp(),
        interference=orchestrator_mod.Interference(),
        lucidity=orchestrator_mod.Lucidity(decision="commit", margin=0.9),
        projector=orchestrator_mod.Projector(),
        decoder=orchestrator_mod.Decoder(),
        episodes=[episode],
        phase=1,
        debug=True,
    )
    orch.run(3)
    status = orch.get_status()
    assert status["patch_history_count"] == 0
    assert status["metrics"]["success_rate"] >= 0.9


def test_end_to_end_failure_goes_to_replay():
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
        perception=orchestrator_mod.Perception(),
        cue_encoder=orchestrator_mod.CueEncoder(),
        dmf=orchestrator_mod.DMF(),
        binding=orchestrator_mod.Binding(),
        context_op=orchestrator_mod.ContextOp(),
        interference=orchestrator_mod.Interference(),
        lucidity=orchestrator_mod.Lucidity(decision="commit", margin=0.9),
        projector=orchestrator_mod.Projector(),
        decoder=orchestrator_mod.Decoder(),
        episodes=[episode],
        phase=1,
        debug=False,
    )
    orch.run_one_step()
    replay_metrics = orch.failure_replay_store.metrics()
    assert replay_metrics["failure_replay_queue_depth"] >= 1
    assert orch.get_status()["rejected_patch_count"] >= 1


def test_projector_runs_only_in_test_consequence_band():
    orchestrator_mod = _load_orchestrator()
    projector = orchestrator_mod.Projector()
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
        perception=orchestrator_mod.Perception(),
        cue_encoder=orchestrator_mod.CueEncoder(),
        dmf=orchestrator_mod.DMF(),
        binding=orchestrator_mod.Binding(),
        context_op=orchestrator_mod.ContextOp(),
        interference=orchestrator_mod.Interference(),
        lucidity=orchestrator_mod.Lucidity(decision="test_consequence", margin=0.5),
        projector=projector,
        decoder=orchestrator_mod.Decoder(),
        episodes=[episode],
        phase=1,
        debug=False,
    )
    orch.run_one_step()
    assert projector.calls >= 1
