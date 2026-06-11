"""Audit logger: readable JSON, manifest, hashes."""

from __future__ import annotations

from pathlib import Path

import pytest

from lucid.audit import AuditLogger, content_hash, print_run, summarize_stage_output
from lucid.ir.binding import BindingOutput, CandidateFrame
from lucid.ir.common import DecoderMode, LucidityDecision, Modality, TaskIntent
from lucid.ir.cue import CueCloud, TraceActivationRequest
from lucid.ir.dmf import ActiveTrace, DmfOutput
from lucid.ir.expression import DecoderOutput
from lucid.ir.lucidity import DecoderPolicy, LucidityOutput
from lucid.ir.perception import CandidateUnit, PerceptionInput, PerceptualEvidenceGraph
from lucid.ir.pipeline import PipelineRun, RunContext, StageName, StageResult
from lucid.ir.training import CostMetrics


@pytest.fixture
def audit_base(tmp_path: Path) -> Path:
    return tmp_path / "audit"


def _sample_run(run_id: str = "run-test-1", session_id: str = "") -> PipelineRun:
    graph = PerceptualEvidenceGraph(
        candidate_units=[CandidateUnit(unit_id="u1", surface="bank", confidence=0.9)],
    )
    cloud = CueCloud(
        primitive_trace_activations=[
            TraceActivationRequest(trace_id="t-financial", weight=0.7),
        ],
    )
    dmf = DmfOutput(active_traces=[ActiveTrace(trace_id="t-financial", activation=0.7)])
    binding = BindingOutput(
        candidate_frames=[CandidateFrame(frame_id="f1", frame_type="word_sense", confidence=0.6)],
    )
    lucidity = LucidityOutput(
        decision=LucidityDecision.PRESERVE_AMBIGUITY,
        decoder_policy=DecoderPolicy(mode=DecoderMode.EXPRESS_UNCERTAINTY.value),
    )
    decoder = DecoderOutput(surface_text="Could mean river bank or financial bank.")

    context = RunContext(
        run_id=run_id,
        session_id=session_id,
        turn_index=2 if session_id else 0,
        task_intent=TaskIntent.ANSWER,
    )

    return PipelineRun(
        context=context,
        perception_input=PerceptionInput(raw_payload="go to the bank", modality=Modality.TEXT),
        evidence_graph=graph,
        cue_cloud=cloud,
        dmf_output=dmf,
        binding_output=binding,
        lucidity_output=lucidity,
        decoder_output=decoder,
        stage_results=[
            StageResult(stage_name=StageName.PERCEPTION, success=True, duration_ms=4.2),
            StageResult(stage_name=StageName.LUCIDITY, success=True, duration_ms=3.1),
            StageResult(stage_name=StageName.DECODER, success=True, duration_ms=1.0),
        ],
        cost_metrics=CostMetrics(wall_time_ms=34.0),
    )


def test_content_hash_stable():
    assert content_hash({"z": 1, "a": [2, 3]}) == content_hash({"a": [2, 3], "z": 1})


def test_stage_json_has_human_summary(audit_base: Path):
    logger = AuditLogger(base_dir=audit_base)
    run_dir = logger.write_pipeline_run(_sample_run())

    record = logger.load_stage_record(run_dir, "perception")
    assert record["schema_version"] == 1
    assert record["summary"]["headline"]
    assert any("bank" in line for line in record["summary"]["lines"])
    assert record["input"]["raw_payload"] == "go to the bank"
    assert record["output"]["candidate_units"][0]["surface"] == "bank"


def test_readme_txt(audit_base: Path):
    logger = AuditLogger(base_dir=audit_base)
    run_dir = logger.write_pipeline_run(_sample_run())
    readme = (run_dir / "README.txt").read_text(encoding="utf-8")
    assert "preserve_ambiguity" in readme
    assert "perception" in readme


def test_manifest_has_summary(audit_base: Path):
    logger = AuditLogger(base_dir=audit_base)
    run_dir = logger.write_pipeline_run(_sample_run())
    manifest = logger.load_manifest(run_dir)
    assert manifest.summary["headline"]
    assert manifest.lucidity_decision == "preserve_ambiguity"


def test_envelope_roundtrip_from_record(audit_base: Path):
    logger = AuditLogger(base_dir=audit_base)
    run_dir = logger.write_pipeline_run(_sample_run())
    envelope = logger.load_stage_envelope(run_dir, "lucidity")
    assert envelope.stage_name == "lucidity"


def test_same_output_same_hash(audit_base: Path):
    logger = AuditLogger(base_dir=audit_base)
    dir_a = logger.write_pipeline_run(_sample_run(run_id="a"))
    dir_b = logger.write_pipeline_run(_sample_run(run_id="b"))
    hash_a = logger.load_stage_record(dir_a, "dmf")["output_hash"]
    hash_b = logger.load_stage_record(dir_b, "dmf")["output_hash"]
    assert hash_a == hash_b


def test_pretty_print(capsys, audit_base: Path):
    logger = AuditLogger(base_dir=audit_base)
    run_dir = logger.write_pipeline_run(_sample_run())
    print_run(run_dir, stage="decoder")
    out = capsys.readouterr().out
    assert "decoder" in out
    assert "bank" in out.lower() or "mean" in out.lower()


def test_no_run_log_json(audit_base: Path):
    logger = AuditLogger(base_dir=audit_base)
    run_dir = logger.write_pipeline_run(_sample_run())
    assert not (run_dir / "run_log.json").exists()


def test_summarize_stage_output():
    s = summarize_stage_output("perception", {"candidate_units": [{"surface": "bank"}]})
    assert "bank" in s["headline"] or any("bank" in line for line in s["lines"])


def test_summarize_decoder_grid_output():
    s = summarize_stage_output("decoder", {"surface_grid": [[0, 1], [0, 0]]})
    assert s["headline"] == "grid 2x2"
    assert "surface_grid: 2x2" in s["lines"]


def test_pipeline_writes_narrative_txt(audit_base: Path):
    logger = AuditLogger(base_dir=audit_base)
    run_dir = logger.write_pipeline_run(_sample_run())
    narrative = (run_dir / "narrative.txt").read_text(encoding="utf-8")
    assert "Pipeline run narrative" in narrative
    assert "DMF" in narrative or "dmf" in narrative.lower()
    assert "Lucidity" in narrative or "lucidity" in narrative.lower()


def test_stage_summary_includes_narrative(audit_base: Path):
    logger = AuditLogger(base_dir=audit_base)
    run_dir = logger.write_pipeline_run(_sample_run())
    dmf = logger.load_stage_record(run_dir, "dmf")
    assert dmf["summary"].get("narrative")
    assert any("active" in line.lower() or "trace" in line.lower() for line in dmf["summary"]["narrative"])
