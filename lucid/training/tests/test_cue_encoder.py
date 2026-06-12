from __future__ import annotations

import json
from pathlib import Path

from lucid.audit.smoke import write_cue_encoder_audit
from lucid.cli import main as lucid_main
from lucid.cognition.input.cue import CueEncoderConfig, encode_cues
from lucid.cognition.input.perception import PerceptionConfig, perceive
from lucid.ir.common import ComputePolicy, Modality
from lucid.ir.cue import CueEncoderInput
from lucid.ir.dmf import DmfInput
from lucid.ir.perception import PerceptionInput
from lucid.cognition.memory.dmf import DmfTraceRecord, DynamicMemoryField


def _bank_cue_input() -> CueEncoderInput:
    graph = perceive(
        PerceptionInput(
            raw_payload="I found money while kayaking and placed it in the bank.",
            modality=Modality.TEXT,
        ),
        config=PerceptionConfig(backend="rule", write_audit=False),
    )
    return CueEncoderInput(
        perceptual_evidence_graph=graph,
        task_intent_hint="answer",
        retrieval_budget=16,
    )


def _ids(cloud) -> list[str]:
    return [request.trace_id for request in cloud.primitive_trace_activations]


def test_cue_encoder_compiles_sparse_surface_relation_and_ambiguity_cues():
    cloud = encode_cues(_bank_cue_input())
    primitive_ids = _ids(cloud)
    relation_ids = [request.trace_id for request in cloud.relational_trace_activations]

    assert {"bank", "found", "money", "kayaking", "placed"}.issubset(primitive_ids)
    assert "i" not in primitive_ids
    assert "it" not in primitive_ids
    assert "the" not in primitive_ids
    assert "object_carryover" in relation_ids
    assert "temporal_subordinate" in relation_ids
    assert "structure:deictic_speaker" in cloud.weak_structure_hints
    assert "structure:pronoun_coreference" in cloud.weak_structure_hints
    assert "marker:locative_marker" in cloud.weak_structure_hints
    assert "uncertainty:polysemy" in cloud.weak_structure_hints
    assert cloud.ambiguity_policy.value == "preserve_plural"

    bank_request = next(request for request in cloud.primitive_trace_activations if request.trace_id == "bank")
    assert bank_request.keep_alive is True
    assert bank_request.evidence_refs == ["u_bank"]
    assert all(request.evidence_refs for request in cloud.primitive_trace_activations)


def test_cue_encoder_output_feeds_existing_dmf_cue_affinity_index():
    cloud = encode_cues(_bank_cue_input())
    dmf = DynamicMemoryField(
        [
            DmfTraceRecord(trace_id="t-money", cue_affinities={"money": 0.9}),
            DmfTraceRecord(trace_id="t-bank", cue_affinities={"bank": 0.8}),
            DmfTraceRecord(trace_id="t-water", cue_affinities={"kayaking": 0.8}),
        ],
        audit_base_dir=None,
    )

    out = dmf.run(DmfInput(cue_cloud=cloud, compute_policy=ComputePolicy(max_active_traces=3)))
    active = {trace.trace_id for trace in out.active_traces}

    assert {"t-money", "t-bank", "t-water"} == active
    assert out.coverage_score > 0.0
    assert out.audit_log["retrieval_mode"] == "activated_threshold_filter"


def test_cue_encoder_evidence_compile_does_not_apply_learned_map_routes(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "cue_encoder_map.json").write_text(
        json.dumps(
            {
                "feature_index": {
                    "surface:bank": [
                        {
                            "cue_key": "river_location_like",
                            "weight": 0.7,
                            "preserve_as_alternative": True,
                        }
                    ]
                },
                "relation_index": {},
                "cue_targets": [],
            }
        ),
        encoding="utf-8",
    )

    cloud = encode_cues(_bank_cue_input(), config=CueEncoderConfig(checkpoint=checkpoint))
    primitive_ids = _ids(cloud)

    assert "bank" in primitive_ids
    assert "river_location_like" not in primitive_ids
    assert cloud.provenance.extra["cue_encoder"]["learned_map_available"] is True


def test_cue_encoder_compiles_grid_change_features():
    raw = {
        "input": [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
        "output": [[0, 0, 0], [0, 0, 2], [0, 0, 0]],
    }
    graph = perceive(
        PerceptionInput(raw_payload=raw, modality=Modality.GRID),
        config=PerceptionConfig(backend="rule", write_audit=False),
    )
    cloud = encode_cues(CueEncoderInput(perceptual_evidence_graph=graph))
    primitive_ids = set(_ids(cloud))

    assert "position_shift_like" in primitive_ids
    assert "color_preserved_like" in primitive_ids
    assert "shape_preserved_like" in primitive_ids


def test_cue_encoder_audit_writer_records_human_and_machine_files(tmp_path: Path):
    cue_input = _bank_cue_input()
    cloud = encode_cues(cue_input)

    run_dir = write_cue_encoder_audit(
        audit_base_dir=tmp_path / "audit",
        cue_input=cue_input,
        cue_cloud=cloud,
        details={"fixture": "bank"},
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage_name"] == "cue_encoder"
    assert manifest["primitive_activation_count"] >= 5
    assert manifest["output_hash"]
    assert (run_dir / "README.txt").exists()
    assert (run_dir / "input.json").exists()
    assert (run_dir / "output.json").exists()


def test_cue_encoder_widen_flag_when_coverage_low():
    cue_input = _bank_cue_input()
    cue_input.upstream_state["dmf_coverage_score"] = 0.2
    cloud = encode_cues(cue_input)

    assert cloud.provenance.extra["cue_encoder"]["widen_applied"] is True


def test_cue_encoder_calibrate_training_patches_missing_routes(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"

    exit_code = lucid_main(
        [
            "train",
            "cue_encoder",
            "--mode",
            "calibrate",
            "--fixture",
            "bank",
            "--checkpoint",
            str(checkpoint),
            "--audit-dir",
            str(tmp_path / "audit"),
        ]
    )

    assert exit_code == 0
    cue_map = json.loads((checkpoint / "cue_encoder_map.json").read_text(encoding="utf-8"))
    assert cue_map["cue_targets"]
    assert cue_map["feature_index"]
    first_entry = next(iter(cue_map["feature_index"].values()))[0]
    assert first_entry.get("feature_pattern")


def test_cue_encoder_training_builds_runtime_feature_index(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"

    exit_code = lucid_main(
        [
            "train",
            "cue_encoder",
            "--mode",
            "seed",
            "--fixture",
            "bank",
            "--checkpoint",
            str(checkpoint),
            "--audit-dir",
            str(tmp_path / "audit"),
        ]
    )

    assert exit_code == 0
    cue_map = json.loads((checkpoint / "cue_encoder_map.json").read_text(encoding="utf-8"))
    assert cue_map["cue_targets"]
    assert cue_map["feature_index"]


def test_lucid_cue_encoder_smoke_command(tmp_path: Path, capsys):
    exit_code = lucid_main(
        [
            "cue-encoder",
            "--fixture",
            "bank",
            "--audit-dir",
            str(tmp_path / "audit"),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert any(req["trace_id"] == "bank" for req in payload["primitive_trace_activations"])
    assert list((tmp_path / "audit").glob("*/manifest.json"))
