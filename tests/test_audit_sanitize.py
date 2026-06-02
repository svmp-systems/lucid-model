from __future__ import annotations

import json

from lucid.audit.scaling import ScalingPoint, _append_point
from lucid.audit.sanitize import redact_string, sanitize_audit_value


def test_redact_openai_key() -> None:
    raw = "Bearer sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"
    assert "[redacted]" in redact_string(raw)
    assert "sk-proj-" not in redact_string(raw)


def test_sanitize_nested_secret_keys() -> None:
    payload = {
        "provenance": {
            "adapter_version": {
                "api_key": "sk-live-should-not-appear",
                "model": "gpt-4o-mini",
                "audit_dir": r"C:\Users\alice\AppData\Local\Temp\run",
            }
        }
    }
    clean = sanitize_audit_value(payload)
    assert clean["provenance"]["adapter_version"]["api_key"] == "[redacted]"
    assert "alice" not in json.dumps(clean)
    assert clean["provenance"]["adapter_version"]["model"] == "gpt-4o-mini"


def test_scaling_append_redacts_before_write(tmp_path) -> None:
    path = tmp_path / "points.jsonl"
    point = ScalingPoint(
        point_id="p1",
        timestamp_utc="2026-06-02T00:00:00+00:00",
        scale_id="test:calibrate:bank:abc",
        event_type="trainer_step",
        training_mode="calibrate",
        module_under_test="cue_encoder",
        run_kind="train",
        provenance={"api_key": "sk-proj-secretvalue1234567890"},
    )
    _append_point(path, point)
    line = path.read_text(encoding="utf-8").strip()
    assert "sk-proj-" not in line
    assert "[redacted]" in line
