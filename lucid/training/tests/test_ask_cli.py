"""``lucid ask`` — one sentence in, answer + compact audit out."""

from __future__ import annotations

from pathlib import Path

from lucid.audit.ask_report import ASK_RUN_REPORT, format_ask_document, format_compact_audit
from lucid.cli import _normalize_argv, main as lucid_main
from lucid.runtime.paths import DEFAULT_ASK_LATEST, resolve_train_path


def test_normalize_argv_maps_bare_sentence_to_ask() -> None:
    assert _normalize_argv(["hello", "world"]) == ["ask", "hello", "world"]
    assert _normalize_argv(["ask", "hi"]) == ["ask", "hi"]
    assert _normalize_argv(["run", "ep.json"]) == ["run", "ep.json"]


def test_ask_prints_sentence_answer_and_audit(tmp_path: Path, capsys) -> None:
    audit_dir = tmp_path / "audit"
    latest = tmp_path / "ask-latest.txt"
    exit_code = lucid_main(
        [
            "ask",
            "I found money while kayaking and placed it in the bank.",
            "--audit-dir",
            str(audit_dir),
            "--latest",
            str(latest),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "sentence" in captured.out
    assert "answer" in captured.out
    assert "audit" in captured.out
    assert "output_file" in captured.out
    assert "latest" in captured.out
    assert "kayaking" in captured.out
    assert "perception" in captured.out
    assert "decoder" in captured.out

    run_dirs = list(audit_dir.iterdir())
    assert len(run_dirs) == 1
    report = run_dirs[0] / ASK_RUN_REPORT
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert text.startswith("sentence\n")
    assert "audit\n" in text
    assert "output_file\n" in text
    assert latest.is_file()
    assert latest.read_text(encoding="utf-8") == text


def test_ask_writes_extra_out_copy(tmp_path: Path, capsys) -> None:
    audit_dir = tmp_path / "audit"
    out_copy = tmp_path / "my-ask.txt"
    exit_code = lucid_main(
        [
            "ask",
            "short test",
            "--audit-dir",
            str(audit_dir),
            "--out",
            str(out_copy),
            "--no-latest",
        ]
    )
    assert exit_code == 0
    assert out_copy.is_file()
    assert "sentence\nshort test\n" in out_copy.read_text(encoding="utf-8")


def test_bare_sentence_invokes_ask(tmp_path: Path, capsys) -> None:
    audit_dir = tmp_path / "audit"
    exit_code = lucid_main(
        [
            "go to the bank",
            "--audit-dir",
            str(audit_dir),
            "--no-latest",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "sentence" in captured.out
    assert "go to the bank" in captured.out


def test_format_compact_audit_lists_stages(tmp_path: Path) -> None:
    audit_dir = tmp_path / "audit"
    lucid_main(["ask", "short test", "--audit-dir", str(audit_dir), "--no-latest"])
    run_dir = next(audit_dir.iterdir())
    body = format_compact_audit(run_dir)
    assert "1. perception" in body
    assert "run:" in body

    doc = format_ask_document(
        sentence="short test",
        answer="ok",
        run_dir=run_dir,
        lucidity_decision="COMMIT",
        report_path=run_dir / ASK_RUN_REPORT,
        latest_path=resolve_train_path(DEFAULT_ASK_LATEST),
    )
    assert doc.splitlines()[0] == "sentence"
    assert "output_file" in doc
