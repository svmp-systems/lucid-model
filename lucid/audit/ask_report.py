"""Compact human-readable report for ``lucid ask`` (sentence → answer → audit)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from lucid.audit.logger import AuditLogger
from lucid.audit.stage_summary import format_stage_summary_block
from lucid.ir.pipeline import PipelineRun
from lucid.runtime.paths import DEFAULT_ASK_LATEST, resolve_train_path

ASK_RUN_REPORT = "report.txt"


def _slug(sentence: str, *, max_len: int = 32) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "-", sentence.strip().lower()).strip("-")
    return (token[:max_len] or "ask").strip("-")


def answer_from_pipeline_run(run: PipelineRun) -> str:
    """User-facing answer text from a completed pipeline run."""
    if run.decoder_output is None:
        return "(pipeline did not reach decoder)"

    dec = run.decoder_output
    if dec.refused:
        reason = (dec.refusal_reason or "refused").strip()
        return f"(refused: {reason})"

    text = (dec.surface_text or "").strip()
    if text:
        return text

    grid = dec.surface_grid
    if isinstance(grid, list) and grid:
        rows = len(grid)
        cols = len(grid[0]) if grid[0] else 0
        return f"(grid output {rows}×{cols})"

    return "(empty answer)"


def format_compact_audit(run_dir: Path | str) -> str:
    """Step-by-step audit from an on-disk pipeline run folder."""
    path = Path(run_dir)
    logger = AuditLogger(base_dir=".")
    manifest = logger.load_manifest(path)

    lines: list[str] = [
        f"run: {path}",
        f"lucidity: {manifest.lucidity_decision or '-'}",
        f"wall: {manifest.wall_time_ms:.0f}ms",
        "",
    ]

    for index, ref in enumerate(manifest.stages, start=1):
        status = "ok" if ref.success else "FAIL"
        record_path = path / ref.file_name
        summary: dict = {}
        if record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            summary = record.get("summary") or {}

        headline = str(summary.get("headline") or "").strip()
        timing = f"{ref.duration_ms:6.1f}ms" if ref.duration_ms else "      -"
        lines.append(f"{index}. {ref.stage_name:<14} {status:<5} {timing} | {headline or '-'}")

        lines.extend(format_stage_summary_block(summary))
        if ref.error_message:
            lines.append(f"     error: {ref.error_message}")
        lines.append("")

    lines.append("files: manifest.json, README.txt, narrative.txt, report.txt, one JSON per stage")
    lines.append("drill-down: lucid-inspect <run>")
    return "\n".join(lines).rstrip()


def format_ask_document(
    *,
    sentence: str,
    answer: str,
    run_dir: Path | str,
    lucidity_decision: str = "",
    report_path: Path | str | None = None,
    latest_path: Path | str | None = None,
) -> str:
    """Full document: sentence, answer, compact audit, output file paths."""
    audit = format_compact_audit(run_dir)
    if lucidity_decision and f"lucidity: {lucidity_decision}" not in audit:
        audit = audit.replace("lucidity: -", f"lucidity: {lucidity_decision}", 1)

    parts = [
        "sentence",
        sentence.strip(),
        "",
        "answer",
        answer.strip() or "(no surface text)",
        "",
        "audit",
        audit,
    ]
    if report_path is not None:
        parts.extend(["", "output_file", str(Path(report_path).resolve())])
    if latest_path is not None:
        parts.extend(["", "latest", str(Path(latest_path).resolve())])
    return "\n".join(parts)


@dataclass(slots=True)
class AskReportPaths:
    run_report: Path
    latest: Path | None = None


def write_ask_report(
    run_dir: Path | str,
    *,
    sentence: str,
    answer: str,
    lucidity_decision: str = "",
    latest_path: Path | str | None = None,
    write_latest: bool = True,
    extra_copy: Path | str | None = None,
) -> AskReportPaths:
    """Write the same document to ``report.txt``, optional latest, and ``--out`` copy."""
    path = Path(run_dir)
    run_report = path / ASK_RUN_REPORT

    latest: Path | None = None
    if write_latest:
        latest = resolve_train_path(latest_path or DEFAULT_ASK_LATEST)
        latest.parent.mkdir(parents=True, exist_ok=True)

    body = format_ask_document(
        sentence=sentence,
        answer=answer,
        run_dir=path,
        lucidity_decision=lucidity_decision,
        report_path=run_report,
        latest_path=latest,
    )
    text = body + "\n"
    run_report.write_text(text, encoding="utf-8")

    if latest is not None:
        latest.write_text(text, encoding="utf-8")

    if extra_copy is not None:
        dest = Path(extra_copy).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")

    return AskReportPaths(run_report=run_report, latest=latest)


def write_ask_artifacts(
    run_dir: Path | str,
    *,
    sentence: str,
    answer: str,
    lucidity_decision: str = "",
) -> Path:
    """Backward-compatible alias → :func:`write_ask_report` run report path."""
    return write_ask_report(
        run_dir,
        sentence=sentence,
        answer=answer,
        lucidity_decision=lucidity_decision,
    ).run_report


def episode_id_for_sentence(sentence: str) -> str:
    return f"ask-{_slug(sentence)}"
