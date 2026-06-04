"""Terminal view of audit runs (`lucid-inspect`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lucid.audit.logger import AuditLogger, RunAuditManifest, StageAuditRef


def format_stage_ref(ref: StageAuditRef) -> str:
    status = "ok" if ref.success else "FAIL"
    timing = f"{ref.duration_ms:.1f}ms" if ref.duration_ms else "-"
    return f"{ref.stage_name:<14} {status:<5} {timing:>10}"


def format_manifest(manifest: RunAuditManifest) -> str:
    if manifest.summary:
        return "\n".join([manifest.summary.get("headline", ""), ""] + manifest.summary.get("lines", []))
    header = [
        f"run_id: {manifest.run_id}",
        f"lucidity: {manifest.lucidity_decision or '-'}",
        "",
        "stages:",
    ]
    return "\n".join(header + [format_stage_ref(ref) for ref in manifest.stages])


def print_run(run_dir: Path | str, *, stage: str | None = None) -> None:
    run_path = Path(run_dir)
    logger = AuditLogger(base_dir=".")
    manifest = logger.load_manifest(run_path)

    print(format_manifest(manifest))
    print()

    if stage:
        record = logger.load_stage_record(run_path, stage)
        summary = record.get("summary") or {}
        print(f"stage: {record.get('stage_name')}")
        print(f"headline: {summary.get('headline', '')}")
        print()
        for line in summary.get("lines", []):
            print(f"  {line}")
        if record.get("error_message"):
            print(f"\nerror: {record['error_message']}")
        return

    for ref in manifest.stages:
        stage_key = ref.stage_name if ref.occurrence <= 1 else f"{ref.stage_name}#{ref.occurrence}"
        file_name = (
            f"{ref.stage_name}.json"
            if ref.occurrence <= 1
            else f"{ref.stage_name}_{ref.occurrence:02d}.json"
        )
        record = json.loads((run_path / file_name).read_text(encoding="utf-8"))
        summary = record.get("summary") or {}
        print(f"--- {stage_key}: {summary.get('headline', '')} ---")
        for line in summary.get("lines", []):
            print(f"  {line}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lucid-inspect", description="View audit run folders")
    parser.add_argument("run_dir", type=Path, help="Folder with manifest.json and README.txt")
    parser.add_argument("--stage", "-s", help="One stage only (e.g. lucidity)")
    args = parser.parse_args(argv)

    if not (args.run_dir / "manifest.json").exists():
        print(f"Not an audit run: {args.run_dir}", file=sys.stderr)
        return 1

    print_run(args.run_dir, stage=args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
