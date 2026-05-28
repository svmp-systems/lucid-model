"""`lucid-run` — run one episode through the pipeline step-by-step."""

from __future__ import annotations

import argparse
import json
import sys
from json import JSONDecodeError
from pathlib import Path

from lucid.ir.serde import from_json
from lucid.ir.training import Episode
from lucid.cognition.input.perception import PerceptionConfig

from .runner import OrchestratorConfig, OrchestratorRunner


def _episode_from_file(path: Path) -> Episode:
    text = path.read_text(encoding="utf-8-sig")
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"empty episode file: {path}")

    try:
        return from_json(stripped, Episode)
    except JSONDecodeError as full_error:
        for line in text.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            try:
                json.loads(candidate)
            except JSONDecodeError:
                break
            return from_json(candidate, Episode)
        raise ValueError(f"invalid Episode JSON in {path}: {full_error}") from full_error


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="lucid-run",
        description="Run one Episode JSON through the pipeline",
    )
    p.add_argument(
        "episode",
        help="Path to an Episode JSON (single object) or JSONL (first line used)",
    )
    p.add_argument("--audit-dir", default="audit", help="Audit base directory (default: audit)")
    p.add_argument(
        "--perception",
        default="",
        choices=["", "rule", "llm"],
        help="Perception backend (default: llm; use rule for offline)",
    )
    args = p.parse_args(argv)

    path = Path(args.episode)
    if not path.exists():
        print(f"missing file: {path}", file=sys.stderr)
        return 2

    try:
        episode = _episode_from_file(path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    perception_cfg = PerceptionConfig.from_env()
    if args.perception:
        perception_cfg.backend = args.perception
    runner = OrchestratorRunner(
        config=OrchestratorConfig(audit_base_dir=args.audit_dir, perception=perception_cfg)
    )
    run = runner.run_episode(episode)
    print(run.context.audit_dir or "(audit written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

