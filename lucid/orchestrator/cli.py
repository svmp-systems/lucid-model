"""`lucid-run` — run one episode through the pipeline step-by-step."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lucid.ir.serde import from_json
from lucid.ir.training import Episode
from lucid.orchestrator.runner import OrchestratorRunner


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lucid-run", description="Run one Episode JSON through the pipeline")
    p.add_argument("episode", help="Path to an Episode JSON (single object) or JSONL (first line used)")
    p.add_argument("--audit-dir", default="audit", help="Audit base directory (default: audit)")
    args = p.parse_args(argv)

    path = Path(args.episode)
    if not path.exists():
        print(f"missing file: {path}", file=sys.stderr)
        return 2

    text = path.read_text(encoding="utf-8")
    line = text.splitlines()[0] if "\n" in text and text.lstrip().startswith("{") is False else None
    payload = line if line and line.strip().startswith("{") else text.strip().splitlines()[0]
    episode = from_json(payload, Episode)

    runner = OrchestratorRunner(config=__import__("lucid.orchestrator.runner", fromlist=["OrchestratorConfig"]).OrchestratorConfig(audit_base_dir=args.audit_dir))
    run = runner.run_episode(episode)
    print(run.context.audit_dir or "(audit written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

