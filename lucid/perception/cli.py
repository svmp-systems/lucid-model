"""`lucid-perceive` — run perception only on raw input."""

from __future__ import annotations

import argparse
import json
import sys

from lucid.ir.common import Modality
from lucid.ir.perception import PerceptionInput
from lucid.ir.serde import to_json
from lucid.perception.config import PerceptionConfig
from lucid.perception.engine import perceive


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lucid-perceive", description="Perception-only: raw input → evidence graph")
    p.add_argument("text", nargs="?", help="Raw text (default: read stdin)")
    p.add_argument("--modality", default="text", choices=[m.value for m in Modality])
    p.add_argument("--backend", default="", help="rule | llm (default: env or rule)")
    p.add_argument("--model", default="", help="LLM model name")
    p.add_argument("--base-url", default="", help="OpenAI-compatible API base URL")
    args = p.parse_args(argv)

    raw = args.text
    if raw is None:
        raw = sys.stdin.read().strip()
    if not raw:
        print("no input", file=sys.stderr)
        return 2

    cfg = PerceptionConfig.from_env()
    if args.backend:
        cfg.backend = args.backend
    if args.model:
        cfg.model = args.model
    if args.base_url:
        cfg.base_url = args.base_url.rstrip("/")

    inp = PerceptionInput(raw_payload=raw, modality=Modality(args.modality))
    graph = perceive(inp, config=cfg)
    print(to_json(graph))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
