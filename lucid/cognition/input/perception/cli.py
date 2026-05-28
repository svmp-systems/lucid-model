"""`lucid-perceive` — raw input → evidence graph."""

from __future__ import annotations

import argparse
import sys

from lucid.ir.common import Modality
from lucid.ir.perception import PerceptionInput
from lucid.ir.serde import to_json
from lucid.perception import PerceptionConfig, perceive


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lucid-perceive")
    p.add_argument("text", nargs="?", help="Raw text (or stdin)")
    p.add_argument("--modality", default="text", choices=[m.value for m in Modality])
    p.add_argument("--backend", default="", choices=["", "rule", "llm"])
    args = p.parse_args(argv)

    raw = args.text if args.text is not None else sys.stdin.read().strip()
    if not raw:
        print("no input", file=sys.stderr)
        return 2

    cfg = PerceptionConfig.from_env()
    if args.backend:
        cfg.backend = args.backend

    graph = perceive(PerceptionInput(raw_payload=raw, modality=Modality(args.modality)), config=cfg)
    print(to_json(graph))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
