"""`lucid-perceive` — raw input → evidence graph."""

from __future__ import annotations

import argparse
import sys

from lucid.ir.common import Modality
from lucid.ir.perception import PerceptionInput
from lucid.cognition.input.perception import PerceptionConfig, perceive
from lucid.cognition.input.perception.compact import to_compact_json
from lucid.ir.serde import to_json


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lucid-perceive")
    p.add_argument("text", nargs="?", help="Raw text (or stdin)")
    p.add_argument("--modality", default="text", choices=[m.value for m in Modality])
    p.add_argument("--backend", default="", choices=["", "rule", "llm"], help="default: llm")
    p.add_argument(
        "--compact",
        action="store_true",
        help="print only non-empty lists and non-default fields",
    )
    args = p.parse_args(argv)

    raw = args.text if args.text is not None else sys.stdin.read().strip()
    if not raw:
        print("no input", file=sys.stderr)
        return 2

    cfg = PerceptionConfig.from_env()
    if args.backend:
        cfg.backend = args.backend

    graph = perceive(PerceptionInput(raw_payload=raw, modality=Modality(args.modality)), config=cfg)
    print(to_compact_json(graph) if args.compact else to_json(graph))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
