"""Global Lucid command surface."""

from __future__ import annotations

import argparse
import sys

from lucid.audit.inspect import main as inspect_main
from lucid.cognition.input.perception import PerceptionConfig, perceive, to_compact_json
from lucid.cognition.orchestrator.cli import main as run_main
from lucid.ir.common import Modality
from lucid.ir.perception import PerceptionInput
from lucid.ir.serde import to_json
from lucid.training.generator.cli import main as gen_main


def _perceive(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lucid perceive")
    parser.add_argument("text", nargs="?", help="Raw text or stdin")
    parser.add_argument("--modality", default="text", choices=[m.value for m in Modality])
    parser.add_argument("--backend", default="", choices=["", "rule", "llm"])
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)

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


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: lucid {perceive|run|gen|inspect} ...")
        return 0

    command, rest = args[0], args[1:]
    if command == "perceive":
        return _perceive(rest)
    if command == "run":
        return run_main(rest)
    if command == "gen":
        return gen_main(rest)
    if command == "inspect":
        return inspect_main(rest)

    print(f"unknown command: {command}", file=sys.stderr)
    print("usage: lucid {perceive|run|gen|inspect} ...", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
