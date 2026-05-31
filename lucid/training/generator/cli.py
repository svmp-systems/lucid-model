"""lucid-gen — generate training episodes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from lucid.ir.serde import to_json
from lucid.training.generator.engine import (
    DEFAULT_BANDS,
    get_recipe,
    list_recipes,
    pick_band,
    parse_band_weights,
    rng_for_seed,
    sample_knob,
    check_batch,
)
from lucid.training.generator.output import build_phase1_pack, read_episodes, write_episodes


def cmd_make(args: argparse.Namespace) -> int:
    recipe = get_recipe(args.recipe)
    rng = rng_for_seed(args.seed)
    bands = parse_band_weights(args.bands) if args.bands else DEFAULT_BANDS

    episodes = []
    for i in range(args.count):
        knob = sample_knob(rng, pick_band(rng, bands))
        episode = recipe.make(rng, knob)
        episode.seed = args.seed + i
        episodes.append(episode)

    errors = check_batch(episodes)
    if errors:
        print("check failed:", file=sys.stderr)
        for err in errors[:10]:
            print(f"  {err}", file=sys.stderr)
        return 1

    if args.preview:
        print(to_json(episodes[0]))
        return 0

    if args.out:
        write_episodes(episodes, args.out)
        print(f"wrote {len(episodes)} episodes -> {args.out}")
    else:
        for episode in episodes:
            print(to_json(episode, indent=None))
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    if args.name != "phase1":
        print(f"unknown pack {args.name!r}", file=sys.stderr)
        return 1
    out = args.out or "data/generated/phase1"
    episodes = build_phase1_pack(out, seed=args.seed)
    print(f"phase1: {len(episodes)} episodes -> {out}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    path = Path(args.path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    failed = 0
    for file in files:
        episodes = read_episodes(file)
        errors = check_batch(episodes)
        if errors:
            print(f"{file.name}: {len(errors)} issue(s)")
            for err in errors[:5]:
                print(f"  {err}")
            failed += len(errors)
        else:
            print(f"{file.name}: ok ({len(episodes)} episodes)")
    return 1 if failed else 0


def cmd_summary(args: argparse.Namespace) -> int:
    episodes = read_episodes(args.path)
    print(
        json.dumps(
            {
                "count": len(episodes),
                "recipes": dict(Counter(ep.template_id for ep in episodes)),
                "lucidity": dict(Counter(ep.gold.lucidity_target for ep in episodes)),
                "modalities": dict(Counter(str(ep.modality) for ep in episodes)),
            },
            indent=2,
        )
    )
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    for name in list_recipes():
        recipe = get_recipe(name)
        print(f"{name}\t{recipe.MODALITY}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lucid-gen")
    sub = parser.add_subparsers(dest="command", required=True)

    make_p = sub.add_parser("make", help="generate episodes from one recipe")
    make_p.add_argument("recipe")
    make_p.add_argument("--count", type=int, default=1)
    make_p.add_argument("--bands", default="30:40:30", help="low:mid:high weight ratio")
    make_p.add_argument("--seed", type=int, default=42)
    make_p.add_argument("--out", default="")
    make_p.add_argument("--preview", action="store_true")
    make_p.set_defaults(func=cmd_make)

    pack_p = sub.add_parser("pack", help="build a named episode bundle")
    pack_p.add_argument("name", choices=["phase1"])
    pack_p.add_argument("--out", default="")
    pack_p.add_argument("--seed", type=int, default=42)
    pack_p.set_defaults(func=cmd_pack)

    sub.add_parser("list", help="list recipes").set_defaults(func=cmd_list)

    check_p = sub.add_parser("check", help="validate episode files")
    check_p.add_argument("path")
    check_p.set_defaults(func=cmd_check)

    sum_p = sub.add_parser("summary", help="summarize an episode file")
    sum_p.add_argument("path")
    sum_p.set_defaults(func=cmd_summary)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
