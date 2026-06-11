"""``lucid checkpoint`` — load/save inference save points separate from training."""

from __future__ import annotations

import argparse
import json
import sys

from lucid.training.checkpoint.slots import (
    clear_loaded_checkpoint,
    format_checkpoint_status,
    list_named_saves,
    promote_to_loaded,
    read_loaded_pointer,
    save_training_snapshot,
)


def _cmd_status(_args: argparse.Namespace) -> int:
    print(format_checkpoint_status())
    return 0


def _cmd_load(args: argparse.Namespace) -> int:
    try:
        dest = promote_to_loaded(args.source, label=args.label)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    pointer = read_loaded_pointer() or {}
    print(
        json.dumps(
            {
                "loaded": str(dest),
                "source": pointer.get("source_path"),
                "label": pointer.get("source_label"),
                "training_steps": pointer.get("training_steps"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_save(args: argparse.Namespace) -> int:
    try:
        path = save_training_snapshot(args.name)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(path)
    return 0


def _cmd_clear(_args: argparse.Namespace) -> int:
    clear_loaded_checkpoint()
    print("loaded checkpoint cleared (inference will run cold until lucid checkpoint load)")
    return 0


def _cmd_list(_args: argparse.Namespace) -> int:
    saves = list_named_saves()
    if not saves:
        print("(no named saves under checkpoints/saves/)")
        return 0
    for name, path in saves:
        print(f"{name}\t{path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lucid checkpoint")
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("status", help="Show training vs loaded checkpoint slots").set_defaults(
        func=_cmd_status
    )

    load_p = sub.add_parser(
        "load",
        help="Copy a checkpoint into the loaded slot (default: training workspace)",
    )
    load_p.add_argument(
        "source",
        nargs="?",
        default="training",
        help="training | saves/<name> | <name> | checkpoints/…",
    )
    load_p.add_argument("--label", default="", help="Optional note stored in loaded.json")
    load_p.set_defaults(func=_cmd_load)

    save_p = sub.add_parser("save", help="Archive training checkpoint to checkpoints/saves/<name>")
    save_p.add_argument("name", help="Save name (e.g. bank-v1)")
    save_p.set_defaults(func=_cmd_save)

    sub.add_parser("clear", help="Remove loaded save point (inference runs cold)").set_defaults(
        func=_cmd_clear
    )
    sub.add_parser("list", help="List named saves").set_defaults(func=_cmd_list)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
