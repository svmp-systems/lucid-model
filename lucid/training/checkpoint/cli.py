"""``lucid checkpoint`` — load/save inference save points separate from training."""

from __future__ import annotations

import argparse
import json
import sys

from lucid.training.checkpoint.metadata import (
    archive_stale_quarantine,
    summarize_metadata_lifecycle,
)
from lucid.training.checkpoint.registry import list_registry
from lucid.training.checkpoint.shards import (
    DEFAULT_MAX_ITEMS_PER_SHARD,
    SHARDABLE_STORES,
    compact_checkpoint,
)
from lucid.training.checkpoint.slots import (
    archive_training_checkpoint,
    clear_loaded_checkpoint,
    format_checkpoint_status,
    list_named_saves,
    promote_to_loaded,
    read_loaded_pointer,
    resolve_checkpoint_ref,
    save_training_snapshot,
)
from lucid.training.checkpoint.store import load_checkpoint, save_checkpoint


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
                "save": pointer.get("save_name"),
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
        if args.from_training:
            record = archive_training_checkpoint(
                source="training",
                name=args.name or None,
                label=args.label,
                command="lucid checkpoint save",
            )
            print(json.dumps(record, indent=2, sort_keys=True))
        else:
            if not args.name:
                print("save name required (or use --from-training for next cp_NNN)", file=sys.stderr)
                return 2
            path = save_training_snapshot(args.name)
            print(path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _cmd_clear(_args: argparse.Namespace) -> int:
    clear_loaded_checkpoint()
    print("loaded checkpoint cleared (inference will run cold until lucid checkpoint load)")
    return 0


def _cmd_list(_args: argparse.Namespace) -> int:
    registry = list_registry()
    if registry:
        print("standard saves (cp_NNN):")
        for row in registry:
            label = f"\t{row['label']}" if row.get("label") else ""
            cmd = f"\t({row['command']})" if row.get("command") else ""
            print(f"{row['name']}\tsteps={row.get('training_steps', '?')}{label}{cmd}")
    saves = list_named_saves()
    known = {row.get("name") for row in registry}
    other = [(name, path) for name, path in saves if name not in known]
    if other:
        print("\nother saves:")
        for name, path in other:
            print(f"{name}\t{path}")
    if not registry and not other:
        print("(no saves under checkpoints/saves/)")
    return 0


def _cmd_lifecycle(args: argparse.Namespace) -> int:
    checkpoint = resolve_checkpoint_ref(args.checkpoint)
    state = load_checkpoint(checkpoint, create=False)
    before = summarize_metadata_lifecycle(
        state,
        stale_quarantine_days=args.max_age_days,
    )
    archived: list[dict[str, object]] = []
    if args.archive_stale:
        archived = archive_stale_quarantine(
            state,
            max_age_days=args.max_age_days,
        )
        if archived:
            save_checkpoint(state, checkpoint, force=True)
    after = summarize_metadata_lifecycle(
        state,
        stale_quarantine_days=args.max_age_days,
    )
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "resolved_checkpoint": checkpoint,
                "before": before,
                "after": after,
                "archived": archived,
                "archive_stale": args.archive_stale,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_compact(args: argparse.Namespace) -> int:
    checkpoint = resolve_checkpoint_ref(args.checkpoint)
    summary = compact_checkpoint(
        checkpoint,
        max_items_per_shard=args.max_items_per_shard,
        stores=args.store or None,
    )
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "resolved_checkpoint": checkpoint,
                **summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lucid checkpoint")
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("status", help="Show training vs loaded checkpoint slots").set_defaults(
        func=_cmd_status
    )

    load_p = sub.add_parser(
        "load",
        help="Pin a checkpoint for inference (default: training workspace)",
    )
    load_p.add_argument(
        "source",
        nargs="?",
        default="training",
        help="training | cp_001 | saves/<name> | checkpoints/…",
    )
    load_p.add_argument("--label", default="", help="Optional note stored in loaded.json")
    load_p.set_defaults(func=_cmd_load)

    save_p = sub.add_parser(
        "save",
        help="Archive training checkpoint to checkpoints/saves/<name> or next cp_NNN",
    )
    save_p.add_argument(
        "name",
        nargs="?",
        default="",
        help="Save name (default: next cp_NNN when --from-training)",
    )
    save_p.add_argument(
        "--from-training",
        action="store_true",
        help="Archive from training workspace and register in cp_NNN registry",
    )
    save_p.add_argument("--label", default="", help="Human note for registry")
    save_p.set_defaults(func=_cmd_save)

    sub.add_parser("clear", help="Remove loaded save point (inference runs cold)").set_defaults(
        func=_cmd_clear
    )
    sub.add_parser("list", help="List standard cp_NNN saves and other archives").set_defaults(
        func=_cmd_list
    )

    lifecycle_p = sub.add_parser(
        "lifecycle",
        help="Summarize learned metadata heat tiers and optionally archive stale quarantine",
    )
    lifecycle_p.add_argument("--checkpoint", default="training")
    lifecycle_p.add_argument("--max-age-days", type=int, default=30)
    lifecycle_p.add_argument("--archive-stale", action="store_true")
    lifecycle_p.set_defaults(func=_cmd_lifecycle)

    compact_p = sub.add_parser(
        "compact",
        help="Compact keyed stores and write manifest-indexed checkpoint shards",
    )
    compact_p.add_argument("--checkpoint", default="training")
    compact_p.add_argument(
        "--max-items-per-shard",
        type=int,
        default=DEFAULT_MAX_ITEMS_PER_SHARD,
    )
    compact_p.add_argument(
        "--store",
        action="append",
        choices=sorted(SHARDABLE_STORES),
        help="Shard one store; repeat for multiple stores. Defaults to all shardable stores.",
    )
    compact_p.set_defaults(func=_cmd_compact)
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
