"""Universal command line entrypoint for Lucid runtime tools."""

from __future__ import annotations

import argparse
import json
import sys
from json import JSONDecodeError
from pathlib import Path

from lucid.cognition.reasoning.context_op import run_context_op
from lucid.cognition.reasoning.interference import run_interference
from lucid.cognition.input.perception import PerceptionConfig, perceive, to_compact_json
from lucid.cognition.orchestrator.runner import OrchestratorConfig, OrchestratorRunner
from lucid.ir.common import Modality
from lucid.ir.binding import CandidateFrame
from lucid.ir.context_op import ContextOpInput
from lucid.ir.dmf import ActiveTrace, ConflictSignal, DmfOutput
from lucid.ir.interference import InterferenceInput
from lucid.ir.perception import PerceptionInput
from lucid.ir.perception import CandidateUnit, PerceptualEvidenceGraph, ReferenceHint
from lucid.ir.serde import from_json, to_json
from lucid.ir.training import Episode


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


def _cmd_perceive(args: argparse.Namespace) -> int:
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


def _cmd_run(args: argparse.Namespace) -> int:
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


def _bank_context_fixture(feedback: list[str] | None = None) -> ContextOpInput:
    graph = PerceptualEvidenceGraph(
        candidate_units=[
            CandidateUnit("u_found", "found"),
            CandidateUnit("u_money", "money"),
            CandidateUnit("u_kayaking", "kayaking"),
            CandidateUnit("u_placed", "placed"),
            CandidateUnit("u_bank", "bank"),
        ],
        reference_hints=[
            ReferenceHint(
                source_unit_id="u_placed",
                target_unit_id="u_money",
                reference_type="shared_theme",
                confidence=0.72,
            )
        ],
    )
    frames = [
        CandidateFrame(
            frame_id="event_one",
            frame_type="event",
            role_assignments={
                "ACTION": "t_found",
                "THEME": "t_money",
                "CONTEXT": "t_kayak",
            },
            member_evidence_refs=["u_found", "u_money", "u_kayaking"],
            confidence=0.76,
        ),
        CandidateFrame(
            frame_id="event_two",
            frame_type="event",
            role_assignments={
                "ACTION": "t_placed",
                "THEME": "t_money",
                "DESTINATION": "t_bank",
            },
            member_evidence_refs=["u_placed", "u_money", "u_bank"],
            confidence=0.74,
            unresolved_slot_names=["bank_sense"],
        ),
    ]
    dmf = DmfOutput(
        active_traces=[
            ActiveTrace("t_found", 0.82),
            ActiveTrace("t_money", 0.79),
            ActiveTrace("t_kayak", 0.76),
            ActiveTrace("t_placed", 0.74),
            ActiveTrace("t_bank", 0.58),
        ],
        conflict_signals=[
            ConflictSignal("t_kayak", "t_bank", severity=0.8),
        ],
        top_margin=0.04,
    )
    return ContextOpInput(
        binding_candidate_frames=frames,
        dmf_output=dmf,
        perceptual_evidence_graph=graph,
        lucidity_feedback=feedback or [],
    )


def _cmd_context_op(args: argparse.Namespace) -> int:
    if args.fixture != "bank":
        print(f"unknown context-op fixture: {args.fixture}", file=sys.stderr)
        return 2
    out = run_context_op(_bank_context_fixture(feedback=args.feedback))
    print(to_json(out))
    return 0


def _cmd_interference(args: argparse.Namespace) -> int:
    if args.fixture != "bank":
        print(f"unknown interference fixture: {args.fixture}", file=sys.stderr)
        return 2

    context_input = _bank_context_fixture(feedback=args.feedback)
    context_output = run_context_op(context_input)
    out = run_interference(
        InterferenceInput(
            context_frames=context_output.context_frames,
            candidate_frames=context_input.binding_candidate_frames,
            dmf_output=context_input.dmf_output,
            interference_gates=context_output.interference_gates,
            scoped_trace_assignments=context_output.scoped_trace_assignments,
            frame_links=context_output.frame_links,
            local_basin_pressures=context_output.local_basin_pressures,
        )
    )
    print(to_json(out))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    from lucid.audit.inspect import main as inspect_main

    return inspect_main(args.args)


def _cmd_gen(args: argparse.Namespace) -> int:
    from lucid.training.generator.cli import main as gen_main

    return gen_main(args.args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lucid")
    sub = parser.add_subparsers(dest="command", required=True)

    perceive_parser = sub.add_parser("perceive", help="Run perception on raw input")
    perceive_parser.add_argument("text", nargs="?", help="Raw text, or stdin when omitted")
    perceive_parser.add_argument("--modality", default="text", choices=[m.value for m in Modality])
    perceive_parser.add_argument("--backend", default="", choices=["", "rule", "llm"])
    perceive_parser.add_argument(
        "--compact",
        action="store_true",
        help="Print only non-empty lists and non-default fields",
    )
    perceive_parser.set_defaults(func=_cmd_perceive)

    run_parser = sub.add_parser("run", help="Run one Episode JSON through the pipeline")
    run_parser.add_argument("episode", help="Path to Episode JSON or JSONL")
    run_parser.add_argument("--audit-dir", default="audit", help="Audit base directory")
    run_parser.add_argument("--perception", default="", choices=["", "rule", "llm"])
    run_parser.set_defaults(func=_cmd_run)

    context_parser = sub.add_parser("context-op", help="Run context-op on a built-in fixture")
    context_parser.add_argument("--fixture", default="bank", choices=["bank"])
    context_parser.add_argument(
        "--feedback",
        action="append",
        default=[],
        help="Lucidity feedback token, e.g. SEARCH_WIDER",
    )
    context_parser.set_defaults(func=_cmd_context_op)

    interference_parser = sub.add_parser("interference", help="Run interference on a built-in fixture")
    interference_parser.add_argument("--fixture", default="bank", choices=["bank"])
    interference_parser.add_argument(
        "--feedback",
        action="append",
        default=[],
        help="Lucidity feedback token passed through context-op first",
    )
    interference_parser.set_defaults(func=_cmd_interference)

    inspect_parser = sub.add_parser("inspect", help="Inspect audit output")
    inspect_parser.add_argument("args", nargs=argparse.REMAINDER)
    inspect_parser.set_defaults(func=_cmd_inspect)

    gen_parser = sub.add_parser("gen", help="Run training generator commands")
    gen_parser.add_argument("args", nargs=argparse.REMAINDER)
    gen_parser.set_defaults(func=_cmd_gen)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
