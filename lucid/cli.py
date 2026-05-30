"""Universal command line entrypoint for Lucid runtime tools."""

from __future__ import annotations

import argparse
import json
import sys
from json import JSONDecodeError
from pathlib import Path

from lucid.cognition.reasoning.context_op import run_context_op
from lucid.cognition.input.perception import PerceptionConfig, perceive, to_compact_json
from lucid.cognition.orchestrator.runner import OrchestratorConfig, OrchestratorRunner
from lucid.ir.common import Modality
from lucid.ir.binding import CandidateFrame
from lucid.ir.context_op import ContextOpInput
from lucid.ir.dmf import ActiveTrace, ConflictSignal, DmfOutput
from lucid.ir.perception import PerceptionInput
from lucid.ir.perception import CandidateUnit, PerceptualEvidenceGraph, ReferenceHint
from lucid.ir.serde import from_json, to_json
from lucid.ir.training import Episode
<<<<<<< Updated upstream
=======
from lucid.memory.dmf import DmfTraceRecord, DynamicMemoryField
from lucid.training.dmf import learn_from_episode
from lucid.training.orchestrator.orchestrator import (
    BlameAssigner,
    RunLog,
    TrainingGovernor,
    UpdatePlanner,
    ValidationResult,
)
from lucid.training.quantization import (
    RetrievalQualitySample,
    binary_signature,
    measure_candidate_quality,
    rank_by_popcount,
)
>>>>>>> Stashed changes


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


<<<<<<< Updated upstream
=======
def _parse_cue(text: str) -> TraceActivationRequest:
    if "=" not in text:
        raise ValueError(f"cue must look like cue_key=weight, got {text!r}")
    key, raw_weight = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError("cue key cannot be empty")
    return TraceActivationRequest(trace_id=key, weight=float(raw_weight.strip()))


def _dmf_fixture(audit_dir: str) -> tuple[DynamicMemoryField, CueCloud]:
    dmf = DynamicMemoryField(
        [
            DmfTraceRecord(
                trace_id="t0001",
                alias="money/value-like",
                cue_affinities={"money": 0.92, "cash": 0.72},
                cluster_id="c_value",
                maturity_state=MaturityState.ACTIVE.value,
            ),
            DmfTraceRecord(
                trace_id="t0002",
                alias="placed/transfer-like",
                cue_affinities={"placed": 0.84, "deposit": 0.76},
                cluster_id="c_transfer",
                maturity_state=MaturityState.ACTIVE.value,
            ),
            DmfTraceRecord(
                trace_id="t0003",
                alias="outdoor/water-like",
                cue_affinities={"kayaking": 0.88, "river": 0.7},
                cluster_id="c_outdoor",
                maturity_state=MaturityState.ACTIVE.value,
            ),
            DmfTraceRecord(
                trace_id="t0004",
                alias="bank ambiguity-like",
                cue_affinities={"bank": 0.82},
                cluster_id="c_place",
                maturity_state=MaturityState.ACTIVE.value,
            ),
        ],
        audit_base_dir=audit_dir,
    )
    cue = CueCloud(
        primitive_trace_activations=[
            TraceActivationRequest(trace_id="money", weight=0.9, evidence_refs=["u_money"]),
            TraceActivationRequest(trace_id="placed", weight=0.8, evidence_refs=["u_placed"]),
            TraceActivationRequest(trace_id="bank", weight=0.75, evidence_refs=["u_bank"]),
            TraceActivationRequest(trace_id="kayaking", weight=0.65, evidence_refs=["u_kayaking"]),
        ],
        retrieval_budget_used=4,
    )
    return dmf, cue


def _cmd_dmf(args: argparse.Namespace) -> int:
    if args.fixture != "bank":
        print(f"unknown DMF fixture: {args.fixture}", file=sys.stderr)
        return 2
    try:
        dmf, cue = _dmf_fixture(args.audit_dir)
        if args.cue:
            cue = CueCloud(primitive_trace_activations=[_parse_cue(item) for item in args.cue])
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.learn:
        learn_from_episode(dmf, cue, winning_trace_indices=[0], spawn_if_novel=False)

    out = dmf.run(
        DmfInput(
            cue_cloud=cue,
            compute_policy=ComputePolicy(max_active_traces=args.max_active),
        )
    )
    print(to_json(out))
    return 0


def _governor_fixture(kind: str) -> tuple[RunLog, ValidationResult]:
    if kind == "high-margin":
        return (
            RunLog(
                episode_id="governor-high-margin",
                raw_input="ok",
                evidence_graph={"entities": ["ok"]},
                cue_cloud={"cue": "ok"},
                active_traces=["t0001"],
                trace_clusters=[],
                candidate_bindings=[{"binding_id": "bind-1"}],
                context_frames=[{"frame_id": "ctx-1"}],
                scoped_trace_assignments={"t0001": "ctx-1"},
                interference_edges=[],
                active_basins=[{"basin_id": "b0001"}],
                basin_assemblies={"answer": "ok"},
                lucidity_features={},
                lucidity_decision="commit",
                lucidity_margin=0.91,
                projection_result=None,
                decoder_output={"answer": "ok"},
                validator_result={},
                cost_metrics={"stages_run": 8, "projector_called": False},
            ),
            ValidationResult(True, 1.0, [], {"answer": "ok"}, 1.0),
        )
    if kind == "failure":
        return (
            RunLog(
                episode_id="governor-failure",
                raw_input="bad",
                evidence_graph={"entities": ["bad"]},
                cue_cloud={"cue": "bad"},
                active_traces=["t0001"],
                trace_clusters=[],
                candidate_bindings=[{"binding_id": "bind-1"}],
                context_frames=[{"frame_id": "ctx-1"}],
                scoped_trace_assignments={"t0001": "ctx-1"},
                interference_edges=[],
                active_basins=[{"basin_id": "b0001"}],
                basin_assemblies={"answer": "wrong"},
                lucidity_features={},
                lucidity_decision="commit",
                lucidity_margin=0.88,
                projection_result=None,
                decoder_output={"answer": "wrong"},
                validator_result={"expected_state": {"answer": "right"}},
                cost_metrics={"stages_run": 8, "projector_called": False},
            ),
            ValidationResult(False, 0.0, ["exact_match_failed"], {"answer": "right"}, 1.0),
        )
    raise ValueError(f"unknown governor fixture: {kind}")


def _cmd_governor(args: argparse.Namespace) -> int:
    try:
        run_log, validation = _governor_fixture(args.fixture)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    governor = TrainingGovernor()
    decision = governor.observe(run_log, validation)
    if decision.action == "UPDATE":
        diagnosis = BlameAssigner().diagnose(run_log, validation)
        proposal = UpdatePlanner().plan(diagnosis, run_log)
        decision = governor.decide_update(run_log, validation, diagnosis, proposal)
    print(to_json(decision))
    return 0


def _cmd_quantization(args: argparse.Namespace) -> int:
    if args.fixture != "retrieval":
        print(f"unknown quantization fixture: {args.fixture}", file=sys.stderr)
        return 2

    cue = binary_signature({"money": 1.0, "bank": 1.0, "kayaking": 0.0})
    records = {
        "t0001": binary_signature({"money": 1.0, "bank": 1.0}),
        "t0002": binary_signature({"kayaking": 1.0, "river": 1.0}),
        "t0003": binary_signature({"bank": 1.0}),
    }
    ranked = rank_by_popcount(cue, records, top_k=2)
    measurement = measure_candidate_quality(
        [
            RetrievalQualitySample(
                sample_id="bank-fixture",
                exact_top_ids=["t0001", "t0003"],
                candidate_top_ids=ranked,
                exact_margin=0.2,
                candidate_margin=0.2,
            )
        ],
        k=2,
    )
    print(
        json.dumps(
            {
                "ranked_ids": ranked,
                "measurement": json.loads(to_json(measurement)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
=======
    dmf_parser = sub.add_parser("dmf", help="Run DMF on a built-in tracebank fixture")
    dmf_parser.add_argument("--fixture", default="bank", choices=["bank"])
    dmf_parser.add_argument(
        "--cue",
        action="append",
        default=[],
        help="Override fixture cue with cue_key=weight; repeat for multiple cues",
    )
    dmf_parser.add_argument("--max-active", type=int, default=4)
    dmf_parser.add_argument("--audit-dir", default="audit/dmf")
    dmf_parser.add_argument("--learn", action="store_true", help="Apply one audited learning step")
    dmf_parser.set_defaults(func=_cmd_dmf)

    governor_parser = sub.add_parser("governor", help="Run training governor on a fixture")
    governor_parser.add_argument(
        "--fixture",
        default="high-margin",
        choices=["high-margin", "failure"],
    )
    governor_parser.set_defaults(func=_cmd_governor)

    quant_parser = sub.add_parser(
        "quantization",
        help="Run training quantization measurement fixture",
    )
    quant_parser.add_argument("--fixture", default="retrieval", choices=["retrieval"])
    quant_parser.set_defaults(func=_cmd_quantization)

>>>>>>> Stashed changes
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
