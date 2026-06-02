"""Universal command line entrypoint for Lucid runtime tools."""

from __future__ import annotations

import argparse
import json
import sys
from json import JSONDecodeError
from pathlib import Path

from lucid.audit.cue import write_cue_encoder_audit
from lucid.cognition.input.cue import CueEncoderConfig, encode_cues
from lucid.cognition.input.perception import PerceptionConfig, perceive, to_compact_json
from lucid.cognition.orchestrator.runner import OrchestratorConfig, OrchestratorRunner
from lucid.cognition.projector import run_projector
from lucid.cognition.reasoning.binding import BindingConfig, run_binding
from lucid.cognition.reasoning.context_op import run_context_op
from lucid.audit.binding import write_binding_audit
from lucid.ir.binding import BindingInput
from lucid.memory.dmf import load_dynamic_memory_field
from lucid.ir.binding import CandidateFrame
from lucid.ir.common import AmbiguityPolicy, ComputePolicy, Modality, MaturityState
from lucid.ir.cue import CueCloud, CueEncoderInput, TraceActivationRequest
from lucid.ir.context_op import ContextOpInput
from lucid.ir.dmf import ActiveTrace, ConflictSignal, DmfInput, DmfOutput
from lucid.ir.lucidity import SearchDirectives
from lucid.ir.perception import CandidateUnit, PerceptionInput, PerceptualEvidenceGraph, ReferenceHint
from lucid.ir.projector import ProjectionConstraints, ProjectionGridPair, ProjectorInput
from lucid.ir.serde import from_json, to_json
from lucid.ir.training import Episode
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
        config=OrchestratorConfig(
            audit_base_dir=args.audit_dir,
            perception=perception_cfg,
            checkpoint=args.checkpoint,
        )
    )
    run = runner.run_episode(episode)
    print(run.context.audit_dir or "(audit written)")
    return 0


def _cue_fixture_text(name: str) -> str:
    if name == "bank":
        return "I found money while kayaking and placed it in the bank."
    raise ValueError(f"unknown cue-encoder fixture: {name}")


def _cmd_cue_encoder(args: argparse.Namespace) -> int:
    try:
        raw = args.text or _cue_fixture_text(args.fixture)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    perception_cfg = PerceptionConfig.from_env()
    perception_cfg.backend = args.backend
    graph = perceive(
        PerceptionInput(raw_payload=raw, modality=Modality(args.modality)),
        config=perception_cfg,
    )
    cue_input = CueEncoderInput(
        perceptual_evidence_graph=graph,
        task_intent_hint=args.task_intent,
        retrieval_budget=args.retrieval_budget,
        ambiguity_policy_in=AmbiguityPolicy(args.ambiguity_policy),
    )
    cloud = encode_cues(cue_input, config=CueEncoderConfig(checkpoint=args.checkpoint))
    write_cue_encoder_audit(
        audit_base_dir=args.audit_dir,
        cue_input=cue_input,
        cue_cloud=cloud,
        details={"checkpoint": args.checkpoint, "fixture": args.fixture, "text": raw},
    )
    print(to_json(cloud))
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
    out = run_context_op(_bank_context_fixture(feedback=args.feedback))
    print(to_json(out))
    return 0


def _grid_move_projector_fixture(max_rollouts: int) -> ProjectorInput:
    return ProjectorInput(
        projection_request=SearchDirectives(
            projector_targets=["asy_grid_candidate"],
            max_rollouts=max_rollouts,
        ),
        constraints=ProjectionConstraints(
            train_pairs=[
                ProjectionGridPair(
                    pair_id="train_0",
                    input_grid=[[0, 1, 0], [0, 0, 0]],
                    output_grid=[[0, 0, 1], [0, 0, 0]],
                )
            ],
            test_inputs=[[[2, 0, 0], [0, 0, 0]]],
            max_rollouts=max_rollouts,
        ),
        task_intent="solve_grid",
    )


def _cmd_projector(args: argparse.Namespace) -> int:
    if args.fixture != "grid-move":
        print(f"unknown projector fixture: {args.fixture}", file=sys.stderr)
        return 2
    out = run_projector(_grid_move_projector_fixture(args.max_rollouts))
    print(to_json(out))
    return 0


def _binding_bank_dmf_output(cue: CueCloud) -> DmfOutput:
    return DmfOutput(
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
        coverage_score=0.7,
    )


def _cmd_bind(args: argparse.Namespace) -> int:
    try:
        raw = args.text or _cue_fixture_text(args.fixture)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    perception_cfg = PerceptionConfig.from_env()
    perception_cfg.backend = args.backend
    graph = perceive(
        PerceptionInput(raw_payload=raw, modality=Modality(args.modality)),
        config=perception_cfg,
    )
    cue_input = CueEncoderInput(
        perceptual_evidence_graph=graph,
        task_intent_hint=args.task_intent,
        retrieval_budget=args.retrieval_budget,
    )
    cloud = encode_cues(
        cue_input,
        config=CueEncoderConfig(checkpoint=args.checkpoint),
    )

    if args.checkpoint:
        dmf = load_dynamic_memory_field(args.checkpoint, audit_base_dir=args.audit_dir)
        dmf_output = dmf.run(
            DmfInput(
                cue_cloud=cloud,
                compute_policy=ComputePolicy(max_active_traces=args.max_active),
            )
        )
    else:
        dmf_output = _binding_bank_dmf_output(cloud)

    binding_input = BindingInput(
        dmf_output=dmf_output,
        perceptual_evidence_graph=graph,
        cue_cloud=cloud,
    )
    binding_output = run_binding(
        binding_input,
        config=BindingConfig(checkpoint=args.checkpoint or None),
    )
    write_binding_audit(
        audit_base_dir=args.audit_dir,
        binding_input=binding_input,
        binding_output=binding_output,
        details={"checkpoint": args.checkpoint, "fixture": args.fixture, "text": raw},
    )
    print(to_json(binding_output))
    return 0


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
    success = kind == "high-margin"
    return (
        RunLog(
            episode_id=f"governor-{kind}",
            raw_input="ok" if success else "bad",
            evidence_graph={"entities": ["x"]},
            cue_cloud={"cue": "x"},
            active_traces=["t0001"],
            trace_clusters=[],
            candidate_bindings=[{"binding_id": "bind-1"}],
            context_frames=[{"frame_id": "ctx-1"}],
            scoped_trace_assignments={"t0001": "ctx-1"},
            interference_edges=[],
            active_basins=[{"basin_id": "b0001"}],
            basin_assemblies={"answer": "ok" if success else "wrong"},
            lucidity_features={},
            lucidity_decision="commit",
            lucidity_margin=0.91 if success else 0.88,
            projection_result=None,
            decoder_output={"answer": "ok" if success else "wrong"},
            validator_result={},
            cost_metrics={"stages_run": 8, "projector_called": False},
        ),
        ValidationResult(
            success,
            1.0 if success else 0.0,
            [] if success else ["exact_match_failed"],
            {"answer": "ok"},
            1.0,
        ),
    )


def _cmd_governor(args: argparse.Namespace) -> int:
    run_log, validation = _governor_fixture(args.fixture)
    governor = TrainingGovernor()
    decision = governor.observe(run_log, validation)
    if decision.action == "UPDATE":
        diagnosis = BlameAssigner().diagnose(run_log, validation)
        proposal = UpdatePlanner().plan(diagnosis, run_log)
        decision = governor.decide_update(run_log, validation, diagnosis, proposal)
    print(to_json(decision))
    return 0


def _cmd_quantization(args: argparse.Namespace) -> int:
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


def _cmd_inspect(args: argparse.Namespace) -> int:
    from lucid.audit.inspect import main as inspect_main

    return inspect_main(args.args)


def _cmd_gen(args: argparse.Namespace) -> int:
    from lucid.training.generator.cli import main as gen_main

    return gen_main(args.args)


def _cmd_scaling_summary(args: argparse.Namespace) -> int:
    from lucid.audit.scaling import ScalingConfig, format_summary, load_points, summarize_file, summarize_points

    cfg = ScalingConfig.from_env()
    if args.scale_id:
        print(format_summary(summarize_file(cfg, scale_id=args.scale_id)))
        return 0
    points = load_points(cfg.points_path)
    if not points:
        print(f"no points at {cfg.points_path}", file=sys.stderr)
        return 1
    by_scale: dict[str, list] = {}
    for row in points:
        by_scale.setdefault(str(row.get("scale_id") or ""), []).append(row)
    for sid in sorted(by_scale):
        print(format_summary(summarize_points(by_scale[sid], scale_id=sid)))
        print()
    return 0


def _cmd_scaling_export(args: argparse.Namespace) -> int:
    from lucid.audit.scaling import ScalingConfig, export_summary_csv, load_points

    cfg = ScalingConfig.from_env()
    points = load_points(cfg.points_path, scale_id=args.scale_id or None)
    if not points:
        print(f"no points at {cfg.points_path}", file=sys.stderr)
        return 1
    out = cfg.exports_dir / (args.out or "summary_by_scale_id.csv")
    export_summary_csv(points, out)
    print(out)
    return 0


def _cmd_scaling_path(_args: argparse.Namespace) -> int:
    from lucid.audit.scaling import ScalingConfig

    print(ScalingConfig.from_env().points_path)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from lucid.training.cli import main as train_main

    return train_main(args.args)


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
    run_parser.add_argument("--checkpoint", default="", help="Checkpoint for runtime stores")
    run_parser.set_defaults(func=_cmd_run)

    cue_parser = sub.add_parser("cue-encoder", help="Run cue encoder on text or a fixture")
    cue_parser.add_argument("text", nargs="?", help="Raw text; fixture is used when omitted")
    cue_parser.add_argument("--fixture", default="bank", choices=["bank"])
    cue_parser.add_argument("--checkpoint", default="", help="Checkpoint with cue_encoder_map.json")
    cue_parser.add_argument("--backend", default="rule", choices=["rule", "llm"])
    cue_parser.add_argument("--modality", default="text", choices=["text"])
    cue_parser.add_argument("--task-intent", default="answer")
    cue_parser.add_argument("--retrieval-budget", type=int, default=128)
    cue_parser.add_argument(
        "--ambiguity-policy",
        default=AmbiguityPolicy.PRESERVE_PLURAL.value,
        choices=[policy.value for policy in AmbiguityPolicy],
    )
    cue_parser.add_argument("--audit-dir", default="audit/cue_encoder")
    cue_parser.set_defaults(func=_cmd_cue_encoder)

    context_parser = sub.add_parser("context-op", help="Run context-op on a built-in fixture")
    context_parser.add_argument("--fixture", default="bank", choices=["bank"])
    context_parser.add_argument(
        "--feedback",
        action="append",
        default=[],
        help="Lucidity feedback token, e.g. SEARCH_WIDER",
    )
    context_parser.set_defaults(func=_cmd_context_op)

    projector_parser = sub.add_parser("projector", help="Run projector on a built-in fixture")
    projector_parser.add_argument("--fixture", default="grid-move", choices=["grid-move"])
    projector_parser.add_argument("--max-rollouts", type=int, default=1)
    projector_parser.set_defaults(func=_cmd_projector)

    bind_parser = sub.add_parser("bind", help="Run binding on text or a fixture")
    bind_parser.add_argument("text", nargs="?", help="Raw text; fixture is used when omitted")
    bind_parser.add_argument("--fixture", default="bank", choices=["bank"])
    bind_parser.add_argument("--checkpoint", default="", help="Checkpoint for cue encoder and DMF")
    bind_parser.add_argument("--backend", default="rule", choices=["rule", "llm"])
    bind_parser.add_argument("--modality", default="text", choices=["text"])
    bind_parser.add_argument("--task-intent", default="answer")
    bind_parser.add_argument("--retrieval-budget", type=int, default=128)
    bind_parser.add_argument("--max-active", type=int, default=8)
    bind_parser.add_argument("--audit-dir", default="audit/binding")
    bind_parser.set_defaults(func=_cmd_bind)

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

    train_parser = sub.add_parser("train", help="Run module or global training commands")
    train_parser.add_argument("args", nargs=argparse.REMAINDER)
    train_parser.set_defaults(func=_cmd_train)

    inspect_parser = sub.add_parser("inspect", help="Inspect audit output")
    inspect_parser.add_argument("args", nargs=argparse.REMAINDER)
    inspect_parser.set_defaults(func=_cmd_inspect)

    gen_parser = sub.add_parser("gen", help="Run training generator commands")
    gen_parser.add_argument("args", nargs=argparse.REMAINDER)
    gen_parser.set_defaults(func=_cmd_gen)

    scaling_parser = sub.add_parser("scaling", help="Scaling observatory (cost/quality receipts)")
    scaling_sub = scaling_parser.add_subparsers(dest="scaling_cmd", required=True)

    scaling_summary = scaling_sub.add_parser("summary", help="Rollup of scaling points")
    scaling_summary.add_argument("--scale-id", default="", help="Filter to one scale_id")
    scaling_summary.set_defaults(func=_cmd_scaling_summary)

    scaling_export = scaling_sub.add_parser("export", help="CSV aggregate by scale_id")
    scaling_export.add_argument("--scale-id", default="", help="Filter before export")
    scaling_export.add_argument("--out", default="", help="Filename under audit/scaling/exports/")
    scaling_export.set_defaults(func=_cmd_scaling_export)

    scaling_path = scaling_sub.add_parser("path", help="Print points.jsonl path")
    scaling_path.set_defaults(func=_cmd_scaling_path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
