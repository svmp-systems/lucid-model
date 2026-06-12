"""Training CLI for checkpoint-backed module and global runs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from lucid.audit.logger import content_hash
from lucid.runtime.paths import DEFAULT_AUDIT_TRAINING_RUNS, DEFAULT_AUDIT_VALIDATION, DEFAULT_TRAINING_CHECKPOINT, resolve_train_path
from lucid.ir.serde import to_dict
from lucid.ir.training import Episode
from lucid.training.corpus import adapters
from lucid.training.checkpoint.store import (
    CheckpointState,
    checkpoint_summary,
    load_checkpoint,
    save_checkpoint,
)
from lucid.training.checkpoint.slots import archive_training_checkpoint, checkpoint_ref, promote_to_loaded
from lucid.training.loop.orchestrator import (
    FailureDiagnosis,
    RunLog,
    TrainingGovernor,
    UpdatePlanner,
    ValidationResult,
)
from lucid.training.modules import get_trainer, trainer_names
from dataclasses import replace

from lucid.training.modules.base import TrainingResult, utc_now_iso
from lucid.training.validate.gold import (
    GoldEpisodeValidator,
    L3ModuleGoldValidator,
    to_training_episode,
    validate_episode_pack,
)
from lucid.training.loop.promotion import CheckpointPromotionHook
from lucid.training.loop.pipeline import PipelineRunExecutor
from lucid.training.loop.orchestrator import TrainingOrchestrator
from lucid.training.loop.pipeline import _PipelineStage


MODULE_TO_DIAGNOSIS = {
    "perception": "perception",
    "cue_encoder": "cue_encoder",
    "dmf": "dmf",
    "binding": "binding",
    "context-op": "context_op",
    "interference": "interference_or_basin",
    "basins": "basins",
    "lucidity": "lucidity_too_strict",
    "projector": "projector",
    "decoder": "decoder",
}


def _safe_part(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return (clean.strip("_") or "train")[:80]


def _new_run_id(prefix: str, checkpoint_id: str) -> str:
    return f"{_safe_part(prefix)}_{_safe_part(checkpoint_id)}_{uuid4().hex[:10]}"


def _finalize_training(args: argparse.Namespace, *, command: str) -> dict[str, Any] | None:
    """Archive training workspace to cp_NNN and optionally pin for inference."""
    if args.dry_run:
        return None

    label = getattr(args, "save_label", "") or ""
    save_as = getattr(args, "save_as", "") or ""
    record: dict[str, Any] | None = None
    if not getattr(args, "no_save", False):
        record = archive_training_checkpoint(
            source=args.checkpoint,
            name=save_as or None,
            label=label,
            command=command,
        )

    if getattr(args, "pin", False):
        if record:
            promote_to_loaded(record["name"], label=label or str(record.get("label", "")))
            record["pinned"] = True
        else:
            promote_to_loaded(args.checkpoint, label=label or command)
            record = {
                "pinned": True,
                "name": checkpoint_ref(args.checkpoint),
                "path": str(resolve_train_path(args.checkpoint)),
            }
    return record


def _load_episodes(args: argparse.Namespace) -> list[Episode]:
    if args.episodes:
        return adapters.load_training_episodes(args.episodes)
    return adapters.fixture_episodes(args.fixture)


def _bounded_episodes(episodes: list[Episode], steps: int) -> list[Episode]:
    if not episodes:
        raise RuntimeError("no episodes available for training")
    count = max(1, int(steps))
    return [episodes[idx % len(episodes)] for idx in range(count)]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_train_manifest(
    run_dir: Path,
    *,
    command: str,
    checkpoint: str,
    results: list[TrainingResult],
    dry_run: bool,
    before: dict[str, Any],
    after: dict[str, Any],
    governor_records: list[dict[str, Any]] | None = None,
) -> None:
    updates = [result for result in results if result.action == "UPDATE"]
    files = {
        "manifest": "manifest.json",
        "governor_decision": "governor_decision.json",
        "module_update": "module_update.json",
        "before": "before.json",
        "after": "after.json",
        "metrics": "metrics.json",
        "readme": "README.txt",
    }
    metrics = {
        "schema_version": 1,
        "step_count": len(results),
        "update_count": len(updates),
        "defer_count": sum(1 for result in results if result.action == "DEFER"),
        "no_update_count": sum(1 for result in results if result.action == "NO_UPDATE"),
        "dry_run": dry_run,
    }
    module_update = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "command": command,
        "results": [
            {
                "module": result.module,
                "action": result.action,
                "episode_id": result.episode_id,
                "updated_objects": result.updated_objects,
                "before_hash": result.before_hash,
                "after_hash": result.after_hash,
                "audit_path": result.audit_path,
                "reason": result.reason,
                "metrics": result.metrics,
            }
            for result in results
        ],
    }
    governor_payload = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "command": command,
        "records": governor_records
        or [
            {
                "action": "NOT_APPLICABLE",
                "reason": "direct_module_training_command",
            }
        ],
    }
    payload = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "command": command,
        "checkpoint": checkpoint,
        "dry_run": dry_run,
        "step_count": metrics["step_count"],
        "update_count": metrics["update_count"],
        "defer_count": metrics["defer_count"],
        "files": files,
    }
    _write_json(run_dir / files["before"], before)
    _write_json(run_dir / files["after"], after)
    _write_json(run_dir / files["metrics"], metrics)
    _write_json(run_dir / files["module_update"], module_update)
    _write_json(run_dir / files["governor_decision"], governor_payload)
    _write_json(run_dir / files["manifest"], payload)
    (run_dir / files["readme"]).write_text(
        "\n".join(
            [
                f"{command} training run",
                "=" * (len(command) + 13),
                "",
                f"checkpoint: {checkpoint}",
                f"dry_run: {dry_run}",
                f"steps: {len(results)}",
                f"updates: {len(updates)}",
                f"defers: {metrics['defer_count']}",
                "",
                "files:",
                *[f"- {name}: {file_name}" for name, file_name in files.items()],
                "",
            ]
        ),
        encoding="utf-8",
    )


def _run_module_training(args: argparse.Namespace) -> int:
    trainer = get_trainer(args.target)
    episodes = _bounded_episodes(_load_episodes(args), args.steps)
    state = load_checkpoint(args.checkpoint, create=True)
    before_summary = checkpoint_summary(state)
    run_dir = resolve_train_path(args.audit_dir) / _new_run_id(args.target, state.checkpoint_id)

    results: list[TrainingResult] = []
    working_state = copy.deepcopy(state) if args.dry_run else state
    for step_index, episode in enumerate(episodes, start=1):
        step_dir = run_dir / f"step_{step_index:06d}_{_safe_part(episode.episode_id)}"
        step_episode = episode
        if args.target == "cue_encoder" and getattr(args, "mode", None):
            step_episode = replace(episode, meta={**episode.meta, "train_mode": args.mode})
        result = trainer.train(step_episode, working_state, step_dir)
        results.append(result)

    if not args.dry_run:
        save_checkpoint(state, args.checkpoint, force=args.force, step_delta=len(results))

    after_summary = checkpoint_summary(working_state)
    _write_train_manifest(
        run_dir,
        command=f"train {args.target}",
        checkpoint=args.checkpoint,
        results=results,
        dry_run=args.dry_run,
        before=before_summary,
        after=after_summary,
    )
    archived = _finalize_training(args, command=f"train {args.target}")
    payload: dict[str, Any] = {
        "target": args.target,
        "checkpoint": args.checkpoint,
        "audit_dir": str(run_dir),
        "dry_run": args.dry_run,
        "steps": len(results),
        "updates": sum(1 for result in results if result.action == "UPDATE"),
        "defers": sum(1 for result in results if result.action == "DEFER"),
        "checkpoint_summary": checkpoint_summary(working_state),
    }
    if archived:
        payload["archived_save"] = archived
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _known_trace_families(state: CheckpointState) -> set[str]:
    return {
        str(record.get("trace_family"))
        for record in state.ensure_store("tracebank").get("records", [])
    }


def _known_basin_families(state: CheckpointState) -> set[str]:
    return {
        str(record.get("family_hint"))
        for record in state.ensure_store("basin_bank").get("records", [])
    }


def _global_target_module(episode: Episode, state: CheckpointState) -> str:
    trace_families = {target["trace_family"] for target in adapters.dmf_targets(episode)}
    if trace_families - _known_trace_families(state):
        return "dmf"

    lucidity = adapters.lucidity_target(episode)["decision"]
    lucidity_counts = state.ensure_store("lucidity_policy").get("decision_counts", {})
    if lucidity and lucidity not in lucidity_counts:
        return "lucidity"

    decoder = adapters.decoder_target(episode)
    decoder_records = state.ensure_store("decoder_adapter").get("render_targets", [])
    if decoder["expected_answer"] is not None and not any(
        record.get("episode_id") == episode.episode_id for record in decoder_records
    ):
        return "decoder"

    context = adapters.context_targets(episode)
    context_store = state.ensure_store("context_policy")
    if context["scope_assignments"] and not context_store.get("scope_patterns"):
        return "context-op"

    if context["interference_gates"] and not state.ensure_store("interference_graph").get("gates"):
        return "interference"

    basin_families = {target["family_hint"] for target in adapters.basin_targets(episode)}
    if basin_families - _known_basin_families(state):
        return "basins"

    if adapters.projector_target(episode) and not state.ensure_store("projector_examples").get(
        "examples"
    ):
        return "projector"

    if any(adapters.perception_targets(episode).values()) and not state.ensure_store(
        "perception_examples"
    ).get("examples"):
        return "perception"

    if adapters.cue_encoder_targets(episode)["trace_targets"] and not state.ensure_store(
        "cue_encoder_map"
    ).get("cue_targets"):
        return "cue_encoder"

    if adapters.binding_targets(episode) and not state.ensure_store("binding_affordances").get(
        "patterns"
    ):
        return "binding"

    return ""


def _global_run_log(episode: Episode, state: CheckpointState, target_module: str) -> RunLog:
    trace_ids = [
        record.get("trace_id", "")
        for record in state.ensure_store("tracebank").get("records", [])
        if record.get("trace_id")
    ]
    basin_ids = [
        record.get("basin_id", "")
        for record in state.ensure_store("basin_bank").get("records", [])
        if record.get("basin_id")
    ]
    return RunLog(
        episode_id=episode.episode_id,
        raw_input=episode.raw_input,
        evidence_graph={"template_id": episode.template_id, "modality": str(episode.modality)},
        cue_cloud={"trace_targets": adapters.dmf_targets(episode)},
        active_traces=trace_ids,
        trace_clusters=[],
        candidate_bindings=adapters.binding_targets(episode),
        context_frames=adapters.context_targets(episode)["scope_assignments"],
        scoped_trace_assignments={},
        interference_edges=adapters.interference_targets(episode),
        active_basins=[{"basin_id": basin_id} for basin_id in basin_ids],
        basin_assemblies={},
        lucidity_features={"target": adapters.lucidity_target(episode)},
        lucidity_decision="commit" if episode.gold.lucidity_target == "COMMIT" else "reject",
        lucidity_margin=0.9 if not target_module else 0.45,
        projection_result=None,
        decoder_output=adapters.decoder_target(episode)["expected_answer"],
        validator_result={},
        cost_metrics={"stages_run": 8, "projector_called": False},
    )


def _global_validation(target_module: str) -> ValidationResult:
    if not target_module:
        return ValidationResult(True, 1.0, [], "checkpoint already covers episode", 1.0)
    return ValidationResult(
        False,
        0.0,
        [f"missing_{target_module}_training_region"],
        f"train {target_module}",
        1.0,
    )


def _global_governor_decision(
    episode: Episode,
    state: CheckpointState,
    target_module: str,
    governor: TrainingGovernor,
):
    run_log = _global_run_log(episode, state, target_module)
    validation = _global_validation(target_module)
    if not target_module:
        return governor.observe(run_log, validation)
    diagnosis_name = MODULE_TO_DIAGNOSIS[target_module]
    diagnosis = FailureDiagnosis(
        diagnosis_name,
        [target_module],
        0.9,
        {"reason": f"checkpoint_missing_{target_module}_region"},
        {
            "perception": 9,
            "cue_encoder": 2,
            "dmf": 2,
            "binding": 4,
            "context-op": 5,
            "interference": 3,
            "basins": 6,
            "lucidity": 8,
            "projector": 8,
            "decoder": 9,
        }.get(target_module, 4),
    )
    proposal = UpdatePlanner().plan(diagnosis, run_log)
    return governor.decide_update(run_log, validation, diagnosis, proposal)


def _run_global_training(args: argparse.Namespace) -> int:
    episodes = _bounded_episodes(_load_episodes(args), args.steps)
    state = load_checkpoint(args.checkpoint, create=True)
    before_summary = checkpoint_summary(state)
    governor = TrainingGovernor()
    run_dir = resolve_train_path(args.audit_dir) / _new_run_id("global", state.checkpoint_id)
    results: list[TrainingResult] = []
    governor_records: list[dict[str, Any]] = []
    promoted = 0

    for step_index, episode in enumerate(episodes, start=1):
        step_dir = run_dir / f"step_{step_index:06d}_{_safe_part(episode.episode_id)}"
        target_module = _global_target_module(episode, state)
        decision = _global_governor_decision(episode, state, target_module, governor)
        _write_json(
            step_dir / "governor_decision.json",
            {
                "schema_version": 1,
                "decision": decision,
            },
        )
        governor_records.append(
            {
                "step_index": step_index,
                "episode_id": episode.episode_id,
                "target_module": target_module or "none",
                "decision": decision,
            }
        )

        if decision.action != "UPDATE" or not target_module:
            result = TrainingResult(
                module=target_module or "global",
                action=decision.action,
                episode_id=episode.episode_id,
                before_hash=content_hash(checkpoint_summary(state)),
                after_hash=content_hash(checkpoint_summary(state)),
                reason=decision.reason,
                audit_path=str(step_dir),
            )
            results.append(result)
            continue

        trainer = get_trainer(target_module)
        shadow_state = copy.deepcopy(state)
        shadow_result = trainer.train(episode, shadow_state, step_dir / "shadow")
        if shadow_result.action == "UPDATE" and shadow_result.before_hash != shadow_result.after_hash:
            if not args.dry_run:
                live_result = trainer.train(episode, state, step_dir / "live")
                results.append(live_result)
                promoted += 1
            else:
                results.append(shadow_result)
        else:
            results.append(shadow_result)

    if not args.dry_run:
        save_checkpoint(state, args.checkpoint, force=args.force, step_delta=len(results))

    after_summary = checkpoint_summary(state)
    _write_train_manifest(
        run_dir,
        command="train global",
        checkpoint=args.checkpoint,
        results=results,
        dry_run=args.dry_run,
        before=before_summary,
        after=after_summary,
        governor_records=governor_records,
    )
    archived = _finalize_training(args, command="train global")
    payload: dict[str, Any] = {
        "target": "global",
        "checkpoint": args.checkpoint,
        "audit_dir": str(run_dir),
        "dry_run": args.dry_run,
        "steps": len(results),
        "promoted_updates": promoted,
        "checkpoint_summary": checkpoint_summary(state),
    }
    if archived:
        payload["archived_save"] = archived
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_list(_args: argparse.Namespace) -> int:
    for name in trainer_names():
        print(name)
    for command in ("global", "loop", "validate", "golden"):
        print(command)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    state = load_checkpoint(args.checkpoint, create=True)
    print(json.dumps(checkpoint_summary(state), indent=2, sort_keys=True))
    return 0


def _training_episodes(args: argparse.Namespace) -> list:
    episodes = _load_episodes(args)
    training = [to_training_episode(episode) for episode in episodes]
    for row in training:
        row.validator_type = "gold_episode"
    return training


def _cmd_loop(args: argparse.Namespace) -> int:
    training = _training_episodes(args)
    executor = PipelineRunExecutor(
        checkpoint=args.checkpoint,
        audit_base_dir=args.audit_dir,
        perception_backend=args.perception,
    )
    orchestrator = TrainingOrchestrator(
        _PipelineStage("perception"),
        _PipelineStage("cue_encoder"),
        _PipelineStage("dmf"),
        _PipelineStage("binding"),
        _PipelineStage("context_op"),
        _PipelineStage("interference"),
        _PipelineStage("lucidity"),
        _PipelineStage("projector"),
        _PipelineStage("decoder"),
        training,
        audit_base_dir=args.audit_dir,
        executor=executor,
    )
    if not args.dry_run:
        orchestrator.live_state["checkpoint_hook"] = CheckpointPromotionHook(
            args.checkpoint,
            audit_dir=args.audit_dir,
        )
    orchestrator.run(max(1, int(args.steps)))
    status = orchestrator.get_status()
    archived = _finalize_training(args, command="train loop")
    if archived:
        status["archived_save"] = archived
    print(json.dumps(status, indent=2, sort_keys=True))
    if orchestrator.live_state.get("checkpoint_hook") is not None:
        print(json.dumps(orchestrator.live_state["checkpoint_hook"].summary(), indent=2, sort_keys=True))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    episodes_path = args.episodes
    if not episodes_path:
        from lucid.training.corpus.output import build_phase1_pack
        from lucid.runtime.paths import train_root

        out = train_root() / "data/generated/phase1"
        build_phase1_pack(out, seed=42)
        episodes_path = str(out / "all.jsonl")
    report = validate_episode_pack(
        episodes_path,
        checkpoint=args.checkpoint,
        audit_dir=args.audit_dir,
        limit=max(0, int(args.limit)),
    )
    out_path = Path(args.report) if args.report else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["crashes"] > 0:
        return 1
    if args.require_l3:
        failed = [row for row in report.get("l3_module_gold", []) if not row.get("passed")]
        if failed:
            return 2
    return 0


def _cmd_golden(args: argparse.Namespace) -> int:
    from lucid.training.corpus.engine import AmbiguityKnob, rng_for_seed
    from lucid.training.corpus.recipes import bank_destination, grid_move

    episodes = []
    if "bank" in args.suite:
        episodes.append(bank_destination.make(rng_for_seed(7), AmbiguityKnob(0.9)))
    if "grid" in args.suite:
        episodes.append(grid_move.make(rng_for_seed(9), AmbiguityKnob(0.95)))
    validator = GoldEpisodeValidator()
    from lucid.cognition.pipe_orchestrator.runner import OrchestratorConfig, OrchestratorRunner
    from lucid.cognition.input.perception import PerceptionConfig

    runner = OrchestratorRunner(
        config=OrchestratorConfig(
            audit_base_dir=args.audit_dir,
            perception=PerceptionConfig(backend="rule", write_audit=False),
            checkpoint=args.checkpoint or None,
        )
    )
    reports = []
    for episode in episodes:
        run = runner.run_episode(episode)
        reports.append(validator.evaluate_run(run, episode))
    print(json.dumps([{"episode_id": r.episode_id, "success": r.success, "signals": r.failure_signals} for r in reports], indent=2))
    return 0 if all(report.success for report in reports) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lucid train")
    sub = parser.add_subparsers(dest="target", required=True)
    sub.add_parser("list", help="List trainable modules").set_defaults(func=_cmd_list)

    status_p = sub.add_parser("status", help="Inspect checkpoint summary")
    status_p.add_argument("--checkpoint", default=DEFAULT_TRAINING_CHECKPOINT)
    status_p.set_defaults(func=_cmd_status)

    for name in trainer_names():
        p = sub.add_parser(name, help=f"Train {name}")
        _add_common_args(p)
        if name == "cue_encoder":
            p.add_argument(
                "--mode",
                default="calibrate",
                choices=["seed", "calibrate"],
                help="seed stores all gold routes; calibrate patches only missing cues",
            )
        p.set_defaults(func=_run_module_training)

    global_p = sub.add_parser("global", help="Run governor-directed global training")
    _add_common_args(global_p)
    global_p.set_defaults(func=_run_global_training)

    loop_p = sub.add_parser("loop", help="Run training orchestrator on the real pipeline")
    _add_common_args(loop_p)
    loop_p.add_argument("--perception", default="rule", choices=["rule", "llm"])
    loop_p.set_defaults(func=_cmd_loop)

    validate_p = sub.add_parser("validate", help="Score episodes against generator gold (L3)")
    validate_p.add_argument("--episodes", default="")
    validate_p.add_argument("--checkpoint", default=DEFAULT_TRAINING_CHECKPOINT)
    validate_p.add_argument("--audit-dir", default=DEFAULT_AUDIT_VALIDATION)
    validate_p.add_argument("--limit", type=int, default=0)
    validate_p.add_argument("--report", default="")
    validate_p.add_argument("--require-l3", action="store_true")
    validate_p.set_defaults(func=_cmd_validate)

    golden_p = sub.add_parser("golden", help="Run named phase-1 golden fixtures")
    golden_p.add_argument("--suite", nargs="+", default=["bank", "grid"], choices=["bank", "grid"])
    golden_p.add_argument("--checkpoint", default=DEFAULT_TRAINING_CHECKPOINT)
    golden_p.add_argument("--audit-dir", default=DEFAULT_AUDIT_VALIDATION)
    golden_p.set_defaults(func=_cmd_golden)
    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", default="", help="Episode JSONL path")
    parser.add_argument("--fixture", default="phase1-mini", choices=["phase1-mini", "bank"])
    parser.add_argument("--checkpoint", default=DEFAULT_TRAINING_CHECKPOINT)
    parser.add_argument("--audit-dir", default=DEFAULT_AUDIT_TRAINING_RUNS)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip auto archive to checkpoints/saves/cp_NNN after training",
    )
    parser.add_argument(
        "--save-as",
        default="",
        help="Archive as this name instead of the next cp_NNN",
    )
    parser.add_argument(
        "--save-label",
        default="",
        help="Human note stored in the checkpoint registry",
    )
    parser.add_argument(
        "--pin",
        action="store_true",
        help="After archive, copy the save into the loaded slot for lucid ask / run",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileExistsError, FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
