"""Universal command line entrypoint for Lucid runtime tools."""

from __future__ import annotations

import argparse
import json
import sys
from json import JSONDecodeError
from pathlib import Path

from lucid.audit.smoke import (
    write_basins_audit,
    write_binding_audit,
    write_cue_encoder_audit,
    write_decoder_audit,
    write_lucidity_audit,
)
from lucid.audit.chat import load_session_memory, new_session_memory, save_session_memory
from lucid.chat import list_sessions, run_chat_turn, start_session
from lucid.cognition.output.decoder import run_decoder
from lucid.cognition.output.lucidity import run_lucidity
from lucid.cognition.input.cue import CueEncoderConfig, encode_cues
from lucid.cognition.input.perception import PerceptionConfig, perceive, to_compact_json
from lucid.cognition.pipe_orchestrator.runner import OrchestratorConfig, OrchestratorRunner
from lucid.cognition.output.projector import run_projector
from lucid.cognition.reasoning.basins import BasinsConfig, run_basins
from lucid.cognition.reasoning.binding import BindingConfig, run_binding
from lucid.cognition.reasoning.context_op import run_context_op
from lucid.cognition.reasoning.interference import run_interference
from lucid.cognition.reasoning.interference import (
    DEFAULT_INTERFERENCE_LEARNING_AUDIT,
    DEFAULT_INTERFERENCE_STORE,
    learn_interference,
    load_learned_interference_links,
)
from lucid.ir.basins import BasinInput
from lucid.ir.binding import BindingInput, BindingOutput
from lucid.cognition.memory.dmf import load_dynamic_memory_field
from lucid.ir.binding import CandidateFrame
from lucid.ir.common import AmbiguityPolicy, ComputePolicy, Modality, MaturityState, TaskIntent
from lucid.ir.cue import CueCloud, CueEncoderInput, TraceActivationRequest
from lucid.ir.context_op import ContextOpInput
from lucid.ir.dmf import ActiveTrace, ConflictSignal, DmfInput, DmfOutput
from lucid.ir.basins import BasinOutput, CandidateBasinState, CompetitionSummary
from lucid.ir.common import DecoderMode, LucidityDecision
from lucid.ir.expression import DecoderInput
from lucid.ir.interference import InterferenceInput, InterferenceOutput
from lucid.ir.lucidity import (
    DecoderPolicy,
    LucidityInput,
    LucidityOutput,
    LucidityRenderPacket,
    RenderUnit,
    SearchDirectives,
    SourceRef,
)
from lucid.ir.projector import ProjectorOutput
from lucid.ir.perception import CandidateUnit, PerceptionInput, PerceptualEvidenceGraph, ReferenceHint
from lucid.ir.projector import ProjectionConstraints, ProjectionGridPair, ProjectorInput
from lucid.ir.serde import from_json, to_json
from lucid.ir.training import Episode
from lucid.cognition.memory.dmf import DmfTraceRecord, DynamicMemoryField
from lucid.training.learn.dmf import learn_from_episode
from lucid.training.loop.orchestrator import (
    BlameAssigner,
    RunLog,
    TrainingGovernor,
    UpdatePlanner,
    ValidationResult,
)
from lucid.runtime.paths import (
    DEFAULT_ASK_LATEST,
    DEFAULT_AUDIT_BINDING,
    DEFAULT_AUDIT_CUE_ENCODER,
    DEFAULT_AUDIT_DMF,
    DEFAULT_AUDIT_LUCIDITY,
    DEFAULT_AUDIT_MEMORY,
    DEFAULT_AUDIT_RUNS,
    DEFAULT_TRAINING_CHECKPOINT,
    smoke_audit_dir,
)
from lucid.training.checkpoint.slots import resolve_inference_checkpoint
from lucid.training.learn.quant import (
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
            checkpoint=resolve_inference_checkpoint(args.checkpoint, cold=args.cold) or "",
        )
    )
    run = runner.run_episode(episode)
    print(run.context.audit_dir or "(audit written)")
    return 0


def _cmd_run_batch(args: argparse.Namespace) -> int:
    path = Path(args.episode)
    if not path.exists():
        print(f"missing file: {path}", file=sys.stderr)
        return 2

    from lucid.training.corpus.output import read_episodes

    episodes = read_episodes(path)
    if args.limit > 0:
        episodes = episodes[: args.limit]

    perception_cfg = PerceptionConfig.from_env()
    if args.perception:
        perception_cfg.backend = args.perception
    runner = OrchestratorRunner(
        config=OrchestratorConfig(
            audit_base_dir=args.audit_dir,
            perception=perception_cfg,
            checkpoint=resolve_inference_checkpoint(args.checkpoint, cold=args.cold) or "",
        )
    )

    crashes = 0
    for index, episode in enumerate(episodes, start=1):
        try:
            runner.run_episode(episode)
        except Exception as exc:  # noqa: BLE001
            crashes += 1
            print(f"[{index}/{len(episodes)}] {episode.episode_id} crash: {exc}", file=sys.stderr)

    print(json.dumps({"episodes": len(episodes), "crashes": crashes}, indent=2, sort_keys=True))
    return 1 if crashes else 0


def _cmd_ask(args: argparse.Namespace) -> int:
    """Run the full pipeline on one sentence; print sentence, answer, compact audit."""
    sentence = " ".join(args.sentence).strip()
    if not sentence:
        print("no sentence", file=sys.stderr)
        return 2

    from lucid.audit.ask_report import (
        answer_from_pipeline_run,
        episode_id_for_sentence,
        write_ask_report,
    )

    episode = Episode(
        episode_id=episode_id_for_sentence(sentence),
        modality=Modality.TEXT,
        raw_input=sentence,
        task_intent=TaskIntent.ANSWER,
    )

    perception_cfg = PerceptionConfig.from_env()
    perception_cfg.backend = args.perception or "rule"

    runner = OrchestratorRunner(
        config=OrchestratorConfig(
            audit_base_dir=args.audit_dir,
            perception=perception_cfg,
            checkpoint=resolve_inference_checkpoint(args.checkpoint, cold=args.cold) or "",
        )
    )
    try:
        run = runner.run_episode(episode)
    except Exception as exc:  # noqa: BLE001 — surface pipeline failure to the user
        print(f"pipeline failed: {exc}", file=sys.stderr)
        return 1

    answer = answer_from_pipeline_run(run)
    lucidity = ""
    if run.lucidity_output is not None:
        decision = run.lucidity_output.decision
        lucidity = decision.value if hasattr(decision, "value") else str(decision)

    audit_dir = run.context.audit_dir
    if not audit_dir:
        print("audit directory missing after run", file=sys.stderr)
        return 1
    run_dir = Path(audit_dir)

    paths = write_ask_report(
        run_dir,
        sentence=sentence,
        answer=answer,
        lucidity_decision=lucidity,
        latest_path=args.latest or None,
        write_latest=not args.no_latest,
        extra_copy=args.out or None,
    )
    body = paths.run_report.read_text(encoding="utf-8").rstrip()
    try:
        print(body)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(body.encode(encoding, errors="replace").decode(encoding))
    return 0


def _cmd_edit(args: argparse.Namespace) -> int:
    import json as json_lib

    from lucid.cognition.memory.edit import edit_basin, edit_trace, list_basins, list_traces

    if args.list == "traces":
        print(json.dumps(list_traces(args.checkpoint), indent=2, sort_keys=True))
        return 0
    if args.list == "basins":
        print(json.dumps(list_basins(args.checkpoint), indent=2, sort_keys=True))
        return 0

    patch = json_lib.loads(args.patch)
    if args.kind == "trace":
        result = edit_trace(args.checkpoint, args.record_id, patch=patch, audit_dir=args.audit_dir)
    else:
        result = edit_basin(args.checkpoint, args.record_id, patch=patch, audit_dir=args.audit_dir)
    print(json.dumps({"store": result.store, "record_id": result.record_id, "audit_path": result.audit_path}, indent=2))
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


def _bank_interference_input(
    context_input: ContextOpInput,
    context_output: object,
    *,
    learned_interference_links: list | None = None,
) -> InterferenceInput:
    return InterferenceInput(
        context_frames=context_output.context_frames,
        candidate_frames=context_input.binding_candidate_frames,
        dmf_output=context_input.dmf_output,
        interference_gates=context_output.interference_gates,
        scoped_trace_assignments=context_output.scoped_trace_assignments,
        frame_links=context_output.frame_links,
        local_basin_pressures=context_output.local_basin_pressures,
        learned_interference_links=learned_interference_links or [],
    )


def _cmd_interference(args: argparse.Namespace) -> int:
    if args.fixture != "bank":
        print(f"unknown interference fixture: {args.fixture}", file=sys.stderr)
        return 2

    context_input = _bank_context_fixture(feedback=args.feedback)
    context_output = run_context_op(context_input)
    learned_links = load_learned_interference_links(args.store) if args.use_store else []
    out = run_interference(
        _bank_interference_input(
            context_input,
            context_output,
            learned_interference_links=learned_links,
        )
    )
    print(to_json(out))
    return 0


def _cmd_interference_learn(args: argparse.Namespace) -> int:
    if args.fixture != "bank":
        print(f"unknown interference learning fixture: {args.fixture}", file=sys.stderr)
        return 2

    context_input = _bank_context_fixture(feedback=args.feedback)
    context_output = run_context_op(context_input)
    learned_links = load_learned_interference_links(args.store)
    inp = _bank_interference_input(
        context_input,
        context_output,
        learned_interference_links=learned_links,
    )
    out = run_interference(inp)
    result = learn_interference(
        inp,
        out,
        validation_success=args.outcome == "success",
        failure_type=args.failure_type,
        store_path=args.store,
        audit_dir=args.audit_dir,
    )
    print(to_json(result))
    return 0


def _lucidity_bank_input(*, pass_kind: str, checkpoint: str) -> LucidityInput:
    context_in = _bank_context_fixture()
    context_out = run_context_op(context_in)
    interference_out = run_interference(_bank_interference_input(context_in, context_out))
    basin_input = BasinInput(
        interference_output=interference_out,
        candidate_frames=context_in.binding_candidate_frames,
        context_frames=context_out.context_frames,
        local_basin_pressures=context_out.local_basin_pressures,
    )
    basin_out = run_basins(basin_input, config=BasinsConfig(checkpoint=checkpoint or None))
    return LucidityInput(
        basin_output=basin_out,
        binding_output=BindingOutput(
            candidate_frames=context_in.binding_candidate_frames,
            binding_stability_score=0.72,
        ),
        context_op_output=context_out,
        interference_output=interference_out,
        dmf_output=context_in.dmf_output,
        perceptual_evidence_graph=context_in.perceptual_evidence_graph,
        task_intent="answer",
        risk_level="medium",
        pass_kind=pass_kind,
    )


def _lucidity_grid_input(*, pass_kind: str, projection: ProjectorOutput | None) -> LucidityInput:
    context_out = run_context_op(_bank_context_fixture())
    interference_out = InterferenceOutput()
    basin_out = BasinOutput(
        candidate_basin_states=[
            CandidateBasinState(basin_id="b_move", energy=0.9, margin_vs_next=0.03),
            CandidateBasinState(basin_id="b_alt", energy=0.5, margin_vs_next=0.0),
        ],
        competition_summary=CompetitionSummary(
            top_basin_id="b_move",
            second_basin_id="b_alt",
            top_margin=0.03,
            active_basin_count=2,
        ),
    )
    return LucidityInput(
        basin_output=basin_out,
        binding_output=BindingOutput(candidate_frames=[], binding_stability_score=0.8),
        context_op_output=context_out,
        interference_output=interference_out,
        dmf_output=DmfOutput(coverage_score=0.9),
        perceptual_evidence_graph=PerceptualEvidenceGraph(),
        task_intent="solve_grid",
        risk_level="high",
        pass_kind=pass_kind,
        projection_output=projection,
    )


def _cmd_lucidity(args: argparse.Namespace) -> int:
    if args.fixture == "bank":
        inp = _lucidity_bank_input(pass_kind=args.pass_kind, checkpoint=args.checkpoint)
    elif args.fixture == "grid":
        projection = None
        if args.pass_kind == "final_check":
            projection = run_projector(_grid_move_projector_fixture(args.max_rollouts))
        inp = _lucidity_grid_input(pass_kind=args.pass_kind, projection=projection)
    else:
        print(f"unknown lucidity fixture: {args.fixture}", file=sys.stderr)
        return 2

    out = run_lucidity(inp)
    write_lucidity_audit(
        audit_base_dir=args.audit_dir,
        lucidity_input=inp,
        lucidity_output=out,
        details={"fixture": args.fixture, "pass_kind": args.pass_kind},
    )
    print(to_json(out))
    return 0


def _cmd_basins(args: argparse.Namespace) -> int:
    context_in = _bank_context_fixture()
    context_out = run_context_op(context_in)
    basin_input = BasinInput(
        interference_output=InterferenceOutput(),
        candidate_frames=context_in.binding_candidate_frames,
        context_frames=context_out.context_frames,
        local_basin_pressures=context_out.local_basin_pressures,
        compute_policy=ComputePolicy(max_active_basins=args.max_active),
    )
    out = run_basins(
        basin_input,
        config=BasinsConfig(checkpoint=args.checkpoint or None, min_energy=args.min_energy),
    )
    write_basins_audit(
        audit_base_dir=args.audit_dir,
        basin_input=basin_input,
        basin_output=out,
        details={"checkpoint": args.checkpoint, "fixture": args.fixture},
    )
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


def _decoder_bank_packet() -> LucidityRenderPacket:
    return LucidityRenderPacket(
        packet_id="smoke-bank",
        decision=LucidityDecision.COMMIT,
        render_mode="committed",
        output_format="text",
        approved_units=[
            RenderUnit(
                unit_id="claim-bank",
                unit_type="claim",
                scope_frame_id="F2",
                payload={"bank_sense": "financial_storage"},
                required=True,
                source_refs=[
                    SourceRef(ref_type="trace", ref_id="t_money"),
                    SourceRef(ref_type="trace", ref_id="t_bank"),
                    SourceRef(ref_type="basin", ref_id="b_financial"),
                ],
            ),
            RenderUnit(
                unit_id="caveat-kayak",
                unit_type="caveat",
                payload={"kayaking_scope": "separate_event"},
                required=False,
                source_refs=[SourceRef(ref_type="frame", ref_id="F1")],
            ),
        ],
    )


def _cmd_decoder(args: argparse.Namespace) -> int:
    if args.fixture == "bank":
        packet = _decoder_bank_packet()
        policy = DecoderPolicy(
            mode=DecoderMode.EXPRESS_COMMITTED.value,
            output_channel="chat",
            max_sentences=args.max_sentences,
        )
    elif args.fixture == "plural":
        packet = LucidityRenderPacket(
            packet_id="smoke-plural",
            decision=LucidityDecision.PRESERVE_AMBIGUITY,
            render_mode="plural",
            preserved_alternatives=[
                {
                    "basin_id": "b_fin",
                    "narrative_hint": "financial bank",
                    "source_refs": [SourceRef(ref_type="basin", ref_id="b_fin")],
                },
                {
                    "basin_id": "b_river",
                    "narrative_hint": "river bank",
                    "source_refs": [SourceRef(ref_type="basin", ref_id="b_river")],
                },
            ],
        )
        policy = DecoderPolicy(
            mode=DecoderMode.EXPRESS_PLURAL.value,
            forbid_single_answer=True,
            output_channel="chat",
        )
    else:
        print(f"unknown decoder fixture: {args.fixture}", file=sys.stderr)
        return 2

    lucidity_out = LucidityOutput(
        decision=packet.decision,
        decoder_policy=policy,
        render_packet=packet,
    )
    decoder_input = DecoderInput(
        lucidity_output=lucidity_out,
        render_packet=packet,
        decoder_policy=policy,
        output_channel="chat",
    )
    out = run_decoder(decoder_input)
    write_decoder_audit(
        audit_base_dir=args.audit_dir,
        decoder_input=decoder_input,
        decoder_output=out,
        details={"fixture": args.fixture},
    )
    print(to_json(out))
    return 0 if not out.refused else 1


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


def _cmd_audit(args: argparse.Namespace) -> int:
    if args.audit_cmd == "list":
        from lucid.audit.layout import format_run_list, list_runs

        print(format_run_list(list_runs(args.kind, module=args.module, limit=args.limit)))
        return 0
    if args.audit_cmd == "checkpoints":
        from lucid.audit.layout import format_checkpoint_list, list_checkpoints

        print(format_checkpoint_list(list_checkpoints()))
        return 0
    if args.audit_cmd == "layout":
        from lucid.audit.layout import write_train_readmes

        write_train_readmes()
        print("wrote lucid/training/tree/README.txt")
        return 0
    print("unknown audit command", file=sys.stderr)
    return 2


def _cmd_gen(args: argparse.Namespace) -> int:
    from lucid.training.corpus.cli import main as gen_main

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


def _cmd_chat_start(args: argparse.Namespace) -> int:
    try:
        session_id = start_session(session_id=args.session_id or None, audit_dir=args.audit_dir)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(session_id)
    return 0


def _cmd_chat_send(args: argparse.Namespace) -> int:
    text = args.text if args.text is not None else sys.stdin.read().strip()
    try:
        result = run_chat_turn(
            text,
            session_id=args.session_id,
            audit_dir=args.audit_dir,
            perception_backend=args.perception,
            checkpoint=args.checkpoint,
            learn_to_dmf=args.learn_to_dmf,
            learning_rate=args.learning_rate,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(to_json(result))
    else:
        print(result.assistant_output)
        print(f"session_audit: {result.session_audit_path}", file=sys.stderr)
        print(f"run_audit: {result.run_audit_dir}", file=sys.stderr)
    return 0


def _cmd_chat_list(args: argparse.Namespace) -> int:
    try:
        for session_id in list_sessions(audit_dir=args.audit_dir):
            print(session_id)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _cmd_chat_memory(args: argparse.Namespace) -> int:
    try:
        memory = load_session_memory(args.audit_dir, args.session_id)
        if memory is None:
            memory = new_session_memory(args.session_id)
            if args.create:
                save_session_memory(args.audit_dir, memory)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(to_json(memory))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from lucid.training.cli import main as train_main

    return train_main(args.args)


def _cmd_checkpoint(args: argparse.Namespace) -> int:
    from lucid.training.checkpoint.cli import main as checkpoint_main

    return checkpoint_main(args.args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lucid")
    sub = parser.add_subparsers(dest="command", required=True)

    ask_parser = sub.add_parser(
        "ask",
        help="One sentence through the full pipeline (sentence, answer, compact audit)",
    )
    ask_parser.add_argument(
        "sentence",
        nargs="+",
        help='Words to run (quote spaces: lucid ask "go to the bank")',
    )
    ask_parser.add_argument(
        "--audit-dir",
        default=DEFAULT_AUDIT_RUNS,
        help="Pipeline audit base directory",
    )
    ask_parser.add_argument("--perception", default="", choices=["", "rule", "llm"])
    ask_parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint override: cp_001, training, loaded, or path (default: pinned loaded save)",
    )
    ask_parser.add_argument(
        "--cold",
        action="store_true",
        help="Run without any checkpoint even if a save point is loaded",
    )
    ask_parser.add_argument(
        "--out",
        default="",
        help="Also write the full report to this path (e.g. last-ask.txt)",
    )
    ask_parser.add_argument(
        "--latest",
        default="",
        help=f"Stable latest copy (default: {DEFAULT_ASK_LATEST} under train tree)",
    )
    ask_parser.add_argument(
        "--no-latest",
        action="store_true",
        help="Do not update the stable latest.txt copy",
    )
    ask_parser.set_defaults(func=_cmd_ask)

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
    run_parser.add_argument(
        "--audit-dir",
        default=DEFAULT_AUDIT_RUNS,
        help="Audit base directory",
    )
    run_parser.add_argument("--perception", default="", choices=["", "rule", "llm"])
    run_parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint override: cp_001, training, loaded, or path (default: pinned loaded save)",
    )
    run_parser.add_argument(
        "--cold",
        action="store_true",
        help="Run without any checkpoint even if a save point is loaded",
    )
    run_parser.set_defaults(func=_cmd_run)

    run_batch_parser = sub.add_parser("run-batch", help="Run many episodes from JSONL without crash")
    run_batch_parser.add_argument("episode", help="Path to Episode JSONL")
    run_batch_parser.add_argument("--audit-dir", default=DEFAULT_AUDIT_RUNS)
    run_batch_parser.add_argument("--perception", default="", choices=["", "rule", "llm"])
    run_batch_parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint override: cp_001, training, loaded, or path (default: pinned loaded save)",
    )
    run_batch_parser.add_argument(
        "--cold",
        action="store_true",
        help="Run without any checkpoint even if a save point is loaded",
    )
    run_batch_parser.add_argument("--limit", type=int, default=0)
    run_batch_parser.set_defaults(func=_cmd_run_batch)

    edit_parser = sub.add_parser("edit", help="Edit tracebank or basin_bank in a checkpoint")
    edit_parser.add_argument("kind", choices=["trace", "basin"])
    edit_parser.add_argument("record_id", nargs="?", default="")
    edit_parser.add_argument("--checkpoint", default=DEFAULT_TRAINING_CHECKPOINT)
    edit_parser.add_argument("--patch", default="{}", help="JSON object merged into the record")
    edit_parser.add_argument("--list", default="", choices=["", "traces", "basins"])
    edit_parser.add_argument("--audit-dir", default=DEFAULT_AUDIT_MEMORY)
    edit_parser.set_defaults(func=_cmd_edit)

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
    cue_parser.add_argument("--audit-dir", default=DEFAULT_AUDIT_CUE_ENCODER)
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

    interference_parser = sub.add_parser("interference", help="Run interference on a built-in fixture")
    interference_parser.add_argument("--fixture", default="bank", choices=["bank"])
    interference_parser.add_argument(
        "--store",
        default=str(DEFAULT_INTERFERENCE_STORE),
        help="Path to learned interference links JSON",
    )
    interference_parser.add_argument(
        "--use-store",
        action="store_true",
        help="Load learned links from --store before running",
    )
    interference_parser.add_argument(
        "--feedback",
        action="append",
        default=[],
        help="Lucidity feedback token passed through context-op first",
    )
    interference_parser.set_defaults(func=_cmd_interference)

    interference_learn_parser = sub.add_parser(
        "interference-learn",
        help="Learn scoped interference links from a built-in fixture",
    )
    interference_learn_parser.add_argument("--fixture", default="bank", choices=["bank"])
    interference_learn_parser.add_argument(
        "--outcome",
        default="success",
        choices=["success", "failure"],
        help="Validated outcome to learn from",
    )
    interference_learn_parser.add_argument(
        "--failure-type",
        default="interference_or_basin",
        help="Failure label used when --outcome failure",
    )
    interference_learn_parser.add_argument(
        "--store",
        default=str(DEFAULT_INTERFERENCE_STORE),
        help="Path to learned interference links JSON",
    )
    interference_learn_parser.add_argument(
        "--audit-dir",
        default=DEFAULT_INTERFERENCE_LEARNING_AUDIT,
        help="Folder for human and machine readable learning audit logs",
    )
    interference_learn_parser.add_argument(
        "--feedback",
        action="append",
        default=[],
        help="Lucidity feedback token passed through context-op first",
    )
    interference_learn_parser.set_defaults(func=_cmd_interference_learn)

    basins_parser = sub.add_parser("basins", help="Run basins on a built-in fixture")
    basins_parser.add_argument("--fixture", default="bank", choices=["bank"])
    basins_parser.add_argument("--checkpoint", default="", help="Checkpoint with basin_bank.json")
    basins_parser.add_argument("--max-active", type=int, default=16)
    basins_parser.add_argument("--min-energy", type=float, default=0.15)
    basins_parser.add_argument("--audit-dir", default=smoke_audit_dir("basins"))
    basins_parser.set_defaults(func=_cmd_basins)

    lucidity_parser = sub.add_parser("lucidity", help="Run lucidity gate on a built-in fixture")
    lucidity_parser.add_argument("--fixture", default="bank", choices=["bank", "grid"])
    lucidity_parser.add_argument(
        "--pass-kind",
        default="pre_check",
        choices=["pre_check", "final_check", "recheck"],
    )
    lucidity_parser.add_argument("--checkpoint", default="", help="Checkpoint for basin bank")
    lucidity_parser.add_argument("--max-rollouts", type=int, default=1)
    lucidity_parser.add_argument("--audit-dir", default=DEFAULT_AUDIT_LUCIDITY)
    lucidity_parser.set_defaults(func=_cmd_lucidity)

    decoder_parser = sub.add_parser("decoder", help="Render a lucidity script into chat text")
    decoder_parser.add_argument("--fixture", default="bank", choices=["bank", "plural"])
    decoder_parser.add_argument("--max-sentences", type=int, default=3)
    decoder_parser.add_argument("--audit-dir", default=smoke_audit_dir("decoder"))
    decoder_parser.set_defaults(func=_cmd_decoder)

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
    bind_parser.add_argument("--audit-dir", default=DEFAULT_AUDIT_BINDING)
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
    dmf_parser.add_argument("--audit-dir", default=DEFAULT_AUDIT_DMF)
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

    checkpoint_parser = sub.add_parser(
        "checkpoint",
        help="Load/save inference save points (separate from training workspace)",
    )
    checkpoint_parser.add_argument("args", nargs=argparse.REMAINDER)
    checkpoint_parser.set_defaults(func=_cmd_checkpoint)

    audit_parser = sub.add_parser("audit", help="List audits and checkpoints under lucid/training/tree/")
    audit_sub = audit_parser.add_subparsers(dest="audit_cmd", required=True)

    audit_list = audit_sub.add_parser("list", help="List recent audit runs")
    audit_list.add_argument(
        "--kind",
        choices=["smoke", "training", "pipeline", "scaling"],
        default="smoke",
    )
    audit_list.add_argument("--module", default="")
    audit_list.add_argument("--limit", type=int, default=20)
    audit_list.set_defaults(func=_cmd_audit, audit_cmd="list")

    audit_ckpt = audit_sub.add_parser("checkpoints", help="List checkpoint directories")
    audit_ckpt.set_defaults(func=_cmd_audit, audit_cmd="checkpoints")

    audit_layout = audit_sub.add_parser("layout", help="Write lucid/training/tree/README.txt")
    audit_layout.set_defaults(func=_cmd_audit, audit_cmd="layout")

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
    scaling_export.add_argument(
        "--out",
        default="",
        help="Filename under train/audit/scaling/exports/",
    )
    scaling_export.set_defaults(func=_cmd_scaling_export)

    scaling_path = scaling_sub.add_parser("path", help="Print points.jsonl path")
    scaling_path.set_defaults(func=_cmd_scaling_path)

    chat_parser = sub.add_parser("chat", help="Run audited session chat turns")
    chat_sub = chat_parser.add_subparsers(dest="chat_cmd", required=True)

    chat_start = chat_sub.add_parser("start", help="Create or reuse a chat session")
    chat_start.add_argument("--session-id", default="", help="Use a stable caller-provided session id")
    chat_start.add_argument("--audit-dir", default="audit/chat", help="Chat audit base directory")
    chat_start.set_defaults(func=_cmd_chat_start)

    chat_send = chat_sub.add_parser("send", help="Send one message into a chat session")
    chat_send.add_argument("text", nargs="?", help="Message text, or stdin when omitted")
    chat_send.add_argument("--session-id", required=True, help="Session id from lucid chat start")
    chat_send.add_argument("--audit-dir", default="audit/chat", help="Chat audit base directory")
    chat_send.add_argument("--perception", default="", choices=["", "rule", "llm"])
    chat_send.add_argument("--checkpoint", default="", help="Checkpoint for runtime stores")
    chat_send.add_argument(
        "--learn-to-dmf",
        action="store_true",
        help="Persist this turn's cues into the checkpoint DMF tracebank",
    )
    chat_send.add_argument("--learning-rate", type=float, default=0.2)
    chat_send.add_argument("--json", action="store_true", help="Print machine-readable turn result")
    chat_send.set_defaults(func=_cmd_chat_send)

    chat_list = chat_sub.add_parser("list", help="List chat sessions under the audit directory")
    chat_list.add_argument("--audit-dir", default="audit/chat", help="Chat audit base directory")
    chat_list.set_defaults(func=_cmd_chat_list)

    chat_memory = chat_sub.add_parser("memory", help="Print one session's chat memory audit")
    chat_memory.add_argument("--session-id", required=True, help="Session id from lucid chat start")
    chat_memory.add_argument("--audit-dir", default="audit/chat", help="Chat audit base directory")
    chat_memory.add_argument("--create", action="store_true", help="Create an empty memory file if missing")
    chat_memory.set_defaults(func=_cmd_chat_memory)
    return parser


_KNOWN_COMMANDS = frozenset(
    {
        "ask",
        "perceive",
        "run",
        "run-batch",
        "edit",
        "cue-encoder",
        "context-op",
        "interference",
        "interference-learn",
        "basins",
        "lucidity",
        "decoder",
        "projector",
        "bind",
        "dmf",
        "governor",
        "quantization",
        "train",
        "checkpoint",
        "audit",
        "inspect",
        "gen",
        "scaling",
        "chat",
    }
)


def _normalize_argv(argv: list[str]) -> list[str]:
    """``lucid "your sentence"`` → ``lucid ask "your sentence"``."""
    if not argv:
        return argv
    if argv[0] in _KNOWN_COMMANDS:
        return argv
    if argv[0].startswith("-"):
        return ["ask", *argv]
    return ["ask", *argv]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    raw = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(_normalize_argv(raw))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
