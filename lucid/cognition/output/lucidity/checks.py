"""Nine lucidity checks — scored, auditable pass/fail per input."""

from __future__ import annotations

from lucid.cognition.output.lucidity.config import (
    LucidityConfig,
    normalize_pass_kind,
    normalize_risk_level,
    normalize_task_intent,
)
from lucid.ir.basins import BasinOutput, CandidateBasinState
from lucid.ir.binding import BindingOutput, CandidateFrame
from lucid.ir.context_op import ContextOpOutput
from lucid.ir.interference import InterferenceOutput
from lucid.ir.lucidity import (
    CheckResult,
    ConfidenceSummary,
    LucidityCheckResults,
    LucidityInput,
)
from lucid.ir.perception import PerceptualEvidenceGraph
from lucid.ir.projector import ProjectorOutput


def _check(
    *,
    passed: bool,
    score: float,
    threshold: float,
    details: dict | None = None,
) -> CheckResult:
    return CheckResult(
        passed=passed,
        score=score,
        threshold=threshold,
        details=details or {},
    )


def _evidence_refs_for_frame(frame: CandidateFrame) -> set[str]:
    refs: set[str] = set(frame.member_evidence_refs)
    for values in frame.slot_evidence_refs.values():
        refs.update(values)
    return refs


def _coverage_score(
    graph: PerceptualEvidenceGraph,
    binding: BindingOutput,
    dmf_coverage: float,
    *,
    salience_cutoff: float,
) -> tuple[float, dict]:
    salient = [unit for unit in graph.candidate_units if unit.salience >= salience_cutoff]
    if not salient:
        base = max(dmf_coverage, 0.75)
        return base, {"salient_units": 0, "covered_units": 0, "source": "dmf_default"}

    covered = 0
    all_refs: set[str] = set()
    for frame in binding.candidate_frames:
        all_refs.update(_evidence_refs_for_frame(frame))

    for unit in salient:
        if unit.unit_id in all_refs:
            covered += 1

    score = covered / len(salient)
    if dmf_coverage > 0:
        score = max(score, min(1.0, dmf_coverage))
    return score, {
        "salient_units": len(salient),
        "covered_units": covered,
        "dmf_coverage": dmf_coverage,
    }


def _coherence_score(binding: BindingOutput, basin_states: list) -> tuple[float, dict]:
    if not binding.candidate_frames:
        top_coherence = basin_states[0].coherence_score if basin_states else 0.0
        return top_coherence, {"frame_count": 0}

    penalties = 0
    frame_scores: list[float] = []
    for frame in binding.candidate_frames:
        frame_score = frame.confidence
        if frame.conflicting_trace_ids and not frame.unresolved_slot_names:
            penalties += 1
            frame_score *= 0.5
        frame_scores.append(frame_score)

    basin_coherence = max((state.coherence_score for state in basin_states), default=0.0)
    if basin_coherence <= 0.0 and frame_scores:
        basin_coherence = sum(frame_scores) / len(frame_scores)
    raw = (sum(frame_scores) / len(frame_scores) + basin_coherence) / 2.0
    score = max(0.0, raw - 0.12 * penalties)
    return score, {
        "frame_count": len(binding.candidate_frames),
        "role_conflicts": penalties,
        "basin_coherence": basin_coherence,
    }


def _basin_by_id(basins: BasinOutput) -> dict[str, CandidateBasinState]:
    return {state.basin_id: state for state in basins.candidate_basin_states if state.basin_id}


def _maturity_trace_ids(basins: BasinOutput) -> set[str]:
    top_id = basins.competition_summary.top_basin_id
    top = next(
        (state for state in basins.candidate_basin_states if state.basin_id == top_id),
        None,
    )
    trace_ids: set[str] = set()
    if top is not None:
        trace_ids.update(trace_id for trace_id in top.supporting_trace_ids if trace_id)
        for member_id in top.member_basin_ids:
            member = next(
                (
                    state
                    for state in basins.candidate_basin_states
                    if state.basin_id == member_id
                    and set(state.scope_frame_ids) == set(top.scope_frame_ids)
                ),
                None,
            )
            if member is not None:
                trace_ids.update(trace_id for trace_id in member.supporting_trace_ids if trace_id)
    if trace_ids:
        return trace_ids
    for state in basins.candidate_basin_states[:3]:
        trace_ids.update(trace_id for trace_id in state.supporting_trace_ids if trace_id)
    return trace_ids


_BASIN_FACET_SUFFIXES = {
    "capability",
    "challenge",
    "contrast",
    "definition",
    "mechanism",
    "measurement",
    "property",
    "speech",
}


def _basin_root(basin_id: str) -> str:
    text = str(basin_id or "").strip()
    if text.startswith("b_"):
        text = text[2:]
    parts = [part for part in text.split("_") if part]
    if len(parts) > 1 and parts[-1] in _BASIN_FACET_SUFFIXES:
        parts = parts[:-1]
    return "_".join(parts)


def _blocking_basin_conflict_count(basins: BasinOutput) -> tuple[int, int]:
    top_id = basins.competition_summary.top_basin_id
    blocking = 0
    ignored = 0
    for conflict in basins.unresolved_conflicts:
        ids = [basin_id for basin_id in conflict.basin_ids if basin_id]
        if conflict.conflict_type == "low_margin_competition":
            if top_id and top_id not in ids:
                ignored += 1
                continue
            roots = {_basin_root(basin_id) for basin_id in ids if _basin_root(basin_id)}
            if len(roots) == 1:
                ignored += 1
                continue
        blocking += 1
    return blocking, ignored


def _scope_score(
    context_op: ContextOpOutput,
    binding: BindingOutput,
    basins: BasinOutput,
    interference: InterferenceOutput,
) -> tuple[float, dict]:
    violations: list[str] = []
    gates = context_op.interference_gates
    if not gates:
        return 1.0, {"gate_count": 0}

    states = _basin_by_id(basins)
    blocked_by_scope = {
        gate.scope_frame_id: set(gate.blocked_trace_ids)
        for gate in gates
        if gate.scope_frame_id and gate.blocked_trace_ids
    }

    for gate in gates:
        blocked = set(gate.blocked_trace_ids)
        if not blocked:
            continue
        for ctx in context_op.context_frames:
            if gate.scope_frame_id and ctx.context_frame_id != gate.scope_frame_id:
                continue
            member_ids = set(ctx.member_frame_ids)
            for frame in binding.candidate_frames:
                if member_ids and frame.frame_id not in member_ids:
                    continue
                for slot, trace_id in frame.role_assignments.items():
                    if trace_id in blocked:
                        violations.append(f"{frame.frame_id}/{slot}:{trace_id}")
                for trace_id in frame.supporting_trace_ids:
                    if trace_id in blocked:
                        violations.append(f"{frame.frame_id}/support:{trace_id}")

        for state in basins.candidate_basin_states:
            if gate.scope_frame_id and state.scope_frame_ids and gate.scope_frame_id not in state.scope_frame_ids:
                continue
            for trace_id in state.supporting_trace_ids:
                if trace_id in blocked:
                    violations.append(f"{state.basin_id}/basin_support:{trace_id}")

        for edge in interference.trace_frame_edges:
            if edge.delta <= 0 or edge.trace_id not in blocked:
                continue
            frame_scope = next(
                (
                    ctx.context_frame_id
                    for ctx in context_op.context_frames
                    if edge.frame_id in ctx.member_frame_ids
                ),
                edge.frame_id,
            )
            if gate.scope_frame_id == frame_scope:
                violations.append(f"{edge.frame_id}/interference_support:{edge.trace_id}")

    for edge in interference.frame_basin_edges:
        if edge.delta <= 0:
            continue
        state = states.get(edge.basin_id)
        if state is None:
            continue
        frame_scope = next(
            (
                ctx.context_frame_id
                for ctx in context_op.context_frames
                if edge.frame_id in ctx.member_frame_ids
            ),
            edge.frame_id,
        )
        blocked = blocked_by_scope.get(frame_scope, set())
        if blocked.intersection(state.supporting_trace_ids):
            traces = ",".join(sorted(blocked.intersection(state.supporting_trace_ids)))
            violations.append(f"{edge.basin_id}/scoped_basin_edge:{traces}")

    if violations:
        return max(0.0, 1.0 - 0.25 * len(violations)), {"violations": violations, "gate_count": len(gates)}
    return 1.0, {"gate_count": len(gates), "violations": []}


def _projection_fit(projection: ProjectorOutput | None, *, threshold: float) -> CheckResult:
    if projection is None:
        return _check(passed=True, score=1.0, threshold=threshold, details={"status": "not_applicable"})

    best = None
    if projection.best_rollout_id:
        best = next((r for r in projection.rollouts if r.rollout_id == projection.best_rollout_id), None)
    if best is None and projection.rollouts:
        best = projection.rollouts[0]

    score = best.fit_scores.aggregate_fit if best is not None else 0.0
    recommendation = projection.recommendation_to_lucidity or projection.recommendation
    passed = score >= threshold and recommendation in {"suggest_commit", "commit", ""}
    return _check(
        passed=passed,
        score=score,
        threshold=threshold,
        details={
            "recommendation": recommendation,
            "best_rollout_id": projection.best_rollout_id,
            "rollout_count": len(projection.rollouts),
        },
    )


def run_checks(inp: LucidityInput, config: LucidityConfig) -> tuple[LucidityCheckResults, ConfidenceSummary]:
    task = normalize_task_intent(inp.task_intent)
    pass_kind = normalize_pass_kind(inp.pass_kind)
    risk_level = normalize_risk_level(inp.risk_level)

    summary = inp.basin_output.competition_summary
    margin_score = summary.top_margin
    margin_threshold = config.margin_threshold(inp.task_intent, pass_kind)
    margin = _check(
        passed=margin_score >= margin_threshold,
        score=margin_score,
        threshold=margin_threshold,
        details={
            "top_basin_id": summary.top_basin_id,
            "second_basin_id": summary.second_basin_id,
            "task_intent": task,
            "pass_kind": pass_kind,
        },
    )

    coverage_score, coverage_details = _coverage_score(
        inp.perceptual_evidence_graph,
        inp.binding_output,
        inp.dmf_output.coverage_score,
        salience_cutoff=config.salience_cutoff,
    )
    coverage = _check(
        passed=coverage_score >= config.coverage_threshold,
        score=coverage_score,
        threshold=config.coverage_threshold,
        details=coverage_details,
    )

    coherence_score, coherence_details = _coherence_score(
        inp.binding_output,
        inp.basin_output.candidate_basin_states,
    )
    coherence = _check(
        passed=coherence_score >= config.coherence_threshold,
        score=coherence_score,
        threshold=config.coherence_threshold,
        details=coherence_details,
    )

    binding_score = inp.binding_output.binding_stability_score
    if binding_score <= 0 and inp.basin_output.binding_stability_hint > 0:
        binding_score = inp.basin_output.binding_stability_hint
    binding = _check(
        passed=binding_score >= config.binding_stability_threshold,
        score=binding_score,
        threshold=config.binding_stability_threshold,
        details={
            "unresolved_frames": sum(
                1 for frame in inp.binding_output.candidate_frames if frame.unresolved_slot_names
            ),
        },
    )

    scope_score, scope_details = _scope_score(
        inp.context_op_output,
        inp.binding_output,
        inp.basin_output,
        inp.interference_output,
    )
    scope = _check(
        passed=scope_score >= 0.85,
        score=scope_score,
        threshold=0.85,
        details=scope_details,
    )

    projection = _projection_fit(inp.projection_output, threshold=config.projection_fit_threshold)

    hard_conflicts = [
        signal
        for signal in inp.dmf_output.conflict_signals
        if signal.severity >= config.contradiction_severity_threshold
    ]
    interference_reports = [
        report
        for report in getattr(inp.interference_output, "conflict_reports", [])
        if getattr(report, "severity", 0.0) >= config.contradiction_severity_threshold
    ]
    severe_basin_edges = [
        edge
        for edge in inp.interference_output.basin_basin_edges
        if edge.relation == "compete" and edge.delta <= -config.contradiction_severity_threshold
    ]
    basin_conflicts, ignored_basin_conflicts = _blocking_basin_conflict_count(inp.basin_output)
    contradiction_score = 1.0
    if hard_conflicts:
        contradiction_score -= min(0.6, 0.2 * len(hard_conflicts))
    if interference_reports:
        contradiction_score -= min(0.8, 0.35 * len(interference_reports))
    if severe_basin_edges:
        contradiction_score -= min(0.5, 0.2 * len(severe_basin_edges))
    if basin_conflicts:
        contradiction_score -= min(0.4, 0.15 * basin_conflicts)
    contradiction_score = max(0.0, contradiction_score)
    contradiction = _check(
        passed=contradiction_score >= 0.7,
        score=contradiction_score,
        threshold=0.7,
        details={
            "hard_conflict_count": len(hard_conflicts),
            "interference_conflict_count": len(interference_reports),
            "severe_basin_edge_count": len(severe_basin_edges),
            "basin_conflict_count": basin_conflicts,
            "ignored_basin_conflict_count": ignored_basin_conflicts,
            "interference_conflict_ids": [
                getattr(report, "report_id", "") for report in interference_reports
            ],
        },
    )

    active = inp.dmf_output.active_traces
    active_by_id = {trace.trace_id: trace for trace in active if trace.trace_id}
    relevant_ids = _maturity_trace_ids(inp.basin_output)
    relevant = [
        active_by_id[trace_id]
        for trace_id in relevant_ids
        if trace_id in active_by_id
    ]
    maturity_pool = relevant or active
    if not maturity_pool:
        maturity_score = 0.5
    else:
        hot = sum(1 for trace in maturity_pool if trace.heat_tier in {"hot", "warm", "stabilized"})
        maturity_score = hot / len(maturity_pool)
    maturity_pass = maturity_score >= config.maturity_hot_fraction or risk_level == "low"
    if risk_level == "high" and maturity_score < config.maturity_hot_fraction and pass_kind != "final_check":
        maturity_pass = False
    maturity = _check(
        passed=maturity_pass,
        score=maturity_score,
        threshold=config.maturity_hot_fraction,
        details={
            "active_trace_count": len(active),
            "relevant_trace_count": len(maturity_pool),
            "relevant_trace_ids": sorted(trace.trace_id for trace in maturity_pool if trace.trace_id),
            "risk_level": risk_level,
        },
    )

    needs_projection = task == "solve_grid" or risk_level == "high"
    if task == "solve_grid" and pass_kind == "pre_check":
        risk_pass = True
    elif task == "solve_grid" and pass_kind == "final_check":
        risk_pass = projection.passed
    elif needs_projection and inp.projection_output is None:
        risk_pass = False
    elif needs_projection and pass_kind == "final_check":
        risk_pass = projection.passed
    else:
        risk_pass = True

    risk = _check(
        passed=risk_pass,
        score=1.0 if risk_pass else 0.0,
        threshold=1.0,
        details={
            "needs_projection": needs_projection,
            "has_projection": inp.projection_output is not None,
            "stakes_policy": inp.stakes_policy,
        },
    )

    checks = LucidityCheckResults(
        margin_check=margin,
        coverage_check=coverage,
        coherence_check=coherence,
        binding_stability_check=binding,
        scope_check=scope,
        projection_fit_check=projection,
        contradiction_check=contradiction,
        maturity_check=maturity,
        risk_check=risk,
    )

    projection_fit = projection.score if inp.projection_output is not None else None
    overall = (margin.score + coverage.score + coherence.score + binding.score + scope.score) / 5.0
    confidence = ConfidenceSummary(
        overall_confidence=overall,
        margin=margin.score,
        coverage=coverage.score,
        coherence=coherence.score,
        projection_fit=projection_fit,
    )
    return checks, confidence
