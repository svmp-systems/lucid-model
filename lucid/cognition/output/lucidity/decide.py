"""Map check results to a lucidity decision and decoder policy."""

from __future__ import annotations

from lucid.cognition.output.lucidity.commit import (
    build_committed_state,
    preserved_hypotheses_from_basins,
    projector_targets,
)
from lucid.cognition.output.lucidity.chat_speech import try_social_speech_decision
from lucid.cognition.output.lucidity.config import (
    LucidityConfig,
    normalize_pass_kind,
    normalize_task_intent,
)
from lucid.ir.common import DecoderMode, LucidityDecision, SearchTarget
from lucid.ir.lucidity import (
    ConfidenceSummary,
    DecoderPolicy,
    LucidityCheckResults,
    LucidityInput,
    LucidityOutput,
    SearchDirectives,
)


def _all_named_checks_pass(checks: LucidityCheckResults, *, skip_projection: bool) -> bool:
    items = [
        checks.margin_check,
        checks.coverage_check,
        checks.coherence_check,
        checks.binding_stability_check,
        checks.scope_check,
        checks.contradiction_check,
        checks.maturity_check,
        checks.risk_check,
    ]
    if not skip_projection and checks.projection_fit_check is not None:
        if checks.projection_fit_check.details.get("status") != "not_applicable":
            items.append(checks.projection_fit_check)
    return all(item is not None and item.passed for item in items)


def _has_supported_local_graph(inp: LucidityInput) -> bool:
    for frame in inp.binding_output.candidate_frames:
        for graph in frame.local_graphs:
            if graph.family != "concept":
                continue
            if any(
                edge.edge_kind == "relation"
                and edge.confidence >= 0.5
                and bool(edge.provenance_refs)
                and not edge.edge_id.startswith("alias_")
                for edge in graph.edges
            ):
                return True
    return False


def _top_source_backed_relation_basin_ready(
    inp: LucidityInput,
    checks: LucidityCheckResults,
) -> bool:
    task = normalize_task_intent(inp.task_intent)
    if task not in {"answer", "chat"}:
        return False

    summary = inp.basin_output.competition_summary
    if not summary.top_basin_id:
        return False

    required_checks = [
        checks.margin_check,
        checks.coverage_check,
        checks.scope_check,
        checks.contradiction_check,
        checks.maturity_check,
        checks.risk_check,
    ]
    if any(item is None or not item.passed for item in required_checks):
        return False

    top = next(
        (
            state
            for state in inp.basin_output.candidate_basin_states
            if state.basin_id == summary.top_basin_id
        ),
        None,
    )
    if top is None or top.energy <= 0.0:
        return False
    if top.coherence_score < 0.7:
        return False

    payload = top.quantized_payload if isinstance(top.quantized_payload, dict) else {}
    relations = payload.get("relations")
    if not isinstance(relations, list) or not relations:
        return False

    for relation in relations:
        if not isinstance(relation, dict):
            continue
        rel_name = str(relation.get("relation") or "").strip()
        target = str(relation.get("target") or "").strip()
        source_refs = relation.get("source_refs") or top.source_refs
        has_source_ref = any(str(ref).strip() for ref in source_refs)
        confidence = float(relation.get("confidence", top.energy) or 0.0)
        if rel_name and target and has_source_ref and confidence >= 0.5:
            return True
    return False


def decoder_policy_for(
    decision: LucidityDecision,
    *,
    task_intent: str,
    checks: LucidityCheckResults,
) -> DecoderPolicy:
    task = normalize_task_intent(task_intent)
    if decision == LucidityDecision.COMMIT:
        return DecoderPolicy(
            mode=DecoderMode.EXPRESS_COMMITTED.value,
            output_format="grid" if task == "solve_grid" else "text",
            forbid_invented_facts=True,
            require_source_refs_per_sentence=True,
            max_sentences=4 if task != "solve_grid" else 0,
        )
    if decision == LucidityDecision.PRESERVE_AMBIGUITY:
        return DecoderPolicy(
            mode=DecoderMode.EXPRESS_PLURAL.value,
            forbid_single_answer=True,
            forbid_invented_facts=True,
            show_alternatives=True,
            show_scope=True,
            output_format="text",
        )
    if decision == LucidityDecision.REQUEST_PROJECTION:
        return DecoderPolicy(mode=DecoderMode.HOLD.value, output_format="grid" if task == "solve_grid" else "text")
    if decision == LucidityDecision.SEARCH_WIDER:
        return DecoderPolicy(mode=DecoderMode.HOLD.value)
    if decision == LucidityDecision.RECHECK_BINDING:
        return DecoderPolicy(mode=DecoderMode.HOLD.value)
    return DecoderPolicy(
        mode=DecoderMode.EXPRESS_UNCERTAINTY.value,
        forbid_invented_facts=True,
        show_confidence=True,
    )


def decide(
    inp: LucidityInput,
    checks: LucidityCheckResults,
    confidence: ConfidenceSummary,
    config: LucidityConfig,
) -> LucidityOutput:
    task = normalize_task_intent(inp.task_intent)
    pass_kind = normalize_pass_kind(inp.pass_kind)
    notes: list[str] = [f"lucidity:task={task}", f"lucidity:pass={pass_kind}"]

    social = try_social_speech_decision(inp)
    if social is not None:
        social.audit_notes = [*notes, *social.audit_notes]
        return social

    if inp.iteration_count >= config.max_iterations:
        notes.append("lucidity:iteration_cap")
        return LucidityOutput(
            decision=LucidityDecision.PRESERVE_AMBIGUITY,
            decoder_policy=decoder_policy_for(LucidityDecision.PRESERVE_AMBIGUITY, task_intent=inp.task_intent, checks=checks),
            preserved_hypotheses=preserved_hypotheses_from_basins(inp),
            audit_notes=notes,
        )

    if task == "solve_grid" and pass_kind == "pre_check" and config.require_projection_on_grid_pre_check:
        if LucidityDecision.REQUEST_PROJECTION in inp.prior_decisions:
            notes.append("lucidity:skip_repeat_projection")
        else:
            notes.append("lucidity:grid_pre_check_projection")
            return LucidityOutput(
                decision=LucidityDecision.REQUEST_PROJECTION,
                decoder_policy=decoder_policy_for(
                    LucidityDecision.REQUEST_PROJECTION,
                    task_intent=inp.task_intent,
                    checks=checks,
                ),
                search_directives=SearchDirectives(
                    projector_targets=projector_targets(inp),
                    max_rollouts=inp.compute_policy.max_projector_rollouts,
                    rollout_mode="none",
                    search_target=SearchTarget.BASINS,
                ),
                audit_notes=notes,
            )

    if task == "solve_grid" and pass_kind == "final_check":
        projection = inp.projection_output
        recommendation = ""
        if projection is not None:
            recommendation = projection.recommendation_to_lucidity or projection.recommendation
        projection_ok = checks.projection_fit_check is not None and checks.projection_fit_check.passed
        if projection is not None and recommendation == "suggest_commit" and projection_ok:
            committed = build_committed_state(inp, projection=projection)
            notes.append("lucidity:grid_commit_after_projection")
            return LucidityOutput(
                decision=LucidityDecision.COMMIT,
                decoder_policy=decoder_policy_for(LucidityDecision.COMMIT, task_intent=inp.task_intent, checks=checks),
                committed_state=committed,
                audit_notes=notes,
            )
        notes.append("lucidity:grid_projection_failed")
        return LucidityOutput(
            decision=LucidityDecision.SEARCH_WIDER,
            decoder_policy=decoder_policy_for(LucidityDecision.SEARCH_WIDER, task_intent=inp.task_intent, checks=checks),
            search_directives=SearchDirectives(
                search_target=SearchTarget.ALL,
                cue_budget_multiplier=1.5,
                allow_new_frames=True,
                extra={"allow_provisional_basins": True},
            ),
            audit_notes=notes,
        )

    margin = checks.margin_check
    coverage = checks.coverage_check
    binding = checks.binding_stability_check
    scope = checks.scope_check
    contradiction = checks.contradiction_check
    risk = checks.risk_check

    commit_ready = _all_named_checks_pass(checks, skip_projection=inp.projection_output is None)

    if risk is not None and not risk.passed and risk.details.get("needs_projection"):
        if inp.projection_output is not None:
            notes.append("lucidity:search_wider_projection_failed")
            return LucidityOutput(
                decision=LucidityDecision.SEARCH_WIDER,
                decoder_policy=decoder_policy_for(
                    LucidityDecision.SEARCH_WIDER,
                    task_intent=inp.task_intent,
                    checks=checks,
                ),
                search_directives=SearchDirectives(
                    search_target=SearchTarget.ALL,
                    cue_budget_multiplier=1.5,
                    allow_new_frames=True,
                    projector_targets=projector_targets(inp),
                    max_rollouts=inp.compute_policy.max_projector_rollouts,
                    extra={"reason": "projection_required_but_failed"},
                ),
                audit_notes=notes,
            )
        notes.append("lucidity:request_projection_high_risk")
        return LucidityOutput(
            decision=LucidityDecision.REQUEST_PROJECTION,
            decoder_policy=decoder_policy_for(
                LucidityDecision.REQUEST_PROJECTION,
                task_intent=inp.task_intent,
                checks=checks,
            ),
            search_directives=SearchDirectives(
                search_target=SearchTarget.BASINS,
                projector_targets=projector_targets(inp),
                max_rollouts=inp.compute_policy.max_projector_rollouts,
                rollout_mode="single_step",
                rollout_depth=1,
                extra={"reason": "high_risk_requires_projection"},
            ),
            audit_notes=notes,
        )

    if contradiction is not None and not contradiction.passed:
        if scope is not None and not scope.passed:
            notes.append("lucidity:recheck_binding_scope")
            return LucidityOutput(
                decision=LucidityDecision.RECHECK_BINDING,
                decoder_policy=decoder_policy_for(
                    LucidityDecision.RECHECK_BINDING,
                    task_intent=inp.task_intent,
                    checks=checks,
                ),
                search_directives=SearchDirectives(search_target=SearchTarget.BINDING),
                audit_notes=notes,
            )
        notes.append("lucidity:preserve_ambiguity_contradiction")
        return LucidityOutput(
            decision=LucidityDecision.PRESERVE_AMBIGUITY,
            decoder_policy=decoder_policy_for(
                LucidityDecision.PRESERVE_AMBIGUITY,
                task_intent=inp.task_intent,
                checks=checks,
            ),
            preserved_hypotheses=preserved_hypotheses_from_basins(inp),
            audit_notes=notes,
        )

    if _top_source_backed_relation_basin_ready(inp, checks):
        notes.append("lucidity:commit_source_backed_basin")
        return LucidityOutput(
            decision=LucidityDecision.COMMIT,
            decoder_policy=decoder_policy_for(
                LucidityDecision.COMMIT,
                task_intent=inp.task_intent,
                checks=checks,
            ),
            committed_state=build_committed_state(inp),
            audit_notes=notes,
        )

    if binding is not None and not binding.passed and coverage is not None and coverage.passed:
        rebind_ids = [frame.frame_id for frame in inp.binding_output.candidate_frames if frame.unresolved_slot_names]
        if not rebind_ids:
            rebind_ids = [frame.frame_id for frame in inp.binding_output.candidate_frames[:2]]
        notes.append("lucidity:recheck_binding")
        return LucidityOutput(
            decision=LucidityDecision.RECHECK_BINDING,
            decoder_policy=decoder_policy_for(LucidityDecision.RECHECK_BINDING, task_intent=inp.task_intent, checks=checks),
            search_directives=SearchDirectives(
                search_target=SearchTarget.BINDING,
                rebind_frame_ids=rebind_ids,
            ),
            audit_notes=notes,
        )

    if coverage is not None and not coverage.passed:
        notes.append("lucidity:search_wider_coverage")
        return LucidityOutput(
            decision=LucidityDecision.SEARCH_WIDER,
            decoder_policy=decoder_policy_for(LucidityDecision.SEARCH_WIDER, task_intent=inp.task_intent, checks=checks),
            search_directives=SearchDirectives(
                search_target=SearchTarget.ALL,
                cue_budget_multiplier=1.5,
                allow_new_frames=True,
                extra={"allow_provisional_basins": True},
            ),
            audit_notes=notes,
        )

    if (
        _has_supported_local_graph(inp)
        and coverage is not None
        and coverage.passed
        and checks.coherence_check is not None
        and checks.coherence_check.passed
        and binding is not None
        and binding.passed
        and scope is not None
        and scope.passed
        and risk is not None
        and risk.passed
    ):
        notes.append("lucidity:commit_local_graph")
        return LucidityOutput(
            decision=LucidityDecision.COMMIT,
            decoder_policy=decoder_policy_for(
                LucidityDecision.COMMIT,
                task_intent=inp.task_intent,
                checks=checks,
            ),
            committed_state=build_committed_state(inp),
            audit_notes=notes,
        )

    if margin is not None and not margin.passed:
        if task in {"act", "plan"} or inp.stakes_policy in {"high", "strict"}:
            notes.append("lucidity:request_projection_low_margin")
            return LucidityOutput(
                decision=LucidityDecision.REQUEST_PROJECTION,
                decoder_policy=decoder_policy_for(
                    LucidityDecision.REQUEST_PROJECTION,
                    task_intent=inp.task_intent,
                    checks=checks,
                ),
                search_directives=SearchDirectives(
                    search_target=SearchTarget.BASINS,
                    projector_targets=projector_targets(inp),
                    max_rollouts=inp.compute_policy.max_projector_rollouts,
                    rollout_mode="single_step",
                    rollout_depth=1,
                    extra={"reason": "low_margin_high_stakes"},
                ),
                audit_notes=notes,
            )
        if coverage and coverage.passed and checks.coherence_check and checks.coherence_check.passed:
            notes.append("lucidity:preserve_ambiguity_margin")
            return LucidityOutput(
                decision=LucidityDecision.PRESERVE_AMBIGUITY,
                decoder_policy=decoder_policy_for(
                    LucidityDecision.PRESERVE_AMBIGUITY,
                    task_intent=inp.task_intent,
                    checks=checks,
                ),
                preserved_hypotheses=preserved_hypotheses_from_basins(inp),
                audit_notes=notes,
            )

    if commit_ready:
        notes.append("lucidity:commit")
        return LucidityOutput(
            decision=LucidityDecision.COMMIT,
            decoder_policy=decoder_policy_for(LucidityDecision.COMMIT, task_intent=inp.task_intent, checks=checks),
            committed_state=build_committed_state(inp),
            audit_notes=notes,
        )

    notes.append("lucidity:preserve_ambiguity_fallback")
    return LucidityOutput(
        decision=LucidityDecision.PRESERVE_AMBIGUITY,
        decoder_policy=decoder_policy_for(LucidityDecision.PRESERVE_AMBIGUITY, task_intent=inp.task_intent, checks=checks),
        preserved_hypotheses=preserved_hypotheses_from_basins(inp),
        audit_notes=notes,
    )
