"""Build LucidityRenderPacket — decoder script from lucidity output."""

from __future__ import annotations

from uuid import uuid4

from lucid.ir.common import DecoderMode, LucidityDecision
from lucid.ir.lucidity import (
    CommittedState,
    DecoderPolicy,
    ExplicitOmission,
    FaithfulnessContract,
    LucidityInput,
    LucidityOutput,
    LucidityRenderPacket,
    PreservedHypothesis,
    RenderConstraints,
    RenderUnit,
    SourceRef,
    StructuredClaim,
)


def _mode_to_render_mode(policy: DecoderPolicy, decision: LucidityDecision) -> str:
    raw = (policy.mode or "").strip().lower()
    if raw == DecoderMode.HOLD.value:
        return "hold"
    if raw == DecoderMode.EXPRESS_REFUSAL.value:
        return "refusal"
    if raw == DecoderMode.EXPRESS_PLURAL.value or policy.forbid_single_answer:
        return "plural"
    if raw == DecoderMode.EXPRESS_UNCERTAINTY.value:
        return "uncertainty"
    if decision == LucidityDecision.PRESERVE_AMBIGUITY:
        return "plural"
    if decision == LucidityDecision.COMMIT:
        return "committed"
    return "uncertainty"


def _grid_from_artifact(artifact: dict) -> list[list[int]] | None:
    grid = artifact.get("grid_output")
    if isinstance(grid, list) and grid and isinstance(grid[0], list):
        return [[int(cell) for cell in row] for row in grid]
    test_outputs = artifact.get("test_outputs")
    if isinstance(test_outputs, list) and test_outputs:
        first = test_outputs[0]
        if isinstance(first, list) and first and isinstance(first[0], list):
            return [[int(cell) for cell in row] for row in first]
        if isinstance(first, list):
            return [[int(cell) for cell in first]]
    return None


def _claim_unit_from_structured(claim: StructuredClaim, *, unit_id: str) -> RenderUnit:
    refs: list[SourceRef] = []
    if claim.subject_ref:
        refs.append(SourceRef(ref_type="trace", ref_id=claim.subject_ref, scope_frame_id=claim.scope_frame_id))
    if claim.predicate_ref:
        refs.append(SourceRef(ref_type="trace", ref_id=claim.predicate_ref, scope_frame_id=claim.scope_frame_id))
    return RenderUnit(
        unit_id=unit_id,
        unit_type="claim",
        scope_frame_id=claim.scope_frame_id,
        text_intent="answer",
        payload={
            "claim_type": claim.claim_type,
            "subject_ref": claim.subject_ref,
            "predicate_ref": claim.predicate_ref,
        },
        confidence=claim.confidence,
        required=True,
        source_refs=refs,
    )


def _units_from_committed(committed: CommittedState) -> list[RenderUnit]:
    if committed.render_units:
        return list(committed.render_units)

    if committed.claims:
        return [_claim_unit_from_structured(claim, unit_id=f"claim-{index}") for index, claim in enumerate(committed.claims)]

    units: list[RenderUnit] = []
    grid = _grid_from_artifact(committed.projection_artifact)
    if grid is not None:
        refs = [SourceRef(ref_type="projection", ref_id=aid, role="supports") for aid in committed.assembly_ids]
        units.append(
            RenderUnit(
                unit_id="artifact-grid",
                unit_type="artifact",
                text_intent="answer",
                payload={"grid_output": grid},
                required=True,
                source_refs=refs,
            )
        )
        return units

    if committed.primary_basin_id:
        units.append(
            RenderUnit(
                unit_id="claim-primary",
                unit_type="claim",
                text_intent="answer",
                payload={
                    "summary": f"committed reading ({committed.primary_basin_id})",
                    "basin_id": committed.primary_basin_id,
                },
                required=True,
                source_refs=[
                    SourceRef(ref_type="basin", ref_id=committed.primary_basin_id, role="supports"),
                ],
            )
        )

    for index, frame in enumerate(committed.frame_commits):
        units.append(
            RenderUnit(
                unit_id=f"frame-{index}",
                unit_type="frame_summary",
                scope_frame_id=frame.context_frame_id,
                text_intent="reason",
                payload={
                    "frame_type": frame.frame_type,
                    "basin_id": frame.basin_id,
                    "summary": frame.scope_notes or f"frame {frame.frame_type}",
                },
                required=index == 0,
                source_refs=[
                    SourceRef(
                        ref_type="frame",
                        ref_id=frame.context_frame_id,
                        scope_frame_id=frame.context_frame_id,
                    ),
                    SourceRef(
                        ref_type="basin",
                        ref_id=frame.basin_id,
                        scope_frame_id=frame.context_frame_id,
                    ),
                ],
            )
        )
    return units


def _alternatives_from_hypotheses(hypotheses: list[PreservedHypothesis]) -> list[dict]:
    rows: list[dict] = []
    for item in hypotheses:
        rows.append(
            {
                "hypothesis_id": item.hypothesis_id,
                "scope_frame_id": item.frame_id,
                "basin_id": item.basin_id,
                "narrative_hint": item.narrative_hint,
                "source_refs": [
                    SourceRef(ref_type="basin", ref_id=item.basin_id, scope_frame_id=item.frame_id),
                ],
            }
        )
    return rows


def build_render_packet(
    output: LucidityOutput,
    *,
    lucidity_input: LucidityInput | None = None,
) -> LucidityRenderPacket | None:
    policy = output.decoder_policy
    render_mode = _mode_to_render_mode(policy, output.decision)
    output_format = policy.output_format or "text"

    if render_mode == "hold":
        return LucidityRenderPacket(
            packet_id=str(uuid4()),
            decision=output.decision,
            render_mode="hold",
            output_format=output_format,
        )

    approved_units: list[RenderUnit] = []
    preserved = _alternatives_from_hypotheses(output.preserved_hypotheses)
    omissions: list[ExplicitOmission] = []

    committed = output.committed_state
    if committed is not None:
        approved_units = _units_from_committed(committed)
        if committed.unresolved:
            omissions.append(
                ExplicitOmission(
                    reason="unsupported",
                    forbidden_claim_refs=list(committed.unresolved),
                    user_visible=True,
                )
            )

    if lucidity_input is not None and not preserved and output.decision == LucidityDecision.PRESERVE_AMBIGUITY:
        top = lucidity_input.basin_output.competition_summary.top_basin_id
        margin = lucidity_input.basin_output.competition_summary.top_margin
        for state in lucidity_input.basin_output.candidate_basin_states[:3]:
            if not state.basin_id:
                continue
            preserved.append(
                {
                    "hypothesis_id": state.basin_id,
                    "scope_frame_id": state.supporting_frame_ids[0] if state.supporting_frame_ids else "",
                    "basin_id": state.basin_id,
                    "narrative_hint": state.basin_id,
                    "source_refs": [SourceRef(ref_type="basin", ref_id=state.basin_id)],
                }
            )
        margin_threshold = 0.08
        if output.check_results.margin_check is not None:
            margin_threshold = output.check_results.margin_check.threshold
        if top and margin < margin_threshold:
            omissions.append(
                ExplicitOmission(
                    reason="low_margin",
                    forbidden_claim_refs=[f"single_winner:{top}"],
                    user_visible=True,
                )
            )

    contradiction = output.check_results.contradiction_check
    if contradiction is not None and not contradiction.passed:
        omissions.append(
            ExplicitOmission(
                reason="unsupported",
                forbidden_claim_refs=[
                    ref for ref in contradiction.details.get("interference_conflict_ids", []) if ref
                ],
                user_visible=True,
            )
        )

    risk = output.check_results.risk_check
    if risk is not None and not risk.passed:
        omissions.append(
            ExplicitOmission(
                reason="high_risk",
                forbidden_claim_refs=[],
                user_visible=True,
            )
        )

    max_sentences = policy.max_sentences or 4
    if output_format == "grid":
        max_sentences = 0

    return LucidityRenderPacket(
        packet_id=str(uuid4()),
        decision=output.decision,
        render_mode=render_mode,
        output_format=output_format,
        approved_units=approved_units,
        preserved_alternatives=preserved,
        explicit_omissions=omissions,
        render_constraints=RenderConstraints(
            max_sentences=max_sentences,
            max_tokens=policy.max_tokens,
            tone="careful" if render_mode in {"plural", "uncertainty"} else "neutral",
        ),
        faithfulness_contract=FaithfulnessContract(
            forbid_new_entities=policy.forbid_invented_facts,
            require_source_refs_per_sentence=policy.require_source_refs_per_sentence
            or policy.require_cite_traces,
        ),
        provenance_chain=list(committed.provenance_chain) if committed else [],
    )


def attach_render_packet(
    output: LucidityOutput,
    *,
    lucidity_input: LucidityInput | None = None,
) -> LucidityOutput:
    output.render_packet = build_render_packet(output, lucidity_input=lucidity_input)
    return output
