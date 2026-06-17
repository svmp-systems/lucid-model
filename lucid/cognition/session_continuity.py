"""Session carryover for context-op scopes and binding concept continuity."""

from __future__ import annotations

from typing import Any

from lucid.ir.context_op import ContextFrame
from lucid.ir.lucidity import LucidityInput, LucidityOutput, PreservedHypothesis
from lucid.ir.pipeline import PipelineRun
from lucid.ir.serde import to_dict
from lucid.training.source_context import parse_concept_query, parse_concept_query_with_context


def context_frames_from_carryover(carryover: dict[str, Any] | None) -> list[ContextFrame]:
    if not isinstance(carryover, dict):
        return []
    frames: list[ContextFrame] = []
    for item in carryover.get("prior_context_frames") or []:
        if not isinstance(item, dict):
            continue
        frame_id = str(item.get("context_frame_id") or "").strip()
        if not frame_id:
            continue
        frames.append(
            ContextFrame(
                context_frame_id=frame_id,
                member_frame_ids=[str(mid) for mid in item.get("member_frame_ids") or [] if str(mid)],
                scope_notes=str(item.get("scope_notes") or ""),
                heat_policy=str(item.get("heat_policy") or "active"),
            )
        )
    return frames


def concept_topics_from_carryover(carryover: dict[str, Any] | None) -> list[str]:
    if not isinstance(carryover, dict):
        return []
    topics: list[str] = []
    seen: set[str] = set()
    for topic in carryover.get("concept_topics") or []:
        key = str(topic or "").strip()
        if key and key not in seen:
            seen.add(key)
            topics.append(key)
    return topics


def carryover_trace_ids(carryover: dict[str, Any] | None) -> list[str]:
    if not isinstance(carryover, dict):
        return []
    return [str(trace_id) for trace_id in carryover.get("carryover_trace_ids") or [] if str(trace_id).strip()]


def unresolved_from_carryover(carryover: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(carryover, dict):
        return []
    return [dict(item) for item in carryover.get("unresolved_items") or [] if isinstance(item, dict)]


def pipeline_carryover_from_session_context(session_context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(session_context, dict):
        return {}
    carryover = session_context.get("pipeline_carryover")
    if isinstance(carryover, dict):
        return dict(carryover)
    return {}


def extract_pipeline_carryover(run: PipelineRun) -> dict[str, Any]:
    concept_topics: list[str] = []
    seen_topics: set[str] = set()
    carryover_traces: list[str] = []
    seen_traces: set[str] = set()
    prior_frames: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    raw_text = ""
    session_context: dict[str, Any] = {}
    if run.evidence_graph is not None:
        extra = run.evidence_graph.provenance.extra or {}
        raw = extra.get("raw_text")
        if isinstance(raw, str):
            raw_text = raw.strip()
        session_ctx = extra.get("session_context")
        if isinstance(session_ctx, dict):
            session_context = session_ctx

    parsed = parse_concept_query_with_context(raw_text, session_context) if raw_text else None
    if parsed:
        concept_id = parsed[1]
        if concept_id not in seen_topics:
            seen_topics.add(concept_id)
            concept_topics.append(concept_id)
        trace_id = f"t_term_{concept_id}"
        if trace_id not in seen_traces:
            seen_traces.add(trace_id)
            carryover_traces.append(trace_id)

    if run.binding_output is not None:
        for frame in run.binding_output.candidate_frames:
            frame_id = str(frame.frame_id or "")
            concept_id = ""
            if frame_id.startswith("concept_query_"):
                concept_id = frame_id[len("concept_query_") :].split("__", 1)[0]
            elif frame_id.startswith("concept_mechanism_"):
                concept_id = frame_id[len("concept_mechanism_") :].split("__", 1)[0]
            if concept_id and concept_id not in seen_topics:
                seen_topics.add(concept_id)
                concept_topics.append(concept_id)
            for trace_id in frame.supporting_trace_ids:
                tid = str(trace_id)
                if tid and tid not in seen_traces:
                    seen_traces.add(tid)
                    carryover_traces.append(tid)

    if run.context_op_output is not None:
        for context_frame in run.context_op_output.context_frames[-6:]:
            prior_frames.append(to_dict(context_frame))

    if run.lucidity_output is not None:
        unresolved.extend(_unresolved_from_lucidity(run.lucidity_output, concept_topics=concept_topics))

    prior_carryover = pipeline_carryover_from_session_context(session_context)
    for topic in concept_topics_from_carryover(prior_carryover):
        if topic not in seen_topics:
            seen_topics.add(topic)
            concept_topics.append(topic)

    return {
        "concept_topics": concept_topics[-8:],
        "prior_context_frames": prior_frames[-8:],
        "carryover_trace_ids": carryover_traces[-16:],
        "unresolved_items": unresolved[-12:],
    }


def _unresolved_from_lucidity(
    output: LucidityOutput,
    *,
    concept_topics: list[str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    decision = output.decision.value if output.decision else ""
    if decision == "commit" and concept_topics:
        items.append(
            {
                "kind": "committed_concept",
                "concept_id": concept_topics[-1],
                "decision": decision,
            }
        )
    for hypothesis in output.preserved_hypotheses or []:
        items.append(_hypothesis_to_unresolved(hypothesis, decision=decision))
    return items


def _hypothesis_to_unresolved(hypothesis: PreservedHypothesis, *, decision: str) -> dict[str, Any]:
    return {
        "kind": "preserved_hypothesis",
        "hypothesis_id": hypothesis.hypothesis_id,
        "frame_id": hypothesis.frame_id,
        "basin_id": hypothesis.basin_id,
        "narrative_hint": hypothesis.narrative_hint,
        "confidence": hypothesis.confidence,
        "decision": decision,
    }


def merge_session_context(session_context: dict[str, Any], carryover: dict[str, Any]) -> dict[str, Any]:
    merged = dict(session_context)
    merged["pipeline_carryover"] = carryover
    merged["concept_topics"] = list(carryover.get("concept_topics") or [])
    merged["unresolved_items"] = list(carryover.get("unresolved_items") or [])
    merged["carryover_trace_ids"] = list(carryover.get("carryover_trace_ids") or [])
    return merged
