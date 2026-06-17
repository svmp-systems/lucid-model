"""Shared evidence-quality helpers for lucidity checks and commit gating."""

from __future__ import annotations

from lucid.cognition.input.cue.encoder import normalize_cue_key
from lucid.cognition.output.lucidity.config import normalize_task_intent
from lucid.ir.basins import CandidateBasinState
from lucid.ir.lucidity import LucidityInput
from lucid.training.source_context import (
    GERUND_TARGET_RE,
    MECHANISM_RELATIONS,
    MIN_DEFINITION_COMMIT_SCORE,
    MIN_DEFINITION_RENDER_SCORE,
    concept_definition_primary_basin,
    is_cross_sense_target,
    is_mechanism_like_target,
    is_renderable_definition_target,
    is_speech_basin,
    parse_concept_query_with_context,
    preferred_definition_concept,
    score_definition_target_for_concept,
)


def raw_text_from_input(inp: LucidityInput) -> str:
    graph = inp.perceptual_evidence_graph
    if graph is None:
        return ""
    raw = graph.provenance.extra.get("raw_text")
    return raw.strip() if isinstance(raw, str) else ""


def session_context_from_input(inp: LucidityInput) -> dict[str, object]:
    extra = inp.perceptual_evidence_graph.provenance.extra or {}
    session_context = extra.get("session_context")
    if isinstance(session_context, dict):
        return session_context
    return {}


def knowledge_query_from_input(inp: LucidityInput) -> tuple[str, str, str] | None:
    raw = raw_text_from_input(inp)
    if raw:
        return parse_concept_query_with_context(raw, session_context_from_input(inp))
    return None


def is_knowledge_query(inp: LucidityInput) -> bool:
    return knowledge_query_from_input(inp) is not None


def top_basin_state(inp: LucidityInput) -> CandidateBasinState | None:
    summary = inp.basin_output.competition_summary
    if not summary.top_basin_id:
        return None
    return next(
        (
            state
            for state in inp.basin_output.candidate_basin_states
            if state.basin_id == summary.top_basin_id
        ),
        None,
    )


def _relation_rows(state: CandidateBasinState) -> list[dict]:
    payload = state.quantized_payload if isinstance(state.quantized_payload, dict) else {}
    relations = payload.get("relations")
    if not isinstance(relations, list):
        return []
    return [row for row in relations if isinstance(row, dict)]


def _score_basin_relation(row: dict, *, concept_id: str = "") -> float:
    rel_name = str(row.get("relation") or "").strip().lower()
    target = str(row.get("target") or "").strip()
    if not target:
        return -1.0
    if rel_name in {"type_of", "is_a", "kind_of"}:
        if GERUND_TARGET_RE.match(target):
            return -1.0
        if not is_renderable_definition_target(target, relation=rel_name, concept_id=concept_id):
            return -1.0
        return score_definition_target_for_concept(target, concept_id)
    if rel_name in MECHANISM_RELATIONS:
        if len(target.split()) < 4:
            return -1.0
        return min(len(target) / 120.0, 0.5) + 0.35
    if rel_name in {"capability", "uses", "property", "has_property"}:
        if len(target.split()) < 3:
            return -1.0
        return 0.4
    return -1.0


def basin_has_renderable_relations(
    state: CandidateBasinState | None,
    *,
    concept_id: str = "",
    relation_names: set[str] | frozenset[str] | None = None,
) -> bool:
    if state is None or is_speech_basin(state.basin_id, state):
        return False
    source_refs = [str(ref).strip() for ref in state.source_refs if str(ref).strip()]
    for row in _relation_rows(state):
        rel_name = str(row.get("relation") or "").strip().lower()
        target = str(row.get("target") or "").strip()
        if relation_names is not None and rel_name not in relation_names:
            continue
        row_refs = [str(ref).strip() for ref in row.get("source_refs") or source_refs if str(ref).strip()]
        if not target or not row_refs:
            continue
        score = _score_basin_relation(row, concept_id=concept_id)
        if score >= MIN_DEFINITION_RENDER_SCORE:
            return True
        if rel_name in MECHANISM_RELATIONS and score >= 0.35:
            return True
    return False


def source_backed_renderable_basin(inp: LucidityInput) -> bool:
    target = concept_target_basin(inp) if is_knowledge_query(inp) else top_basin_state(inp)
    if target is None:
        return False
    if not any(str(ref).strip() for ref in target.source_refs):
        return False
    parsed = knowledge_query_from_input(inp)
    concept_id = parsed[1] if parsed else ""
    relation_names: frozenset[str] | None = None
    if parsed and parsed[2] == "definition_query":
        concept_id = preferred_definition_concept(concept_id)
        relation_names = frozenset({"type_of", "is_a", "kind_of"})
    return basin_has_renderable_relations(target, concept_id=concept_id, relation_names=relation_names)


def concept_target_basin(inp: LucidityInput) -> CandidateBasinState | None:
    parsed = knowledge_query_from_input(inp)
    if not parsed:
        return None
    _, concept_id, frame_type = parsed
    basin_id = concept_definition_primary_basin(
        inp.basin_output.candidate_basin_states,
        concept_id,
        frame_type=frame_type,
    )
    if not basin_id:
        return None
    return next(
        (state for state in inp.basin_output.candidate_basin_states if state.basin_id == basin_id),
        None,
    )


def binding_has_resolved_claims(inp: LucidityInput) -> bool:
    for frame in inp.binding_output.candidate_frames:
        if frame.unresolved_slot_names:
            continue
        if frame.role_assignments and frame.confidence >= 0.45:
            return True
        if frame.supporting_trace_ids and frame.confidence >= 0.45:
            return True
    return False


def binding_graph_render_units(inp: LucidityInput) -> list[dict[str, object]]:
    parsed = knowledge_query_from_input(inp)
    if not parsed:
        return []
    _, concept_id, frame_type = parsed
    render_concept = preferred_definition_concept(concept_id)
    concept_key = normalize_cue_key(render_concept)
    units: list[dict[str, object]] = []
    for frame in inp.binding_output.candidate_frames:
        if frame.frame_type not in {"definition_query", "mechanism_query", "concept_query"}:
            continue
        for graph in frame.local_graphs:
            if graph.family != "concept" or not graph.edges:
                continue
            for edge in graph.edges:
                if edge.edge_kind != "relation" or not edge.label:
                    continue
                subject = ""
                for node in graph.nodes:
                    if node.node_id == edge.source_id:
                        subject = node.label
                target = ""
                for node in graph.nodes:
                    if node.node_id == edge.target_id:
                        target = node.label
                if frame_type == "definition_query":
                    if edge.label not in {"type_of", "is_a", "kind_of"}:
                        continue
                    if not is_renderable_definition_target(
                        target,
                        relation=edge.label,
                        concept_id=render_concept,
                    ):
                        continue
                    if is_cross_sense_target(target, render_concept):
                        continue
                    if score_definition_target_for_concept(target, render_concept) < MIN_DEFINITION_COMMIT_SCORE:
                        continue
                elif edge.label in MECHANISM_RELATIONS:
                    if len(target.split()) < 4:
                        continue
                    if is_cross_sense_target(target, render_concept):
                        continue
                elif edge.label in {"property", "has_property"} and is_mechanism_like_target(target):
                    if is_cross_sense_target(target, render_concept):
                        continue
                else:
                    continue
                if concept_key and normalize_cue_key(subject) not in {concept_key, concept_id}:
                    continue
                if edge.provenance_refs or frame.supporting_trace_ids:
                    units.append(
                        {
                            "subject": subject,
                            "relation": edge.label,
                            "target": target,
                            "confidence": edge.confidence,
                        }
                    )
    return units


def renderability_score(inp: LucidityInput) -> tuple[float, dict]:
    task = normalize_task_intent(inp.task_intent)
    if task == "solve_grid":
        return 1.0, {"status": "not_applicable", "task_intent": task}

    parsed = knowledge_query_from_input(inp)
    if parsed:
        _, concept_id, frame_type = parsed
        render_concept = preferred_definition_concept(concept_id)
        graph_units = binding_graph_render_units(inp)
        if graph_units:
            return 1.0, {
                "source": "binding_local_graph",
                "concept_id": render_concept,
                "unit_count": len(graph_units),
            }
        state = concept_target_basin(inp)
        if state is None:
            return 0.0, {"reason": "no_concept_basin", "concept_id": concept_id}
        relation_names = (
            frozenset({"type_of", "is_a", "kind_of"})
            if frame_type == "definition_query"
            else None
        )
        if basin_has_renderable_relations(
            state,
            concept_id=render_concept,
            relation_names=relation_names,
        ):
            return 1.0, {"source": "concept_basin", "basin_id": state.basin_id, "concept_id": render_concept}
        return 0.0, {"reason": "unrenderable_concept_memory", "basin_id": state.basin_id, "concept_id": render_concept}

    if source_backed_renderable_basin(inp):
        top = top_basin_state(inp)
        return 1.0, {"source": "source_backed_basin", "basin_id": top.basin_id if top else ""}

    if binding_has_resolved_claims(inp):
        return 0.75, {"source": "binding_claims"}

    top = top_basin_state(inp)
    if top is not None and is_speech_basin(top.basin_id, top):
        return 0.0, {"reason": "speech_basin_only", "basin_id": top.basin_id}

    if task in {"chat", "answer"}:
        return 0.0, {"reason": "no_renderable_evidence", "top_basin_id": top.basin_id if top else ""}
    return 1.0, {"status": "not_applicable", "task_intent": task}


def knowledge_query_competition_margin(inp: LucidityInput) -> tuple[float, dict[str, object]] | None:
    """Concept-scoped basin margin for definition/mechanism queries (excludes speech basins)."""
    parsed = knowledge_query_from_input(inp)
    if not parsed:
        return None
    _, concept_id, frame_type = parsed
    preferred = preferred_definition_concept(concept_id)
    lookup_ids = {normalize_cue_key(preferred)}
    if frame_type == "mechanism_query" and normalize_cue_key(concept_id) != normalize_cue_key(preferred):
        lookup_ids.add(normalize_cue_key(concept_id))
    facet = "mechanism" if frame_type == "mechanism_query" else "definition"

    from lucid.training.source_context import parse_concept_basin_id

    best_by_basin_id: dict[str, CandidateBasinState] = {}
    for state in inp.basin_output.candidate_basin_states:
        if is_speech_basin(state.basin_id, state):
            continue
        parsed_basin = parse_concept_basin_id(state.basin_id)
        basin_concept = ""
        basin_facet = ""
        if parsed_basin is not None:
            basin_concept, basin_facet = parsed_basin
        else:
            payload = state.quantized_payload if isinstance(state.quantized_payload, dict) else {}
            basin_concept = str(payload.get("concept_id") or "")
            basin_facet = str(payload.get("facet") or "definition")
        if basin_facet != facet:
            continue
        if normalize_cue_key(basin_concept) not in lookup_ids:
            continue
        current = best_by_basin_id.get(state.basin_id)
        if current is None or state.energy > current.energy:
            best_by_basin_id[state.basin_id] = state

    scoped = list(best_by_basin_id.values())

    if not scoped:
        if binding_graph_render_units(inp):
            return 1.0, {"source": "binding_graph", "concept_id": preferred}
        target = concept_target_basin(inp)
        if target is not None and basin_has_renderable_relations(target, concept_id=preferred):
            return 1.0, {"source": "concept_target_basin", "basin_id": target.basin_id}
        return None

    ordered = sorted(scoped, key=lambda item: item.energy, reverse=True)
    top = ordered[0]
    second = ordered[1] if len(ordered) > 1 else None
    margin = top.energy - second.energy if second else top.energy
    details: dict[str, object] = {
        "source": "concept_scoped",
        "concept_id": preferred,
        "facet": facet,
        "top_basin_id": top.basin_id,
        "second_basin_id": second.basin_id if second else "",
        "scoped_basin_count": len(scoped),
    }
    if binding_graph_render_units(inp) and margin < 0.08:
        margin = max(margin, 0.12)
        details["binding_graph_boost"] = True
    return round(max(0.0, margin), 4), details


def perception_surfaces(inp: LucidityInput) -> set[str]:
    surfaces: set[str] = set()
    graph = inp.perceptual_evidence_graph
    if graph is None:
        return surfaces
    for unit in graph.candidate_units:
        key = normalize_cue_key(unit.surface)
        if key:
            surfaces.add(key)
    return surfaces
