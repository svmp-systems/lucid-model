"""Source ingest learning: facet caps, contradiction branches, consolidation, audit metrics."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lucid.training.source_context import (
    GERUND_TARGET_RE,
    MECHANISM_VERB_SURFACES,
    SOURCE_ENTITY_BY_ARTICLE,
    VENDOR_ARTIFACT_RE,
    VENDOR_REDIRECT_TARGETS,
)

FACET_BY_RELATION = {
    "type_of": "definition",
    "property": "definition",
    "related_to": "definition",
    "uses": "mechanism",
    "enables": "mechanism",
    "capability": "mechanism",
    "measurement": "mechanism",
    "contrast": "contrast",
    "challenge": "challenge",
}

CONFLICT_RELATION_TYPES = frozenset({"type_of"})

NEGATION_MARKERS = re.compile(
    r"\b(?:not|no|never|none|without|cannot|can't|couldn't|won't|doesn't|don't|"
    r"impossible|unlikely|merely|only a|nothing more than)\b",
    re.I,
)
BENIGN_NEGATION_RE = re.compile(
    r"\bnot(?:\s+(?:much|many|only|just|necessarily|always|yet|all|simply|exactly|entirely|quite|directly|strictly|limited to)\b|-much\b)",
    re.I,
)
NEGATIVE_CLAIM_PHRASES = (
    "no practical",
    "not useful",
    "theoretical curiosity",
    "too noisy",
    "error-prone",
    "error prone",
    "not scalable",
    "cannot be used",
    "unable to",
    "impractical",
    "rudimentary",
)
POSITIVE_CLAIM_PHRASES = (
    "practical",
    "useful",
    "industrial",
    "real-world",
    "real world",
    "scalable",
    "reliable",
    "powerful",
    "solve complex",
    "solve problems",
)
CONTRAST_PHRASES = (
    "instead of",
    "rather than",
    "unlike",
    "opposite of",
    " as opposed to ",
)

DEFAULT_MAX_RELATIONS_PER_FACET = 8
DEFAULT_MAX_RELATIONS_PER_CONCEPT = 24
DEFAULT_MAX_CANDIDATE_TERMS = 120


@dataclass(slots=True)
class SentenceAudit:
    total_raw: int = 0
    eligible: int = 0
    extracted: int = 0
    skipped_no_subject: int = 0
    skipped_no_relation: int = 0
    skipped_self_target: int = 0
    skip_reasons: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_raw": self.total_raw,
            "eligible": self.eligible,
            "extracted": self.extracted,
            "skipped_no_subject": self.skipped_no_subject,
            "skipped_no_relation": self.skipped_no_relation,
            "skipped_self_target": self.skipped_self_target,
            "skip_reasons": dict(self.skip_reasons),
            "coverage_ratio": round(self.extracted / max(1, self.eligible), 4),
        }


@dataclass(slots=True)
class IngestLearningReport:
    sentence_audit: SentenceAudit
    concepts_before_split: int = 0
    concepts_after_split: int = 0
    contradiction_splits: int = 0
    contradiction_events: list[dict[str, Any]] = field(default_factory=list)
    warm_concepts: int = 0
    probation_concepts: int = 0
    quarantine_concepts: int = 0
    traces_before_consolidation: int = 0
    traces_after_consolidation: int = 0
    traces_deduplicated: int = 0
    article_sentence_counts: dict[str, int] = field(default_factory=dict)
    crosstalk_pass: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence_audit": self.sentence_audit.to_dict(),
            "concepts_before_split": self.concepts_before_split,
            "concepts_after_split": self.concepts_after_split,
            "contradiction_splits": self.contradiction_splits,
            "contradiction_events": self.contradiction_events,
            "warm_concepts": self.warm_concepts,
            "probation_concepts": self.probation_concepts,
            "quarantine_concepts": self.quarantine_concepts,
            "traces_before_consolidation": self.traces_before_consolidation,
            "traces_after_consolidation": self.traces_after_consolidation,
            "traces_deduplicated": self.traces_deduplicated,
            "article_sentence_counts": self.article_sentence_counts,
            "crosstalk_pass": self.crosstalk_pass,
        }


def normalize_target_key(target: object) -> str:
    text = str(target or "").strip().lower()
    return " ".join(text.split())


def _target_token_overlap(left: str, right: str) -> float:
    left_tokens = set(normalize_target_key(left).split())
    right_tokens = set(normalize_target_key(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    shared = len(left_tokens & right_tokens)
    return shared / min(len(left_tokens), len(right_tokens))


def _target_has_negation(text: str) -> bool:
    scrubbed = BENIGN_NEGATION_RE.sub(" ", text)
    if NEGATION_MARKERS.search(scrubbed):
        return True
    return any(phrase in text for phrase in NEGATIVE_CLAIM_PHRASES)


def _polarity_clash(left: str, right: str) -> bool:
    left_positive = any(phrase in left for phrase in POSITIVE_CLAIM_PHRASES)
    right_positive = any(phrase in right for phrase in POSITIVE_CLAIM_PHRASES)
    left_negative = _target_has_negation(left)
    right_negative = _target_has_negation(right)
    return (left_positive and right_negative) or (right_positive and left_negative)


def _explicit_contradiction(left: str, right: str) -> bool:
    key_left = normalize_target_key(left)
    key_right = normalize_target_key(right)
    if _target_has_negation(key_left) or _target_has_negation(key_right):
        return True
    if _polarity_clash(key_left, key_right):
        return True
    return any(phrase in key_left or phrase in key_right for phrase in CONTRAST_PHRASES)


def relation_targets_conflict(relation: str, left: str, right: str) -> bool:
    if relation not in CONFLICT_RELATION_TYPES:
        return False
    key_left = normalize_target_key(left)
    key_right = normalize_target_key(right)
    if not key_left or not key_right or key_left == key_right:
        return False
    if key_left in key_right or key_right in key_left:
        return False
    if _target_token_overlap(key_left, key_right) >= 0.65:
        return False
    return _explicit_contradiction(key_left, key_right)


def relations_can_coexist(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_rel = str(left.get("relation") or "")
    right_rel = str(right.get("relation") or "")
    if left_rel != right_rel:
        return True
    return not relation_targets_conflict(left_rel, str(left.get("target") or ""), str(right.get("target") or ""))


def _cluster_same_relation(relations: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not relations:
        return []
    size = len(relations)
    parent = list(range(size))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    relation_type = str(relations[0].get("relation") or "")
    for left in range(size):
        for right in range(left + 1, size):
            if not relation_targets_conflict(
                relation_type,
                str(relations[left].get("target") or ""),
                str(relations[right].get("target") or ""),
            ):
                union(left, right)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, relation in enumerate(relations):
        grouped[find(index)].append(relation)
    return list(grouped.values())


def _shared_source_refs(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_refs = {str(ref) for ref in left.get("source_refs", []) if str(ref)}
    right_refs = {str(ref) for ref in right.get("source_refs", []) if str(ref)}
    return bool(left_refs & right_refs)


def cap_relations_by_facet(
    relations: list[dict[str, Any]],
    *,
    max_per_facet: int = DEFAULT_MAX_RELATIONS_PER_FACET,
    max_total: int = DEFAULT_MAX_RELATIONS_PER_CONCEPT,
) -> list[dict[str, Any]]:
    by_facet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relation in relations:
        facet = FACET_BY_RELATION.get(str(relation.get("relation") or ""), "definition")
        by_facet[facet].append(relation)

    capped: list[dict[str, Any]] = []
    for facet in sorted(by_facet):
        rows = sorted(
            by_facet[facet],
            key=lambda item: (
                -float(item.get("confidence", 0.0) or 0.0),
                str(item.get("relation") or ""),
                str(item.get("target") or ""),
            ),
        )
        capped.extend(rows[:max_per_facet])
    capped.sort(
        key=lambda item: (
            FACET_BY_RELATION.get(str(item.get("relation") or ""), "definition"),
            -float(item.get("confidence", 0.0) or 0.0),
            str(item.get("target") or ""),
        )
    )
    return capped[:max_total]


def split_concept_for_conflicts(
    concept: dict[str, Any],
    *,
    branch_hash: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    concept_id = str(concept.get("concept_id") or "")
    relations = [dict(relation) for relation in concept.get("relations", []) if isinstance(relation, dict)]
    if len(relations) <= 1:
        return [concept], []

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relation in relations:
        by_type[str(relation.get("relation") or "")].append(relation)

    conflict_clusters: list[list[dict[str, Any]]] = []
    for relation_type in sorted(by_type):
        if relation_type not in CONFLICT_RELATION_TYPES:
            continue
        clusters = _cluster_same_relation(by_type[relation_type])
        if len(clusters) > 1:
            conflict_clusters.extend(clusters)

    if not conflict_clusters:
        return [concept], []

    non_conflict = [
        relation for relation in relations if str(relation.get("relation") or "") not in CONFLICT_RELATION_TYPES
    ]

    events: list[dict[str, Any]] = []
    split_concepts: list[dict[str, Any]] = []
    for index, cluster in enumerate(conflict_clusters):
        branch_relations = list(cluster)
        for relation in non_conflict:
            if any(_shared_source_refs(relation, seed) for seed in cluster):
                branch_relations.append(relation)
        if index == 0:
            for relation in non_conflict:
                if relation not in branch_relations:
                    branch_relations.append(relation)

        branch_id = concept_id if index == 0 else f"{concept_id}__reading_{branch_hash(cluster[0])}"
        record = dict(concept)
        record["concept_id"] = branch_id
        record["relations"] = branch_relations
        record["source_refs"] = sorted(
            {
                str(source_ref)
                for relation in branch_relations
                for source_ref in relation.get("source_refs", [])
                if str(source_ref)
            }
        )
        if index > 0:
            record["parent_concept_id"] = concept_id
            record["branch_reason"] = "contradiction_split"
            events.append(
                {
                    "base_concept_id": concept_id,
                    "branch_concept_id": branch_id,
                    "relation": str(cluster[0].get("relation") or ""),
                    "target": str(cluster[0].get("target") or ""),
                    "source_refs": list(cluster[0].get("source_refs") or []),
                }
            )
        extraction = dict(record.get("extraction") or {})
        extraction["contradiction_branch"] = index > 0
        record["extraction"] = extraction
        split_concepts.append(record)
    return split_concepts, events


def apply_contradiction_branches(
    concepts: list[dict[str, Any]],
    *,
    branch_hash: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expanded: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for concept in concepts:
        split_rows, split_events = split_concept_for_conflicts(concept, branch_hash=branch_hash)
        expanded.extend(split_rows)
        events.extend(split_events)
    return expanded, events


def consolidate_trace_records(traces: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Merge duplicate claim traces that share family, relation alias, and target."""

    term_traces: list[dict[str, Any]] = []
    claim_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    deduped = 0

    for trace in traces:
        trace_id = str(trace.get("trace_id") or "")
        if trace_id.startswith("t_term_") or trace_id.startswith("t_basic_"):
            term_traces.append(trace)
            continue
        if not trace_id.startswith("t_claim_"):
            term_traces.append(trace)
            continue

        family = str(trace.get("trace_family") or "")
        alias = str(trace.get("alias") or "")
        relation_part = alias[len(family) + 1 :] if alias.startswith(f"{family}_") else ""
        affinities = trace.get("cue_affinities") or {}
        target_key = ""
        for key, weight in sorted(affinities.items(), key=lambda item: -float(item[1])):
            if key in {family, relation_part} or key in CONFLICT_RELATION_TYPES:
                continue
            if float(weight) >= 0.5:
                target_key = normalize_target_key(key)
                break
        dedupe_key = (family, relation_part, target_key)

        existing = claim_index.get(dedupe_key)
        if existing is None:
            claim_index[dedupe_key] = dict(trace)
            continue

        deduped += 1
        existing["source_refs"] = sorted(
            set(list(existing.get("source_refs") or []) + list(trace.get("source_refs") or []))
        )
        existing["activation_count"] = int(existing.get("activation_count", 0) or 0) + int(
            trace.get("activation_count", 0) or 0
        )
        existing["trust_score"] = max(
            float(existing.get("trust_score", 0.0) or 0.0),
            float(trace.get("trust_score", 0.0) or 0.0),
        )
        existing["last_update_summary"] = "ingest_consolidation_merge"

    consolidated = term_traces + list(claim_index.values())
    consolidated.sort(key=lambda row: str(row.get("trace_id") or ""))
    return consolidated, deduped


def count_concept_heat_tiers(
    concepts: list[dict[str, Any]],
    metadata_objects: dict[str, Any],
) -> tuple[int, int, int]:
    warm = probation = quarantine = 0
    for concept in concepts:
        concept_id = str(concept.get("concept_id") or "")
        tier = str(metadata_objects.get(f"concept:{concept_id}", {}).get("heat_tier") or "quarantine")
        if tier == "warm":
            warm += 1
        elif tier == "probation":
            probation += 1
        else:
            quarantine += 1
    return warm, probation, quarantine


def evaluate_crosstalk(
    concepts: list[dict[str, Any]],
    *,
    base_concept_id: str,
) -> bool:
    """Pass when incompatible readings split into separate branched concept ids."""

    base = normalize_target_key(base_concept_id)
    branches = [
        concept
        for concept in concepts
        if normalize_target_key(str(concept.get("concept_id") or "")).startswith(base)
    ]
    if len(branches) < 2:
        return False

    reading_ids = {
        str(concept.get("concept_id") or "")
        for concept in branches
        if str(concept.get("branch_reason") or "") == "contradiction_split"
        or "__reading_" in str(concept.get("concept_id") or "")
    }
    if len(reading_ids) < 1:
        return False

    type_targets: list[str] = []
    for concept in branches:
        for relation in concept.get("relations", []):
            if str(relation.get("relation") or "") == "type_of":
                type_targets.append(normalize_target_key(relation.get("target")))
    unique_targets = {target for target in type_targets if target}
    if len(unique_targets) < 2:
        return False

    for left in unique_targets:
        for right in unique_targets:
            if left >= right:
                continue
            if relation_targets_conflict("type_of", left, right):
                return True
    return False


def write_ingest_audit_report(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def format_ingest_audit_text(report: IngestLearningReport) -> str:
    audit = report.sentence_audit
    lines = [
        "Source ingest learning report",
        "=" * 28,
        f"sentences eligible: {audit.eligible}",
        f"sentences extracted: {audit.extracted}",
        f"coverage ratio: {audit.to_dict()['coverage_ratio']:.3f}",
        f"concepts before split: {report.concepts_before_split}",
        f"concepts after split: {report.concepts_after_split}",
        f"contradiction splits: {report.contradiction_splits}",
        f"warm / probation / quarantine: {report.warm_concepts} / {report.probation_concepts} / {report.quarantine_concepts}",
        f"traces deduplicated: {report.traces_deduplicated}",
        f"crosstalk pass: {report.crosstalk_pass}",
    ]
    if report.contradiction_events:
        lines.append("")
        lines.append("contradiction events:")
        for event in report.contradiction_events[:12]:
            lines.append(
                f"  {event.get('base_concept_id')} -> {event.get('branch_concept_id')} "
                f"({event.get('relation')}: {event.get('target')})"
            )
    return "\n".join(lines)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_relation_target(relation: str, target: str) -> tuple[str, str]:
    """Remap invalid type_of gerund complements to capability."""

    cleaned = str(target or "").strip()
    if relation == "type_of" and GERUND_TARGET_RE.match(cleaned):
        return "capability", cleaned
    return relation, cleaned


def is_vendor_artifact_concept(concept_id: str) -> bool:
    return bool(VENDOR_ARTIFACT_RE.match(str(concept_id or "").strip()))


def build_mechanism_relation_aliases() -> list[dict[str, Any]]:
    aliases: list[dict[str, Any]] = []
    for surface in sorted(MECHANISM_VERB_SURFACES):
        aliases.append(
            {
                "alias_id": f"alias_mechanism_{surface.replace(' ', '_')}",
                "surface_pattern": surface,
                "relation_candidates": ["uses"],
                "confidence": 0.78,
                "source": "scale_ingest_mechanism",
            }
        )
    for article_id, entity in SOURCE_ENTITY_BY_ARTICLE.items():
        key = article_id.split("_")[0]
        aliases.append(
            {
                "alias_id": f"alias_source_{key}",
                "surface_pattern": entity,
                "relation_candidates": ["source_entity", article_id],
                "confidence": 0.82,
                "source": "scale_ingest_source_entity",
            }
        )
        aliases.append(
            {
                "alias_id": f"alias_vendor_{key}_quantum_concept",
                "surface_pattern": f"{key} quantum",
                "relation_candidates": ["concept", "quantum_computer"],
                "confidence": 0.8,
                "source": "scale_ingest_source_entity",
            }
        )
        if key == "google":
            aliases.append(
                {
                    "alias_id": "alias_google_quantum_ai",
                    "surface_pattern": "google quantum ai",
                    "relation_candidates": ["concept", "quantum_computer"],
                    "confidence": 0.84,
                    "source": "scale_ingest_source_entity",
                }
            )
    return aliases


def consolidate_vendor_artifact_concepts(concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop vendor n-gram concepts and attach their claims to domain concepts."""

    by_id = {str(concept["concept_id"]): dict(concept) for concept in concepts}
    redirects: list[tuple[str, dict[str, Any]]] = []

    for concept_id, concept in list(by_id.items()):
        if not is_vendor_artifact_concept(concept_id):
            continue
        for relation in concept.get("relations") or []:
            if not isinstance(relation, dict):
                continue
            source_refs = [str(ref) for ref in relation.get("source_refs") or [] if str(ref)]
            article_id = source_refs[0] if source_refs else ""
            targets = VENDOR_REDIRECT_TARGETS.get(article_id, ["quantum_computer", "quantum_computing"])
            record = dict(relation)
            entity = SOURCE_ENTITY_BY_ARTICLE.get(article_id, "")
            if entity:
                record["source_entity"] = entity
            for target_id in targets:
                if target_id in by_id and target_id != concept_id:
                    redirects.append((target_id, record))
                    break
        del by_id[concept_id]

    for target_id, relation in redirects:
        concept = by_id[target_id]
        relations = list(concept.get("relations") or [])
        relations.append(relation)
        concept["relations"] = cap_relations_by_facet(
            merge_relation_list(relations),
            max_per_facet=DEFAULT_MAX_RELATIONS_PER_FACET,
            max_total=DEFAULT_MAX_RELATIONS_PER_CONCEPT,
        )
        concept["source_refs"] = sorted(
            set(list(concept.get("source_refs") or []) + list(relation.get("source_refs") or []))
        )

    cleaned = list(by_id.values())
    cleaned.sort(key=lambda row: str(row.get("concept_id") or ""))
    return cleaned


def merge_relation_list(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for relation in relations:
        key = (str(relation.get("relation") or ""), normalize_target_key(relation.get("target")))
        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(relation)
            continue
        existing["confidence"] = max(
            float(existing.get("confidence", 0.0) or 0.0),
            float(relation.get("confidence", 0.0) or 0.0),
        )
        existing["source_refs"] = sorted(
            set(list(existing.get("source_refs") or []) + list(relation.get("source_refs") or []))
        )
        if relation.get("source_entity") and not existing.get("source_entity"):
            existing["source_entity"] = relation.get("source_entity")
    return list(merged.values())
