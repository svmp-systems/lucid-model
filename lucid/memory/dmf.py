"""Dynamic memory field runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import log2
from pathlib import Path

from lucid.ir.common import HeatTier, MaturityState
from lucid.ir.cue import CueCloud
from lucid.audit.logger import content_hash
from lucid.ir.dmf import (
    ActiveTrace,
    ConflictSignal,
    DmfInput,
    DmfOutput,
    NoveltySignal,
    TraceCluster,
)
from lucid.cognition.reasoning.cue_routes import competing_cue_keys
from lucid.memory.cue_match import best_affinity_for_cue
from lucid.ir.serde import to_dict
from lucid.runtime.paths import DEFAULT_AUDIT_DMF, resolve_checkpoint, resolve_train_path
from lucid.training.checkpoint.slots import resolve_checkpoint_ref

# DMF emits activated traces (capped by compute max_active_traces); lucidity decides winners.
DMF_ACTIVATION_FLOOR_PERCENTILE = 0.05  # drop bottom 5% of scored activations (top ~95% band)
DMF_MIN_ACTIVATION_SCORE = 0.0


@dataclass(slots=True)
class DmfAuditEvent:
    event_type: str
    summary: str
    trace_index: int = -1
    trace_id_before: str = ""
    trace_id_after: str = ""
    cue_keys: list[str] = field(default_factory=list)
    details: dict[str, str | int | float | bool] = field(default_factory=dict)
    before_hash: str = ""
    after_hash: str = ""
    audit_path: str = ""


@dataclass(slots=True)
class DmfTraceRecord:
    """One runtime trace entry in DMF memory.

    `trace_id` intentionally defaults to empty string for early training.
    """

    trace_id: str = ""
    alias: str = ""
    cue_affinities: dict[str, float] = field(default_factory=dict)
    cluster_id: str = ""
    heat_tier: str = HeatTier.HOT.value
    maturity_state: str = MaturityState.PROVISIONAL.value
    activation_bias: float = 0.0
    coactivation_links: dict[int, float] = field(default_factory=dict)
    conflict_links: dict[int, float] = field(default_factory=dict)
    activation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    created_from_cues: list[str] = field(default_factory=list)
    created_from_examples: list[str] = field(default_factory=list)
    description: str = ""
    last_update_summary: str = ""


def _index_link_map(raw: object) -> dict[int, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[int, float] = {}
    for key, value in raw.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        out[idx] = float(value)
    return out


def trace_record_from_store(record: dict[str, object]) -> DmfTraceRecord:
    """Build a runtime trace record from a checkpoint ``tracebank`` store row."""

    cue_affinities = {
        str(key): float(value)
        for key, value in dict(record.get("cue_affinities") or {}).items()
        if str(key)
    }
    family = str(record.get("trace_family") or record.get("alias") or "").strip()
    if family and family not in cue_affinities:
        cue_affinities[family] = max(cue_affinities.get(family, 0.0), 0.5)

    created_from = record.get("created_from_cues") or record.get("created_from_episodes") or []
    if not isinstance(created_from, list):
        created_from = []
    examples = record.get("created_from_examples") or []
    if not isinstance(examples, list):
        examples = []

    return DmfTraceRecord(
        trace_id=str(record.get("trace_id") or ""),
        alias=str(record.get("alias") or family),
        cue_affinities=cue_affinities,
        cluster_id=str(record.get("cluster_id") or family),
        heat_tier=str(record.get("heat_tier") or HeatTier.HOT.value),
        maturity_state=str(record.get("maturity_state") or MaturityState.PROVISIONAL.value),
        activation_bias=float(record.get("activation_bias") or 0.0),
        coactivation_links=_index_link_map(record.get("coactivation_links")),
        conflict_links=_index_link_map(record.get("conflict_links")),
        activation_count=int(record.get("activation_count") or 0),
        success_count=int(record.get("success_count") or 0),
        failure_count=int(record.get("failure_count") or 0),
        created_from_cues=[str(item) for item in created_from if str(item)],
        created_from_examples=[str(item) for item in examples if str(item)],
        description=str(record.get("description") or ""),
        last_update_summary=str(record.get("last_update_summary") or ""),
    )


def tracebank_from_checkpoint(path: str | Path) -> list[DmfTraceRecord]:
    """Load tracebank rows from a Lucid checkpoint directory."""

    root = resolve_checkpoint(resolve_checkpoint_ref(path))
    store_path = root / "tracebank.json"
    if not store_path.exists():
        return []
    store = json.loads(store_path.read_text(encoding="utf-8"))
    records = store.get("records", [])
    if not isinstance(records, list):
        return []
    return [trace_record_from_store(item) for item in records if isinstance(item, dict)]


def load_dynamic_memory_field(
    checkpoint: str | Path | None = None,
    *,
    audit_base_dir: str | Path | None = None,
) -> DynamicMemoryField:
    """Construct DMF runtime state from an optional checkpoint tracebank."""

    tracebank: list[DmfTraceRecord] = []
    if checkpoint:
        root = resolve_checkpoint(resolve_checkpoint_ref(checkpoint))
        if root.exists():
            tracebank = tracebank_from_checkpoint(root)
    return DynamicMemoryField(tracebank=tracebank, audit_base_dir=audit_base_dir)


class DynamicMemoryField:
    """Lean DMF runtime.

    This stage returns an activation landscape and does not collapse to final meaning.
    """

    def __init__(
        self,
        tracebank: list[DmfTraceRecord] | None = None,
        *,
        audit_base_dir: str | Path | None = DEFAULT_AUDIT_DMF,
    ):
        self.tracebank: list[DmfTraceRecord] = tracebank or []
        self.audit_events: list[DmfAuditEvent] = []
        if audit_base_dir is None:
            self.audit_base_dir = None
        else:
            self.audit_base_dir = resolve_train_path(audit_base_dir, mkdir=True)
        self._cue_index: dict[str, set[int]] = {}
        self._trace_id_index: dict[str, int] = {}
        self._cluster_index: dict[str, set[int]] = {}
        self._trace_index_keys: dict[int, set[str]] = {}
        self._trace_index_id: dict[int, str] = {}
        self._trace_index_cluster: dict[int, str] = {}
        self._indexed_trace_count = 0
        self.rebuild_index()

    def snapshot_id(self) -> str:
        """Content hash of the current tracebank for audit snapshots."""

        return content_hash([to_dict(trace) for trace in self.tracebank])

    def record_audit_event(self, event: DmfAuditEvent) -> None:
        self.audit_events.append(event)

    def recent_audit_summaries(self, limit: int = 8) -> list[str]:
        if limit <= 0:
            return []
        return [event.summary for event in self.audit_events[-limit:]]

    def rebuild_index(self) -> None:
        """Rebuild sparse retrieval indexes from the editable tracebank."""

        self._cue_index.clear()
        self._trace_id_index.clear()
        self._cluster_index.clear()
        self._trace_index_keys.clear()
        self._trace_index_id.clear()
        self._trace_index_cluster.clear()
        self._indexed_trace_count = 0
        for idx in range(len(self.tracebank)):
            self.reindex_trace(idx)
        self._indexed_trace_count = len(self.tracebank)

    def reindex_trace(self, idx: int) -> None:
        if not 0 <= idx < len(self.tracebank):
            return

        old_keys = self._trace_index_keys.pop(idx, set())
        for key in old_keys:
            indices = self._cue_index.get(key)
            if indices is not None:
                indices.discard(idx)
                if not indices:
                    self._cue_index.pop(key, None)

        old_trace_id = self._trace_index_id.pop(idx, "")
        if old_trace_id and self._trace_id_index.get(old_trace_id) == idx:
            self._trace_id_index.pop(old_trace_id, None)

        old_cluster = self._trace_index_cluster.pop(idx, "")
        if old_cluster:
            members = self._cluster_index.get(old_cluster)
            if members is not None:
                members.discard(idx)
                if not members:
                    self._cluster_index.pop(old_cluster, None)

        record = self.tracebank[idx]
        keys = {key for key in record.cue_affinities if key}
        self._trace_index_keys[idx] = keys
        for key in keys:
            self._cue_index.setdefault(key, set()).add(idx)

        trace_id = record.trace_id.strip()
        if trace_id:
            self._trace_id_index[trace_id] = idx
            self._trace_index_id[idx] = trace_id

        cluster_id = record.cluster_id.strip()
        if cluster_id:
            self._cluster_index.setdefault(cluster_id, set()).add(idx)
            self._trace_index_cluster[idx] = cluster_id

    def index_new_traces(self) -> None:
        while self._indexed_trace_count < len(self.tracebank):
            self.reindex_trace(self._indexed_trace_count)
            self._indexed_trace_count += 1

    @staticmethod
    def cue_key_weights(cue_cloud: CueCloud) -> dict[str, float]:
        cue_weights: dict[str, float] = {}
        for req in cue_cloud.primitive_trace_activations:
            key = req.trace_id.strip()
            if not key:
                continue
            cue_weights[key] = max(cue_weights.get(key, 0.0), float(req.weight))
        for req in cue_cloud.relational_trace_activations:
            key = req.trace_id.strip()
            if not key:
                continue
            cue_weights[key] = max(cue_weights.get(key, 0.0), float(req.weight))
        for key, weight in cue_cloud.soft_context_priors.items():
            if key:
                cue_weights[key] = max(cue_weights.get(key, 0.0), float(weight))
        return cue_weights

    @staticmethod
    def _cue_reason_routes(cue_cloud: CueCloud) -> dict[str, list[dict[str, object]]]:
        routes: dict[str, list[dict[str, object]]] = {}
        for req in cue_cloud.primitive_trace_activations:
            key = req.trace_id.strip()
            if not key:
                continue
            routes.setdefault(key, []).append(
                {
                    "route_type": "primitive",
                    "weight": float(req.weight),
                    "evidence_refs": [ref for ref in req.evidence_refs if ref],
                    "relation_refs": [],
                    "endpoint_unit_ids": [],
                }
            )
        for req in cue_cloud.relational_trace_activations:
            key = req.trace_id.strip()
            if not key:
                continue
            routes.setdefault(key, []).append(
                {
                    "route_type": "relational",
                    "weight": float(req.weight),
                    "evidence_refs": [],
                    "relation_refs": [ref for ref in req.relation_refs if ref],
                    "endpoint_unit_ids": [ref for ref in req.endpoint_unit_ids if ref],
                }
            )
        for key, weight in cue_cloud.soft_context_priors.items():
            if not key:
                continue
            routes.setdefault(key, []).append(
                {
                    "route_type": "soft_context",
                    "weight": float(weight),
                    "evidence_refs": [],
                    "relation_refs": [],
                    "endpoint_unit_ids": [],
                }
            )
        return routes

    @staticmethod
    def _trace_activation_reasons(
        record: DmfTraceRecord,
        cue_routes: dict[str, list[dict[str, object]]],
    ) -> list[dict[str, object]]:
        reasons: list[dict[str, object]] = []
        for cue_key, affinity in sorted(record.cue_affinities.items()):
            if affinity <= 0:
                continue
            routes = cue_routes.get(cue_key, [])
            for route in routes:
                cue_weight = float(route.get("weight") or 0.0)
                if cue_weight <= 0:
                    continue
                contribution = max(0.0, min(1.0, cue_weight * float(affinity)))
                reasons.append(
                    {
                        "cue_key": cue_key,
                        "cue_weight": round(cue_weight, 6),
                        "trace_affinity": round(float(affinity), 6),
                        "contribution": round(contribution, 6),
                        "evidence_refs": list(route.get("evidence_refs") or []),
                        "relation_refs": list(route.get("relation_refs") or []),
                        "endpoint_unit_ids": list(route.get("endpoint_unit_ids") or []),
                        "route_type": str(route.get("route_type") or "unknown"),
                        "summary": (
                            f"matched cue '{cue_key}' with affinity "
                            f"{float(affinity):.3f}"
                        ),
                    }
                )
        reasons.sort(key=lambda item: (-float(item["contribution"]), str(item["cue_key"])))
        return reasons

    @staticmethod
    def _heat_multiplier(heat_tier: str, heat_policy: str) -> float:
        base = {
            HeatTier.HOT.value: 1.0,
            HeatTier.WARM.value: 0.85,
            HeatTier.COLD.value: 0.65,
            HeatTier.FROZEN.value: 0.45,
        }.get(heat_tier, 0.8)
        if heat_policy == "favor_hot":
            return base * (1.15 if heat_tier == HeatTier.HOT.value else 1.0)
        if heat_policy == "favor_cold":
            return base * (1.15 if heat_tier == HeatTier.COLD.value else 1.0)
        return base

    def _score_trace(
        self,
        record: DmfTraceRecord,
        cue_weights: dict[str, float],
        carryover: set[str],
        heat_policy: str,
        quarantine_filter: bool,
    ) -> float:
        if quarantine_filter and record.maturity_state == MaturityState.PROVISIONAL.value:
            # Keep provisional traces from dominating unless explicitly allowed.
            provisional_cap = 0.35
        else:
            provisional_cap = 1.0

        score = 0.0
        matched = 0
        for emitted_cue, weight in cue_weights.items():
            if weight <= 0:
                continue
            affinity = best_affinity_for_cue(emitted_cue, record.cue_affinities)
            if affinity > 0:
                score += float(weight) * affinity
                matched += 1
        if record.trace_id and record.trace_id in carryover:
            score += 0.12
        score += record.activation_bias
        score *= self._heat_multiplier(record.heat_tier, heat_policy)

        co_boost = 0.0
        for linked_idx, link_weight in record.coactivation_links.items():
            if 0 <= linked_idx < len(self.tracebank):
                linked = self.tracebank[linked_idx]
                if linked.trace_id and linked.trace_id in carryover:
                    co_boost += max(0.0, float(link_weight)) * 0.2
        score += co_boost

        if matched == 0:
            score = max(score, 0.02)

        score = max(0.0, min(1.0, score))
        return min(score, provisional_cap)

    def _filter_activated_traces(
        self,
        scored: list[tuple[int, DmfTraceRecord, float]],
        max_traces: int,
        cue_cloud: CueCloud | None = None,
        *,
        floor_percentile: float = DMF_ACTIVATION_FLOOR_PERCENTILE,
        min_score: float = DMF_MIN_ACTIVATION_SCORE,
    ) -> tuple[list[tuple[int, DmfTraceRecord, float]], int]:
        """Emit activated traces above the score floor, capped at ``max_traces``.

        When cue routes compete on evidence, seed one best-matching trace per route
        (if it clears the floor) so diverse hypotheticals are not drowned by one cluster.
        """
        activated = [item for item in scored if item[2] > min_score]
        if not activated:
            return [], 0

        scores = sorted(item[2] for item in activated)
        if len(scores) == 1:
            threshold = scores[0]
        else:
            index = max(0, min(len(scores) - 1, int(floor_percentile * (len(scores) - 1))))
            threshold = scores[index]

        passing = [item for item in activated if item[2] >= threshold]
        if not passing:
            passing = list(activated)

        chosen_ids: set[int] = set()
        chosen: list[tuple[int, DmfTraceRecord, float]] = []
        route_seeded = 0

        if cue_cloud is not None:
            for cue_keys in competing_cue_keys(cue_cloud).values():
                for cue_key in cue_keys:
                    best = self._best_scored_for_cue_key(passing, cue_key)
                    if best is None or best[2] < threshold:
                        continue
                    if best[0] in chosen_ids:
                        continue
                    chosen.append(best)
                    chosen_ids.add(best[0])
                    route_seeded += 1
                    if len(chosen) >= max_traces:
                        return chosen[:max_traces], route_seeded

        for item in passing:
            if item[0] in chosen_ids:
                continue
            chosen.append(item)
            chosen_ids.add(item[0])
            if len(chosen) >= max_traces:
                break
        return chosen, route_seeded

    def _best_scored_for_cue_key(
        self,
        scored: list[tuple[int, DmfTraceRecord, float]],
        cue_key: str,
    ) -> tuple[int, DmfTraceRecord, float] | None:
        best: tuple[int, DmfTraceRecord, float] | None = None
        for item in scored:
            record = item[1]
            affinity = best_affinity_for_cue(cue_key, record.cue_affinities)
            if affinity <= 0:
                continue
            if best is None or item[2] > best[2]:
                best = item
        return best

    def _candidate_indices(
        self,
        cue_weights: dict[str, float],
        carryover: set[str],
        max_traces: int,
        *,
        cue_cloud: CueCloud | None = None,
    ) -> set[int]:
        self.index_new_traces()
        candidates: set[int] = set()

        for cue_key in cue_weights:
            candidates.update(self._cue_index.get(cue_key, set()))
            for idx, record in enumerate(self.tracebank):
                if best_affinity_for_cue(cue_key, record.cue_affinities) > 0.0:
                    candidates.add(idx)

        if cue_cloud is not None:
            for cue_keys in competing_cue_keys(cue_cloud).values():
                for cue_key in cue_keys:
                    candidates.update(self._cue_index.get(cue_key, set()))

        for trace_id in carryover:
            idx = self._trace_id_index.get(trace_id)
            if idx is not None:
                candidates.add(idx)

        expanded = set(candidates)
        expansion_cap = max(max_traces * 4, max_traces + 8)
        for idx in sorted(candidates):
            if len(expanded) >= expansion_cap:
                break
            if not 0 <= idx < len(self.tracebank):
                continue
            record = self.tracebank[idx]
            if record.cluster_id:
                for peer_idx in sorted(self._cluster_index.get(record.cluster_id, set())):
                    expanded.add(peer_idx)
                    if len(expanded) >= expansion_cap:
                        break
            for linked_idx in sorted(record.coactivation_links):
                if 0 <= linked_idx < len(self.tracebank):
                    expanded.add(linked_idx)
                if len(expanded) >= expansion_cap:
                    break

        return expanded

    @staticmethod
    def _entropy(values: list[float]) -> float:
        total = sum(values)
        if total <= 0:
            return 0.0
        probs = [v / total for v in values if v > 0]
        if not probs:
            return 0.0
        return -sum(p * log2(p) for p in probs)

    def run(self, dmf_input: DmfInput) -> DmfOutput:
        cue_weights = self.cue_key_weights(dmf_input.cue_cloud)
        cue_routes = self._cue_reason_routes(dmf_input.cue_cloud)
        carryover = set(dmf_input.prior_active_trace_ids)
        max_traces = max(1, int(dmf_input.compute_policy.max_active_traces))
        candidate_indices = self._candidate_indices(
            cue_weights,
            carryover,
            max_traces,
            cue_cloud=dmf_input.cue_cloud,
        )

        scored: list[tuple[int, DmfTraceRecord, float]] = []
        for idx in sorted(candidate_indices):
            if not 0 <= idx < len(self.tracebank):
                continue
            record = self.tracebank[idx]
            act = self._score_trace(
                record,
                cue_weights,
                carryover,
                dmf_input.heat_policy,
                dmf_input.quarantine_filter,
            )
            scored.append((idx, record, act))
        scored.sort(key=lambda x: x[2], reverse=True)
        top, route_seeded = self._filter_activated_traces(
            scored,
            max_traces,
            dmf_input.cue_cloud,
        )

        active_traces: list[ActiveTrace] = []
        adjusted_activations: dict[str, float] = {}
        activation_reasons: dict[str, list[dict[str, object]]] = {}
        for idx, record, activation in top:
            inhibition = 0.0
            for linked_idx, conflict_weight in record.conflict_links.items():
                for ranked_idx, _, ranked_act in top[:8]:
                    if ranked_idx == linked_idx and ranked_act > activation:
                        inhibition += max(0.0, float(conflict_weight)) * 0.15
            adj = max(0.0, activation - inhibition)
            active_traces.append(
                ActiveTrace(
                    trace_id=record.trace_id,
                    activation=adj,
                    cluster_id=record.cluster_id,
                    heat_tier=record.heat_tier,
                    is_novel=False,
                )
            )
            key = record.trace_id if record.trace_id else f"idx:{idx}"
            adjusted_activations[key] = adj
            reasons = self._trace_activation_reasons(record, cue_routes)
            if reasons:
                activation_reasons[key] = reasons

        cluster_map: dict[str, list[ActiveTrace]] = {}
        for trace in active_traces:
            cid = trace.cluster_id if trace.cluster_id else "unclustered"
            cluster_map.setdefault(cid, []).append(trace)
        trace_clusters: list[TraceCluster] = []
        for cid, members in cluster_map.items():
            acts = [m.activation for m in members]
            strength = sum(acts) / max(1, len(acts))
            coherence = 1.0 - (max(acts) - min(acts) if len(acts) > 1 else 0.0)
            trace_clusters.append(
                TraceCluster(
                    cluster_id=cid,
                    member_trace_ids=[m.trace_id for m in members],
                    cluster_strength=max(0.0, min(1.0, strength)),
                    cluster_coherence=max(0.0, min(1.0, coherence)),
                )
            )
        trace_clusters.sort(key=lambda c: c.cluster_strength, reverse=True)

        top_idx_to_rank = {idx: rank for rank, (idx, _, _) in enumerate(top[:16])}
        conflict_signals: list[ConflictSignal] = []
        for idx, record, _ in top[:16]:
            for linked_idx, weight in record.conflict_links.items():
                if linked_idx in top_idx_to_rank and linked_idx > idx:
                    other = self.tracebank[linked_idx]
                    conflict_signals.append(
                        ConflictSignal(
                            trace_id_a=record.trace_id,
                            trace_id_b=other.trace_id,
                            severity=max(0.0, min(1.0, float(weight))),
                            scope_frame_id="",
                        )
                    )

        known_cues = set(self._cue_index)
        unmatched = [k for k in cue_weights if k not in known_cues]
        novelty_signals: list[NoveltySignal] = []
        if unmatched:
            novelty_score = min(1.0, len(unmatched) / max(1, len(cue_weights)))
            action = "spawn_provisional" if novelty_score >= 0.6 else "widen_search"
            novelty_signals.append(
                NoveltySignal(
                    region_or_evidence_ref=unmatched[0],
                    novelty_score=novelty_score,
                    suggested_action=action,
                )
            )
        coverage_score = 0.0
        if cue_weights:
            matched_count = len([k for k in cue_weights if k in known_cues])
            coverage_score = matched_count / len(cue_weights)

        activations = [t.activation for t in active_traces]
        top_margin = 0.0
        second_margin = 0.0
        if len(activations) >= 2:
            top_margin = activations[0] - activations[1]
        cluster_strengths = [c.cluster_strength for c in trace_clusters]
        if len(cluster_strengths) >= 2:
            second_margin = cluster_strengths[0] - cluster_strengths[1]
        activation_entropy = self._entropy(activations)
        if top_margin >= 0.2 and second_margin >= 0.2:
            uncertainty = "low"
        elif top_margin < 0.05 or second_margin < 0.05:
            uncertainty = "high"
        else:
            uncertainty = "medium"

        return DmfOutput(
            active_traces=active_traces,
            trace_clusters=trace_clusters,
            novelty_signals=novelty_signals,
            conflict_signals=conflict_signals,
            top_margin=max(0.0, top_margin),
            second_margin=max(0.0, second_margin),
            activation_entropy=activation_entropy,
            coverage_score=coverage_score,
            adjusted_activations=adjusted_activations,
            activation_reasons=activation_reasons,
            uncertainty_summary=uncertainty,
            tracebank_snapshot_id=dmf_input.tracebank_snapshot_id,
            provenance=dmf_input.cue_cloud.provenance,
            audit_log={
                "tracebank_size": len(self.tracebank),
                "candidate_traces": len(candidate_indices),
                "selected_traces": len(active_traces),
                "compute_limit": max_traces,
                "retrieval_mode": "activated_threshold_filter",
                "activation_floor_percentile": DMF_ACTIVATION_FLOOR_PERCENTILE,
                "max_emit_traces": max_traces,
                "route_diversity_seeded": route_seeded,
                "competing_evidence_refs": len(competing_cue_keys(dmf_input.cue_cloud)),
                "trace_reason_count": sum(len(items) for items in activation_reasons.values()),
                "provisional_traces": len(
                    [
                        trace
                        for trace in self.tracebank
                        if trace.maturity_state == MaturityState.PROVISIONAL.value
                    ]
                ),
                "active_or_better_traces": len(
                    [
                        trace
                        for trace in self.tracebank
                        if trace.maturity_state != MaturityState.PROVISIONAL.value
                    ]
                ),
            },
        )
