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
from lucid.ir.serde import to_dict
from lucid.paths import DEFAULT_AUDIT_DMF, resolve_train_path


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
        last_update_summary=str(record.get("last_update_summary") or ""),
    )


def tracebank_from_checkpoint(path: str | Path) -> list[DmfTraceRecord]:
    """Load tracebank rows from a Lucid checkpoint directory."""

    store_path = Path(path) / "tracebank.json"
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
        root = Path(checkpoint)
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
        for cue_key, affinity in record.cue_affinities.items():
            w = cue_weights.get(cue_key, 0.0)
            if w > 0 and affinity > 0:
                score += w * affinity
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

    def _candidate_indices(
        self,
        cue_weights: dict[str, float],
        carryover: set[str],
        max_traces: int,
    ) -> set[int]:
        self.index_new_traces()
        candidates: set[int] = set()

        for cue_key in cue_weights:
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
        carryover = set(dmf_input.prior_active_trace_ids)
        max_traces = max(1, int(dmf_input.compute_policy.max_active_traces))
        candidate_indices = self._candidate_indices(cue_weights, carryover, max_traces)

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
        top = scored[:max_traces]

        active_traces: list[ActiveTrace] = []
        adjusted_activations: dict[str, float] = {}
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
            uncertainty_summary=uncertainty,
            tracebank_snapshot_id=dmf_input.tracebank_snapshot_id,
            provenance=dmf_input.cue_cloud.provenance,
            audit_log={
                "tracebank_size": len(self.tracebank),
                "candidate_traces": len(candidate_indices),
                "selected_traces": len(active_traces),
                "compute_limit": max_traces,
                "retrieval_mode": "indexed_top_k",
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
