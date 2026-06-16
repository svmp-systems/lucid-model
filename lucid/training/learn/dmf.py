"""Training updates for the dynamic memory field."""

from __future__ import annotations

from typing import Any

from lucid.audit.dmf import DmfUpdateAuditLogger
from lucid.audit.logger import content_hash
from lucid.ir.common import HeatTier, MaturityState
from lucid.ir.cue import CueCloud
from lucid.ir.serde import to_dict
from lucid.cognition.memory.dmf import DmfAuditEvent, DmfTraceRecord, DynamicMemoryField

MAX_LEARNED_LINK_DEGREE = 16


def _next_trace_id(dmf: DynamicMemoryField) -> str:
    max_seen = 0
    for trace in dmf.tracebank:
        raw = trace.trace_id.strip()
        if len(raw) > 1 and raw[0] == "t" and raw[1:].isdigit():
            max_seen = max(max_seen, int(raw[1:]))
    return f"t{max_seen + 1:04d}"


def _tracebank_snapshot_id(dmf: DynamicMemoryField) -> str:
    return content_hash([to_dict(trace) for trace in dmf.tracebank])


def _emit_audit_event(
    dmf: DynamicMemoryField,
    event: DmfAuditEvent,
    *,
    trace_before: Any = None,
    trace_after: Any = None,
    snapshot_before: str = "",
) -> None:
    snapshot_after = _tracebank_snapshot_id(dmf)
    if dmf.audit_base_dir is not None:
        DmfUpdateAuditLogger(dmf.audit_base_dir).write_event(
            event,
            trace_before=trace_before,
            trace_after=trace_after,
            tracebank_snapshot_before=snapshot_before,
            tracebank_snapshot_after=snapshot_after,
        )
    dmf.record_audit_event(event)


def _best_trace_for_cue(
    dmf: DynamicMemoryField,
    cue_key: str,
    min_affinity: float,
) -> int | None:
    best_idx: int | None = None
    best_affinity = min_affinity
    for idx, trace in enumerate(dmf.tracebank):
        affinity = float(trace.cue_affinities.get(cue_key, 0.0))
        if affinity >= best_affinity:
            best_idx = idx
            best_affinity = affinity
    return best_idx


def _link_coactivations(
    dmf: DynamicMemoryField,
    trace_indices: list[int],
    learning_rate: float,
) -> None:
    unique = sorted(set(trace_indices))
    for idx in unique:
        if not 0 <= idx < len(dmf.tracebank):
            continue
        trace = dmf.tracebank[idx]
        snapshot_before = _tracebank_snapshot_id(dmf)
        trace_before = to_dict(trace)
        changed = 0
        for other_idx in unique:
            if other_idx == idx:
                continue
            old = trace.coactivation_links.get(other_idx, 0.0)
            new = min(1.0, old + learning_rate)
            if new != old:
                trace.coactivation_links[other_idx] = new
                changed += 1
        pruned = _prune_link_map(trace.coactivation_links, max_degree=MAX_LEARNED_LINK_DEGREE)
        if changed or pruned:
            _emit_audit_event(
                dmf,
                DmfAuditEvent(
                    event_type="link_coactivation",
                    summary=(
                        f"Updated {changed} coactivation link(s) for trace {idx}; "
                        f"pruned {pruned} stale link(s)."
                    ),
                    trace_index=idx,
                    trace_id_after=trace.trace_id,
                    cue_keys=list(trace.cue_affinities),
                    details={
                        "learning_rate": learning_rate,
                        "linked_trace_count": changed,
                        "max_link_degree": MAX_LEARNED_LINK_DEGREE,
                        "pruned_link_count": pruned,
                    },
                ),
                trace_before=trace_before,
                trace_after=trace,
                snapshot_before=snapshot_before,
            )


def _prune_link_map(links: dict[int, float], *, max_degree: int) -> int:
    if max_degree <= 0 or len(links) <= max_degree:
        return 0
    keep = {
        idx
        for idx, _weight in sorted(
            links.items(),
            key=lambda item: (-float(item[1]), int(item[0])),
        )[:max_degree]
    }
    before = len(links)
    for idx in list(links):
        if idx not in keep:
            links.pop(idx, None)
    return before - len(links)


def _select_cues_for_winner(
    trace: DmfTraceRecord,
    cue_weights: dict[str, float],
    *,
    winner_position: int,
    winner_count: int,
) -> dict[str, float]:
    existing = {
        key: weight
        for key, weight in cue_weights.items()
        if key in trace.cue_affinities or key in trace.created_from_cues
    }
    if existing:
        return existing

    ranked = sorted(cue_weights.items(), key=lambda item: (-item[1], item[0]))
    if not ranked:
        return {}
    if winner_count <= len(ranked):
        key, weight = ranked[winner_position % len(ranked)]
        return {key: weight}
    key, weight = ranked[0]
    return {key: weight}


def learn_from_episode(
    dmf: DynamicMemoryField,
    cue_cloud: CueCloud,
    winning_trace_indices: list[int] | None = None,
    learning_rate: float = 0.2,
    spawn_if_novel: bool = True,
    match_threshold: float = 0.05,
) -> list[int]:
    """Reinforce matching traces or spawn quarantined provisional traces.

    New traces stay anonymous (`trace_id=""`) until lucidity-gated promotion.
    """

    cue_weights = dmf.cue_key_weights(cue_cloud)
    if not cue_weights:
        return []
    winners = winning_trace_indices or []
    updated: list[int] = []
    if winners:
        valid_winners = [idx for idx in winners if 0 <= idx < len(dmf.tracebank)]
        for winner_position, idx in enumerate(valid_winners):
            if 0 <= idx < len(dmf.tracebank):
                rec = dmf.tracebank[idx]
                selected_cues = _select_cues_for_winner(
                    rec,
                    cue_weights,
                    winner_position=winner_position,
                    winner_count=len(valid_winners),
                )
                if not selected_cues:
                    continue
                snapshot_before = _tracebank_snapshot_id(dmf)
                trace_before = to_dict(rec)
                for key, weight in selected_cues.items():
                    old = rec.cue_affinities.get(key, 0.0)
                    rec.cue_affinities[key] = min(1.0, old + learning_rate * weight)
                rec.activation_count += 1
                rec.last_update_summary = "reinforced selected cue links for training winner"
                updated.append(idx)
                dmf.reindex_trace(idx)
                _emit_audit_event(
                    dmf,
                    DmfAuditEvent(
                        event_type="reinforce_trace",
                        summary=(
                            f"Reinforced trace {idx} from selected training winner "
                            f"with {len(selected_cues)} cue link(s)."
                        ),
                        trace_index=idx,
                        trace_id_after=rec.trace_id,
                        cue_keys=list(selected_cues),
                        details={
                            "learning_rate": learning_rate,
                            "candidate_cues": len(cue_weights),
                            "selected_cues": len(selected_cues),
                        },
                    ),
                    trace_before=trace_before,
                    trace_after=rec,
                    snapshot_before=snapshot_before,
                )
        _link_coactivations(dmf, updated, learning_rate * 0.5)
        for idx in updated:
            dmf.reindex_trace(idx)
        return updated

    for key, weight in cue_weights.items():
        idx = _best_trace_for_cue(dmf, key, match_threshold)
        if idx is not None:
            rec = dmf.tracebank[idx]
            snapshot_before = _tracebank_snapshot_id(dmf)
            trace_before = to_dict(rec)
            old = rec.cue_affinities.get(key, 0.0)
            rec.cue_affinities[key] = min(1.0, old + learning_rate * weight)
            rec.activation_count += 1
            rec.last_update_summary = "matched recurring cue pattern"
            updated.append(idx)
            dmf.reindex_trace(idx)
            _emit_audit_event(
                dmf,
                DmfAuditEvent(
                    event_type="update_existing_trace",
                    summary=f"Updated existing trace {idx} for recurring cue '{key}'.",
                    trace_index=idx,
                    trace_id_after=rec.trace_id,
                    cue_keys=[key],
                    details={
                        "old_affinity": old,
                        "new_affinity": rec.cue_affinities[key],
                    },
                ),
                trace_before=trace_before,
                trace_after=rec,
                snapshot_before=snapshot_before,
            )
            continue

        if spawn_if_novel:
            snapshot_before = _tracebank_snapshot_id(dmf)
            dmf.tracebank.append(
                DmfTraceRecord(
                    trace_id="",
                    cue_affinities={key: min(1.0, weight)},
                    maturity_state=MaturityState.PROVISIONAL.value,
                    heat_tier=HeatTier.HOT.value,
                    created_from_cues=[key],
                    last_update_summary="spawned provisional trace from novel cue",
                )
            )
            new_idx = len(dmf.tracebank) - 1
            updated.append(new_idx)
            dmf.reindex_trace(new_idx)
            _emit_audit_event(
                dmf,
                DmfAuditEvent(
                    event_type="spawn_provisional_trace",
                    summary=f"Spawned anonymous provisional trace {new_idx} for novel cue '{key}'.",
                    trace_index=new_idx,
                    cue_keys=[key],
                    details={"initial_affinity": min(1.0, weight)},
                ),
                trace_before=None,
                trace_after=dmf.tracebank[new_idx],
                snapshot_before=snapshot_before,
            )
    _link_coactivations(dmf, updated, learning_rate * 0.5)
    for idx in updated:
        dmf.reindex_trace(idx)
    return sorted(set(updated))


def apply_lucidity_trace_feedback(
    dmf: DynamicMemoryField,
    trace_indices: list[int],
    *,
    passed_lucidity: bool,
    promotion_threshold: int = 3,
    failure_quarantine_threshold: int = 2,
) -> list[int]:
    """Apply local success/failure feedback and promote only after lucidity passes."""

    promoted: list[int] = []
    for idx in sorted(set(trace_indices)):
        if not 0 <= idx < len(dmf.tracebank):
            continue
        trace = dmf.tracebank[idx]
        before_id = trace.trace_id
        snapshot_before = _tracebank_snapshot_id(dmf)
        trace_before = to_dict(trace)
        if passed_lucidity:
            trace.success_count += 1
            if (
                trace.trace_id == ""
                and trace.maturity_state == MaturityState.PROVISIONAL.value
                and trace.success_count >= promotion_threshold
            ):
                trace.trace_id = _next_trace_id(dmf)
                trace.maturity_state = MaturityState.ACTIVE.value
                trace.last_update_summary = "promoted after lucidity-gated successes"
                promoted.append(idx)
                dmf.reindex_trace(idx)
                _emit_audit_event(
                    dmf,
                    DmfAuditEvent(
                        event_type="promote_trace",
                        summary=(
                            f"Promoted trace {idx} from anonymous provisional slot "
                            f"to learned id {trace.trace_id}."
                        ),
                        trace_index=idx,
                        trace_id_before=before_id,
                        trace_id_after=trace.trace_id,
                        cue_keys=list(trace.cue_affinities),
                        details={"success_count": trace.success_count},
                    ),
                    trace_before=trace_before,
                    trace_after=trace,
                    snapshot_before=snapshot_before,
                )
            else:
                trace.last_update_summary = "lucidity success counted; not yet promoted"
                dmf.reindex_trace(idx)
                _emit_audit_event(
                    dmf,
                    DmfAuditEvent(
                        event_type="count_lucidity_success",
                        summary=f"Counted lucidity success for trace {idx}; promotion threshold not met.",
                        trace_index=idx,
                        trace_id_before=before_id,
                        trace_id_after=trace.trace_id,
                        cue_keys=list(trace.cue_affinities),
                        details={"success_count": trace.success_count},
                    ),
                    trace_before=trace_before,
                    trace_after=trace,
                    snapshot_before=snapshot_before,
                )
        else:
            trace.failure_count += 1
            trace.last_update_summary = "failed lucidity feedback; kept quarantined"
            if trace.failure_count >= failure_quarantine_threshold:
                trace.maturity_state = MaturityState.PROVISIONAL.value
                trace.heat_tier = HeatTier.HOT.value
            dmf.reindex_trace(idx)
            _emit_audit_event(
                dmf,
                DmfAuditEvent(
                    event_type="quarantine_trace",
                    summary=f"Kept trace {idx} quarantined after failed lucidity feedback.",
                    trace_index=idx,
                    trace_id_after=trace.trace_id,
                    cue_keys=list(trace.cue_affinities),
                    details={"failure_count": trace.failure_count},
                ),
                trace_before=trace_before,
                trace_after=trace,
                snapshot_before=snapshot_before,
            )
    return promoted
