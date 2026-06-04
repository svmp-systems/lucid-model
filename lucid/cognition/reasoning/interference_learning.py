"""Interference learning: small scoped updates for learned support/conflict links."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from lucid.audit.logger import content_hash
from lucid.ir.context_op import InterferenceGate
from lucid.ir.interference import (
    InterferenceInput,
    InterferenceLearningPatch,
    InterferenceLearningResult,
    InterferenceOutput,
    LearnedInterferenceLink,
)
from lucid.ir.serde import to_dict, to_json

SCHEMA_VERSION = 1
DEFAULT_INTERFERENCE_STORE = Path("audit/interference_learning/interference_links.json")
DEFAULT_INTERFERENCE_LEARNING_AUDIT_DIR = Path("audit/interference_learning/runs")


@dataclass(slots=True)
class StoredInterferenceLink:
    source_id: str
    target_id: str
    scope_hint: str = ""
    weight: float = 0.0
    positive_updates: int = 0
    negative_updates: int = 0
    last_reason: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class InterferenceStoreSnapshot:
    schema_version: int = SCHEMA_VERSION
    links: list[StoredInterferenceLink] = field(default_factory=list)
    audit_notes: list[str] = field(default_factory=list)


def learn_interference(
    inp: InterferenceInput,
    out: InterferenceOutput,
    *,
    validation_success: bool,
    failure_type: str = "",
    store_path: str | Path = DEFAULT_INTERFERENCE_STORE,
    audit_dir: str | Path = DEFAULT_INTERFERENCE_LEARNING_AUDIT_DIR,
    learning_rate: float = 0.08,
) -> InterferenceLearningResult:
    """Build, persist, and audit scoped interference link updates."""
    patches = build_interference_patches(
        inp,
        out,
        validation_success=validation_success,
        failure_type=failure_type,
        learning_rate=learning_rate,
    )
    store = InterferenceGraphStore(store_path)
    store.apply_patches(patches)
    audit_path = write_interference_learning_audit(
        audit_dir,
        inp,
        out,
        patches,
        store_path=store.path,
        validation_success=validation_success,
        failure_type=failure_type,
    )
    return InterferenceLearningResult(
        patches=patches,
        store_path=str(store.path),
        audit_path=str(audit_path),
        audit_notes=[
            f"patches={len(patches)} validation_success={validation_success}",
            f"store={store.path}",
            f"audit={audit_path}",
        ],
    )


def build_interference_patches(
    inp: InterferenceInput,
    out: InterferenceOutput,
    *,
    validation_success: bool,
    failure_type: str = "",
    learning_rate: float = 0.08,
) -> list[InterferenceLearningPatch]:
    """Convert a validated run into tiny trace-link updates."""
    if validation_success:
        return _success_patches(inp, learning_rate)
    if failure_type and "interference" not in failure_type and "basin" not in failure_type:
        return []
    return _failure_patches(inp, out, learning_rate)


def load_learned_interference_links(
    store_path: str | Path = DEFAULT_INTERFERENCE_STORE,
) -> list[LearnedInterferenceLink]:
    """Load persisted learned links in the shape consumed by run_interference."""
    return InterferenceGraphStore(store_path).to_learned_links()


class InterferenceGraphStore:
    """JSON-backed signed compatibility store for interference."""

    def __init__(self, path: str | Path = DEFAULT_INTERFERENCE_STORE) -> None:
        self.path = Path(path)

    def load(self) -> InterferenceStoreSnapshot:
        if not self.path.exists():
            return InterferenceStoreSnapshot(audit_notes=["new_store"])
        data = json.loads(self.path.read_text(encoding="utf-8"))
        links = [
            StoredInterferenceLink(
                source_id=item["source_id"],
                target_id=item["target_id"],
                scope_hint=item.get("scope_hint", ""),
                weight=float(item.get("weight", 0.0)),
                positive_updates=int(item.get("positive_updates", 0)),
                negative_updates=int(item.get("negative_updates", 0)),
                last_reason=item.get("last_reason", ""),
                updated_at=item.get("updated_at", ""),
            )
            for item in data.get("links", [])
        ]
        return InterferenceStoreSnapshot(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            links=links,
            audit_notes=list(data.get("audit_notes", [])),
        )

    def save(self, snapshot: InterferenceStoreSnapshot) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.links.sort(key=lambda link: (link.scope_hint, link.source_id, link.target_id))
        snapshot.audit_notes = [
            f"links={len(snapshot.links)}",
            f"updated_at={_utc_now_iso()}",
        ]
        self.path.write_text(to_json(snapshot), encoding="utf-8")

    def apply_patches(self, patches: list[InterferenceLearningPatch]) -> InterferenceStoreSnapshot:
        snapshot = self.load()
        by_key = {_link_key(link.source_id, link.target_id, link.scope_hint): link for link in snapshot.links}
        now = _utc_now_iso()
        for patch in patches:
            source_id, target_id = _ordered_pair(patch.source_id, patch.target_id)
            key = _link_key(source_id, target_id, patch.scope_hint)
            link = by_key.get(key)
            if link is None:
                link = StoredInterferenceLink(source_id, target_id, scope_hint=patch.scope_hint)
                by_key[key] = link
            link.weight = _clamp_round(link.weight + patch.delta)
            if patch.delta >= 0:
                link.positive_updates += 1
            else:
                link.negative_updates += 1
            link.last_reason = patch.reason
            link.updated_at = now
        snapshot.links = [link for link in by_key.values() if abs(link.weight) >= 0.001]
        self.save(snapshot)
        return snapshot

    def to_learned_links(self) -> list[LearnedInterferenceLink]:
        return [
            LearnedInterferenceLink(link.source_id, link.target_id, link.weight, link.scope_hint)
            for link in self.load().links
            if abs(link.weight) >= 0.001
        ]


def write_interference_learning_audit(
    audit_dir: str | Path,
    inp: InterferenceInput,
    out: InterferenceOutput,
    patches: list[InterferenceLearningPatch],
    *,
    store_path: Path,
    validation_success: bool,
    failure_type: str,
) -> Path:
    audit_base = Path(audit_dir)
    run_dir = audit_base / _audit_run_id(validation_success, inp)
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "component": "interference_learning",
        "validation_success": validation_success,
        "failure_type": failure_type,
        "store_path": str(store_path),
        "input_hash": content_hash(inp),
        "output_hash": content_hash(out),
        "patch_count": len(patches),
        "summary": {
            "headline": f"{len(patches)} interference learning patches",
            "lines": [
                f"validation_success: {validation_success}",
                f"failure_type: {failure_type or '-'}",
                f"store: {store_path}",
                f"patches: {len(patches)}",
            ],
        },
        "patches": to_dict(patches),
    }
    (run_dir / "interference_learning.json").write_text(
        json.dumps(record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    readme = "\n".join(
        [
            record["summary"]["headline"],
            "=" * len(record["summary"]["headline"]),
            "",
            *record["summary"]["lines"],
            "",
        ]
    )
    (run_dir / "README.txt").write_text(readme, encoding="utf-8")
    return run_dir


def _success_patches(
    inp: InterferenceInput,
    learning_rate: float,
) -> list[InterferenceLearningPatch]:
    patches: list[InterferenceLearningPatch] = []
    gates = {gate.scope_frame_id: gate for gate in inp.interference_gates}
    frames_by_scope = _frames_by_scope(inp)
    for scope_id, frame_ids in frames_by_scope.items():
        allowed = _allowed_trace_ids(scope_id, gates.get(scope_id))
        for frame in inp.candidate_frames:
            if frame.frame_id not in frame_ids:
                continue
            frame_trace_ids = _frame_trace_ids(frame)
            trace_ids = (
                [trace_id for trace_id in frame_trace_ids if trace_id in allowed]
                if allowed
                else frame_trace_ids
            )
            for trace_a, trace_b in combinations(sorted(set(trace_ids)), 2):
                patches.append(
                    InterferenceLearningPatch(
                        source_id=trace_a,
                        target_id=trace_b,
                        scope_hint=scope_id,
                        delta=_round(learning_rate),
                        reason="validated_success_local_coactivation",
                        evidence_refs=[f"frame:{frame.frame_id}"],
                    )
                )
    return patches


def _failure_patches(
    inp: InterferenceInput,
    out: InterferenceOutput,
    learning_rate: float,
) -> list[InterferenceLearningPatch]:
    patches: list[InterferenceLearningPatch] = []
    for report in out.conflict_reports:
        trace_ids = sorted(member for member in report.members if member.startswith("t_"))
        for trace_a, trace_b in combinations(trace_ids, 2):
            patches.append(
                InterferenceLearningPatch(
                    source_id=trace_a,
                    target_id=trace_b,
                    scope_hint=report.scope_frame_id,
                    delta=-_round(learning_rate * max(0.25, min(1.0, report.severity))),
                    reason=f"validated_failure_{report.conflict_type}",
                    evidence_refs=[f"conflict:{report.conflict_type}"],
                )
            )
    if patches:
        return patches
    for patch in _success_patches(inp, learning_rate * 0.5):
        patch.delta = -patch.delta
        patch.reason = "validated_failure_local_coactivation"
        patch.evidence_refs = [*patch.evidence_refs, "fallback:no_trace_conflict_report"]
        patches.append(patch)
    return patches


def _frames_by_scope(inp: InterferenceInput) -> dict[str, set[str]]:
    out = {context.context_frame_id: set(context.member_frame_ids) for context in inp.context_frames}
    if out:
        first_scope = inp.context_frames[0].context_frame_id
        assigned = {frame_id for frame_ids in out.values() for frame_id in frame_ids}
        for frame in inp.candidate_frames:
            if frame.frame_id not in assigned:
                out.setdefault(first_scope, set()).add(frame.frame_id)
    return out


def _allowed_trace_ids(scope_id: str, gate: InterferenceGate | None) -> set[str]:
    if gate is None:
        return set()
    return set(gate.allowed_trace_ids) - set(gate.blocked_trace_ids)


def _frame_trace_ids(frame: object) -> list[str]:
    values: list[str] = []
    values.extend(_string_values(getattr(frame, "role_assignments", {}).values()))
    values.extend(_string_values(getattr(frame, "relation_assignments", {}).values()))
    values.extend(getattr(frame, "supporting_trace_ids", []))
    values.extend(getattr(frame, "conflicting_trace_ids", []))
    return list(dict.fromkeys(values))


def _string_values(values: object) -> list[str]:
    return [str(value) for value in values if isinstance(value, str) and value.strip()]


def _link_key(source_id: str, target_id: str, scope_hint: str) -> tuple[str, str, str]:
    source_id, target_id = _ordered_pair(source_id, target_id)
    return source_id, target_id, scope_hint


def _ordered_pair(source_id: str, target_id: str) -> tuple[str, str]:
    return tuple(sorted((source_id, target_id)))  # type: ignore[return-value]


def _clamp_round(value: float) -> float:
    return _round(max(-1.0, min(1.0, value)))


def _round(value: float) -> float:
    return round(float(value), 3)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _audit_run_id(validation_success: bool, inp: InterferenceInput) -> str:
    label = "success" if validation_success else "failure"
    return f"{_utc_now_iso().replace(':', '').replace('+', 'Z')}_{label}_{content_hash(inp)[:10]}"
