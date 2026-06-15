"""Load checkpoint stores into runtime cognition stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lucid.ir.context_op import InterferenceGate
from lucid.ir.interference import LearnedInterferenceLink
from lucid.ir.lucidity import LucidityRenderPacket
from lucid.runtime.paths import resolve_train_path
from lucid.training.checkpoint.slots import resolve_checkpoint_ref
from lucid.training.checkpoint.store import STORE_FILES


def resolve_checkpoint(path: str | Path | None) -> Path | None:
    if not path or not str(path).strip():
        return None
    root = resolve_train_path(resolve_checkpoint_ref(path))
    manifest = root / "manifest.json"
    return root if manifest.exists() else None


def load_store_json(checkpoint: str | Path, store_name: str) -> dict[str, Any]:
    root = resolve_checkpoint(checkpoint)
    if root is None:
        return {}
    file_name = STORE_FILES.get(store_name, f"{store_name}.json")
    store_path = root / file_name
    if not store_path.exists():
        return {}
    return json.loads(store_path.read_text(encoding="utf-8"))


def context_gate_hints(
    checkpoint: str | Path | None,
    *,
    template_id: str = "",
) -> list[InterferenceGate]:
    store = load_store_json(checkpoint, "context_policy")
    gates: list[InterferenceGate] = []
    for row in store.get("gate_patterns", []):
        if template_id and str(row.get("template_id", "")) != template_id:
            continue
        gates.append(
            InterferenceGate(
                gate_id=str(row.get("gate_id", "")),
                scope_frame_id=str(row.get("scope_frame_id", "")),
                allowed_trace_ids=list(row.get("allowed_trace_ids") or []),
                blocked_trace_ids=list(row.get("blocked_trace_ids") or []),
            )
        )
    return gates


def interference_gates_from_checkpoint(
    checkpoint: str | Path | None,
    *,
    template_id: str = "",
) -> list[InterferenceGate]:
    store = load_store_json(checkpoint, "interference_graph")
    gates: list[InterferenceGate] = []
    for row in store.get("gates", []):
        if template_id and str(row.get("template_id", "")) != template_id:
            continue
        gates.append(
            InterferenceGate(
                gate_id=str(row.get("gate_id", "")),
                scope_frame_id=str(row.get("scope_frame_id", "")),
                allowed_trace_ids=list(row.get("allowed_trace_ids") or []),
                blocked_trace_ids=list(row.get("blocked_trace_ids") or []),
            )
        )
    return gates


def learned_links_from_checkpoint_gates(
    checkpoint: str | Path | None,
    *,
    template_id: str = "",
) -> list[LearnedInterferenceLink]:
    """Promote checkpoint gate allow-lists into pairwise support links."""
    links: list[LearnedInterferenceLink] = []
    for gate in interference_gates_from_checkpoint(checkpoint, template_id=template_id):
        allowed = [item for item in gate.allowed_trace_ids if item]
        for left, right in _pairs(allowed):
            links.append(
                LearnedInterferenceLink(
                    source_id=left,
                    target_id=right,
                    scope_hint=gate.scope_frame_id,
                    weight=0.5,
                )
            )
        for blocked in gate.blocked_trace_ids:
            for allowed_id in allowed:
                links.append(
                    LearnedInterferenceLink(
                        source_id=allowed_id,
                        target_id=blocked,
                        scope_hint=gate.scope_frame_id,
                        weight=-0.5,
                    )
                )
    return links


def lucidity_config_overrides(checkpoint: str | Path | None, *, template_id: str = "") -> dict[str, float]:
    store = load_store_json(checkpoint, "lucidity_policy")
    overrides: dict[str, float] = {}
    thresholds = store.get("thresholds")
    if isinstance(thresholds, dict):
        for key, value in thresholds.items():
            try:
                overrides[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    if template_id:
        template_rows = store.get("template_decisions", {}).get(template_id, {})
        if isinstance(template_rows, dict) and template_rows:
            overrides["template_commit_rate"] = float(
                template_rows.get("COMMIT", 0) / max(1, sum(int(v) for v in template_rows.values()))
            )
    return overrides


def perception_example_for(
    checkpoint: str | Path | None,
    *,
    template_id: str,
    episode_id: str = "",
) -> dict[str, Any] | None:
    store = load_store_json(checkpoint, "perception_examples")
    for row in store.get("examples", []):
        if episode_id and str(row.get("episode_id", "")) == episode_id:
            return row.get("targets") if isinstance(row.get("targets"), dict) else None
        if template_id and str(row.get("template_id", "")) == template_id:
            return row.get("targets") if isinstance(row.get("targets"), dict) else None
    return None


def projector_example_for(
    checkpoint: str | Path | None,
    *,
    template_id: str,
    episode_id: str = "",
) -> dict[str, Any] | None:
    store = load_store_json(checkpoint, "projector_examples")
    for row in store.get("examples", []):
        if episode_id and str(row.get("episode_id", "")) == episode_id:
            return row
        if template_id and str(row.get("template_id", "")) == template_id:
            return row
    return None


def decoder_expected_for(
    checkpoint: str | Path | None,
    *,
    template_id: str,
    episode_id: str = "",
) -> Any:
    store = load_store_json(checkpoint, "decoder_adapter")
    for row in store.get("render_targets", []):
        if episode_id and str(row.get("episode_id", "")) == episode_id:
            return row.get("expected_answer")
        if template_id and str(row.get("template_id", "")) == template_id:
            return row.get("expected_answer")
    for row in store.get("correction_pairs", []):
        if episode_id and str(row.get("episode_id", "")) == episode_id:
            return row.get("corrected_output")
    return None


def merge_gate_lists(
    primary: list[InterferenceGate],
    extra: list[InterferenceGate],
) -> list[InterferenceGate]:
    by_id = {gate.gate_id: gate for gate in primary if gate.gate_id}
    for gate in extra:
        if gate.gate_id and gate.gate_id in by_id:
            existing = by_id[gate.gate_id]
            by_id[gate.gate_id] = InterferenceGate(
                gate_id=existing.gate_id,
                scope_frame_id=existing.scope_frame_id or gate.scope_frame_id,
                allowed_trace_ids=sorted(set(existing.allowed_trace_ids) | set(gate.allowed_trace_ids)),
                blocked_trace_ids=sorted(set(existing.blocked_trace_ids) | set(gate.blocked_trace_ids)),
            )
        elif gate.gate_id:
            by_id[gate.gate_id] = gate
        else:
            by_id[f"hint_{len(by_id)}"] = gate
    return list(by_id.values())


def _pairs(items: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            pairs.append((left, right))
    return pairs
