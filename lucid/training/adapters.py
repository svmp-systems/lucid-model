"""Adapters from generated Episodes to module-specific training targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lucid.ir.common import Modality
from lucid.ir.cue import CueCloud, CueEncoderInput, TraceActivationRequest
from lucid.ir.perception import PerceptionInput
from lucid.ir.training import Episode
from lucid.training.generator.output import read_episodes
from lucid.training.generator.recipes import bank_destination, grid_move
from lucid.training.generator.engine import AmbiguityKnob, rng_for_seed


def load_training_episodes(path: str | Path) -> list[Episode]:
    return read_episodes(path)


def fixture_episodes(name: str) -> list[Episode]:
    if name == "bank":
        return [bank_destination.make(rng_for_seed(7), AmbiguityKnob(0.9))]
    if name == "phase1-mini":
        return [
            bank_destination.make(rng_for_seed(7), AmbiguityKnob(0.9)),
            bank_destination.make(rng_for_seed(8), AmbiguityKnob(0.1)),
            grid_move.make(rng_for_seed(9), AmbiguityKnob(0.8)),
        ]
    raise ValueError(f"unknown training fixture: {name}")


def episode_to_dmf_cue_cloud(episode: Episode) -> CueCloud:
    return CueCloud(
        primitive_trace_activations=[
            TraceActivationRequest(
                trace_id=target.trace_family,
                weight=target.weight,
                evidence_refs=[target.evidence_ref] if target.evidence_ref else [],
                keep_alive=target.keep_alive,
            )
            for target in episode.gold.trace_activations
        ],
        ambiguity_policy=episode.gold.ambiguity_policy,
        retrieval_budget_used=len(episode.gold.trace_activations),
    )


def dmf_targets(episode: Episode) -> list[dict[str, Any]]:
    return [
        {
            "trace_family": target.trace_family,
            "weight": float(target.weight),
            "evidence_ref": target.evidence_ref,
            "keep_alive": bool(target.keep_alive),
        }
        for target in episode.gold.trace_activations
    ]


def perception_targets(episode: Episode) -> dict[str, Any]:
    return {
        "spans": [
            {
                "span_id": span.span_id,
                "surface": span.surface,
                "kind_hint": span.kind_hint,
                "position": span.position,
            }
            for span in episode.gold.spans
        ],
        "markers": [
            {
                "marker_id": marker.marker_id,
                "surface": marker.surface,
                "marker_type_hints": list(marker.marker_type_hints),
            }
            for marker in episode.gold.markers
        ],
        "regions": [
            {
                "region_id": region.region_id,
                "role_hint": region.role_hint,
                "member_span_ids": list(region.member_span_ids),
            }
            for region in episode.gold.regions
        ],
    }


def cue_encoder_targets(episode: Episode) -> dict[str, Any]:
    cue_cloud = episode_to_dmf_cue_cloud(episode)
    return {
        "trace_targets": dmf_targets(episode),
        "ambiguity_policy": str(cue_cloud.ambiguity_policy),
    }


def episode_to_cue_encoder_input(episode: Episode) -> CueEncoderInput:
    """Build a cue-encoder input by running rule perception on the episode."""

    from lucid.cognition.input.perception import PerceptionConfig, perceive

    modality = (
        episode.modality if isinstance(episode.modality, Modality) else Modality(str(episode.modality))
    )
    graph = perceive(
        PerceptionInput(raw_payload=episode.raw_input, modality=modality),
        config=PerceptionConfig(backend="rule", write_audit=False),
    )
    task_intent = (
        episode.task_intent.value
        if hasattr(episode.task_intent, "value")
        else str(episode.task_intent)
    )
    return CueEncoderInput(
        perceptual_evidence_graph=graph,
        task_intent_hint=task_intent,
        retrieval_budget=max(16, len(episode.gold.trace_activations) * 2),
    )


def binding_targets(episode: Episode) -> list[dict[str, Any]]:
    return [
        {
            "span_id": assignment.span_id,
            "primary_frame": assignment.primary_frame,
            "secondary_frames": list(assignment.secondary_frames),
        }
        for assignment in episode.gold.scope_assignments
    ]


def binding_frame_targets(episode: Episode) -> list[dict[str, Any]]:
    return [
        {
            "frame_id": target.frame_id,
            "frame_type": target.frame_type,
            "slot_targets": [
                {
                    "slot_id": slot.slot_id,
                    "trace_family": slot.trace_family,
                    "member_span_ids": list(slot.member_span_ids),
                    "affinity_hints": dict(slot.affinity_hints),
                    "confidence": float(slot.confidence),
                }
                for slot in target.slot_targets
            ],
            "role_assignments": dict(target.role_assignments),
            "member_span_ids": list(target.member_span_ids),
            "unresolved_slot_names": list(target.unresolved_slot_names),
            "confidence": float(target.confidence),
        }
        for target in episode.gold.frame_targets
    ]


def context_targets(episode: Episode) -> dict[str, Any]:
    return {
        "scope_assignments": binding_targets(episode),
        "interference_gates": [
            {
                "gate_id": gate.gate_id,
                "scope_frame_id": gate.scope_frame_id,
                "allowed_trace_ids": list(gate.allowed_trace_ids),
                "blocked_trace_ids": list(gate.blocked_trace_ids),
            }
            for gate in episode.gold.interference_gates
        ],
    }


def interference_targets(episode: Episode) -> list[dict[str, Any]]:
    return context_targets(episode)["interference_gates"]


def basin_targets(episode: Episode) -> list[dict[str, Any]]:
    return [
        {
            "family_hint": target.family_hint,
            "frame_id": target.frame_id,
            "confidence": float(target.confidence),
        }
        for target in episode.gold.basin_families
    ]


def lucidity_target(episode: Episode) -> dict[str, Any]:
    return {
        "decision": episode.gold.lucidity_target,
        "rationale": episode.gold.lucidity_rationale,
        "ambiguity_policy": episode.gold.ambiguity_policy,
    }


def decoder_target(episode: Episode) -> dict[str, Any]:
    return {
        "expected_answer": episode.gold.expected_answer,
        "lucidity_target": episode.gold.lucidity_target,
        "validator": episode.validator,
    }


def projector_target(episode: Episode) -> dict[str, Any]:
    if not isinstance(episode.raw_input, dict):
        return {}
    return {
        "raw_input": episode.raw_input,
        "expected_answer": episode.gold.expected_answer,
        "task_intent": str(episode.task_intent),
    }


def episode_family(episode: Episode) -> str:
    return episode.template_id or str(episode.modality)
