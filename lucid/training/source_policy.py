"""Trust policy for training evidence.

Generated episodes are useful canaries, but their gold labels should not become
checkpoint-changing evidence by default. This module keeps that boundary explicit
for module trainers, global training, and the training orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from lucid.ir.training import Episode

GENERATOR_BLOCK_REASON = "generator_gold_validation_only"

_TRUSTED_SOURCE_PREFIXES = (
    "article",
    "benchmark",
    "human",
    "imported",
    "lucid.chat",
    "observed",
    "real",
    "validator",
)


@dataclass(frozen=True, slots=True)
class TrainingSourcePolicy:
    source_kind: str
    source_role: str
    validator_type: str
    promotion_eligible: bool
    block_reason: str = ""
    generator_gold_allowed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "source_role": self.source_role,
            "validator_type": self.validator_type,
            "promotion_eligible": self.promotion_eligible,
            "block_reason": self.block_reason,
            "generator_gold_allowed": self.generator_gold_allowed,
        }


def episode_source_kind(episode: Episode) -> str:
    meta = episode.meta if isinstance(episode.meta, dict) else {}
    source = str(meta.get("source") or meta.get("origin") or "").strip().lower()

    if meta.get("recipe") or meta.get("generator") or meta.get("synthetic"):
        return "generator"
    if source.startswith(("generator", "synthetic", "corpus")):
        return "generator"
    if source and source.startswith(_TRUSTED_SOURCE_PREFIXES):
        return "trusted"
    return "unknown"


def training_source_policy(
    episode: Episode,
    *,
    allow_generator_gold: bool = False,
) -> TrainingSourcePolicy:
    source_kind = episode_source_kind(episode)
    is_generator = source_kind == "generator"
    promotion_eligible = not is_generator or allow_generator_gold
    source_role = "training_evidence" if promotion_eligible else "validation_canary"
    return TrainingSourcePolicy(
        source_kind=source_kind,
        source_role=source_role,
        validator_type="gold_episode",
        promotion_eligible=promotion_eligible,
        block_reason="" if promotion_eligible else GENERATOR_BLOCK_REASON,
        generator_gold_allowed=bool(allow_generator_gold),
    )


def policy_metadata(policy: TrainingSourcePolicy) -> dict[str, Any]:
    return {"training_policy": policy.as_dict()}


def policy_from_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    policy = metadata.get("training_policy")
    return policy if isinstance(policy, Mapping) else {}


def training_episode_promotion_eligible(training_episode: Any) -> bool:
    policy = policy_from_metadata(getattr(training_episode, "metadata", {}))
    if "promotion_eligible" not in policy:
        return True
    return bool(policy.get("promotion_eligible"))


def training_episode_block_reason(training_episode: Any) -> str:
    policy = policy_from_metadata(getattr(training_episode, "metadata", {}))
    return str(policy.get("block_reason") or GENERATOR_BLOCK_REASON)


def training_episode_policy_dict(training_episode: Any) -> dict[str, Any]:
    policy = policy_from_metadata(getattr(training_episode, "metadata", {}))
    return dict(policy)
