"""Bootstrap general conversational language into a checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from lucid.runtime.paths import resolve_train_path
from lucid.training.checkpoint.metadata import ensure_metadata, record_support
from lucid.training.checkpoint.registry import register_checkpoint
from lucid.training.checkpoint.slots import _normalize_slot_path
from lucid.training.checkpoint.store import CheckpointState, checkpoint_summary, load_checkpoint, save_checkpoint
from lucid.training.corpus.engine import generate
from lucid.training.corpus.output import GENERAL_LANGUAGE_COUNTS, write_episodes

DEFAULT_GENERAL_LANGUAGE_CHECKPOINT = "v0.0.1"

SOCIAL_TRACES = [
    {
        "trace_id": "t9001",
        "trace_family": "social_speech_like",
        "alias": "social_speech_like",
        "cue_affinities": {
            "social:greeting": 0.95,
            "social:thanks": 0.9,
            "social:farewell": 0.9,
            "social:how_are_you": 0.88,
            "social:capability": 0.86,
            "phrase:hi": 0.95,
            "phrase:hello": 0.95,
            "phrase:hey": 0.92,
        },
        "created_from_episodes": ["general_language_bootstrap"],
        "activation_count": 1,
        "success_count": 1,
        "failure_count": 0,
        "maturity_state": "stable",
        "heat_tier": "warm",
    }
]

RELATION_ALIASES = [
    {"alias_id": "alias_hi", "surface_pattern": "hi", "relation_candidates": ["social"], "confidence": 0.95},
    {"alias_id": "alias_hello", "surface_pattern": "hello", "relation_candidates": ["social"], "confidence": 0.95},
    {"alias_id": "alias_thanks", "surface_pattern": "thanks", "relation_candidates": ["social"], "confidence": 0.9},
    {"alias_id": "alias_bye", "surface_pattern": "bye", "relation_candidates": ["social"], "confidence": 0.9},
]

DECODER_TARGETS = [
    {
        "template_id": "chat_greeting",
        "episode_id": "general-language-greeting",
        "expected_answer": "Hello.",
        "lucidity_target": "COMMIT",
        "validator": "exact_social",
    },
    {
        "template_id": "chat_thanks",
        "episode_id": "general-language-thanks",
        "expected_answer": "You're welcome.",
        "lucidity_target": "COMMIT",
        "validator": "exact_social",
    },
    {
        "template_id": "chat_farewell",
        "episode_id": "general-language-farewell",
        "expected_answer": "Goodbye.",
        "lucidity_target": "COMMIT",
        "validator": "exact_social",
    },
    {
        "template_id": "chat_how_are_you",
        "episode_id": "general-language-how-are-you",
        "expected_answer": "I'm here and ready to help.",
        "lucidity_target": "COMMIT",
        "validator": "exact_social",
    },
    {
        "template_id": "chat_capability",
        "episode_id": "general-language-capability",
        "expected_answer": "I'm Lucid. I answer from audited pipeline state, not open-ended guessing.",
        "lucidity_target": "COMMIT",
        "validator": "exact_social",
    },
]

LUCIDITY_TEMPLATE_DECISIONS = {
    "chat_greeting": {"COMMIT": 1},
    "chat_thanks": {"COMMIT": 1},
    "chat_farewell": {"COMMIT": 1},
    "chat_how_are_you": {"COMMIT": 1},
    "chat_capability": {"COMMIT": 1},
}

PARAPHRASE_TRACES = [
    {
        "trace_id": "t9002",
        "trace_family": "concept_query_like",
        "alias": "concept_query_like",
        "cue_affinities": {
            "concept_query": 0.92,
            "definition_query": 0.9,
            "mechanism_query": 0.88,
            "query:what_is": 0.9,
            "query:explain": 0.88,
            "query:tell_me_about": 0.86,
        },
        "created_from_episodes": ["general_language_paraphrase_bootstrap"],
        "activation_count": 1,
        "success_count": 1,
        "failure_count": 0,
        "maturity_state": "stable",
        "heat_tier": "warm",
    }
]

PARAPHRASE_ALIASES = [
    {"alias_id": "alias_what_is", "surface_pattern": "what is", "relation_candidates": ["concept_query"], "confidence": 0.9},
    {"alias_id": "alias_explain", "surface_pattern": "explain", "relation_candidates": ["concept_query"], "confidence": 0.88},
    {"alias_id": "alias_tell_me_about", "surface_pattern": "tell me about", "relation_candidates": ["concept_query"], "confidence": 0.88},
    {"alias_id": "alias_how_does", "surface_pattern": "how does", "relation_candidates": ["mechanism_query"], "confidence": 0.86},
]


def _upsert_by_key(rows: list[dict[str, Any]], key: str, record: dict[str, Any]) -> None:
    value = record.get(key)
    for index, row in enumerate(rows):
        if row.get(key) == value:
            rows[index] = {**row, **record}
            return
    rows.append(dict(record))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def bootstrap_general_language_stores(state: CheckpointState) -> None:
    trace_store = state.ensure_store("tracebank")
    alias_store = state.ensure_store("relation_aliases").setdefault("aliases", [])
    decoder_store = state.ensure_store("decoder_adapter")
    lucidity_store = state.ensure_store("lucidity_policy")

    for trace in SOCIAL_TRACES + PARAPHRASE_TRACES:
        _upsert_by_key(trace_store["records"], "trace_id", trace)
        record_support(state, f"trace:{trace['trace_id']}", "trace")

    for alias in RELATION_ALIASES + PARAPHRASE_ALIASES:
        _upsert_by_key(alias_store, "alias_id", alias)
        record_support(state, f"alias:{alias['alias_id']}", "relation_alias")

    render_targets = decoder_store.setdefault("render_targets", [])
    for target in DECODER_TARGETS:
        _upsert_by_key(render_targets, "template_id", target)

    template_decisions = lucidity_store.setdefault("template_decisions", {})
    for template_id, counts in LUCIDITY_TEMPLATE_DECISIONS.items():
        bucket = template_decisions.setdefault(template_id, {})
        for decision, count in counts.items():
            bucket[decision] = int(bucket.get(decision, 0)) + count

    ensure_metadata(
        state,
        "speech:general_language",
        "speech_pack",
        source="general_language_bootstrap",
    )


def train_general_language(
    checkpoint: str | Path = DEFAULT_GENERAL_LANGUAGE_CHECKPOINT,
    *,
    episode_count: int | None = None,
    seed: int = 42,
    run_module_train: bool = True,
    registry_name: str | None = None,
) -> dict[str, Any]:
    root = _normalize_slot_path(checkpoint)
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(
            f"checkpoint not found at {root} — run ingest train first or pass --checkpoint"
        )

    state = load_checkpoint(root, create=False)
    bootstrap_general_language_stores(state)
    save_checkpoint(state, root, force=True, step_delta=1)

    pack_dir = resolve_train_path("data/generated/general_language")
    social_count = episode_count or GENERAL_LANGUAGE_COUNTS.get("chat_social", 120)
    paraphrase_count = GENERAL_LANGUAGE_COUNTS.get("chat_qa_paraphrase", 200)
    social_episodes = generate("chat_social", social_count, seed=seed)
    paraphrase_episodes = generate("chat_qa_paraphrase", paraphrase_count, seed=seed + 1)
    episodes = [*social_episodes, *paraphrase_episodes]
    write_episodes(social_episodes, pack_dir / "chat_social.jsonl")
    write_episodes(paraphrase_episodes, pack_dir / "chat_qa_paraphrase.jsonl")
    write_episodes(episodes, pack_dir / "all.jsonl")

    checkpoint_arg = str(checkpoint)
    module_runs: dict[str, Any] = {}
    if run_module_train:
        train_modules = ("lucidity", "decoder", "dmf", "cue_encoder", "binding")
        steps = str(min(len(episodes), 80))
        for module in train_modules:
            cmd = [
                sys.executable,
                "-m",
                "lucid.training.cli",
                module,
                "--checkpoint",
                checkpoint_arg,
                "--episodes",
                str(pack_dir / "all.jsonl"),
                "--steps",
                steps,
                "--no-save",
            ]
            completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
            module_runs[module] = {
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout.strip()[-400:],
                "stderr_tail": completed.stderr.strip()[-400:],
            }

    save_checkpoint(load_checkpoint(root, create=False), root, force=True, step_delta=1)
    summary = checkpoint_summary(load_checkpoint(root, create=False))
    name = registry_name or (
        str(checkpoint) if isinstance(checkpoint, str) and "/" not in str(checkpoint).replace("\\", "/") else DEFAULT_GENERAL_LANGUAGE_CHECKPOINT
    )
    archived = register_checkpoint(
        name=name,
        path=root,
        label="articles + general language + paraphrase binding",
        command="lucid.training.general_language",
        summary=summary,
    )
    return {
        "checkpoint": str(root),
        "archived": archived,
        "episodes": len(episodes),
        "social_episodes": len(social_episodes),
        "paraphrase_episodes": len(paraphrase_episodes),
        "episode_pack": str(pack_dir / "all.jsonl"),
        "module_runs": module_runs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap general conversational language checkpoint")
    parser.add_argument("--checkpoint", default=DEFAULT_GENERAL_LANGUAGE_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-module-train", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            train_general_language(
                args.checkpoint,
                episode_count=args.episodes,
                seed=args.seed,
                run_module_train=not args.skip_module_train,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
