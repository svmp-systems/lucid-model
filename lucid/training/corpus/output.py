"""Write and read generated episodes."""

from __future__ import annotations

from pathlib import Path

from lucid.ir.serde import from_json, to_json
from lucid.ir.training import Episode
from lucid.training.corpus.engine import (
    PHASE1_COUNTS,
    PHASE1_RECIPES,
    check_batch,
    generate,
)


def write_episodes(episodes: list[Episode], path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(to_json(episode, indent=None))
            handle.write("\n")


def read_episodes(path: Path | str) -> list[Episode]:
    episodes: list[Episode] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                episodes.append(from_json(line, Episode))
    return episodes


def build_phase1_pack(out_dir: Path | str, *, seed: int = 42) -> list[Episode]:
    """420 episodes: 300 text + 120 grid, split across clarity bands."""
    root = Path(out_dir)
    all_episodes: list[Episode] = []

    for index, name in enumerate(PHASE1_RECIPES):
        batch = generate(name, PHASE1_COUNTS[name], seed=seed + index * 1000)
        write_episodes(batch, root / f"{name}.jsonl")
        all_episodes.extend(batch)

    errors = check_batch(all_episodes)
    if errors:
        raise RuntimeError("phase1 pack failed checks:\n" + "\n".join(errors[:20]))

    write_episodes(all_episodes, root / "all.jsonl")
    (root / "readme.txt").write_text(
        "\n".join(
            [
                "Lucid phase-1 generated episodes",
                f"total: {len(all_episodes)}",
                f"seed: {seed}",
                "",
                "per recipe:",
                *[f"  {name}: {PHASE1_COUNTS[name]}" for name in PHASE1_RECIPES],
            ]
        ),
        encoding="utf-8",
    )
    return all_episodes


GENERAL_LANGUAGE_COUNTS: dict[str, int] = {
    "chat_social": 120,
    "chat_qa_paraphrase": 200,
}


def build_general_language_pack(out_dir: Path | str, *, seed: int = 42) -> list[Episode]:
    """General conversational language episodes for social speech training."""
    root = Path(out_dir)
    all_episodes: list[Episode] = []

    for index, (name, count) in enumerate(GENERAL_LANGUAGE_COUNTS.items()):
        batch = generate(name, count, seed=seed + index * 1000)
        write_episodes(batch, root / f"{name}.jsonl")
        all_episodes.extend(batch)

    errors = check_batch(all_episodes)
    if errors:
        raise RuntimeError("general language pack failed checks:\n" + "\n".join(errors[:20]))

    write_episodes(all_episodes, root / "all.jsonl")
    (root / "readme.txt").write_text(
        "\n".join(
            [
                "Lucid general language episodes",
                f"total: {len(all_episodes)}",
                f"seed: {seed}",
            ]
        ),
        encoding="utf-8",
    )
    return all_episodes
