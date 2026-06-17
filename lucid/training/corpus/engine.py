"""Generation engine: ambiguity knob, recipes, checks."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Protocol

from lucid.ir.common import Modality
from lucid.ir.training import Episode

# --- Ambiguity knob (0 = max ambiguous, 1 = clear) ---


@dataclass(frozen=True, slots=True)
class AmbiguityKnob:
    level: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.level <= 1.0:
            raise ValueError(f"ambiguity knob level must be in [0, 1], got {self.level}")


@dataclass(frozen=True, slots=True)
class ClarityBand:
    """Sample range for the knob, with a relative weight."""

    low: float
    high: float
    weight: int = 1


PHASE1_BANDS: tuple[ClarityBand, ...] = (
    ClarityBand(0.0, 0.33, weight=20),
    ClarityBand(0.33, 0.66, weight=20),
    ClarityBand(0.66, 1.0, weight=20),
)

DEFAULT_BANDS: tuple[ClarityBand, ...] = (
    ClarityBand(0.0, 0.3, weight=30),
    ClarityBand(0.3, 0.7, weight=40),
    ClarityBand(0.7, 1.0, weight=30),
)


def sample_knob(rng: Random, band: ClarityBand) -> AmbiguityKnob:
    return AmbiguityKnob(level=rng.uniform(band.low, band.high))


def parse_band_weights(spec: str) -> tuple[ClarityBand, ...]:
    parts = [int(p.strip()) for p in spec.split(":")]
    if len(parts) != 3:
        raise ValueError("band weights must look like 30:40:30")
    return (
        ClarityBand(0.0, 0.3, weight=parts[0]),
        ClarityBand(0.3, 0.7, weight=parts[1]),
        ClarityBand(0.7, 1.0, weight=parts[2]),
    )


def pick_band(rng: Random, bands: tuple[ClarityBand, ...]) -> ClarityBand:
    return rng.choices(list(bands), weights=[b.weight for b in bands], k=1)[0]


def rng_for_seed(seed: int) -> Random:
    return Random(seed)


# --- Checks (before recipes load) ---

_COMPETING_TRACES = (("financial_action_like", "river_location_like"),)
_GRID_COMPETING = (("position_shift_like", "recolor_like"),)


def _nonzero_cells(grid: list[list[int]]) -> list[tuple[int, int, int]]:
    return [(r, c, v) for r, row in enumerate(grid) for c, v in enumerate(row) if v != 0]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _grid_reachability_errors(episode: Episode) -> list[str]:
    errors: list[str] = []
    raw = episode.raw_input
    if not isinstance(raw, dict):
        return errors
    inp = raw.get("input")
    out = raw.get("output") or episode.gold.expected_answer
    if not isinstance(inp, list) or not isinstance(out, list):
        return []

    in_cells = _nonzero_cells(inp)
    out_cells = _nonzero_cells(out)
    if len(in_cells) != 1 or len(out_cells) != 1:
        errors.append(
            f"grid expects exactly one nonzero cell in/out (got {len(in_cells)}/{len(out_cells)})"
        )
        return errors

    (in_r, in_c, in_v), (out_r, out_c, out_v) = in_cells[0], out_cells[0]
    in_pos, out_pos = (in_r, in_c), (out_r, out_c)
    template = episode.template_id or ""

    if template == "grid_move":
        if in_v != out_v:
            errors.append("grid_move requires preserved color between input and output")
        elif in_pos == out_pos:
            errors.append("grid_move requires a position change")
        else:
            meta = episode.meta or {}
            direction = str(meta.get("direction") or "")
            steps = int(meta.get("steps") or 0)
            distance = _manhattan(in_pos, out_pos)
            if steps and distance != steps:
                # steps is requested stride; walls can shorten the actual move
                if distance > steps or distance == 0:
                    errors.append(
                        f"grid_move steps={steps} but manhattan distance={distance}"
                    )
            if direction == "left" and not (out_c < in_c and out_r == in_r):
                errors.append("grid_move direction=left inconsistent with cell delta")
            elif direction == "right" and not (out_c > in_c and out_r == in_r):
                errors.append("grid_move direction=right inconsistent with cell delta")
            elif direction == "up" and not (out_r < in_r and out_c == in_c):
                errors.append("grid_move direction=up inconsistent with cell delta")
            elif direction == "down" and not (out_r > in_r and out_c == in_c):
                errors.append("grid_move direction=down inconsistent with cell delta")

    if template == "grid_recolor":
        if in_pos != out_pos:
            errors.append("grid_recolor requires fixed cell position")
        elif in_v == out_v:
            errors.append("grid_recolor requires a color change")

    return errors


class CheckError(Exception):
    pass


def check_episode(episode: Episode) -> list[str]:
    errors: list[str] = []
    gold = episode.gold

    for span in gold.spans:
        if not span.surface.strip():
            errors.append(f"empty surface on span {span.span_id}")

    span_ids = {s.span_id for s in gold.spans}
    region_ids = {r.region_id for r in gold.regions}
    frame_ids = {f.frame_id for f in gold.frame_targets}
    valid_scope_frames = frame_ids | region_ids

    for assignment in gold.scope_assignments:
        if assignment.span_id and assignment.span_id not in span_ids:
            errors.append(f"scope points at missing span {assignment.span_id}")
        if assignment.primary_frame and valid_scope_frames and assignment.primary_frame not in valid_scope_frames:
            errors.append(
                f"scope primary_frame {assignment.primary_frame!r} not in frame_targets or regions"
            )
        for secondary in assignment.secondary_frames:
            if valid_scope_frames and secondary not in valid_scope_frames:
                errors.append(f"scope secondary_frame {secondary!r} not in frame_targets or regions")

    for gate in gold.interference_gates:
        if gate.scope_frame_id and valid_scope_frames and gate.scope_frame_id not in valid_scope_frames:
            errors.append(
                f"gate scope_frame_id {gate.scope_frame_id!r} not in frame_targets or regions"
            )

    weights = {t.trace_family: t.weight for t in gold.trace_activations}
    competing_pairs = _COMPETING_TRACES
    if episode.modality in (Modality.GRID, "grid"):
        competing_pairs = (*_COMPETING_TRACES, *_GRID_COMPETING)
    for a, b in competing_pairs:
        total = weights.get(a, 0.0) + weights.get(b, 0.0)
        if total > 1.0 + 1e-6:
            errors.append(f"{a} + {b} weights sum to {total:.3f}")

    top = max((t.weight for t in gold.trace_activations), default=0.0)
    if gold.lucidity_target == "COMMIT" and top <= 0.6:
        errors.append(f"COMMIT needs top trace weight > 0.6 (got {top:.3f})")
    if gold.lucidity_target == "PRESERVE_AMBIGUITY" and top > 0.6:
        errors.append(f"PRESERVE_AMBIGUITY needs top weight ≤ 0.6 (got {top:.3f})")

    if episode.modality in (Modality.GRID, "grid"):
        raw = episode.raw_input
        if not isinstance(raw, dict):
            errors.append("grid input must be a dict")
        else:
            inp, out = raw.get("input"), raw.get("output") or gold.expected_answer
            if not isinstance(inp, list) or not isinstance(out, list):
                errors.append("grid input/output must be 2d arrays")
            elif len(inp) != len(out):
                errors.append("grid height mismatch")
            else:
                errors.extend(_grid_reachability_errors(episode))

    return errors


def check_batch(episodes: list[Episode]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for episode in episodes:
        if episode.episode_id in seen:
            errors.append(f"duplicate id {episode.episode_id}")
        seen.add(episode.episode_id)
        for msg in check_episode(episode):
            errors.append(f"{episode.episode_id}: {msg}")
    return errors


def require_valid(episode: Episode) -> None:
    errors = check_episode(episode)
    if errors:
        raise CheckError(f"{episode.episode_id}: {'; '.join(errors)}")


# --- Recipe registry ---


class Recipe(Protocol):
    NAME: str
    MODALITY: str

    def make(self, rng: Random, knob: AmbiguityKnob) -> Episode: ...


_recipes: dict[str, Recipe] | None = None


def _load_recipes() -> dict[str, Recipe]:
    from lucid.training.corpus.recipes import (
        bank_destination,
        chat_qa_paraphrase,
        chat_social,
        grid_move,
        grid_recolor,
        scoped_instruction,
        two_events,
    )

    modules = (
        bank_destination,
        chat_qa_paraphrase,
        chat_social,
        two_events,
        scoped_instruction,
        grid_move,
        grid_recolor,
    )
    return {module.NAME: module for module in modules}


def recipes() -> dict[str, Recipe]:
    global _recipes
    if _recipes is None:
        _recipes = _load_recipes()
    return _recipes


PHASE1_RECIPES: tuple[str, ...] = (
    "bank_destination",
    "two_events",
    "scoped_instruction",
    "grid_move",
    "grid_recolor",
)

PHASE1_COUNTS: dict[str, int] = {
    "bank_destination": 100,
    "two_events": 100,
    "scoped_instruction": 100,
    "grid_move": 60,
    "grid_recolor": 60,
}


def get_recipe(name: str) -> Recipe:
    key = name.strip()
    catalog = recipes()
    if key not in catalog:
        known = ", ".join(sorted(catalog))
        raise KeyError(f"unknown recipe {name!r}; known: {known}")
    return catalog[key]


def list_recipes(*, modality: str | None = None) -> list[str]:
    catalog = recipes()
    names = sorted(catalog)
    if modality is None:
        return names
    return [n for n in names if catalog[n].MODALITY == modality]


def generate(
    recipe_name: str,
    count: int,
    *,
    seed: int = 42,
    bands: tuple[ClarityBand, ...] = PHASE1_BANDS,
) -> list[Episode]:
    recipe = get_recipe(recipe_name)
    rng = rng_for_seed(seed)
    episodes: list[Episode] = []
    per_band = count // len(bands)
    remainder = count % len(bands)

    for band_index, band in enumerate(bands):
        n = per_band + (1 if band_index < remainder else 0)
        for _ in range(n):
            knob = sample_knob(rng, band)
            episode = recipe.make(rng, knob)
            episode.seed = seed
            episodes.append(episode)

    return episodes
