"""Phase-1 generator gold must supervise all module trainers."""

from __future__ import annotations

from lucid.training.generator.engine import (
    PHASE1_BANDS,
    AmbiguityKnob,
    check_batch,
    generate,
    rng_for_seed,
    sample_knob,
)
from lucid.training.generator.output import build_phase1_pack
from lucid.training.generator.recipes import (
    bank_destination,
    grid_move,
    grid_recolor,
    scoped_instruction,
)


def _has_gold(episode, field: str) -> bool:
    gold = episode.gold
    return {
        "spans": bool(gold.spans),
        "scope": bool(gold.scope_assignments),
        "frames": bool(gold.frame_targets),
        "basins": bool(gold.basin_families),
        "gates": bool(gold.interference_gates),
        "traces": bool(gold.trace_activations),
    }[field]


def test_phase1_pack_has_full_module_gold(tmp_path):
    episodes = build_phase1_pack(tmp_path, seed=42)
    assert len(episodes) == 420
    assert not check_batch(episodes)

    for field in ("spans", "scope", "frames", "basins", "traces"):
        missing = [ep.template_id for ep in episodes if not _has_gold(ep, field)]
        assert not missing, f"missing {field} for recipes: {set(missing)}"

    missing_gates = [ep.template_id for ep in episodes if not _has_gold(ep, "gates")]
    assert not missing_gates, f"missing gates for recipes: {set(missing_gates)}"


def test_phase1_bands_cover_ambiguity_range():
    rng = rng_for_seed(99)
    levels: list[float] = []
    for _ in range(500):
        band = rng.choices(list(PHASE1_BANDS), weights=[b.weight for b in PHASE1_BANDS], k=1)[0]
        levels.append(sample_knob(rng, band).level)
    assert min(levels) < 0.05
    assert max(levels) > 0.95
    assert any(0.25 <= level <= 0.4 for level in levels)
    assert any(0.6 <= level <= 0.75 for level in levels)


def test_grid_knob_changes_lucidity():
    clear = grid_move.make(rng_for_seed(1), AmbiguityKnob(0.95))
    fuzzy = grid_move.make(rng_for_seed(1), AmbiguityKnob(0.05))
    assert clear.gold.lucidity_target == "COMMIT"
    assert fuzzy.gold.lucidity_target == "PRESERVE_AMBIGUITY"


def test_recipe_samples_pass_checks():
    rng = rng_for_seed(7)
    for recipe in (bank_destination, scoped_instruction, grid_move, grid_recolor):
        for level in (0.05, 0.5, 0.95):
            episode = recipe.make(rng, AmbiguityKnob(level))
            assert not check_batch([episode])


def test_generate_recipe_batch_valid():
    for name in ("bank_destination", "two_events", "scoped_instruction", "grid_move", "grid_recolor"):
        episodes = generate(name, 30, seed=11)
        assert len(episodes) == 30
        assert not check_batch(episodes)
