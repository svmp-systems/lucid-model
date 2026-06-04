from lucid.training.generator.engine import AmbiguityKnob, rng_for_seed
from lucid.training.generator.recipes import grid_move


def test_grid_move_shape():
    episode = grid_move.make(rng_for_seed(4), AmbiguityKnob(0.5))
    raw = episode.raw_input
    gold = episode.gold
    assert len(raw["input"]) == 6
    assert episode.gold.expected_answer == raw["output"]
    assert gold.spans
    assert gold.frame_targets
    assert gold.scope_assignments
    assert gold.basin_families
