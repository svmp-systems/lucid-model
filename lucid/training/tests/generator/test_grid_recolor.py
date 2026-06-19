from lucid.training.corpus.engine import AmbiguityKnob, check_episode, rng_for_seed
from lucid.training.corpus.recipes import grid_recolor


def test_grid_recolor_full_gold():
    episode = grid_recolor.make(rng_for_seed(2), AmbiguityKnob(0.8))
    gold = episode.gold
    assert gold.spans
    assert gold.frame_targets
    assert gold.scope_assignments
    assert gold.basin_families
    assert not check_episode(episode)
