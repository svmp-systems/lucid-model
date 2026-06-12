from lucid.training.corpus.engine import AmbiguityKnob, check_episode, rng_for_seed
from lucid.training.corpus.recipes import scoped_instruction


def test_scoped_instruction_full_gold():
    episode = scoped_instruction.make(rng_for_seed(3), AmbiguityKnob(0.85))
    gold = episode.gold
    assert gold.frame_targets
    assert gold.basin_families
    assert gold.interference_gates
    assert gold.scope_assignments
    assert not check_episode(episode)
