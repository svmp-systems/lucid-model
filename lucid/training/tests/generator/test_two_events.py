from lucid.training.corpus.engine import AmbiguityKnob, rng_for_seed
from lucid.training.corpus.recipes import two_events


def test_two_events_recipe():
    low = two_events.make(rng_for_seed(2), AmbiguityKnob(0.1))
    high = two_events.make(rng_for_seed(3), AmbiguityKnob(0.9))
    assert low.gold.lucidity_target == "PRESERVE_AMBIGUITY"
    assert high.gold.lucidity_target == "COMMIT"
