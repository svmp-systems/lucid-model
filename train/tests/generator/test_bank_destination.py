from lucid.training.generator.engine import AmbiguityKnob, check_episode, rng_for_seed
from lucid.training.generator.recipes import bank_destination


def test_low_ambiguity_stays_open():
    episode = bank_destination.make(rng_for_seed(1), AmbiguityKnob(0.1))
    assert episode.gold.lucidity_target == "PRESERVE_AMBIGUITY"
    assert not check_episode(episode)


def test_high_ambiguity_commits():
    episode = bank_destination.make(rng_for_seed(99), AmbiguityKnob(0.9))
    assert episode.gold.lucidity_target == "COMMIT"
    assert episode.gold.expected_answer == "financial"
