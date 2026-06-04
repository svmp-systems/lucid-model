from lucid.training.generator.engine import AmbiguityKnob, rng_for_seed
from lucid.training.generator.output import build_phase1_pack, read_episodes, write_episodes
from lucid.training.generator.recipes import scoped_instruction
from lucid.ir.serde import from_dict, to_dict
from lucid.ir.training import Episode


def test_jsonl_roundtrip(tmp_path):
    episodes = [
        scoped_instruction.make(rng_for_seed(5), AmbiguityKnob(0.5)) for _ in range(3)
    ]
    path = tmp_path / "episodes.jsonl"
    write_episodes(episodes, path)
    restored = read_episodes(path)
    assert len(restored) == 3
    for original, loaded in zip(episodes, restored, strict=True):
        assert from_dict(to_dict(original), Episode) == loaded


def test_phase1_pack(tmp_path):
    episodes = build_phase1_pack(tmp_path, seed=42)
    assert len(episodes) == 420
    assert (tmp_path / "all.jsonl").exists()
