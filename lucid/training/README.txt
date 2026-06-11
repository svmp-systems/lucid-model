Lucid training — local artifact tree (gitignored except tests/)

Run tests:
  py -m pytest lucid/training/tests

Artifact root (tree/):
  checkpoints/training/   mutable weights (`lucid train` writes here)
  checkpoints/loaded/     pinned inference save (`lucid checkpoint load` / `lucid train --pin`)
  checkpoints/saves/        standard archives cp_001, cp_002, … (auto after each train run)
  checkpoints/saves/registry.json   human-readable save index
  data/generated/         episode JSONL (`lucid-gen pack`)
  audit/runs/training/    training step audits
  audit/runs/pipeline/    pipeline run audits
  audit/runs/smoke/       CLI smoke audits
  audit/scaling/          cost/quality points.jsonl

Regenerate phase-1 pack:
  py -c "from lucid.training.corpus.cli import main; main(['pack','phase1','--out','tree/data/generated/phase1'])"

Never git-add tree/checkpoints tree/audit tree/data — or repo-root audit/ train/ checkpoints/.
