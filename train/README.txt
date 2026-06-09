Lucid training artifacts — LOCAL ONLY (not pushed to GitHub)

Everything under train/ is in .gitignore except this file and train/tests/.
Never: git add train/checkpoints/ train/data/ train/audit/

  tests/                 pytest suite (py -m pytest from repo root)

  checkpoints/local/     weights (lucid train / lucid run --checkpoint train/checkpoints/local)
  data/generated/        episode JSONL (lucid-gen pack)
  audit/runs/training/   training step audits
  audit/runs/pipeline/   pipeline run audits
  audit/runs/smoke/      CLI smoke audits
  audit/scaling/         cost/quality points.jsonl (secrets redacted on write)

Regenerate phase-1 pack:
  py -c "from lucid.training.generator.cli import main; main(['pack','phase1','--out','train/data/generated/phase1'])"
