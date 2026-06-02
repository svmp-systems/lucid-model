Lucid training artifacts (local only — not committed except this file)

  checkpoints/local/     default weights (lucid train, lucid run --checkpoint)
  audit/runs/smoke/      CLI smoke audits per module (readable folder names)
  audit/runs/training/   lucid train module runs (steps.jsonl + snapshots)
  audit/runs/pipeline/   lucid run full pipeline audits
  audit/scaling/         cost/quality points.jsonl (secrets redacted on write)
  data/generated/        lucid-gen output

Default checkpoint: train/checkpoints/local
