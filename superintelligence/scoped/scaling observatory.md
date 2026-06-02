# Scaling observatory

Scoped architecture spec for lucid-model **cost–quality tracking**, **internal scaling**, and **frontier-facing comparison**.

This is not frontier-lab “parameters vs loss” scaling. It is:

```
When we spend more (time, GPU, $, data, memory, sparsity),
    what improves?
    what does each success cost?
    where do we plateau or fall off?
    how do we compare on public benchmarks without fake precision?
```

Aligned with:

```
build.md — Phase 3 go/no-go, Phase 4 spend only after measured gain
training/training orchestrator.md — capability gained per update dollar
training/training quantization.md — NO_UPDATE on high-margin success
```

Implementation: `lucid/audit/scaling.py` (Layer 1–2). Data: `audit/scaling/points.jsonl` (`LUCID_SCALING_DIR`). CLI: `lucid scaling summary|export|path`. Frontier template: `audit/scaling/frontier_reference.csv`.

---

## Document structure

| Section | Purpose |
|---------|---------|
| **§1 Role** | What the observatory is and is not |
| **§2 Implementation — what to track** | Raw fields, hooks, storage — **build this** |
| **§3 Post-implementation — inference from data** | Summaries, curves, graphs, frontier comparison — **derive after logging** |
| **§4 Training modes & scale_id** | Never mix curves without tags |
| **§5 Plateau & fall off** | Rules inferred from trends |
| **§6 Build order** | Phases vs deliverables |
| **§7 Anti-patterns** | Misleading charts and comparisons |

**Rule:** Implementation only **appends facts**. Graphs, scaling “laws,” and frontier points on charts are **computed later** from those facts (and optional external benchmark tables).

---

## §1 Role

The scaling observatory is a **receipt log + query/export layer**. It does **not** train, patch, or commit lucidity decisions.

```
Implementation:   run finished → append ScalingPoint (and optional CampaignManifest)
Post-implementation:  aggregate → summarize → plot → write PHASE3_GO_NO_GO snippets
```

Use for:

- **Mode A** — module calibration vs generator gold
- **Mode B** — stacked stages (real upstream + module under test)
- **Mode C** — full pipeline + orchestrator
- **Inference** — chat/CLI/API turns (cost per turn, latency)
- **Eval campaigns** — Arc, BabyLM, private holdout (benchmark score vs campaign GPU-h / $)

**Never** plot Mode A and Mode C on one curve without `training_mode` and `module_under_test` labels.

---

## §2 Implementation — what to track

Everything in this section is **written at runtime** (Layer 1). No graphs required.

### 2.1 Core record: `ScalingPoint`

Append one row per logical unit. Choose unit by event type:

| Event type | One ScalingPoint per |
|------------|----------------------|
| Pipeline episode run | `run_id` |
| Orchestrator training step | `orchestrator_step_id` |
| Module trainer batch/epoch | `trainer_step_id` |
| Benchmark eval job | `eval_job_id` (or per-task if cheap) |
| Chat/API turn | `session_id` + `turn_index` |

```text
ScalingPoint {
    // --- Identity (required) ---
    point_id                    // uuid
    timestamp_utc
    scale_id                    // see §4
    event_type                  // pipeline_run | orchestrator_step | trainer_step | benchmark_eval | inference_turn
    training_mode               // calibrate | stack | full_loop | inference_only | benchmark_only
    module_under_test           // perception | cue_encoder | dmf | … | pipeline | none
    run_kind                    // train | eval | inference | shadow | diagnostic

    episode_id                  // optional
    run_id                      // optional
    session_id                  // optional
    turn_index                  // optional
    corpus_slice                // phase1_fixtures | chat_7k | arc_holdout | bank_destination | …
    build_phase                 // 1 | 2 | 3 | 4 | 5
    version_tag                 // e.g. v0.2.0-chat, git describe

    // --- Quality (record what applies; null if N/A) ---
    validator_success           // bool
    validator_score             // float 0–1
    validator_name              // exact_sense | exact_grid | …
    module_gold_score           // float 0–1 — Mode A aggregate for this step
    module_gold_detail          // json — per-metric breakdown (cue recall@k, span F1, …)
    lucidity_decision           // commit | preserve_ambiguity | …
    lucidity_false_commit       // bool — post-hoc or validator contradiction
    lucidity_false_reject       // bool
    decoder_faithfulness        // float 0–1 — render vs committed IR
    benchmark_name              // arc_agi_2 | arc_agi_3 | babylm | …
    benchmark_split             // public | private_holdout | official_submit
    benchmark_score             // task-specific (%, accuracy, etc.)
    benchmark_tasks_total       // int
    benchmark_tasks_succeeded   // int
    benchmark_attempts          // int — if multi-attempt protocol
    audit_complete              // bool — full pipeline audit without crash
    capability_matrix_checks    // json — optional Phase 3 checklist pass/fail

    // --- Orchestrator / learning (Mode C; null in pure inference) ---
    governor_action             // NO_UPDATE | UPDATE | …
    update_level                // 0–10 ladder
    primary_blame_module        // string
    patch_promoted              // bool
    patch_rejected              // bool
    failure_replay_cleared      // bool — 3-streak + shadow rules met
    retention_canary_pass       // bool
    shadow_episodes_run         // int
    consecutive_successes       // int — on replay entry

    // --- Cost / compute (required where measurable) ---
    wall_time_ms                // float
    gpu_seconds                 // float — 0 if CPU-only
    cpu_seconds                 // float — optional
    llm_prompt_tokens           // int
    llm_completion_tokens       // int
    llm_cost_usd                // float — computed from price table + tokens
    cloud_cost_usd              // float — optional explicit billing
    hardware_class              // laptop_3050ti | a100_80gb | …
    cloud_provider              // local | aws | …

    // --- Sparsity / pipeline heaviness ---
    active_trace_count
    active_basin_count
    candidate_frame_count
    retrieval_budget
    projector_rollout_count
    lucidity_iteration_count
    stage_times_ms              // json — per-stage breakdown

    // --- Memory / scale context ---
    tracebank_size              // int — count at run start
    basin_count
    tracebank_snapshot_id
    episodes_in_corpus          // corpus version size
    episodes_seen_lifetime      // optional cumulative counter
    config_hash                 // retrieval, stubs, registry, budgets
    git_sha

  provenance                    // adapter versions, backend (rule|llm), notes
}
```

**Do not compute** `cost_per_success`, `score_per_gpu_hour`, or frontier comparison fields at append time — those belong in §3.

### 2.2 Campaign record: `CampaignManifest`

One manifest per **benchmark campaign** or **training campaign** (not per episode). Links many `ScalingPoint` rows.

```text
CampaignManifest {
    campaign_id
    campaign_type               // benchmark | module_train | full_loop_train | ablation
    benchmark_name              // optional
    version_tag
    build_phase
    started_at
    finished_at

    protocol_doc                // path or url — holdout rules, seeds, attempt limits
    hardware_class
    total_gpu_seconds           // sum from points or job scheduler
    total_llm_cost_usd
    total_wall_time_ms

    checkpoint_ids              // list
    config_hash
    git_sha

    notes
}
```

### 2.3 Tracking categories (implementation checklist)

Implement hooks so these categories are coverable. Each maps to `ScalingPoint` fields above.

#### A — Internal efficiency (lucid-only; not frontier-comparable)

| Track | Fields | Hook |
|-------|--------|------|
| Module gold | `module_gold_score`, `module_gold_detail` | `lucid/training/trainers/*` after calibrate |
| Validator E2E | `validator_success`, `validator_score` | pipeline run + orchestrator |
| Orchestrator learning | `governor_action`, `patch_*`, `shadow_*`, blame | `TrainingOrchestrator.run_one_step` |
| Sparsity cost | `active_trace_count`, `retrieval_budget`, `projector_rollout_count` | `PipelineRun` / `RunLog` |
| Memory scale | `tracebank_size`, `snapshot_id` | trace/basin store at run start |
| Quality of gate | `lucidity_false_commit`, `lucidity_false_reject` | validator vs lucidity post-hoc |
| Decoder fidelity | `decoder_faithfulness` | re-parse or claim check vs `CommittedState` |
| Scope leaks | tag in `provenance` or dedicated bool | tests / manual audit sample |
| Retention | `retention_canary_pass` | shadow evaluator |
| Replay health | `failure_replay_cleared`, `consecutive_successes` | `FailureReplayStore` |

#### B — Frontier-comparable (public scoreboard)

| Track | Fields | Hook |
|-------|--------|------|
| Benchmark score | `benchmark_*` | eval scripts Phase 3+ |
| Audit coverage | `audit_complete` | pipeline runner — no silent crash |
| Reproducibility | `config_hash`, `git_sha`, seed in `CampaignManifest` | campaign + fixed episode pack |
| Attempts protocol | `benchmark_attempts` | eval harness |
| **Campaign totals** | `CampaignManifest.total_gpu_seconds`, $ | sum points or job wrapper |

#### C — Differentiation (your architecture proof; not on leaderboards)

| Track | Fields | Hook |
|-------|--------|------|
| Capability matrix | `capability_matrix_checks` | Phase 3 test suite |
| Plural until lucidity | log `lucidity_decision` + basin margins in `provenance` | existing run logs |
| Lazy collapse | audit sample: no early commit | manual or automated audit scan |

#### D — Inference vs training (separate `run_kind`)

| Track | Fields | Hook |
|-------|--------|------|
| Training | `run_kind=train`, orchestrator fields populated | trainers + orchestrator |
| Inference | `run_kind=inference`, `training_mode=inference_only` | `OrchestratorRunner`, `lucid chat`, API |
| Eval | `run_kind=eval`, `benchmark_*` populated | eval CLI |
| Shadow | `run_kind=shadow` | shadow evaluator |

**Separate storage or filter** when plotting — do not mix train GPU-h with inference ms/turn in one “cost efficiency” chart without labeling.

### 2.4 External reference table (implementation: static file, not runtime)

Maintain a **human-curated** file (not auto-generated from secrets):

```text
lucid/audit/scaling/frontier_reference.csv
```

Columns:

```text
system_label,benchmark_name,benchmark_score,benchmark_date,
train_compute_note,train_gpu_hours_est,train_tokens_est,params_est,
inference_cost_note,source_url,confidence  // measured | leaderboard | estimated | order_of_magnitude
```

Updated manually when papers/leaderboards change. **Never** pretend these are measured from your runs.

### 2.5 Implementation layout

```text
lucid/audit/scaling.py        # types, record, extract, summarize, export

audit/scaling/                # runtime data (repo audit tree)
  points.jsonl
  exports/
  frontier_reference.csv
```

### 2.6 Hooks (required)

| Source | When to record |
|--------|----------------|
| `OrchestratorRunner._execute_episode` | End of episode; `event_type=pipeline_run` |
| `TrainingOrchestrator.run_one_step` | After validation + promote/reject; `orchestrator_step` |
| `lucid/training/trainers/*` | End of calibrate batch/epoch; `trainer_step` |
| Benchmark eval CLI | Per campaign + optional per-task; `benchmark_eval` |
| `lucid chat` / API turn | Per turn; `inference_turn` |
| `lucid-gen pack` | Optional single point for corpus build wall time |

Read existing structures — do not duplicate instrumentation:

- `RunLog`, `CostMetrics` (`lucid/ir/training.py`)
- `PipelineRun.cost_metrics`, `stage_records`
- `TrainingOrchestrator.get_status()`, `TrainingGovernor.metrics()`
- `FailureReplayStore.metrics()`

### 2.7 CLI (universal `lucid` entrypoint)

```bash
lucid scaling summary
lucid scaling summary --scale-id cue_encoder:calibrate:bank_destination:abc123
lucid scaling export --out summary_by_scale_id.csv
lucid scaling path
```

Layer 1 works without plots; `export` writes CSV under `audit/scaling/exports/`.

---

## §3 Post-implementation — inference from tracked data

Everything here is **derived** from `ScalingPoint` + `CampaignManifest` + `frontier_reference.csv`. Safe to re-run when formulas change.

### 3.1 Derived metrics (formulas)

Compute in `lucid/audit/scaling.py` (`summarize_points`, `export_summary_csv`).

#### Per window (scale_id + date range)

```text
success_rate           = count(validator_success) / count(points)
gold_mean              = mean(module_gold_score)
cost_per_success       = sum(gpu_seconds) / max(1, count(validator_success))
                        # also wall_time_ms, llm_cost_usd variants
successes_per_gpu_hour = count(validator_success) / (sum(gpu_seconds)/3600)
patches_per_gpu_hour   = count(patch_promoted) / (sum(gpu_seconds)/3600)
no_update_rate         = count(governor_action==NO_UPDATE) / count(orchestrator steps)
replay_queue_depth     = from FailureReplayStore snapshot at window end
```

#### Between two checkpoints (marginal)

```text
delta_score            = benchmark_score_B - benchmark_score_A
delta_gpu_hours        = (sum gpu_seconds)_B - (sum gpu_seconds)_A
score_per_100_gpu_h    = delta_score / delta_gpu_hours * 100
score_per_10k_usd      = delta_score / (delta_usd / 10000)
marginal_gain          = delta_quality / delta_compute   # generic
```

#### Plateau heuristic (advisory)

```text
plateau_detected if:
  last N scale steps: abs(delta_score) < epsilon
  AND sum(additional_gpu_seconds) > plateau_min_compute
```

#### Fall off heuristic (advisory)

```text
falloff_detected if:
  success_rate dropped > falloff_threshold vs previous window
  OR failure_replay_queue_depth rising while patch_promotion_rate high
  OR retention_canary_pass rate dropped
```

### 3.2 Layer 2 — Text summaries (no charts)

Output of `lucid scaling summary`:

```text
scale_id: cue_encoder:calibrate:phase1:abc123
  points: 420  |  gold_mean: 0.91  |  wall_ms p50: 38
  plateau_warning: false

scale_id: pipeline:full_loop:phase1:abc123
  validator_pass: 0.71  |  cost_per_success: 2.1s  |  projector_rate: 0.12
```

### 3.3 Layer 3 — Internal curves (lucid-only)

| Chart ID | X | Y | Filter |
|----------|---|---|--------|
| `L3-module-gold` | episodes trained or trainer step | `module_gold_score` | Mode A, one `module_under_test` |
| `L3-validator-pass` | corpus size or campaign day | `success_rate` | Mode C |
| `L3-cost-per-success` | build_phase or week | `cost_per_success` | Mode C |
| `L3-orchestrator` | week | `patches_per_gpu_hour`, `no_update_rate` | orchestrator steps |
| `L3-plateau` | cumulative gpu_seconds | benchmark or validator score | one `scale_id` |

Fit (optional, Phase 3+): log-linear in episodes or power-law in compute — **advisory**, report R² and wide confidence bands.

### 3.4 Publication charts — frontier comparison

Use **only** for storytelling where shared benchmark exists. Always footnote frontier points.

#### Chart P1 — Score vs total spend (primary “cheaper” chart)

```text
X: total_gpu_hours OR total_usd (log scale) for campaign
Y: benchmark_score (single task per chart)
Points:
  - lucid: from CampaignManifest + aggregate (measured)
  - frontier: from frontier_reference.csv (estimated / leaderboard)
```

**Caption template:**

```text
Lucid points: measured train+eval GPU-hours for stated version.
Frontier points: public leaderboard scores; training compute estimated (see appendix).
```

#### Chart P2 — Marginal efficiency (bar)

```text
Bars: score_per_100_gpu_h between consecutive lucid checkpoints
Optional: single estimated bar for a named external system (wide error bar or "order of magnitude")
```

#### Chart P3 — Campaign learning curve

```text
X: cumulative gpu_hours within one Arc campaign
Y: holdout score
Only lucid measured points — do not draw frontier training curve (unknown)
Optional: horizontal band = public SOTA range from leaderboard
```

#### Chart P4 — Inference cost (separate figure)

```text
X: ms per turn OR $ per 1000 turns
Y: task success or user-facing quality proxy
Compare: lucid local run vs API baseline (your measured token usage × public $/token)
Title must say INFERENCE not TRAINING
```

#### Chart P5 — Differentiation dashboard (not vs frontier)

Capability matrix pass rate, audit coverage %, false commit rate — internal only.

### 3.5 What frontier labs do *not* give you

Do not expect public data for:

- Per-module gold recall curves
- Cost per promoted patch
- Your `ScalingPoint` schema

**Do expect** public benchmark scores; occasional order-of-magnitude train compute in papers; API $ for inference proxies.

Your graphs are **more transparent on lucid cost-to-score** than most labs; frontier dots on P1/P2 are **estimated**, not copied from their internal logs.

### 3.6 Export pipeline (post-implementation)

```text
audit/scaling/points.jsonl
        │
        └─► lucid scaling export ──► audit/scaling/exports/*.csv
                                      (plot externally; frontier_reference.csv for comparison rows)
```

Inputs to Phase 3 doc generator:

```text
PHASE3_GO_NO_GO.md ← template + aggregate metrics + plateau/falloff flags + P1 chart path
```

### 3.7 Minimum points before claiming a “scaling law”

| Claim | Minimum data |
|-------|----------------|
| Module plateau | ≥3 episode scales on same `scale_id`, gold delta &lt; ε on last two |
| Benchmark marginal gain | ≥3 cumulative GPU levels on same benchmark protocol |
| Cheaper than frontier | ≥1 measured lucid campaign + sourced frontier_reference row + log-scale X |
| Phase 4 spend go | `build.md` thresholds + plateau not detected + falloff false |

---

## §4 Training modes & `scale_id`

From `build.md`:

| Mode | `training_mode` | `module_under_test` |
|------|-------------------|---------------------|
| A — Calibrate | `calibrate` | `perception`, `cue_encoder`, … |
| B — Stack | `stack` | module under test |
| C — Full loop | `full_loop` | `pipeline` |
| Inference only | `inference_only` | `pipeline` or `none` |
| Benchmark eval | `benchmark_only` | `pipeline` |

```text
scale_id = {module_under_test}:{training_mode}:{corpus_slice}:{config_hash}
```

`config_hash` changes when: retrieval budget, tracebank snapshot, stub versions, seed policy, trainer hyperparameters.

---

## §5 Plateau vs fall off (inferred in §3)

### Plateau

Spend more; quality barely moves. **Action:** stop scaling that module/corpus; move bottleneck.

### Fall off

Spend more; quality drops or replay/retention worsens. **Action:** fix orchestrator/memory; do not increase Phase 4 budget (`build.md`).

Detection: §3.1 heuristics on aggregated windows — not single-run flags.

---

## §6 Build order vs `build.md`

| Phase | Implementation (§2) | Post-implementation (§3) |
|-------|---------------------|---------------------------|
| First training | `ScalingPoint` append + hooks on runner + one trainer | `scaling summary` text only |
| Phase 1 done | All trainers + orchestrator step | `L3-module-gold`, cost per success baseline |
| Phase 2 | Inference turn points; LLM $ | P4 inference chart optional |
| Phase 3 | `CampaignManifest`; benchmark fields | P1, P2, P3; `PHASE3_GO_NO_GO` export; frontier_reference.csv |
| Phase 4 | Memory tier tags | Cost vs tier; governor no-update vs $ |

Do **not** block Phase 1 pipeline on plots or frontier CSV.

---

## §7 Anti-patterns

### Implementation

- Storing derived metrics in JSONL without raw fields (can't recompute)
- Omitting `config_hash` / `scale_id`
- Mixing train and inference in one point without `run_kind`
- Auto-scraping frontier training logs as “measured”

### Post-implementation / graphs

- Lucid module gold vs frontier MMLU on same axes
- Linear X from 1 GPU-h to frontier pretrain FLOPs without log scale
- Training $ for lucid vs API inference $ for frontier on one chart
- Frontier points without `confidence` / source in appendix
- “10,000× cheaper” without shared benchmark + sourced estimates
- Plotting Mode A and Mode C as one line

### Comparison narrative

**Defensible:** “Arc holdout X% after Y measured GPU-hours; published systems at similar scores are commonly associated with far larger training compute (see frontier_reference.csv).”

**Indefensible:** “We beat GPT-5 on cost” without shared task and sourced compute.

---

## §8 Relation to other specs

| Spec | Relationship |
|------|----------------|
| `training/training orchestrator.md` | Orchestrator fields, replay clear, retention canary |
| `training/training quantization.md` | Log `NO_UPDATE` for true cost per learning event |
| `memory quantization.md` | `tracebank_snapshot_id`, tier changes |
| `build.md` | Phase 3 matrix + benchmark gates consume §3 exports |

---

## §9 Summary

```text
IMPLEMENTATION (track):
  ScalingPoint per run/step/turn/campaign
  CampaignManifest per benchmark/train campaign
  frontier_reference.csv (manual, external)
  Hooks: runner, orchestrator, trainers, eval CLI
  Store: audit/scaling/points.jsonl

POST-IMPLEMENTATION (infer):
  Aggregates: cost_per_success, score_per_100_gpu_h, plateau/falloff flags
  Internal charts: L3-* (module, validator, orchestrator)
  Public charts: P1 score vs spend, P2 marginal bars, P3 campaign curve, P4 inference separate
  Outputs: PHASE3_GO_NO_GO snippets, model card efficiency appendix

North star: capability gained per update dollar (train) and score per GPU-hour (eval campaigns)
```

The observatory separates **honest measurement** (implementation) from **interpretation and comparison** (graphs and docs derived later).
