# Scaling observatory

Scoped architecture spec for lucid-model **cost–quality tracking** across training and inference.

This is not frontier-lab “parameters vs loss” scaling. It is **your** question: when we spend more time, GPU, data, or memory, **what do we get back**, and when does that **stop helping** or **start hurting**?

Aligned with:

```
build.md — Phase 3 go/no-go, Phase 4 spend only after measured gain
training/training orchestrator.md — capability gained per update dollar
training/training quantization.md — NO_UPDATE on high-margin success
```

Implementation target (when built): `lucid/training/scaling/` + hooks in orchestrator runner and training orchestrator. Audits under `lucid/audit/scaling/`.

---

## Role

The scaling observatory is a **receipt log + reporting layer**. It does not train anything. It records every meaningful run so you can answer:

```
When we do more work (episodes, GPU-hours, bigger memory, heavier retrieval),
    what improves?
    what does each success cost?
    where do we plateau (diminishing returns)?
    where do we fall off (regressions, thrashing, pollution)?
```

Use it for **module-by-module calibration** (Mode A), **stacked stages** (Mode B), and **full pipeline + orchestrator** (Mode C) — **never mixed on one curve without labels**.

---

## Plain-language overview

Lucid-model thinks in **steps** (perception → cue → memory → binding → … → answer). Training mostly **fixes the smallest piece that broke**, not one giant backprop through everything.

A scaling observatory is the system’s **notebook**:

| Question | Example answer from the log |
|----------|-----------------------------|
| Did more episodes help cue? | Gold recall 72% → 91% from 1k → 10k episodes, then flat |
| Did more GPU on Arc help? | +3% holdout per 100 GPU-hours until ~400h, then flat |
| Are we wasting money? | Cost per success doubled; patch promotion high but replay queue growing |
| Ready for Phase 4 cloud spend? | Marginal gain per $500/month below threshold in `build.md` |

**Cost-based scaling law** (practical meaning): a **measured trend** you plot after enough points, e.g. “roughly +X% benchmark score per 100 GPU-hours until Y hours, then plateau.” Not a theorem — a **decision input** for `PHASE3_GO_NO_GO.md` and Phase 4 budget.

---

## What we scale (axes)

Frontier labs scale **parameters and tokens**. Lucid-model scales different knobs:

| Axis | Examples |
|------|----------|
| **Data** | 420 → 7k → 150k → 1.5M episodes; synthetic vs verifier mix |
| **Compute** | Laptop CPU ms/run; cloud GPU-seconds; LLM API $ for perception/decoder |
| **Sparsity / cost per run** | `retrieval_budget`, active traces K, basins H, projector on-demand |
| **Memory** | Trace/basin count; hot vs frozen tiers; quantization |
| **Training mode** | Calibrate one module (A) vs stack (B) vs full loop + patches (C) |
| **Quality** | Validator pass; module gold match; benchmark holdout; patches that stay fixed |

North-star metric (from training orchestrator):

```
capability gained per update dollar
```

Also track **successes per GPU-hour** and **cost per successful episode** when GPU or cloud $ is involved.

---

## Training modes — separate curves

From `build.md`:

| Mode | Plain meaning | What scaling measures |
|------|---------------|------------------------|
| **A — Calibrate** | Train or score **one stage** against generator gold; rest stubbed or gold | Learning curve for that module: gold recall, F1, etc. vs episodes or steps |
| **B — Stack** | Real upstream + module under test + stub/gold downstream | Composition curve: does real cue + gold DMF beat stub cue? |
| **C — Full loop** | End-to-end run + validator + orchestrator patches | System curve: validator pass vs corpus size; cost vs success; patch efficiency |

**Rule:** same log file, different `training_mode` and `module_under_test` tags. Do not plot Mode A cue calibration and Mode C full pipeline on one line without labeling.

Recommended grouping key:

```
scale_id = {module_under_test}:{training_mode}:{corpus_version}:{config_hash}
```

`config_hash` must change when retrieval budget, tracebank snapshot, seed policy, or stub versions change — otherwise curves are not comparable.

---

## Cost-based scaling: spend vs return

### What counts as spend (inputs)

| Field | Why |
|-------|-----|
| `wall_time_ms` | Every environment |
| `gpu_seconds` | Cloud / GPU eval |
| `llm_tokens` or `llm_cost_usd` | LLM perception or decoder adapters |
| `episodes_processed` | Data scale |
| `shadow_episodes` | Hidden tax per patch (orchestrator) |
| `active_trace_count`, `active_basin_count` | How “heavy” the cognition pass was |
| `projector_rollout_count` | Expensive consequence testing |
| `lucidity_iterations` | SEARCH_WIDER / RECHECK_BINDING loops |
| `patches_applied`, `shadow_tests_run` | Training churn |

Populate from `RunLog.cost_metrics`, pipeline `CostMetrics`, and orchestrator status (`success_rate`, `patch_promotion_rate`, `failure_replay_queue_depth`).

### What counts as return (outputs)

| Field | Why |
|-------|-----|
| `validator_success`, `validator_score` | End-to-end truth when available |
| `module_gold_match` | Mode A (e.g. cue trace recall vs `TraceTarget`) |
| `benchmark_score` | Arc / BabyLM when run |
| `patch_promoted_and_episode_cleared` | Real fix, not apply-only |
| `retention_canary_pass` | No regression after patch |
| `no_update` (high margin success) | Cheap good outcome — still log it |

### Derived metrics (reporting)

```
cost_per_success        = total_compute_spent / successes
successes_per_gpu_hour  = successes / gpu_seconds * 3600
capability_per_dollar   = delta_quality / delta_spend   (over a window)
marginal_gain           = delta_quality / delta_compute (between two scale steps)
```

---

## Plateau vs fall off

### Plateau (diminishing returns — not necessarily bad)

**Plateau** = you spend more, quality barely moves.

Examples:

- Cue encoder already 95% gold on Phase 1 templates; 10× more episodes on the same recipe adds &lt;1%.
- Arc holdout flat for 200 GPU-hours while corpus still growing.

**Action:** stop pouring compute on that module or corpus slice; move to the next bottleneck (binding, full pipeline, harder templates).

### Fall off (bad — detect early)

**Fall off** = you spend more and things get **worse**.

Examples:

| Symptom | Likely cause |
|---------|----------------|
| Pass rate down while episode count up | Trace/basin pollution; wrong patches promoted |
| `failure_replay_queue_depth` up, `patch_promotion_rate` high | Patches applied but not verified; stale replay |
| Cost per success up, quality flat | Shadow eval or retrieval too heavy without gain |
| Retention canary failures rising | Overfitting patches to target episode only |
| Bigger bank, same K retrieval | Wrong traces crowd out good ones |

**Action:** freeze scale-up; fix orchestrator clear rules, retention suite, or module blame — do not open Phase 4 spend (`build.md`).

---

## Three layers (what to build when)

### Layer 1 — Receipt (build with first real training)

After each inference run, calibration batch, or orchestrator step, append one **ScalingPoint** (JSONL or SQLite).

```
ScalingPoint {
    timestamp
    scale_id
    training_mode              // calibrate | stack | full_loop
    module_under_test          // perception | cue_encoder | … | pipeline

    episode_id
    run_id
    corpus_slice               // phase1_fixtures | chat_7k | arc_holdout | …
    phase                      // 1 | 2 | 3 | 4 | 5

    // Quality
    validator_success
    validator_score
    module_gold_score          // optional Mode A
    lucidity_decision
    benchmark_name             // optional
    benchmark_score              // optional
    patch_promoted
    failure_replay_cleared

    // Cost
    wall_time_ms
    gpu_seconds
    llm_tokens
    llm_cost_usd
    active_trace_count
    active_basin_count
    projector_rollout_count
    shadow_episodes
    updates_this_episode

    // Scale context
    episodes_seen_in_run
    tracebank_size
    basin_count
    retrieval_budget
    tracebank_snapshot_id
    config_hash
    git_sha                      // optional

    provenance
}
```

No fitting. No dashboard required. **Do not block** perception/cue work on Layer 1 — a thin append is enough.

### Layer 2 — Summaries (Phase 1–2)

Periodic human-readable rollups, e.g. CLI `lucid scaling summary --scale-id …`:

```
Last 500 cue-calibrate runs on bank_destination:
  gold recall: 82%  |  avg 38 ms/run  |  0 GPU
Full pipeline on phase1 pack:
  validator pass: 71%  |  avg 2.1 s/run  |  projector 12% of runs
Orchestrator (7d):
  patch promotion: 34%  |  replay queue depth: 12  |  cost per promoted fix: 8 shadow eps
```

### Layer 3 — Curves and laws (Phase 3+ gate)

When you have **several compute levels** on the **same** `scale_id` (e.g. 20h, 100h, 500h GPU on same Arc setup):

- Plot quality vs total GPU-hours or $
- Fit simple trends (log-linear in episodes or power-law in compute) — **advisory only**
- Emit snippets for `PHASE3_GO_NO_GO.md`: marginal gain, plateau estimate, fall-off flags
- Phase 4: compare projected gain to monthly budget cap

Require minimum points per curve (e.g. ≥3 scale steps, ≥100 episodes per step) before auto-recommending spend.

---

## Integration points

```
OrchestratorRunner (inference / episode run)
    → record ScalingPoint at end of _execute_episode

Module trainers (lucid/training/trainers/*)
    → record after each calibrate batch or epoch

TrainingOrchestrator.run_one_step
    → record after validation + governor decision + patch promote/reject

lucid-gen pack / benchmark eval scripts
    → record corpus build and eval campaigns with gpu_seconds
```

Read existing signals; do not duplicate logic:

- `RunLog`, `CostMetrics` (`lucid/ir/training.py`)
- `TrainingOrchestrator.get_status()` metrics
- `TrainingGovernor.metrics()` margin stats
- `FailureReplayStore.metrics()`
- Audit logger run directories

Write artifacts to `lucid/audit/scaling/{scale_id}.jsonl` or central `scaling_points.jsonl` with `scale_id` in each row.

---

## Module vs pipeline — example questions

| Training | Question the observatory answers |
|----------|----------------------------------|
| Cue Mode A | Does gold trace recall keep rising with episodes, or plateau? |
| Perception Mode A | Span/marker quality vs wall time (rule vs LLM $)? |
| Stack Mode B | Real cue + gold DMF → binding pass better than stub? |
| Full Mode C | Validator pass vs 150k corpus; cost per success vs Phase 2 baseline? |
| Orchestrator | Promoted patches per GPU-hour; replay queue stuck? |
| Phase 3 eval | Arc holdout % vs GPU-hours; audit coverage % vs crash rate |

---

## Build order vs `build.md`

| When | Observatory deliverable |
|------|-------------------------|
| **First module training** | Layer 1 append only |
| **Phase 1 complete** | Summaries for phase1 fixtures; baseline cost per success |
| **Phase 2 chat** | Add latency vs session length; LLM decoder $ if used |
| **Phase 3** | Curves for Arc/BabyLM vs GPU-hours; auto-sections for go/no-go |
| **Phase 4** | Cost per success vs memory tier; governor no-update rate vs $ |

Do **not** delay Phase 1 pipeline proof waiting for Layer 3 fits.

---

## Anti-patterns

**Do not use one global “loss” curve.**

There is no single cross-entropy. Use validator pass, gold match, lucidity checks, benchmarks.

**Do not merge Mode A and Mode C without tags.**

Cue calibration looks “better” than full pipeline — that is expected, not contradiction.

**Do not treat patch count like gradient steps.**

Patches are sparse and correlated. Track **promoted fixes that clear failure replay** (patch + episode shadow + 3-streak).

**Do not compare scales with different `config_hash`.**

Changing retrieval budget, tracebank snapshot, or stubs invalidates the curve.

**Do not extrapolate Phase 4 spend from &lt;3 compute points.**

Plateau detection needs multiple measured levels on the same setup.

**Do not block training on dashboards.**

Receipt log first; fitting and UI later.

---

## Relation to other specs

| Spec | Relationship |
|------|----------------|
| `training/training orchestrator.md` | Source of patch, shadow, replay, promotion metrics |
| `training/training quantization.md` | `NO_UPDATE` runs are cheap wins — log them for true cost per **learning** event |
| `memory quantization.md` | Tier promotion changes cost per retrieval — tag `tracebank_snapshot_id` |
| `build.md` | Phase 3 capability matrix + benchmark thresholds consume observatory reports |

---

## Summary

```
Scaling observatory = receipt log + cost/quality reports + (later) curves for spend decisions
Measures:   quality vs compute, data, sparsity, memory — not parameter count
Modes:      separate curves for calibrate (A), stack (B), full loop (C)
Plateau:    spend more, gain little → move on or stop
Fall off:   spend more, get worse → fix before scaling cloud budget
Build:      Layer 1 with first training; Layer 3 for Phase 3 go/no-go
North star: capability gained per update dollar (and successes per GPU-hour)
```

The observatory makes “is scaling spend justified?” a **measured** question instead of a guess.
