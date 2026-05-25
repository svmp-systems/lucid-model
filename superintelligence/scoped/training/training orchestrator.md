The training orchestrator is not “another model.” It is the **cost-control and learning-control system**.

Its job:

```
run episodelog everythingvalidate resultassign blamechoose cheapest updatetest update in shadowpromote only if betterfreeze/quantize stable parts
```

Your cue encoder document already gives the right architectural constraint: cue encoding should preserve ambiguity and produce cue-cloud activation pressure, not final interpretation. The orchestrator’s job is to make every training step obey that principle.

---

# 1. Core principle

```
Do not train the whole system.Train only the smallest responsible part,only when validation says learning is needed,and only promote changes that improve quality/cost/retention.
```

This is how the system stays cheap.

The orchestrator should make **no-update** the default outcome.

```
high-confidence success → no updatelow-margin success → light reinforcementfailure → targeted local repairnovel pattern → quarantinerepeated stable success → quantize/freeze
```

---

# 2. Orchestrator responsibilities

The orchestrator controls seven things:

```
1. Episode selection2. Run execution3. Logging4. Validation5. Blame assignment6. Local update selection7. Shadow testing + promotion
```

It is the thing that prevents your system from becoming “10 trained models.”

---

# 3. Main components

## A. Episode Scheduler

Chooses what to train on next.

Input queues:

```
real_text_chat_queuesynthetic_structure_queueverifiable_task_queuemultimodal_queueagent_episode_queuefailure_replay_queueretention_test_queue
```

Output:

```
TrainingEpisode
```

Sampling rule:

```
prioritize:    unresolved failures    weak basins    high-value new traces    under-tested modules    family-held-out generalization    retention checks
```

Do not sample randomly forever. Train where the system is weak.

### Failure replay queue — entry and **clear** rules

An episode enters `failure_replay_queue` when:

```
validation failed  OR  patch was rejected  OR  patch promoted but episode not yet re-validated
```

**Do not remove an episode because a patch was merely applied.** Applied-but-unverified patches quietly corrupt the learning signal and the replay queue fills with stale failures.

An episode **clears** from `failure_replay_queue` only when **all** of the following are true:

```
1. patch_promoted     — a patch that targeted this episode was promoted to live state
2. shadow_pass        — shadow evaluator passed on this specific episode_id
                       (same input/validator; live_state after promotion)
3. stability_streak   — 3 consecutive successful runs on this episode_id
                       without triggering a new patch proposal for that episode
```

Formal state on each replay entry:

```
FailureReplayEntry {
    episode_id
    run_log_id_last_failure
    patch_ids_applied[]          // promoted patches tied to this episode
    shadow_passed: bool           // episode-specific re-run after last promote
    consecutive_successes: int   // reset to 0 on any new failure or new patch
    entered_at
    last_attempt_at
}
```

Clear transition:

```
if patch_promoted AND shadow_passed AND consecutive_successes >= 3:
    failure_replay_store.remove(episode_id)
else:
    keep in queue (scheduler may resample with higher weight)
```

If a **new** patch is proposed or applied for this `episode_id`, reset `consecutive_successes` to 0 and set `shadow_passed = false` until episode-specific shadow re-run passes again.

If patch is **rejected**, increment replay priority; do not clear.

Metrics to track: `failure_replay_queue_depth`, `episodes_stuck_past_N_attempts`, `cleared_without_shadow_pass` (must stay 0).

---

## B. Run Executor

Runs one episode through the architecture.

```
raw_input→ perception→ cue encoder→ DMF→ binding→ context-op→ interference→ basins→ lucidity→ optional projector→ decoder/action
```

The executor must support two modes:

```
inference_onlytraining_observation
```

Most runs should start as inference-only. Training happens after validation.

---

## C. Run Logger

Logs every intermediate state.

This is critical.

```
RunLog {    episode_id    raw_input    evidence_graph    cue_cloud    active_traces    trace_clusters    candidate_bindings    context_frames    scoped_trace_assignments    interference_edges    active_basins    basin_assemblies    lucidity_features    lucidity_decision    projection_result_optional    decoder_output    validator_result    cost_metrics}
```

Without this log, you cannot cheaply assign blame.

---

## D. Validator

Computes whether the result worked.

Validator types:

```
exact_matchunit_testsimulator_rewardformal_checkerretrieval_groundinghuman_feedbackself_consistencydecoder_faithfulness
```

Output:

```
ValidationResult {    success: bool    score: float    failure_signals    expected_state_optional    confidence}
```

For chat/text, validation may be softer. For code/math/grid/tool tasks, validation should be hard.

---

## E. Blame Assigner

Determines what failed.

This is the key cost-saving module.

Input:

```
RunLog + ValidationResult
```

Output:

```
FailureDiagnosis {    primary_failure_module    secondary_failure_modules    blame_confidence    evidence    recommended_update_type}
```

Failure table:

|Symptom|Likely update|
|---|---|
|Missing object/span/region|perception|
|Evidence exists but correct trace absent|cue encoder / DMF|
|Correct traces active but wrong role structure|binding|
|Correct structure but evidence leaked globally|context-op|
|Correct scope but wrong support/suppression|interference|
|Wrong basin won|basin links / basin split|
|Correct basin but premature commit|lucidity|
|Needed consequence test but skipped it|lucidity/projector policy|
|Good committed state, bad wording/output|decoder|
|New recurring pattern|seed trace/basin in quarantine|

This module is why training can be cheap.

---

## F. Update Planner

Chooses the cheapest valid update.

Update ladder:

```
Level 0: no updateLevel 1: counter/stat updateLevel 2: trace reinforcementLevel 3: interference edge updateLevel 4: binding affordance updateLevel 5: context scope updateLevel 6: basin link updateLevel 7: basin split/mergeLevel 8: tiny policy updateLevel 9: perception/cue/decoder gradient patchLevel 10: global recalibration
```

Rules:

```
Always try lowest level first.Never update more than 1–2 modules unless diagnosis is low-confidence.Never do global recalibration unless repeated failures prove local repair is insufficient.
```

---

## G. Patch Builder

Creates a candidate update, but does not immediately apply it to live memory.

Patch types:

```
TracePatchInterferencePatchBindingPatchContextPatchBasinPatchLucidityPatchDecoderPatchPerceptionPatchCueEncoderPatch
```

Example:

```
InterferencePatch {    edge: trace_t42 → basin_b19    scope: context_frame_f2    delta_positive: +0.04    delta_negative: 0    reason: "appeared in successful committed path"}
```

Example:

```
ContextPatch {    pattern: [kayaking, found, money, placed, bank]    change:        reduce kayaking influence on placed-bank frame        increase money bridge weight}
```

---

## H. Shadow Evaluator

Applies the patch in a shadow copy.

Never directly mutate the live system.

```
live_state+ patch→ shadow_state→ run targeted tests→ compare live vs shadow
```

Test bundle (assembled by **RetentionSuiteManager** — see § H.1):

```
1. original failed episode          (always — target fix)
2. nearby variants                (same family / perturbation)
3. retention canary suite         (curated subset — NOT full corpus at scale)
4. adversarial ambiguity cases    (from retention suite tags)
5. cost check
```

Patch promotion rule:

```
promote if:    fixes target failure    does not regress retention suite    does not increase cost too much    does not reduce lucidity calibration    does not increase basin pollution
```

Otherwise reject or quarantine.

After promotion, Shadow Evaluator (or Promotion Manager) must run **episode-specific shadow pass** on the original failed `episode_id` before that entry can count toward failure-replay clear (see Failure replay clear rules).

---

### H.1. RetentionSuiteManager

Shadow testing runs a bundle on **every** patch candidate. At Phase 1 (~420 fixtures) the full retention set is fine. By Phase 4 (1M+ episodes) running the full corpus per patch makes shadow eval the bottleneck.

**RetentionSuiteManager** maintains a **small, representative canary suite** — not the training corpus.

```
RetentionSuiteManager {
    canary_episodes[]           // fixed-size curated set (see budgets below)
    suite_version
    family_coverage_map         // at least one canary per task family / module
    last_full_audit_at          // when suite was refreshed against corpus stats
    regression_log              // episodes that regressed after any promote → auto-canary
}
```

Responsibilities:

```
select_shadow_bundle(patch, target_episode) → episode_ids[]   // bounded size
on_promote(patch_result) → update canary if regression detected
on_corpus_growth() → periodic refresh (not every patch)
audit_suite_vs_corpus() → flag under-covered families
```

**Suite maintenance rules:**

| Trigger | Action |
|---------|--------|
| Patch causes retention regression on any episode | Add that `episode_id` to canary (or bump weight); never drop silently |
| New task family appears in training | Add ≥1 canary within 1k episodes or block promote for that family |
| Module blame spike (e.g. context-op) | Add 3–5 episodes from `failure_replay` that fixed after context patch |
| Scheduled audit (weekly Phase 4+, daily Phase 5) | Stratified sample from corpus → compare live vs shadow on 2× suite size; swap weakest 10% canaries |
| Canary age > 90 days without hit | Review for staleness; demote only if covered by newer canary in same family |

**Shadow bundle size budgets (hard caps):**

| Phase | Corpus scale | Max episodes per shadow test |
|-------|----------------|------------------------------|
| 1 | ~420 fixtures | full retention OK (≤500) |
| 2–3 | 10k–100k | ≤200 (target + variants + canary) |
| 4 | 100k–1.5M | ≤80 canary + target bundle ≤20 |
| 5 | 1.5M+ | ≤50 canary + target bundle ≤15; high-risk patches ≤120 |

Target bundle = failed episode + nearby variants only. Canary suite is **shared and stable** across patches until audit refresh.

```
shadow_test_bundle =
    [target_episode] +
    nearby_variants(patch, max=5) +
    retention_suite_manager.sample(canary_budget) +
    cost_probe
```

**Representativeness without full scans:**

```
stratify by: modality, task_family, primary_module_in_last_blame, lucidity_decision_type
ensure: min 1 canary per family in coverage_map
weight: failure_replay_cleared episodes, recent regressions, held-out gold tasks
```

Phase 1: canary suite may equal all fixtures. Phase 4+: never pass `full_corpus` into Shadow Evaluator.

---

## I. Promotion Manager

If shadow test passes on **target bundle + retention suite**:

```
apply patch to live
record patch history
link patch_id → FailureReplayEntry.patch_ids_applied for target episode(s)
run episode-specific shadow on original failed episode_id → set shadow_passed
reset consecutive_successes = 0 on those replay entries (streak restarts after promote)
increase confidence
maybe quantize/freeze if stable
```

If it fails:

```
reject patch
store failed patch as negative research memory
possibly escalate to higher update level
scheduler.add_to_failure_replay(run_log)  // do not clear existing entries
```

Promotion does **not** clear failure replay by itself — only the 3-condition clear rule does.

---

## J. Quantizer / Freezer

Runs after updates.

Rules:

```
stable trace → low-bit codestable basin → low-bit attractorstable interference edge → ternary sign + confidence buckettrusted module state → frozen until failure evidence
```

Do not quantize new parts early.

Lifecycle:

```
provisional→ plastic→ stable→ quantized→ frozen→ thawed only on repeated failure
```

---

# 4. Orchestrator data model

## TrainingEpisode

```
TrainingEpisode {    episode_id    raw_input    modality    task_intent    context    constraints    allowed_tools    expected_output_optional    validator    metadata}
```

## RunLog

```
RunLog {    episode_id    evidence_graph    cue_cloud    active_traces    candidate_bindings    context_frames    interference_edges    active_basins    lucidity_decision    projection_result_optional    decoder_output    validator_result    cost_metrics}
```

## FailureDiagnosis

```
FailureDiagnosis {    primary_module    failure_type    confidence    responsible_objects    suggested_update_level}
```

## UpdateProposal

```
UpdateProposal {    patch_type    target_objects    update_level    expected_fix    risk_level    shadow_test_bundle}
```

## PatchResult

```
PatchResult {    fixed_target    retention_passed    cost_delta    quality_delta    promoted: bool    episode_shadow_passed: bool    retention_suite_version    notes}
```

## FailureReplayEntry

```
FailureReplayEntry {    episode_id    run_log_id_last_failure    patch_ids_applied[]    shadow_passed    consecutive_successes    entered_at    last_attempt_at}
```

## RetentionSuiteSnapshot

```
RetentionSuiteSnapshot {    suite_version    canary_count    family_coverage    episode_ids[]    last_audit_at}
```

---

# 5. Cheapness rules

The orchestrator enforces these.

## Rule 1: No-update is common

```
if success and lucidity_margin high:    do not update learned weights    only update usage counters
```

## Rule 2: Local updates only

```
if binding failed:    update binding affordance table    do not touch perception, DMF, decoder
```

## Rule 3: Sparse activation

```
max_active_traces = Kmax_candidate_bindings = Bmax_active_basins = H
```

Example starting values:

```
K = 128 tracesB = 32 candidate framesH = 16 basin candidates
```

Increase only when lucidity requests wider search.

## Rule 4: Projector is on-demand

```
run projector only if:    task requires consequence generation    lucidity margin is low    output must be externally validated    high-risk answer
```

## Rule 5: Mature memory is compressed

```
stable traces/basins/interference edges are low-bitplastic zones are small
```

## Rule 6: Update budgets

Per episode:

```
max_modules_updated = 1 by defaultmax_modules_updated = 2 if blame uncertainmax_global_updates = 0 except scheduled recalibration
```

## Rule 7: Patch must beat live

No patch gets promoted because it “seems good.”

It must beat current live state on a test bundle.

---

# 6. Orchestrator loop

```
while training:
    episode = scheduler.sample()   # failure_replay weighted until cleared
    run_log = executor.run(episode, mode="training_observation")
    validation = validator.evaluate(run_log)

    if validation.success and run_log.lucidity.margin_high:
        stats.update_success(run_log)
        quantizer.maybe_freeze(run_log)
        if failure_replay_store.contains(episode.id):
            failure_replay_store.record_success(episode.id)   # bump consecutive_successes
            failure_replay_store.try_clear(episode.id)          # clear only if 3-streak + shadow_pass + patch_promoted
        continue

    diagnosis = blame_assigner.diagnose(run_log, validation)
    update_proposal = update_planner.plan(diagnosis, run_log)
    if update_proposal.level == 0:
        scheduler.add_or_refresh_failure_replay(run_log)
        continue

    patch = patch_builder.build(update_proposal, run_log)
    bundle = retention_suite_manager.select_shadow_bundle(patch, run_log.episode_id)
    shadow_result = shadow_evaluator.test(patch, bundle)

    if shadow_result.promote:
        promotion_manager.apply(patch)
        promotion_manager.run_episode_shadow(run_log.episode_id, patch)
        failure_replay_store.on_patch_promoted(run_log.episode_id, patch.id)
        # consecutive_successes reset; shadow_passed set true only if episode shadow ok
    else:
        promotion_manager.reject(patch)
        failure_replay_store.on_patch_rejected(run_log.episode_id)
        scheduler.add_or_refresh_failure_replay(run_log)

    quantizer.maybe_quantize_stable_parts()
```

This is the heart of the system.

**Invariant:** `failure_replay_queue` depth should correlate with *unresolved* failures, not with *promoted-but-unverified* patches. Alert if `episodes_stuck_past_N_attempts` grows while `patch_promotion_rate` is high.

---

# 7. Blame assignment logic

Use deterministic checks first.

```
if expected evidence missing:    blame perceptionelif correct evidence exists but no matching traces activated:    blame cue_encoder_or_DMFelif correct traces active but no good candidate frame:    blame bindingelif good candidate frame exists but wrong context influence:    blame context_opelif correct frame/scope exists but wrong basin energy:    blame interference_or_basinelif correct basin high but lucidity rejected:    blame lucidity_too_strictelif wrong basin committed despite conflict:    blame lucidity_too_looseelif committed state correct but output wrong:    blame decoder
```

When unsure, do diagnostic reruns:

```
force correct evidenceforce correct tracesforce correct bindingforce correct contextforce correct basin
```

Whichever forced intervention fixes the result identifies the failing module.

This is extremely important.

---

# 8. Diagnostic forcing

To cheaply locate failure, run controlled interventions.

Example:

```
Original run failed.Test A:    give gold evidence graph.    If fixed → perception problem.Test B:    force correct trace activation.    If fixed → cue/DMF problem.Test C:    force correct binding frame.    If fixed → binding problem.Test D:    force correct context scope.    If fixed → context-op problem.Test E:    force correct basin candidate.    If fixed → interference/basin problem.Test F:    force correct committed state.    If output still wrong → decoder problem.
```

These do not have to run every time. Use them when blame confidence is low.

This is how you prevent random global updates.

---

# 9. Corpus interface

The orchestrator should consume episodes from a **corpus factory**, not one static dataset.

Episode sources:

```
real_chatpublic_textsynthetic_structureverifiable_reasoningtool_tasksmultimodal_tasksagent_tasksself_failure_replayretention_tests
```

Scheduler sampling policy:

```
sample_weight =    base_weight  + failure_rate  + novelty_value  + undertrained_module_value  + retention_risk  - overfitting_risk
```

If context-op is failing often, the scheduler should sample more context-scope episodes.

If decoder is weak, sample committed-state rendering tasks.

If basins are polluted, sample retention and ambiguity tests.

---

# 10. Metrics the orchestrator must track

## Cost metrics

```
active_traces_per_episodeactive_basins_per_episodeprojector_calls_per_episodeupdates_per_episodemodules_updated_per_episodeshadow_tests_per_patchcost_per_successful_learning_event
```

## Quality metrics

```
success_ratelucidity_false_commit_ratelucidity_false_reject_ratedecoder_faithfulnessprojection_pass_ratebinding_stabilitycontext_leak_ratebasin_pollution_rateretention_score
```

## Learning efficiency

```
new_skill_acquisition_costfailures_until_fixpatch_promotion_ratepatch_regression_ratetrace_promotion_ratebasin_split_success_rate
```

The most important metric:

```
capability gained per update dollar
```

---

# 11. Promotion thresholds

Example defaults:

```
Patch promotion requires:target_fix_rate >= 90% on target bundleretention_score_drop <= 1%  (measured on RetentionSuiteManager canary — not full corpus)cost_increase <= 5%lucidity_false_commit_not_increaseddecoder_faithfulness_not_decreased
```

**Separate from patch promotion:** failure-replay entry clear requires episode-specific shadow pass + 3 consecutive successes (§3.A). Do not conflate “patch promoted” with “episode resolved.”

For high-risk core changes:

```
retention_score_drop must be 0larger shadow suite requiredhuman/external review optional
```

For tiny local edge updates:

```
small test bundle enoughauto-promote if safe
```

---

# 12. State storage

Use separated stores:

```
TraceStoreBasinStoreInterferenceGraphStoreAffordanceStoreContextPolicyStoreLucidityPolicyStoreDecoderAdapterStorePatchHistoryStoreRunLogStoreFailureReplayStoreRetentionSuiteStore
```

`FailureReplayStore` implements clear rules (patch + episode shadow + 3-streak). `RetentionSuiteStore` holds canary `episode_id`s + `suite_version`; Shadow Evaluator reads only through RetentionSuiteManager.

Do not blend everything into one opaque weight blob.

---

# 13. Build order

## MVP 1: Offline orchestrator

Build:

```
EpisodeRunLogValidatorBlameAssignerUpdatePlannerPatchBuilderShadowEvaluatorPromotionManagerFailureReplayStore (with explicit clear rules)RetentionSuiteManager (full suite = all fixtures in Phase 1)
```

Use simple rule-based modules first.

Goal:

```
prove update-only-failing-part loop
prove failure replay does not clear until episode re-validated (3-streak)
```

## MVP 2: Sparse trace/basin learning

Add:

```
TraceStoreBasinStoreInterferenceGraphlocal updatesquantization lifecycle
```

Goal:

```
system learns from repeated episodes without global training
```

## MVP 3: Context and binding repair

Add:

```
binding affordance updatescontext scoping updatesdiagnostic forcing
```

Goal:

```
fix mixed-signal text and structured tasks locally
```

## MVP 4: Projector and tool validation

Add:

```
optional projectorexternal validatorstool tasks
```

Goal:

```
lucidity learns when to test consequences
```

## MVP 5: Self-research loop

Add:

```
patch proposal generationbroader architecture changesshadow regression suitespromotion gatesRetentionSuiteManager audit refresh at scale (canary caps per phase)
```

Goal:

```
safe continual improvement without shadow eval becoming the training bottleneck
```

---

# 14. Why this is your cost edge

Frontier training:

```
train huge dense model on enormous corpus
```

Your orchestrator:

```
generate structured episodesrun sparse architecturevalidateupdate only small responsible objectshadow-testpromote/freeze
```

The cost advantage comes from:

```
no global backprop through everythingno full memory scansno mandatory projectionno updates on high-margin successno independent giant model per modulequantized mature memory
```

The orchestrator is the thing that enforces this.

---

# 15. Short final spec

```
TrainingOrchestrator {    Scheduler    Executor    RunLogger    Validator    BlameAssigner    UpdatePlanner    PatchBuilder    RetentionSuiteManager    ShadowEvaluator    PromotionManager    FailureReplayStore    QuantizerFreezer    MetricsTracker}
```

Primary loop:

```
sample episoderun architecturelog all internalsvalidate resultif high-margin success: no update; if in failure_replay, bump streak and try clearelse diagnose failurebuild smallest patchtest patch in shadow (curated retention suite, not full corpus)promote only if better; episode-specific shadow on targetquantize/freeze stable partsfailure replay clears only: patch promoted + episode shadow pass + 3 consecutive successes without new patch
```

One-sentence definition:

```
The training orchestrator is the system that turns experience into the smallest safe local update instead of an expensive global training step.
```