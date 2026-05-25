# Training quantization

Scoped architecture spec for the lucid-model cognitive pipeline.

---

## Role

Training quantization defines **how and when the system learns** — not just how memory is stored. The goal is **maximum learning efficiency**: update only what failed, skip stable high-margin regions, and assign credit to the responsible subsystem.

Aligned with principles:

```
training must be optimized to the fullest (principle.md)
full auditability — every update/no-update decision logged
full editability — manual overrides to governor decisions
lazy collapse + lucidity — pass/fail from lucidity drives consolidation
```

Split/merge/repair of quantized structures and full end-to-end quantization-aware training are **deferred** until core loop subsystems are proven — but the **training governor**, **no-update policy**, **active-region-only updates**, and **responsibility classifier** are first-class now.

---

## Pipeline touchpoints

```
Lucidity outcome     → training governor input
DMF margins          → no-update on high margin
Interference/basins  → active-region-only link updates
Binding failures     → binding-only training pairs
Decoder errors       → decoder-only correction pairs
Memory quantization  → promote/demote precision tiers on consolidation
```

---

## Training governor

Central orchestrator for all learning events. Full loop (scheduler, shadow eval, failure replay clear rules, retention canary suite) lives in **`training orchestrator.md`** — governor implements `NO_UPDATE | UPDATE | …` decisions; orchestrator owns patch shadow/promote and replay queue lifecycle.

**Failure replay invariant:** an episode leaves the replay queue only when `patch_promoted AND episode_shadow_pass AND consecutive_successes >= 3` (new patch resets streak). Promotion alone does not clear.

**Retention at scale:** shadow tests use `RetentionSuiteManager` curated canaries (Phase 4 cap ~80), not the full 1M+ episode corpus per patch.

```
TrainingGovernorInput {
    lucidity_output               // decision + check_results
    task_outcome                  // success | partial | fail
    competition_summary           // margins
    active_trace_ids
    active_basin_ids
    touched_subsystems            // inferred or classified
    iteration_cost                // search/projection cost
    heat_state_snapshot
}

TrainingGovernorOutput {
    action                        // UPDATE | NO_UPDATE | DEFER | QUARANTINE | THAW
    update_regions: [             // active-region-only
        { subsystem, target_ids, update_kind, magnitude }
    ]
    responsibility_assignment
    consolidation_directives      // heat tier changes, quantize promote
    audit_log
}
```

### Governor rules (priority order)

1. **Lucidity COMMIT + task success + high margin → NO_UPDATE** (default protect stable attractors)
2. **Lucidity COMMIT + task fail → responsibility classifier → targeted UPDATE**
3. **PRESERVE_AMBIGUITY + user clarification → narrow UPDATE on disambiguated region**
4. **SEARCH_WIDER exhausted + fail → provisional trace/basin QUARANTINE or spawn**
5. **RECHECK_BINDING → binding-only UPDATE, no basin destruction**

---

## No-update on high margin

**No-update is a first-class outcome**, not a missing gradient bug.

```
NoUpdateDecision {
    reason: high_margin_stable_pass
    margin: .14
    threshold: .08
    protected_ids: [t0142, b12441, ...]
    snapshot_id
}
```

When:

```
lucidity.decision == COMMIT
task_outcome == success
competition_summary.top_margin >= margin_no_update_threshold
binding_stability pass
no novelty flags unresolved
```

Then:

```
action: NO_UPDATE
```

**Rationale:** stable basins should not be perturbed by noise — matches lazy collapse cost story and continual learning fossilization.

Exceptions (still update):

```
explicit user correction
distribution shift detector fires
manual editor override
quarantine promotion after repeated success in new domain
```

---

## Active-region-only updates

Never backprop/update entire bank for a single failure.

```
UpdateRegion {
    subsystem: dmf | binding | context_op | interference | basins | projector | decoder
    target_ids: [specific trace/basin/link ids]
    update_kind: strengthen | weaken | split_candidate | merge_candidate
    magnitude: low | medium  // capped step
}
```

### Subsystem scopes

| Failure signature | Update region |
|-------------------|---------------|
| Perception evidence wrong | perception adapter only (if learned) |
| Wrong traces activated | DMF retrieval links for implicated ids |
| Wrong frames | binding weights/rules for implicated frames |
| Scope leak | context-op gates |
| Wrong support/conflict | interference links (ternary) |
| Basin collision | basin field + interference |
| Projection misfit | projector params + basin assembly links |
| Render wrong but state correct | decoder correction pairs only |

---

## Responsibility classifier

Assigns blame when `COMMIT` or expression fails.

```
ResponsibilityClassifierInput {
    lucidity_check_results        // which checks failed
    binding_stability_score
    projection_fit_scores
    decoder_render_match          // did output match committed state?
    perceptual_evidence_diff      // optional re-parse
}

ResponsibilityClassifierOutput {
    primary_blame: subsystem
    secondary_blame: [subsystem]
    confidence
    recommended_update_regions
}
```

### Heuristic table (starting point)

| Symptom | Primary blame |
|---------|---------------|
| coverage_check fail | DMF / cue encoder retrieval |
| binding_stability fail | binding |
| scope_check fail | context-op |
| conflict_reports wrong | interference |
| margin ok but wrong answer | basins or training data gap |
| projection_fit fail | projector + basin assembly |
| committed ok, render wrong | decoder |
| evidence missing | perception |

Classifier outputs are **auditable** and **overridable** by human editors.

---

## Decoder-only correction pairs

When lucidity committed correctly but decoder misrendered:

```
DecoderCorrectionPair {
    committed_state
    decoder_policy
    wrong_output
    corrected_output
    update_scope: decoder_only
}
```

No updates to basins/traces for decoder phrasing errors.

---

## Margin metrics for training

Track rolling statistics:

```
BasinMarginMetrics {
    basin_id
    mean_margin_on_success
    mean_margin_on_fail
    no_update_count
    last_update_timestamp
}
```

Use to tune thresholds and identify basins needing split (deferred) or reheat.

---

## Integration with memory quantization

On successful consolidation:

```
hot → warm → cold → frozen
fp16 → 8-bit → 4-bit → binary + optional PQ residuals
```

On failure in frozen region:

```
THAW policy → limited fp16 updates → re-quantize after lucidity pass
```

Training governor emits `consolidation_directives` — not silent quantization.

---

## Text example: bank / kayaking training

### Case A — high margin correct disambiguation

```
lucidity: PRESERVE_AMBIGUITY (correct)
user selects financial reading
lucidity: COMMIT b00491
margin: .12
→ NO_UPDATE on b12441 outdoor basin (still valid in F1)
→ UPDATE binding link: strengthen F2 DESTINATION slot for t00491 only
```

### Case B — wrong commit

```
committed b08810 (river) but user intent financial
margin was .03 (low — should not have COMMIT)
responsibility: lucidity threshold misconfig + binding unresolved_slots ignored
→ UPDATE lucidity risk_check thresholds
→ RECHECK_BINDING training pair for bank frames
→ NO_UPDATE on unrelated traces
```

---

## Grid example (visual stress test)

### Projection fail

```
REQUEST_PROJECTION → assembly ASY1 fail pair 2
responsibility: projector + basin assembly links
→ UPDATE interference cooperation between b5502/b3340
→ NO_UPDATE on b9100 legend basin (orthogonal scope)
```

### High margin synthetic task

```
COMMIT after exact projection, margin .18
→ NO_UPDATE entire active region
→ promote b5502 toward cold tier (consolidation_directive)
```

---

## Deferred (explicitly out of scope for v1)

Document but do not require in first implementation:

```
full quantized backprop through all subsystems
automatic basin split/merge at scale
global end-to-end loss replacing lucidity governor
```

Prove core loop first; add aggressive compaction after metrics justify it.

---

## Anti-patterns

**Do not update whole bank on every batch.**

```
BAD:  global backprop all traces
GOOD: active-region-only update_regions list
```

**Do not update on high-margin success.**

```
BAD:  always apply gradient
GOOD: NO_UPDATE first-class with audit reason
```

**Do not blame decoder for basin errors.**

```
BAD:  fix wording by updating basins
GOOD: responsibility classifier → subsystem match
```

**Do not train through lucidity bypass.**

```
BAD:  decoder generates, lucidity ignored, train on output
GOOD: lucidity outcome required for governor input
```

**Do not silently unfreeze frozen traces.**

```
THAW action must appear in audit_log
```

---

## Audit requirements

Every training event logs:

```
governor_input_hash
action (UPDATE | NO_UPDATE | ...)
update_regions or no_update reason
responsibility_classifier output
before/after snapshot ids for touched ids
human_override flag if any
```

---

## Summary

```
Training quantization = efficient, targeted learning policy
Core:     training governor
Key rule: NO_UPDATE on high margin success (protect stable attractors)
Scope:    active-region-only updates per subsystem
Blame:    responsibility classifier
Pairs:    decoder-only corrections when state was right
Deferred: full split/merge/quantized backprop until core loop proven
```

Training efficiency is not an afterthought — it is how the architecture keeps sparse memory stable while still learning from failure.
