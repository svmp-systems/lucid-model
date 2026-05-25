# Lucidity

Scoped architecture spec for the lucid-model cognitive pipeline.

---

## Role

Lucidity is the **controlled collapse gate** — where lazy collapse becomes an explicit, auditable decision. Every upstream stage preserves plural hypotheses; lucidity alone decides whether to commit, preserve ambiguity, request projection, widen search, or recheck binding.

In your architecture:

```
cue encoder
→ DMF / tracebank
→ binding
→ context-op
→ interference
→ basins
→ lucidity check
→ [projector if REQUEST_PROJECTION]
→ decoder
```

Lucidity is where lazy collapse becomes controlled collapse.

Lucidity does **not** replace binding, context-op, or basins. It **evaluates** their outputs against thresholds, margins, coverage, coherence, and optional projection fit — then sets **decoder policy** so the decoder cannot hallucinate past approval.

---

## Pipeline position

### Default path (no projection)

```
basins → lucidity → decoder
```

Use when margin, coverage, coherence, and binding stability are already sufficient for the task and risk level.

### Projection path

```
basins → lucidity (pre-check) → projector → lucidity (final-check) → decoder
```

Use when consequences must be tested before commit (grid exact match, planning rollouts, high-stakes disambiguation).

The projector is **optional**. Lucidity decides via `REQUEST_PROJECTION`.

---

## Core principles

1. **Only lucidity collapses** — no earlier stage may set `COMMIT`.
2. **Plural inputs, explicit decisions** — one primary `decision` enum per pass; *how* to commit is `committed_state.commit_shape`, not extra decision types.
3. **Decoder is gated** — decoder policy flows from lucidity output; decoder never decides truth.
4. **Learned IDs throughout** — commits reference `t####`, `b####`, frame IDs — not hard-coded concepts.
5. **ARC is stress test** — grid tasks use same checks; no hard-coded solvers in lucidity.
6. **Audit everything** — every check score, threshold, and decision reason is logged.

---

## Input contract

```
LucidityInput {
    // From basins
    basin_output
    candidate_basin_states
    basin_assemblies
    competition_summary

    // From upstream
    binding_output
    binding_stability_score
    context_op_output
    context_frames
    interference_output
    conflict_reports
    dmf_margin_info
    perceptual_evidence_graph

    // Task & risk
    task_intent                 // answer | solve_grid | act | retrieve
    risk_level                  // low | medium | high
    stakes_policy               // when to require projection

    // Projection (if re-entry after projector)
    projection_result           // optional
    projection_fit_scores

    // Loop state
    pass_kind                   // pre_check | final_check | recheck
    iteration_count
    prior_lucidity_decisions

    compute_policy
}
```

---

## Output contract

```
LucidityOutput {
    decision                    // primary enum — see Decision types
    secondary_decisions         // optional parallel actions

    committed_state             // populated only if decision == COMMIT (or partial commit)
    preserved_hypotheses        // populated if PRESERVE_AMBIGUITY
    search_directives           // populated if SEARCH_WIDER or RECHECK_BINDING

    decoder_policy              // mandatory — gates decoder behavior

    check_results: {
        margin_check
        coverage_check
        coherence_check
        binding_stability_check
        scope_check
        projection_fit_check      // N/A if no projection
        contradiction_check
        maturity_check
        risk_check
    }

    confidence_summary: {
        overall_confidence
        margin
        coverage
        coherence
        projection_fit            // optional
    }

    audit_log
    provenance
}
```

---

## Decision types

Lucidity must emit exactly one **primary** `decision` per pass (secondary actions optional).

### COMMIT

Collapse to a committed internal state for decoder rendering.

```
When:
    margin_check pass
    coverage_check pass
    coherence_check pass
    binding_stability pass
    scope_check pass
    projection_fit pass (if projection ran or required by stakes_policy)
    no unresolved hard contradictions
```

```
LucidityOutput {
    decision: COMMIT
    committed_state: CommittedState { ... }
    decoder_policy: { mode: express_committed, uncertainty: low }
}
```

Partial commit is allowed when task allows scoped answers (e.g. commit F2 destination sense while marking F1 narrative as auxiliary).

#### Commit shapes (`committed_state.commit_shape`)

When `decision == COMMIT`, set exactly one **commit shape** — metadata for decoder and audit, **not** separate lucidity enums.

Do **not** add `COMMIT_SINGLE`, `COMMIT_MULTI_FRAME`, `COMMIT_ASSEMBLY`, or `REQUEST_ROLLOUT` as primary decisions. Use the table below.

| `commit_shape` | When | Typical task |
|----------------|------|--------------|
| `single` | One frame, one dominant basin, margin high | Simple disambiguation ("bank" + withdrew money) |
| `per_frame` | Multiple independent frames, each with its own basin commit | Multi-event text (found while kayaking / placed in bank) |
| `assembly` | Cooperative basins explain more together than alone | ARC move + recolor; compositional rules |
| `rollout_plan` | Future states/actions matter; commit includes planned rollouts | Agents, world models, AGI-3-style exploration |

```
LucidityOutput {
    decision: COMMIT
    committed_state: {
        commit_shape: per_frame
        frame_commits: [
            { context_frame_id: F1, basin_id: b12441, ... },
            { context_frame_id: F2, basin_id: b00491, ... }
        ]
        ...
    }
}
```

Basins still emit plural candidates; lucidity chooses shape from `frame_type`, margins, coverage, `task_intent`, and projection results.

#### Choosing commit shape (heuristic)

```
if one salient frame and margin_check pass:
    commit_shape = single

elif multiple independent frames with stable bindings each:
    commit_shape = per_frame

elif basin_assemblies beat singletons on coverage/coherence:
    commit_shape = assembly
    (often after REQUEST_PROJECTION + projection_fit pass)

elif task_intent in { act, plan } and stakes need consequence test:
    decision may be REQUEST_PROJECTION first
    then COMMIT with commit_shape = rollout_plan

elif margin low and ambiguity legitimate:
    decision = PRESERVE_AMBIGUITY (not single)
```

Domain (text vs ARC vs agent) does **not** hard-code shape — `frame_type` + checks do.

---

### PRESERVE_AMBIGUITY

Do not collapse; keep plural hypotheses alive; decoder must not invent a single story.

```
When:
    margin below threshold but evidence quality high
    legitimate ambiguity (bank polysemy, multiple valid grid rules)
    task allows non-collapsed response (explain alternatives)
```

```
LucidityOutput {
    decision: PRESERVE_AMBIGUITY
    preserved_hypotheses: [hypothesis_A, hypothesis_B, ...]
    decoder_policy: { mode: express_plural, forbid_single_answer: true }
}
```

---

### REQUEST_PROJECTION

Authorize expensive consequence testing before commit — including **grid fit**, **assembly verification**, and **agent/world-model rollouts**.

There is no separate `REQUEST_ROLLOUT` decision. Rollouts are projection with rollout directives.

```
When:
    task_intent requires exact validation (solve_grid)
    stakes_policy high and margin low
    top basins tie after interference
    assembly needs fit verification
    act | plan intent and consequence must be tested before commit
```

```
LucidityOutput {
    decision: REQUEST_PROJECTION
    search_directives: {
        projector_targets: [basin_ids | assembly_ids]
        max_rollouts: k
        rollout_mode: none | single_step | multi_step    // none for static grid fit
        rollout_depth: int                               // agent / world model
        counterfactual_frames: [frame_id]                // optional
    }
    decoder_policy: { mode: hold — no external output yet }
}
```

After projector returns, lucidity re-enters with `pass_kind: final_check`. Successful rollout may yield `COMMIT` with `commit_shape: rollout_plan` and `projection_artifact` populated.

```
Flow (planning):
    state basins → REQUEST_PROJECTION (rollout_mode: multi_step)
    → projector → predicted state basins
    → final_check → COMMIT | PRESERVE_AMBIGUITY | SEARCH_WIDER
```

---

### SEARCH_WIDER

Widen retrieval, binding, or basin search — loop back upstream with directives.

```
When:
    coverage_check fail (evidence unexplained)
    all top hypotheses weak
    projection failed for all top-k assemblies
    novelty signals unresolved
```

```
LucidityOutput {
    decision: SEARCH_WIDER
    search_directives: {
        cue_encoder: { ambiguity_policy: force_widen, budget_multiplier: 1.5 }
        binding: { allow_new_frames: true }
        basins: { allow_provisional: true }
    }
    decoder_policy: { mode: hold | express_partial_progress }
}
```

Cap `iteration_count`; after cap, fall back to `PRESERVE_AMBIGUITY` or best-effort partial with explicit uncertainty.

---

### RECHECK_BINDING

Binding frames unstable or inconsistent with scoped evidence — request re-bind, not wider DMF.

```
When:
    binding_stability_check fail
    role conflicts inside a frame
    frame-evidence misalignment
    context-op scope correct but frames wrong
```

```
LucidityOutput {
    decision: RECHECK_BINDING
    search_directives: {
        binding: { rebind_frames: [F2], preserve_traces: true }
        context_op: { re_scope: false }
    }
    decoder_policy: { mode: hold }
}
```

Distinguish from `SEARCH_WIDER`: recheck binding does **not** necessarily widen trace retrieval.

---

## Check suite

Each check returns `{pass: bool, score: float, threshold: float, details: ...}`.

### 1. Margin check

```
margin = competition_summary.top_margin
       or assembly_margin if assemblies dominate

pass if margin >= margin_threshold(task_intent, risk_level)

task-specific notes:
    solve_grid: often stricter after projection than before
    answer (text): may allow lower margin if PRESERVE_AMBIGUITY acceptable
```

```
margin_check {
    pass: false
    score: .04
    threshold: .08
    details: { top: b00491, second: b08810, scope: F2 }
}
```

---

### 2. Coverage check

```
coverage = fraction of salient evidence with supporting active traces + frames + basins

pass if coverage >= coverage_threshold

fail triggers SEARCH_WIDER or RECHECK_BINDING depending on whether frames explain gaps
```

Example fail: legend region detected in perception but no supporting basin in F_legend.

---

### 3. Coherence check

```
coherence = internal consistency of top candidate(s):
    - no contradictory role assignments in same frame
    - assembly members non-conflicting
    - scoped traces align with frame membership

pass if coherence >= coherence_threshold
```

Example fail: single frame assigns both `t_bank_fin` and `t_bank_river` as DESTINATION without unresolved_slots flag.

---

### 4. Binding stability check

```
binding_stability = binding_output.binding_stability_score
                  adjusted by frame competition churn

pass if stable across micro-iterations or explicit unresolved_slots present
```

Unstable binding → `RECHECK_BINDING`.

---

### 5. Scope check

```
verify context-op gates honored:
    kayaking evidence not used to commit finance basin globally
    F_legend scopes do not suppress F_content without evidence

pass if no cross-scope contamination detected
```

---

### 6. Projection fit check

Only when `projection_result` present or `stakes_policy` requires it.

```
projection_fit = aggregate fit over train pairs / rollouts

for grids: exact match per cell where required
for text: consistency with committed frames (not fluency)

pass if fit >= fit_threshold and no overfit flags
```

```
projection_fit_check {
    pass: true
    score: 1.0
    details: { pairs_passed: 3/3, test_outputs_generated: 1 }
}
```

---

### 7. Contradiction check

```
hard_contradictions = interference conflict_reports above severity threshold
                  inside committed scope

pass if no hard contradictions remain in committed_state candidate
```

Soft conflicts may survive under `PRESERVE_AMBIGUITY`.

---

### 8. Maturity check

```
traces/basins involved meet minimum heat tier for risk_level
provisional-only commits blocked on high stakes unless projection passes
```

---

### 9. Risk check

```
maps risk_level + task_intent to required checks:
    high + finance advice → require projection or explicit uncertainty
    low + retrieval → may COMMIT with moderate margin
```

---

## CommittedState IR

When `decision == COMMIT`, lucidity emits a clean IR for decoder — not raw activations.

```
CommittedState {
    commit_id
    commit_shape                  // single | per_frame | assembly | rollout_plan

    primary_basin_id              // b#### — dominant for single; optional for per_frame
    assembly_ids                  // required when commit_shape == assembly
    member_basin_ids              // explicit list when assembly

    frame_commits: [               // required when commit_shape == per_frame; optional otherwise
        {
            context_frame_id
            frame_type              // from binding — auditable
            basin_id
            role_map               // slot → t####
            scope_notes
        }
    ]

    rollout_steps: [               // when commit_shape == rollout_plan
        {
            step_index
            action_ref              // trace or tool ref
            predicted_state_basin_id
            fit_score
        }
    ]

    claims: [                      // structured, not prose
        {claim_type, subject_ref, predicate_ref, confidence, scope}
    ]
    unresolved: [...]              // optional explicit leftover ambiguity
    projection_artifact           // optional grid IR, plan IR
    provenance_chain
}
```

| `commit_shape` | Required fields |
|----------------|-----------------|
| `single` | `primary_basin_id`, usually one `frame_commits[0]` |
| `per_frame` | `frame_commits` (one basin per independent frame) |
| `assembly` | `assembly_ids` or `member_basin_ids` + `projection_artifact` if grid |
| `rollout_plan` | `rollout_steps` + `projection_artifact` |

---

## Decoder policy

Lucidity **always** sets decoder policy. Decoder expresses; lucidity decides.

```
DecoderPolicy {
    mode: express_committed
        | express_plural
        | express_uncertainty
        | express_refusal
        | hold

    forbid_single_answer: bool
    forbid_invented_facts: bool
    require_cite_traces: bool
    max_detail_level: low | medium | high
    output_format: text | grid | action | plan | tool_call

    uncertainty_presentation: {
        show_alternatives: bool
        show_confidence: bool
        show_scope: bool
    }

    refusal_reason               // if mode == express_refusal
}
```

### Policy mapping

| Decision | Typical decoder mode |
|----------|---------------------|
| COMMIT | express_committed |
| PRESERVE_AMBIGUITY | express_plural |
| REQUEST_PROJECTION | hold |
| SEARCH_WIDER | hold or express_partial_progress |
| RECHECK_BINDING | hold |

**Anti-hallucination rule:** if lucidity is weak, decoder policy must express uncertainty or refuse — never fill gaps fluently.

---

## Pre-check vs final-check

### Pre-check (before projector)

```
Inputs: basin_output, no projection_result
Decisions allowed: COMMIT, PRESERVE_AMBIGUITY, REQUEST_PROJECTION, SEARCH_WIDER, RECHECK_BINDING

Typical grid hard task:
    margin low + task_intent solve_grid → REQUEST_PROJECTION
```

### Final-check (after projector)

```
Inputs: basin_output + projection_result + projection_fit_scores
Decisions allowed: COMMIT, PRESERVE_AMBIGUITY, SEARCH_WIDER, RECHECK_BINDING
(request projection usually disallowed — prevent loops unless iteration budget)

Typical:
    projection_fit pass + margin ok → COMMIT
    projection_fit fail all top-k → SEARCH_WIDER
```

Why optional projector:

```
projection is expensive
many tasks do not need simulated consequences to commit safely
decoder must never run on unapproved basin state regardless
```

---

## Text example: bank / kayaking sentence

### Pass 1 — pre-check

```
LucidityInput {
    competition_summary: { top_margin: .04 }
    binding_output: { F1, F2 with bank unresolved }
    task_intent: answer
    risk_level: medium
    pass_kind: pre_check
}
```

```
LucidityOutput {
    decision: PRESERVE_AMBIGUITY
    preserved_hypotheses: [
        {frame: F1, narrative: outdoor_discovery, basin: b12441},
        {frame: F2, narrative: financial_storage, basin: b00491},
        {frame: F2, narrative: river_bank, basin: b08810}
    ]
    check_results: {
        margin_check: { pass: false, score: .04, threshold: .08 }
        coverage_check: { pass: true, score: .86 }
        coherence_check: { pass: true, score: .81 }
        binding_stability_check: { pass: true }
        scope_check: { pass: true }
    }
    decoder_policy: {
        mode: express_plural
        forbid_single_answer: true
        show_alternatives: true
        output_format: text
    }
}
```

Decoder renders both readings without picking one falsely.

### Pass 2 — after user asks "financial account"

External evidence narrows F2; margin improves.

```
decision: COMMIT
commit_shape: single
committed_state: {
    frame_commits: [{ frame: F2, frame_type: word_sense, basin: b00491, role_map: { DESTINATION: t00491 } }]
}
decoder_policy: { mode: express_committed, forbid_invented_facts: true }
```

If both F1 and F2 were committed in one pass (multi-clause answer):

```
commit_shape: per_frame
frame_commits: [
    { frame: F1, frame_type: event, basin: b12441, ... },
    { frame: F2, frame_type: event, basin: b00491, ... }
]
```

---

## Grid example (visual stress test)

### Pre-check

```
task_intent: solve_grid
risk_level: high
competition_summary: { top_margin: .03 }
assembly ASY1 = {b5502, b3340}
```

```
decision: REQUEST_PROJECTION
search_directives: { projector_targets: [ASY1, b9100], max_rollouts: 2 }
decoder_policy: { mode: hold }
```

### Final-check (after projector)

```
projection_fit_check: { pass: true, score: 1.0, pairs: 3/3 }
margin_check: { pass: true, score: .12 }
```

```
decision: COMMIT
commit_shape: assembly
committed_state: {
    assembly_ids: [ASY1]
    member_basin_ids: [b5502, b3340]
    projection_artifact: { grid_output, ir_program_ref }
}
decoder_policy: { mode: express_committed, output_format: grid }
```

If projection fit fails:

```
decision: SEARCH_WIDER
search_directives: { cue_encoder: force_widen, basins: allow_provisional }
```

---

## Loopback integration

| Decision | Upstream effect |
|----------|----------------|
| SEARCH_WIDER | cue encoder budget+, binding new frames, provisional basins |
| RECHECK_BINDING | binding rebind; context-op may adjust gates |
| REQUEST_PROJECTION | invoke projector module |
| PRESERVE_AMBIGUITY | optional user clarification loop |
| COMMIT | decoder only |

Training governor reads lucidity pass/fail to assign credit (see `training quantization.md`).

---

## Anti-patterns

**Do not COMMIT on argmax energy alone.**

```
BAD:  if top basin highest → COMMIT
GOOD: full check suite + decoder policy
```

**Do not let decoder override lucidity.**

```
BAD:  decoder picks fluent answer when decision PRESERVE_AMBIGUITY
GOOD: decoder_policy.forbid_single_answer = true
```

**Do not run projection by default on every pass.**

```
BAD:  always project top basin
GOOD: REQUEST_PROJECTION only when checks + task require it
```

**Do not global-classify then commit.**

```
BAD:  task is finance → commit finance
GOOD: frame-scoped commits with scope_check
```

**Do not multiply primary decision enums for commit variants.**

```
BAD:  COMMIT_SINGLE | COMMIT_MULTI_FRAME | COMMIT_ASSEMBLY | REQUEST_ROLLOUT
GOOD: decision: COMMIT + committed_state.commit_shape
GOOD: decision: REQUEST_PROJECTION + search_directives.rollout_mode
```

**Do not hard-code ARC solvers in lucidity.**

```
BAD:  if legend_task: commit legend_mapping
GOOD: coverage + projection_fit on universal CommittedState
```

**Do not silent loop forever.**

```
iteration_count cap → PRESERVE_AMBIGUITY or express_refusal
```

---

## Training signal

Lucidity outcomes feed back:

```
COMMIT + downstream success → strengthen traces, links, basins involved
COMMIT + failure → responsibility classifier assigns blame
PRESERVE_AMBIGUITY + user clarification → targeted update
SEARCH_WIDER cost logged → optimize pruning
```

Lucidity is both **runtime gate** and **learning referee**.

---

## Summary table

| Check | Fail tendency |
|-------|---------------|
| margin | PRESERVE_AMBIGUITY or REQUEST_PROJECTION |
| coverage | SEARCH_WIDER |
| coherence | RECHECK_BINDING or PRESERVE_AMBIGUITY |
| binding stability | RECHECK_BINDING |
| scope | RECHECK_BINDING / fix context-op |
| projection fit | SEARCH_WIDER |
| maturity + risk | REQUEST_PROJECTION or refuse |

---

## Summary

```
Lucidity = controlled collapse gate
Input:     basin_output + frames + task_intent + optional projection
Output:    primary decision + CommittedState (with commit_shape) + decoder_policy + checks
Decisions: COMMIT | PRESERVE_AMBIGUITY | REQUEST_PROJECTION | SEARCH_WIDER | RECHECK_BINDING
Commit:    commit_shape = single | per_frame | assembly | rollout_plan (under COMMIT only)
Rollout:   via REQUEST_PROJECTION + search_directives — not a separate decision
Principle: only lucidity collapses; basins stay plural; frame-scoped competition
Paths:     basins → lucidity → decoder  OR  basins → lucidity → projector → lucidity → decoder
```

Lucidity is the system's **conscience and brake** — the point where ambiguity becomes responsibility, and expression becomes trustworthy.
