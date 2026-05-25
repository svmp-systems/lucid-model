# Projector & decoder

Scoped architecture spec for the lucid-model cognitive pipeline.

Two modules: **projector** (optional — tests implications) and **decoder** (mandatory for output — expression only, gated by lucidity).

---

```
Projector = tests / expands what a basin implies (optional)
Decoder = expresses the committed state as output (always gated by lucidity)
```

The projector is closer to **simulation / prediction / implication**.

The decoder is closer to **translation / rendering / action formatting**.

---

# Default flow vs projection flow

**Default** (most clear text, retrieval, classification):

```
basins → lucidity → decoder
```

**Projection** (when consequences must be tested):

```
basins → lucidity → projector → lucidity → decoder
```

The projector is **not a mandatory stage**. Lucidity decides whether projection is needed.

Use projection when the system must test consequences, for example:

```
grid output generation (exact validation)
planning / action rollouts
counterfactuals
science hypotheses
self-research
ambiguous interpretation checks under high stakes
```

Often **skip projection** for:

```
simple text interpretation
basic retrieval
clear low-ambiguity classification
```

Architecture position:

```
basins
→ lucidity check
→ [projector if REQUEST_PROJECTION]
→ lucidity final check (if projection ran)
→ decoder
```

Or when projection is not needed:

```
Basin state
   ↓
Lucidity: commit or restrict decoder
   ↓
Decoder: express the committed state
```

---

## Pipeline summary

| Stage | Mandatory? | Decides truth? |
|-------|------------|----------------|
| Projector | No — lucidity requests | No — tests implications |
| Decoder | Yes for external output | No — renders CommittedState |

---

# Part 1 — Projector

## Role

Given candidate basin state(s), the projector **generates or simulates consequences** — candidate output grids, plan steps, counterfactual states — so lucidity can score fit before commit.

The projector is **not** a hand-coded ARC solver. It applies **committed or candidate basin assemblies** through a universal internal IR, parameterized by learned basins.

---

## Input contract

```
ProjectorInput {
    projection_request            // from lucidity REQUEST_PROJECTION
    target_assemblies             // basin ids or assembly ids
    candidate_frames
    context_frames
    perceptual_evidence_graph
    committed_or_candidate_state  // partial CommittedState if any
    constraints: {
        output_shape_rules
        train_pair_refs
        test_input_refs
        max_rollouts
    }
    task_intent
    compute_policy
}
```

---

## Output contract

```
ProjectorOutput {
    rollouts: [
        {
            rollout_id
            assembly_id
            implied_artifact          // grid, plan, state delta
            fit_scores: {
                per_train_pair
                aggregate_fit
                unexplained_cells       // grids
                consistency_score       // text/plans
            }
            ir_program_ref              // auditable internal program
            failure_point               // optional — which pair broke
        }
    ]
    best_rollout_id
    recommendation_to_lucidity      // suggest COMMIT | SEARCH_WIDER — not binding
    audit_log
}
```

---

# 11. Where projector sits relative to lucidity

The projector is **optional**. Lucidity is the gate.

**Path A — no projection:**

```
basins → lucidity → decoder
```

Use when margin, coverage, and binding stability are already sufficient.

**Path B — projection requested:**

```
basins → lucidity (REQUEST_PROJECTION?) → projector → lucidity → decoder
```

Use when the task requires consequence testing before commit.

Lucidity pre-check may output `REQUEST_PROJECTION`. Only then does the projector run. After projection, lucidity final-check decides `COMMIT`, `SEARCH_WIDER`, `PRESERVE_AMBIGUITY`, etc.

Why optional:

```
projection is expensive
many tasks do not need simulated consequences to commit safely
decoder must never run on unapproved basin state regardless
```

Flow when projection runs:

```
1. Basins produce candidate state.
2. Lucidity asks: is projection required for this task / ambiguity level?
3. If yes, projector tests implications.
4. Lucidity final-check: did projection validate the basin?
5. Decoder expresses only after lucidity approval.
```

---

# 12. Projector on grid / visual stress tests

For grid tasks (e.g. ARC-style benchmarks), projection is often **needed** because exact output validation is the test.

The projector applies a **committed or candidate basin assembly** and checks fit against training pairs. This is universal consequence testing — not a hand-coded legend solver, frame solver, or color-map branch wired into the architecture.

Input:

```
basin assembly: [b2011, b5502]
candidate frames from binding + context-op
constraints: output shape, training-pair fit
```

Projector function:

```
apply candidate hypothesis to each training input
compare generated output to training output
if fit good, apply to test input(s)
```

Lucidity uses fit scores to `COMMIT` or `SEARCH_WIDER`. Decoder renders the final grid only after commit.

### Grid IR (internal, auditable)

Basins select and parameterize ops — ops are not hand-wired task branches:

```
Move(obj_ref, dx, dy)
Recolor(obj_ref, c_from, c_to)
Fill(region_ref, color)
Copy(src_ref, dst_ref)
MapSymbol(glyph_ref, color_ref)
PredictOutputShape(rule_ref)
```

`CommittedState.projection_artifact` stores `(basin_ids, ir_program, params)`.

---

## Text example (counterfactual check)

High-stakes disambiguation before commit:

```
ProjectorInput {
    target_assemblies: [b00491]
    task_intent: answer
    constraints: { counterfactual: "If bank means financial institution, what follows?" }
}
```

Projector simulates consequence consistency with F2 frame — lucidity scores coherence, does not generate fluent prose.

---

# Part 2 — Decoder

# 6. Decoder

## Decoder's role

The decoder **expresses** truth; it does **not decide** truth.

It answers:

```
How do we express the lucidity-approved committed state?
```

It should not decide meaning.

It should not invent new claims.

It should not override lucidity.

Its job is:

```
committed internal state + decoder policy → external form
```

External form can be:

```
text answer
grid output
action command
tool call
plan
diagram
speech
```

The decoder is a **renderer**, gated by lucidity.

---

## Input contract

```
DecoderInput {
    lucidity_output               // must include decision + decoder_policy
    committed_state               // required for COMMIT; optional for plural modes
    preserved_hypotheses          // for PRESERVE_AMBIGUITY
    projection_artifact           // optional — e.g. final grid from projector
    output_format_hint
    provenance
}
```

Decoder **must reject** input without valid `decoder_policy` from lucidity.

---

## Output contract

```
DecoderOutput {
    rendered_payload              // text, grid tensor, action struct, etc.
    render_mode                   // committed | plural | uncertainty | refusal
    citations: [                  // trace/basin/frame refs used
        {ref_type, ref_id, role}
    ]
    uncertainty_block             // if policy requires
    refusal_message               // if policy express_refusal
    audit_log
}
```

---

## Decoder modes (from policy)

### express_committed

Render `CommittedState` only. No new entities, no gap-filling.

```
Input:  CommittedState with F2 → b00491, DESTINATION t00491
Output: "I placed the money in the financial institution." (or structured answer)
```

### express_plural

List preserved hypotheses without false collapse.

```
Output: "The sentence supports outdoor discovery (kayaking) and placing money in either a financial account or river bank."
```

### express_uncertainty

Shorter than plural; confidence bounds required.

### express_refusal

When checks fail and stakes high:

```
"I cannot commit a single reading with sufficient confidence."
```

### hold

No external output (projection pending or internal loop).

---

## CommittedState → rendering

Decoder reads **IR**, not raw DMF activations:

```
CommittedState.claims → template or structured generation
CommittedState.projection_artifact → grid renderer
CommittedState.frame_commits → scoped sections in text
```

**Non-next-token principle:** decoder is not autoregressive guessing; it maps approved structure to surface form. (Hybrid implementations may use small language modules constrained by IR.)

---

## Anti-patterns

**Do not run decoder without lucidity approval.**

```
BAD:  basins → decoder
GOOD: basins → lucidity → decoder
```

**Do not run projector without REQUEST_PROJECTION.**

```
BAD:  project every basin
GOOD: lucidity pre-check gates expensive rollouts
```

**Do not embed ARC-specific solvers in projector.**

```
BAD:  legend_solver(), frame_solver() branches
GOOD: universal grid IR + basin-parameterized programs
```

**Do not let decoder invent facts.**

```
BAD:  fluent paragraph adding unstated details
GOOD: forbid_invented_facts in decoder_policy
```

**Do not decode from raw activations.**

```
BAD:  DecoderInput { active_traces only }
GOOD: CommittedState + decoder_policy
```

If lucidity is weak, decoder policy must express uncertainty or refuse — never fill gaps fluently.

---

# 17. Final simple summary

```
Projector:    optional — tests what a basin implies when lucidity requires it
Decoder:    mandatory for output — expresses what lucidity approved only
```

Default flow:

```
Basins → Lucidity → Decoder
```

When projection is required:

```
Basins → Lucidity → Projector → Lucidity → Decoder
```

If lucidity is weak, decoder policy must express uncertainty or refuse — never fill gaps fluently.

---

## Summary

```
Projector:  optional consequence testing; universal IR; not task-specific solvers
Decoder:    expression-only renderer; gated by lucidity decoder_policy
Principle:  express truth, don't decide truth
Never:      decoder before lucidity; projector without REQUEST_PROJECTION
```

Together they close the loop from **internal hypothesis** to **external, auditable output** — without collapsing into an uncontrolled language model.
