# Binding & context-op

Scoped architecture spec for the lucid-model cognitive pipeline.

Two distinct stages: **binding** proposes plural structures; **context-op** assigns local scope and gates.

---

## Pipeline position

```
DMF / tracebank
→ binding             (plural candidate frames)
→ context-op          (scoped frames + gates — no global classification)
→ interference        (local support/conflict)
→ basins
→ lucidity
```

---

# Part 1 — Binding

## Role

Binding takes **active traces** and proposes **plural candidate structures** — event-like frames, relation frames, transform frames — with role or relation assignments. It does not choose the final meaning.

Binding output should be plural.

```
Binding does not say: "This is the meaning."
Binding says: "Here are plausible structures."
```

### Event-style example (text)

For:

```
I found money while kayaking which I placed in the bank.
```

Binding may output **multiple event frames**, not one collapsed parse:

```
CandidateFrame F1 {
    frame_type: event-like
    role_assignments:
        ACTION = t_found
        THEME  = t_money
        CONTEXT = t_kayaking
    confidence: .76
}

CandidateFrame F2 {
    frame_type: event-like
    role_assignments:
        ACTION = t_placed
        THEME  = t_money
        DESTINATION = t_bank
    confidence: .74
    unresolved_slots:
        bank_sense ambiguous
}
```

Binding preserves both. Context-op scopes them; lucidity commits later.

---

## Binding input contract

```
BindingInput {
    active_traces                 // from DMF
    trace_clusters
    novelty_signals
    conflict_signals
    perceptual_evidence_graph     // structural hints, markers, grouping
    affordance_hints              // optional learned role affordances
    relation_like_traces          // relational activations from DMF
    structural_hints              // event-like, transform-like (soft)
    prior_candidate_frames        // optional carryover
    compute_policy
}
```

---

## Binding output contract

```
BindingOutput {
    candidate_frames: [
        CandidateFrame {
            frame_id
            frame_type              // event-like | relation-like | transform-like | symbol-region-like
            role_assignments        // slot → trace_id
            relation_assignments    // optional
            member_evidence_refs
            confidence
            unresolved_slots        // explicit ambiguity
            support_traces
            conflict_traces
        }
    ]
    frame_competition_graph       // soft edges between frames
    binding_stability_score       // for lucidity
    audit_log
}
```

---

# Part 2 — Context-op

## 2. Context-op

Context-op is **scope control**, not context classification.

Its job:

```
Assign local scope and control policy so evidence does not globally contaminate all hypotheses.
```

It does **not** decide final meaning.

It does **not** label the whole input with one domain or task type.

It should **not** say:

```
the whole input is finance
the whole task is symbolic
the whole grid is a movement task
```

It **should** say:

```
this trace belongs to this local frame
this context affects this part, not that part
these hypotheses should remain alive
this part needs wider search
```

It answers:

```
Which traces should influence each other, under which local frame?
```

This is crucial because all active traces should **not** interfere globally.

---

## Context-op input contract

```
ContextOpInput {
    candidate_frames              // from binding
    active_traces
    trace_clusters
    uncertainty_margin_info       // from DMF
    perceptual_evidence_graph
    prior_context_frames          // discourse / multi-turn carryover
    lucidity_feedback             // SEARCH_WIDER, RECHECK_BINDING, etc.
    compute_policy
}
```

---

## Context-op output contract

```
ContextOpOutput {
    context_frames
    scoped_trace_assignments
    frame_link_graph
    scoped_hypothesis_pressure    // local soft pressure per frame — not global task label
    interference_gates
    local_basin_pressure          // soft local routing — does not choose basins
    ambiguity_policy
    compute_policy
}
```

---

## A. Context frames

Context-op creates **local scopes**, not a single global reading.

For the sentence:

```
I found money while kayaking which I placed in the bank.
```

Output:

```
ContextFrame F1 {
    frame_kind: event-like
    traces: [found, money, kayaking]
    scoped_hypothesis_pressure:   // local only — not a global domain label
        outdoor_discovery-like .61
        water_activity-like .54
}

ContextFrame F2 {
    frame_kind: event-like
    traces: [placed, money, bank]
    scoped_hypothesis_pressure:
        storage-like .51
        financial-institution-like .67
        destination-like .44
        river_bank-like .18
}
```

The important part:

```
kayaking mostly affects F1.
money affects both F1 and F2.
bank is mostly scoped to F2.
No frame owns the whole sentence alone.
```

---

## B. Scoped trace assignments

```
scoped_trace_assignments = [
    {trace: t_kayak,   primary_frame: F1, secondary_frames: [], gate: open},
    {trace: t_money,   primary_frame: F1, secondary_frames: [F2], gate: open},
    {trace: t_bank,    primary_frame: F2, secondary_frames: [], gate: open},
    {trace: t_found,   primary_frame: F1, secondary_frames: [], gate: open},
    {trace: t_placed,  primary_frame: F2, secondary_frames: [], gate: open}
]
```

---

## C. Frame link graph

Soft links between context frames (not hard merge):

```
frame_link_graph = [
    {from: F1, to: F2, link_type: shared_theme, weight: .63, shared_trace: t_money}
]
```

---

## D. Interference gates

Gates control **which frames may interfere with each other**:

```
interference_gates = [
    {frame_a: F1, frame_b: F2, gate: partial, reason: shared t_money only},
    {frame_a: F1, frame_b: F_global, gate: closed, reason: prevent kayak→finance leak}
]
```

Interference runs **inside** scoped frames; gates prevent global contamination.

---

## E. Local basin pressure

Context-op does **not** choose a basin. It may emit **soft local pressure** inside a scoped frame only.

```
local_basin_pressure = [
    {
        frame: F1,
        basin_pressure: [
            {basin: b12441, weight: .61},
            {basin: b3302, weight: .54}
        ]
    },
    {
        frame: F2,
        basin_pressure: [
            {basin: b00491, weight: .67},
            {basin: b08810, weight: .18}
        ]
    }
]
```

Again: soft, scoped, non-committing.

---

## F. Ambiguity & compute policy

```
ambiguity_policy: preserve_plural | allow_narrow | force_widen
compute_policy:   standard | deep_scope | cheap_pass
```

Lucidity loopback may force `force_widen` to add competing local policies (especially on contextual grid tasks).

---

# Grid / visual stress tests (e.g. ARC)

Same contracts. Use universal evidence types — not task-specific solver branches.

## Binding output

Candidate frames (plural):

```
CandidateFrame A { frame_type: transform-like, ACTION = t_move_like, THEME = t_object, ... }
CandidateFrame B { frame_type: attribute-map-like, THEME = t_object, RESULT = t_color_delta, ... }
CandidateFrame C { frame_type: symbol-region-like, SYMBOL = t_glyph, REGION = t_target, ... }
```

## Context-op output

Scope by region and example — not by hard-coded task family:

```
ContextFrame F_legend {
    scope: legend_or_key_region
    traces: [...]
    scoped_hypothesis_pressure: symbol_mapping-like .55
}

ContextFrame F_content {
    scope: main_canvas
    traces: [...]
}

ContextFrame F_pair_1 {
    scope: example_pair_1
    traces: [input_traces, output_traces]
}

ContextFrame F_cross {
    scope: cross_example_invariant
    traces: [...]
}
```

Interference then applies **local** support/conflict inside each scope. Projector may validate exact output when lucidity requests it.

---

# Context-op in one sentence

```
Context-op assigns local scope and gates so evidence supports the right hypotheses in the right places — without global contamination or whole-input classification.
```

---

# Final simple version

```
Binding input:    active traces + affordances + relation-like traces + structural hints
Binding output:    possible event/relation/transform frames (plural)

Context-op input:    possible frames + active traces + uncertainty + previous context
Context-op output:    local scoped frames + scoped trace influence + interference gates + ambiguity/compute policy
```

---

## Anti-patterns

**Do not emit one collapsed parse (binding).**

```
BAD:  single CandidateFrame with all roles filled and bank_sense=financial
GOOD: F1 + F2 + unresolved_slots on bank
```

**Do not globally classify domain (context-op).**

```
BAD:  task_domain: finance
GOOD: scoped_hypothesis_pressure inside F2 only
```

**Do not choose basins in context-op.**

```
BAD:  winner_basin: b00491
GOOD: local_basin_pressure soft weights only
```

**Do not disable interference globally.**

```
BAD:  all_frames_interfere: true
GOOD: interference_gates with partial/closed edges
```

**Do not hard-code ARC solver branches.**

```
BAD:  if legend_frame: activate_legend_solver
GOOD: F_legend scope + symbol-region candidate frames
```

---

## Summary

```
Binding:     plural candidate frames from active traces
Context-op:  local scope + gates + soft pressure (not classification)
Principle:   kayaking must not force bank=river globally; legend scope ≠ whole grid task label
Next:        interference applies local support/conflict inside scoped frames
Never:       final commit, projection, decoder output
```

Binding asks **what structures are plausible**; context-op asks **where each trace and hypothesis may act** — together they implement lazy collapse before basins.
