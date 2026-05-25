# Cue encoder & cue cloud

Scoped architecture spec for the lucid-model cognitive pipeline.

---

## Role

The cue encoder is the **trace-address compiler**. It converts perceptual evidence (and optional upstream state) into a **CueCloud**: a sparse, weighted activation request over the DMF / tracebank.

With **lazy collapse**, the cue encoder's role is:

```
Turn incoming evidence into an ambiguity-preserving activation request for the DMF / tracebank.
```

**Core principle:** the cue encoder decides which possible meanings **deserve to compete** — not which meaning is true.

It should **not** decide the final interpretation.

It should **not** bind everything into one meaning.

It should **not** collapse context early.

It should **not** say:

```
this definitely means X
```

It should say:

```
these possible meanings deserve to compete
```

The cue encoder emits **pressure**, not **commitment**. Learned trace IDs (`t0012`, `t8840`) appear here for the first time in the pipeline — as activation targets, not as hard-coded human concepts.

---

## Pipeline position

```
perception / upstream state
        ↓
cue encoder
        ↓
CueCloud:
    primitive trace activations
    relational trace activations
    soft context priors
    weak structure hints
    ambiguity policy
        ↓
DMF / tracebank
        ↓
active traces, clusters, novelty, conflict, margins
        ↓
binding / context-op / interference
        ↓
basins
        ↓
lucidity
        ↓
[projector if REQUEST_PROJECTION]
        ↓
decoder
```

---

## Input contract

```
CueEncoderInput {
    perceptual_evidence_graph     // primary source
    upstream_state                // optional: prior turn, working memory handles
    task_intent_hint              // optional routing hint — not global classification
    retrieval_budget              // max traces to request, width, depth
    ambiguity_policy_in           // preserve | narrow | widen (from lucidity loopback)
    compute_policy                // cheap | standard | deep retrieval
    provenance
}
```

---

## Output contract

```
CueCloud {
    primitive_trace_activations   // [{trace_id, weight, evidence_refs}]
    relational_trace_activations  // [{trace_id, weight, relation_refs, endpoints}]
    soft_context_priors           // local soft pressure — not domain labels
    weak_structure_hints          // frame-like, event-like, transform-like — soft only
    ambiguity_policy              // preserve_plural | allow_narrow | force_widen
    retrieval_budget_used
    suppression_list              // traces explicitly downweighted this pass
    provenance
}

CuePacket {
    // compact serial form for DMF lookup / logging
    cloud_id
    top_k_trace_ids
    activation_vector_or_sparse_list
    policy_flags
}
```

### Activation entry rules

```
weight ∈ [0, 1]           // soft request strength
evidence_refs             // links back to PerceptualEvidenceGraph nodes
no hard winner            // multiple competing activations expected
floor_threshold           // drop noise below ε, but keep ambiguity bands
```

---

## Internal processing

### 1. Evidence → activation mapping

Map surface evidence to **learned trace families** without naming human concepts internally:

```
noun_span "bank"     → activate competing trace clusters (not one winner)
change: position_shift → activate motion-like trace families
region: legend_band  → activate symbol-region trace families (soft)
```

Human aliases (`BANK`, `MOVE_LEFT`) may exist for debugging; runtime uses `t####` only.

### 2. Relational compilation

Relation markers and arrangement hints compile to **relational trace activations**:

```
u1_left_of_u2  → relational activation with endpoints [u1, u2]
u_money carryover → relational activation linking spans/clauses
```

### 3. Ambiguity policy

```
preserve_plural (default):
    keep competing activations above floor
    do not zero losers early

allow_narrow:
    when lucidity previously signaled stable margin upstream

force_widen:
    when lucidity returned SEARCH_WIDER — increase retrieval_budget
```

### 4. Retrieval budget

Controls DMF cost:

```
retrieval_budget {
    max_primitive_activations
    max_relational_activations
    max_cluster_expansion_depth
}
```

Wide clouds cost more; lucidity and training governor feed back when widening was necessary.

---

## Text example: bank / kayaking sentence

Input evidence (from perception): spans, clauses, markers, `bank` polysemy flag.

CueCloud (abbreviated):

```
CueCloud {
    primitive_trace_activations: [
        {trace: t_found_like,   weight: .82, evidence: [u_found]},
        {trace: t_money_like,   weight: .79, evidence: [u_money]},
        {trace: t_kayak_like,   weight: .76, evidence: [u_kayak]},
        {trace: t_placed_like,  weight: .74, evidence: [u_placed]},
        {trace: t_bank_fin_like, weight: .58, evidence: [u_bank]},
        {trace: t_bank_river_like, weight: .41, evidence: [u_bank]}
    ]
    relational_trace_activations: [
        {trace: t_while_subord, weight: .71, endpoints: [u_found, u_kayak]},
        {trace: t_object_carry, weight: .63, endpoints: [u_money, u_placed]}
    ]
    weak_structure_hints: [
        {hint: event_like, weight: .68},
        {hint: locative_destination, weight: .52}
    ]
    ambiguity_policy: preserve_plural
}
```

Note: both bank-related traces stay alive. Cue encoder does not pick financial vs river.

---

## Grid example (visual stress test)

Input evidence: legend region uncertain, glyph template, object tracks, change hints.

```
CueCloud {
    primitive_trace_activations: [
        {trace: t_object_like,      weight: .77, evidence: [u1, u2]},
        {trace: t_glyph_like,       weight: .62, evidence: [g1]},
        {trace: t_legend_region_like, weight: .55, evidence: [r_legend]},
        {trace: t_position_shift_like, weight: .71, evidence: [change_hints]}
    ]
    relational_trace_activations: [
        {trace: t_inside_like, weight: .54, endpoints: [u1, r_canvas]},
        {trace: t_in_legend_like, weight: .48, endpoints: [g1, r_legend]}
    ]
    ambiguity_policy: preserve_plural
    retrieval_budget_used: {primitive: 12, relational: 6}
}
```

No branch says `run_legend_solver`. Multiple transform and symbol hypotheses compete via trace activation.

---

## Anti-patterns

**Do not collapse to one trace per surface form.**

```
BAD:  bank → t_bank_fin_only
GOOD: bank → competing trace activations + ambiguity_policy preserve_plural
```

**Do not bind roles or frames here.**

```
BAD:  ACTION=t_found, THEME=t_money in CueCloud
GOOD: primitive + relational activations only; binding comes later
```

**Do not globally classify domain or task type.**

```
BAD:  soft_context_priors: {finance: .9}
GOOD: local soft priors tied to evidence refs, plural hypotheses
```

**Do not skip audit links.**

Every activation must cite `evidence_refs` back to the perceptual graph.

**Do not use human concept names as internal IDs.**

```
BAD:  trace: WITHDRAW
GOOD: trace: t2931 (alias optional in debug tooling)
```

---

## Loopback from lucidity

When lucidity returns `SEARCH_WIDER` or `RECHECK_BINDING`:

```
increase retrieval_budget
force_widen ambiguity_policy
optionally re-compile from edited perception graph
```

When lucidity returns high margin pass on prior pass:

```
allow_narrow may reduce redundant activations next cycle
```

---

## Summary

```
Cue encoder = trace-address compiler
Input:     PerceptualEvidenceGraph + policy + budget
Output:    CueCloud / CuePacket (sparse activation hypergraph)
Principle: these meanings deserve to compete — not this is the meaning
Next:      DMF returns activation landscape (traces, clusters, margins)
Never:     final interpretation, role binding, basin choice, decoder output
```

The cue cloud is the bridge from **editable evidence** to **learned trace memory**. Its quality determines whether the right hypotheses enter competition at all.
