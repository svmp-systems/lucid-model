# DMF / tracebank

Scoped architecture spec for the lucid-model cognitive pipeline.

---

# DMF / tracebank

Traces are **learned IDs**, not manually inserted concepts:

```
t0012
t8840
t2931
```

Do **not** hard-code internal concepts like `WITHDRAW`, `MONEY`, `BANK`, or `LEFT_OF`. Human-readable names may become **aliases later**; internally the system uses learned trace IDs only.

A trace gains meaning from use:

```
what activates it
what it coactivates with
which basins it supports
which outputs it helps
which lucidity checks it survives
```

The DMF is not a dictionary. It returns an **activation landscape** for downstream stages to bind, scope, and test.

---

# Required DMF output (summary)

The DMF must output at minimum:

```
active traces
trace clusters
novelty signals
conflict signals
uncertainty / margin info
```

Not a final interpretation.

---

## Role

The **Dynamic Memory Field (DMF)** / **tracebank** is the learned associative memory that responds to a CueCloud with an activation landscape. It retrieves and adjusts trace activations; it does not parse input or commit answers.

---

## Pipeline position

```
cue encoder → DMF / tracebank → binding → context-op → interference → basins → lucidity
```

---

# What input goes into the DMF?

```
DMFInput {
    cue_cloud                    // CuePacket or full CueCloud
    tracebank_snapshot_id        // versioned bank state for audit
    heat_policy                  // hot / warm / cold trace tiers (continual learning)
    quarantine_filter            // exclude or cap provisional traces
    prior_active_traces          // optional carryover from previous cognitive step
    compute_policy
}
```

---

## Output contract

```
DMFOutput {
    active_traces: [
        {
            trace_id               // t0012
            activation
            heat_tier              // hot | warm | cold | frozen
            cluster_id             // optional
            evidence_alignment     // how well cue refs match
            provenance
        }
    ]
    trace_clusters: [
        {
            cluster_id
            member_trace_ids
            cluster_activation
            cluster_coherence
        }
    ]
    novelty_signals: [
        {
            region_or_evidence_ref
            novelty_score
            suggested_action         // spawn_provisional | widen_search | flag_quarantine
        }
    ]
    conflict_signals: [
        {
            trace_a
            trace_b
            conflict_strength
            scope_hint               // local frame if known from prior context
        }
    ]
    uncertainty_margin_info: {
        top_trace_margin
        cluster_margin
        activation_entropy
        coverage_score               // fraction of evidence with supporting traces
    }
    adjusted_activations           // post-inhibition / post-boost landscape
    audit_log
}
```

---

## Trace lifecycle

Traces move through explicit states (aligned with continual learning):

```
seed → provisional → active → stabilized → crystallized → aliased → frozen
```

| State | Description |
|-------|-------------|
| seed | initialized from perceptual ops or synthetic curriculum |
| provisional | quarantined; weak links; hot storage |
| active | participates in binding/interference |
| stabilized | repeated lucidity success |
| crystallized | high margin, low edit rate |
| aliased | human-readable label attached (optional) |
| frozen | cold/quantized; edits require explicit thaw |

Promotion and demotion are **lucidity-gated**, not silent weight drift.

---

## Internal mechanics

### 1. Sparse retrieval

Given CueCloud activations, retrieve top-k traces and expand clusters to bounded depth. Never materialize full bank per pass.

### 2. Coactivation adjustment

Boost traces that coactivated historically in similar contexts; apply soft inhibition to conflicting pairs (learned interference links feed later stages).

### 3. Novelty detection

When evidence aligns poorly with all active traces:

```
novelty_score high → suggest provisional trace spawn (quarantine)
```

### 4. Margin computation

Compute margins early for training governor and lucidity:

```
top_trace_margin = act(top1) - act(top2)
cluster_margin   = act(best_cluster) - act(second_cluster)
```

High margin downstream may trigger **no-update** during training.

---

## Text example: bank / kayaking sentence

CueCloud activates `t_found_like`, `t_money_like`, `t_kayak_like`, `t_placed_like`, competing bank traces.

```
DMFOutput {
    active_traces: [
        {trace: t0142, activation: .81, cluster: c_event},
        {trace: t0881, activation: .78, cluster: c_transfer},
        {trace: t2204, activation: .74, cluster: c_outdoor},
        {trace: t00491, activation: .62, cluster: c_finance},
        {trace: t08810, activation: .44, cluster: c_location_water}
    ]
    trace_clusters: [
        {cluster: c_event, members: [t0142, t2204], cluster_activation: .79},
        {cluster: c_transfer, members: [t0881, t00491], cluster_activation: .71}
    ]
    conflict_signals: [
        {trace_a: t00491, trace_b: t08810, conflict_strength: .55, scope_hint: bank_span}
    ]
    uncertainty_margin_info: {
        top_trace_margin: .03,
        cluster_margin: .08,
        activation_entropy: .71,
        coverage_score: .84
    }
}
```

Low top trace margin → binding and lucidity must preserve plural hypotheses.

---

## Grid example (visual stress test)

CueCloud activates object, glyph, legend-region, and motion-like families.

```
DMFOutput {
    active_traces: [
        {trace: t5502, activation: .76, cluster: c_transform},
        {trace: t2011, activation: .73, cluster: c_transform},
        {trace: t9100, activation: .68, cluster: c_symbol},
        {trace: t3340, activation: .61, cluster: c_color_map}
    ]
    novelty_signals: [
        {region: r_legend, novelty_score: .42, suggested_action: widen_search}
    ]
    uncertainty_margin_info: {
        top_trace_margin: .03,
        cluster_margin: .05,
        coverage_score: .79
    }
}
```

Multiple clusters stay active for binding to form plural transform/symbol frames.

---

## Anti-patterns

**Do not store human concept names as primary keys.**

```
BAD:  trace_name: MONEY
GOOD: trace_id: t0881, alias: "money-like" (debug only)
```

**Do not return a single winning trace as final answer.**

```
BAD:  winner: t00491
GOOD: active_traces + margins + conflicts (plural until lucidity)
```

**Do not silently create traces without quarantine.**

All new traces enter as **provisional** with audit record.

**Do not global-classify input domain in DMF.**

No field `domain: finance`. Use clusters and conflict signals only.

**Do not skip margin / coverage outputs.**

Lucidity and training governor depend on these fields.

---

## Editability & auditability

Human and machine editors must be able to:

```
inspect activation path for a trace
adjust trace heat tier
merge/split clusters (with audit)
freeze / thaw traces
view coactivation and basin link history
```

Every DMFOutput logs `tracebank_snapshot_id` for reproducibility.

---

## Summary

```
DMF / tracebank = learned trace activation field
Input:     CueCloud + bank snapshot + heat/quarantine policy
Output:    active traces, clusters, novelty, conflict, margins
Principle: learned IDs (t####); meaning from use; not a dictionary lookup
Next:      binding proposes structures from active traces (plural)
Never:     final interpretation, scoped frames, basin commit, decoder output
```

The DMF is where **memory meets ambiguity**. It amplifies what deserves to compete and reports whether the bank recognizes the evidence — without collapsing early.
