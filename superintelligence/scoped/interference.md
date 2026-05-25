# Interference

Scoped architecture spec for the lucid-model cognitive pipeline.

---

## Role

Interference is the module that takes the **scoped candidate structures** from context-op and turns them into **local support / conflict pressure** over basins.

**Key rule:** interference is **not global**. Traces affect basins **inside scoped frames**, not across the whole input at once.

In your architecture:

```
DMF / tracebank
→ binding
→ context-op
→ interference        (local only)
→ learned basin field
→ lucidity collapse
```

Interference answers:

```
Inside each scoped frame:
    which traces, bindings, and structures support each other?
    which ones conflict?
    how should basin energies change before lucidity chooses?
```

It does **not** choose the final answer.  
It shapes the energy landscape with **soft pressure** — not hard deletion of alternatives.

It outputs:

```
positive support edges
negative support edges
basin energy changes
cooperation / competition maps
conflict reports
```

---

## Pipeline position

```
context-op → interference → basins → lucidity
```

Interference reads **interference_gates** from context-op and never applies pressure across closed gates.

---

## Input contract

```
InterferenceInput {
    active_traces
    adjusted_trace_activations_from_DMF
    candidate_frames_from_binding
    context_frames_from_contextop
    scoped_trace_assignments
    frame_link_graph
    local_basin_pressure          // from context-op — scoped only
    learned_interference_links
    uncertainty_map
    conflict_map
    task_constraints
    compute_policy
}
```

Also acceptable compact form:

```
InterferenceInput {
    active_traces
    candidate_frames
    context_frames
    scoped_trace_assignments
    frame_link_graph
    local_basin_pressure
    learned_interference_links
    task_constraints
    uncertainty_map
    compute_policy
}
```

---

## Output contract

```
InterferenceOutput {
    positive_support_edges: [
        {scope: frame_id, source: trace_or_frame, target: trace_or_frame_or_basin, weight}
    ]
    negative_support_edges: [
        {scope: frame_id, source, target, weight}
    ]
    basin_energy_deltas: [
        {scope: frame_id, basin_id, delta, reason_refs}
    ]
    cooperation_maps: [
        {scope: frame_id, basin_ids: [...], cooperation_strength}
    ]
    competition_maps: [
        {scope: frame_id, basin_ids: [...], competition_strength}
    ]
    conflict_reports: [
        {scope: frame_id, conflict_type, members, severity}
    ]
    interference_summary: {
        frames_processed
        gates_honored
        net_energy_shift_by_basin
    }
    audit_log
}
```

---

## Four interference levels (all local to a scope)

### Level 1 — trace ↔ trace

Inside a context frame, learned links modulate coactivation:

```
t_kayak + t_outdoor  → positive support
t_kayak + t_finance  → negative support (within wrong scope — gated)
```

Links may be stored as **ternary** (+1 / 0 / -1) in quantized memory (see `memory quantization.md`).

### Level 2 — trace ↔ frame

Candidate frames gain energy when member traces reinforce role assignments:

```
F1 + t_found + t_money  → positive support to frame F1
F2 + t_bank_fin + t_bank_river simultaneously → conflict report
```

### Level 3 — frame ↔ basin

Frames push soft energy toward basins that historically explain similar assemblies:

```
F2 transform-like frame → +ΔE on b5502, b2011 (within F_content scope)
```

Uses **local_basin_pressure** as prior only — never as winner.

### Level 4 — basin ↔ basin

Basins cooperate (assembly) or compete inside a scope:

```
b_move + b_recolor → cooperation_map (assembly candidate)
b_move vs b_mirror → competition_map
```

---

## E. Local basin pressure (from context-op)

Context-op may pass soft scoped pressure. Interference treats it as **local prior only** — never a global basin winner.

```
local_basin_pressure = [
    {frame: F2, basin: b00491, weight: .67},
    {frame: F2, basin: b08810, weight: .18}
]
```

Interference still computes support/conflict from traces and bindings **inside F2**.

---

## Processing algorithm (conceptual)

```
for each context_frame F (respecting interference_gates):
    collect traces and candidate_frames scoped to F
    apply learned_interference_links (trace level)
    score frame internal consistency (trace ↔ frame)
    translate frame support into basin_energy_deltas (frame ↔ basin)
    detect basin cooperation/competition (basin ↔ basin)
    merge local_basin_pressure as soft prior
    emit edges + deltas + conflict_reports (all tagged with scope F)
```

No cross-frame delta unless `frame_link_graph` and gates explicitly allow partial sharing (e.g. shared `t_money`).

---

## Text example: bank / kayaking sentence

```
InterferenceOutput {
    positive_support_edges: [
        {scope: F1, source: t_kayak, target: t_outdoor_cluster, weight: .71},
        {scope: F1, source: t_found, target: F1, weight: .68},
        {scope: F2, source: t_placed, target: F2, weight: .66},
        {scope: F2, source: t_bank_fin, target: b00491, weight: .59}
    ]
    negative_support_edges: [
        {scope: F1, source: t_kayak, target: t_bank_fin, weight: .62},
        {scope: F2, source: t_bank_fin, target: t_bank_river, weight: .48}
    ]
    basin_energy_deltas: [
        {scope: F1, basin: b12441, delta: +.12, reason: outdoor frame support},
        {scope: F2, basin: b00491, delta: +.15, reason: finance-local pressure + frame support},
        {scope: F2, basin: b08810, delta: +.04, reason: weak river_bank pressure}
    ]
    conflict_reports: [
        {scope: F2, conflict_type: ambiguous_destination, members: [b00491, b08810], severity: medium}
    ]
}
```

Interference keeps both bank basins alive in F2 with a conflict report — lucidity decides later.

---

# 6. Example: grid / visual stress test

Scopes: `F_legend`, `F_content`, `F_pair_1`, `F_cross`.

```
InterferenceOutput {
    basin_energy_deltas: [
        {scope: F_legend, basin: b9100, delta: +.18, reason: glyph + legend scope},
        {scope: F_content, basin: b5502, delta: +.14, reason: transform frame A},
        {scope: F_content, basin: b3340, delta: +.11, reason: attribute-map frame B},
        {scope: F_cross, basin: b5502, delta: +.09, reason: cross-pair invariant}
    ]
    cooperation_maps: [
        {scope: F_content, basin_ids: [b5502, b3340], cooperation_strength: .52}
    ]
    competition_maps: [
        {scope: F_content, basin_ids: [b5502, b9100], competition_strength: .44}
    ]
}
```

Legend scope does not globally suppress canvas transform hypotheses — gates keep scopes separate.

---

## Anti-patterns

**Do not apply global basin energy shifts.**

```
BAD:  b00491 += .2 everywhere
GOOD: basin_energy_deltas tagged with scope frame_id
```

**Do not hard-delete losing basins.**

```
BAD:  remove b08810 from field
GOOD: negative_support_edges + small delta; lucidity collapses
```

**Do not treat local_basin_pressure as final decision.**

```
BAD:  winner = argmax(local_basin_pressure)
GOOD: prior merged with trace/frame evidence inside scope
```

**Do not interfere across closed gates.**

Kayaking traces must not penalize finance basins in F2 via F1 leakage.

**Do not commit output or run projection.**

Interference shapes energy only.

---

## Quantization note

At scale, interference links and basin energy proxies may use **popcount** over binary/ternary codes (see `memory quantization.md`). Logical semantics remain: local soft pressure, auditable edges.

---

## Summary

```
Interference = local support/conflict pressure over basins
Input:     scoped frames, gates, traces, frames, local_basin_pressure, learned links
Output:    support edges, energy deltas, cooperation/competition maps, conflicts
Principle: never global; soft pressure; plural hypotheses survive until lucidity
Next:      basins accumulate energy and form assemblies
Never:     final commit, projection, decoder
```

Interference is the **physics layer** of hypothesis competition — local forces, not global verdicts.
