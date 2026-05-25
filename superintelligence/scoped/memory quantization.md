# Memory quantization

Scoped architecture spec for the lucid-model cognitive pipeline.

---

## Role

Memory quantization makes traces, basins, and interference links **compact and fast to retrieve** while preserving enough fidelity for lazy collapse and lucidity. Quantization is **tiered** — hot/warm traces stay higher precision; crystallized/frozen structures move to low-bit codes.

Aligned with core principles:

```
memory must be quantizable (principle.md)
full auditability — quantized codes must decode to inspectable form
full editability — thaw/repair paths for frozen entries
```

This module defines **storage and retrieval formats**, not training policy (see `training quantization.md`).

---

## Pipeline touchpoints

```
DMF / tracebank     → quantized trace codes
Interference        → ternary interference links
Basins              → low-bit basin density vectors
Lucidity / search   → popcount energy proxies for tier-0 pruning
Ambiguous cases     → residual PQ vectors alongside binary/ternary base
```

---

## Design goals

1. **Cheap top-k retrieval** — popcount / Hamming distance over binary codes before fp16 refinement.
2. **Preserve conflict semantics** — ternary (+1 / 0 / -1) interference links.
3. **Tiered precision** — fp16 → 8-bit → 4-bit → binary by heat tier.
4. **Residual PQ for hard cases** — when margin low, load residuals without promoting everything to fp16.
5. **Auditable decode** — every quantized record stores `codebook_id`, `version`, `thaw_recipe`.

---

## Trace quantization

### Binary traces (cold / frozen)

Each trace `t####` stored as:

```
QuantizedTrace {
    trace_id
    binary_code              // e.g. 256-bit signature
    codebook_id
    heat_tier: cold | frozen
    alias_optional
    decode_metadata
}
```

**Use:** fast candidate retrieval — "which traces share bits with this cue pattern?"

Retrieval:

```
candidate_set = top_k_by_popcount(cue_binary_proxy, trace_bank, k)
refine with 8-bit or fp16 on hot/warm subset only
```

### Ternary traces (warm)

For traces needing signed feature dimensions:

```
TernaryTraceCode {
    trace_id
    ternary_vector           // per dimension: +1, 0, -1 packed
    packing: 2 bits per dim or dedicated ternary codec
    heat_tier: warm
}
```

Ternary captures **directional** features (e.g. axis of motion, polarity of relation) better than binary alone.

---

## Interference link quantization

Learned interference stored as **ternary links**:

```
QuantizedInterferenceLink {
    trace_a
    trace_b
    link_value               // +1 support, 0 none, -1 conflict
    scope_family             // optional bucket — not global task label
    precision_tier
}
```

Four interference levels (trace↔trace, trace↔frame, frame↔basin, basin↔basin) may use separate codebooks but same ternary semantics.

**Popcount energy proxy:** approximate basin energy delta without full fp16 matmul:

```
energy_proxy += popcount(trace_activation_bits AND basin_density_bits)
energy_penalty -= popcount(conflict_link_bits)
```

Used in **tier-0 search** before partial projection (see lucidity + ARC stress tests).

---

## Basin quantization

Basins `b####` use **low-bit density vectors**:

```
QuantizedBasin {
    basin_id
    density_code             // 4-bit or binary vector over trace/basis dimensions
    assembly_link_codes      // optional compressed cooperation edges
    heat_tier
    margin_history_summary   // for training governor no-update decisions
}
```

Basin assemblies reference member basin codes + assembly coherence hash.

---

## Tiered precision ladder

| Tier | Heat | Storage | When loaded |
|------|------|---------|-------------|
| fp16 | hot | full vector | active provisional traces, quarantine testing |
| 8-bit | warm | scalar quantized | active retrieval refinement |
| 4-bit | warm/cold | density codes | basin field scan |
| binary | cold | trace/basin signatures | tier-0 popcount retrieval |
| ternary | warm/cold | interference + directional traces | conflict/support fast path |

Promotion/demotion tied to continual learning heat types (see `continual learning.md`).

---

## Residual PQ (Product Quantization)

When binary/ternary retrieval returns **ambiguous top-k** (low margin):

```
QuantizedRecordWithResidual {
    base_code                // binary or ternary
    pq_residual_subvectors   // M subcodes × log2(K) bits each
    residual_scale
}
```

**Use:**

```
tier-0: popcount on binary base → top-64
tier-0.5: add PQ distance → top-16
tier-1: load fp16 hot traces for final top-k only
```

Avoids storing full fp16 for entire bank while preserving accuracy on hard disambiguation (bank/kayaking, legend/grid tasks).

---

## Popcount energy (tier-0 proxy)

Before expensive binding/basin/projection passes:

```
PopcountEnergyInput {
    cue_binary_proxy
    active_trace_binary_codes
    basin_density_codes
    ternary_conflict_mask
}

PopcountEnergyOutput {
    basin_rankings_approx
    margin_approx
    recommend_widen: bool
    recommend_load_residuals: bool
}
```

If `margin_approx` above threshold → skip deep retrieval widening (compute savings).

If below → load PQ residuals or force lucidity `SEARCH_WIDER`.

---

## Text example: bank / kayaking

Cue compiles to binary proxy activating money/bank/event bits.

```
tier-0 retrieval:
    t00491 (finance) popcount 142
    t08810 (river)   popcount 118
    margin_approx low → load PQ residuals for bank cluster
tier-1:
    refine t00491, t08810 in fp16 warm tier
```

Interference ternary link `(t_kayak, t_finance) = -1` applied in F1 scope only — encoded in scoped link bucket.

---

## Grid example (visual stress test)

```
tier-0:
    b5502 popcount high on motion bits
    b9100 popcount high on glyph bits
    separate scopes — no global merge
tier-0.5:
    assembly {b5502,b3340} cooperation bits reinforce
before projection:
    lucidity uses margin_approx to pick top-3 assemblies only
```

---

## Anti-patterns

**Do not quantize away auditability.**

```
BAD:  opaque blob with no decode_metadata
GOOD: codebook_id + version + thaw_recipe
```

**Do not use single global binary code for whole input.**

Scoped frames need scoped code buckets or tagged dimensions.

**Do not skip residuals on all low-margin cases.**

Binary-only retrieval fails on polysemy and symbol tasks — use PQ.

**Do not store interference as float32 dense matrix at scale.**

Use ternary sparse + popcount.

**Do not thaw frozen traces silently.**

Thaw requires explicit policy log entry.

---

## Measurement requirements (pre-production)

Before relying on quantization for cost claims:

```
measure top-k recall@k: binary vs fp16 on held-out activation logs
measure margin preservation after popcount proxy
measure ARC-style search time tier-0 vs tier-2
```

Walkthrough notes: win only materializes if low-bit retrieval preserves top-k quality.

---

## Summary

```
Memory quantization = compact trace/basin/link storage + fast retrieval
Formats:   binary/ternary traces, ternary interference, low-bit basin density
Speed:     popcount energy proxy for tier-0 pruning
Quality:   tiered precision + residual PQ for ambiguous cases
Principle: quantizable, auditable, editable via thaw/repair
```

Quantization shifts the cost curve — it does not replace lucidity or plural hypothesis competition.
