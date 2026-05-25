# Perception layer

Scoped architecture spec for the lucid-model cognitive pipeline.

---

## Role

Perception is the **first stage** of the pipeline. It converts raw multimodal input into a structured **evidence graph** that downstream modules can address without committing to meaning.

## Perception's role

Perception is **layered evidence parsing**, not object detection or token labeling with a single meaning attached.

It should answer:

```
What candidate parts, regions, relations, changes, and uncertainties are present?
```

It should **not** answer:

```
What does this definitely mean?
What is the rule?
What is the final interpretation?
```

So perception outputs an **evidence graph**, not interpretation.

Clean role:

```
Perception = layered evidence extractor
Cue encoder = trace-address compiler
DMF = learned trace activation field
```

---

# Layered evidence (universal)

Perception extracts **layers of candidate evidence** without deciding which layer or hypothesis wins.

**Text layers:**

```
words / spans
clauses
possible events
relation markers
reference hints
ambiguity flags
```

**Grid / image layers:**

```
cells / pixels
regions
objects
frames / containers
symbols / glyphs
background / canvas
spatial relations
before-after changes
uncertainty
```

Each layer adds structure to the evidence graph. None of them commit to a final rule or meaning.

Perception should say:

```
these are the candidate parts, regions, relations, changes, and uncertainties
```

Not:

```
this is the rule
this is the final meaning
```

Grid benchmarks (e.g. ARC) are **stress tests** for this pipeline — they should not hard-code task-specific solvers into perception.

---

## Pipeline position

```
raw input
→ perception          outputs PerceptualEvidenceGraph (layered, no meaning committed)
→ cue encoder         outputs CueCloud / CuePacket
→ DMF / tracebank     outputs active traces, clusters, novelty, conflict, margins
→ binding             outputs candidate structures (plural)
→ context-op          outputs scoped frames + gates (no global classification)
→ interference        outputs local support/conflict inside scopes
→ basins              outputs hypothesis states + assemblies
→ lucidity            commits / preserves ambiguity / may REQUEST_PROJECTION
→ projector           optional — only when lucidity requires consequence testing
→ lucidity            final check if projection ran
→ decoder             expresses committed state only
```

Perception has **no trace IDs**, **no basin IDs**, and **no learned semantics**. It emits surface structure only.

---

## Input contract

```
PerceptionInput {
    raw_payload              // text, grid tensor, audio frames, game frame, etc.
    modality                 // text | grid | image | audio | interactive
    task_intent_hint         // optional: solve_grid | answer | act | observe
    prior_context            // optional carryover from previous turn / frame
    compute_policy           // recall vs speed tradeoff for segmentation depth
    provenance_seed          // source id, timestamp, session id
}
```

### Grid / visual input

```
input_grid
output_grid examples
test_grid
```

### Text input

```
token_stream_or_sentence
optional_discourse_context
```

---

## Output contract

```
PerceptualEvidenceGraph {
    candidate_units          // spans, objects, cells, glyphs
    candidate_regions        // clauses, canvas, legend bands, background
    candidate_containers     // frames, hollow regions, event groupings
    candidate_markers        // relation markers, spatial relation hints
    arrangement_hints        // order, adjacency, alignment
    change_hints             // before-after deltas
    grouping_hints           // clause groups, object tracks, example pairs
    reference_hints          // pronouns, carryover, cross-frame links
    salience_scores
    confidence_scores
    uncertainty_flags
    provenance
}
```

## Perception output

```
PerceptualEvidenceGraph {
    candidate_units: [...]
    candidate_regions: [...]
    candidate_containers: [...]
    candidate_markers: [...]
    arrangement_hints: [...]
    change_hints: [...]
    grouping_hints: [...]
    reference_hints: [...]
    uncertainty_flags: [...]
    provenance: {...}
}
```

### Field definitions

```
CandidateUnit {
    unit_id
    raw_surface_or_region
    modality
    kind_hint                // span, connected_component, glyph_template, etc.
    type_hints               // soft labels only — not semantic IDs
    feature_signature        // color hash, shape hash, token form, etc.
    position_or_time
    confidence
    salience
    uncertainty              // optional
}

CandidateRegion {
    region_id
    role_hint                // background, legend_band, clause, canvas, etc.
    member_unit_ids
    confidence
    uncertainty
}

CandidateContainer {
    container_id
    kind_hint                // frame, hollow_region, event_grouping
    border_signature
    interior_region_id
    confidence
}

CandidateMarker {
    marker_id
    marker_surface_or_pattern
    marker_type_hints        // relation, temporal, causal-at-surface
    possible_targets
    confidence
}

ArrangementHint {
    hint_type                // left_of, inside, before, aligned_with
    source_unit
    target_unit
    weight
}

ChangeHint {
    before_unit
    after_unit
    change_type              // position_shift, color_delta, shape_preserved, created, deleted
    weight
}

GroupingHint {
    group_id
    member_units
    grouping_reason          // example_pair, object_track, clause_group
    confidence
}

ReferenceHint {
    source_unit
    target_unit_or_span
    reference_type           // pronoun, carryover, cross_frame
    confidence
}

UncertaintyFlag {
    target
    uncertainty_type
    severity                 // low | medium | high
}

Provenance {
    source
    modality
    timestamps_or_spans
    segmentation_pass_id     // auditable: which pass produced this graph
}
```

---

## Processing stages

A good perception layer has five stages:

```
1. normalize input
2. segment candidates across layers (units, regions, containers, symbols)
3. detect markers / relations at surface level
4. extract arrangement, change, grouping, and reference hints
5. emit layered evidence graph with uncertainty
```

### Stage notes

**Normalize:** unify coordinate systems, tokenization, color palettes, frame indexing.

**Segment across layers:** emit multiple plausible segmentations when needed; do not pick one winner.

**Markers at surface level:** detect "which", "while", "in", adjacency, alignment — not what they mean.

**Hints:** arrangement, change, grouping, reference — all weighted, all plural where ambiguity exists.

**Emit with uncertainty:** high recall, low commitment. Missing evidence is worse than extra candidates.

---

## Text example: bank / kayaking sentence

Input:

```
I found money while kayaking which I placed in the bank.
```

Perception output (abbreviated):

```
PerceptualEvidenceGraph {
    candidate_units: [
        {id: u_found,  kind_hint: verb_span,      surface: "found",      confidence: .98},
        {id: u_money,  kind_hint: noun_span,      surface: "money",      confidence: .97},
        {id: u_kayak,  kind_hint: gerund_clause,  surface: "kayaking",   confidence: .95},
        {id: u_placed, kind_hint: verb_span,      surface: "placed",     confidence: .96},
        {id: u_bank,   kind_hint: noun_span,      surface: "bank",       confidence: .94}
    ]
    candidate_regions: [
        {id: r_main, role_hint: main_clause, members: [u_found, u_money, u_kayak]},
        {id: r_rel,  role_hint: relative_clause, members: [u_placed, u_bank]}
    ]
    candidate_markers: [
        {id: m_while, marker_type_hints: [temporal_subordinate], targets: [u_kayak], confidence: .91},
        {id: m_which, marker_type_hints: [relative_pronoun], targets: [u_placed], confidence: .93},
        {id: m_in,    marker_type_hints: [locative_preposition], targets: [u_bank], confidence: .89}
    ]
    arrangement_hints: [
        {hint: u_kayak_subordinate_to_u_found, weight: .84},
        {hint: u_placed_follows_u_found_event, weight: .79}
    ]
    reference_hints: [
        {source: u_placed, target: u_money, reference_type: object_carryover, confidence: .72}
    ]
    uncertainty_flags: [
        {target: u_bank, uncertainty_type: "polysemy_surface_form", severity: medium}
    ]
}
```

Perception does **not** say financial bank vs river bank. It flags polysemy and emits structure for the cue encoder.

---

## Grid example (visual stress test)

Input: grid training pair (e.g. ARC-style task).

Perception output — layered, no rule committed:

```
PerceptualEvidenceGraph {
    candidate_regions: [
        {id: r_canvas, role_hint: background, confidence: .89},
        {id: r_legend, role_hint: key_or_legend_band, confidence: .71, uncertainty: high}
    ]
    candidate_units: [
        {id: u1, kind_hint: connected_component, color_signature: c1, bbox: ..., confidence: .96},
        {id: u2, kind_hint: connected_component, color_signature: c2, bbox: ..., confidence: .93}
    ]
    candidate_containers: [
        {id: f1, kind_hint: frame_or_hollow_region, border_signature: ..., confidence: .68}
    ]
    candidate_units_symbols: [
        {id: g1, kind_hint: glyph_template, pattern_hash: ..., confidence: .74, uncertainty: medium}
    ]
    arrangement_hints: [
        {hint: u1_left_of_u2, weight: .82},
        {hint: u1_inside_region_r_canvas, weight: .58},
        {hint: g1_in_region_r_legend, weight: .55}
    ]
    change_hints: [
        {before: u1_input, after: u1_output, change: position_shift, weight: .88},
        {before: u1_input, after: u1_output, change: shape_preserved, weight: .93},
        {before: u1_input, after: u1_output, change: color_delta, weight: .61}
    ]
    grouping_hints: [
        {group: example_pair_1, members: [input_grid_1, output_grid_1], confidence: 1.0},
        {group: object_track_1, members: [u1_input, u1_output], confidence: .81}
    ]
    uncertainty_flags: [
        {target: object_track_1, uncertainty_type: "identity_match_uncertain", severity: medium},
        {target: r_legend, uncertainty_type: "region_role_uncertain", severity: medium}
    ]
}
```

Then cue encoder generates a cue cloud with pressure toward learned traces — not toward a hand-built legend solver or frame solver.

---

## Anti-patterns

**Do not attach learned trace IDs in perception.**

```
BAD:  unit_id: t_money
GOOD: unit_id: u_money, kind_hint: noun_span
```

**Do not emit a single global parse or segmentation.**

```
BAD:  one object correspondence, one clause tree, one rule family
GOOD: multiple tracks, multiple region hypotheses, uncertainty flags
```

**Do not classify the whole task.**

```
BAD:  task_type: symbolic_legend_task
GOOD: role_hint: key_or_legend_band, uncertainty: high
```

**Do not hard-code ARC solvers.**

```
BAD:  if legend_detected: run_legend_solver()
GOOD: emit legend region + glyph units + change hints; let pipeline compete
```

**Do not collapse ambiguity early.**

```
BAD:  bank_sense: financial_institution
GOOD: uncertainty_flag on u_bank; reference_hints for carryover
```

**Do not produce low-recall segmentation to save compute on hard inputs.**

Perception should prefer **extra candidates** over **missing structure**. Downstream lazy collapse and lucidity handle cost.

---

## Auditability requirements

Every perception run must log:

```
input hash
modality adapter version
segmentation pass list
full PerceptualEvidenceGraph (or diff from prior frame)
uncertainty summary
```

Human editors must be able to:

```
add/remove candidate units
split/merge regions
adjust change hints
re-run cue encoder from edited graph
```

---

## Summary

```
Perception = layered evidence extractor
Input:     raw multimodal payload
Output:    PerceptualEvidenceGraph (plural, uncertain, no meaning)
Next:      cue encoder compiles activation requests from evidence
Principle: lazy collapse starts here — never commit rule or final meaning
Stress:    grids (ARC-style) use same contract; no task-specific solver branches
```

Perception is the foundation of auditability. If this stage lies, nothing downstream can recover honestly — but if it preserves ambiguity faithfully, the rest of the pipeline can compete hypotheses under lucidity.
