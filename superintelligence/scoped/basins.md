# Basins

Scoped architecture spec for the lucid-model cognitive pipeline.

---

## Role

Basins are **learned hypothesis attractors** — not hand-coded rules like `move_left` or `financial_bank`. Internally they use learned IDs:

```
b0001
b12441
b00491
```

A basin accumulates energy from local interference, cooperates or competes with other basins, and may form **basin assemblies** (e.g. move + recolor; found-event + placed-event). Basins propose **candidate interpretations**; they do **not** final-commit. Lucidity does that.

**Universal definition:**

```
A basin is a learned stable pattern:
  "these traces + bindings + context + interference belong together
   for this frame."
```

Meaning comes from what it stabilizes (supporting traces, frames, projection history) — not from a hand-assigned label like `financial_bank`. Decoder aliases may exist for humans; internal IDs stay `b####`.

**Key rule:** a basin is never the answer. It is a candidate stable state **inside a frame**. Lucidity decides whether that becomes a single commit, per-frame commits, an assembly, a rollout plan, or preserved ambiguity (see `lucidity.md` → commit shapes).

---

## Frames, not domains

Basins do not change mechanism per benchmark (text vs ARC vs agent). **Binding and context-op produce frames**; basins attach to those frames.

```
Frame {
    frame_id
    frame_type              // learned tag + optional affinity hints — see below
    traces
    roles
    context_frame_id
    task_intent_hint
}
```

Example `frame_type` tags (illustrative — not a fixed ontology baked into code):

```
word_sense | event | relation | rule | goal | state | transition | action | world
```

The system asks:

```
What frame are we resolving?
→ which basins compete or cooperate inside that frame only
```

| Domain | Typical frames | Basin behavior |
|--------|----------------|----------------|
| Text | `word_sense`, `event` | Often **compete** within one frame (e.g. financial vs river sense for "bank") |
| Text (multi-clause) | multiple `event` frames | **One hypothesis per frame** — do not merge F1 energy into F2 |
| ARC-AGI 2 | `rule`, transform | Often **cooperate** into assemblies (move + recolor) |
| ARC-AGI 3 / agents | `goal`, `state`, `transition`, `action` | Multiple hypotheses stay alive; projection/rollout via lucidity |
| World model | `world`, `state`, `transition` | Competing world hypotheses; transition links across steps |

Do **not** hard-code "text gets one pass, ARC gets many." Pass count is a **lucidity** choice from margins, coverage, `task_intent`, and whether projection is required — not a domain flag in the basin stage.

---

## Pipeline position

In your architecture:

```
cue encoder
→ DMF / tracebank
→ binding
→ context-op
→ interference
→ basins
→ lucidity check
→ [projector if lucidity requests it]
→ decoder
```

Basins are the stage where ambiguity starts becoming structure.

Many tasks need **basin assemblies** — not a single winner (e.g. move + recolor; found-event + placed-event).

But basins still do not final-commit. Lucidity does that.

---

## Current implementation

The runtime implementation lives in `lucid/cognition/reasoning/basins.py` and is backed by checkpoint records loaded through `lucid/memory/basin_bank.py`.

At runtime basins are:

1. loaded from `basin_bank.json` in the selected checkpoint;
2. indexed by stable basin ID and normalized family hint;
3. shortlisted per context frame from frame affinity, context-op basin pressure, or activation-signature matches;
4. scored with local frame confidence, pressure, trace coherence, activation match, trust, heat tier, interference deltas, continuity bias, and suppression penalties;
5. ranked with `compute_policy.max_active_basins` while preserving scope coverage before filling the remaining budget;
6. assembled only from active basins in the same scope;
7. emitted with audit notes that summarize bank size, active states, assemblies, evidence handles, source refs, suppression links, conflicts, and snapshot ID.

The direct manual smoke command is:

```bash
python -m lucid.cli basins --audit-dir audit/basins
```

That command runs the built-in bank ambiguity fixture, prints `BasinOutput` JSON, and writes a normal Lucid audit run under `audit/basins/runs/<run_id>/`.

---

## Input contract

```
BasinInput {
    interference_output          // energy deltas, cooperation/competition maps
    candidate_frames             // binding candidate frames
    context_frames               // context-op scopes
    local_basin_pressures        // scoped priors from context-op
    basin_field_snapshot_id      // optional versioned basin-memory snapshot
    prior_basin_state            // optional carryover for continuity bias
    compute_policy
}
```

---

## Output contract

```
BasinOutput {
    candidate_basin_states: [
        {
            basin_id                 // b12441
            energy
            assembly_id              // optional — if part of cooperative set
            member_basins            // optional assembly members
            supporting_traces
            supporting_frames
            scope_frames             // which context frames contributed
            margin_vs_next
            coherence_score
            activation_signature
            semantic_signature
            evidence_handles
            relation_handles
            source_refs
            trust_score
            heat_tier
            quantized_payload
        }
    ]
    basin_assemblies: [
        {
            assembly_id
            member_basin_ids
            combined_energy
            assembly_coherence
            scope_frames
            evidence_handles
            relation_handles
            source_refs
            quantized_payload
        }
    ]
    competition_summary: {
        top_basin_id
        second_basin_id
        top_margin
        active_basin_count
    }
    unresolved_conflicts         // passed to lucidity
    binding_stability_hint
    audit_notes
}
```

---

## Basin lifecycle

Mirrors trace lifecycle (continual learning):

```
seed → provisional → active → stabilized → crystallized → frozen
```

| State | Behavior |
|-------|----------|
| provisional | quarantined; weak energy contribution |
| active | full participation in competition |
| stabilized | repeated lucidity + projection success |
| crystallized | high margin; candidate for quantization |
| frozen | cold storage; thaw requires explicit policy |

Basins gain meaning from:

```
which traces support them
which frames scope them
which projections they survive
which decoder outputs they produce under lucidity
```

---

## Internal mechanics

### 1. Energy accumulation

Apply `basin_energy_deltas` from interference **per scope**, aggregate into basin state:

```
E(b) += Σ scoped deltas + assembly cooperation bonuses - competition penalties
```

### 2. Assembly formation

When `cooperation_maps` link basins inside a scope:

```
assembly A = {b_move, b_recolor} with combined_energy
```

Assemblies compete as units in `competition_summary`.

### 3. Margin computation

```
top_margin = E(top) - E(second)
assembly_margin = E(top_assembly) - E(second_assembly)
```

Low margin → lucidity preserves ambiguity or requests projection.

### 4. Plural outputs

Always emit **multiple** candidate_basin_states above energy floor — never single winner only.

Even when one sense basin will eventually win for a simple sentence, **BasinOutput must still list alternatives and margins** — lucidity performs collapse, not basins.

### 5. Competition vs cooperation (learned)

Basins learn how they relate to each other **per frame type**, from outcomes — not from manual domain rules.

| Relation | When it emerges | Effect |
|----------|-----------------|--------|
| **Compete** | Choosing one basin usually excludes another in the same frame | Negative basin–basin interference (e.g. two `word_sense` candidates for "bank") |
| **Cooperate** | Two basins together explain more evidence | Positive links → **assembly** (e.g. movement + recolor in a `rule` frame) |
| **Transition** | One basin/state leads to another after action/event | `transition_links`: action/event trace + `next_basin_id` (agents, world models) |

Training strengthens or weakens these links when lucidity/projector validation succeeds or fails (see training orchestrator).

---

## Basin memory record (editable store)

Persistent basin entries (separate from per-run `BasinOutput`):

```
BasinRecord {
    id                          // b####

    frame_affinities: {         // soft weights — not "this basin IS a text basin"
        frame_type_tag → weight
    }

    support_links: {
        trace_id → weight
        binding_pattern_id → weight
        context_pattern_id → weight
    }

    suppression_links: {
        trace_id → weight
        basin_id → weight           // competition within frame
    }

    cooperation_links: {
        basin_id → weight           // assembly candidates
    }

    transition_links: [
        { trigger_trace_id, next_basin_id, weight }
    ]

    maturity                    // seed → provisional → active → stabilized → crystallized → frozen
    success_rate
    failure_modes[]             // auditable history for editors and governor
}
```

Editors may inspect/adjust links, heat tier, and freeze/thaw — same editability standard as tracebank. Quantization applies to stable links per `memory quantization.md`.

---

## Text example: bank / kayaking sentence

```
BasinOutput {
    candidate_basin_states: [
        {basin: b12441, energy: .78, scope: [F1], supporting_traces: [t_kayak, t_found], margin_vs_next: .06},
        {basin: b00491, energy: .74, scope: [F2], supporting_traces: [t_bank_fin, t_placed], margin_vs_next: .05},
        {basin: b08810, energy: .69, scope: [F2], supporting_traces: [t_bank_river], margin_vs_next: .05}
    ]
    basin_assemblies: [
        {assembly: A1, members: [b12441], combined_energy: .78, scope: [F1]},
        {assembly: A2, members: [b00491], combined_energy: .74, scope: [F2]}
    ]
    competition_summary: {
        top_basin_id: b12441,
        second_basin_id: b00491,
        top_margin: .04,
        active_basin_count: 3
    }
    unresolved_conflicts: [
        {scope: F2, type: destination_ambiguity, basins: [b00491, b08810]}
    ]
}
```

Low global margin; lucidity must not treat b12441 as sentence-wide winner (different scopes).

---

## Grid example (visual stress test)

```
BasinOutput {
    candidate_basin_states: [
        {basin: b5502, energy: .81, assembly_id: ASY1, member_basins: [b5502, b3340], scope: [F_content, F_cross]},
        {basin: b9100, energy: .76, scope: [F_legend]},
        {basin: b2011, energy: .72, scope: [F_content]}
    ]
    basin_assemblies: [
        {assembly: ASY1, members: [b5502, b3340], combined_energy: .84, scope: [F_content]}
    ]
    competition_summary: {
        top_margin: .03,
        active_basin_count: 4
    }
}
```

Move+recolor assembly competes with symbol-mapping basin in separate scopes; lucidity may `REQUEST_PROJECTION`.

---

## Anti-patterns

**Do not use semantic basin names as IDs.**

```
BAD:  basin: financial_bank
GOOD: basin: b00491
```

**Do not collapse to single winner in BasinOutput.**

```
BAD:  winner_basin only
GOOD: candidate_basin_states + competition_summary + margins
```

**Do not final-commit or decode.**

```
BAD:  committed_basin in BasinOutput
GOOD: candidate states for lucidity
```

**Do not merge scopes when aggregating energy.**

F1 outdoor basin energy must not silently dominate F2 finance competition.

**Do not hard-code ARC rule basins.**

Basins emerge from training; assemblies compose at runtime.

**Do not treat basin "modes" as built-in types.**

```
BAD:  global enum SenseBasin | RuleBasin | GoalBasin
GOOD: frame_type on Frame + frame_affinities on BasinRecord
```

**Do not argmax inside the basin stage.**

```
BAD:  winner_basin_id only in output
GOOD: plural candidates + competition_summary → lucidity
```

---

## How basins are learned

```
1. Traces activate; binding forms candidate frames.
2. Context-op scopes them; interference applies support/conflict per scope.
3. Basin candidates accumulate energy (per frame).
4. Lucidity / projector validates outcome.
5. Success → strengthen trace/frame/context → basin links; promote maturity.
6. Failure → weaken, split, quarantine, or suppress via training governor.
```

A basin is never manually assigned as "financial bank." It emerges when `money + deposit + bank + storage context` repeatedly stabilizes the same attractor and passes lucidity.

---

## Editability

Editors must be able to:

```
inspect basin energy trace for a run
adjust basin heat tier
edit support / suppression / cooperation / transition links on BasinRecord
split/merge basins (with audit, post-stabilization)
view assembly history
freeze/thaw basin fields
```

---

## Summary

```
Basins = learned hypothesis attractors (b####) scoped to frames
Input:     interference output + candidate_frames + context_frames + local pressure
Output:    plural candidate states, assemblies, margins, conflicts (per frame)
Learn:     compete / cooperate / transition links from validated outcomes
Principle: plural until lucidity; same machinery for text, ARC, agents, world models
Next:      lucidity commit shape (+ optional projector)
Never:     final commit, decoder output, cross-frame energy merge, semantic basin IDs
```

Basins are where **competition becomes structured** — but still reversible until lucidity speaks.

**Mental model:**

```
Trace      = what is active
Binding    = what structure could exist
Context-op = where evidence applies
Interference = support/conflict inside scope
Basin      = stable hypothesis for that frame
Lucidity   = whether/how to commit
Projector  = test consequences if needed
Decoder    = express approved state
```
