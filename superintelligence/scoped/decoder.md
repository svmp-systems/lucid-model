# Decoder

Scoped architecture spec for the lucid-model cognitive pipeline.

---

## Role

The decoder is the expression layer. It turns lucidity-approved state into an external answer, grid, action, plan, tool call, or structured payload.

It is not a better next-token predictor. It is a renderer:

```text
LucidityRenderPacket + DecoderPolicy -> DecoderOutput
```

The system's advantage is that most of the hard work has already happened before decoding:

```text
perception -> cue -> DMF -> binding -> context-op -> interference -> basins -> lucidity
```

By the time the decoder runs, it should not rediscover meaning. It should express what lucidity approved, at the detail level lucidity allowed.

---

## Why this can beat next-token decoding

Classic next-token decoding makes the surface model do many jobs at once:

```text
retrieve facts
track scope
resolve ambiguity
choose confidence
plan structure
avoid hallucination
sound natural
emit tokens
```

Lucid should split those jobs. The decoder receives:

```text
committed facts
preserved alternatives
source refs
forbidden claims
output mode
detail budget
uncertainty policy
```

That creates the 10x to 100x opportunity:

```text
cost win:    render bounded state instead of sampling open-ended text
quality win: output cannot claim more than lucidity approved
audit win:   every sentence can point back to traces, basins, frames, tools, or validators
```

---

## Pipeline position

Default path:

```text
basins -> lucidity -> decoder
```

Projection path:

```text
basins -> lucidity -> projector -> lucidity -> decoder
```

The decoder never runs before lucidity and never runs on raw basin state.

---

## Input contract

```text
DecoderInput {
    lucidity_output
    render_packet               // required unless decoder_policy.mode == hold
    decoder_policy

    user_facing_context          // optional channel/user wrapper
    output_channel               // chat | cli | api | action_bus | grid
}
```

The decoder may read `lucidity_output` for audit, but it should render from `render_packet` and `decoder_policy`.

If `render_packet` is missing and policy mode is not `hold`, the decoder must refuse with an internal contract error.

---

## LucidityRenderPacket

Lucidity prepares the decoder's working set.

```text
LucidityRenderPacket {
    packet_id
    decision
    render_mode                 // committed | plural | uncertainty | refusal | hold
    output_format               // text | grid | action | plan | tool_call | structured_json

    approved_units: [
        RenderUnit {
            unit_id
            unit_type            // claim | frame_summary | alternative | caveat | artifact | action
            scope_frame_id
            text_intent          // answer | reason | caveat | next_step | refusal
            payload              // structured content, not prose-first
            confidence
            required: bool
            source_refs: [SourceRef]
        }
    ]

    preserved_alternatives: [
        {
            hypothesis_id
            scope_frame_id
            basin_id
            contrast_with
            source_refs: [SourceRef]
        }
    ]

    explicit_omissions: [
        {
            reason               // unsupported | low_margin | high_risk | projection_failed
            forbidden_claim_refs
            user_visible: bool
        }
    ]

    render_constraints: {
        max_sentences
        max_tokens
        detail_level             // terse | normal | expanded | audit
        audience_level           // child | general | expert | machine
        tone                     // neutral | direct | careful | instructional
        must_include_refs
        forbidden_refs
    }

    faithfulness_contract: {
        forbid_new_entities
        forbid_new_causal_links
        require_source_refs_per_sentence
        require_reparse_check
    }

    provenance_chain
}
```

### SourceRef

```text
SourceRef {
    ref_type        // trace | basin | frame | evidence | conflict | projection | tool | validator
    ref_id
    scope_frame_id
    role            // supports | contradicts | bounds | formats | verifies
}
```

If a rendered sentence cannot point to source refs, it should not be emitted except for harmless connective text.

---

## Output contract

```text
DecoderOutput {
    surface_text
    surface_grid
    surface_action
    structured_payload

    render_mode
    cited_refs: [SourceRef]
    sentence_refs: [
        {sentence_id, source_refs}
    ]

    uncertainty_presentation
    refused
    refusal_reason

    faithfulness_report: {
        unsupported_sentence_count
        omitted_required_units
        policy_violations
        reparse_match_score
    }

    audit_notes
}
```

The output is both user-facing payload and auditable render artifact.

---

## Render modes

### committed

Use when lucidity decided `COMMIT`.

The decoder renders only approved committed units. It may compress wording, but it may not add a new claim.

```text
Input:  claim, scope F2, payload bank_sense = financial_storage
Output: "Here, bank means a financial/storage bank."
```

### plural

Use when lucidity decided `PRESERVE_AMBIGUITY`.

The decoder must not pick one winner. It shows alternatives and why they remain live.

### uncertainty

Use when a concise caveat is better than listing all alternatives.

### refusal

Use when risk or missing evidence prevents a safe answer.

### hold

No external answer. Projection, search, or re-binding is still pending.

---

## Renderer architecture

The decoder should be a set of small renderers, not one giant generator.

```text
DecoderRouter:
    inspect decoder_policy.output_format
    inspect render_packet.render_mode
    choose renderer

Renderers:
    TextClaimRenderer
    PluralHypothesisRenderer
    GridArtifactRenderer
    ActionRenderer
    PlanRenderer
    ToolCallRenderer
    RefusalRenderer
    StructuredJsonRenderer
```

Each renderer is testable because it consumes a bounded packet.

---

## Text rendering

Text output should be built in layers:

```text
1. select approved_units required by policy
2. order by scope, answer priority, and dependency
3. choose sentence patterns by unit_type
4. attach sentence_refs
5. run faithfulness check
6. optionally run tiny polish pass constrained by sentence_refs
```

The optional polish pass is not allowed to add claims. It may only improve grammar, flow, or brevity.

---

## Grid rendering

For grids, the decoder should not describe when a grid artifact is requested.

```text
projection_artifact.grid_output -> surface_grid
```

Text explanation is optional and policy-controlled.

---

## Action rendering

For actions, decoder output must be structured first:

```text
surface_action {
    action_type
    target_ref
    arguments
    safety_bounds
    source_refs
}
```

Natural language can accompany the action, but the action struct is authoritative.

---

## Cost controls

Use this ladder:

```text
Level 0: direct deterministic render
Level 1: template with slot filling
Level 2: grammar/format adapter
Level 3: tiny constrained language pass
Level 4: larger language fallback, only when policy allows
```

Most routine answers should stay at Level 0 or Level 1.

Escalation requires:

```text
policy.allow_language_fallback = true
render_packet.faithfulness_contract still enforced
post-render reparse check passes
cost budget available
```

---

## Faithfulness check

After rendering:

```text
for each output sentence:
    extract claims
    map claims to approved_units
    fail if claim lacks source refs
    fail if claim contradicts explicit_omissions
    fail if policy forbids single answer and sentence collapses alternatives
```

If faithfulness fails:

```text
try simpler deterministic render
else express uncertainty/refusal
log decoder_faithfulness failure
```

---

## Training signal

Decoder training data is not:

```text
input text -> answer
```

It is:

```text
LucidityRenderPacket + DecoderPolicy -> faithful surface output
```

When committed state is correct but output is bad:

```text
update decoder renderer/correction pairs only
do not update traces, basins, interference, or lucidity
```

Correction record:

```text
DecoderCorrectionPair {
    render_packet
    decoder_policy
    bad_output
    corrected_output
    faithfulness_error
    update_scope: decoder_only
}
```

---

## Implementation impact

To implement this decoder design, the following repo modules should change.

### IR contracts

```text
lucid/ir/lucidity.py
```

Add typed structures for:

```text
SourceRef
RenderUnit
ExplicitOmission
RenderConstraints
FaithfulnessContract
LucidityRenderPacket
```

Then add `render_packet` to `LucidityOutput`, and extend `DecoderPolicy` with:

```text
require_source_refs_per_sentence
max_sentences
max_tokens
allow_language_fallback
fallback_budget
```

```text
lucid/ir/expression.py
```

Update `DecoderInput` so it receives `render_packet` directly. Update `DecoderOutput` with:

```text
structured_payload
render_mode
cited_refs
sentence_refs
faithfulness_report
audit_notes
```

### Lucidity implementation

```text
lucid/cognition/.../lucidity
lucid/cognition/orchestrator/stub_stages.py
```

Wherever lucidity is implemented, it should build the render packet before the decoder runs. The stub lucidity stage should also emit a minimal packet so end-to-end audits prove the contract.

### Decoder implementation

```text
lucid/cognition/.../decoder
```

Add a real decoder module with a router and small renderers:

```text
TextClaimRenderer
PluralHypothesisRenderer
GridArtifactRenderer
ActionRenderer
PlanRenderer
RefusalRenderer
StructuredJsonRenderer
```

The current decoder path should reject missing packets unless policy mode is `hold`.

### Orchestrator handoff

```text
lucid/cognition/orchestrator/runner.py
lucid/training/orchestrator/orchestrator.py
```

Pass `lucidity_output.render_packet` into `DecoderInput`. Audit should record the packet and decoder faithfulness report.

### Audit and tests

```text
lucid/audit/logger.py
tests/test_ir_roundtrip.py
tests/test_orchestrator_runner.py
```

Audit summaries should include render mode, cited refs, and faithfulness failures. Roundtrip tests should cover the new IR. Runner tests should prove decoder cannot output unsupported claims.

### Training data

```text
lucid/ir/training.py
superintelligence/scoped/training/training data.md
```

Add decoder correction pairs in the concrete IR if they are not already represented:

```text
render_packet + decoder_policy + bad_output -> corrected_output
```

These updates should stay decoder-only when committed state is correct.

---

## Examples

### Bank sentence

```text
LucidityRenderPacket {
    render_mode: committed
    output_format: text
    approved_units: [
        {
            unit_type: claim
            scope_frame_id: F2
            payload: {bank_sense: financial_storage}
            source_refs: [t_money, t_placed, t_bank, b_financial_destination_like]
        },
        {
            unit_type: caveat
            scope_frame_id: F1
            payload: {kayaking_scope: separate_event}
            source_refs: [cf_event_one, gate_cf_event_two]
        }
    ]
}
```

Possible output:

```text
In the placed-money frame, bank means a financial/storage bank. The kayaking context belongs to the separate earlier frame.
```

### Ambiguous reading

```text
render_mode: plural
preserved_alternatives: [
    {basin_id: b_financial_destination_like, scope: F2},
    {basin_id: b_river_bank_like, scope: F2}
]
```

Possible output:

```text
The sentence does not force one bank sense. The money/placed frame supports a financial bank reading, while another bank reading remains unresolved.
```

### Grid

```text
render_mode: committed
output_format: grid
approved_units: [
    {unit_type: artifact, payload: {grid_output: [[1,0],[0,1]]}}
]
```

Output:

```text
surface_grid = [[1,0],[0,1]]
```

---

## Anti-patterns

**Do not make the decoder reason from raw text.**

```text
BAD:  raw user prompt -> decoder -> answer
GOOD: lucidity render packet -> decoder -> answer
```

**Do not let fluency override policy.**

```text
BAD:  low margin, but decoder sounds confident
GOOD: policy mode plural or uncertainty
```

**Do not invent citations after the fact.**

```text
BAD:  generate paragraph, then attach random trace refs
GOOD: source_refs drive sentence creation
```

**Do not use large generation by default.**

```text
BAD:  every answer goes through an LLM-style generator
GOOD: deterministic render first; constrained fallback only if needed
```

---

## Summary

```text
Decoder = faithful renderer of lucidity-approved state
Input:    LucidityRenderPacket + DecoderPolicy
Output:   surface payload + citations + faithfulness report
Never:    decide truth, add unsupported facts, override lucidity
Win:      lower cost and higher quality by rendering checked structure instead of predicting tokens
```

The decoder should be boring in the best way: small, auditable, cheap, and faithful.
