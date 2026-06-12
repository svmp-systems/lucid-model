# Cost-Quantized Continual Training

This note defines the long-term training strategy for Lucid if the goal is not
merely to reduce cost, but to escape frontier-transformer cost structure.

The target is aggressive:

```text
1000x lower total training and adaptation cost than frontier-style dense
pretraining, while keeping the system inspectable, editable, and continually
learnable.
```

The path is not:

```text
more generator data -> bigger model -> better model
```

The path is:

```text
real experience + validators + local patches + quarantine + replay
  -> consolidation
  -> heat-tiered memory
  -> quantized stable knowledge
  -> sparse active compute
```

The generator remains important, but it is not the scalable teacher. It is a
curriculum compiler, test mutator, canary builder, and failure amplifier.

## External Motivation

Public cost estimates for frontier model training already sit in the tens to
hundreds of millions of dollars for leading systems. Stanford HAI's AI Index
coverage cites estimates around 78 million USD for GPT-4 and 191 million USD
for Gemini Ultra. Epoch tracks frontier training cost growth and algorithmic
efficiency trends, both of which imply that a small lab cannot win by copying
dense scaling economics.

Synthetic data also cannot be the entire answer. Work on model collapse shows
that recursively training on generated data can degrade the modeled
distribution when synthetic data displaces grounded data.

Continual-learning work gives useful warnings and tools. Experience replay,
elastic weight consolidation, and generative replay all address catastrophic
forgetting in dense neural systems. Lucid can take the core lesson, but avoid
the hardest version of the problem by making much of its knowledge explicit,
localized, auditable, and quarantineable.

References:

- Stanford HAI AI Index cost discussion: https://hai.stanford.edu/news/inside-new-ai-index-expensive-new-models-targeted-investments-and-more
- Epoch AI trends: https://epoch.ai/trends
- Model collapse: https://www.nature.com/articles/s41586-024-07566-y
- Elastic weight consolidation: https://www.pnas.org/doi/10.1073/pnas.1611835114
- Experience replay for continual learning: https://arxiv.org/abs/1811.11682
- Deep generative replay: https://papers.nips.cc/paper/6892-continual-learning-with-deep-generative-replay

## Core Claims

1. Generator-only training is not scalable.
2. Dense next-token pretraining is not the main training mode.
3. Long-term training must be continual, failure-driven, and local.
4. New knowledge must enter quarantine before it can affect committed answers.
5. Stable knowledge must become cheaper over time through heat tiers and
   quantization.
6. Lucidity remains the only commit gate.
7. Every learned object must carry provenance, support, contradiction, replay,
   heat, and quantization metadata.

## What The Generator Is For

The generator should not be treated as the world.

It should be treated as a microscope.

Good generator uses:

- Create exact validator-backed episodes.
- Create contrast pairs around a discovered failure.
- Mutate real failures into nearby boundary cases.
- Generate canaries for regression testing.
- Generate adversarial variants of a promoted patch.
- Build curriculum ramps for new operators.
- Stress test lucidity decisions, especially hold, plural, and refusal.
- Produce small, clean, inspectable datasets for module calibration.

Bad generator uses:

- Replacing real experience.
- Expanding synthetic corpora without validators.
- Training the decoder as a general language model.
- Letting generated text become the source of facts.
- Treating template pass rate as real intelligence.
- Training on synthetic outputs recursively without grounding.

Long-term rule:

```text
Real or validated events supply the seed.
The generator supplies the neighborhood around the seed.
```

## Long-Term Data Sources

Training should draw from multiple streams, each with different trust.

| Source | Trust | Primary Use |
| --- | --- | --- |
| User corrections | Medium to high | Local patches, decoder corrections, concept edits |
| Benchmark failures | High if validator is exact | Binding, projector, lucidity, operator repair |
| Tool-verified facts | High | Concept banks, source-backed traces |
| Real interaction logs | Medium | Cue routing, ambiguity handling, session memory |
| Public corpora parsed by Lucid | Medium | Concept discovery, language coverage |
| Projector rollouts | Medium | Consequence testing, operator stats |
| Generator mutations | Low to medium | Tests, canaries, contrastive boundaries |
| LLM proposal adapters | Low | Candidate perception or paraphrase proposals only |

The trust level controls the initial heat tier and commit permission.

## Heat Tiers

Every learned object should have a heat tier. This applies to traces,
operators, aliases, basin links, context gates, decoder corrections, lucidity
threshold patches, and projection rules.

| Tier | Meaning | Commit Permission |
| --- | --- | --- |
| `quarantine` | New, untrusted, locally plausible | Cannot support commit alone |
| `probation` | Passed target fix and small shadow bundle | Can influence plural hypotheses |
| `warm` | Repeatedly useful, no recent regression | Can support commit with other evidence |
| `hot` | Frequently useful and high margin | Normal active support |
| `cold` | Stable, mature, rarely edited | Quantized retrieval support |
| `archived` | Not active; provenance and replay only | Rehydrate only if needed |

Tier movement is evidence-driven. No object jumps directly from quarantine to
hot because it fixed one episode.

## Learned Object Metadata

Every learned object should carry this shape, or an equivalent local schema:

```json
{
  "object_id": "operator:motion_propagates_through_coupling",
  "object_type": "operator",
  "source": "benchmark_failure",
  "created_at": "2026-06-12T00:00:00Z",
  "heat_tier": "quarantine",
  "precision_tier": "fp16",
  "commit_permission": "support_only",
  "support_count": 1,
  "contradiction_count": 0,
  "target_fix_count": 1,
  "shadow_pass_count": 0,
  "canary_pass_rate": 0.0,
  "last_failed_replay": "",
  "source_refs": [],
  "audit_refs": [],
  "quantization_candidate": false
}
```

This makes continual learning inspectable. A bad memory is not a mysterious
weight update. It is an object with a source, history, permission level, and
rollback path.

## Continual Learning Loop

The normal loop should be:

```text
observe
  -> run full pipeline
  -> validate outcome
  -> if high-margin success: no update
  -> if low-margin success: observe more
  -> if failure: assign blame
  -> propose smallest local patch
  -> quarantine patch
  -> shadow test
  -> promote to probation if safe
  -> replay over time
  -> consolidate repeated patches
  -> quantize stable knowledge
```

This is more important than batch training. Batch training can still exist, but
it should be composed of the same local operations.

## Cost Quantization Strategy

The 1000x target needs several compounding savings.

| Lever | Intended Saving |
| --- | --- |
| Replace internet-scale next-token pretraining with structured episodes | 30x to 100x |
| Train only on surprise, failure, uncertainty, and correction | 5x to 20x |
| Apply local patches instead of global updates | 10x to 100x per correction |
| Use sparse active traces, basins, operators, and gates | 10x to 50x runtime |
| Reuse graph operators instead of memorizing examples | 10x to 100x in covered domains |
| Quantize stable hot/warm/cold memory | 4x to 32x memory bandwidth |
| Cache deterministic stage outputs for replay | 2x to 10x training iteration cost |

These multipliers are targets, not assumptions. The scaling observatory must
measure them.

## Training Modes

Lucid should use four training modes.

### 1. Calibration

Single module under test, gold or stubbed neighbors.

Use for:

- New module development.
- Generator-backed exact episodes.
- Operator primitive validation.
- Decoder faithfulness checks.

### 2. Stack Training

Real upstream, module under test, gold or stub downstream.

Use for:

- Perception to cue calibration.
- Cue to DMF activation quality.
- Binding from real perception and DMF.
- Lucidity from real basins and context.

### 3. Full Loop Training

End-to-end run, validator, blame, patch, shadow, promote.

Use for:

- Real failures.
- Benchmark episodes.
- Continual learning.
- Cross-module regressions.

### 4. Sleep Consolidation

Offline compaction of repeated local patches into cleaner memory.

Use for:

- Merging aliases.
- Strengthening or pruning traces.
- Creating graph operators from repeated successful patches.
- Moving stable memory to colder precision.
- Building canary suites from past failures.

## Module Training Map

| Module | Needs Training? | Normal Training | Special Training Needed |
| --- | --- | --- | --- |
| Perception | Yes | Parse raw input into evidence graph | Must train for abstention, uncertainty flags, span provenance, and proposal rejection |
| Cue encoder | Yes | Route evidence to trace activations | Must optimize recall under budget, not just precision |
| DMF | Yes | Activate traces, track conflict and novelty | Must train heat-aware retrieval, sparse activation, and quantized trace tiers |
| Binding | Yes, heavily | Build plural candidate frames | Needs graph/operator training, local frame seeds, role ambiguity, and provenance |
| Context-op | Yes | Scope frames, link traces, gate interference | Needs anti-leak training and cross-frame reference resolution |
| Interference | Yes | Support/conflict edges | Needs contradiction mining, false-conflict suppression, and family-local gates |
| Basins | Yes | Candidate state settling and competition | Needs energy calibration, basin assembly, collapse delay, and quantized attractors |
| Projector/reasoner | Yes, specially | Test consequences and grid/state transforms | Needs exact validators, operator reuse, rollout pruning, and proof receipts |
| Lucidity | Yes, specially | Commit, hold, search, recheck, preserve ambiguity | Needs risk calibration, cost-aware decisions, and adversarial false-commit training |
| Decoder | Yes, narrowly | Approved meaning to text | Must not learn facts; train canvas-to-surface and faithfulness repair only |
| Governor | Yes | Decide update, defer, no-update, consolidation | Needs cost/quality reward and regression sensitivity |
| Generator | Yes, as tool | Produce tests and mutations | Must learn from real failure clusters, not invent ground truth |
| Checkpoint/memory manager | Yes | Promotion, rollback, tier movement | Needs heat-tier policy, quarantine rules, and quantization lifecycle |

## Modules That Need Special Treatment

### Binding

Binding is the biggest unlock for cost reduction.

Current binding can build local candidate frames from perception and DMF traces.
The long-term version must add:

- Universal graph substrate.
- Operator bank.
- Operator matcher and applier.
- Relation alias bank.
- Concept bank.
- Local graph candidates attached to `CandidateFrame`.
- Evidence and provenance for every inferred node and edge.

Binding should not learn by memorizing sentences. It should learn reusable
local structures.

Example:

```text
inside(X, Y) + moves(Y, L) -> at(X, L)
```

One operator like this replaces many examples.

### Projector / Reasoner

The projector is not only for grids. It should become the consequence tester
for operator candidates.

It should answer:

```text
If this candidate graph is true, what follows, and does that match the validator?
```

Special requirements:

- Rollouts must be capped.
- Operators must carry receipts.
- Failed projections must create training data.
- Successful projections can strengthen operator stats.
- High rollout cost should feed back into lucidity and the governor.

### Lucidity

Lucidity is where many systems will fail.

It must be trained on negative cases:

- Tempting but unsupported answer.
- Correct answer from wrong evidence.
- High confidence but wrong binding.
- Good projection but bad source grounding.
- Ambiguous input with one attractive basin.
- User asks for unsupported details.
- Decoder can make answer sound better than evidence supports.

Lucidity should optimize for calibrated commitment, not maximum answer rate.

### Decoder

Decoder training must be narrow.

Allowed:

- Canvas to fluent surface.
- Grammar and flow.
- Compression.
- Sentence ordering.
- Faithfulness repair.

Forbidden:

- Learning facts.
- Re-answering the prompt.
- Adding examples.
- Adding analogies unless approved.
- Using raw prompt as a completion request.

Decoder should be cheap because it is not the knowledge system.

### Governor

The governor is a cost-control learner.

It should learn when to:

- Update nothing.
- Defer.
- Patch one module.
- Allow two modules when blame is uncertain.
- Expand replay.
- Promote.
- Reject.
- Consolidate.
- Quantize.

This is where cost reduction becomes policy, not hope.

## Non-Happy Path Cases

The training plan must explicitly handle failure modes.

### False Positive Commit

Problem:

```text
Lucidity commits to an answer that is unsupported or wrong.
```

Response:

- Immediate high-priority failure.
- Patch lucidity thresholds or check logic first.
- Inspect binding and basin support.
- Add canary episode.
- Demote any object that supported the false commit.
- Quarantine related new objects until replay passes.

### False Hold

Problem:

```text
Lucidity refuses or preserves ambiguity when answer is actually supported.
```

Response:

- Lower cost than false commit, but still important.
- Check coverage, margin, and binding stability.
- Add positive support examples.
- Train lucidity to distinguish uncertainty from missing calibration.

### Bad User Correction

Problem:

```text
User correction is wrong, malicious, ambiguous, or context-specific.
```

Response:

- Enter quarantine only.
- Never directly overwrite hot memory.
- Require source support, repeated agreement, or canary pass.
- Store correction as scoped preference if global truth is unclear.

### Poisoned or Low-Trust Source

Problem:

```text
External data introduces false facts or adversarial patterns.
```

Response:

- Low initial trust.
- Source-level reputation.
- Contradiction checks against existing memory.
- No commit permission until verified.
- Batch quarantine for suspicious source clusters.

### Generator Overfit

Problem:

```text
Model passes generator templates but fails real variants.
```

Response:

- Track real-vs-generator pass gap.
- Penalize patches that only improve generated cases.
- Force generator mutations from real failures.
- Keep generator data below a configured share of promotion evidence.

### Cross-Module Blame Error

Problem:

```text
Failure is blamed on decoder, but true cause is binding or lucidity.
```

Response:

- Allow temporary multi-hypothesis blame.
- Prefer patches with smallest blast radius.
- Shadow test alternate patches when blame confidence is low.
- Record blame accuracy over time.

### Catastrophic Forgetting

Problem:

```text
New patch improves target but regresses older skills.
```

Response:

- Canary replay.
- Family-balanced retention suite.
- Heat-tier demotion.
- Patch rejection.
- Archive and replay failure for sleep consolidation.

### Memory Bloat

Problem:

```text
Continual learning accumulates too many small objects.
```

Response:

- Merge repeated aliases.
- Consolidate local patches into operators.
- Prune unsupported objects.
- Move stable low-use objects to cold quantized tiers.
- Archive stale quarantine objects.

### Scalability Cliff

Problem:

```text
Continual learning turns small local updates into global work.
```

The foreseeable cliff is not one bad model update. It is write and retrieval
amplification:

- Trace retrieval drifts back toward scanning the whole tracebank.
- Coactivation or conflict links grow toward a dense graph.
- Checkpoints rewrite and hash large stores after tiny changes.
- Shadow tests replay too many unrelated episodes per proposed patch.
- Audit artifacts grow faster than the useful memory.

Response:

- Keep the hot tier small, mutable, and fully in RAM.
- Write learning events to an append-only journal, then compact into shards.
- Cap coactivation and conflict degree per object; decay and prune old links.
- Retrieve through cue indexes, family shards, and bounded candidate sets.
- Give quarantine objects TTL, merge, and duplicate-detection rules.
- Select shadow bundles by affected family or shard plus global canaries.
- Make cold and archived tiers mostly immutable, quantized, and rehydratable.

The rule is:

```text
continual learning may add experience every turn,
but it must not touch the whole memory every turn
```

### Cost Regression

Problem:

```text
Quality improves but active traces, rollouts, or wall time increase too much.
```

Response:

- Promotion fails unless cost delta is within budget.
- Governor learns cost-sensitive update policy.
- Lucidity may prefer search narrowing over wider retrieval.
- Projector rollouts must be capped and audited.

### Validator Is Wrong

Problem:

```text
Training signal is incorrect.
```

Response:

- Validators have trust levels too.
- Disagreement between validators creates quarantine.
- Human review path for high-value contradictions.
- Do not promote patches from a single low-trust validator.

### Contradictory Memories

Problem:

```text
Two mature memories conflict.
```

Response:

- Preserve plural if context cannot resolve.
- Add scope qualifiers.
- Prefer source-backed and recent evidence when task demands it.
- Do not globally delete unless contradiction is proven.

## Promotion Rules

A patch can move from quarantine to probation only if:

- It improves the target failure.
- It passes a local shadow bundle.
- It does not exceed cost regression budget.
- It has provenance.
- It has a rollback path.

A patch can move from probation to warm only if:

- It succeeds across repeated replays.
- It does not regress family canaries.
- It survives at least one sleep consolidation pass.
- It has no unresolved contradiction.

A patch can move to hot only if:

- It is frequently used.
- It supports high-margin successes.
- It remains stable across domain variants.
- It has known scope boundaries.

A patch can move to cold only if:

- It is mature.
- It is rarely edited.
- It can be represented cheaply.
- Quantization does not harm replay.

## Quantization Lifecycle

Quantization is not only numeric precision. It is a lifecycle.

| Stage | Action |
| --- | --- |
| New | Store full precision and full provenance |
| Quarantine | Do not quantize |
| Probation | Measure stability and access frequency |
| Warm | Candidate for partial quantization |
| Hot | Keep fast; quantize only if replay-safe |
| Cold | Aggressively quantize, compress, index |
| Archived | Store provenance, receipts, and rehydration path |

Quantization must be reversible or replayable. If a quantized object causes a
failure, Lucid should be able to rehydrate it or fall back to a safer version.

## Cost Metrics To Track

The scaling observatory should track:

- Cost per successful validated episode.
- Cost per promoted patch.
- Cost per retained capability.
- Active trace count.
- Active basin count.
- Active operator count.
- Projector rollout count.
- Retrieval budget.
- Wall time per stage.
- Memory tier distribution.
- Quantized object ratio.
- Replay failure rate by tier.
- Generator-vs-real pass gap.
- No-update rate.
- Patch promotion rate.
- Cost delta on promoted patches.
- Checkpoint write amplification.
- Journal compaction cost.
- Audit bytes per promoted patch.
- Retrieval candidate count per DMF run.
- Mean and max link degree by memory family.
- Quarantine TTL expiry and merge rate.
- False commit rate.
- False hold rate.

The key metric is not raw pass rate. It is:

```text
validated quality gain per unit cost, with no unacceptable regression
```

## Training The Unhappy Paths

Every module should see adversarial or negative examples, not just clean gold.

| Module | Negative Training Cases |
| --- | --- |
| Perception | Missing spans, ambiguous spans, noisy text, malformed grids, adversarial punctuation |
| Cue encoder | Distractor cues, rare aliases, high-recall pressure, budget-limited retrieval |
| DMF | Conflicting traces, stale traces, cold trace retrieval, novelty with low support |
| Binding | Unresolved roles, local/global leakage, metaphor, pronoun ambiguity, wrong operator family |
| Context-op | Cross-frame leakage, false reference links, scope confusion, session carryover errors |
| Interference | False contradictions, missed contradictions, irrelevant conflicts |
| Basins | Premature collapse, basin crowding, low-margin winners, assembly mistakes |
| Projector | Spurious transform fit, rollout explosion, invalid operator consequence |
| Lucidity | Tempting false commit, excessive refusal, unsafe projection trust, unsupported decoder packet |
| Decoder | Added facts, dropped required units, relation distortion, over-polishing |
| Governor | Over-updating, under-updating, bad blame, cost-insensitive promotion |
| Memory manager | Bad promotion, stale quarantine, unsafe quantization, failed rollback |

## Implementation Direction

Do not rewrite the whole system at once.

Recommended sequence:

1. Add heat-tier metadata to learned checkpoint objects.
2. Add quarantine and commit-permission fields.
3. Add cost regression checks to promotion.
4. Add generator mutation from real failure seeds.
5. Add graph/operator IR for binding.
6. Add a tiny operator matcher/applier.
7. Add operator/concept/relation checkpoint stores.
8. Attach local graph candidates to `CandidateFrame`.
9. Let projector/reasoner reuse operators.
10. Add replay canaries by family and heat tier.
11. Add sleep consolidation.
12. Add quantization lifecycle and rehydration.
13. Add append-only learning journals and checkpoint compaction.
14. Shard checkpoint stores by object family and heat tier.
15. Replace tracebank scan paths with bounded indexed retrieval.
16. Add coactivation/conflict degree caps, decay, and pruning.
17. Add quarantine TTL, duplicate detection, and merge rules.
18. Replace decoder surface path with canvas -> realizer -> faithfulness.

## Minimal First Target

The first proof should be small:

```text
Real or generated seed:
  "A is inside B. B moves to C. Where is A?"

Binding:
  local graph with inside(A, B), moves(B, C)

Operator:
  inside + move -> propagated location

Projector:
  derives at(A, C)

Lucidity:
  commits only if support is sufficient

Decoder:
  says "A is at C."

Training:
  if wrong, create local patch, quarantine, shadow, promote

Quantization:
  once stable, move operator stats toward cold compact storage
```

This proves the whole cost thesis in miniature:

- One operator replaces many examples.
- Failure creates local patch.
- Patch starts quarantined.
- Replay protects old behavior.
- Stable knowledge gets cheaper.

## Summary

The generator is not the long-term source of intelligence.

Lucid's long-term source of intelligence is continual interaction with real and
validated environments, plus a training system that turns failure into small,
auditable, quarantineable patches.

The generator scales only when it is used to expand, mutate, and validate
failures discovered elsewhere.

The 1000x cost target depends on Lucid becoming better without repeatedly
retraining a dense global model. Knowledge must enter locally, prove itself,
survive replay, consolidate into reusable structure, and become cheaper over
time.

That is the central advantage of this architecture:

```text
experience increases capability,
but stable experience decreases marginal cost.
```
