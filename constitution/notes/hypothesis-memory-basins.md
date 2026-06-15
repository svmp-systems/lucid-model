# Hypothesis-Memory Basins

Basins must remain hypothesis buckets. They are not answer templates, static
paragraph stores, or hidden prompt-completion modules. A basin is the cheap,
auditable runtime object that says: "this family of evidence is likely active,
these source-backed handles support it, these related hypotheses may cooperate
or conflict, and this compact payload is enough to rehydrate detail when the
system needs it."

## Why This Exists

The model needs memory that scales past hand-authored cases while staying
editable, quantizable, and safe for continual learning. Transformer-style
pretraining hides knowledge in dense weights; this system should keep knowledge
in explicit stores that can be inspected, patched, demoted, replayed, compressed,
or deleted.

That means basins must do two jobs at once:

1. Act as runtime hypotheses that compete, combine, and preserve uncertainty.
2. Point into large, cheap knowledge memory without forcing all knowledge into
   active context.

## Basin Field vs Stored Basins

The general basin field is the live activation space. It receives traces,
binding frames, local basin pressures, source trust, and interference signals.
It decides which stored basins should wake up, compete, assemble, or stay cold.

Stored basins are compact hypothesis-memory prototypes. They live cheaply in
the checkpoint and are recalled only when their sparse signatures match the
current evidence. They should be heavily quantizable because they store handles
and signatures, not prose.

The split is essential:

- The basin field stays general and learns how to activate new basin families.
- Stored basins can scale to very large knowledge corpora.
- Cold basins do not tax inference until traces and binding make them relevant.
- Multiple basins can be active together, letting answers blend evidence from
  several hypothesis buckets.

## What A Basin Stores

A basin should store only general-purpose fields:

- `basin_id`: stable editable identity.
- `family_hint`: coarse hypothesis family for routing and pressure.
- `activation_signature`: sparse cue and trace weights used to wake the basin.
- `semantic_signature`: compact concept/operator tokens used for matching and
  compression.
- `frame_affinities`: learned compatibility with frame shapes.
- `evidence_handles`: pointers to concepts, claims, traces, examples, or source
  chunks.
- `relation_handles`: pointers to relation records the decoder or reasoner can
  rehydrate.
- `source_refs`: auditable source identifiers.
- `trust_score`: source-backed reliability estimate.
- `heat_tier`: quarantine, probation, warm, hot, cold, or archive.
- `cooperation_links`: other basins that compose well with this basin.
- `suppression_links`: basins that should be downweighted under conflict.
- `quantized_payload`: tiny structured payload, such as top terms, relation
  summaries, confidence buckets, and compression metadata.

It should not store broad natural-language answers as the primary memory. Fluent
text should be produced later by the decoder from render units, source handles,
and rehydrated relations.

## Training Rule

Basins are trained from real, source-backed evidence and from system failures.
Generator outputs are allowed only as canaries, tests, or mutations of real
seeds. They must not be treated as ground truth for durable knowledge unless a
source-backed validator accepts them.

The training loop should prioritize:

- Failures, low margins, contradictions, and high-surprise episodes.
- Exact validators where available.
- Single-module blame before patching.
- Quarantine for all new knowledge.
- Replay by family and heat tier before promotion.
- Patch pruning and consolidation into operators when patterns repeat.
- Quantization only after replay shows the patch remains safe.

High-margin wins should usually be skipped. Repeated episodes should dedupe into
stronger handles, not duplicate memory.

## Activation And Assembly

At inference, the field activates basins through overlapping signals:

- Trace IDs and cue aliases.
- Binding frame roles and supporting evidence.
- Local basin pressure from context.
- Interference boosts and penalties.
- Prior active basin state.
- Trust and heat tier.

Activated basins compete by energy. Compatible basins assemble when cooperation
links or interference maps say their hypotheses are mutually useful. Assemblies
must carry merged evidence handles, source refs, and relation handles so the
decoder can build an answer from several basins.

## Decoder Contract

Basins should feed fluent decoding without becoming templates. The decoder sees
render units, not raw basin internals. A basin can contribute:

- Source-backed relation claims.
- Evidence handles that can be rehydrated.
- Omission markers when a required detail is absent.
- Confidence, heat, and trust metadata for audit.

Simple factual answers can use deterministic renderers. Open-ended answers
should go through a canvas builder and a small realizer later, but the realizer
must stay faithful to approved render units. There should always be a literal
fallback when fluency would risk inventing unsupported content.

## Scale And Quantization

The long-term target is a very large cold basin store with cheap sparse recall.
Each basin should be compressible to low-bit signatures and handles:

- Quantized activation and semantic vectors.
- Small relation summaries with confidence buckets.
- Shared dictionaries for terms, sources, and relation labels.
- Cold payload compression with rehydration on failure.
- Basin compaction when many basins become equivalent.

The system should pay full cost only for active basins. Cold basins should stay
cheap, auditable, and editable.

## Implementation Standard

An implementation counts as aligned only if it:

- Keeps basins as hypotheses rather than templates.
- Stores source refs and handles for audit.
- Allows multiple basins to compose.
- Supports quarantine and later promotion.
- Avoids generator-gold training as durable knowledge.
- Lets the decoder produce fluent text from approved units.
- Keeps the memory representation general across grids, language, FAQ, article
  QA, aptitude questions, and future domains.
