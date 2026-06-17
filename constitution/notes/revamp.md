# Lucid Revamp — Strategy, Architecture, and MVP

This note consolidates the architecture review from the v.0.3-ai-ml ingest work,
Q&A quality investigation, and product-direction discussion. It is the operating
document for what Lucid is, what it is not, and what to build next.

## What Lucid Is

Lucid is **not** a frontier LM trained end-to-end on the internet. It is a
**cognitive stack** over **structured, continually learnable memory**:

```text
perception → cue encoding → DMF → binding → basins → lucidity → decoder
```

| Layer | Job |
|-------|-----|
| Perception | Evidence only (text, grid, …) — no reasoning |
| Cue + DMF | Activate traces from cues — association, not logic |
| Binding | Interpret input: situational graphs, question type, concept edges |
| Basins | Energy-weighted memory activation after traces bind |
| Lucidity | Commit / refuse / preserve ambiguity — picks licensed answer shape |
| Decoder | Surface form only — realizes committed units |

Checkpoints are JSON stores (`concept_bank`, `tracebank`, `basin_bank`,
`operator_bank`, …), not neural weight matrices. Training is trace promotion,
ingest merge, cue affinity updates — not backprop through a monolith.

**Product thesis:** behave like an LLM at the interface (Q&A, agents, follow-ups)
while staying **cheaper, auditable, editable, and continually learnable**. Logic
lives in the system; the mouth LM (if any) does not think.

## What Lucid Is Not

- Not GPT with extra JSON.
- Not “100 Wikipedia articles = trained model.” Ingest builds **memory**, not
  parametric intelligence.
- Not RAG — unless lucidity commits thin fragments and the decoder LM fills gaps
  from its own weights (the failure mode to avoid).

The v.0.3-ai-ml re-ingest (97 articles, ~3.5h) succeeded technically but
exposed the real bottleneck: **memory quality**, not pipeline stage count.

| Query | Result | Cause |
|-------|--------|-------|
| "how does a transformer work" | Good | `transformer_architecture` has strong `uses` edge |
| "what is a transformer" | Weak fragment | No real definition in memory; scorer picked only passing `type_of` |
| "what is machine learning" | Junk fragment | Wikipedia lead not captured as clean `type_of` |
| "what is AI" | Honest refuse | `artificial_intelligence` has **zero** `type_of` relations |

Regex ingest at scale adds **capacity**, not **definitions**.

---

## MVP Pipeline (What to Run Daily)

Keep the full stack in the repo; **bypass** expensive middle layers on the simple
path first.

### MVP spine (single-turn Q&A)

```text
cue → DMF → binding → basins → lucidity → decoder
```

### Defer on MVP path (do not delete)

- **context-op** — scope linking, session frame wiring, interference gates
- **interference** — cross-frame contradiction pressure

Turn these back on for: multi-frame ambiguity ("bank" river vs money),
multi-scope agents, grid tasks.

### Do not cut

- **Binding** — concept local graphs, event frames, interrogative seeds
- **Lucidity** — commit/refuse/margin; this is the decision layer, not bloat
- **Multi-frame in binding** — hypotheses live here; lucidity collapses them

Cutting “the hypothesis system” entirely returns you to KB lookup + templates.

---

## Binding Revamp — Logic World Model

Binding should grow into a **situation + question interpreter**, implemented as
**three internal passes** (one pipeline stage, auditable as `binding`):

```text
A. Frame parse     — speech/clauses → CandidateFrame + LocalGraph
B. World inference — operators infer edges (inferred: true)
C. Interrogate     — question → answer_schema + evidence relations
```

### A. Frame parse (events)

Example: *"Human drove the car"*

- Build `event: drive(agent=human, patient=car)` with entity/event nodes
- Start with ~20–30 high-frequency verbs (drive, give, put, go, make, …)

### B. World inference (operators)

Existing IR: `GraphNode`, `GraphEdge`, `OperatorReceipt`, `_apply_operators()`.

Bootstrap operators (extend `operator_bank`):

| Family | Pattern | Effect |
|--------|---------|--------|
| `agent_action` | X verbed Y | event(verb, agent, patient) |
| `vehicle_motion` | drive(X, Y) | move(X), move(Y) |
| `coupling` | coupled motion | location/state propagation |

World logic stays in **operators + graphs**, not DMF, not ingest prose, not LM.

### C. Interrogate (question typing)

Extend beyond `definition_query` / `mechanism_query` to an
**InterrogativeProfile** on `CandidateFrame`:

- `answer_schema`: definition | mechanism | cause | verify | agent_fill | …
- `target_focus`: concept or event id in graph
- `evidence_relations`: which relation families to commit from

**Binding pass C** states what answer is required; **lucidity** enforces it
against evidence. Do not discover question intent only at decode time.

---

## Memory and Ingest (Fix Before More Scale)

### What ingest actually does

- Sequential Wikipedia fetch + regex `classify_relation` per sentence
- O(sentences × candidate_terms) — ~3h for 100 articles is expected, not a hang
- Checkpoint saves **once** at end; use `--progress-interval 5` on future runs

### Why lead definitions fail

- Transformer article: no clean "X is a …" lead in extracted sentences
- ML article: no "Machine learning is …"; wrong subjects on nearby sentences
- AI article: lead is *"It is a field of research…"* — `best_subject` picks
  `intelligence` (late in sentence), not `artificial_intelligence`

The classifier works on **canonical** sentences; Wikipedia structure + subject
detection loses them.

### Memory priorities (not more blind scale)

1. **Bootstrap definitional `type_of`** for core concepts (transformer, ML, AI)
2. **Article-lead / pronoun resolution** for ingest (map "It" → article subject)
3. **Merge `transformer` → `transformer_architecture`** — stop splitting concepts
4. **Penalize comparative/usage `type_of`** in commit scoring ("more parallelizable",
   "becoming a field of study")
5. **Domain packs** over raw 100-article regex marathon when quality matters

---

## Decoder and LM — Not RAG If Done Right

### Same surface, different contract

| | RAG + LM | Lucid + constrained decoder |
|--|----------|-------------------------------|
| Who picks facts | LM + retriever | Lucidity + binding + memory |
| LM receives | Raw chunks | Approved `RenderUnit`s only |
| Hallucination | Soft grounding | Hard: `forbid_invented_facts` |
| Learning | Re-embed / fine-tune | Checkpoint merge, trace promotion |

### The circle trap

If committed units are **fragments**, the LM pads from its weights → you pay for
Lucid **and** duplicate world knowledge in the LM → **complicated RAG**.

**No circle** when:

- Lucidity commits **render-ready full claims**
- Decoder realizes them (grammar, discourse glue) — **no new entities or claims**
- Lucidity **refuses** when too thin

| Path | Outcome |
|------|---------|
| Thin memory + big LM decoder | RAG circle — LM does real work |
| Rich memory + small LM / template | Thesis holds — LM is mouth only |
| Rich memory + constrained LM | LLM-like **in domains you own** |

Expert **tone** comes from **what you commit**, not LM PhD knowledge.

---

## LLM-Level Capability Without Logic in the LM

Decompose what GPT does:

| Capability | Where it lives (not decoder LM) |
|------------|----------------------------------|
| Broad factual Q&A | Structured memory at scale |
| Paraphrase | Cue encoder / embeddings / paraphrase traces |
| Follow-ups | Session + binding interrogative profile |
| Agents | Harness loop; lucidity commits `tool_call` per turn |
| Domain reasoning | Operators, event graphs, harness validators |
| Math / science | SymPy, sim, units — tools, not weights |
| Fluency | Decoder realizes committed text |

### Open-ended reasoning / “thought”

Not one lucidity collapse per user message. **Thought = episode orchestrator:**

```text
while not done:
  perceive (goal + observation + prior steps)
  cue → DMF → binding → basins
  lucidity commits ONE: subgoal | inference | tool_call | branch | refuse
  harness acts → observe → promote traces
```

- **Search-mode lucidity** / projector for branching (grid already does this)
- Logic in **loop + operators + validators**, not hidden CoT in LM

**Ceiling:** open-ended **agent** reasoning in fed domains — not GPT-style
know-everything parametric intuition.

### Code generation

**Strong fit:** coding **agent** (policy, repo memory, test loop)  
**Weak fit:** one-shot full file from `concept_bank`

```text
read → plan → edit → run tests → read error → retry
```

Lucid commits **structured actions**; harness materializes patches.

Codegen syntax can live in a **bounded patch tool** (optional harness LM).
Lucid owns plan; tests own truth.

**Requires:** code memory (repo ingest, error→fix traces), binding for symbols/
stack traces, `operator_bank` tools (`read`, `edit`, `run`, `grep`), episode
orchestrator.

---

## Novel Math / Science (If Fundamental)

Text-only `type_of` claims are insufficient. Add:

1. **Representational layer** — equations, rules, quantities, derivation state
2. **Reasoning layer** — apply operators; search over partial derivations;
   validate via harness
3. **Agent loop** — one licensed step per turn

Wikipedia-scale ingest of prose **moves away** from this unless you extract
**rules**, not trivia.

---

## context-op + interference

Not bloat — **defer for MVP**:

- Needed for multi-frame disambiguation, session multi-scope, grid
- Bad Q&A was **not** caused by these stages

---

## Immediate Next Steps (Priority Order)

1. **Bootstrap definitions** — transformer, ML, AI (`type_of` render-ready)
2. **Harden binding → lucidity → decoder** for `definition_query` /
   `mechanism_query`
3. **Binding internal passes** — event parse + 5–10 motion operators +
   InterrogativeProfile
4. **Episode orchestrator** spec — subgoals, `tool_call`, stop rules (reasoning
   + code share this)
5. **One code-fix episode** on this repo (prove agent path)
6. **Constrained decoder policy** — audit that output ⊆ committed units
7. **Ingest improvements** — lead sentences, pronoun→subject, progress logging
8. **context-op + interference** — re-enable when ambiguity/session demands it

**Stop doing:** expecting more Wikipedia regex ingest alone to fix Q&A or to
constitute “training a model.”

---

## One-Sentence Position

> **GPT-like in domains you feed; honest and cheap; logic in the system and
> harness, fluency in a constrained mouth — memory quality is the gating
> variable, not parameter count.**
