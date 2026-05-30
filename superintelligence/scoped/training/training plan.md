```
We train one shared sparse cognitive substrate.Most modules are cheap operators over that substrate.
```

---

# 1. What is actually being trained?

Not ten full networks.

The trainable substrate is:

```
Tracebank:    learned primitive/relation/operator tracesBasin bank:    learned hypothesis attractorsInterference graph:    learned support/conflict linksAffordance tables:    what traces can do in rolesContext policy:    small controller for local scopeLucidity gate:    small calibrated decision systemDecoder:    renderer from committed state to output
```

Most modules are not large neural models.

```
Binding = structured search over active tracesContext-op = local scope/routing controllerInterference = signed compatibility graphBasins = sparse learned attractorsLucidity = validation gateDecoder = expression layer
```

Only these may need conventional neural training:

```
perception front-endcue encoder mappersome decoder surface layermaybe tiny context/lucidity policies
```

Everything else should mostly learn through **local updates**.

---

# 2. Universal training phases

## Phase 1: Train perception as evidence extraction

This is not ARC-specific.

For text/chat, perception extracts:

```
tokens/spansentitiesactionsrelation markersclausesquestions vs assertionsreferencesspeaker intent hintsuncertainty
```

For images/grids/audio, perception extracts:

```
objectsregionsmotionsymbolsboundarieschangesspatial relationsuncertainty
```

Training objective:

```
input → evidence graph
```

Not:

```
input → final answer
```

For text, use cheap signals:

```
span predictionmasked span recoverysentence segmentationreference predictionnext-event predictionquestion/assertion classificationparaphrase consistency
```

For images/grids:

```
object reconstructionsame/differentbefore/after change detectionregion groupingmotion/change prediction
```

This trains perception to expose evidence, not solve everything.

---

## Phase 2: Train cue encoder as high-recall trace addressing

Cue encoder takes evidence and emits a cue cloud.

Training goal:

```
activate all plausible traces needed for interpretation
```

Not top-1 meaning.

For text:

```
"bank" should activate:    financial-bank-like trace    river-bank-like trace    generic-place-like trace
```

For chat:

```
"I don't get it"should activate:    confusion    explanation request    possible frustration    previous-topic reference
```

Training objective:

```
correct useful traces appear in top-kbad traces are ranked lowerambiguity is preserved when needed
```

### Implementation (current)

The cue encoder is a **layered evidence compiler** with checkpoint **promoted routes** — see `scoped/cue encoder & cue cloud.md`.

Training is route promotion into `cue_encoder_map.json`, not neural backprop:

| Mode | When | What it does |
|------|------|--------------|
| **calibrate** | default during Phase 1–3 module training | run encoder on episode → measure recall vs generator gold → patch **only missing** routes → `NO_UPDATE` when recall sufficient |
| **seed** | corpus bootstrap, cold checkpoint | store all gold cue targets + routes from each episode |

Commands:

```
lucid train cue_encoder --mode calibrate --episodes data/phase1/all.jsonl --checkpoint checkpoints/local
lucid train cue_encoder --mode seed --fixture bank --checkpoint checkpoints/local
```

Integration with orchestrator: failures blamed on `cue_encoder_or_DMF` produce `CueEncoderPatch` entries using the same route store; governor shadow-tests before promote.

### Phase map for cue encoder

| Build phase | Corpus scale | Cue encoder focus |
|-------------|--------------|-------------------|
| **Phase 1** | ~420 generator episodes | calibrate on bank/grid/two-event templates; seed checkpoint; prove recall on gold |
| **Phase 2** | ~7k (+ chat pack) | calibrate on chat templates (clarify, follow-up, carryover); relation_index for discourse markers |
| **Phase 3** | ~150k | calibrate on ARC failure mining; widen path tuned via scaling observatory |
| **Phase 4** | 1.5M+ | route index compaction / quantization (binary feature patterns); hot/warm route tiers |

How to train (runtime loop, all phases):

```
1. Run text/input through perception.
2. Cue encoder compiles evidence + promoted routes → cue cloud.
3. Downstream system tries to reconstruct, answer, or act.
4. If recall vs gold is low (calibrate) or validator fails (orchestrator), promote smallest missing route.
5. If high-margin success, governor skips update.
6. If wrong collapse, demote or reject route on shadow failure.
```

This is retrieval-style training, not full reasoning training.

---

## Phase 3: Grow the DMF / tracebank through online clustering

Traces start anonymous:

```
t0001t0002t0003
```

They are not manually inserted concepts.

They form from recurring evidence/cue patterns.

Training rule:

```
if cue pattern matches existing trace:    activate and update trace slightlyif no match:    create provisional traceif provisional trace repeatedly helps:    promote itif it is noisy:    decay or quarantine
```

A trace gains meaning from behavior:

```
what activates itwhat it coactivates withwhich basins it supportswhich bindings it participates inwhich outputs it helps
```

This is how the system learns concepts, relations, conversational intents, object types, reasoning moves, and task patterns.

Use shadow precision while plastic:

```
new trace = higher precisionstable trace = quantizedtrusted trace = frozen/low-bit
```

That is how it stays cheap.

---

## Phase 4: Train binding from successful structure formation

Binding does not need a giant model.

Binding takes active traces and proposes candidate structures:

```
event framesrelation framesquestion framesgoal framestool-use framesvisual object frames
```

For chat/text:

```
"I found money while kayaking which I placed in the bank."
```

Binding should propose:

```
Frame 1:    found money while kayakingFrame 2:    placed money in bank
```

For conversation:

```
"Can you explain that again?"
```

Binding should propose:

```
user_intent = request_explanationtarget = previous_assistant_contentstyle = simpler
```

Training signal:

```
if frame led to correct/lucid response:    strengthen trace-role affinitiesif frame caused wrong answer:    weaken those role assignments
```

This is sparse table learning:

```
trace tX fits ROLE_ACTIONtrace tY fits ROLE_THEMEtrace tZ fits ROLE_DESTINATION
```

Not large backprop.

---

## Phase 5: Train context-op as scope control

Context-op prevents evidence from becoming global noise.

For text/chat, this is huge.

Example:

```
I found money while kayaking which I placed in the bank.
```

Context-op learns:

```
kayaking scopes to the found-eventbank scopes to the placed-eventmoney bridges both
```

For chat:

```
"Actually, make it shorter"
```

Context-op learns:

```
"it" refers to previous output"shorter" applies to output stylenot to the user’s original subject
```

Training objective:

```
assign traces to local context framescontrol which evidence influences which basinpreserve ambiguity when scope is uncertain
```

Training signal:

```
wrong answer due to evidence leakage → update context-opcorrect scoped response → reinforce scope pattern
```

Again, this is a small routing/scoping policy, not a giant model.

---

## Phase 6: Train interference as signed compatibility

Interference learns:

```
what supports whatwhat conflicts with whatwhat should cooperatewhat should compete
```

For text:

```
money + bank + deposit→ positive support for financial-bank basinkayaking + bank→ positive support for river-bank only in the right local framemoney + placed in bank→ weak negative against river-bank reading
```

For chat:

```
"explain simply" + "I'm confused"→ supports beginner-explanation basin"give code only"→ suppress long explanation basin
```

Training rule:

```
if edge helped successful response:    strengthen positive linkif edge caused wrong basin:    strengthen negative linkif uncertain:    keep weak
```

This is a signed graph update.

Very cheap.

---

## Phase 7: Train basins as hypothesis attractors

Basins are learned hypothesis states.

They are not hand-coded labels.

A basin might become:

```
b1042 = financial-bank interpretation patternb8841 = user wants concise rewriteb3290 = causal explanation requestb7112 = code debugging modeb5510 = visual object transformation
```

But internally they are just learned attractors.

Training rule:

```
if basin wins and lucidity passes:    strengthen trace→basin links    increase basin maturityif basin wins and fails:    weaken links    add negative interference    possibly split basinif several basins cooperate:    create basin assembly
```

For general intelligence, basin assemblies matter:

```
user asks for code review= code-understanding basin+ bug-finding basin+ explanation-style basin+ safety/constraint basin
```

For chat, the winning state is often an assembly, not one basin.

---

## Phase 8: Train lucidity as the commitment gate

Lucidity decides:

```
commitpreserve ambiguityask clarificationsearch widerrequest projectionreject
```

For chat/text, lucidity checks:

```
did we understand the user?is the answer grounded?are there unresolved references?is confidence high enough?is the decoder allowed to answer directly?
```

For reasoning:

```
are steps coherent?is there contradiction?did we skip a dependency?
```

For tool/action tasks:

```
do we need verification?do we need a calculator?do we need a web check?
```

Training signal:

```
wrong confident answer → raise thresholdcorrect cautious answer but unnecessary hedge → lower thresholdasked clarification when not needed → adjustfailed to ask clarification when needed → adjust
```

Lucidity is tiny but extremely important.

---

## Phase 9: Train decoder as expression, not intelligence

Decoder receives committed state.

It should not decide truth.

For chat/text, decoder training is:

```
committed internal state+ style policy+ user intent→ natural response
```

Training data:

```
dialogueshelpful answerssummariesexplanationscode responsestool result descriptions
```

But it should be constrained by lucidity.

Bad:

```
decoder invents fluent answer from vague basin state
```

Good:

```
decoder expresses only committed state and uncertainty
```

If the system wants to be a strong chatbot, yes, decoder quality matters. But it should not carry the core reasoning.

---

# 3. Universal training loop

The exact loop should be:

```
for each training example / interaction:    evidence = perception(input)    cue_cloud = cue_encoder(evidence)    active_traces = DMF.activate_or_seed(cue_cloud)    candidate_frames = binding.propose(active_traces)    scoped_frames = contextop.scope(candidate_frames)    support_graph = interference.compute(scoped_frames)    basin_state = basins.settle(support_graph)    decision = lucidity.check(basin_state)    if decision requires projection/tool:        projection = projector.run(basin_state)        decision = lucidity.recheck(basin_state, projection)    output = decoder.render(decision.committed_state)    score = validator(output)    failure_type = assign_blame(run_log, score)    update_only(failure_type)
```

The important part:

```
update_only(failure_type)
```

Not everything.

---

# 4. How the system gets supervised without ARC labels

For general chat/text, you have many validation signals.

## Reconstruction

```
input → internal state → reconstruct meaning/text
```

Not the only objective, but useful early.

## Next-turn prediction

```
given conversation state, predict useful next response type
```

Not raw next-token prediction as the main engine.

## Question answering

```
answer must match known answer or source
```

## Entailment / contradiction

```
does committed state imply correct relation?does it contradict known fact?
```

## Tool-verifiable tasks

```
mathcode executionunit testsretrieval QAlogic puzzlesdata transforms
```

## Human preference / critique

Later, use feedback to improve lucidity/decoder/policies.

## Self-consistency

Run multiple basin paths. If decoder outputs conflict, lucidity learns to preserve ambiguity or search wider.

---

# 5. Why this can still be cheaper than frontier labs

The number of named modules does not determine cost.

Cost depends on:

```
dense global updates vs sparse local updatesfull model activation vs top-k activationfull retraining vs trace/basin editshigh precision everywhere vs quantized mature memory
```

Your architecture should be cheap because:

```
perception and decoder are the only conventional-ish front/back endsDMF is sparse memorybinding is search over active tracescontext-op is a tiny scope policyinterference is signed graph updatesbasins are online attractor clusterslucidity is a small gateprojector is optional
```

You are not training ten GPTs.

You are training:

```
one sparse trace-basin graphwith small controllers around it
```

---

# 6. The correct cheapness strategy

## Make most learning non-gradient

Use gradient learning only for:

```
perception front-endcue encoder mappingdecoder renderertiny lucidity/context policies if needed
```

Use local updates for:

```
trace creationtrace strengtheningtrace affordancesbinding role affinitiescontext scope weightsinterference edgesbasin centersbasin assemblies
```

## Make mature memory low-bit

```
new/plastic = higher precisionstable = low-bittrusted = frozen/quantized
```

## Make most examples no-update

If the system is already correct with high lucidity:

```
do not train
```

Maybe update counters only.

## Make projector optional

Do not pay for simulation when direct lucidity is enough.

## Train on verifiable tasks for reasoning

Use chat data for language and interaction.

Use verifiable tasks for truth/reasoning:

```
mathcodelogicretrieval QAstructured transformationsformal checkssimulatorstool outputs
```

This gives hard signals without needing giant preference data.

---

# 7. Blame assignment is the core

Every failure should update one responsible piece.

|Failure|Update|
|---|---|
|Missed evidence|perception|
|Good evidence, wrong traces|cue encoder / DMF|
|Good traces, wrong structure|binding|
|Good structure, wrong scope|context-op|
|Good scope, wrong support/conflict|interference|
|Wrong hypothesis wins|basin links/split|
|Good hypothesis, bad commit decision|lucidity|
|Good committed state, bad wording|decoder|
|Need consequence test|projector/tool path|
|Novel stable pattern|new trace/basin|

This is how you prevent the “10 modules = 10x training cost” problem.

---

# 8. First practical training build

For a generally intelligent text/chat model, I’d build in this order:

```
1. Text perception + evidence graph2. Cue encoder + DMF trace activation3. Simple decoder reconstruction4. Binding for event/relation frames5. Context-op for local scope6. Interference + basins7. Lucidity gate8. Tool/projector for verifiable tasks9. Continual learning updates10. Multimodal/grid/game adapters later
```

Do not start with ARC.

Start with language and simple verifiable reasoning.

Then add:

```
structured grid tasksimage evidenceagent tasksscientific tools
```

This keeps the architecture general.

---

# 9. Training curriculum for general intelligence

## Level 1: Language grounding

```
sentence meaningword sense ambiguityreference resolutionevent framesquestion vs assertionconversation intent
```

## Level 2: Structured reasoning

```
relationscausesif-thencomparisonsplansmulti-step instructions
```

## Level 3: Verifiable tasks

```
mathcodelogicdata transformsretrieval QA
```

## Level 4: Multimodal structure

```
imagesgridsdiagramstables
```

## Level 5: Agent loop

```
stategoalactionfeedbackplanning
```

ARC becomes one evaluation/training class under Level 4, not the foundation.