train it on **episodes**, not just examples.

A training episode should look like:

```
TrainingEpisode {    raw_input    modality    conversation_or_task_context    task_intent    allowed_tools_or_environment    expected_output_optional    validator_or_reward    feedback    run_log}
```

Your cue encoder design already assumes this kind of structure: it expects candidate evidence, relation markers, task intent, context state, uncertainty, constraints, and budget — not just raw text or final labels.

So the data should be layered.

---

# 1. Raw input data

This is what the model sees first.

Use broad general data, not only ARC-style tasks.

## Text / chat

```
user messageconversation historydocumentsquestionsinstructionsdialogue turnstool results
```

Examples:

```
"Explain quantum tunneling simply.""Rewrite this email.""I found money while kayaking which I placed in the bank.""Debug this Python error.""What should I do next?"
```

## Structured reasoning

```
math problemslogic puzzlescode tasksdata transformationsproof snippetscausal reasoning tasksplanning tasks
```

Examples:

```
input → expected answercode → unit testsclaim → entailment/contradiction labelplan state → valid next action
```

## Perception / multimodal

```
imagesdiagramstablesscreenshotsgridsvideos/frame sequencesenvironment observations
```

## Agent/environment data

```
stateactionnext statereward/successgoal hidden or explicittool callsfeedback
```

This is needed for planning and self-research later.

---

# 2. Evidence-graph supervision

For perception, the target is not the final answer.

The target is:

```
raw input → evidence graph
```

You want data that teaches perception to extract:

```
candidate unitscandidate relation markersgrouping hintsposition/order hintschange hintsuncertainty flags
```

## For text

Training target:

```
TextEvidenceGraph {    spans    entities    actions    relation markers    clauses    references    intent hints    uncertainty}
```

Example input:

```
"I found money while kayaking which I placed in the bank."
```

Evidence target:

```
units:    found, money, kayaking, placed, bankmarkers:    while, which, ingrouping hints:    found-money-kayaking    placed-money-bankuncertainty:    bank sense ambiguous    which attachment uncertain
```

## For images/grids

Training target:

```
VisualEvidenceGraph {    objects    regions    boundaries    symbols    spatial relations    before/after deltas    uncertainty}
```

You can create lots of this synthetically.

This is cheap and useful because the system is only learning to expose evidence, not solve everything.

---

# 3. Cue-cloud training data

The cue encoder should learn:

```
evidence graph → cue cloud
```

The cue cloud should contain:

```
weighted trace pressuresrelational pressuressoft context priorsstructural hintsambiguity policyretrieval policy
```

Early on, you will not have mature trace IDs. So use two stages.

## Stage A: pseudo-traces

Cluster evidence patterns into temporary trace IDs.

```
money-like pattern → t042bank-like ambiguous pattern → t119source-marker-like pattern → t301
```

## Stage B: successful-run traces

After the system starts solving tasks, use lucidity-approved paths as positive training targets.

Training target:

```
CueCloudTarget {    useful_traces    useful_relation_traces    useful_context_priors    traces_to_preserve_as_alternatives    harmful_early_collapses}
```

The goal is **high recall**, not immediate top-1 interpretation.

---

# 4. Tracebank / DMF training data

The DMF learns from streams of cue clouds.

Input:

```
CueCloud
```

Training signal:

```
which traces activatedwhich traces helped successwhich traces caused failurewhich novel patterns repeated
```

You do not need manual labels like:

```
t042 = money
```

Instead, each run provides:

```
trace t042 activatedtrace t042 participated in successful basin b17trace t042 often appears with t301 and t119trace t042 causes failure when bound as source
```

So DMF data is mostly:

```
cue cloud streams + success/failure credit
```

---

# 5. Binding training data

Binding needs examples of:

```
active traces → candidate structures
```

Good data types:

```
event extractionsemantic role labelingrelation extractionquestion parsingtool-use intent parsingobject-relation parsinggrid/object transform frames
```

But do not force one answer too early.

Binding targets should be candidate frames.

Example target:

```
Input traces:    found, money, kayaking, placed, bankGood candidate frames:    F1 = found(theme=money, context=kayaking)    F2 = placed(theme=money, destination=bank)Bad candidate frame:    placed(theme=bank, destination=money)
```

Training signal:

```
successful frame → strengthen trace-role affinitiesfailed frame → weaken trace-role affinities
```

---

# 6. Context-op training data

Context-op needs data where **scope matters**.

Use lots of examples with mixed signals.

## Text examples

```
"I found money while kayaking which I placed in the bank.""She saw the bat near the cave and later used it in the game.""Make the second paragraph shorter, but keep the first one detailed.""Actually, apply that only to the Python version."
```

Targets:

```
which evidence belongs to which local framewhich traces should influence each otherwhich references carry across frameswhich context should stay local
```

## General task examples

```
same object, different rule depending on local conditionsame word, different sense by clausesame action, different consequence by environmentsame symbol, different local meaning by legend/rule
```

Context-op training signal comes from downstream success:

```
bad scope caused wrong collapse → update context-opgood scope preserved correct interpretation → reinforce
```

---

# 7. Interference training data

Interference needs examples of support and conflict.

Data should include:

```
positive co-occurrencenegative co-occurrencecontradictionscompatible structuresincompatible structurescooperating hypothesescompeting hypotheses
```

Example:

```
money + deposit + bank→ supports financial-bank basinkayaking + river + bank→ supports river-bank basinmoney + placed in bank→ weakly suppresses river-bank basinobject moved + object recolored→ can cooperate
```

Input:

```
scoped candidate frames
```

Target/update:

```
edges that helped successful result → positiveedges that caused wrong result → negative
```

This data mostly comes from run logs.

---

# 8. Basin training data

Basins train from repeated successful hypothesis patterns.

Input:

```
active tracescandidate framesscoped contextsinterference edges
```

Signal:

```
did this basin or basin assembly lead to a correct/lucid result?
```

You need tasks that force hypothesis competition:

```
ambiguous word sensemulti-step instructionscausal reasoningtool-use decisionsvisual interpretationobject transformationsplanning under uncertainty
```

Basin target is not a human label. It is:

```
this attractor pattern workedthis one failedthese two basins cooperatethese two compete
```

---

# 9. Lucidity training data

Lucidity needs examples of:

```
when to commitwhen to preserve ambiguitywhen to ask clarificationwhen to search widerwhen to request projection/toolwhen to reject
```

Good data sources:

```
QA with answer confidenceambiguous questionsunderspecified instructionstool-verifiable tasksmath/code taskscontradiction examplesfailed reasoning traces
```

Training input:

```
basin energymargincoveragecoherenceconflictcontext consistencybinding stabilityprojection fitmaturity
```

Target:

```
COMMIT / PRESERVE / PROJECT / CLARIFY / SEARCH / REJECT
```

This is a small but critical dataset.

---

# 10. Projector training data

Only needed for domains where consequences must be generated.

Use:

```
math executioncode unit testssimulatorsgrid transformationsplanning environmentsformal checkerstool-use outcomesscience simulations
```

Input:

```
candidate basin state
```

Target:

```
predicted consequencecandidate outputpass/fail result
```

For general chat, projector is optional.

For reasoning and agents, it becomes important.

---

# 11. Decoder training data

Decoder data is:

```
committed internal state → external output
```

For text/chat:

```
committed frames + decoder policy → helpful natural language answer
```

For tools:

```
committed action state → tool call
```

For grids:

```
committed output grid → exact grid format
```

Decoder should not be trained to invent truth. Train it to render.

Add faithfulness data:

```
decoder output → re-encoded claims → compare to committed state
```

If mismatch, train decoder correction.

---

# 12. The most important data type: run logs

After bootstrapping, the system mostly trains from its own run logs.

Every example creates:

```
RunLog {    evidence_graph    cue_cloud    active_traces    candidate_bindings    context_frames    interference_edges    active_basins    lucidity_decision    projector_result    decoder_output    validator_result    failure_type}
```

This is your main training data.

Why?

Because it tells you exactly what to update.

Without run logs, you have to train everything.

With run logs, you can update only:

```
bindingor context-opor interferenceor basin linksor lucidityor decoder
```

---

# 13. Best data mix

For a general intelligent model, I’d use this mix:

```
30% text/chat/instruction data20% structured reasoning and QA15% code/math/tool-verifiable tasks15% multimodal/visual/grid evidence tasks10% planning/environment episodes10% ambiguity/contradiction/underspecification tasks
```

But for early prototypes, prioritize verifiable tasks because they provide clean training signals.

A better early mix:

```
40% verifiable reasoning tasks25% text/chat intent + reference + ambiguity20% synthetic evidence/structure tasks10% tool-use/action episodes5% open-ended conversation polish
```

Why?

Because early on you need strong validators more than style polish.

---

# 14. Concrete example training episode

## Input

```
User:"I found money while kayaking which I placed in the bank."
```

## Target/validator

```
Expected interpretation:bank most likely financial/storage destination;kayaking scopes to found-event.
```

## Episode teaches

```
perception:    identify spans and markerscue encoder:    preserve bank ambiguitybinding:    split into found-event and placed-eventcontext-op:    kayaking scopes to found-event    bank scopes to placed-eventinterference:    money+placed+bank supports financial/storage basinbasins:    financial/storage interpretation wins in frame 2lucidity:    commit because margin highdecoder:    explain without overclaiming
```

This one example trains multiple things, but only the parts that need updating should be changed.

---

# 15. What not to train on too early

Avoid too much:

```
open-ended internet text with no validatorpure next-token predictionstyle-only chat databenchmark-specific templatesunverified reasoning traces
```

Those can make the decoder fluent while leaving the cognitive substrate weak.

Use unverified text mainly for:

```
language exposuresurface fluencycandidate concept formation
```

Use verified tasks for:

```
reasoningluciditybasinsinterferencecontext-opself-repair
```

---

# 16. Final answer

The data going into training should be:

```
1. Raw inputs:   text, chat, images, grids, tools, environments.2. Evidence supervision:   spans, objects, regions, markers, changes, groupings, uncertainty.3. Cue-cloud targets:   useful trace candidates, relation pressures, context priors, ambiguity policies.4. Run logs:   active traces, bindings, contexts, interference, basins, lucidity decisions.5. Validators:   exact answers, unit tests, simulators, tool results, human feedback, formal checks.6. Failure labels:   perception miss, wrong trace, bad binding, bad scope, bad interference, wrong basin, bad lucidity, bad decoder.7. Continual streams:   new concepts, changing contexts, new tools, new domains, old-task retention checks.
```

The cleanest summary:

```
Train perception from evidence labels.Train cue/DMF from high-recall trace activation.Train binding/context/interference/basins from run logs.Train lucidity from commit/fail outcomes.Train decoder from committed-state rendering.Use validators wherever possible.
```