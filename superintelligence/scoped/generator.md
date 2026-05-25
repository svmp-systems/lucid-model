# Generator build spec — lucid-model synthetic episode factory

Sits alongside the main build plan. Produces the data fixtures that `p1-26`, `p1-28`, and `p1-29` depend on, plus the larger training corpora for Phase 3+.

---

## Repo location

```
lucid/
  generator/
    __init__.py
    core/
      episode.py          # Episode dataclass + JSON serialiser
      gold.py             # GoldLabels dataclass
      validator.py        # Validator protocol + built-in impls
    templates/
      text/
        ambiguous_destination.py
        multi_event_reference.py
        local_override.py
        causal_chain.py       # Phase 3+
        nested_scope.py       # Phase 3+
      grid/
        position_shift.py
        recolor.py
        symbol_map.py         # Phase 3+
        assembly.py           # Phase 3+ (move + recolor combined)
      agent/                  # Phase 4+
    samplers/
      slot_sampler.py         # weighted random slot filling
      ambiguity_dial.py       # controls how resolved each episode is
      seed.py                 # seeded RNG wrapper
    exporters/
      jsonl.py
      hf_dataset.py           # Phase 3+
    cli.py                    # lucid-gen command
  tests/
    generator/
      test_ambiguous_destination.py
      test_multi_event_reference.py
      test_local_override.py
      test_grid_position_shift.py
      test_episode_roundtrip.py
```

---

## Core data types

### Episode

Every generated sample is an Episode. This is the unit that trains any module.

```python
@dataclass
class Episode:
    episode_id: str          # uuid4
    modality: str            # "text" | "grid" | "agent"
    template_id: str         # e.g. "ambiguous_destination_v1"
    raw_input: str | dict    # surface form shown to model
    gold: GoldLabels         # full ground truth
    validator: str           # validator key e.g. "exact_sense"
    seed: int                # RNG seed — fully reproducible
    meta: dict               # template params used (for debugging)
```

### GoldLabels

All targets in one place. Each module reads what it needs.

```python
@dataclass
class GoldLabels:
    # perception targets
    spans: list[Span]              # id, surface, kind_hint, position
    markers: list[Marker]          # id, surface, marker_type_hints
    regions: list[Region]          # id, role_hint, member_span_ids
    uncertainty_flags: list[UncertaintyFlag]

    # cue encoder targets
    trace_activations: list[TraceTarget]
    # TraceTarget: { trace_family, weight, evidence_ref, keep_alive: bool }
    ambiguity_policy: str          # "preserve_plural" | "allow_narrow"

    # context-op targets
    scope_assignments: list[ScopeAssignment]
    # ScopeAssignment: { span_id, primary_frame, secondary_frames }
    interference_gates: list[GateDirective]

    # basin / lucidity targets
    basin_families: list[BasinTarget]
    # BasinTarget: { family_hint, frame_id, confidence }
    lucidity_target: str           # "COMMIT" | "PRESERVE_AMBIGUITY"
    lucidity_rationale: str        # human-readable reason

    # decoder target
    expected_answer: str | dict | None
    validator_result: bool | None  # None = no auto-check available
```

---

## Template contract

Every template implements this protocol:

```python
class EpisodeTemplate(Protocol):
    template_id: str
    version: str

    def sample(self, rng: Random, dial: AmbiguityDial) -> Episode:
        ...

    def validate(self, episode: Episode) -> bool:
        # self-check: gold labels are internally consistent
        ...
```

Templates never call external models. Pure Python + RNG.

---

## Ambiguity dial

Controls how resolved each generated episode is. Passed into every template at sample time.

```python
@dataclass
class AmbiguityDial:
    resolution: float   # 0.0 = maximally ambiguous, 1.0 = fully resolved
    # Generator uses this to choose:
    #   - how strongly scoping context appears
    #   - what the gold lucidity_target should be
    #   - what weight ratio to assign competing trace families
```

Example effect on ambiguous_destination template:

```
dial.resolution = 0.1  →  no scoping context, bank_sense [0.52, 0.48]
                           lucidity_target: PRESERVE_AMBIGUITY

dial.resolution = 0.7  →  strong scoping context present
                           bank_sense [0.84, 0.16]
                           lucidity_target: COMMIT

dial.resolution = 1.0  →  unambiguous surface form used
                           lucidity_target: COMMIT, confidence high
```

---

## Phase 1 templates (text)

### T1 — ambiguous_destination

Pattern: two events sharing a theme, ambiguous final location.

Slot tables:

```python
OUTDOOR_CONTEXTS = [
    "while kayaking", "during a hike", "while fishing",
    "after swimming", "on a trail run", "while canoeing"
]

FINANCIAL_ACTIONS = [
    "deposited", "placed", "put", "stored", "left"
]

AMBIGUOUS_LOCATIONS = [
    "bank", "vault", "safe"   # "bank" is the main one; others for variety
]

THEMES = [
    "the cash", "some money", "the coins", "her savings",
    "the funds", "the bills"
]

AGENTS = ["Alex", "Jamie", "she", "he", "they", "I", "Sam", "Jordan"]

FIND_VERBS = ["found", "discovered", "picked up", "came across"]
```

Render logic (simplified):

```python
def sample(self, rng, dial):
    agent = rng.choice(AGENTS)
    theme = rng.choice(THEMES)
    find_verb = rng.choice(FIND_VERBS)
    deposit_verb = rng.choice(FINANCIAL_ACTIONS)
    location = rng.choice(AMBIGUOUS_LOCATIONS)

    # dial controls whether outdoor context is present
    if dial.resolution > 0.3:
        ctx = rng.choice(OUTDOOR_CONTEXTS)
        raw = f"{agent} {find_verb} {theme} {ctx} and later {deposit_verb} it in the {location}."
        # outdoor context present → river sense suppressed
        river_weight = max(0.05, 0.45 - dial.resolution * 0.4)
    else:
        raw = f"{agent} {find_verb} {theme} and later {deposit_verb} it in the {location}."
        river_weight = 0.45

    financial_weight = 1.0 - river_weight

    gold = GoldLabels(
        spans=[
            Span(find_verb, "verb_span"),
            Span(theme, "noun_span"),
            Span(location, "noun_span", uncertainty="polysemy" if river_weight > 0.2 else None),
            ...
        ],
        trace_activations=[
            TraceTarget("financial_action_like", financial_weight, keep_alive=True),
            TraceTarget("river_location_like", river_weight, keep_alive=(river_weight > 0.2)),
            TraceTarget("outdoor_context_like", 0.7 if ctx else 0.0, keep_alive=False),
        ],
        scope_assignments=[
            ScopeAssignment(find_verb, primary_frame="F1"),
            ScopeAssignment(ctx_span, primary_frame="F1"),
            ScopeAssignment(deposit_verb, primary_frame="F2"),
            ScopeAssignment(location, primary_frame="F2"),
            ScopeAssignment(theme, primary_frame="F1", secondary_frames=["F2"]),
        ],
        lucidity_target="COMMIT" if dial.resolution > 0.5 else "PRESERVE_AMBIGUITY",
        expected_answer="financial" if financial_weight > 0.7 else None,
        validator_result=True,
    )
    return Episode(raw_input=raw, gold=gold, template_id=self.template_id, ...)
```

---

### T2 — multi_event_reference

Pattern: pronoun or noun carries across two events; system must track reference without collapsing scope.

Slot tables:

```python
EVENT_PAIRS = [
    {
        "e1": ("found", "it", "near the river"),
        "e2": ("sold", "it", "at the market"),
        "theme": ["an old coin", "a carved stone", "a glass bottle"],
    },
    {
        "e1": ("spotted", "her", "in the park"),
        "e2": ("recognised", "her", "from the photo"),
        "theme": ["the woman", "the girl", "the stranger"],
    },
    ...
]
```

Gold adds a reference_hint between the two pronoun spans. Scope assignments ensure e1 context doesn't leak into e2 resolution.

Lucidity target: usually COMMIT (reference is resolvable), but PRESERVE_AMBIGUITY when the referent is genuinely unclear.

---

### T3 — local_override

Pattern: instruction applies only to a scoped part; system must not apply it globally.

Examples:

```
"Make the second paragraph shorter, but keep the first one detailed."
"Apply that change only to the Python version, not the JavaScript one."
"Actually, revert just the last edit."
```

Gold encodes:

- which instruction token scopes to which region
- interference gate: instruction must not leak to non-target region
- lucidity target: COMMIT to scoped interpretation

Scope leak is the failure mode. This template specifically trains context-op to prevent it.

---

## Phase 1 templates (grid)

### G1 — position_shift

Generate a before/after grid pair where one object moves.

```python
GRID_SIZE = (6, 6)   # small for Phase 1

def sample(self, rng, dial):
    color = rng.choice(COLORS)
    shape = rng.choice(SHAPES)     # "square" | "cross" | "dot"
    pos_before = random_position(rng, GRID_SIZE)
    direction = rng.choice(["left","right","up","down"])
    distance = rng.randint(1, 3)
    pos_after = apply_move(pos_before, direction, distance, GRID_SIZE)

    input_grid = empty_grid(GRID_SIZE)
    output_grid = empty_grid(GRID_SIZE)
    place(input_grid, shape, color, pos_before)
    place(output_grid, shape, color, pos_after)

    gold = GoldLabels(
        change_hints=[ChangeHint("position_shift", pos_before, pos_after, direction, distance)],
        trace_activations=[
            TraceTarget("position_shift_like", 0.88),
            TraceTarget("shape_preserved_like", 0.93),
            TraceTarget("color_preserved_like", 0.91),
        ],
        lucidity_target="COMMIT",
        expected_answer=output_grid,
        validator_result=True,
    )
    return Episode(raw_input={"input": input_grid, "output": output_grid}, gold=gold, ...)
```

---

### G2 — recolor

Same structure but color changes, position preserved. Trains the system to distinguish attribute-change from position-change. Combined with G1 to create ambiguous cases (both happen simultaneously) in Phase 3.

---

## Ambiguity dial distribution per corpus build

For Phase 1 micro fixtures (used in integration tests):

```
20 episodes resolution 0.0–0.2   (maximally ambiguous)
20 episodes resolution 0.4–0.6   (moderate)
20 episodes resolution 0.8–1.0   (clearly resolved)
```

For Phase 3 training corpus (100k text episodes):

```
30% resolution 0.0–0.3
40% resolution 0.3–0.7
30% resolution 0.7–1.0
```

Don't over-represent easy (high-resolution) episodes. The system learns the hard cases from the ambiguous band.

---

## Validators

Each template registers a validator key. Validators run at episode generation time (self-check) and again at eval time.

```python
class Validator(Protocol):
    key: str
    def check(self, episode: Episode, model_output: str | dict) -> bool: ...

# Built-ins
ExactSenseValidator       # "bank" → "financial" matches expected
ExactGridValidator        # output grid matches gold grid cell-by-cell
ScopeConsistencyValidator # model's scope assignments match gold
LucidityDecisionValidator # model emitted correct COMMIT/PRESERVE decision
ReferenceResolutionValidator
```

---

## CLI

```bash
# generate and write to disk
lucid-gen text --template ambiguous_destination \
               --count 10000 \
               --resolution-dist "30:40:30" \
               --seed 42 \
               --out data/episodes/ambiguous_destination_10k.jsonl

# generate and preview one episode
lucid-gen text --template ambiguous_destination --count 1 --preview

# generate full Phase 1 fixture pack
lucid-gen pack phase1 --out data/fixtures/phase1/

# validate a generated file (self-check gold consistency)
lucid-gen validate data/episodes/ambiguous_destination_10k.jsonl

# stats on a generated file
lucid-gen stats data/episodes/ambiguous_destination_10k.jsonl
```

---

## Self-check requirements

Every generated episode must pass before being written to disk:

```
1. Gold spans cover all surface tokens they claim
2. Scope assignments reference only declared frame IDs
3. Trace activations sum ≤ 1.0 per competing family pair
4. lucidity_target == COMMIT only when top trace weight > 0.6
5. Grid episodes: output grid is reachable from input via declared change
6. No duplicate episode_ids in a batch
```

Run with `lucid-gen validate` before any training job.

---

## Commit placement in build plan

Canonical IDs live in `superintelligence/build.md`. Generator block (after cognitive `p1-24`, before orchestrator `p1-26`):

```
p1-25a–h  generator (Episode IR → templates → lucid-gen → phase1 pack)
p1-26–26e training orchestrator MVP
p1-27     seed trace/basin packs compiled FROM phase1 fixture gold
```

Do not reuse `p1-25a/b` for orchestrator components — those IDs are reserved for generator in the unified plan.

---

## Volume targets by phase

|Phase|Text episodes|Grid episodes|Total|
|---|---|---|---|
|1 (fixtures)|300|120|420|
|2 (smoke)|5k|2k|7k|
|3 (training)|100k|50k|150k|
|4 (scale)|1M|500k|1.5M|
|5 (full)|5M+|2M+|7M+|

Phase 1–2 generated on laptop in minutes. Phase 3 generated on laptop overnight. Phase 4–5 generated on cheap cloud CPU (parallelised, trivial cost).

---

## What this does not include

- LLM-assisted paraphrasing (add in Phase 3 for surface diversity)
- Agent/environment episodes (Phase 4)
- Multimodal image episodes (Phase 4)
- Human-in-the-loop episode correction UI (Phase 3+)

These are extensions. The core generator is pure Python, no models, no external dependencies beyond standard library + dataclasses.