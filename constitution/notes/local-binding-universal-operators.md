# Local binding and universal operators

This note describes a direction for making Lucid understand relations, actions,
state changes, definitions, grids, and analytical tasks without turning binding
into a brittle pile of hand-coded cases.

The short version:

```text
shared graph IR + checkpointed operator bank

inside Lucid:
  local binding uses operators to build candidate frames

outside / beside Lucid:
  projector or reasoner uses the same operators to simulate consequences

lucidity:
  remains the only commit gate
```

## Problem

Lucid needs common-sense structure such as:

```text
if X is inside Y, and Y moves, X probably moves too
if X is holding Y, and X moves, Y probably moves too
if X gives Y to Z, possession changes
if X breaks Y, Y may stop functioning
```

But implementing this as direct code like this would be wrong:

```text
if verb == "walk": move person
if verb == "drive": move person and car
if verb == "carry": move held object
```

That design is narrow, hard to train, and fragile under metaphor, missing
entities, new domains, grids, analytical questions, or definitions.

The fix is not a giant world-model object. The fix is a small universal
relational substrate plus a checkpoint-backed bank of composable operators.

## Design Goals

- Binding is local, not global.
- Operators are graph patterns, not Python if/else branches.
- Rules are defaults with confidence, not absolute truth.
- The same checkpointed knowledge can be used by binding and by external
  consequence testing.
- The substrate supports physical common sense, concepts, grids, analytical
  constraints, explanations, and aptitude-style tasks.
- Every inferred structure carries evidence and provenance.
- Lucidity decides when an interpretation is safe to express.

## Core Boundary

Binding should not answer questions by itself.

Binding should answer:

```text
Within this local frame, what entities, events, roles, relations, states,
constraints, or transforms might explain the evidence?
```

The projector / reasoner should answer:

```text
If these candidate relations and events are true, what follows?
```

Lucidity should answer:

```text
Is the result supported enough to commit, or should Lucid preserve ambiguity,
search wider, request projection, or recheck binding?
```

## Pipeline Shape

The preferred architecture is a two-pass context relationship:

```text
perception
  -> context_seed
  -> local_binding
  -> context_op
  -> interference
  -> basins
  -> projector / reasoner when needed
  -> lucidity
  -> decoder
```

### Context Seed

`context_seed` creates provisional local regions or frames from perception:

```text
"The man walked to the car, then drove it away."

seed_frame_1:
  span: "The man walked to the car"
  event_hint: walk
  units: man, walked, car

seed_frame_2:
  span: "then drove it away"
  event_hint: drive
  units: drove, it, away
```

These frames are not conclusions. They are local workspaces.

### Local Binding

Binding runs inside each provisional frame:

```text
bind(seed_frame_1, operator_bank)
bind(seed_frame_2, operator_bank)
```

It should not globally mix all entities and events in the whole input.

Example local outputs:

```text
frame_1:
  event = walk
  agent = man
  destination = car
  moved_entity = man

frame_2:
  event = drive
  driver = man or unresolved
  vehicle = it
  moved_entities = driver and vehicle
```

### Context-Op

After local binding, context-op links frames and scopes interactions:

```text
frame_2 after frame_1
"it" in frame_2 may refer to car from frame_1
man persists across frames
car changes role from destination object to vehicle
```

Context-op also gates interference so unrelated traces do not leak across local
frames.

## Universal Graph Substrate

The substrate should be a graph, not a table of domain-specific fields.

### Node Kinds

Keep node kinds small and general:

```text
entity
event
state
value
relation
constraint
region
concept
operator
artifact
```

These are broad enough for many tasks:

```text
physical:
  person, car, bag, move event, inside relation

conceptual:
  qubit concept, definition relation, property relation

grid:
  cell, object/component, color value, transform event

analytical:
  variable, greater_than constraint, sequence relation
```

### Edge Kinds

Edges should also be general:

```text
role(event, participant)
relation(source, target)
state(entity, value)
property(entity_or_concept, value)
type_of(entity_or_concept, class)
before(event_a, event_b)
causes(a, b)
supports(a, b)
conflicts(a, b)
evidence(edge_or_node, source_ref)
```

Avoid adding fields like:

```text
car_location
driver_location
bag_inside_car
qubit_definition_text
grid_red_cell_count
```

Those are brittle. Represent them as graph structure instead.

## Operator Bank

An operator is a reusable graph pattern with possible effects.

```text
operator:
  id
  family
  pattern
  effects
  constraints
  default_confidence
  exceptions
  evidence_policy
  learned_stats
```

Operators are checkpoint records, not hardcoded stage logic. Code provides the
matcher and applier. Checkpoints provide most of the domain knowledge.

### Example: Motion Propagation

Instead of special-casing driving, use a more general operator:

```text
operator: motion_propagates_through_coupling

pattern:
  coupled(X, Y, strength=S)
  moves(Y, destination=L)

effect:
  at(X, L) with confidence S
```

Other operators can derive coupling:

```text
inside(X, Y)      -> coupled(X, Y, strong)
attached(X, Y)    -> coupled(X, Y, very_strong)
holding(Y, X)     -> coupled(X, Y, strong)
on_top_of(X, Y)   -> coupled(X, Y, medium)
wearing(Y, X)     -> coupled(X, Y, strong)
```

Driving then becomes a composition:

```text
drive(person, vehicle, destination)
  -> controls(person, vehicle)
  -> inside_or_on(person, vehicle)
  -> moves(vehicle, destination)
  -> coupled(person, vehicle, strong)
  -> at(vehicle, destination)
  -> at(person, destination)
```

Walking is simpler:

```text
walk(person, destination)
  -> moves(person, destination)
  -> at(person, destination)
```

No special "man and car both move" rule is needed.

## Operator Families

The same substrate supports different families.

### Physical Common Sense

```text
inside
holding
attached
on_top_of
at
controls
opens
breaks
repairs
transfers_possession
moves_with
```

### Concept and Definition Questions

For a question like:

```text
what is a qubit?
```

The local frame is not physical. It is a definition query:

```text
frame:
  intent = define
  target_concept = qubit
  expected_answer_type = concept_definition
```

The concept memory may contain:

```text
concept: qubit
type_of: unit_of_quantum_information
analog_of: classical_bit
property: can_exist_in_superposition
measurement_result: classical 0 or 1
domain: quantum_computing
```

The answer comes from concept retrieval and lucidity checks, not from physical
motion rules.

### Grid and Pattern Tasks

Grid tasks use the same graph idea:

```text
cell_at(object, row, col)
color(object, red)
connected_component(object)
translate(object, dx, dy)
recolor(object, from_color, to_color)
mirror(object, axis)
```

The projector can test candidate transforms against train pairs before lucidity
commits.

### Analytical and Aptitude Tasks

Analytical tasks can be represented as constraints:

```text
greater_than(A, B)
greater_than(B, C)
  -> greater_than(A, C)

before(A, B)
before(B, C)
  -> before(A, C)

member_of(X, Set)
count(Set, N)
```

The reasoner applies constraint operators, then lucidity checks whether the
derived answer is forced, likely, or still ambiguous.

### Explanation Tasks

Explanation tasks can use causal and support edges:

```text
causes(A, B)
enables(A, B)
prevents(A, B)
requires(A, B)
because_of(claim, support)
```

The decoder should express only the support graph that lucidity approved.

## Routing

Lucid should not force every input into one operator family.

The system should route local frames by evidence:

```text
definition query:
  use concept operators and concept memory

physical event:
  use event, relation, state-change, and coupling operators

grid task:
  use grid transform operators and projector checks

analytical question:
  use constraint operators and exact/near-exact consequence testing

explanation:
  use causal/support graph operators
```

Routing is itself plural. A frame can carry multiple candidate families until
lucidity or projection narrows them.

## Checkpoint Shape

This can be backed by a single shared checkpoint area with multiple stores:

```text
schema_bank.json
operator_bank.json
concept_bank.json
relation_aliases.json
operator_stats.json
```

These could later be folded into existing checkpoint stores if that is cleaner,
but the conceptual split matters.

### `operator_bank.json`

Stores reusable graph rewrite operators:

```json
{
  "operators": [
    {
      "operator_id": "motion_propagates_through_coupling",
      "family": "physical",
      "pattern": [
        ["relation", "coupled", "X", "Y"],
        ["event", "moves", "Y", "L"]
      ],
      "effects": [
        ["state", "at", "X", "L"]
      ],
      "default_confidence": 0.75,
      "exceptions": ["metaphor", "fragile_support", "uncertain_occupancy"]
    }
  ]
}
```

### `relation_aliases.json`

Maps language evidence to graph relations without baking mechanics into words:

```json
{
  "aliases": [
    {
      "surface_pattern": "in",
      "relation_candidates": ["inside", "located_in"],
      "confidence": 0.6
    },
    {
      "surface_pattern": "wearing",
      "relation_candidates": ["wearing", "attached_soft"],
      "confidence": 0.8
    }
  ]
}
```

### `concept_bank.json`

Stores auditable concept facts and definition support:

```json
{
  "concepts": [
    {
      "concept_id": "qubit",
      "relations": [
        ["type_of", "unit_of_quantum_information"],
        ["analog_of", "classical_bit"],
        ["property", "superposition_possible"]
      ],
      "sources": []
    }
  ]
}
```

## Learning

Most knowledge should enter through training data, not manual code.

### Bootstrap Seeds

It is acceptable to seed a tiny set of primitives:

```text
inside
at
holding
attached
before
after
equals
greater_than
color
cell_at
type_of
property
```

But the rule should be:

```text
primitive code is small
operator knowledge lives in checkpoints
```

### Synthetic Training

Generate many small episodes:

```text
Initial:
  A is inside B.
  B is inside C.
  C moves to D.

Question:
  Where is A?

Gold:
  D
```

Vary relation chains, exceptions, metaphors, distractors, grid transforms, and
constraint puzzles.

### Contrastive Updates

Train operators by contrast:

```text
winner:
  inside + move implies location propagation

loser:
  mere proximity + move does not necessarily imply propagation
```

This teaches the model reusable boundaries instead of memorizing examples.

### Projection-Based Validation

For tasks with checkable consequences, projector/reasoner should test candidate
operators:

```text
candidate operator -> predicted graph/grid/state -> validator -> score
```

High-confidence correct predictions can strengthen operator stats. Failed
predictions create audit-backed training cases.

## Avoiding Brittleness

Use these constraints:

```text
1. No action-specific Python branches except minimal bootstrap primitives.
2. Operators produce candidates with confidence, not final truth.
3. Word aliases are separate from mechanics.
4. Binding applies operators locally, never globally.
5. Context-op links local frames after binding.
6. Exceptions lower confidence or fork alternatives.
7. Every inferred node/edge has evidence/provenance.
8. Projector/reasoner validates consequences when possible.
9. Lucidity remains the only commit gate.
10. New task types add operator families, not new one-off IR fields.
```

## Example: Walking vs Driving

Input:

```text
The man walked to the car, then drove it away.
```

Context seed:

```text
frame_1: "The man walked to the car"
frame_2: "then drove it away"
```

Local binding:

```text
frame_1:
  walk(man, car)
  moves(man, car)
  at(man, car)

frame_2:
  drive(man?, it, away)
  controls(man?, it)
  moves(it, away)
```

Context-op:

```text
it -> car
man? -> man
frame_2 after frame_1
```

Reasoner:

```text
inside_or_on(man, car) likely during drive
coupled(man, car, strong)
moves(car, away)
therefore at(man, away) and at(car, away)
```

Lucidity:

```text
commit if role resolution and confidence pass
preserve ambiguity if "it" is unresolved or "drove" is metaphorical
```

## Example: What Is a Qubit?

Input:

```text
what is a qubit?
```

Context seed:

```text
frame_1:
  intent = define
  target = qubit
```

Local binding:

```text
definition_query(target_concept=qubit)
```

Retriever:

```text
concept_bank.lookup("qubit")
```

Reasoner:

```text
type_of(qubit, unit_of_quantum_information)
analog_of(qubit, classical_bit)
property(qubit, superposition_possible)
```

Lucidity:

```text
commit only if concept support is present
otherwise preserve uncertainty or request retrieval/source
```

Decoder:

```text
A qubit is a unit of quantum information. It is analogous to a classical bit,
but it can represent quantum state information such as superposition before
measurement.
```

## Example: Grid Transform

Input:

```text
train pair: red object shifts one cell right
test grid: red object at new position
```

Local binding:

```text
object(component_1)
color(component_1, red)
transform_candidate(translate, dx=1, dy=0)
```

Projector:

```text
apply translate(dx=1, dy=0)
score against train output
apply to test input
```

Lucidity:

```text
commit if projection fit passes
preserve ambiguity if multiple transforms fit
```

## Implementation Direction

Do not start by rewriting the whole pipeline.

A practical sequence:

```text
1. Add graph/operator IR types.
2. Add a tiny operator matcher/applier.
3. Add checkpoint store loaders for operator/concept banks.
4. Let binding attach local graph candidates to CandidateFrame payloads.
5. Let context-op link local frame graphs.
6. Let projector/reasoner apply operators for consequence tests.
7. Add audits for matched operators, inferred edges, and rejected alternatives.
8. Add synthetic training recipes for physical, concept, grid, and constraint tasks.
```

The first working version should be small but end-to-end:

```text
inside + move -> location propagation
definition query -> concept lookup
grid translate -> projection check
greater_than transitivity -> analytical check
```

That proves universality without pretending the world model is complete.

## Summary

The architecture should not make binding a global common-sense brain.

It should make binding a local structure builder that consults a shared,
checkpointed operator bank. The same bank can then power external reasoning and
projection. This gives Lucid reusable common-sense mechanics while keeping the
system plural, auditable, trainable, and broad enough for physical questions,
concept definitions, grid patterns, analytical tasks, and explanations.
