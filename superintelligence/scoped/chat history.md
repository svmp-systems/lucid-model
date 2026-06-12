# Chat History And Session Memory

## Purpose

Chat history is the session-local memory layer for Lucid chat. It lets the system
carry meaning from one turn to the next without creating a separate chatbot path.
Every chat turn still goes through the normal Lucid pipeline:

1. perception
2. cue encoder
3. DMF
4. binding
5. context-op
6. interference
7. basins
8. lucidity
9. decoder

The chat layer prepares session context for the pipeline, records what happened,
and updates only the current session memory.

## Core Rule

Chat history must not be a domain engine.

It should not hardcode cases such as:

```text
if colour question, remember colour
if aptitude question, use aptitude frame
if JEE question, use JEE frame
if grid question, use grid frame
```

Instead, chat history preserves neutral session state. The pipeline interprets
what the current turn means.

The rule is:

```text
chat history preserves session state
the pipeline figures out the domain and meaning
lucidity decides what is safe to commit
```

## Core Behavior

A chat session is one conversation. Turns inside that conversation can refer to
earlier turns in the same session.

Example:

```text
session: blue-chat

turn 1
user: remember the colour blue
assistant: I will remember that the colour is blue.

turn 2
user: what colour did I tell you?
assistant: Blue.
```

The example is only a smoke case. The implementation must work through generic
session memory and pipeline binding, not a special colour rule.

A new session starts with new memory.

```text
session: new-chat

turn 1
user: what colour did I tell you?
assistant: I do not have a colour stored in this session.
```

The important rule is simple: session memory must not leak across sessions.

## Session Boundary

Each session owns its own files under:

```text
audit/chat/<session_id>/
```

Expected files:

```text
session.json
transcript.txt
history.jsonl
memory.json
```

The chat runtime must load only:

```text
audit/chat/<current_session_id>/
```

It must not load:

```text
audit/chat/*
global chat memory
latest previous session
another user's session
```

Cross-session memory can only exist later as an explicit import/export feature,
not as default behavior.

## Turn Memory

Turn memory means the current message can use earlier turns from the same
session. It should include:

- recent turns for immediate flow
- active session-local memories
- active bindings
- unresolved unclear items
- user preferences stated in this session
- compact summaries of older session history

It should not blindly stuff the entire transcript into every pipeline run.
History selection must be bounded and auditable.

## Long Session Behavior

One session should be able to continue for a reasonable number of turns without
breaking or becoming slow because of transcript growth.

The design should separate saved history from active context:

```text
full audit history: saved forever
active context: bounded
older context: summarized
durable memory: structured
```

For each new turn, the pipeline should receive only the useful session context:

- last N turns
- relevant active memories
- active bindings
- unresolved items
- compact summaries
- carryover evidence, trace, and frame refs

The complete transcript remains available in audit files, but the next pipeline
run should not receive unlimited raw history.

## Domain-Neutral Memory

Memory should be flexible enough for normal chat, math, code, grid reasoning,
planning, corrections, and future modules without adding special cases to chat
history.

Use neutral memory records such as:

```json
{
  "memory_id": "m_001",
  "kind": "fact",
  "content": {
    "subject": "user_provided_reference",
    "predicate": "has_value",
    "object": "blue"
  },
  "source_turn_index": 1,
  "scope": "session",
  "confidence": 0.92,
  "status": "active",
  "refs": [],
  "metadata": {}
}
```

The schema is generic. The pipeline can later interpret the content as a colour,
equation, variable, grid object, tool result, user preference, or anything else.

Recommended generic memory kinds:

- `entity`
- `fact`
- `constraint`
- `preference`
- `correction`
- `task_state`
- `tool_result`
- `summary`
- `unclear_item`

These are broad storage roles, not domain classifiers.

## Binding And Rebinding

Binding connects a phrase or reference in the current turn to a session-local
target.

The target can be any memory, evidence item, trace, frame, grid object, equation,
tool result, or pipeline-produced reference.

Generic binding shape:

```json
{
  "binding_id": "b_001",
  "surface": "that value",
  "target_ref": "memory:m_004",
  "target_type": "memory",
  "source_turn_index": 3,
  "status": "active",
  "confidence": 0.9,
  "metadata": {}
}
```

Examples of what this generic binding can support:

```text
"the colour" -> prior session value
"that number" -> prior numeric value
"this grid" -> prior grid object
"that equation" -> prior equation
"my preference" -> prior user preference
```

Those are examples only. Chat history should not hardcode them.

Rebinding updates an existing reference when the user changes or corrects it.

```text
user: actually use the other value
```

The current active binding should change, while audit keeps the previous target.

```json
{
  "event_type": "binding_rebound",
  "binding_id": "b_001",
  "old_target_ref": "memory:m_004",
  "new_target_ref": "memory:m_009",
  "source_turn_index": 6,
  "reason": "user_correction"
}
```

The latest active binding is used for future turns, but the old path remains
auditable.

## Pipeline-Owned Interpretation

The chat-history layer should not decide the domain of a question. It should pass
session state into the pipeline and let the normal stages interpret it.

Expected flow:

```text
current user turn
-> load current session state
-> select bounded neutral session context
-> create chat episode
-> pass SessionState into OrchestratorRunner
-> perception extracts evidence
-> cue encoder creates retrieval cues
-> DMF retrieves relevant traces and memories
-> binding links current references to prior session items
-> context-op builds task context
-> interference handles conflicts
-> basins stabilize candidate meaning
-> lucidity decides commit/hold/recheck/search
-> decoder replies
-> accepted session changes are saved
```

This keeps chat history pipeline-native and future-proof.

## Memory Updates

Memory updates should be event-sourced. Do not silently overwrite state without a
record.

Example event:

```json
{
  "event_type": "memory_upserted",
  "turn_index": 4,
  "run_id": "20260612T000000Z_session_turn_0004_abc123",
  "memory_id": "m_002",
  "operation": "rebind",
  "old_target": "memory:m_001",
  "new_target": "memory:m_002",
  "reason": "user_correction",
  "lucidity_decision": "commit"
}
```

Memory updates should come from pipeline outputs when possible:

- committed lucidity decisions
- decoder render packets
- cue cloud evidence
- binding output
- context-op frame links
- explicit user correction turns
- validator or tool results when available

Keyword matching can exist only as a small conservative smoke fallback. It should
not be the main memory system.

## Pipeline Alignment

Chat history must remain a layer around the pipeline, not a replacement for it.

The chat runtime should:

1. load the current session
2. build a bounded session context
3. create a chat episode with `task_intent: chat`
4. pass `SessionState` into `OrchestratorRunner`
5. run the normal pipeline
6. read committed outputs
7. update only the current session memory
8. write chat audit files

The orchestrator should still decide through lucidity before memory becomes
durable. If the turn is unclear, the memory update should be deferred or marked
unresolved instead of silently becoming a fact.

## SessionState

`SessionState` is the bridge between chat history and the pipeline.

It should carry:

- `session_id`
- turn records from the current session
- active session memories
- active session bindings
- carryover evidence refs
- carryover trace ids
- carryover frame ids
- unresolved items
- compact summaries

Pipeline stages should receive this through normal run context. They should not
read arbitrary chat files directly.

## Audit Requirements

Every chat turn must leave a clear audit trail.

`session.json` should contain the session state:

- session id
- turn list
- summary
- active memory
- active bindings
- unresolved items
- links to pipeline run audits

`memory.json` should contain current session memory:

- active memory records
- active bindings
- summaries
- unresolved items
- schema version

`transcript.txt` should be readable by a person:

- turn number
- user text
- assistant text
- linked run audit directory

`history.jsonl` should be append-only:

- one event per turn
- memory change events
- binding and rebinding events
- unclear item events
- summary refresh events

Each memory change should answer:

- what changed
- which turn caused it
- which pipeline run caused it
- why it was accepted
- whether it replaced an older value
- whether the change is active, superseded, deferred, or rejected

## CLI Shape

The universal Lucid CLI should remain the manual smoke path.

Expected commands:

```text
lucid chat start --session-id session-a
lucid chat send "remember this value for this session" --session-id session-a
lucid chat send "what value did I give you?" --session-id session-a
lucid chat start --session-id session-b
lucid chat send "what value did I give you?" --session-id session-b
lucid chat list
```

Useful future inspection commands:

```text
lucid chat inspect --session-id session-a
lucid chat memory --session-id session-a
```

These commands should read the current session audit only.

## Correctness Tests

The chat-history implementation should be tested end to end:

- session A remembers a user-provided item
- session A can answer a later reference to that item
- session B does not know session A's item
- session A can rebind a previous reference
- session A uses the latest active binding after rebinding
- unresolved turns do not become durable facts
- long sessions do not inject unlimited raw history
- old turns remain auditable after summarization
- important early memory remains retrievable later in the same session
- `history.jsonl` has one turn event per turn
- `session.json` links each turn to a pipeline run audit
- `memory.json` belongs to one session only
- invalid session ids cannot escape the audit directory

## Non-Goals

This layer should not:

- create a separate chatbot outside the Lucid pipeline
- use one global memory for all sessions
- inject unlimited transcript history into every turn
- define domain-specific chat-history logic for aptitude, JEE, grid, code, or any other domain
- turn keyword guesses into permanent facts without lucidity
- let old sessions affect new sessions by default
- hide memory changes from audit logs

## Desired End State

Chat history should feel natural turn by turn while staying auditable,
domain-neutral, and pipeline-native.

The system should remember within a session, forget across new sessions, bind
and rebind references as the conversation changes, keep long sessions bounded,
and trace every memory update back to a specific turn and pipeline run.
