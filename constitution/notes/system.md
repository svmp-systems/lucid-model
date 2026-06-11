# What lucid-model is

Inferred from the repository layout, IR types, pipeline runner, and CLI. This describes what the code implements today, not a product roadmap.

## Purpose

**lucid-model** (package `lucid-model`, import `lucid`) is a Python cognitive pipeline that turns raw input (mostly text or small grids) into structured intermediate representations, decides whether to commit to an interpretation, and renders a user-facing answer. The design separates **reasoning/commit** from **expression**: only the lucidity stage commits; the decoder only renders what lucidity approves.

Version in `pyproject.toml`: **0.1.0**. Target Python: **≥3.11**.

## Core architecture

### Thinking pipeline

Stages are defined in `lucid/ir/pipeline.py` as `StageName` and executed sequentially by `lucid/cognition/pipe_orchestrator/runner.py` (`OrchestratorRunner`):

```
perception → cue_encoder → dmf → binding → context_op → interference → basins → lucidity → [projector] → decoder
```

Each stage reads typed inputs and writes typed outputs under `lucid/ir/`. Stage implementations live under `lucid/cognition/`:

| Stage | Role (from code behavior) |
|-------|---------------------------|
| **Perception** | Raw text/grid → `PerceptualEvidenceGraph` (candidate units, regions, markers, hints). Backends: rule-based tokenizer (`backend=rule`) or LLM with JSON schema (`backend=llm`, OpenAI-compatible API). |
| **Cue encoder** | Evidence graph → `CueCloud` (primitive and relational activations tied to traces). |
| **DMF** | Dynamic memory field: activates traces from cue cloud against a **tracebank** checkpoint store; produces `DmfOutput` with active traces, conflicts, clusters. |
| **Binding** | Builds plural **candidate frames** (role assignments, evidence refs, unresolved slots) from perception + DMF. |
| **Context-op** | Scopes frames into **context frames**, scoped trace assignments, gates. |
| **Interference** | Trace–trace, trace–frame, frame–basin edges; scoped energy deltas; conflict reports. Can load learned links from checkpoint. |
| **Basins** | Competing **basin states** and optional assemblies; `CompetitionSummary` (top basin, margin). |
| **Lucidity** | Runs named checks (margin, coverage, coherence, binding stability, scope, projection fit, contradiction, maturity, risk). Emits a **decision** (`commit`, `preserve_ambiguity`, `request_projection`, `search_wider`, `recheck_binding`) and a **render packet** (`LucidityRenderPacket`) for the decoder. |
| **Projector** | Optional grid rollout / implied artifact when task is grid-like or lucidity requests projection. |
| **Decoder** | Semantic graph → discourse plan → realization program → surface text (or grid), with **faithfulness** checks and optional chat polish. Falls back to simpler text render if strict checks fail. |

### Commit vs express

- **Lucidity** (`lucid/cognition/output/lucidity/`) builds `CommittedState` with `RenderUnit` payloads (claims, frame summaries, artifacts, actions) when decision is `commit`.
- **Decoder** (`lucid/cognition/output/decoder/`) must not invent facts beyond the render packet; `check_faithfulness` and `check_structural_faithfulness` enforce that.
- Decoder modes (`DecoderMode` in `lucid/ir/common.py`) include express committed, plural, uncertainty, refusal, and hold.

### Intermediate representation (IR)

Shared datatypes under `lucid/ir/`:

- **Perception**: `CandidateUnit`, regions, markers, arrangement/change/grouping hints.
- **Cue / DMF**: cue activations, `ActiveTrace`, conflict/novelty signals.
- **Binding**: `CandidateFrame`, competition edges.
- **Context**: `ContextFrame`, scoped assignments.
- **Interference / basins**: edges, basin records, assemblies.
- **Lucidity**: `LucidityRenderPacket`, `RenderUnit`, `DecoderPolicy`, `FaithfulnessContract`.
- **Expression**: `DecoderOutput`, `FaithfulnessReport`, sentence refs with `SourceRef` provenance.
- **Training**: `Episode`, `RunLog`, validator hooks.

Serialization via `lucid/ir/serde.py` (`to_json`, `from_json`, `to_dict`).

## Memory and checkpoints

Train-time artifacts live under **`lucid/training/tree/`** (resolved by `lucid/paths.py`; overridable with `LUCID_TRAIN_ROOT`).

A **checkpoint** is a directory with `manifest.json` and JSON stores (`lucid/training/checkpoints.py`):

| Store file | Contents |
|------------|----------|
| `tracebank.json` | DMF trace records |
| `basin_bank.json` | Basin memory |
| `cue_encoder_map.json` | Cue targets / feature index |
| `binding_affordances.json` | Binding patterns |
| `interference_graph.json` | Gates and edges |
| `context_policy.json` | Scope patterns |
| `lucidity_policy.json` | Decision thresholds / templates |
| `decoder_adapter.json` | Render targets, correction pairs |
| `perception_examples.json`, `projector_examples.json` | Training examples |

**Checkpoint slots** (`lucid/training/checkpoint_slots.py`):

- `checkpoints/training/` — written by `lucid train`
- `checkpoints/loaded/` — inference save point; `lucid ask` uses this when a manifest exists
- `checkpoints/saves/<name>/` — named archives
- Pointer file: `checkpoints/loaded.json`

Memory is **editable**: `lucid edit` and `lucid/cognition/memory/edit.py` patch tracebank/basin records with audit trails.

## Auditing

`lucid/audit/logger.py` (`AuditLogger`) writes per-run folders: `manifest.json`, one JSON file per stage, human-readable summaries. Pipeline runs land in `audit/runs/pipeline/<timestamp>_<episode>/`. Module-specific audits exist (DMF, binding, lucidity, decoder, etc.).

`lucid-inspect` and `lucid audit list` drill into runs. `lucid ask` writes `report.txt` (sentence / answer / compact audit) and optionally `audit/runs/ask/latest.txt`.

## Training and data generation

Three supporting systems referenced in `constitution/plan/build.md` and partially implemented:

1. **Generator** (`lucid-gen`, `lucid/training/generator/`) — synthetic episodes with gold labels; recipes include bank/kayaking ambiguity, two-event text, grid move/recolor, scoped instructions. Ambiguity controlled by a knob (`AmbiguityKnob`, clarity bands).

2. **Module trainers** (`lucid/training/treeers/`) — per-stage training for perception, cue_encoder, dmf, binding, context_op, interference, basins, lucidity, projector, decoder.

3. **Orchestrator** (`lucid/training/orchestrator/orchestrator.py`) — MVP loop: sample episode → run pipeline → validate → blame → patch → shadow → promote/reject → failure replay. `lucid train loop` wires this to the real pipeline runner.

Training CLI (`lucid train`): per-module commands, `global`, `loop`, `validate` (gold L3 scoring), `golden` (named fixtures).

## CLI surface

Entry points (`pyproject.toml`):

- **`lucid`** — main runtime: `ask`, `run`, `run-batch`, `perceive`, stage smokes (`bind`, `dmf`, `basins`, `lucidity`, `decoder`, …), `train`, `checkpoint`, `edit`, `audit`, `scaling`, `gen`
- **`lucid-inspect`** — audit inspection
- **`lucid-gen`** — generator packs

`lucid ask "…"` (or bare `lucid "…"`) runs the full pipeline on one sentence and prints sentence, answer, and compact audit.

## Modality and tasks

`Modality`: text, grid, image, audio, interactive (enum in IR; implementation depth varies).

`TaskIntent`: chat, answer, solve_grid, act, observe, retrieve. Lucidity and decoder policies branch on task (e.g. grid output format, projection checks).

## Design constraints visible in code

- **Fail-closed expression**: decoder refuses or holds when render packet is missing; faithfulness failure can trigger literal/text fallback rather than silent invention.
- **Plural hypotheses**: basins and lucidity can preserve alternatives instead of forcing a single reading (`preserve_ambiguity`, plural decoder mode).
- **Auditable stages**: every pipeline step is timed, success-flagged, and persisted.
- **Phase 1 scope** (`build.md`): pipeline proof on laptop fixtures; not yet a full multi-turn chat product (session types exist in IR but Phase 2).

## Repository layout (runtime code)

```
lucid/
  ir/                 # shared datatypes
  cognition/
    input/            # perception, cue encoder
    reasoning/        # dmf hookup, binding, context_op, interference, basins
    output/           # lucidity, projector, decoder
    memory/           # DMF runtime, edit API
    pipe_orchestrator/  # pipeline runner, stub stages, checkpoint runtime
  training/           # trainers, generator, orchestrator, checkpoints, validation
  audit/              # loggers, ask report, stage-specific audit writers
  cli.py              # universal CLI
  paths.py            # train tree resolution
constitution/plan/    # build phases and exit criteria
superintelligence/    # scoped design specs (reference, not runtime)
```

Tests live in `lucid/training/tests/` (212 tests at last count).
