# What has been accomplished

Inferred from implemented modules, CLI commands, tests, and audit artifacts. Status is **Phase 1–oriented** unless noted.

## Pipeline end-to-end

- **Full stage chain** runs without manual wiring: perception through decoder, orchestrated by `OrchestratorRunner` with per-stage JSON audit and manifest.
- **Episode-driven runs**: `lucid run` / `run-batch` accept `Episode` JSON; grid pairs detected from raw input when present.
- **Single-sentence UX**: `lucid ask` runs the pipeline on plain text and emits sentence, answer, compact step audit, and `report.txt` under the run directory.
- **212 pytest tests** in `lucid/training/tests/` covering IR, stages, generator, orchestrator, paths, checkpoints, decoder, and CLI smoke paths.

## Perception

- **Rule backend**: tokenizes text, skips function-word markers, emits span units with character offsets, regions (main vs subordinate clause heuristics), polysemy flags (e.g. `bank`), reference hints.
- **LLM backend**: OpenAI-compatible chat with JSON schema validation, retry loop, dedicated perception audit files under each run.
- **`infer_unit_positions`**: after perception, fills missing span offsets by scanning the source sentence so downstream binding keeps word order (especially for LLM graphs without positions).

## Reasoning stages

- **Cue encoder**: primitive/relational activations; checkpoint-backed cue map; CLI smoke and trainer.
- **DMF**: trace activation from cue cloud against tracebank; conflict/novelty signals; runtime audit events; `lucid train dmf` and edit API for traces.
- **Binding**: frame seeds from regions, verb splits, grouping hints; role slots ordered by sentence position; candidate frames with unresolved slots (e.g. `bank_sense`); trainer + affordances in checkpoint.
- **Context-op**: context frames, scoped traces, gates; tests on multi-frame bank/kayaking fixtures.
- **Interference**: scoped basin deltas, conflict reports, learnable link store; `interference learn` CLI path.
- **Basins**: candidate basin states, assemblies, competition summary; basin bank in checkpoint; edit API.

## Lucidity gate

- **Named checks** implemented and wired: margin, coverage, coherence, binding stability, scope, projection fit (when applicable), contradiction, maturity, risk.
- **Decisions** map to decoder policies (committed, plural, uncertainty, hold for projection/search/recheck).
- **`CommittedState`** with render units: primary basin claim, per-frame summaries (role-ordered evidence surfaces), optional projection artifact.
- **Checkpoint overrides** for margin/coverage thresholds via `lucidity_policy.json` and `checkpoint_runtime`.

## Decoder

- **Semantic path**: build semantic graph → discourse plan → realization ops → surface realizer.
- **Render modes**: committed, plural, uncertainty, refusal, hold; grid renderer for grid packets.
- **Faithfulness**: token/reparse checks, required-unit coverage, structural program checks; literal/text fallback when polish fails.
- **Frame summaries**: surface phrases from committed frames; composed multi-frame answers; bank-sense caveat when unresolved slots include `bank`.
- **Chat polish**: light dedupe and sentence clipping (`polish_for_chat`).

## Projector

- Grid pair handling, rollouts, implied artifacts; lucidity can request projection before commit; tests and trainer stub.

## Memory, checkpoints, paths

- **Canonical train tree**: `lucid/training/tree/` via `paths.py`; legacy `train/` prefix stripping; `LUCID_TRAIN_ROOT` override.
- **Checkpoint stores**: manifest + hashed store files; summary and load helpers.
- **Training vs loaded slots**: `lucid checkpoint status|load|save|clear|list`; inference resolves loaded save point by default; `--cold` and `--checkpoint` overrides on ask/run.
- **Editable memory**: `lucid edit` list/patch traces and basins with audit diffs.

## Training and validation

- **Per-module trainers** registered for all major stages.
- **`lucid train`**: module targets, global governor-directed training, orchestrator loop on real pipeline.
- **`lucid train validate`**: score episodes against generator gold (L3 labels).
- **`lucid train golden`**: named Phase 1 golden fixtures (bank/kayaking ambiguity, grid cases).
- **Failure replay** tests for orchestrator patch/shadow/promote rules.
- **Quantization helpers** and governor smoke for retrieval-quality experiments.

## Generator (`lucid-gen`)

- Recipe engine with ambiguity knob and clarity bands.
- Recipes: bank destination, two events, scoped instruction, grid move, grid recolor.
- Phase 1 pack concept in build plan; generator tests for gold coverage and episode roundtrip.

## Auditing and inspection

- **AuditLogger**: run manifests, stage records, timing, success flags.
- **Stage-specific audit writers** (binding, cue, basins, lucidity, decoder, …).
- **`lucid-inspect`**, `lucid audit list|checkpoints|layout`.
- **Ask reports**: human-readable sentence/answer/audit; `latest.txt` pointer for quick access.
- **Scaling observatory** CLI (`lucid scaling summary|export|path`) for cost/quality receipts.

## CLI coverage (smokeable stages)

Individual stage commands exist for: perceive, cue-encoder, bind, dmf, context-op, interference, basins, lucidity, decoder, projector, governor, quant, train, checkpoint, edit, audit, gen, scaling, run, ask.

## Verified behaviors (from tests and fixtures)

- IR JSON roundtrip for major datatypes.
- Bank/kayaking template: lucidity can commit or preserve ambiguity rather than silently forcing one reading.
- Grid fixtures run end-to-end with performance gate tests.
- Decoder tests: committed claims, caveats, artifacts, frame-summary composition, faithfulness pass paths.
- Path resolution tests prevent stray repo-root `train/` remapping bugs.
- Ask CLI and checkpoint slot tests for inference defaulting to loaded weights.

## What is not yet implemented (gaps visible in code vs build.md Phase 2+)

- **Multi-turn chat product**: `SessionState` / `TurnRecord` exist in IR; no full Phase 2 conversation loop in CLI.
- **Natural conversational prose**: decoder is faithfulness-first phrase composition, not a trained conversational LM renderer; `decoder_adapter.render_targets` store exists but full Phase 2 polish/training loop is incomplete.
- **Public release / benchmark harnesses**: Phase 3+ in build plan; eval harness not present as first-class CLI.
- **PHASE1_REPORT single command**: build.md references it; assembly is currently manual (pytest + validate + golden + checkpoint status + demo ask).
- **CI badge / v0.1.0-laptop tag**: build plan milestone; verify separately from this note.

## Recent engineering fixes (decoder / ask path)

These are implemented in the codebase and covered by tests:

- Decoder no longer dumps raw IR key-value blobs when semantic realization passes faithfulness (composed sentence cites all consumed units including skipped basin telemetry).
- Basin-primary claim optional when frame summaries carry content.
- Binding preserves sentence order via position keys and graph index instead of alphabetical unit IDs.
- Multi-frame answers join with clause boundaries rather than treating unrelated frames as one “X and Y” noun list.

## How to verify locally

```bash
py -m pytest lucid/training/tests -q
lucid train golden --checkpoint training
lucid train validate --limit 50 --require-l3 --checkpoint training
lucid checkpoint load checkpoints/local   # if weights exist
lucid ask "I found money while kayaking and placed it in the bank." --perception rule
lucid checkpoint status
```

Expected: pipeline audit under `lucid/training/tree/audit/runs/pipeline/`, readable ask report, lucidity decision and decoder surface text in decoder stage summary.
