# Build plan

From pipeline proof on your laptop → local chatbot → local web POC → benchmark gate → big training → **public release** targeting SOTA on **Arc-AGI 2**, **Arc-AGI 3**, and **BabyLM**.

Each phase ends with something you can **run and judge**, not just more specs.

**Deep specs:** `scoped/` (especially `generator.md`, `training/training orchestrator.md`, `perception layer.md`, `lucidity.md`).

---

## What stays the same every phase

### Thinking pipeline

```
perception → cue → DMF → binding → context → interference → basins → lucidity → [projector] → decoder
```

- Only **lucidity** commits; **decoder** only expresses approved state.
- Every stage logs auditable JSON.
- Memory (traces, basins, links) stays editable.

### Three supporting systems

| System | Role |
|--------|------|
| **Generator** (`lucid-gen`) | Synthetic episodes + gold labels |
| **Orchestrator** | Run → validate → blame → patch → shadow → promote |
| **Eval** (Phase 3+) | Official benchmark harnesses |

### Training modes (when to use which)

| Mode | When |
|------|------|
| **A — Calibrate** | Right after you build a stage; compare to generator gold |
| **B — Stack** | Real upstream + module under test + stub/gold downstream |
| **C — Full loop** | End-to-end run + validator + orchestrator patches |

Don't run full **C** until the pipeline can score an episode end-to-end.

---

## Phase map (your milestones)

| Phase | Where | What you get |
|-------|--------|----------------|
| **1** | Laptop | Pipeline + generator + orchestrator MVP on fixtures |
| **2** | Laptop | **Local chatbot** — multi-turn CLI, basic conversation |
| **2.1** | Laptop (localhost) | **Chat website** — POC UI, not public product |
| **3** | Laptop + spot GPU eval | Deeper model + **verified capabilities** + promising benchmark runs → **go/no-go for Phase 4** |
| **4** | Cloud ($500–2k/mo) | Big memory + real training campaigns |
| **5** | Cloud (~10k GPU-h) | **Public release** + SOTA attempt on all three benchmarks |

```text
Phase 1   prove architecture
Phase 2   talk to it (CLI)
Phase 2.1 talk to it (browser, local)
Phase 3   prove it's worth scaling (benchmarks + capability report)
Phase 4   scale memory + training
Phase 5   ship publicly + SOTA
```

---


---

# Phase 1 — Pipeline proof (laptop)

**Machine:** RTX 3050 Ti, 16 GB — enough  
### Goal

Prove the architecture on small text + micro grids: audit, edit, generator, minimal orchestrator. **No chat product yet** — single-shot `lucid run` is enough.

### Success → Phase 2

- All stages run; 100 fixture runs without crash.
- `lucid-gen pack phase1` → 420 episodes, validation passes.
- Orchestrator on 50+ fixtures: log → validate → patch → shadow → promote/reject.
- Lucidity ≥6 checks; `COMMIT` produces valid `CommittedState`.
- CLI: `run`, `inspect`, `edit` + audit diffs.
- ≥3 grid fixtures end-to-end; grid pass &lt;2s on GPU.
- CI green.

### Golden

- Bank/kayaking: `PRESERVE_AMBIGUITY` or plural output, not wrong forced commit.
- ≥1 grid uses projection before commit.
- Failure-replay clear rules pass tests (patch + episode shadow + 3-streak).

### Commits (46)

| ID           | What                                          |
| ------------ | --------------------------------------------- |
| p1-01        | Python project, pytest, lint                  |
| p1-02        | Docs: link scoped specs, ARCHITECTURE.md      |
| p1-03        | Core enums + AuditEnvelope                    |
| p1-04        | IR: perception, cues, frames                  |
| p1-05        | IR: context, basins, lucidity, CommittedState |
| p1-06        | Audit logger + pretty-print                   |
| p1-07        | Pipeline runner                               |
| p1-08        | Trace store + edit API                        |
| p1-09        | Basin store + edit API                        |
| p1-10        | Text perception                               |
| p1-11        | Grid perception                               |
| p1-12–p1-21  | Cue → decoder stages                          |
| p1-22–p1-24  | CLI run, inspect, edit                        |
| p1-25a–h     | Generator + phase1 fixtures                   |
| p1-26–p1-26e | Orchestrator MVP                              |
| p1-27        | Seed memory from fixture gold                 |
| p1-28–p1-37  | Tests, CI                                     |
| p1-38        | `v0.1.0-laptop` + PHASE1_REPORT               |

---

# Phase 2 — Local chatbot (CLI)

**Machine:** Same laptop, local only  
**Time:** ~4–8 weeks after Phase 1

### Goal

A **multi-turn chat** you can use daily: greet, short Q&A, clarify ambiguity, explain what the system is uncertain about. Still runs the **full pipeline**. Inference stays **local**.

### What Phase 2 adds (vs Phase 1)

| Piece                                | Purpose                                                                                               |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| **Session / working memory**         | `session_id`, turn history, `prior_context` into perception/cue/DMF                                   |
| **`task_intent: chat`**              | Lucidity tuned for dialogue (hold, clarify, commit per turn)                                          |
| **Chat decoder**                     | Render `CommittedState` → natural reply; small **local** surface model OK if gated by lucidity policy |
| **Conversation generator templates** | Greetings, follow-ups, clarification turns, simple factual                                            |
| **`lucid chat` REPL**                | Interactive terminal chat                                                                             |
| **Light training**                   | Mode B/C on chat episodes; patch cue + interference links first                                       |

### What Phase 2 is not

- Not public hosting (that's 2.1).
- Not ARC-scale eval (that's 3).
- Not big memory or cloud training (that's 4).

### Perception for chat

Keep **own** rule-based perception (see perception policy): spans, markers, uncertainty on vague words. Optional: spaCy locally as **proposal** adapter only — output still `PerceptualEvidenceGraph`.

### Decoder for chat (pragmatic)

1. **v0:** Template + IR-to-text from `CommittedState` (fast, fully auditable).  
2. **v1:** Small local LM (e.g. 1–3B class) as **renderer only** — input is committed IR + policy, not raw user prompt to completion.

Lucidity must set `decoder_policy` every turn; LM cannot bypass commit.

### Success → Phase 2.1

- `lucid chat` holds 20+ turn session without crash.
- Coherent handling of: hello, short question, follow-up, “what did you mean?”, ambiguous phrase (plural/hold when appropriate).
- Each turn produces full audit under `audit/{session_id}/{turn}/`.
- Latency: median turn &lt;5s on laptop (template decoder) or &lt;15s with small local LM.
- 7k smoke episodes generated; chat templates ≥30% of mix.
- You can use it for **basic stuff** for a week without fixing crashes daily.

### Golden

- Clarification loop: ambiguous input → plural/hold → user narrows → committed answer.
- Session remembers last 5 turns in evidence/cue carryover (verifiable in audit).
- 100 chat episodes in eval set; ≥70% validator pass on expected lucidity decision.

### Commits (~24)

| ID | What |
|----|------|
| p2-01 | `session/` — SessionState, turn log, carryover into RunContext |
| p2-02 | Pipeline: multi-turn driver (run turn N with prior) |
| p2-03 | Lucidity: chat thresholds + clarify / hold policies |
| p2-04 | Decoder: `express_committed` text templates from IR |
| p2-05 | Decoder: optional local LM adapter (env-flagged, gated) |
| p2-06 | Generator: chat templates (greet, follow-up, clarify, simple QA) |
| p2-07 | `lucid chat` REPL + slash commands (`/audit`, `/reset`, `/inspect`) |
| p2-08 | Chat validators + 100-episode chat eval pack |
| p2-09 | Orchestrator: chat episode scheduler + patch cue/links on failures |
| p2-10 | `lucid-gen` smoke 7k (include chat slice) |
| p2-11 | Tests: multi-turn session, ambiguity clarification path |
| p2-12 | Docs: chat architecture + local LM install (optional) |
| p2-13 | Perf: cap active traces/basins per turn for laptop |
| p2-14 | Tag `v0.2.0-chat` |
| p2-15 | PHASE2_REPORT.md |

---

# Phase 2.1 — Local chat website (POC)

**Machine:** Laptop — **localhost only** (`127.0.0.1`)  
**Time:** ~2–4 weeks after Phase 2  
**Not:** public deploy, production auth, or paid hosting

### Goal

Same chat brain as Phase 2 in a **browser**: message box, reply stream, optional “show thinking” audit panel. For **you** to test and demo locally — not for the internet.

### Success → Phase 3

- `docker compose up` or `lucid serve --local` → UI at `http://127.0.0.1:PORT`.
- Parity: same session/seed as CLI produces same audit hash for a turn.
- UI: send message, see reply, expand stage timeline, download audit JSON for turn.
- No listen on `0.0.0.0` by default; README warns POC-only.
- Works offline (no external API required for core path).

### Golden

- Edit trace/basin in UI → rerun turn → visible diff.
- Screen recording: 3-minute ambiguous chat → clarify → commit.

### Commits (~14)

| ID | What |
|----|------|
| p2.1-01 | FastAPI: `/chat/turn`, `/session`, `/audit/{turn}` |
| p2.1-02 | Static or Vite chat UI (messages + input) |
| p2.1-03 | Audit drawer (stage timeline, pretty-print) |
| p2.1-04 | Bind localhost only; CORS for local dev |
| p2.1-05 | docker-compose (api + ui, CPU) |
| p2.1-06 | CLI/API parity tests |
| p2.1-07 | Optional: toggle “verbose pipeline” in UI |
| p2.1-08 | Docs: local POC runbook |
| p2.1-09 | Tag `v0.2.1-chat-poc` |
| p2.1-10 | PHASE2.1_REPORT.md |

---

# Phase 3 — Capability proof + benchmark gate

**Machine:** Laptop dev; **spot GPU** for eval batches  
**Time:** ~10–16 weeks  
**Output:** **PHASE3_GO_NO_GO.md** — explicit approval before Phase 4 money

### Goal

A **more built-out** system than chat POC: stronger grid path, real benchmark plumbing, measured capabilities, and **promising** (not necessarily SOTA) official or private benchmark runs. This phase answers: *is scaling spend justified?*

### What you build

- Perception/grid upgrades for ARC-style tasks.
- Projector v2; lucidity grid checks.
- Eval harness: Arc-AGI 2, Arc-AGI 3, BabyLM ingest + eval stubs.
- Generator scale: **~150k** episodes; orchestrator task-family training.
- **Capability matrix** (see below) — signed checklist, not vibes.

### Capability matrix (verify before Phase 4)

| Capability | How verified |
|------------|----------------|
| Lazy collapse | Audit: plural basins until lucidity; no early commit in logs |
| Frame-scoped competition | Bank/kayaking + multi-frame commits in report |
| Assembly (grid) | Move+recolor fixtures + projection path |
| Chat + grid same pipeline | Same IR, different `modality` |
| Learning from failure | Replay queue stable; promoted patches improve canary |
| Reproducibility | Same seed → same audit on 50 fixed episodes |
| ARC plumbing | Submission artifacts build without manual hacks |
| BabyLM path | Official-format eval runs (score logged) |

### Benchmark targets (success = go to Phase 4)

| Benchmark | Success (go) | Golden |
|-----------|--------------|--------|
| **Arc-AGI 2** | ≥1 official submit; private holdout **≥20%** OR clear gain vs Phase 2 baseline | ≥35% holdout or top-50% early leaderboard slice |
| **Arc-AGI 3** | ≥1 official submit; private holdout **≥15%** | ≥25% holdout |
| **BabyLM** | Official eval run complete; within **2.5×** chosen baseline | Within **1.5×** baseline |
| **Ops** | ≥70% ARC tasks return full audit (no silent crash) | Median GPU-s/task within budget in report |

**No-go:** If Arc holdout &lt;10% after planned training, or replay queue grows unbounded, or capability matrix &lt;6/8 — **stay in Phase 3**, do not open Phase 4 spend.

### Success → Phase 4

- `PHASE3_GO_NO_GO.md` = **GO** with dated scores and cost log.
- Capability matrix 8/8 or documented exceptions.
- Reproducible eval scripts + submission bundles archived.
- 150k corpus validated; orchestrator stable on failure-only updates.

### Commits (~32)

| ID | What |
|----|------|
| p3-01 | Arc-AGI 2 loader + schema |
| p3-02 | Arc-AGI 3 loader + adapters |
| p3-03 | Grid perception v2 |
| p3-04–08 | Grid cue → binding → projector → lucidity → submission export |
| p3-09 | Generator: harder templates + 150k build |
| p3-10 | Orchestrator: corpus mix + retention ≤200 |
| p3-11 | Private holdout splits + metrics dashboard |
| p3-12 | Capability test suite + report generator |
| p3-13 | Cloud eval runner (spot, resumable) |
| p3-14 | BabyLM ingest + eval |
| p3-15 | Submit Arc-AGI 2 run 1 |
| p3-16 | Submit Arc-AGI 3 run 1 |
| p3-17 | BabyLM eval run 1 |
| p3-18 | Failure clustering → template priorities |
| p3-19 | `PHASE3_GO_NO_GO.md` |
| p3-20 | Tag `v0.3.0-gate` |
| p3-21–32 | Tests, fixes, optional paraphrase layer, alignment tests (buffer) |

---

# Phase 4 — Big memory + real training

**Only if Phase 3 = GO.**  
**Budget:** ~$500–2k/month + multi-GPU windows  
**Time:** ~12–20 weeks

### Goal

Scale what worked in Phase 3: **100k–10M traces**, quantized tiers, full training governor, **1.5M+** episodes, serious train/eval campaigns. Target **strong** benchmark scores — still pre-release.

### Success → Phase 5

- Hot/warm/cold memory operational.
- Governor skips ≥50% updates on high-margin wins.
- Shadow bundles ≤80 episodes always.
- Arc 2 **≥45%** (official or private); Arc 3 **≥30%**; BabyLM within **1.5×** baseline.
- 10k-step train without memory corruption; audit replay holds.
- Checkpoint + provenance for Phase 5 campaigns.

### Golden

- Arc 55% / 40%; BabyLM **1.2×** baseline; 3× $/performance vs Phase 3.

### Commits (~36)

Memory tiers p4-01–03; 1.5M generator p4-04–05; full orchestrator + governor p4-06–14; scale indexes + encoders p4-15–17; distributed train p4-18–23; big eval submissions p4-24–33; report p4-36–38. (Same substance as prior plan; IDs p4-xx preserved where possible.)

---

# Phase 5 — Public release + SOTA

**Budget:** ~10k GPU-hours total across benchmark campaigns (you set cap)  
**Time:** ~4–8+ months

### Goal

**Publicly releasable** model artifacts: weights/checkpoints (as you choose), docs, license, eval reproduction package. **Best effort SOTA** on Arc-AGI 2, Arc-AGI 3, BabyLM within budget.

### Public release checklist

| Item | Required |
|------|----------|
| Model card | Architecture summary, limits, training data overview |
| Repro script | Seed + checkpoint → benchmark submission files |
| License + weights hosting | Clear terms |
| Safety / misuse section | What it is not |
| Ablations | Lucidity off, global-update off — show deltas |
| No secrets in repo | Keys, private checkpoints gitignored |

### Success (ship)

| Benchmark | Target (set in PHASE5_REPORT at phase start) |
|-----------|-----------------------------------------------|
| Arc-AGI 2 | Competitive: e.g. ≥70% or top-10 per rules at launch |
| Arc-AGI 3 | Competitive: e.g. ≥50% or top-10 |
| BabyLM | Within **1.05×** named SOTA reference |

Plus: one-command repro, full cost/score table, operator manual.

### Golden

- **SOTA or tied #1** on each benchmark in target window.
- 2× score per GPU-hour vs Phase 4.

### Commits (~28)

Campaigns p5-08–12; unified submit pipeline p5-13–14; ablations p5-15–18; final submits p5-19–21; release packaging p5-22–27; `v1.0.0` p5-27; PHASE5_REPORT p5-28.

---

## Cross-phase rules

1. **Same pipeline shape** — phases add product surfaces (chat, web) and scale, not new cognitive religion per phase.
2. **Phase 2 chat** must not skip lucidity; no raw prompt-to-LM product path.
3. **Phase 2.1** is localhost POC — not a substitute for Phase 3 hard eval.
4. **Phase 4 requires Phase 3 GO doc** — no exceptions for enthusiasm.
5. **Generator before heavy orchestrator** in Phase 1; chat templates in Phase 2.
6. **Perception:** own graph; external models = proposal adapters only (`scoped/perception layer.md`).
7. **Training:** calibrate each new stage (A/B) before full loop (C).
8. **Replay clear:** patch promoted + episode shadow pass + 3 consecutive successes.

---

## Training & data by phase

| Phase | Episodes (order of magnitude) | Training focus |
|-------|------------------------------|----------------|
| 1 | 420 | Orchestrator MVP; link patches |
| 2 | 7k (+ chat pack) | Chat turns; cue + interference |
| 2.1 | — | Inference only (no new training required) |
| 3 | 150k | ARC failures; capability canaries |
| 4 | 1.5M | Governor + big memory |
| 5 | 7M+ | Campaign HPO; compaction |

---

## If you get stuck

| Situation | Action |
|-----------|--------|
| Phase 1 OK but chat bad | Fix decoder + session carryover before 2.1 web |
| Chat OK but ARC hopeless | Expected early; Phase 3 is the gate — don't skip to Phase 4 |
| Phase 3 no-go | More generator templates + failure mining; no cloud scale |
| Local LM bypasses lucidity | Bug — decoder must read `decoder_policy` only |
| Want public demo before Phase 5 | OK for **2.1 localhost demo**; not production hosting |

---

## Commit totals (approximate)

| Phase | Commits |
|-------|---------|
| 1 Pipeline | 46 |
| 2 Chat CLI | 15 |
| 2.1 Local web | 10 |
| 3 Gate | 32 |
| 4 Big train | 36 |
| 5 Release | 28 |
| **Total** | **~167** |

---

## One-line north star

**Phase 1** prove the machine · **Phase 2** talk to it · **Phase 2.1** click to talk · **Phase 3** prove it's worth millions of cycles · **Phase 4** spend those cycles · **Phase 5** ship and win the benchmarks.
