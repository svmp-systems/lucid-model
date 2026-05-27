# lucid-model


lucid models is an attempt at superintelligence using hopfield networks and EBMs in addition to some novel innovations such as lazy collapse and interference/contextop/binding. 

ideally this competes with frontier models by quantizing memory costs heavily, quantizing training costs heavily due to being fully inspectable, and is better at reasoning due to lazy collapse architecture.

besides that, we believe there exists an open path to continual learning and self research from a system like this.

## IR package (`lucid/ir`)

Seven layers of typed, JSON-serializable contracts:

| Layer | Module | Contents |
|-------|--------|----------|
| 1 | `common.py` | Enums, `Provenance`, `ComputePolicy`, `AuditEnvelope` |
| 2 | `perception.py` | `PerceptionInput` → `PerceptualEvidenceGraph` |
| 3 | `cue.py`, `dmf.py` | Cue cloud, tracebank field |
| 4 | `binding.py`, `context_op.py`, `interference.py`, `basins.py` | Frames, scope, competition, basins |
| 5 | `lucidity.py`, `projector.py`, `expression.py` | Commit gate, optional projector, decoder |
| 6 | `training.py`, `memory.py` | Episodes, run logs, trace/basin records |
| 7 | `pipeline.py` | `RunContext`, `SessionState`, `PipelineRun` |

```bash
python -m pytest tests/
```