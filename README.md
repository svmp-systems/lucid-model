# lucid-model

lucid models is an attempt at superintelligence using hopfield networks and EBMs in addition to some novel innovations such as lazy collapse and interference/contextop/binding.

ideally this competes with frontier models by quantizing memory costs heavily, quantizing training costs heavily due to being fully inspectable, and is better at reasoning due to lazy collapse architecture.

besides that, we believe there exists an open path to continual learning and self research from a system like this.

## Perception (LLM)

Put your key in `.env` at the repo root (loaded automatically):

```text
OPENAI_API_KEY=sk-...
```

Or set `$env:OPENAI_API_KEY` in PowerShell. Then:

```powershell
py -m lucid.cli perceive "go to the bank."
```
