# lucid-model

lucid models is an attempt at superintelligence using hopfield networks and EBMs in addition to some novel innovations such as lazy collapse and interference/contextop/binding. 

ideally this competes with frontier models by quantizing memory costs heavily, quantizing training costs heavily due to being fully inspectable, and is better at reasoning due to lazy collapse architecture.

besides that, we believe there exists an open path to continual learning and self research from a system like this.

## Perception (model = evidence only)

The perception stage emits a `PerceptualEvidenceGraph` — spans, markers, hints, uncertainty — never meaning or answers.

- **Offline (default):** `rule` backend — no API key.
- **LLM:** set `LUCID_PERCEPTION_BACKEND=llm` and `OPENAI_API_KEY` (or `LUCID_PERCEPTION_API_KEY`). Optional: `LUCID_PERCEPTION_BASE_URL` for Ollama/local OpenAI-compatible servers.

```bash
lucid-perceive "Alex found money and put it in the bank."
lucid-run episode.json --perception llm
```

