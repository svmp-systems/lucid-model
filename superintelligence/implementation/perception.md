Short-term perception: **LLM default** (`lucid/perception/`).

- **Input (prompt):** raw payload + short task instructions only.
- **Output (API):** `response_format: json_schema` with `PerceptualEvidenceGraph` schema; falls back to `json_object` if unsupported.
- **After response:** `normalize_graph_dict` + semantic checks in `parse.py`.

Offline: `LUCID_PERCEPTION_BACKEND=rule`.
