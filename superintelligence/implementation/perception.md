a short term implementation of perception using LLMs for fast prototyping. Ideally we finish training our own perception layer by the end.

works by sending raw package to LLM, asking it to parse, and then return back to us in a schema

gives output in these three forms
- candidate units  - main individual traces
- candidate regions - splits into 2 regions (so primitive traces and relational traces)
- candidate containers - for grid parsing, marks things in boxes and borders and stuff.
- candidate markers - supporting words / relational traces
