"""Diagnostic script for weak definition queries."""
from __future__ import annotations

import json
from pathlib import Path

from lucid.chat import run_chat_turn, start_session
from lucid.training.source_context import parse_concept_query, resolve_concept_topic

CKPT = Path("checkpoints/saves/v.0.3-ai-ml")
concepts = json.loads((CKPT / "concept_bank.json").read_text(encoding="utf-8"))["concepts"]
by_id = {c["concept_id"]: c for c in concepts}

QUERIES = [
    "what is a transformer",
    "what is machine learning",
    "what is AI",
    "how does a transformer work",
]


def show_concept(cid: str) -> None:
    c = by_id.get(cid)
    if not c:
        print(f"  concept {cid}: MISSING")
        return
    rels = c.get("relations", [])
    type_ofs = [r for r in rels if r.get("relation") == "type_of"]
    uses = [r for r in rels if r.get("relation") == "uses"]
    print(f"  concept {cid}: {len(rels)} rels ({len(type_ofs)} type_of, {len(uses)} uses)")
    for label, rows in [("type_of", type_ofs[:8]), ("uses", uses[:4])]:
        for r in rows:
            print(f"    [{label}] {(r.get('target') or '')[:130]}")


for q in QUERIES:
    parsed = parse_concept_query(q)
    topic_surface, concept_id, frame_type = parsed if parsed else (None, None, None)
    resolved = resolve_concept_topic(topic_surface) if topic_surface else None
    print("===", q)
    print("  parsed:", parsed)
    print("  resolved:", resolved)
    for cid in filter(
        None,
        [concept_id, resolved, topic_surface, "transformer_architecture", "machine_learning", "artificial_intelligence"],
    ):
        show_concept(cid)
    print()

print("=== LIVE TURNS ===")
sid = start_session()
for q in ["what is a transformer", "what is machine learning", "what is AI"]:
    r = run_chat_turn(
        q,
        session_id=sid,
        checkpoint=str(CKPT),
        perception_backend="rule",
        audit=True,
    )
    print("Q:", q)
    print("A:", r.assistant_output)
    print("audit:", r.audit_path)
    if not r.audit_path:
        print()
        continue
    root = Path(r.audit_path)
    for name in ("binding.json", "lucidity.json", "commit.json" if (root / "commit.json").exists() else "decoder.json"):
        p = root / name
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if name == "binding.json":
            for g in d.get("local_graphs") or []:
                print(f"  binding seed={g.get('seed_concept_id')} edges={len(g.get('edges') or [])}")
                for e in (g.get("edges") or [])[:6]:
                    print(f"    {e.get('relation')} -> {(e.get('target') or '')[:100]}")
        if name == "lucidity.json":
            print("  lucidity decision:", d.get("decision"), "reason:", d.get("reason_code"))
            ru = d.get("render_units") or d.get("committed_render_units") or []
            for u in ru[:5]:
                if isinstance(u, dict):
                    print(f"    unit: {u.get('text') or u.get('content') or u}")
                else:
                    print(f"    unit: {u}")
        if name == "decoder.json":
            print("  decoder output:", (d.get("text") or d.get("rendered_text") or "")[:200])
    print()
