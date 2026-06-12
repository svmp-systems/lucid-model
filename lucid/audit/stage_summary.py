"""Human-readable stage summaries for audit JSON and run reports."""

from __future__ import annotations

from typing import Any

from lucid.ir.serde import to_dict


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return to_dict(value)


def _fmt(value: float | int | None, *, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 100 or number == int(number):
        return str(int(number)) if number == int(number) else f"{number:.{digits}f}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _unit_surface_map(*sources: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for source in sources:
        data = _as_dict(source)
        graph = data.get("perceptual_evidence_graph") or data
        for unit in graph.get("candidate_units") or []:
            if not isinstance(unit, dict):
                continue
            uid = str(unit.get("unit_id") or "").strip()
            surface = str(unit.get("surface") or "").strip()
            if uid and surface:
                mapping[uid] = surface
    return mapping


def _surface(ref: str, surfaces: dict[str, str]) -> str:
    ref = str(ref or "").strip()
    if not ref:
        return "?"
    if ref in surfaces:
        return surfaces[ref]
    if ref.startswith("u_"):
        return ref[2:].replace("_", " ")
    return ref


def _section(title: str) -> str:
    return f"-- {title} --"


def _summarize_perception(data: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    units = data.get("candidate_units") or []
    flags = data.get("uncertainty_flags") or []
    regions = data.get("candidate_regions") or []
    surfaces = [str(u.get("surface") or "") for u in units if u.get("surface")]
    headline = f"{len(units)} evidence units"
    if surfaces:
        headline += f": {', '.join(surfaces[:5])}"
        if len(surfaces) > 5:
            headline += f", +{len(surfaces) - 5} more"

    lines = [
        f"candidate_units: {len(units)}",
        f"uncertainty_flags: {len(flags)}",
        f"candidate_regions: {len(regions)}",
    ]
    if surfaces:
        lines.append(f"surfaces: {', '.join(surfaces)}")

    narrative = [_section("Perception")]
    if units:
        for unit in units:
            uid = unit.get("unit_id", "")
            surface = unit.get("surface", "")
            pos = unit.get("position_or_time", "")
            pos_bit = f" @{pos}" if str(pos).strip() else ""
            narrative.append(f"  {uid}: \"{surface}\"{pos_bit}")
    else:
        narrative.append("  (no candidate units)")
    if flags:
        narrative.append("  Uncertainty:")
        for flag in flags[:6]:
            target = flag.get("target_id", "?")
            kind = flag.get("uncertainty_type", "unknown")
            narrative.append(f"    {target}: {kind}")
    if regions:
        narrative.append("  Regions:")
        for region in regions[:4]:
            members = region.get("member_unit_ids") or []
            narrative.append(
                f"    {region.get('region_id', '?')} ({region.get('role_hint', 'region')}): "
                f"{len(members)} units"
            )
    return headline, lines, narrative


def _summarize_cue_encoder(data: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    prim = data.get("primitive_trace_activations") or []
    rel = data.get("relational_trace_activations") or []
    budget = data.get("retrieval_budget_used", 0)
    headline = f"{len(prim)} cue->trace routes, {len(rel)} relational"
    lines = [
        f"primitive_activations: {len(prim)}",
        f"relational_activations: {len(rel)}",
        f"retrieval_budget_used: {budget}",
    ]

    narrative = [_section("Cue encoder")]
    ranked = sorted(prim, key=lambda item: float(item.get("weight") or 0), reverse=True)
    if ranked:
        narrative.append("  Primitive routes (trace <- evidence):")
        for item in ranked[:10]:
            trace = item.get("trace_id", "?")
            weight = _fmt(item.get("weight"))
            refs = item.get("evidence_refs") or []
            ref_text = ", ".join(str(ref) for ref in refs[:3]) or "-"
            narrative.append(f"    {trace} ({weight}) <- {ref_text}")
    if rel:
        narrative.append("  Relational routes:")
        for item in rel[:6]:
            trace = item.get("trace_id", "?")
            weight = _fmt(item.get("weight"))
            endpoints = item.get("endpoint_unit_ids") or []
            narrative.append(f"    {trace} ({weight}) between {', '.join(endpoints) or '-'}")
    hints = data.get("weak_structure_hints") or []
    if hints:
        narrative.append(f"  Structure hints: {', '.join(str(h) for h in hints[:8])}")
    return headline, lines, narrative


def _summarize_dmf(data: dict[str, Any], *, stage_input: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    active = data.get("active_traces") or []
    margin = data.get("top_margin", 0.0)
    coverage = data.get("coverage_score", 0.0)
    audit_log = data.get("audit_log") or {}
    headline = f"{len(active)} active traces, margin {_fmt(margin)}, coverage {_fmt(coverage)}"

    lines = [
        f"active_traces: {len(active)}",
        f"top_margin: {margin}",
        f"coverage_score: {coverage}",
    ]
    if audit_log:
        lines.append(
            f"sparse_recall: {audit_log.get('selected_traces', '?')}/"
            f"{audit_log.get('tracebank_size', '?')} tracebank"
        )

    narrative = [_section("DMF - dynamic memory field")]
    if audit_log:
        narrative.append(
            f"  Sparse recall: selected {audit_log.get('selected_traces', '?')} of "
            f"{audit_log.get('tracebank_size', '?')} tracebank records "
            f"(mode={audit_log.get('retrieval_mode', '?')}, limit={audit_log.get('compute_limit', '?')})"
        )
        narrative.append(
            f"  Candidates considered: {audit_log.get('candidate_traces', '?')} | "
            f"provisional in bank: {audit_log.get('provisional_traces', '?')}"
        )
    cue_meta = (
        (data.get("provenance") or {}).get("extra") or {}
    ).get("cue_encoder") or (
        (stage_input.get("cue_cloud") or {}).get("provenance") or {}
    ).get("extra", {}).get("cue_encoder", {})
    if cue_meta:
        narrative.append(
            f"  Cue routing: {cue_meta.get('primitive_candidate_count', '?')} primitive + "
            f"{cue_meta.get('relational_candidate_count', '?')} relational candidates; "
            f"exact hits={cue_meta.get('exact_route_hits', '?')}, "
            f"similar hits={cue_meta.get('similar_route_hits', '?')}"
        )

    if active:
        narrative.append("  Activated traces (ranked):")
        ranked = sorted(active, key=lambda item: float(item.get("activation") or 0), reverse=True)
        for item in ranked[:12]:
            trace = item.get("trace_id", "?")
            act = _fmt(item.get("activation"))
            cluster = item.get("cluster_id") or "-"
            tier = item.get("heat_tier") or "-"
            novel = " novel" if item.get("is_novel") else ""
            narrative.append(f"    {trace}: energy {act}, cluster={cluster}, heat={tier}{novel}")
    else:
        narrative.append("  No traces activated (empty tracebank or no cue match).")

    conflicts = data.get("conflict_signals") or []
    if conflicts:
        narrative.append("  Trace conflicts:")
        for signal in conflicts[:4]:
            narrative.append(
                f"    {signal.get('trace_id_a', '?')} vs {signal.get('trace_id_b', '?')} "
                f"({signal.get('conflict_type', 'conflict')})"
            )
    novelty = data.get("novelty_signals") or []
    if novelty:
        narrative.append("  Novelty:")
        for signal in novelty[:4]:
            narrative.append(
                f"    {signal.get('region_or_evidence_ref', '?')}: "
                f"score {_fmt(signal.get('novelty_score'))} -> {signal.get('suggested_action', '?')}"
            )
    narrative.append(
        f"  Competition margin: {_fmt(margin)} | entropy {_fmt(data.get('activation_entropy'))} | "
        f"uncertainty: {data.get('uncertainty_summary', '-')}"
    )
    return headline, lines, narrative


def _summarize_binding(data: dict[str, Any], *, stage_input: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    frames = data.get("candidate_frames") or []
    score = data.get("binding_stability_score", 0.0)
    surfaces = _unit_surface_map(stage_input)
    headline = f"{len(frames)} candidate frames (stability {_fmt(score)})"
    lines = [
        f"candidate_frames: {len(frames)}",
        f"binding_stability_score: {score}",
    ]

    narrative = [_section("Binding - candidate frames")]
    for frame in frames:
        frame_id = frame.get("frame_id", "?")
        frame_type = frame.get("frame_type", "?")
        conf = _fmt(frame.get("confidence"))
        unresolved = frame.get("unresolved_slot_names") or []
        narrative.append(f"  {frame_id} [{frame_type}, confidence {conf}]")
        roles = frame.get("role_assignments") or {}
        slot_refs = frame.get("slot_evidence_refs") or {}
        for role in sorted(roles.keys()):
            trace = roles[role]
            refs = slot_refs.get(role) or []
            evidence = ", ".join(_surface(ref, surfaces) for ref in refs) or "?"
            narrative.append(f"    {role}: \"{evidence}\" -> trace {trace}")
        supporting = frame.get("supporting_trace_ids") or []
        if supporting:
            narrative.append(f"    supporting traces: {', '.join(supporting)}")
        if unresolved:
            narrative.append(f"    unresolved: {', '.join(unresolved)}")
        if frame.get("conflicting_trace_ids"):
            narrative.append(f"    conflicts: {', '.join(frame['conflicting_trace_ids'])}")
    if not frames:
        narrative.append("  (no frames materialized)")
    return headline, lines, narrative


def _summarize_context_op(data: dict[str, Any], *, stage_input: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    ctx_frames = data.get("context_frames") or []
    scoped = data.get("scoped_trace_assignments") or []
    links = data.get("frame_links") or []
    gates = data.get("interference_gates") or []
    pressures = data.get("local_basin_pressures") or []
    headline = f"{len(ctx_frames)} context frames, {len(scoped)} scoped traces, {len(gates)} gates"
    lines = [
        f"context_frames: {len(ctx_frames)}",
        f"scoped_trace_assignments: {len(scoped)}",
        f"frame_links: {len(links)}",
        f"interference_gates: {len(gates)}",
        f"local_basin_pressures: {len(pressures)}",
    ]
    notes = data.get("audit_notes") or []
    if notes:
        lines.append(f"audit: {notes[0]}")

    narrative = [_section("Context-op - scoped interpretation")]
    if ctx_frames:
        narrative.append("  Context frames:")
        for ctx in ctx_frames:
            ctx_id = ctx.get("context_frame_id", "?")
            members = ", ".join(ctx.get("member_frame_ids") or []) or "-"
            notes_text = ctx.get("scope_notes") or ""
            heat = ctx.get("heat_policy") or "-"
            narrative.append(f"    {ctx_id}: frames [{members}] heat={heat}")
            if notes_text:
                narrative.append(f"      notes: {notes_text}")
    if scoped:
        narrative.append("  Scoped trace assignments:")
        for row in scoped[:12]:
            narrative.append(
                f"    {row.get('trace_id', '?')} -> {row.get('primary_context_frame_id', '?')} "
                f"(weight {_fmt(row.get('weight'))})"
            )
    if links:
        narrative.append("  Frame links:")
        for link in links[:6]:
            narrative.append(
                f"    {link.get('source_frame_id', '?')} -{link.get('link_type', '?')}-> "
                f"{link.get('target_frame_id', '?')} ({_fmt(link.get('weight'))})"
            )
    if gates:
        narrative.append("  Interference gates (allowed / blocked traces per scope):")
        for gate in gates[:6]:
            allowed = ", ".join(gate.get("allowed_trace_ids") or []) or "-"
            blocked = ", ".join(gate.get("blocked_trace_ids") or []) or "-"
            narrative.append(f"    {gate.get('scope_frame_id', gate.get('gate_id', '?'))}:")
            narrative.append(f"      allow: {allowed}")
            if blocked != "-":
                narrative.append(f"      block: {blocked}")
    if pressures:
        narrative.append("  Local basin family hints:")
        for row in pressures:
            ctx_id = row.get("context_frame_id", "?")
            hints = row.get("basin_family_hints") or {}
            ranked = sorted(hints.items(), key=lambda item: item[1], reverse=True)
            hint_text = ", ".join(f"{name}={_fmt(weight)}" for name, weight in ranked[:5])
            narrative.append(f"    {ctx_id}: {hint_text or '-'}")
    return headline, lines, narrative


def _summarize_interference(data: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    tt = len(data.get("trace_trace_edges") or [])
    tf = len(data.get("trace_frame_edges") or [])
    fb = len(data.get("frame_basin_edges") or [])
    scoped = data.get("scoped_basin_energy_deltas") or []
    conflicts = data.get("conflict_reports") or []
    headline = f"{tf} trace->frame links, {len(scoped)} scoped basin deltas, {len(conflicts)} conflicts"
    lines = [
        f"trace_trace_edges: {tt}",
        f"trace_frame_edges: {tf}",
        f"frame_basin_edges: {fb}",
        f"scoped_basin_energy_deltas: {len(scoped)}",
        f"conflict_reports: {len(conflicts)}",
    ]
    notes = data.get("audit_notes") or []
    if notes:
        lines.append(f"audit: {notes[0]}")

    narrative = [_section("Interference - competition between readings")]
    tf_edges = data.get("trace_frame_edges") or []
    if tf_edges:
        narrative.append("  Trace supports frame:")
        for edge in tf_edges[:10]:
            narrative.append(
                f"    {edge.get('trace_id', '?')} -> {edge.get('frame_id', '?')} "
                f"(d={_fmt(edge.get('delta'))})"
            )
    fb_edges = data.get("frame_basin_edges") or []
    if fb_edges:
        narrative.append("  Frame supports basin family:")
        for edge in fb_edges[:8]:
            narrative.append(
                f"    {edge.get('frame_id', '?')} -> {edge.get('basin_id', '?')} "
                f"(d={_fmt(edge.get('delta'))})"
            )
    if scoped:
        narrative.append("  Scoped basin energy shifts:")
        for delta in scoped[:10]:
            reasons = ", ".join(delta.get("reason_refs") or []) or "-"
            narrative.append(
                f"    {delta.get('scope_frame_id', '?')} / {delta.get('basin_id', '?')}: "
                f"d={_fmt(delta.get('delta'))} ({reasons})"
            )
    if conflicts:
        narrative.append("  Conflicts:")
        for report in conflicts[:6]:
            members = ", ".join(report.get("members") or []) or "-"
            narrative.append(
                f"    {report.get('scope_frame_id', '?')}: {report.get('conflict_type', '?')} "
                f"[{members}] severity {_fmt(report.get('severity'))}"
            )
    cooperation = data.get("cooperation_maps") or {}
    competition = data.get("competition_maps") or {}
    if cooperation:
        narrative.append(f"  Cooperation maps: {len(cooperation)} scopes")
    if competition:
        narrative.append("  Competing basin families:")
        for scope, members in list(competition.items())[:4]:
            narrative.append(f"    {scope}: {', '.join(members) if isinstance(members, list) else members}")
    return headline, lines, narrative


def _summarize_basins(data: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    basins = data.get("candidate_basin_states") or []
    summary = data.get("competition_summary") or {}
    assemblies = data.get("basin_assemblies") or []
    top = summary.get("top_basin_id", "")
    margin = summary.get("top_margin", 0.0)
    headline = f"{len(basins)} competing basins; leader {top or '-'} (margin {_fmt(margin)})"
    lines = [
        f"candidate_basin_states: {len(basins)}",
        f"top_basin: {top}",
        f"top_margin: {margin}",
        f"basin_assemblies: {len(assemblies)}",
    ]

    narrative = [_section("Basins - competing interpretations")]
    ranked = sorted(basins, key=lambda item: float(item.get("energy") or 0), reverse=True)
    for index, basin in enumerate(ranked[:8], start=1):
        basin_id = basin.get("basin_id", "?")
        energy = _fmt(basin.get("energy"))
        margin_next = _fmt(basin.get("margin_vs_next"))
        frames = ", ".join(basin.get("supporting_frame_ids") or []) or "-"
        scopes = ", ".join(basin.get("scope_frame_ids") or []) or "-"
        traces = ", ".join(basin.get("supporting_trace_ids") or []) or "-"
        label = " <- leading" if index == 1 else ""
        narrative.append(
            f"  #{index} {basin_id}: energy {energy}, margin vs next {margin_next}{label}"
        )
        narrative.append(f"      frames: {frames} | scopes: {scopes}")
        narrative.append(f"      traces: {traces}")
    if assemblies:
        narrative.append("  Assemblies:")
        for assembly in assemblies[:4]:
            members = ", ".join(assembly.get("member_basin_ids") or []) or "-"
            narrative.append(
                f"    {assembly.get('assembly_id', '?')}: members [{members}] "
                f"combined energy {_fmt(assembly.get('combined_energy'))}"
            )
    if not basins:
        narrative.append("  (no basin states - DMF/binding may be empty)")
    return headline, lines, narrative


def _summarize_lucidity(data: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    decision = data.get("decision", "")
    committed = data.get("committed_state") or {}
    checks = data.get("check_results") or {}
    preserved = data.get("preserved_hypotheses") or []
    primary = committed.get("primary_basin_id", "") if isinstance(committed, dict) else ""
    headline = f"decision: {decision}"
    if primary:
        headline += f" (primary basin {primary})"
    lines = [f"decision: {decision}"]
    if primary:
        lines.append(f"primary_basin_id: {primary}")

    narrative = [_section(f"Lucidity - decision: {decision}")]
    check_rows = [
        ("margin", checks.get("margin_check")),
        ("coverage", checks.get("coverage_check")),
        ("coherence", checks.get("coherence_check")),
        ("binding_stability", checks.get("binding_stability_check")),
        ("scope", checks.get("scope_check")),
        ("projection_fit", checks.get("projection_fit_check")),
        ("contradiction", checks.get("contradiction_check")),
        ("maturity", checks.get("maturity_check")),
        ("risk", checks.get("risk_check")),
    ]
    narrative.append("  Checks:")
    for name, result in check_rows:
        if not result:
            continue
        mark = "pass" if result.get("passed") else "FAIL"
        score = _fmt(result.get("score"))
        threshold = _fmt(result.get("threshold"))
        narrative.append(f"    {name}: {mark} (score {score}, threshold {threshold})")

    if isinstance(committed, dict) and committed:
        shape = committed.get("commit_shape", "")
        narrative.append(f"  Commit shape: {shape or '-'}")
        frame_commits = committed.get("frame_commits") or []
        if frame_commits:
            narrative.append("  Frame commits:")
            for commit in frame_commits:
                narrative.append(
                    f"    {commit.get('context_frame_id', '?')} / {commit.get('frame_type', '?')} "
                    f"-> basin {commit.get('basin_id', '?')}"
                )
        units = committed.get("render_units") or []
        if units:
            narrative.append(f"  Render units prepared for decoder: {len(units)}")
            for unit in units[:6]:
                payload = unit.get("payload") or {}
                summary = payload.get("summary") or unit.get("unit_type", "?")
                narrative.append(f"    [{unit.get('unit_type', '?')}] {summary}")

    if preserved:
        narrative.append(f"  Preserved hypotheses ({len(preserved)}):")
        for hypo in preserved[:6]:
            narrative.append(
                f"    {hypo.get('basin_id', hypo.get('hypothesis_id', '?'))}: "
                f"{hypo.get('narrative_hint', '-')} (conf {_fmt(hypo.get('confidence'))})"
            )
    return headline, lines, narrative


def _summarize_decoder(data: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    refused = data.get("refused", False)
    report = data.get("faithfulness_report") or {}
    grid = data.get("surface_grid")
    text = (data.get("surface_text") or "").strip()
    if isinstance(grid, list) and grid and isinstance(grid[0], list):
        rows = len(grid)
        cols = len(grid[0]) if grid[0] else 0
        headline = f"grid {rows}x{cols}"
    elif refused:
        headline = "refused"
    else:
        headline = text[:72] + ("…" if len(text) > 72 else "") if text else "empty"
    lines: list[str] = []
    if isinstance(report, dict):
        passed = report.get("passed", True)
        lines.append(f"faithfulness: {'pass' if passed else 'fail'}")
        violations = report.get("policy_violations") or []
        if violations:
            lines.append(f"violations: {', '.join(str(v) for v in violations[:3])}")
    render_mode = data.get("render_mode", "")
    if render_mode:
        lines.append(f"render_mode: {render_mode}")
    if isinstance(grid, list) and grid and isinstance(grid[0], list):
        rows = len(grid)
        cols = len(grid[0]) if grid[0] else 0
        lines.append(f"surface_grid: {rows}x{cols}")
    else:
        lines.append(f"refused: {refused}")
        if text:
            lines.append(f"surface_text: {text[:120]}{'…' if len(text) > 120 else ''}")

    narrative = [_section("Decoder - user-facing output")]
    narrative.append(f"  Refused: {refused}")
    if render_mode:
        narrative.append(f"  Render mode: {render_mode}")
    if isinstance(report, dict):
        narrative.append(f"  Faithfulness: {'pass' if report.get('passed', True) else 'fail'}")
        omitted = report.get("omitted_required_units") or []
        if omitted:
            narrative.append(f"  Omitted required units: {', '.join(omitted)}")
    notes = data.get("audit_notes") or []
    if notes:
        narrative.append(f"  Path: {', '.join(str(n) for n in notes)}")
    if text:
        narrative.append(f"  Answer: {text}")
    elif isinstance(grid, list) and grid:
        narrative.append(f"  Grid: {len(grid)}×{len(grid[0]) if grid[0] else 0}")
    return headline, lines, narrative


def _summarize_projector(data: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    rollouts = data.get("rollouts") or []
    best = data.get("best_rollout_id", "")
    headline = f"{len(rollouts)} rollouts"
    lines = [f"rollouts: {len(rollouts)}", f"best_rollout_id: {best}"]
    narrative = [_section("Projector")]
    if best:
        narrative.append(f"  Best rollout: {best}")
    for rollout in rollouts[:4]:
        narrative.append(
            f"  {rollout.get('rollout_id', '?')}: fit {_fmt(rollout.get('fit_score'))} "
            f"steps {len(rollout.get('steps') or [])}"
        )
    if not rollouts:
        narrative.append("  (no rollouts)")
    return headline, lines, narrative


def summarize_stage_output(
    stage_name: str,
    output: Any,
    *,
    stage_input: Any = None,
) -> dict[str, Any]:
    """Build headline, compact lines, and human narrative for a stage record."""
    if output is None:
        return {"headline": "no output", "lines": ["(no output)"], "narrative": ["(no output)"]}

    data = _as_dict(output)
    inp = _as_dict(stage_input)
    name = stage_name.strip().lower()

    if name == "perception":
        headline, lines, narrative = _summarize_perception(data)
    elif name == "cue_encoder":
        headline, lines, narrative = _summarize_cue_encoder(data)
    elif name == "dmf":
        headline, lines, narrative = _summarize_dmf(data, stage_input=inp)
    elif name == "binding":
        headline, lines, narrative = _summarize_binding(data, stage_input=inp)
    elif name == "context_op":
        headline, lines, narrative = _summarize_context_op(data, stage_input=inp)
    elif name == "interference":
        headline, lines, narrative = _summarize_interference(data)
    elif name == "basins":
        headline, lines, narrative = _summarize_basins(data)
    elif name == "lucidity":
        headline, lines, narrative = _summarize_lucidity(data)
    elif name == "projector":
        headline, lines, narrative = _summarize_projector(data)
    elif name == "decoder":
        headline, lines, narrative = _summarize_decoder(data)
    else:
        keys = ", ".join(sorted(data.keys())[:8])
        headline = stage_name
        lines = [f"fields: {keys}"]
        narrative = [f"Stage {stage_name}: {keys}"]

    return {"headline": headline, "lines": lines, "narrative": narrative}


def format_stage_summary_block(summary: dict[str, Any], *, indent: str = "     ") -> list[str]:
    """Prefer narrative lines for display; fall back to compact lines."""
    narrative = summary.get("narrative") or []
    if narrative:
        return [f"{indent}{line}" if line and not line.startswith("--") else f"{indent}{line}" for line in narrative]
    return [f"{indent}{line}" for line in summary.get("lines") or []]


def build_run_narrative(run_dir_records: list[tuple[str, dict[str, Any]]]) -> str:
    """Full run story from ordered (stage_name, record) pairs."""
    parts: list[str] = ["Pipeline run narrative", "=" * 20, ""]
    for stage_name, record in run_dir_records:
        summary = record.get("summary") or {}
        headline = summary.get("headline") or stage_name
        parts.append(f"## {stage_name}: {headline}")
        for line in summary.get("narrative") or summary.get("lines") or []:
            parts.append(line)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"
