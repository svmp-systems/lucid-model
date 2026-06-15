"""Fluent surface composition — merge approved claims into readable prose."""

from __future__ import annotations

from dataclasses import dataclass, field

from lucid.cognition.output.decoder.phrases import humanize
from lucid.ir.lucidity import RenderUnit, SourceRef


@dataclass(slots=True)
class RelationClaim:
    subject: str
    relation: str
    target: str
    unit_id: str
    source_refs: list[SourceRef] = field(default_factory=list)
    required: bool = True


def _join_phrase_list(items: list[str]) -> str:
    clean = [humanize(item).strip() for item in items if str(item).strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return ", ".join(clean[:-1]) + f", and {clean[-1]}"


def _sentence(text: str) -> str:
    cleaned = " ".join(str(text).strip().split())
    if not cleaned:
        return ""
    return cleaned if cleaned.endswith((".", "!", "?")) else cleaned + "."


def _discourse_clause(text: str) -> str:
    cleaned = humanize(text).strip()
    lowered = cleaned.lower()
    for prefix in ("but ", "however, ", "however "):
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    return _capitalize_sentence(cleaned)


def realize_relation_group(subject: str, relation: str, targets: list[str]) -> str:
    """One natural sentence for a subject/relation and its target objects."""
    subj = humanize(subject).strip()
    rel = str(relation or "").strip().lower()
    target_phrase = _join_phrase_list(targets)
    if not subj or not target_phrase:
        return ""
    subject_phrase = _capitalize_sentence(_with_indefinite_article(subj))

    if rel in {"type_of", "is_a", "kind_of"}:
        return _sentence(f"{subject_phrase} is {_with_indefinite_article(target_phrase)}")
    if rel in {"property", "has_property"}:
        if target_phrase.lower().startswith("can "):
            return _sentence(f"{subject_phrase} {target_phrase}")
        if target_phrase.lower().startswith(("based on ", "built with ")):
            return _sentence(f"{subject_phrase} is {target_phrase}")
        return _sentence(f"{subject_phrase} has {target_phrase}")
    if rel in {"can", "capability"}:
        if target_phrase.lower().startswith("can "):
            return _sentence(f"{subject_phrase} {target_phrase}")
        return _sentence(f"{subject_phrase} can {target_phrase}")
    if rel in {"challenge", "limitation"}:
        return _sentence(f"{subject_phrase} is limited by {target_phrase}")
    if rel in {"uses", "use"}:
        return _sentence(f"{subject_phrase} uses {target_phrase}")
    if rel in {"enables", "supports"}:
        return _sentence(f"{subject_phrase} supports {target_phrase}")
    if rel in {"contrast", "contrasts_with"}:
        return _sentence(_discourse_clause(target_phrase))
    return _sentence(f"{subject_phrase} {humanize(rel)} {target_phrase}")


def extract_relation_claims(units: list[RenderUnit]) -> list[RelationClaim]:
    claims: list[RelationClaim] = []
    for unit in units:
        if unit.unit_type == "artifact":
            continue
        payload = dict(unit.payload)
        subject = str(payload.get("subject") or "").strip()
        relation = str(payload.get("relation") or "").strip()
        target = str(payload.get("target") or "").strip()
        if not subject or not relation or not target:
            continue
        claims.append(
            RelationClaim(
                subject=subject,
                relation=relation,
                target=target,
                unit_id=unit.unit_id,
                source_refs=list(unit.source_refs),
                required=unit.required,
            )
        )
    return claims


def group_relation_claims(claims: list[RelationClaim]) -> list[tuple[str, str, list[RelationClaim]]]:
    grouped: dict[tuple[str, str], list[RelationClaim]] = {}
    order: list[tuple[str, str]] = []
    for claim in claims:
        key = (humanize(claim.subject).strip().lower(), claim.relation.strip().lower())
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(claim)
    return [(subject, relation, grouped[(subject, relation)]) for subject, relation in order]


@dataclass(slots=True)
class FluentLine:
    text: str
    unit_ids: list[str] = field(default_factory=list)
    source_refs: list[SourceRef] = field(default_factory=list)
    required: bool = True


def _capitalize_sentence(text: str) -> str:
    cleaned = " ".join(str(text).strip().split())
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:]


def _needs_article(phrase: str) -> bool:
    lowered = phrase.strip().lower()
    if not lowered:
        return False
    if lowered.startswith(("a ", "an ", "the ", "this ", "that ", "these ", "those ")):
        return False
    if lowered.endswith("s") and not lowered.endswith("ss"):
        return False
    first = lowered.split()[0]
    return first not in {"quantum", "classical"}


def _with_indefinite_article(phrase: str) -> str:
    clean = humanize(phrase).strip()
    if not _needs_article(clean):
        return clean
    lowered = clean.lower()
    article = "an" if clean[0].lower() in {"a", "e", "i", "o", "u"} else "a"
    if lowered.startswith(("unit", "university", "use", "user", "one ")):
        article = "a"
    if lowered.startswith(("hour", "honest", "honor")):
        article = "an"
    return f"{article} {clean}"


def _claim_refs(claims: list[RelationClaim]) -> list[SourceRef]:
    refs: list[SourceRef] = []
    seen: set[tuple[str, str, str]] = set()
    for claim in claims:
        for ref in claim.source_refs:
            key = (ref.ref_type, ref.ref_id, ref.scope_frame_id)
            if key in seen:
                continue
            seen.add(key)
            refs.append(ref)
    return refs


def _claim_unit_ids(claims: list[RelationClaim]) -> list[str]:
    return [claim.unit_id for claim in claims]


def _definition_line_for_subject(subject: str, claims: list[RelationClaim]) -> FluentLine | None:
    """Compose a compact explanatory answer from source-backed relation claims."""

    by_relation: dict[str, list[RelationClaim]] = {}
    for claim in claims:
        by_relation.setdefault(claim.relation.strip().lower(), []).append(claim)

    type_claims = [
        *by_relation.get("type_of", []),
        *by_relation.get("is_a", []),
        *by_relation.get("kind_of", []),
    ]
    property_claims = [
        *by_relation.get("property", []),
        *by_relation.get("has_property", []),
    ]
    capability_claims = [
        *by_relation.get("can", []),
        *by_relation.get("capability", []),
    ]
    challenge_claims = [
        *by_relation.get("challenge", []),
        *by_relation.get("limitation", []),
    ]

    if not type_claims or not (property_claims or capability_claims or challenge_claims):
        return None

    subj = humanize(subject).strip()
    if not subj:
        return None

    first_sentence = ""
    second_parts: list[str] = []
    type_phrase = _join_phrase_list([claim.target for claim in type_claims])
    if type_phrase:
        first_sentence = (
            f"{_capitalize_sentence(_with_indefinite_article(subj))} is "
            f"{_with_indefinite_article(type_phrase)}"
        )

    property_targets = [claim.target for claim in property_claims]
    if property_targets:
        property_phrase = _join_phrase_list(property_targets)
        lowered = property_phrase.lower()
        if lowered.startswith("can "):
            second_parts.append(f"it {property_phrase}")
        elif lowered.startswith(("based on ", "built with ")):
            second_parts.append(f"it is {property_phrase}")
        else:
            second_parts.append(f"it has {property_phrase}")

    capability_targets = [claim.target for claim in capability_claims]
    if capability_targets:
        capability_phrase = _join_phrase_list(capability_targets)
        lowered = capability_phrase.lower()
        if lowered.startswith(("can ", "could ", "may ", "might ")):
            second_parts.append(f"it {capability_phrase}")
        else:
            second_parts.append(f"it can {capability_phrase}")

    challenge_targets = [claim.target for claim in challenge_claims]
    if challenge_targets:
        challenge_phrase = _join_phrase_list(challenge_targets)
        if property_targets:
            second_parts.append(f"but it is limited by {challenge_phrase}")
        else:
            second_parts.append(f"it is limited by {challenge_phrase}")

    sentences = [first_sentence] if first_sentence else []
    if second_parts:
        if len(second_parts) == 1:
            sentences.append(_capitalize_sentence(second_parts[0]))
        elif second_parts[-1].startswith("but "):
            sentences.append(_capitalize_sentence(", ".join(second_parts[:-1]) + f", {second_parts[-1]}"))
        else:
            sentences.append(_capitalize_sentence("; ".join(second_parts)))

    if not sentences:
        return None

    text = " ".join(_sentence(sentence) for sentence in sentences if sentence)
    used_claims = [*type_claims, *property_claims, *capability_claims, *challenge_claims]
    return FluentLine(
        text=text,
        unit_ids=_claim_unit_ids(used_claims),
        source_refs=_claim_refs(used_claims),
        required=any(claim.required for claim in used_claims),
    )


def _line_from_summary_payload(payload: dict) -> str:
    if "bank_sense" in payload:
        sense = humanize(payload["bank_sense"])
        scope = str(payload.get("scope_frame_id") or "").strip()
        if scope:
            return _sentence(f"In scope {scope}, bank is being used in the {sense} sense")
        return _sentence(f"Here, bank is being used in the {sense} sense")
    if "summary" in payload:
        summary = humanize(payload["summary"]).strip()
        if not summary:
            return ""
        unresolved = payload.get("unresolved_slots") or []
        if isinstance(unresolved, list) and unresolved:
            return _sentence(f"{summary} is not fully settled")
        return _sentence(summary)
    if "refusal_reason" in payload:
        return _sentence(str(payload["refusal_reason"]))
    if "action_type" in payload:
        action = humanize(payload.get("action_type", "approved action"))
        target = humanize(payload.get("target_ref", "")).strip()
        if target:
            return _sentence(f"The approved action is {action} for {target}")
        return _sentence(f"The approved action is {action}")
    if payload.get("speech_kind") and payload.get("summary"):
        return _sentence(humanize(payload["summary"]))
    return ""


def _is_relation_payload(payload: dict) -> bool:
    return bool(str(payload.get("subject") or "").strip() and str(payload.get("relation") or "").strip())


def compose_fluent_lines(units: list[RenderUnit]) -> list[FluentLine]:
    """Build fluent lines: grouped relation claims first, then other approved content."""
    relation_units = [unit for unit in units if _is_relation_payload(dict(unit.payload))]
    other_units = [unit for unit in units if unit not in relation_units and unit.unit_type != "artifact"]

    lines: list[FluentLine] = []
    claims = extract_relation_claims(relation_units)
    claims_by_subject: dict[str, list[RelationClaim]] = {}
    subject_order: list[str] = []
    for claim in claims:
        subject_key = humanize(claim.subject).strip().lower()
        if not subject_key:
            continue
        if subject_key not in claims_by_subject:
            claims_by_subject[subject_key] = []
            subject_order.append(subject_key)
        claims_by_subject[subject_key].append(claim)

    consumed_unit_ids: set[str] = set()
    for subject_key in subject_order:
        definition_line = _definition_line_for_subject(subject_key, claims_by_subject[subject_key])
        if definition_line is None:
            continue
        lines.append(definition_line)
        consumed_unit_ids.update(definition_line.unit_ids)

    grouped = group_relation_claims(
        [claim for claim in claims if claim.unit_id not in consumed_unit_ids]
    )
    for subject, relation, claims in grouped:
        targets = [claim.target for claim in claims]
        text = realize_relation_group(subject, relation, targets)
        if not text:
            continue
        refs: list[SourceRef] = []
        seen: set[tuple[str, str, str]] = set()
        unit_ids: list[str] = []
        required = False
        for claim in claims:
            unit_ids.append(claim.unit_id)
            required = required or claim.required
            for ref in claim.source_refs:
                key = (ref.ref_type, ref.ref_id, ref.scope_frame_id)
                if key in seen:
                    continue
                seen.add(key)
                refs.append(ref)
        lines.append(FluentLine(text=text, unit_ids=unit_ids, source_refs=refs, required=required))

    seen_summaries: set[str] = set()
    for unit in other_units:
        payload = dict(unit.payload)
        if unit.unit_type == "claim" and "basin_id" in payload and "energy" in payload:
            continue
        if unit.unit_type == "frame_summary":
            surfaces = [
                str(item).strip()
                for item in payload.get("member_evidence_surfaces") or []
                if str(item).strip()
            ]
            phrase = str(payload.get("summary") or "").strip() or " ".join(surfaces)
            if not phrase:
                continue
            unresolved = {str(item).strip().lower() for item in payload.get("unresolved_slots") or []}
            if "bank_sense" in unresolved or "bank" in unresolved:
                text = _sentence(f"{phrase.rstrip('.')}, but the sense of \"bank\" is not fully settled")
            else:
                text = _sentence(phrase)
        else:
            text = _line_from_summary_payload(payload)
        if not text:
            continue
        stem = text.lower()
        if stem in seen_summaries:
            continue
        seen_summaries.add(stem)
        lines.append(
            FluentLine(
                text=text,
                unit_ids=[unit.unit_id],
                source_refs=list(unit.source_refs),
                required=unit.required,
            )
        )
    return lines
