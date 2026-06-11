"""Shared wording maps for render + faithfulness checks."""

from __future__ import annotations

SENSE_PHRASES: dict[str, str] = {
    "financial_storage": "a financial institution where money is stored",
    "river_bank": "the edge of a river",
    "financial": "a financial institution",
}


def humanize(value: object) -> str:
    if not isinstance(value, str):
        return str(value)
    key = value.strip().lower().replace(" ", "_")
    if key in SENSE_PHRASES:
        return SENSE_PHRASES[key]
    return value.replace("_", " ")
