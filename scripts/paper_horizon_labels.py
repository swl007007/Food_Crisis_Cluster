#!/usr/bin/env python3
"""Paper-facing forecasting horizon display labels."""

from __future__ import annotations

import re


HORIZON_MONTHS_BY_SCOPE = {
    "fs1": 4,
    "fs2": 8,
    "fs3": 12,
}

HORIZON_LABELS = {
    scope: f"{months}-month horizon"
    for scope, months in HORIZON_MONTHS_BY_SCOPE.items()
}

OLD_TO_NEW_DISPLAY_REPLACEMENTS = {
    "4-month-lag": "4-month-horizon",
    "8-month-lag": "8-month-horizon",
    "12-month-lag": "12-month-horizon",
    "4-month lag": "4-month horizon",
    "8-month lag": "8-month horizon",
    "12-month lag": "12-month horizon",
    "Forecasting horizon / lag": "Forecasting horizon",
    "forecasting horizon / lag": "forecasting horizon",
    "horizon / lag": "horizon",
    "Forecasting Horizon / Lag": "Forecasting Horizon",
    "Forecasting horizon (month lag)": "Forecasting horizon",
}

FORBIDDEN_PAPER_LAG_TERMS = tuple(OLD_TO_NEW_DISPLAY_REPLACEMENTS)

ALLOWED_REMAINING_LAG_MARKERS = (
    "Lag Exclude",
    "lagged",
    "_lag1m",
    "_lag4m",
    "_lag8m",
    "_lag12m",
)


def label_for_scope(scope: str) -> str:
    """Return the paper-facing forecasting horizon label for a scope token."""
    return HORIZON_LABELS.get(str(scope), str(scope))


def replace_paper_horizon_terms(text: str) -> str:
    """Replace paper-facing forecast interval labels without touching true lag terms."""
    updated = text
    for old, new in OLD_TO_NEW_DISPLAY_REPLACEMENTS.items():
        pattern = re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(old)}(?![A-Za-z0-9_-])")
        updated = pattern.sub(new, updated)
    return updated


def _term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(term)}(?![A-Za-z0-9_-])")


def is_allowed_remaining_lag_line(line: str) -> bool:
    """Return whether a remaining lag mention is an intended technical term."""
    text = str(line)
    if forbidden_paper_lag_terms(text):
        return False

    lowered = text.lower()
    return any(marker.lower() in lowered for marker in ALLOWED_REMAINING_LAG_MARKERS)


def forbidden_paper_lag_terms(text: str) -> list[str]:
    """Return old paper-facing lag labels still present in text."""
    matches: list[tuple[int, int, str]] = []
    for term in FORBIDDEN_PAPER_LAG_TERMS:
        matches.extend((match.start(), match.end(), term) for match in _term_pattern(term).finditer(str(text)))

    accepted: list[tuple[int, int, str]] = []
    for start, end, term in sorted(matches, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(start < kept_end and end > kept_start for kept_start, kept_end, _ in accepted):
            continue
        accepted.append((start, end, term))

    seen: set[str] = set()
    terms: list[str] = []
    for _, _, term in accepted:
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return terms
