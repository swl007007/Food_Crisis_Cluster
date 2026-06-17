#!/usr/bin/env python3
"""Paper-facing forecasting horizon display labels."""

from __future__ import annotations


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


def label_for_scope(scope: str) -> str:
    """Return the paper-facing forecasting horizon label for a scope token."""
    return HORIZON_LABELS.get(str(scope), str(scope))


def replace_paper_horizon_terms(text: str) -> str:
    """Replace paper-facing forecast interval labels without touching true lag terms."""
    updated = text
    for old, new in OLD_TO_NEW_DISPLAY_REPLACEMENTS.items():
        updated = updated.replace(old, new)
    return updated
