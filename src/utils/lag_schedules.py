"""Lag schedule helpers shared by GeoRF pipelines."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

try:
    from config import DEFAULT_LAGS_MONTHS as _DEFAULT_LAGS_MONTHS
    from config import LEGACY_LAG_VALUES as _LEGACY_LAG_VALUES
except Exception:  # pragma: no cover - fallback for import-order edge cases
    _DEFAULT_LAGS_MONTHS = (4, 8, 12)
    _LEGACY_LAG_VALUES = {3, 6, 9}


def resolve_lag_schedule(lags: Sequence[int], *, context: str = "lag schedule") -> list[int]:
    """Validate and return the active lag schedule."""

    resolved = list(lags)
    if list(dict.fromkeys(resolved)) != resolved:
        raise ValueError(f"Duplicated entries detected in {context}: {resolved}")

    legacy_hits = sorted({lag for lag in resolved if lag in _LEGACY_LAG_VALUES})
    if legacy_hits:
        raise ValueError(
            f"Legacy lag month(s) {legacy_hits} detected in {context}; configure to {_DEFAULT_LAGS_MONTHS}."
        )

    if resolved != list(_DEFAULT_LAGS_MONTHS):
        raise ValueError(
            f"Unsupported {context}: {resolved}. Set to {_DEFAULT_LAGS_MONTHS} to match FEWSNET baseline."
        )

    return resolved


def forecasting_scope_to_lag(scope: int, lags: Sequence[int]) -> int:
    """Map forecasting scope index to lag months with bounds checking."""

    if scope < 1 or scope > len(lags):
        raise ValueError(
            f"Forecasting scope {scope} is out of range for {len(lags)} lag options {list(lags)}."
        )
    return lags[scope - 1]


def log_lag_schedule(lags: Sequence[int], artifacts_root: str) -> str:
    """Append the active lag schedule to the shared log file and return its path."""

    log_path = Path(artifacts_root) / "logs" / "lag_schedule_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp}\t{','.join(str(lag) for lag in lags)}\n")
    return str(log_path)


def assert_no_legacy_values(sequence: Sequence[int], *, context: str) -> None:
    """Raise if legacy lag months are present in ``sequence``."""

    legacy_hits = sorted({value for value in sequence if value in _LEGACY_LAG_VALUES})
    if legacy_hits:
        raise ValueError(
            f"Legacy lag month(s) {legacy_hits} detected in {context}; supported schedule is {_DEFAULT_LAGS_MONTHS}."
        )


def validate_lag_boundaries(active_lag: int, train_end, test_start) -> None:
    """
    Validate that TRAIN window ends exactly ACTIVE_LAG months before TEST starts.

    Parameters
    ----------
    active_lag : int
        Active lag in months (e.g., 4, 8, 12)
    train_end : datetime or pd.Timestamp
        End of TRAIN window
    test_start : datetime or pd.Timestamp
        Start of TEST window

    Raises
    ------
    ValueError
        If train_end is not exactly active_lag months before test_start
    """
    import pandas as pd

    if train_end >= test_start:
        raise ValueError(
            f"TRAIN end ({train_end.date()}) must be before TEST start ({test_start.date()})"
        )

    # Calculate expected train_end based on test_start and active_lag
    expected_train_end = test_start - pd.DateOffset(months=active_lag)

    # Allow for small timestamp differences (same month-start)
    if train_end.to_period('M') != expected_train_end.to_period('M'):
        raise ValueError(
            f"TRAIN end month ({train_end.to_period('M')}) does not match "
            f"expected end {active_lag} months before TEST ({expected_train_end.to_period('M')})"
        )


def assert_max_lag_valid(lags: Sequence[int], active_lag: int, *, context: str = "feature lags") -> None:
    """
    Assert that all lags are less than or equal to the active lag.

    Parameters
    ----------
    lags : Sequence[int]
        List of lag values used in feature engineering
    active_lag : int
        Maximum allowed lag (active lag for this evaluation)
    context : str
        Description of where lags are used (for error messages)

    Raises
    ------
    ValueError
        If any lag exceeds active_lag
    """
    max_lag = max(lags)
    if max_lag > active_lag:
        raise ValueError(
            f"Maximum lag ({max_lag}) in {context} exceeds active lag ({active_lag}). "
            f"This would cause data leakage from the test period into training features."
        )
