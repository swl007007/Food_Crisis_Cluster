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
