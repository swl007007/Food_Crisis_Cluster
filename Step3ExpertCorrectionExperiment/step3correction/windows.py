"""Temporal windows for the Step 3 expert selective-correction experiment.

The outer window reproduces the historical main splitter exactly, including its
off-by-one behaviour (``src/customize/customize.py:407-445``): a *configured*
36-month window resolves to ``W = [O - 35 months, O)``, i.e. 35 monthly
timestamps, where ``O = T - H`` is the forecast origin.

Inside ``W`` the correction adds two further boundaries:

* ``V = [O - 12 months, O)`` - the validation interval used for rule selection
  (revised 2026-09-18, R8).  The source label months are tri-annual and exactly
  four months apart, and ``O = T - H`` with ``H in {4, 8}`` keeps
  ``O % 4 == 2``, so the previous ``[O - 6, O)`` admitted exactly **one**
  observed label month in 24/24 folds and made the approved
  ``distinct months >= 2`` gate structurally unsatisfiable.  Twelve calendar
  months admit exactly **three** observed label months (``O-4``, ``O-8``,
  ``O-12``) while leaving every approved selection gate untouched.
* ``fit`` - outer rows whose label month is strictly before ``V_start - H``.
  This single conservative cutoff is horizon-safe at the earliest validation
  origin and therefore at every later validation origin.
* ``gap`` - rows between the fit cutoff and ``V``; withheld from the initial fit
  and used by neither stage.

Accepted consequence of the twelve-month ``V``, surfaced rather than hidden: the
horizon-isolated first-stage fit retains roughly four observed label months for
fs1 and three for fs2, so partitions under the 50-row minimum abstain as normal.

Nothing here may be relaxed to obtain usable validation evidence: the outer
window stays at 35 timestamps, ``V`` is anchored on ``O`` (never on the last
observed training month), and ``V`` is never widened beyond twelve months.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

MAIN_TRAIN_WINDOW_MONTHS = 36
# Historical off-by-one: configured 36 months -> 35 monthly timestamps.
OUTER_WINDOW_TIMESTAMPS = MAIN_TRAIN_WINDOW_MONTHS - 1
VALIDATION_CALENDAR_MONTHS = 12


@dataclass(frozen=True)
class FoldWindows:
    """Resolved calendar endpoints for one (scope, target month) fold."""

    target_month: pd.Period
    horizon_months: int
    origin: pd.Timestamp
    outer_start: pd.Timestamp
    outer_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    fit_cutoff: pd.Timestamp

    def as_record(self) -> Dict[str, object]:
        """Return a flat, JSON/CSV-friendly description of every endpoint."""
        return {
            "target_month": str(self.target_month),
            "horizon_months": int(self.horizon_months),
            "origin_month": str(self.origin.date()),
            "outer_start": str(self.outer_start.date()),
            "outer_end_exclusive": str(self.outer_end.date()),
            "outer_window_timestamps": OUTER_WINDOW_TIMESTAMPS,
            "validation_calendar_months": VALIDATION_CALENDAR_MONTHS,
            "validation_start": str(self.validation_start.date()),
            "validation_end_exclusive": str(self.validation_end.date()),
            "fit_cutoff_exclusive": str(self.fit_cutoff.date()),
        }


def resolve_fold_windows(target_month, horizon_months: int) -> FoldWindows:
    """Resolve the outer/validation/fit endpoints for one fold."""
    if horizon_months <= 0:
        raise ValueError(f"horizon_months must be positive, got {horizon_months}")
    period = (
        target_month
        if isinstance(target_month, pd.Period)
        else pd.Period(str(target_month), freq="M")
    )
    if period.freqstr != "M":
        raise ValueError(f"target_month must be monthly, got freq {period.freqstr}")

    target_start = period.to_timestamp()
    origin = target_start - pd.DateOffset(months=horizon_months)
    outer_start = origin - pd.DateOffset(months=OUTER_WINDOW_TIMESTAMPS)
    validation_start = origin - pd.DateOffset(months=VALIDATION_CALENDAR_MONTHS)
    fit_cutoff = validation_start - pd.DateOffset(months=horizon_months)

    if origin >= target_start:
        raise ValueError("Forecast origin must precede the target month")
    if not outer_start < fit_cutoff <= validation_start < origin:
        raise ValueError(
            "Inconsistent fold endpoints: "
            f"outer_start={outer_start.date()} fit_cutoff={fit_cutoff.date()} "
            f"validation_start={validation_start.date()} origin={origin.date()}"
        )
    return FoldWindows(
        target_month=period,
        horizon_months=int(horizon_months),
        origin=origin,
        outer_start=outer_start,
        outer_end=origin,
        validation_start=validation_start,
        validation_end=origin,
        fit_cutoff=fit_cutoff,
    )


def outer_mask(dates: pd.Series, windows: FoldWindows) -> np.ndarray:
    """Return the main splitter's outer-window mask ``[outer_start, origin)``."""
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    return ((dates >= windows.outer_start) & (dates < windows.outer_end)).to_numpy()


def target_mask(dates: pd.Series, windows: FoldWindows) -> np.ndarray:
    """Return the single-target-month test mask."""
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    start = windows.target_month.to_timestamp()
    end = (windows.target_month + 1).to_timestamp()
    return ((dates >= start) & (dates < end)).to_numpy()


def validation_mask(dates: pd.Series, windows: FoldWindows) -> np.ndarray:
    """Return the ``[O-12, O)`` validation mask (whole months move together)."""
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    return (
        (dates >= windows.validation_start) & (dates < windows.validation_end)
    ).to_numpy()


def fit_mask(dates: pd.Series, windows: FoldWindows) -> np.ndarray:
    """Return the horizon-isolated initial-fit mask ``[outer_start, V_start - H)``."""
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    return ((dates >= windows.outer_start) & (dates < windows.fit_cutoff)).to_numpy()


def gap_mask(dates: pd.Series, windows: FoldWindows) -> np.ndarray:
    """Return the withheld purge band ``[V_start - H, V_start)``."""
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    return (
        (dates >= windows.fit_cutoff) & (dates < windows.validation_start)
    ).to_numpy()


def group_eligibility_mask(outer_groups: np.ndarray, target_groups: np.ndarray) -> np.ndarray:
    """Reproduce the main splitter's 'train groups must appear in test' filter."""
    return np.isin(np.asarray(outer_groups), np.unique(np.asarray(target_groups)))


def describe_observed_months(dates: pd.Series, mask: np.ndarray) -> Dict[str, object]:
    """Summarise the distinct observed label months selected by ``mask``."""
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    selected = dates[np.asarray(mask)]
    months = sorted({str(value.to_period("M")) for value in selected})
    return {"n_rows": int(len(selected)), "n_months": len(months), "months": months}
