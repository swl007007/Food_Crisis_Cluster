"""Annual calendar (design section 2): fit origins, current blocks and gate dates.

All dates are integer calendar-month ordinals (year*12 + month - 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ipcch_yearly_xgb.errors import TechnicalError


def parse_month(label: str) -> int:
    y, m = label.split("-")
    return int(y) * 12 + int(m) - 1


def month_label(o: int) -> str:
    return f"{o // 12:04d}-{o % 12 + 1:02d}"


def fit_origin(h: int, first_main_target: int, u: int) -> int:
    """F_H - H for months of F_H's year at/after F_H; otherwise January(year(U)) - H."""
    if u // 12 == first_main_target // 12 and u >= first_main_target:
        return first_main_target - h
    return (u // 12) * 12 - h


@dataclass(frozen=True)
class Block:
    block_id: str
    h: int
    period: str
    year: int
    anchor: int  # first scheduled target month of the block (truth or not)
    origin: int  # fit origin O = fit_origin(H, anchor)
    folds: tuple  # fold records (dicts) in target order


def blocks(calendar, h: int, first_main_target: int) -> list[Block]:
    """Group the scheduled folds of H by (period, target year)."""
    folds = calendar[calendar["horizon_months"] == h].sort_values(["target_ord", "period"])
    out = []
    for (period, year), part in folds.groupby([folds["period"], folds["target_ord"] // 12], sort=True):
        anchor = int(part["target_ord"].min())
        out.append(Block(block_id=f"{period[:4]}_h{h:02d}_{int(year)}", h=h, period=str(period), year=int(year),
                         anchor=anchor, origin=fit_origin(h, first_main_target, anchor),
                         folds=tuple(part.to_dict("records"))))
    return sorted(out, key=lambda b: (b.anchor, b.period))


def gate_dates(observed_months, origin: int, max_dates: int = 6) -> np.ndarray:
    """Latest up to six globally observed target months U < O, newest first."""
    months = np.unique(np.asarray(observed_months, dtype=np.int64))
    return months[months < origin][::-1][:max_dates]


def decay_weights(target_ord: np.ndarray, origin: int, half_life: int = 24) -> np.ndarray:
    age = origin - np.asarray(target_ord, dtype=np.int64)
    if (age < 0).any():
        raise TechnicalError("a fitting row is after the fit origin")
    return np.power(0.5, age.astype(np.float64) / float(half_life))
