"""Dense month ordinals ``year * 12 + month - 1`` and ``YYYY-MM`` labels."""

from __future__ import annotations

import numpy as np


def month_ord(year, month) -> np.ndarray:
    return np.asarray(year, dtype=np.int64) * 12 + np.asarray(month, dtype=np.int64) - 1


def parse_month(label: str) -> int:
    year, month = (int(x) for x in str(label).split("-"))
    if not 1 <= month <= 12:
        raise ValueError(f"bad month label {label!r}")
    return year * 12 + month - 1


def month_label(ordinal) -> np.ndarray | str:
    """``YYYY-MM`` for a scalar or array; negative ordinals render empty."""
    if np.isscalar(ordinal):
        o = int(ordinal)
        return "" if o < 0 else f"{o // 12:04d}-{o % 12 + 1:02d}"
    arr = np.asarray(ordinal, dtype=np.int64)
    out = np.array([f"{y:04d}-{m:02d}" for y, m in zip(arr // 12, arr % 12 + 1)], dtype=object)
    out[arr < 0] = ""
    return out
