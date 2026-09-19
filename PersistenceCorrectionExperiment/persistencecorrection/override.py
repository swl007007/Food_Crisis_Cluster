"""Layer 2: the persistence-correction override.

PRD R9/R10/R15. The override has exactly one free parameter, a single global
threshold per scope, and it is **structurally up-only**: :func:`apply_override`
starts from the persistence vector and can only ever write the value ``1`` into
it. There is no code path capable of emitting a ``1 -> 0`` flip, which is the
enforcement mechanism for R15 -- not a check that rejects one after the fact.
"""

from __future__ import annotations

import numpy as np


class OverrideContractError(RuntimeError):
    """Raised when the override contract is violated; the run must halt."""


def apply_override(persistence, p_cal, tau):
    """Return the 2-layer prediction.

    ``y = 1`` where ``persistence == 0 and p_cal > tau``; ``y = persistence``
    everywhere else. Only ``0 -> 1`` is expressible.
    """
    persistence = np.asarray(persistence)
    p_cal = np.asarray(p_cal, dtype=float)
    if persistence.shape != p_cal.shape:
        raise OverrideContractError(
            f"persistence {persistence.shape} and p_cal {p_cal.shape} must align"
        )
    if not np.isin(persistence, (0, 1)).all():
        raise OverrideContractError("persistence must be binary")
    if np.isnan(p_cal).any():
        raise OverrideContractError("p_cal must not contain NaN")

    out = persistence.astype(int).copy()
    # The only write. Restricted to rows where the base layer is 0, so the
    # assignment of 1 can never turn a 1 into a 0.
    out[(persistence == 0) & (p_cal > tau)] = 1
    return out


def flip_report(persistence, y_pred, y_true):
    """Account for every row the override changed."""
    persistence = np.asarray(persistence).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_true = np.asarray(y_true).astype(int)

    changed = y_pred != persistence
    up = changed & (persistence == 0)
    down = changed & (persistence == 1)
    if down.any():
        raise OverrideContractError(
            f"{int(down.sum())} rows flipped 1 -> 0; the override must be up-only"
        )
    fixed = int((up & (y_true == 1)).sum())
    damaged = int((up & (y_true == 0)).sum())
    n = int(up.sum())
    return {
        "flips": n,
        "flips_1_to_0": 0,
        "fixed": fixed,
        "damaged": damaged,
        "test_flip_precision": (fixed / n) if n else None,
    }
