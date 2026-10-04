"""Bounded monotone projection and unrounded decoding of the q2..q5 quartet (R15).

``q_star = argmin_z sum_k (z_k - q_raw_k)^2`` subject to
``1 >= z2 >= z3 >= z4 >= z5 >= 0``, per row, equal weights, using only that
row's raw predictions.

Method. Without bounds, the least-squares non-increasing fit of four ordered
values is the block-mean vector of some partition of (z2..z5) into contiguous
blocks (pool-adjacent-violators). There are 2**3 = 8 such partitions; the
optimum is the feasible (non-increasing) block-mean vector with the smallest
squared error, found exactly by enumeration. With constant bounds [0, 1] on
every coordinate, the bounded optimum is that isotonic solution clipped to
[0, 1]. Clipping *before* the isotonic step is not equivalent and is not used.

Decoding: ``phase = max({1} | {k : q_star_k >= 0.20})`` with no rounding;
after projection this is ``1 + #{k : q_star_k >= 0.20}``.
"""

from __future__ import annotations

import itertools

import numpy as np

from ipcch_geoxgb.errors import TechnicalError

THRESHOLD = 0.20
N_TARGETS = 4


def _partitions(n: int = N_TARGETS) -> list[list[tuple[int, int]]]:
    """All partitions of range(n) into contiguous [start, stop) blocks."""
    out = []
    for cuts in itertools.product((False, True), repeat=n - 1):
        blocks, start = [], 0
        for i, cut in enumerate(cuts, start=1):
            if cut:
                blocks.append((start, i))
                start = i
        blocks.append((start, n))
        out.append(blocks)
    return out


_PARTITIONS = _partitions()


def isotonic_decreasing(raw: np.ndarray) -> np.ndarray:
    """Row-wise least-squares non-increasing fit of an (n, 4) array (unbounded)."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != N_TARGETS:
        raise TechnicalError(f"projection input shape {raw.shape}, expected (n, {N_TARGETS})")
    if not np.isfinite(raw).all():
        raise TechnicalError("projection input contains NaN/Inf (R41)")
    best = np.full(raw.shape, np.nan)
    best_sse = np.full(raw.shape[0], np.inf)
    for blocks in _PARTITIONS:
        candidate = np.empty_like(raw)
        for start, stop in blocks:
            candidate[:, start:stop] = raw[:, start:stop].mean(axis=1, keepdims=True)
        feasible = np.all(np.diff(candidate, axis=1) <= 0.0, axis=1)
        sse = ((candidate - raw) ** 2).sum(axis=1)
        better = feasible & (sse < best_sse)
        best[better] = candidate[better]
        best_sse[better] = sse[better]
    if np.isnan(best).any():  # the all-pooled partition is always feasible
        raise TechnicalError("isotonic projection found no feasible partition")
    return best


def project(raw: np.ndarray) -> np.ndarray:
    """Bounded projection onto 1 >= z2 >= z3 >= z4 >= z5 >= 0."""
    z = np.clip(isotonic_decreasing(raw), 0.0, 1.0)
    if not (np.isfinite(z).all() and np.all(np.diff(z, axis=1) <= 0.0)):
        raise TechnicalError("projected quartet violates the bounded monotone constraint")
    return z


def decode(q_star: np.ndarray) -> np.ndarray:
    """Five-level phase from projected shares, unrounded, inclusive 0.20."""
    q_star = np.asarray(q_star, dtype=np.float64)
    if q_star.ndim != 2 or q_star.shape[1] != N_TARGETS:
        raise TechnicalError(f"decode input shape {q_star.shape}")
    reached = q_star >= THRESHOLD
    phase = np.ones(len(q_star), dtype=np.int64)
    for k in range(N_TARGETS):  # q2..q5 -> phases 2..5
        phase = np.where(reached[:, k], k + 2, phase)
    return phase


def project_and_decode(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_star = project(raw)
    return q_star, decode(q_star)
