"""Bounded monotone projection and unrounded decoding of the q2..q5 quartet (R15).

``q_star = argmin_z sum_k (z_k - q_raw_k)^2`` subject to
``1 >= z2 >= z3 >= z4 >= z5 >= 0``, per row, equal weights, using only that
row's raw predictions.

Method. Pool-adjacent-violators (PAVA) for a non-increasing fit: adjacent
blocks that violate the order are merged and replaced by the mean of their
original values, computed with ``math.fsum`` (exactly rounded); if that sum
would overflow float64 the mean is taken as ``fsum(v / n)`` instead, so every
finite float64 input has a finite result (an SSE-ranking of candidate
partitions loses the comparison when |q_raw| is huge). Rows already
non-increasing are returned unchanged. With constant bounds [0, 1] on every
coordinate, the bounded optimum is that isotonic solution clipped to [0, 1].
Clipping *before* the isotonic step is not equivalent and is not used.

Decoding: ``phase = max({1} | {k : q_star_k >= 0.20})`` with no rounding;
after projection this is ``1 + #{k : q_star_k >= 0.20}``.
"""

from __future__ import annotations

import math

import numpy as np

from ipcch_geoxgb.errors import TechnicalError

THRESHOLD = 0.20
N_TARGETS = 4


def _block_mean(values: list[float]) -> float:
    try:
        return math.fsum(values) / len(values)
    except OverflowError:  # |sum| beyond float64: scale first (each term then fits)
        n = len(values)
        return math.fsum(v / n for v in values)


def _pava_row(values: list[float]) -> list[float]:
    blocks: list[list[float]] = []  # each block: its original values
    means: list[float] = []
    for value in values:
        blocks.append([value])
        means.append(value)
        while len(means) > 1 and means[-2] < means[-1]:  # non-increasing violated
            merged = blocks[-2] + blocks[-1]
            blocks[-2:] = [merged]
            means[-2:] = [_block_mean(merged)]
    out: list[float] = []
    for block, mean in zip(blocks, means):
        out.extend([mean] * len(block))
    return out


def isotonic_decreasing(raw: np.ndarray) -> np.ndarray:
    """Row-wise least-squares non-increasing fit of an (n, 4) array (unbounded)."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != N_TARGETS:
        raise TechnicalError(f"projection input shape {raw.shape}, expected (n, {N_TARGETS})")
    if not np.isfinite(raw).all():
        raise TechnicalError("projection input contains NaN/Inf (R41)")
    out = raw.copy()
    violating = np.flatnonzero(np.any(np.diff(raw, axis=1) > 0.0, axis=1))
    for row in violating:
        out[row] = _pava_row(raw[row].tolist())
    return out


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
