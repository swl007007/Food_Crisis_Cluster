"""Fitting-only preprocessing (PRD R12, design section 4).

Per lawful global fitting pool: float64 medians of observed values (0 for
training-all-missing columns), mean/population-SD after imputation (scale 1
exactly where the variance is zero), all 561 columns kept. Output is
``[standardized values, unscaled missingness flags]`` as contiguous float32
(n, 1122). Never refit on regional, validation or test rows.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass

import numpy as np

from ipcch_mlp.errors import TechnicalError

N_FEATURES = 561


@dataclass(frozen=True)
class Transform:
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    all_missing: np.ndarray
    n_fit: int

    def digest(self) -> str:
        h = hashlib.sha256()
        for name in ("medians", "means", "scales"):
            arr = np.ascontiguousarray(getattr(self, name), dtype=np.float64)
            h.update(name.encode())
            h.update(arr.tobytes())
        h.update(np.ascontiguousarray(self.all_missing, dtype=np.uint8).tobytes())
        h.update(str(self.n_fit).encode())
        return h.hexdigest()

    def to_arrays(self) -> dict:
        return {"medians": self.medians, "means": self.means, "scales": self.scales,
                "all_missing": self.all_missing.astype(np.uint8), "n_fit": np.array([self.n_fit], dtype=np.int64)}

    @staticmethod
    def from_arrays(d: dict) -> "Transform":
        return Transform(np.asarray(d["medians"], np.float64), np.asarray(d["means"], np.float64),
                         np.asarray(d["scales"], np.float64), np.asarray(d["all_missing"]).astype(bool),
                         int(np.asarray(d["n_fit"])[0]))


def _check_raw(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise TechnicalError(f"raw feature matrix must be (n, {N_FEATURES}), got {X.shape}")
    if np.isinf(X).any():
        raise TechnicalError("raw feature matrix contains an infinity")
    return X


def fit_transform(X_fit: np.ndarray) -> Transform:
    X = _check_raw(X_fit)
    if len(X) == 0:
        raise TechnicalError("cannot fit a transform on an empty pool")
    observed = ~np.isnan(X)
    all_missing = ~observed.any(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(X, axis=0)
    medians = np.where(all_missing, 0.0, medians)
    imputed = np.where(observed, X, medians)
    means = imputed.mean(axis=0)
    sd = imputed.std(axis=0, ddof=0)
    scales = np.where(sd == 0.0, 1.0, sd)
    t = Transform(medians.astype(np.float64), means.astype(np.float64), scales.astype(np.float64),
                  all_missing, int(len(X)))
    for name in ("medians", "means", "scales"):
        if not np.isfinite(getattr(t, name)).all():
            raise TechnicalError(f"transform {name} is not finite")
    return t


def apply(t: Transform, X: np.ndarray) -> np.ndarray:
    X = _check_raw(X)
    missing = np.isnan(X)
    imputed = np.where(missing, t.medians, X)
    z = (imputed - t.means) / t.scales
    if not np.isfinite(z).all():
        raise TechnicalError("standardized values are not finite (float64)")
    out = np.ascontiguousarray(np.concatenate([z, missing.astype(np.float64)], axis=1), dtype=np.float32)
    if not np.isfinite(out).all():
        raise TechnicalError("model inputs are not finite after float32 conversion")
    return out


def unseen_missingness(t: Transform, X: np.ndarray) -> dict:
    """Diagnostic: observed values in training-all-missing columns (design section 4)."""
    X = np.asarray(X, dtype=np.float64)
    cols = np.flatnonzero(t.all_missing)
    if len(cols) == 0 or len(X) == 0:
        return {"columns": 0, "rows_affected": 0, "per_column": {}}
    seen = ~np.isnan(X[:, cols])
    return {"columns": int(len(cols)), "rows_affected": int(seen.any(axis=1).sum()),
            "per_column": {int(c): int(n) for c, n in zip(cols, seen.sum(axis=0)) if n}}
