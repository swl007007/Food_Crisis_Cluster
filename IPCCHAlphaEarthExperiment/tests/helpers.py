"""Shared test fixtures: scaled-down contracts (never frozen) and tiny correction pools."""

import numpy as np

from ipcch_alphaearth import contract as ct


def small_contract(**overrides):
    base = ct.load_contract()
    fast = {f"backbone.recipes.{q}.n_estimators": 5 for q in ("q2", "q3", "q4", "q5")}
    fast.update({f"backbone.recipes.{q}.max_depth": 3 for q in ("q2", "q3", "q4", "q5")})
    fast["backbone.fixed.n_jobs"] = 2
    fast.update(overrides)
    return ct.scaled_copy(base, **fast)


def intercept_pool(residual: np.ndarray, admins: np.ndarray, months: np.ndarray, routable=None, weights=None):
    """Pool with an intercept-only design (all other slots zero/inactive), one residual column copied to all q."""
    from ipcch_alphaearth import partition as pt

    n = len(residual)
    V = np.zeros((n, 4, 8))
    V[:, :, 0] = 1.0
    R = np.repeat(np.asarray(residual, float)[:, None], 4, axis=1)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    routable = np.ones(n, bool) if routable is None else np.asarray(routable, bool)
    return pt.Pool(V, R, w, np.asarray(admins, np.int64), np.asarray(months, np.int64), routable)


def intercept_active():
    a = np.zeros((4, 8), bool)
    a[:, 0] = True
    return a
