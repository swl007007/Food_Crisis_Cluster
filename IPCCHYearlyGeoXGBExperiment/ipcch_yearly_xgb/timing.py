"""P0 bounded synthetic timing (design section 8). Synthetic data only; no project data is fitted.

For each H's frozen recipe, one weighted global quartet at the smallest and
largest enumerated global pool sizes, and one weighted L2 local continuation
at a small and the largest enumerated local pool size; fit, save, reload and
predict are timed once each. The full-run estimate interpolates linearly in n
over the enumerated pools (21 global + 160 local quartets).
"""

from __future__ import annotations

import time

import numpy as np

from ipcch_yearly_xgb import quartet, schedule


def _data(n: int, seed: int = 20261007):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 561))
    X[rng.random((n, 561)) < 0.4] = np.nan
    base = 0.25 + 0.1 * np.nan_to_num(X[:, 0]) - 0.05 * np.nan_to_num(X[:, 1])
    Y = np.column_stack([np.clip(base + 0.2, 0, 1), np.clip(base, 0, 1), np.clip(base * 0.4, 0, 1),
                         np.clip(base * 0.05, 0, 1)])
    t = 24000 + rng.integers(0, 120, size=n)
    w = schedule.decay_weights(t, 24000 + 120, 24)
    return X, Y, w


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def probe(contract: dict, enumeration: dict) -> dict:
    out = {"cases": [], "estimate_seconds": {}}
    total = 0.0
    for h, e in enumeration.items():
        recipe = contract["recipes"][h]
        gp, gr = quartet.global_params(contract, recipe[:2])
        lp, lr = quartet.local_params(contract, recipe[2:])
        gsizes = sorted(e["global_pool_sizes"].values())
        lsizes = sorted(e["local_pool_sizes"].values())
        gpts, lpts = [], []
        for n in (gsizes[0], gsizes[-1]):
            X, Y, w = _data(n)
            gq, t_fit = _timed(lambda: quartet.fit_global_quartet(X, Y, w, gp, gr))
            nbytes = sum(len(v) for v in gq.payloads.values())
            _, t_pred = _timed(lambda: gq.predict_raw(X[:2000]))
            gpts.append((n, t_fit))
            out["cases"].append({"H": h, "role": "global", "recipe": recipe[:2], "n": n, "fit_seconds": t_fit,
                                 "predict_2000_seconds": t_pred, "bytes": nbytes})
            for m in (min(2000, lsizes[0]), lsizes[-1]):
                if m > n:
                    continue
                lq, t_l = _timed(lambda: quartet.continue_local_quartet(gq, X[:m], Y[:m], w[:m], lp, lr))
                lpts.append((m, t_l))
                out["cases"].append({"H": h, "role": "local", "recipe": recipe[2:], "n": m, "fit_seconds": t_l,
                                     "bytes": sum(len(v) for v in lq.payloads.values())})
        gx, gy = zip(*sorted(gpts))
        lx, ly = zip(*sorted(set(lpts)))
        est = sum(float(np.interp(n, gx, gy)) for n in gsizes) + sum(float(np.interp(n, lx, ly)) for n in lsizes)
        out["estimate_seconds"][h] = est
        total += est
    out["estimate_seconds"]["total_fit"] = total
    return out
