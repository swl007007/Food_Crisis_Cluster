"""P0 compute benchmark on synthetic data only (no project data fitted, EE never contacted).

1. Supervised search at representative production dimensions (pool rows, mapped admins, 128 descriptors, real
   D20 support floors, eight leaves, depth 3) for one lambda; timed per growth path and extrapolated to the
   24 development + <= 4 final paths.
2. XGB quartets with the frozen D21 recipes (n_jobs 16, hist) on synthetic matrices of the largest and a smaller
   real B1 pool shape (424 columns, ~7% NaN); per-row rate extrapolated over the 61 enumerated quartet pools.
BLAS threads are pinned (threadpoolctl) so ridge batches and XGB never oversubscribe the 32 logical cores.
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_alphaearth import backbone, correction as cr, partition as pt
from ipcch_alphaearth.artifacts import write_json
from ipcch_alphaearth.contract import load_contract

BLAS_THREADS = 16


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def synthetic_pool(n_rows: int, n_admin: int, n_months: int, seed: int = 0, routable_share: float = 0.9):
    rng = np.random.default_rng(seed)
    admin = np.sort(rng.integers(0, n_admin, n_rows))
    months = rng.integers(0, n_months, n_rows)
    P = rng.uniform(0, 1, (n_rows, 4))
    Z = rng.normal(size=(n_rows, 3))
    Z[rng.uniform(size=Z.shape) < 0.05] = np.nan
    D = rng.normal(size=(n_admin, 128))
    # eight response groups on three descriptors force full best-first growth (worst case for the search)
    s0, s1, s2 = (np.where(D[:, j] > 0, 1.0, -1.0) for j in range(3))
    level = 0.30 * s0 + 0.20 * s1 + 0.15 * s2  # every tree level separates a sizeable deviation
    slope = 0.20 * s0 * s1 + 0.15 * s2
    R = 0.05 * rng.normal(size=(n_rows, 4)) + slope[admin][:, None] * np.nan_to_num(Z[:, :1]) + level[admin][:, None]
    w = np.power(2.0, -(n_months - 1 - months) / 24.0)
    t = cr.Transform.fit(P, Z, w)
    routable_admin = rng.uniform(size=n_admin) < routable_share
    pool = pt.Pool(t.design(P, Z), R, w, admin, months, routable_admin[admin])
    return pool, t.active(), D, np.arange(n_admin)


def bench_search(contract: dict, n_rows: int, n_admin: int, n_months: int, lam: float | None = None) -> dict:
    from threadpoolctl import threadpool_limits

    pool, active, D, codes = synthetic_pool(n_rows, n_admin, n_months)
    lam = float(contract["correction"]["regional_lambdas"][0]) if lam is None else float(lam)
    with threadpool_limits(BLAS_THREADS):
        t0 = time.time()
        search = pt.Search(pool, D, codes, active, contract, lam)
        states = search.run(8, int(contract["correction"]["max_depth"]))
        seconds = time.time() - t0
    ledger = search.ledger_frame()
    return {"rows": n_rows, "admins": n_admin, "months": n_months, "lambda_region": lam, "seconds_per_path": round(seconds, 2),
            "leaves_reached": states[-1]["map"].n_leaves, "depth_reached": states[-1]["map"].depth, "candidates_scored": int(len(ledger)),
            "supported_candidates": int(ledger["supported"].sum()), "peak_rss_gb": round(peak_rss_gb(), 2),
            "ledger_bytes_csv_estimate": int(len(ledger) * 230)}


def bench_xgb(contract: dict, n_rows: int, n_features: int = 424, seed: int = 1) -> dict:
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features))
    X[rng.uniform(size=X.shape) < 0.07] = np.nan
    Y = np.clip(rng.uniform(size=(n_rows, 4)) * np.array([1.0, 0.8, 0.4, 0.1]), 0, 1)
    w = rng.uniform(0.2, 1.0, n_rows)
    out = {"rows": n_rows, "features": n_features, "seconds": {}}
    with threadpool_limits(BLAS_THREADS):
        total = 0.0
        for k, q in enumerate(backbone.TARGETS):
            t0 = time.time()
            booster = backbone.fit_scalar(X, Y[:, k], w, backbone.xgb_params(contract, q))
            out["seconds"][q] = round(time.time() - t0, 2)
            total += time.time() - t0
            out.setdefault("ubj_bytes", {})[q] = len(bytes(booster.save_raw("ubj")))
    out["quartet_seconds"] = round(total, 2)
    out["peak_rss_gb"] = round(peak_rss_gb(), 2)
    return out


def run_benchmark(out_dir: Path, prepared_dir: Path | None = None) -> dict:
    contract = load_contract()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    search = [bench_search(contract, 15700, 4430, 60), bench_search(contract, 3700, 2065, 13)]
    xgb = [bench_xgb(contract, 38603), bench_xgb(contract, 10000)]
    rate = xgb[0]["quartet_seconds"] / xgb[0]["rows"]
    estimate = {"xgb_rate_seconds_per_row_quartet": rate}
    if prepared_dir is not None and (Path(prepared_dir) / "schedule_requests.csv.gz").exists():
        req = pd.read_csv(Path(prepared_dir) / "schedule_requests.csv.gz")
        fixed = xgb[1]["quartet_seconds"] - rate * xgb[1]["rows"]
        est = (req["fit_rows"] * rate + max(fixed, 0.0)).sum()
        estimate.update({"quartets": int(len(req)), "total_fit_rows": int(req["fit_rows"].sum()),
                         "xgb_total_hours_estimate": round(est / 3600, 2),
                         "store_bytes_estimate": int(len(req) * sum(xgb[0]["ubj_bytes"].values()))})
    paths = 24 + 4
    estimate["search_total_hours_upper_estimate"] = round(paths * search[0]["seconds_per_path"] / 3600, 2)
    result = {"note": "synthetic data only; estimates are not measured production costs", "blas_threads": BLAS_THREADS,
              "search": search, "xgb": xgb, "estimate": estimate, "elapsed_seconds": round(time.time() - t0, 1)}
    write_json(out_dir / f"benchmark-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json", result)
    return result


def run_search_supplement(out_dir: Path) -> dict:
    """Search-only supplement: an eight-group three-descriptor response fixture that genuinely grows to eight leaves /
    depth 3 under the unchanged production support floors and penalties (the synthetic response strength/pattern is
    the only change). Extrapolation is conservative: every development and final path is charged a full eight-leaf
    growth at the larger representative size."""
    contract = load_contract()
    runs = [bench_search(contract, r, a, m, lam) for r, a, m in ((15700, 4430, 60), (3700, 2065, 13))
            for lam in contract["correction"]["regional_lambdas"]]
    full = [x for x in runs if x["leaves_reached"] == 8 and x["depth_reached"] == 3]
    worst = max(x["seconds_per_path"] for x in full) if full else None
    paths = {"development": 4 * 2 * 3, "final_map": 4}
    result = {"note": "synthetic search supplement; production recipe, support floors and penalties unchanged",
              "fixture": "level = .30*s0 + .20*s1 + .15*s2 and slope = .20*s0*s1 + .15*s2 on signs of descriptors 0-2; noise sd .05",
              "original_measurement_scope": "benchmark-20261010T204213Z.json search rows stopped at 2 leaves (single-effect fixture): "
                                            "early-stop scope only, not an eight-leaf cost",
              "runs": runs, "full_growth_runs": len(full), "paths": paths,
              "conservative_total_seconds": None if worst is None else round(worst * sum(paths.values()), 1),
              "conservative_total_minutes": None if worst is None else round(worst * sum(paths.values()) / 60, 2),
              "replay_note": "replay re-solves accepted states only (<= 7 joint solves per path), far below a full path",
              "peak_rss_gb": round(peak_rss_gb(), 2)}
    write_json(Path(out_dir) / f"benchmark-search-supplement-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json", result)
    return result
