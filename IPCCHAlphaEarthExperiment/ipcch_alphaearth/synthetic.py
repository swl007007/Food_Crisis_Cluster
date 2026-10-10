"""Synthetic inputs on the real calendar for end-to-end tests and the compute benchmark (no project data).

Admins carry random 128-dim descriptors (a share unroutable/NaN). A persistent AR(1) shock x(admin, month) drives a
latent crisis score with a region-specific slope when ``regional=True`` (region = descriptor 0 > 0), so a weak pooled
backbone leaves slope heterogeneity in its residuals for the supervised partition to find. Row inputs at origin
O = T - H use x(admin, O) as the 'precipitation' core input; truth uses x(admin, T). Shares are monotone in k.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ipcch_alphaearth.months import month_label, parse_month
from ipcch_alphaearth.pipeline import HorizonInputs, Inputs
from ipcch_alphaearth.prepare import persistence

CORE = ("prcp_anom_month_ensmean__ma3_at_origin", "tmean_anom_month_ensmean__ma3_at_origin", "event_count_violence__ma3_at_origin")


def make_inputs(contract: dict, n_admin: int = 300, n_countries: int = 12, seed: int = 7, regional: bool = True,
                p_obs_late: float = 0.22, p_obs_early: float = 0.06, unroutable_share: float = 0.1, n_noise: int = 7,
                n_desc: int = 128) -> Inputs:
    rng = np.random.default_rng(seed)
    first, last = parse_month("2013-01"), parse_month("2026-04")
    months = np.arange(first, last + 1)
    admins = np.arange(1000, 1000 + n_admin, dtype=np.int64)
    country = np.array([f"C{c:02d}" for c in rng.integers(0, n_countries, n_admin)], dtype=object)
    D = rng.normal(size=(n_admin, n_desc))
    routable = rng.uniform(size=n_admin) >= unroutable_share
    D[~routable] = np.nan
    region = np.where(routable, D[:, 0] > 0, rng.uniform(size=n_admin) > 0.5)
    slope = np.where(region, 0.9, -0.5) if regional else np.full(n_admin, 0.2)
    base = rng.normal(0.0, 0.6, n_admin)
    x = np.zeros((n_admin, len(months)))
    for j in range(1, len(months)):
        x[:, j] = 0.92 * x[:, j - 1] + rng.normal(0, 0.45, n_admin)
    t2 = rng.normal(size=(n_admin, len(months)))
    viol = rng.poisson(1.0, size=(n_admin, len(months))).astype(float)
    obs = []
    for ai in range(n_admin):
        for j, m in enumerate(months):
            if m < parse_month("2014-01"):
                continue
            p = p_obs_early if m < parse_month("2018-01") else p_obs_late
            if rng.uniform() < p:
                obs.append((ai, j, m))
    obs = np.array(obs)
    ai, ji, t = obs[:, 0], obs[:, 1], obs[:, 2]
    latent = base[ai] + slope[ai] * x[ai, ji] + 0.15 * viol[ai, ji] + rng.normal(0, 0.35, len(ai))
    cuts = np.array([-0.8, 0.0, 0.9, 1.8])
    q = 1.0 / (1.0 + np.exp(-(latent[:, None] - cuts[None, :]) * 2.2))
    q = np.minimum.accumulate(q, axis=1)
    phase = 1 + (q >= 0.20).sum(axis=1)
    valid = pd.DataFrame({"admin_code": admins[ai], "month_ord": t, "phase_truth": phase, "q3": q[:, 1]})
    valid = valid.sort_values(["admin_code", "month_ord"], kind="mergesort").reset_index(drop=True)
    order = np.lexsort((t, admins[ai]))
    ai, ji, t, q, phase = ai[order], ji[order], t[order], q[order], phase[order]
    features = list(CORE) + [f"noise_{k}" for k in range(n_noise)]
    horizons = {}
    for h in contract["calendar"]["horizons_months"]:
        o = t - h
        oj = o - first
        prcp = x[ai, oj] + rng.normal(0, 0.05, len(ai))
        prcp[rng.uniform(size=len(ai)) < 0.03] = np.nan
        tmean = t2[ai, oj]
        vio = viol[ai, oj]
        noise = rng.normal(size=(len(ai), n_noise))
        X = np.column_stack([prcp, tmean, vio, noise])
        keys = pd.DataFrame({"admin_code": admins[ai], "country_key": country[ai], "horizon": h, "target_ord": t,
                             "target_month": month_label(t), "origin_ord": o, "origin_month": month_label(o),
                             "phase_truth": phase, "crisis_truth": (phase >= 3).astype(np.int64),
                             "q2": q[:, 0], "q3": q[:, 1], "q4": q[:, 2], "q5": q[:, 3]})
        keys = pd.concat([keys, persistence(valid, keys["admin_code"].to_numpy(np.int64), o)], axis=1)
        horizons[h] = HorizonInputs(h, keys, X, D[ai], routable[ai], [0, 1, 2])
    names = [f"ae2017_mean_A{j:02d}" for j in range(n_desc // 2)] + [f"ae2017_sd_A{j:02d}" for j in range(n_desc - n_desc // 2)]
    return Inputs(horizons, features, features + names, admins, D, {"synthetic_seed": seed, "regional": regional}, names)
