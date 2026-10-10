"""Exact-count metrics, D23 guards, D13 recipe selection, D22 per-H decisions and the D24 paired country bootstrap.

Classification scores are exact rationals from pooled unweighted confusion counts (fitting weights never enter
evaluation); a class absent from truth and prediction makes macro-F1 undefined (no zero fill). Projected q3 R^2
uses the P6 ``metrics.r_squared`` (raw R^2 is diagnostic only). Undefined required scores never pass a guard.
Bootstrap structure adapted from IPCCHGeoXGBExperiment/ipcch_geoxgb/report.py (bootstrap_delta) at 6798df2 with
the D24 constants (20,000 draws, default_rng(42) reset per H, linear 1.25th percentile, no drop/redraw).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd

from ipcch_alphaearth import metrics
from ipcch_alphaearth.targets import four_class

GUARDED = ("four_class.accuracy", "four_class.macro_f1", "binary.accuracy", "binary.precision", "binary.recall", "binary.f2")


def exact_scores(truth_phase, pred_phase) -> dict:
    """Seven classification scores as Fractions (None when undefined) plus counts."""
    t = np.asarray(truth_phase, dtype=np.int64)
    p = np.asarray(pred_phase, dtype=np.int64)
    c = metrics.crisis_counts(t, p)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]

    def ratio(num, den):
        return Fraction(num, den) if den else None

    out = {"counts": c, "n": int(len(t)),
           "binary.accuracy": ratio(tp + tn, tp + fp + fn + tn), "binary.precision": ratio(tp, tp + fp),
           "binary.recall": ratio(tp, tp + fn), "binary.f1": ratio(2 * tp, 2 * tp + fp + fn),
           "binary.f2": ratio(5 * tp, 5 * tp + 4 * fn + fp)}
    if len(t):
        m = np.bincount(four_class(t) * 4 + four_class(p), minlength=16).reshape(4, 4)
    else:
        m = np.zeros((4, 4), dtype=np.int64)
    diag = np.diag(m)
    f1s = []
    for k in range(4):
        den = int(2 * diag[k] + (m[:, k].sum() - diag[k]) + (m[k].sum() - diag[k]))
        f1s.append(Fraction(2 * int(diag[k]), den) if den else None)
    out["four_class.accuracy"] = ratio(int(diag.sum()), int(m.sum()))
    out["four_class.macro_f1"] = None if any(f is None for f in f1s) else sum(f1s) / 4
    out["four_class.confusion"] = m.tolist()
    reasons = {}
    for name, den in (("binary.accuracy", tp + fp + fn + tn), ("binary.precision", tp + fp), ("binary.recall", tp + fn),
                      ("binary.f1", 2 * tp + fp + fn), ("binary.f2", 5 * tp + 4 * fn + fp), ("four_class.accuracy", int(m.sum()))):
        if den == 0:
            reasons[name] = "zero denominator"
    absent = [("1", "2", "3", "4/5")[k] for k, f in enumerate(f1s) if f is None]
    if absent:
        reasons["four_class.macro_f1"] = f"class absent from truth and prediction: {absent}"
    out["na_reasons"] = reasons
    return out


def panel(truth_phase, pred_phase, q3_true, q3_star, q3_raw) -> dict:
    s = exact_scores(truth_phase, pred_phase)
    r2p, why_p = metrics.r_squared(q3_true, q3_star) if len(q3_true) else (None, "no keys")
    r2r, why_r = metrics.r_squared(q3_true, q3_raw) if len(q3_true) else (None, "no keys")
    s.update({"q3_r2_projected": r2p, "q3_r2_projected_na": why_p, "q3_r2_raw": r2r, "q3_r2_raw_na": why_r})
    return s


def as_float(x):
    return None if x is None else float(x)


def flat(p: dict) -> dict:
    keys = ("binary.f1", *GUARDED)
    out = {k: as_float(p[k]) for k in keys}
    out.update({"q3_r2_projected": p["q3_r2_projected"], "q3_r2_raw": p["q3_r2_raw"], "n": p["n"]})
    return out


def detail(p: dict) -> dict:
    """Machine-readable evidence for one panel: counts, confusion, exact values and a reason per undefined metric."""
    keys = ("binary.f1", *GUARDED)
    reasons = dict(p.get("na_reasons", {}))
    for name in ("q3_r2_projected", "q3_r2_raw"):
        if p.get(name) is None:
            reasons[name] = p.get(f"{name}_na") or "undefined"
    return {"n": p["n"], "binary_counts": p["counts"], "four_class_confusion_rows_truth_cols_pred": p["four_class.confusion"],
            "exact": {k: None if p[k] is None else str(p[k]) for k in keys},
            "q3_r2_projected": p.get("q3_r2_projected"), "q3_r2_raw": p.get("q3_r2_raw"), "na_reasons": reasons}


def guard_check(g: dict, comparators: dict, contract: dict) -> dict:
    """D23 point guards. ``comparators``: {"B1": panel, "B2": panel, "B0": panel}. Undefined -> fail with reason."""
    ev = contract["evaluation"]
    tol_c, tol_r = Fraction(ev["classification_guard"]), float(ev["projected_r2_guard"])
    items = []

    def add(name, ok, reason=""):
        items.append({"guard": name, "pass": bool(ok), "reason": reason})

    for ref in ("B1", "B0"):
        a, b = g["binary.f1"], comparators[ref]["binary.f1"]
        add(f"f1_G>={ref}", a is not None and b is not None and a >= b, "" if a is not None and b is not None else "undefined F1")
    for ref in ("B1", "B2"):
        for m in GUARDED:
            a, b = g[m], comparators[ref][m]
            if a is None or b is None:
                add(f"{m}_vs_{ref}", False, "undefined")
            else:
                add(f"{m}_vs_{ref}", a - b >= -tol_c)
        a, b = g["q3_r2_projected"], comparators[ref]["q3_r2_projected"]
        if a is None or b is None:
            add(f"q3_r2_projected_vs_{ref}", False, "undefined")
        else:
            add(f"q3_r2_projected_vs_{ref}", a - b >= -tol_r)
    return {"pass": all(i["pass"] for i in items), "items": items,
            "failed": [i["guard"] for i in items if not i["pass"]]}


def f1_gain(g: dict, b2: dict):
    a, b = g["binary.f1"], b2["binary.f1"]
    return None if a is None or b is None else a - b


# --------------------------------------------------------------------------- D13 selection


def select_recipe(records: list[dict]) -> dict:
    """``records``: one per recipe with keys recipe_id, leaf_budget, lambda_region, years {y: {gain, guards_pass,
    n_leaves, depth}}. Returns the selection record (or B2 when none is eligible)."""
    table = []
    for r in records:
        years = r["years"]
        gains = [years[y]["gain"] for y in sorted(years)]
        eligible = all(g is not None and g >= 0 for g in gains) and all(years[y]["guards_pass"] for y in years)
        mean = None if any(g is None for g in gains) else sum(gains) / len(gains)
        eligible = eligible and mean is not None and mean > 0
        table.append({**r, "mean_gain": mean, "eligible": eligible,
                      "sum_leaves": sum(years[y]["n_leaves"] for y in years), "max_depth": max(years[y]["depth"] for y in years)})
    ok = [t for t in table if t["eligible"]]
    if not ok:
        return {"selected": "B2", "reason": "no eligible recipe (D13/D23)", "table": table}
    best = min(ok, key=lambda t: (-t["mean_gain"], t["sum_leaves"], t["max_depth"], t["leaf_budget"], -t["lambda_region"], t["recipe_id"]))
    return {"selected": best["recipe_id"], "leaf_budget": best["leaf_budget"], "lambda_region": best["lambda_region"],
            "mean_gain": str(best["mean_gain"]), "reason": "largest eligible equal-year mean gain", "table": table}


# --------------------------------------------------------------------------- D24 paired country bootstrap


def country_counts(frame: pd.DataFrame, phase_col: str, countries: list) -> np.ndarray:
    rows = []
    for c in countries:
        part = frame[frame["country_key"] == c]
        cnt = metrics.crisis_counts(part["phase_truth"].to_numpy(), part[phase_col].to_numpy())
        rows.append([cnt["tp"], cnt["fp"], cnt["fn"], cnt["tn"]])
    return np.array(rows, dtype=np.int64).reshape(len(countries), 4)


def _f1_rows(counts: np.ndarray) -> np.ndarray:
    tp, fp, fn = counts[:, 0], counts[:, 1], counts[:, 2]
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), np.nan)


def paired_bootstrap(frame: pd.DataFrame, col_a: str, col_b: str, contract: dict) -> tuple[dict, pd.DataFrame | None]:
    """Whole-country paired bootstrap of crisis F1(a) - F1(b); lower bound = linear 1.25th percentile."""
    bs = contract["evaluation"]["bootstrap"]
    draws_n, seed, pct = int(bs["draws"]), int(bs["seed"]), float(bs["lower_percentile"])
    countries = sorted(frame["country_key"].unique().tolist())
    k = len(countries)
    fa = metrics.exact_f1(metrics.crisis_counts(frame["phase_truth"].to_numpy(), frame[col_a].to_numpy()))
    fb = metrics.exact_f1(metrics.crisis_counts(frame["phase_truth"].to_numpy(), frame[col_b].to_numpy()))
    point = None if fa is None or fb is None else fa - fb
    record = {"countries": countries, "K": k, "draws": draws_n, "seed": seed, "rng": "numpy.random.default_rng",
              "percentile": pct, "percentile_method": bs["percentile_method"], "point_delta": None if point is None else float(point),
              "point_delta_exact": None if point is None else str(point)}
    if k == 0:
        return {**record, "lower_bound": None, "na_reason": "empty cohort"}, None
    ca, cb = country_counts(frame, col_a, countries), country_counts(frame, col_b, countries)
    rng = np.random.default_rng(seed)
    mult = np.zeros((draws_n, k), dtype=np.int64)
    for d in range(draws_n):
        mult[d] = np.bincount(rng.integers(0, k, size=k), minlength=k)
    f_a, f_b = _f1_rows(mult @ ca), _f1_rows(mult @ cb)
    delta = f_a - f_b
    draws = pd.DataFrame(mult, columns=[f"m::{c}" for c in countries])
    draws.insert(0, "delta", delta)
    draws.insert(0, f"f1_{col_b}", f_b)
    draws.insert(0, f"f1_{col_a}", f_a)
    draws.insert(0, "draw", np.arange(draws_n))
    defined = int(np.isfinite(delta).sum())
    record.update(defined_draws=defined, undefined_draws=draws_n - defined, country_counts={col_a: ca.tolist(), col_b: cb.tolist()})
    reasons = []
    if k < int(bs["min_countries"]):
        reasons.append("K < 2")
    if point is None:
        reasons.append("point delta undefined")
    if defined < draws_n:
        reasons.append(f"{draws_n - defined} undefined replicate deltas")
    if reasons:
        return {**record, "lower_bound": None, "na_reason": "; ".join(reasons)}, draws
    lower = float(np.percentile(delta, pct, method=bs["percentile_method"]))
    descriptive = [float(x) for x in np.percentile(delta, (2.5, 97.5), method="linear")]
    return {**record, "lower_bound": lower, "descriptive_uncorrected_95": descriptive, "na_reason": ""}, draws


# --------------------------------------------------------------------------- D22/D23 per-H decision


def horizon_decision(main: pd.DataFrame, panels: dict, years: dict, boot: dict, contract: dict) -> dict:
    """Per-H report: effect, corrected bound, annual signs, guards; distinguishes the failure modes (D22/D23/D28)."""
    ev = contract["evaluation"]
    gain = f1_gain(panels["G"], panels["B2"])
    floor = Fraction(ev["main_f1_gain_minimum"])
    annual = {}
    for y, p in sorted(years.items()):
        g = f1_gain(p["G"], p["B2"])
        annual[str(y)] = {"gain": None if g is None else str(g), "gain_float": as_float(g),
                          "nonnegative": None if g is None else bool(g >= 0)}
    guards = guard_check(panels["G"], {"B1": panels["B1"], "B2": panels["B2"], "B0": panels["B0"]}, contract)
    c1 = gain is not None and gain >= floor
    c2 = boot.get("lower_bound") is not None and boot["lower_bound"] > 0
    c3 = all(a["nonnegative"] is True for a in annual.values()) and len(annual) > 0
    primary = c1 and c2 and c3
    if gain is None:
        status = "undefined_point_estimate"
    elif primary and guards["pass"]:
        status = "protected_stable_benefit"
    elif primary:
        status = "primary_met_guard_tradeoff"
    elif boot.get("lower_bound") is None:
        status = "inconclusive_bound_undefined"
    elif any(a["nonnegative"] is None for a in annual.values()):
        status = "inconclusive_annual_undefined"
    elif gain <= 0:
        status = "no_gain_or_reversal"
    elif not c1:
        status = "gain_below_practical_floor"
    elif not c2:
        status = "imprecise_bound_not_above_zero"
    else:
        status = "annual_sign_reversal"
    return {"f1_gain": None if gain is None else str(gain), "f1_gain_float": as_float(gain),
            "criterion_gain_floor": c1, "criterion_corrected_lower_bound": c2, "criterion_annual_signs": c3,
            "lower_bound": boot.get("lower_bound"), "bound_na_reason": boot.get("na_reason"),
            "annual": annual, "guards": guards, "status": status}
