"""Independent report recount for replay (no call into ``report``/``evaluation`` producers).

Rebuilds the canonical keyed cohort from the inputs and the frozen schedule, recounts every published panel with
its own counting/R^2/guard/status code, and regenerates the D24 bootstrap from canonical country counts. Each
function returns ``[(check_name, ok, detail)]``. Undefined scores compare as undefined (None) with a reason.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pandas as pd

from ipcch_alphaearth import projection

TARGETS = ("q2", "q3", "q4", "q5")
ARMS = ("B0", "B1", "B2", "G")
GUARDED = ("four_class.accuracy", "four_class.macro_f1", "binary.accuracy", "binary.precision", "binary.recall", "binary.f2")
CANONICAL = ["row", "admin_code", "country_key", "target_ord", "target_month", "origin_month", "phase_truth", "q2", "q3", "q4", "q5",
             "persistence_available", "persistence_phase", "persistence_q3", "persistence_source_month", "persistence_age_months",
             "period", "block_id", "routable"]


# --------------------------------------------------------------------------- canonical cohort


def canonical_block(hi, block) -> pd.DataFrame:
    rows = np.flatnonzero((hi.target >= block.target_lo) & (hi.target <= block.target_hi))
    k = hi.keys.iloc[rows].reset_index(drop=True)
    frame = pd.DataFrame({"row": rows})
    for c in CANONICAL[1:16]:
        frame[c] = k[c].to_numpy()
    frame["period"] = "supplement" if block.role == "supplement" else "main"
    frame["block_id"] = block.block_id
    frame["routable"] = hi.routable[rows].astype(np.int64)
    return frame


def projected_columns_ok(saved: pd.DataFrame) -> bool:
    for arm in ARMS:
        raw = saved[[f"{arm}_{q}_raw" for q in TARGETS]].to_numpy()
        star, phase = projection.project_and_decode(raw)
        if not (np.array_equal(star, saved[[f"{arm}_{q}_star" for q in TARGETS]].to_numpy())
                and np.array_equal(phase, saved[f"{arm}_phase"].to_numpy())):
            return False
    return True


# --------------------------------------------------------------------------- independent scores


def scores(truth, phase, q3_true, q3_star, q3_raw) -> dict:
    t = [int(x) for x in truth]
    p = [int(x) for x in phase]
    tp = sum(a >= 3 and b >= 3 for a, b in zip(t, p))
    fp = sum(a < 3 and b >= 3 for a, b in zip(t, p))
    fn = sum(a >= 3 and b < 3 for a, b in zip(t, p))
    tn = len(t) - tp - fp - fn
    cls = lambda x: min(x, 4) - 1  # noqa: E731
    conf = [[0] * 4 for _ in range(4)]
    for a, b in zip(t, p):
        conf[cls(a)][cls(b)] += 1
    out = {"n": len(t), "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}, "confusion": conf, "na": {}}

    def frac(name, num, den):
        out[name] = Fraction(num, den) if den else None
        if not den:
            out["na"][name] = "zero denominator"

    frac("binary.accuracy", tp + tn, len(t))
    frac("binary.precision", tp, tp + fp)
    frac("binary.recall", tp, tp + fn)
    frac("binary.f1", 2 * tp, 2 * tp + fp + fn)
    frac("binary.f2", 5 * tp, 5 * tp + 4 * fn + fp)
    frac("four_class.accuracy", sum(conf[k][k] for k in range(4)), len(t))
    f1s = []
    for k in range(4):
        tpk = conf[k][k]
        den = 2 * tpk + (sum(conf[r][k] for r in range(4)) - tpk) + (sum(conf[k]) - tpk)
        f1s.append(Fraction(2 * tpk, den) if den else None)
    out["four_class.macro_f1"] = None if any(f is None for f in f1s) else sum(f1s) / 4
    if out["four_class.macro_f1"] is None:
        out["na"]["four_class.macro_f1"] = "class absent"
    out["q3_r2_projected"] = r2(q3_true, q3_star)
    out["q3_r2_raw"] = r2(q3_true, q3_raw)
    return out


def r2(y, f):
    y = np.asarray(y, dtype=float)
    f = np.asarray(f, dtype=float)
    if len(y) < 2 or np.all(y == y[0]):
        return None
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - f) ** 2)) / sst


def arm_scores(frame: pd.DataFrame, arm: str) -> dict:
    if arm == "persistence":
        return scores(frame["phase_truth"], frame["persistence_phase"], frame["q3"], frame["persistence_q3"], frame["persistence_q3"])
    return scores(frame["phase_truth"], frame[f"{arm}_phase"], frame["q3"], frame[f"{arm}_q3_star"], frame[f"{arm}_q3_raw"])


def _num_eq(saved, mine, tol=0.0) -> bool:
    if saved is None or mine is None:
        return saved is None and mine is None
    a, b = float(saved), float(mine)
    return a == b if tol == 0 else abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def flat_ok(saved: dict, mine: dict) -> bool:
    keys = ("binary.f1", *GUARDED)
    return (all(_num_eq(saved[k], None if mine[k] is None else float(mine[k])) for k in keys)
            and _num_eq(saved["q3_r2_projected"], mine["q3_r2_projected"], 1e-12)
            and _num_eq(saved["q3_r2_raw"], mine["q3_r2_raw"], 1e-12) and saved["n"] == mine["n"])


def detail_ok(saved: dict, mine: dict) -> bool:
    keys = ("binary.f1", *GUARDED)
    exact = all((saved["exact"][k] is None and mine[k] is None) or (mine[k] is not None and saved["exact"][k] == str(mine[k])) for k in keys)
    na_keys = {k for k in keys if mine[k] is None} | {k for k in ("q3_r2_projected", "q3_r2_raw") if mine[k] is None}
    return (exact and saved["binary_counts"] == mine["counts"] and saved["four_class_confusion_rows_truth_cols_pred"] == mine["confusion"]
            and saved["n"] == mine["n"] and set(saved["na_reasons"]) == na_keys)


def delta(a: dict, b: dict) -> dict:
    out = {}
    for k in ("binary.f1", *GUARDED):
        out[k] = None if a[k] is None or b[k] is None else float(a[k]) - float(b[k])
    for k in ("q3_r2_projected", "q3_r2_raw"):
        out[k] = None if a[k] is None or b[k] is None else a[k] - b[k]
    return out


def delta_ok(saved: dict, mine: dict) -> bool:
    return all(_num_eq(saved.get(k), v, 1e-12) for k, v in mine.items())


# --------------------------------------------------------------------------- decision (independent re-coding of D22/D23)


def guard_items(g, b1, b2, b0, contract) -> dict:
    """Ordered {guard name: pass} for every D23 point guard (undefined never passes)."""
    tol_c, tol_r = Fraction(contract["evaluation"]["classification_guard"]), float(contract["evaluation"]["projected_r2_guard"])
    items = {}
    for name, ref in (("B1", b1), ("B0", b0)):
        items[f"f1_G>={name}"] = not (g["binary.f1"] is None or ref["binary.f1"] is None or g["binary.f1"] < ref["binary.f1"])
    for name, ref in (("B1", b1), ("B2", b2)):
        for m in GUARDED:
            items[f"{m}_vs_{name}"] = not (g[m] is None or ref[m] is None or g[m] - ref[m] < -tol_c)
        a, b = g["q3_r2_projected"], ref["q3_r2_projected"]
        items[f"q3_r2_projected_vs_{name}"] = not (a is None or b is None or a - b < -tol_r)
    return items


def guards(g, b1, b2, b0, contract) -> tuple[bool, list]:
    items = guard_items(g, b1, b2, b0, contract)
    failed = [k for k, ok in items.items() if not ok]
    return not failed, failed


def status(gain, bound, annual: dict, guard_pass: bool, contract) -> str:
    floor = Fraction(contract["evaluation"]["main_f1_gain_minimum"])
    c1 = gain is not None and gain >= floor
    c2 = bound is not None and bound > 0
    c3 = len(annual) > 0 and all(v is not None and v >= 0 for v in annual.values())
    if gain is None:
        return "undefined_point_estimate"
    if c1 and c2 and c3:
        return "protected_stable_benefit" if guard_pass else "primary_met_guard_tradeoff"
    if bound is None:
        return "inconclusive_bound_undefined"
    if any(v is None for v in annual.values()):
        return "inconclusive_annual_undefined"
    if gain <= 0:
        return "no_gain_or_reversal"
    if not c1:
        return "gain_below_practical_floor"
    if not c2:
        return "imprecise_bound_not_above_zero"
    return "annual_sign_reversal"


# --------------------------------------------------------------------------- bootstrap regeneration


def bootstrap(main: pd.DataFrame, contract: dict) -> dict:
    bs = contract["evaluation"]["bootstrap"]
    n_draws, seed, pct = int(bs["draws"]), int(bs["seed"]), float(bs["lower_percentile"])
    countries = sorted(set(main["country_key"]))
    K = len(countries)
    counts = {}
    for arm in ("G", "B2"):
        rows = []
        for cname in countries:
            part = main[main["country_key"] == cname]
            s = scores(part["phase_truth"], part[f"{arm}_phase"], part["q3"], part[f"{arm}_q3_star"], part[f"{arm}_q3_raw"])["counts"]
            rows.append([s["tp"], s["fp"], s["fn"], s["tn"]])
        counts[arm] = np.array(rows, dtype=np.int64).reshape(K, 4)
    rng = np.random.default_rng(seed)
    mult = np.zeros((n_draws, K), dtype=np.int64)
    for d in range(n_draws):
        mult[d] = np.bincount(rng.integers(0, K, size=K), minlength=K)
    f1 = {}
    for arm in ("G", "B2"):
        agg = mult @ counts[arm]
        den = 2 * agg[:, 0] + agg[:, 1] + agg[:, 2]
        f1[arm] = np.where(den > 0, 2 * agg[:, 0] / np.where(den > 0, den, 1), np.nan)
    d = f1["G"] - f1["B2"]
    defined = np.isfinite(d)
    point = None
    tot = {arm: counts[arm].sum(axis=0) for arm in ("G", "B2")}
    dens = {arm: 2 * tot[arm][0] + tot[arm][1] + tot[arm][2] for arm in ("G", "B2")}
    if K and dens["G"] and dens["B2"]:
        point = Fraction(2 * int(tot["G"][0]), int(dens["G"])) - Fraction(2 * int(tot["B2"][0]), int(dens["B2"]))
    reasons = ["empty cohort"] if K == 0 else []
    if 0 < K < int(bs["min_countries"]):
        reasons.append("K < 2")
    if K and point is None:
        reasons.append("point delta undefined")
    if K and int(defined.sum()) < n_draws:
        reasons.append(f"{n_draws - int(defined.sum())} undefined replicate deltas")

    def linear(p):
        x = np.sort(d)
        h = (len(x) - 1) * p / 100.0
        lo = int(math.floor(h))
        hi_ = min(lo + 1, len(x) - 1)
        return float(x[lo] + (h - lo) * (x[hi_] - x[lo]))

    lower = None if reasons or K == 0 else linear(pct)
    descriptive = None if reasons or K == 0 else [linear(2.5), linear(97.5)]
    return {"countries": countries, "counts": counts, "mult": mult, "f1": f1, "delta": d, "defined": int(defined.sum()),
            "undefined": n_draws - int(defined.sum()), "lower": lower, "point": point, "reasons": reasons, "descriptive": descriptive,
            "K": K, "draws": n_draws, "seed": seed, "percentile": pct}
