"""D54/A28 zero-fit fixed-policy local-increment contrast.

Applies D38's fixed post-hoc transform p_post ∝ p*q (q=.625 at the exact origin
class, .125 elsewhere; uniform, i.e. no change, when the origin is missing) to
the saved D41 root and full Brier-local probabilities (frozen D34 maps/routes),
and compares E3 crisis decisions (four-class argmax -> code >= 2).
Inputs: D41 rows (hash vs research/d41_summary.json), D52 C/E3 rows (via
research/d52_completion.json; authoritative origin codes and original
probabilities), D38 rows_E3 (via research/d39_probability_diagnostic.json and
research/d40_summary.json; E3-only post-hoc reconciliation).
No models, fits, grids, thresholds, gate relearning or Stage 2/3.
numpy / pandas / stdlib only.
"""
import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from fractions import Fraction
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve()
RESEARCH = SCRIPT.parent
D41_RECORD = RESEARCH / "d41_summary.json"
D52_RECORD = RESEARCH / "d52_completion.json"
D39_RECORD = RESEARCH / "d39_probability_diagnostic.json"
D40_RECORD = RESEARCH / "d40_summary.json"
D41_ROWS = r"C:\Users\swl00\geoxgb_runs\d41-local-shrinkage-20261002\rows.csv.gz"
D52_DIR = r"C:\Users\swl00\geoxgb_runs\geoxgb-d52-binary-root-20261002"
D38_DIR = r"C:\Users\swl00\geoxgb_runs\geoxgb-d38-persistence-margin-root-20261002"
DEFAULT_OUT = r"C:\Users\swl00\geoxgb_runs\geoxgb-d54-fixed-policy-local-contrast-20261002"
CLS = ("1", "2", "3", "4或5")
PARTS = ("C", "E3")
HORIZONS = (4, 8, 12)
KEY = ["root", "part", "area", "target_month", "horizon"]
ARMS = ("raw_root", "raw_full", "post_root", "post_full")
PAIRS = (("raw", "raw_full", "raw_root"), ("post", "post_full", "post_root"))
WINS = (("post_full", "post_root"), ("post_full", "persistence"), ("raw_full", "raw_root"))
Q_ORIGIN, Q_OTHER = 0.625, 0.125
TOL = 1e-12


class GateError(ValueError):
    """A stopping discrepancy (fail closed)."""


def local_path(p):
    if sys.platform.startswith("win") or not (len(p) > 1 and p[1] == ":"):
        return Path(p)
    w = PureWindowsPath(p)
    return Path("/mnt", w.drive[0].lower(), *w.parts[1:])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Fraction):
        return f"{x.numerator}/{x.denominator}"
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else float(x)
    return x


def dump_json(obj, path):
    Path(path).write_text(json.dumps(clean(obj), indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")


def write_csv(df, path):
    df.to_csv(path, index=False, na_rep="", float_format="%.17g")


# ---------------------------------------------------------------- core
def probs(df, prefix):
    return df[[f"{prefix}_{c}" for c in CLS]].to_numpy(dtype=np.float64)


def transform(p, code):
    """Fixed D38 post-hoc transform; missing origin returns raw exactly."""
    p = np.asarray(p, dtype=np.float64)
    code = np.asarray(code, dtype=np.float64)
    out = p.copy()
    known = ~np.isnan(code)
    if known.any():
        q = np.full((int(known.sum()), 4), Q_OTHER, dtype=np.float64)
        q[np.arange(q.shape[0]), code[known].astype(int)] = Q_ORIGIN
        w = p[known] * q
        out[known] = w / w.sum(axis=1, keepdims=True)
    return out


def crisis_mass(p):
    return (p[:, 2] + p[:, 3]) / p.sum(axis=1)


def f1_exact(tp, fp, fn):
    d = 2 * tp + fp + fn
    return Fraction(0) if d == 0 else Fraction(2 * tp, d)


def confusion(y, c):
    y = np.asarray(y, dtype=bool)
    c = np.asarray(c, dtype=bool)
    tp, fp = int((y & c).sum()), int((~y & c).sum())
    fn, tn = int((y & ~c).sum()), int((~y & ~c).sum())
    f = f1_exact(tp, fp, fn)
    return {"n": int(y.size), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "f1": float(f), "f1_exact": f}


def same_codes(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.array_equal(np.isnan(a), np.isnan(b)) and \
        np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])


def check_codes(code, name):
    c = np.asarray(code, dtype=np.float64)
    if not np.isin(c[~np.isnan(c)], [0, 1, 2, 3]).all():
        raise GateError(f"{name}: persistence_code outside {{0,1,2,3,NaN}}")


def join_d41_d52(d41, d52):
    """Gates 1-3 and 5; returns joined frame with arm probabilities attached."""
    for name, df in (("D41", d41), ("D52", d52)):
        if df.duplicated(KEY).any():
            raise GateError(f"{name}: duplicate keys")
    r41, r52 = set(d41["root"]), set(d52["root"])
    if r41 != r52:
        raise GateError("D41/D52 root sets differ")
    for part in PARTS:
        k41 = set(map(tuple, d41.loc[d41["part"] == part, KEY].itertuples(index=False)))
        k52 = set(map(tuple, d52.loc[d52["part"] == part, KEY].itertuples(index=False)))
        if k41 != k52:
            raise GateError(f"{part}: D41 and D52 key sets differ "
                            f"({len(k41 - k52)} D41-only, {len(k52 - k41)} D52-only)")
    if set(d41["part"]) - set(PARTS) or set(d52["part"]) - set(PARTS):
        raise GateError("unexpected part label")
    j = d41.merge(d52, on=KEY, how="inner", suffixes=("", "_d52"), validate="one_to_one")
    if len(j) != len(d41) or len(j) != len(d52):
        raise GateError("join lost rows")
    p_root = probs(j, "p_root")
    if not np.array_equal(p_root, probs(j, "p_original")):
        raise GateError("D41 p_root differs from D52 p_original")
    if not np.array_equal(j["truth"].to_numpy(), j["truth_d52"].to_numpy()):
        raise GateError("D41 truth differs from D52 truth")
    if "truth_crisis" in j and not np.array_equal(
            j["truth_crisis"].to_numpy().astype(int), (j["truth"].to_numpy() >= 2).astype(int)):
        raise GateError("D52 truth_crisis != (truth >= 2)")
    code = j["persistence_code_d52"].to_numpy(dtype=np.float64)
    check_codes(code, "D52")
    if not same_codes(j["persistence_code"].to_numpy(dtype=np.float64), code):
        raise GateError("D41 vs D52 persistence_code disagreement")
    p_full = probs(j, "p_full")
    zero = (j["route_type"] == "zero_increment").to_numpy()
    if not set(j["route_type"]) <= {"local", "zero_increment"}:
        raise GateError("unexpected route_type")
    if not np.array_equal(p_full[zero], p_root[zero]):
        raise GateError("zero_increment rows have p_full != p_root")
    for arm, col in (("raw_root", "y_root"), ("raw_full", "y_full")):
        p = p_root if arm == "raw_root" else p_full
        if not np.array_equal(np.argmax(p, axis=1), j[col].to_numpy().astype(int)):
            raise GateError(f"saved {col} != argmax of probabilities")
    post_root, post_full = transform(p_root, code), transform(p_full, code)
    miss = np.isnan(code)
    if not (np.array_equal(post_root[miss], p_root[miss])
            and np.array_equal(post_full[miss], p_full[miss])):
        raise GateError("missing-origin rows not neutral")
    if not np.array_equal(post_full[zero], post_root[zero]):
        raise GateError("zero_increment post(full) != post(root)")
    out = j[KEY + ["truth", "route_type"]].copy()
    out["code"] = code
    out["y_crisis"] = j["truth"].to_numpy() >= 2
    for arm, p in (("raw_root", p_root), ("raw_full", p_full),
                   ("post_root", post_root), ("post_full", post_full)):
        out[f"{arm}_cls"] = np.argmax(p, axis=1)
        out[f"{arm}_s"] = crisis_mass(p)
    out["_post_root_p"] = list(post_root)
    return out


def check_d38(e3, d38):
    """Gate 6 (E3 post-hoc part): D38 p_posthoc/y_posthoc vs post(root)."""
    k = ["root", "area", "target_month", "horizon"]
    if d38.duplicated(k).any():
        raise GateError("D38: duplicate keys")
    a = set(map(tuple, e3[k].itertuples(index=False)))
    b = set(map(tuple, d38[k].itertuples(index=False)))
    if a != b:
        raise GateError(f"D38 E3 key set differs ({len(a - b)} / {len(b - a)})")
    j = e3.merge(d38, on=k, how="inner", suffixes=("", "_d38"), validate="one_to_one")
    if not np.array_equal(j["truth"].to_numpy(), j["truth_d38"].to_numpy()):
        raise GateError("D38 truth differs")
    pr = np.vstack(j["_post_root_p"].to_numpy())
    diff = float(np.max(np.abs(pr - probs(j, "p_posthoc")))) if len(j) else 0.0
    if not diff <= TOL:
        raise GateError(f"post(root) vs D38 p_posthoc max diff {diff}")
    if not np.array_equal(j["post_root_cls"].to_numpy(), j["y_posthoc"].to_numpy().astype(int)):
        raise GateError("argmax post(root) != D38 y_posthoc")
    per_root = {}
    for root, g in j.groupby("root", sort=True):
        ours = confusion(g["y_crisis"], g["post_root_cls"] >= 2)
        theirs = confusion(g["truth_d38"] >= 2, g["y_posthoc"] >= 2)
        if any(ours[x] != theirs[x] for x in ("tp", "fp", "fn", "tn")):
            raise GateError(f"{root}: post-hoc confusion differs from D38 rows")
        per_root[root] = {x: ours[x] for x in ("n", "tp", "fp", "fn", "tn")}
    code_d38 = j["persistence_code"].to_numpy(dtype=np.float64)
    code = j["code"].to_numpy(dtype=np.float64)
    agree = (np.isnan(code_d38) & np.isnan(code)) | (code_d38 == code)
    return {"max_abs_posthoc_diff": diff, "n": int(len(j)), "per_root": per_root,
            "persistence_code_disagreements_d38_vs_d52_recorded": int((~agree).sum())}


def same_conf(rec, conf):
    return all(int(rec[x]) == conf[x] for x in ("n", "tp", "fp", "fn", "tn"))


def reconcile_d41(df, summ):
    """Gate 6 (raw part): raw root/full confusions equal the D41 record."""
    checked = []
    amap = {"root": "raw_root", "full": "raw_full"}
    for part in PARTS:
        rec = summ["parts"][part]
        pdf = df[df["part"] == part]
        sets = {"all": pdf, "matched": pdf[~np.isnan(pdf["code"].to_numpy())]}
        for ks, sub in sets.items():
            for a, arm in amap.items():
                c = confusion(sub["y_crisis"], sub[f"{arm}_cls"] >= 2)
                if not same_conf(rec[ks][a], c):
                    raise GateError(f"D41 {part}/{ks}/{a} confusion mismatch")
                checked.append(f"{part}/{ks}/{a}")
        for name, field, groups in (("per_root", "root", rec["per_root"]),
                                    ("per_horizon", "horizon", rec["per_horizon"]),
                                    ("routes", "route_type", rec["routes"])):
            for gk, grec in groups.items():
                sub = pdf[pdf[field].astype(str) == str(gk)]
                for a, arm in amap.items():
                    c = confusion(sub["y_crisis"], sub[f"{arm}_cls"] >= 2)
                    if not same_conf(grec[a], c):
                        raise GateError(f"D41 {part}/{name}/{gk}/{a} confusion mismatch")
            checked.append(f"{part}/{name}(all keys, {len(groups)} groups)")
    return checked


def reconcile_d39(df_e3, d39):
    """Post(root) E3 confusions equal the committed D39 per-root posthoc record."""
    done = []
    for root, rec in d39["per_root"].items():
        g = df_e3[df_e3["root"] == root]
        for ks, sub in (("all", g), ("matched", g[~np.isnan(g["code"].to_numpy())])):
            r = rec.get(ks, {}).get("posthoc", {}).get("argmax_crisis")
            if r is None:
                raise GateError(f"D39 {root}/{ks} posthoc argmax_crisis record missing")
            c = confusion(sub["y_crisis"], sub["post_root_cls"] >= 2)
            if any(int(r[x]) != c[x] for x in ("tp", "fp", "fn", "tn")):
                raise GateError(f"D39 {root}/{ks} posthoc confusion mismatch")
            done.append(f"{root}/{ks}")
    return done


# ---------------------------------------------------------------- metrics
def arm_metrics(sub, arm):
    y = sub["y_crisis"].to_numpy(dtype=bool)
    if arm == "persistence":
        call = sub["code"].to_numpy() >= 2
        s = call.astype(np.float64)
    else:
        call = sub[f"{arm}_cls"].to_numpy() >= 2
        s = sub[f"{arm}_s"].to_numpy(dtype=np.float64)
    c = confusion(y, call)
    c["brier"] = float(np.mean((s - y) ** 2)) if len(y) else np.nan
    return c


def score(df):
    rows = []
    for part in PARTS:
        for root, g in df[df["part"] == part].groupby("root", sort=True):
            h = int(g["horizon"].iloc[0])
            known = ~np.isnan(g["code"].to_numpy())
            for ks, sub in (("matched", g[known]), ("all", g)):
                arms = ARMS + (("persistence",) if ks == "matched" else ())
                for arm in arms:
                    m = arm_metrics(sub, arm)
                    rows.append({"root": root, "part": part, "horizon": h, "keyset": ks,
                                 "arm": arm, "missing_origin_n": int((~known).sum()),
                                 **m, "f1_exact": clean(m["f1_exact"])})
    return pd.DataFrame(rows)


def pooled(per_root, df):
    out = {}
    for part in PARTS:
        for ks in ("matched", "all"):
            pdf = df[df["part"] == part]
            if ks == "matched":
                pdf = pdf[~np.isnan(pdf["code"].to_numpy())]
            arms = ARMS + (("persistence",) if ks == "matched" else ())
            for hk in ("all",) + tuple(str(h) for h in HORIZONS):
                sub = pdf if hk == "all" else pdf[pdf["horizon"] == int(hk)]
                pr = per_root[(per_root["part"] == part) & (per_root["keyset"] == ks)]
                if hk != "all":
                    pr = pr[pr["horizon"] == int(hk)]
                cell = {"n_roots": int(pr["root"].nunique()),
                        "missing_origin_n": int(np.isnan(sub["code"].to_numpy()).sum())}
                for arm in arms:
                    m = arm_metrics(sub, arm)
                    fr = [Fraction(x) for x in pr.loc[pr["arm"] == arm, "f1_exact"]]
                    cell[arm] = {**{x: m[x] for x in ("n", "tp", "fp", "fn", "tn",
                                                       "f1", "brier")},
                                 "mean_fold_f1": float(sum(fr) / len(fr)) if fr else None}
                wins = {}
                for a, b in WINS:
                    if b not in arms:
                        continue
                    fa = dict(zip(pr.loc[pr["arm"] == a, "root"],
                                  pr.loc[pr["arm"] == a, "f1_exact"].map(Fraction)))
                    fb = dict(zip(pr.loc[pr["arm"] == b, "root"],
                                  pr.loc[pr["arm"] == b, "f1_exact"].map(Fraction)))
                    w = [(fa[r] > fb[r]) - (fa[r] < fb[r]) for r in sorted(fa)]
                    wins[f"{a}_vs_{b}"] = {"wins": w.count(1), "ties": w.count(0),
                                           "losses": w.count(-1)}
                cell["fold_wins"] = wins
                out[f"{part}|{ks}|h{hk}"] = cell
    return out


def changes(df):
    rows = []
    for part in PARTS:
        pdf = df[df["part"] == part]
        for ks, sub0 in (("matched", pdf[~np.isnan(pdf["code"].to_numpy())]), ("all", pdf)):
            for (root, rt), g in sub0.groupby(["root", "route_type"], sort=True):
                y = g["y_crisis"].to_numpy(dtype=bool)
                for name, a, b in PAIRS:
                    ca = g[f"{a}_cls"].to_numpy() >= 2
                    cb = g[f"{b}_cls"].to_numpy() >= 2
                    m_a, m_b = confusion(y, ca), confusion(y, cb)
                    rows.append({"root": root, "part": part, "keyset": ks,
                                 "horizon": int(g["horizon"].iloc[0]), "route_type": rt,
                                 "pair": f"{name}:{a}-{b}", "n": int(len(g)),
                                 "crisis_flips": int((ca != cb).sum()),
                                 "fourclass_flips": int((g[f"{a}_cls"].to_numpy()
                                                         != g[f"{b}_cls"].to_numpy()).sum()),
                                 "d_tp": m_a["tp"] - m_b["tp"], "d_fp": m_a["fp"] - m_b["fp"],
                                 "d_fn": m_a["fn"] - m_b["fn"]})
    ch = pd.DataFrame(rows)
    z = ch[ch["route_type"] == "zero_increment"]
    if (z[["crisis_flips", "fourclass_flips", "d_tp", "d_fp", "d_fn"]].to_numpy() != 0).any():
        raise GateError("zero_increment rows show local-root change")
    return ch


INTERPRETATION = (
    "Zero-fit, zero-parameter diagnostic: D38's fixed q (.625 origin / .125 other; "
    "uniform when origin missing) applied post hoc to saved D41 root and full "
    "Brier-local probabilities on frozen D34 maps/routes. The interaction was chosen "
    "after exposed development results (D38-D53); not pre-registered. The transform "
    "preserves only class log-odds differences between full and root, not probability "
    "differences, and is not a byte-exact margin replay. Post-hoc outputs are "
    "diagnostic only; raw four-class outputs remain the primary model outputs. Not "
    "partition training on an anchored base and no evidence such training would work. "
    "Primary: E3 matched exact-origin crisis F1; persistence only on matched keys; C "
    "is in-window context only. No significance test, no adoption; stop, then a "
    "research decision memo.")


# ---------------------------------------------------------------- selftest
def _synthetic():
    rng = np.random.default_rng(0)
    rows41, rows52 = [], []
    for part in PARTS:
        for i in range(6):
            p = rng.dirichlet(np.ones(4))
            dp = rng.dirichlet(np.ones(4))
            rt = "zero_increment" if i % 3 == 0 else "local"
            pf = p if rt == "zero_increment" else (p + dp) / 2
            code = np.nan if i == 5 else float(i % 4)
            truth = int(i % 4)
            base = {"root": "h4_r", "part": part, "area": str(i), "target_month": "2019-02",
                    "horizon": 4, "truth": truth, "persistence_code": code}
            r41 = dict(base, route_type=rt, y_root=int(np.argmax(p)), y_full=int(np.argmax(pf)))
            r41.update({f"p_root_{c}": p[k] for k, c in enumerate(CLS)})
            r41.update({f"p_full_{c}": pf[k] for k, c in enumerate(CLS)})
            r52 = dict(base, truth_crisis=int(truth >= 2))
            r52.update({f"p_original_{c}": p[k] for k, c in enumerate(CLS)})
            rows41.append(r41)
            rows52.append(r52)
    return pd.DataFrame(rows41), pd.DataFrame(rows52)


def _expect_stop(fn, *a):
    try:
        fn(*a)
    except GateError:
        return
    raise AssertionError("expected a GateError stop")


def selftest():
    # transform values and normalisation
    p = np.array([[0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]])
    t = transform(p, np.array([2.0, 0.0]))
    w = np.array([0.4 * .125, 0.3 * .125, 0.2 * .625, 0.1 * .125])
    assert np.allclose(t[0], w / w.sum(), atol=1e-15, rtol=0)
    assert np.allclose(t[1], [5 / 8, 1 / 8, 1 / 8, 1 / 8], atol=1e-15, rtol=0)
    assert np.allclose(t.sum(axis=1), 1.0, atol=1e-15, rtol=0)
    # missing-origin neutrality (exact)
    t2 = transform(p, np.array([np.nan, 1.0]))
    assert np.array_equal(t2[0], p[0]) and not np.array_equal(t2[1], p[1])
    # happy path, zero-increment identity, change counts
    d41, d52 = _synthetic()
    j = join_d41_d52(d41, d52)
    z = j[j["route_type"] == "zero_increment"]
    assert (z["post_full_cls"] == z["post_root_cls"]).all()
    assert (z["post_full_s"] == z["post_root_s"]).all()
    ch = changes(j)
    assert (ch[ch["route_type"] == "zero_increment"]["d_tp"] == 0).all()
    pr = score(j)
    assert set(pr["arm"]) == set(ARMS) | {"persistence"}
    assert not ((pr["keyset"] == "all") & (pr["arm"] == "persistence")).any()
    pooled(pr, j)
    # key-set mismatch stops
    _expect_stop(join_d41_d52, d41, d52.iloc[1:])
    bad = d52.copy()
    bad.loc[0, "area"] = "999"
    _expect_stop(join_d41_d52, d41, bad)
    # p_root vs p_original mismatch stops (one ulp)
    bad = d52.copy()
    bad.loc[0, "p_original_1"] = np.nextafter(bad.loc[0, "p_original_1"], 1.0)
    _expect_stop(join_d41_d52, d41, bad)
    # origin-code disagreement stops (value and NaN pattern)
    for v in (3.0 if d52.loc[1, "persistence_code"] != 3 else 0.0, np.nan):
        bad = d52.copy()
        bad.loc[1, "persistence_code"] = v
        _expect_stop(join_d41_d52, d41, bad)
    # zero-increment identity violation stops
    bad = d41.copy()
    zi = bad.index[bad["route_type"] == "zero_increment"][0]
    bad.loc[zi, "p_full_1"] = bad.loc[zi, "p_full_1"] * 0.5
    _expect_stop(join_d41_d52, bad, d52)
    # D38 posthoc reconciliation: pass, then mismatch stops
    e3 = j[j["part"] == "E3"].reset_index(drop=True)
    post = np.vstack(e3["_post_root_p"].to_numpy())
    d38 = e3[["root", "area", "target_month", "horizon", "truth"]].copy()
    d38["persistence_code"] = e3["code"]
    d38["y_posthoc"] = np.argmax(post, axis=1)
    for k, c in enumerate(CLS):
        d38[f"p_posthoc_{c}"] = post[:, k]
    rec = check_d38(e3, d38)
    assert rec["max_abs_posthoc_diff"] == 0.0
    bad = d38.copy()
    bad.loc[0, "p_posthoc_1"] += 1e-9
    _expect_stop(check_d38, e3, bad)
    bad = d38.copy()
    bad.loc[0, "y_posthoc"] = (bad.loc[0, "y_posthoc"] + 1) % 4
    _expect_stop(check_d38, e3, bad)
    _expect_stop(check_d38, e3, d38.iloc[1:])
    # truth mismatch stops
    bad = d52.copy()
    bad.loc[0, "truth"] = (bad.loc[0, "truth"] + 1) % 4
    _expect_stop(join_d41_d52, d41, bad)
    # D41 confusion reconciliation: pass, then mismatch stops
    def rec_of(sub):
        return {a: confusion(sub["y_crisis"], sub[f"{arm}_cls"] >= 2)
                for a, arm in (("root", "raw_root"), ("full", "raw_full"))}
    summ = {"parts": {}}
    for part in PARTS:
        pdf = j[j["part"] == part]
        summ["parts"][part] = {
            "all": rec_of(pdf), "matched": rec_of(pdf[~np.isnan(pdf["code"].to_numpy())]),
            "per_root": {r: rec_of(g) for r, g in pdf.groupby("root")},
            "per_horizon": {str(h): rec_of(g) for h, g in pdf.groupby("horizon")},
            "routes": {r: rec_of(g) for r, g in pdf.groupby("route_type")}}
    reconcile_d41(j, summ)
    summ["parts"]["E3"]["routes"]["local"]["full"]["tp"] += 1
    _expect_stop(reconcile_d41, j, summ)
    # D39 post-hoc record: pass, mismatch stops, missing record stops
    d39 = {"per_root": {r: {ks: {"posthoc": {"argmax_crisis": confusion(
        sub["y_crisis"], sub["post_root_cls"] >= 2)}} for ks, sub in
        (("all", g), ("matched", g[~np.isnan(g["code"].to_numpy())]))}
        for r, g in e3.groupby("root")}}
    assert len(reconcile_d39(e3, d39)) == 2
    d39["per_root"]["h4_r"]["all"]["posthoc"]["argmax_crisis"]["fp"] += 1
    _expect_stop(reconcile_d39, e3, d39)
    del d39["per_root"]["h4_r"]["matched"]["posthoc"]
    d39["per_root"]["h4_r"]["all"]["posthoc"]["argmax_crisis"]["fp"] -= 1
    _expect_stop(reconcile_d39, e3, d39)
    # confusion / exact F1 on a tiny case
    c = confusion([1, 1, 0, 0, 1], [1, 0, 1, 0, 1])
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 1)
    assert c["f1_exact"] == Fraction(2, 3) and c["f1"] == 2 / 3
    assert confusion([0, 0], [0, 0])["f1_exact"] == 0
    # JSON allow_nan=False
    json.dumps(clean({"f": Fraction(1, 3), "x": np.nan}), allow_nan=False)
    try:
        json.dumps({"x": float("nan")}, allow_nan=False)
        raise AssertionError("allow_nan=False did not refuse NaN")
    except ValueError:
        pass
    print("SELFTEST OK")


# ---------------------------------------------------------------- real run
def git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


def script_identity():
    top = Path(git(["rev-parse", "--show-toplevel"], SCRIPT.parent)).resolve()
    rel = SCRIPT.relative_to(top).as_posix()
    blob = git(["hash-object", rel], top)
    head = git(["rev-parse", f"HEAD:{rel}"], top)
    dirty = git(["status", "--porcelain", "--", rel], top)
    if blob != head or dirty:
        raise RuntimeError(f"script {rel} not committed at HEAD (blob {blob} head {head} "
                           f"status {dirty!r})")
    return {"path": rel, "git_blob": blob, "head_commit": git(["rev-parse", "HEAD"], top),
            "sha256": sha256(SCRIPT)}


def verify_inputs(d41_rows, d52_dir, d38_dir):
    d41 = json.loads(D41_RECORD.read_text(encoding="utf-8"))
    hashes = {"d41_rows": sha256(d41_rows)}
    if hashes["d41_rows"] != d41["rows_sha256"]:
        raise GateError("D41 rows sha256 differs from d41_summary.json")
    rec_bytes = D52_RECORD.read_bytes()
    if rec_bytes != (d52_dir / "completion.json").read_bytes():
        raise GateError("external D52 completion.json differs from research record")
    outputs = json.loads(rec_bytes)["outputs"]
    roots = sorted({k.split("/")[0] for k in outputs if "/rows_" in k})
    if len(roots) != 21 or any(sum(r.startswith(f"h{h}_") for r in roots) != 7
                               for h in HORIZONS):
        raise GateError("expected 21 roots, 7 per horizon")
    for r in roots:
        for part in PARTS:
            k = f"{r}/rows_{part}.csv.gz"
            got = sha256(d52_dir / k)
            if got != outputs.get(k):
                raise GateError(f"D52 sha256 mismatch for {k}")
            hashes[f"d52/{k}"] = got
    h39 = json.loads(D39_RECORD.read_text(encoding="utf-8"))["input_hashes"]
    h40 = json.loads(D40_RECORD.read_text(encoding="utf-8"))["input_hashes"]
    if h39 != h40 or sorted(h39) != roots:
        raise GateError("D39/D40 input_hashes differ or root set mismatch")
    for r in roots:
        got = sha256(d38_dir / r / "rows_E3.csv.gz")
        if got != h39[r]:
            raise GateError(f"D38 rows_E3 sha256 mismatch for {r}")
        hashes[f"d38/{r}/rows_E3.csv.gz"] = got
    records = {p.name: sha256(p) for p in (D41_RECORD, D52_RECORD, D39_RECORD, D40_RECORD)}
    return roots, hashes, records, d41


def read_csv(path, **kw):
    return pd.read_csv(path, dtype={"area": str, "target_month": str},
                       float_precision="round_trip", **kw)


def real_run(d41_rows, d52_dir, d38_dir, out):
    roots, hashes, records, d41_summary = verify_inputs(d41_rows, d52_dir, d38_dir)
    if out.exists():
        raise FileExistsError(f"output directory exists: {out}")
    ident = script_identity()
    out.mkdir(parents=True)
    dump_json({"script": ident, "records_sha256": records, "inputs_sha256": hashes,
               "sources": {"d41": str(d41_rows), "d52": str(d52_dir), "d38": str(d38_dir)},
               "runtime": {"python": sys.version, "numpy": np.__version__,
                           "pandas": pd.__version__, "platform": platform.platform()}},
              out / "identity.json")
    try:
        cols41 = KEY + ["truth", "route_type", "persistence_code", "y_root", "y_full"] + \
            [f"p_{a}_{c}" for a in ("root", "full") for c in CLS]
        d41 = read_csv(d41_rows, usecols=cols41)
        d52 = pd.concat([read_csv(d52_dir / r / f"rows_{part}.csv.gz").assign(root=r, part=part)
                         for r in roots for part in PARTS], ignore_index=True)
        d52 = d52[KEY + ["truth", "truth_crisis", "persistence_code"] +
                  [f"p_original_{c}" for c in CLS]]
        j = join_d41_d52(d41, d52)
        recon = {"d41_confusions_checked": reconcile_d41(j, d41_summary),
                 "d41_rows_sha256": hashes["d41_rows"]}
        e3 = j[j["part"] == "E3"]
        d38 = pd.concat([read_csv(d38_dir / r / "rows_E3.csv.gz").assign(root=r)
                         for r in roots], ignore_index=True)
        recon["d38_posthoc_E3"] = check_d38(e3, d38)
        d39 = json.loads(D39_RECORD.read_text(encoding="utf-8"))
        recon["d39_posthoc_confusions_checked"] = reconcile_d39(e3, d39)
        j = j.drop(columns=["_post_root_p"])
        per_root = score(j)
        ch = changes(j)
    except GateError as e:
        dump_json({"status": "failed", "stopping_discrepancy": str(e)}, out / "failure.json")
        raise
    pool = pooled(per_root, j)
    ch_tot = ch.groupby(["part", "keyset", "route_type", "pair"], sort=True)[
        ["n", "crisis_flips", "fourclass_flips", "d_tp", "d_fp", "d_fn"]].sum()
    summary = {"definition": "D54 fixed D38 post-hoc q on saved D41 root/full; zero fit",
               "q": {"origin": Q_ORIGIN, "other": Q_OTHER}, "n_roots": len(roots),
               "rows": {p: int((j["part"] == p).sum()) for p in PARTS},
               "missing_origin_rows": {p: int(np.isnan(j.loc[j["part"] == p, "code"]).sum())
                                       for p in PARTS},
               "gates": "all passed", "reconciliation": recon, "pooled": pool,
               "change_totals": [dict(zip(("part", "keyset", "route_type", "pair"), k), **v)
                                 for k, v in ch_tot.to_dict("index").items()],
               "interpretation": INTERPRETATION}
    write_csv(per_root, out / "per_root.csv")
    write_csv(ch, out / "changes.csv")
    dump_json(summary, out / "summary.json")
    files = ("identity.json", "per_root.csv", "changes.csv", "summary.json")
    dump_json({"status": "complete", "outputs": {f: sha256(out / f) for f in files}},
              out / "completion.json")
    print(f"D54 complete: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d41-rows", default=D41_ROWS)
    ap.add_argument("--d52-dir", default=D52_DIR)
    ap.add_argument("--d38-dir", default=D38_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--selftest", action="store_true", help="synthetic checks only")
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    real_run(local_path(a.d41_rows), local_path(a.d52_dir), local_path(a.d38_dir),
             local_path(a.out))


if __name__ == "__main__":
    main()
