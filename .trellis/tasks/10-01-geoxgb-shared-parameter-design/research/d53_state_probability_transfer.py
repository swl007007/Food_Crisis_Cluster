"""D53/A27 zero-fit exact-origin-state probability-level transfer.

Descriptive only: within each exact origin state (persistence_code 0..3, plus a
separate `missing` group) compare the observed crisis rate and the mean score of
the original four-class crisis mass (s_original) and the D52 binary score
(p_binary) across FIT, C and E3. Reads only the 63 saved D52 row files after
byte/hash checks against the committed research/d52_completion.json.
No models, snapshots, fits, calibration, thresholds or Stage 2/3.
numpy / pandas / stdlib only.
"""
import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve()
RECORD = SCRIPT.parent / "d52_completion.json"
DEFAULT_SOURCE = r"C:\Users\swl00\geoxgb_runs\geoxgb-d52-binary-root-20261002"
DEFAULT_OUT = r"C:\Users\swl00\geoxgb_runs\geoxgb-d53-state-probability-transfer-20261002"
ROLES = ("FIT", "C", "E3")
GROUPS = ("0", "1", "2", "3", "missing")
STATES = ("0", "1", "2", "3")
SCORES = ("s_original", "p_binary")
COLUMNS = ("area", "target_month", "horizon", "truth_crisis",
           "persistence_code", "s_original", "p_binary")
CONTRASTS = (("E3_minus_FIT", "E3", "FIT"), ("E3_minus_C", "E3", "C"))
HORIZONS = (4, 8, 12)
MAX_MONTH = "2020-12"
TOL = 1e-12


def local_path(p):
    """Map a Windows path to /mnt/<drive>/... when running under WSL/Linux."""
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
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else float(x)
    return x


def dump_json(obj, path):
    Path(path).write_text(json.dumps(clean(obj), indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")


def write_csv(df, path):
    df.to_csv(path, index=False, na_rep="", float_format="%.17g")


# ---------------------------------------------------------------- metrics
def cell_metrics(df):
    n = int(len(df))
    rec = {"n": n}
    if n == 0:
        rec.update(positives=0, negatives=0, n_unique_areas=0,
                   n_unique_label_months=0, min_label_month=None,
                   max_label_month=None, rate=np.nan, support_flag="empty",
                   null_reason="empty")
        for s in SCORES:
            rec.update({f"mean_{s}": np.nan, f"bias_{s}": np.nan,
                        f"brier_{s}": np.nan})
        return rec
    y = df["truth_crisis"].to_numpy(dtype=np.float64)
    pos = int(y.sum())
    neg = n - pos
    rate = pos / n
    rec.update(positives=pos, negatives=neg,
               n_unique_areas=int(df["area"].nunique()),
               n_unique_label_months=int(df["target_month"].nunique()),
               min_label_month=str(df["target_month"].min()),
               max_label_month=str(df["target_month"].max()), rate=rate,
               support_flag=("no_positive" if pos == 0 else
                             "no_negative" if neg == 0 else "both"),
               null_reason=None)
    for s in SCORES:
        p = df[s].to_numpy(dtype=np.float64)
        m = float(np.mean(p))
        rec.update({f"mean_{s}": m, f"bias_{s}": m - rate,
                    f"brier_{s}": float(np.mean((p - y) ** 2))})
    return rec


def validate_rows(df, horizon, name):
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    if df.duplicated(["area", "target_month", "horizon"]).any():
        raise ValueError(f"{name}: duplicate (area,target_month,horizon) keys")
    if df["target_month"].isna().any() or (df["target_month"] > MAX_MONTH).any():
        raise ValueError(f"{name}: target_month null or > {MAX_MONTH}")
    if horizon is not None and not (df["horizon"].astype(int) == horizon).all():
        raise ValueError(f"{name}: horizon column != {horizon}")
    if not df["truth_crisis"].isin([0, 1]).all():
        raise ValueError(f"{name}: truth_crisis outside {{0,1}}")
    pc = df["persistence_code"]
    if not (pc.isna() | pc.isin([0, 1, 2, 3])).all():
        raise ValueError(f"{name}: persistence_code outside {{0,1,2,3,NaN}}")
    for s in SCORES:
        v = df[s].to_numpy(dtype=np.float64)
        if not (np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()):
            raise ValueError(f"{name}: {s} not finite in [0,1]")


def group_of(df):
    pc = df["persistence_code"]
    return np.where(pc.isna(), "missing",
                    pc.fillna(-1).astype(int).astype(str))


def role_tables(df, meta, role):
    """Return (role rows, month rows) with add-back checks."""
    df = df.assign(_group=group_of(df))
    months = sorted(df["target_month"].unique())
    role_rows, month_rows = [], []
    for g in GROUPS:
        sub = df[df["_group"] == g]
        rr = dict(meta, role=role, group=g, **cell_metrics(sub))
        role_rows.append(rr)
        n_sum = pos_sum = 0
        for m in months:
            ms = sub[sub["target_month"] == m]
            mr = dict(meta, role=role, group=g, label_month=m, **cell_metrics(ms))
            month_rows.append(mr)
            n_sum += mr["n"]
            pos_sum += mr["positives"]
        if n_sum != rr["n"] or pos_sum != rr["positives"]:
            raise ValueError(f"{meta}/{role}/{g}: month cells do not add back")
    if sum(r["n"] for r in role_rows) != len(df):
        raise ValueError(f"{meta}/{role}: group cells do not add back to rows")
    return role_rows, month_rows


def contrasts_for(role_rows):
    idx = {(r["root"], r["role"], r["group"]): r for r in role_rows}
    roots = sorted({r["root"] for r in role_rows})
    out = []
    for root in roots:
        for g in STATES:
            for cname, a, b in CONTRASTS:
                ra, rb = idx[(root, a, g)], idx[(root, b, g)]
                rec = {"root": root, "horizon": ra["horizon"], "target": ra["target"],
                       "group": g, "contrast": cname,
                       "n_E3": ra["n"], "n_reference": rb["n"]}
                empty = [x for x, r in ((a, ra), (b, rb)) if r["n"] == 0]
                keys = ["delta_rate"] + [f"delta_{k}_{s}" for s in SCORES
                                         for k in ("mean", "bias")]
                if empty:
                    rec.update({k: np.nan for k in keys})
                    rec["null_reason"] = "empty_" + "_".join(empty)
                else:
                    rec["delta_rate"] = ra["rate"] - rb["rate"]
                    for s in SCORES:
                        dm = ra[f"mean_{s}"] - rb[f"mean_{s}"]
                        db = ra[f"bias_{s}"] - rb[f"bias_{s}"]
                        if abs(db - (dm - rec["delta_rate"])) > TOL:
                            raise ValueError(f"{root}/{g}/{cname}/{s}: bias identity")
                        rec[f"delta_mean_{s}"] = dm
                        rec[f"delta_bias_{s}"] = db
                    rec["null_reason"] = None
                out.append(rec)
    return out


METRICS = ["delta_rate"] + [f"delta_{k}_{s}" for s in SCORES for k in ("mean", "bias")]


def summarize(contrasts, roots_per_h=7):
    res = {}
    for h in HORIZONS:
        for g in STATES:
            for cname, _, _ in CONTRASTS:
                recs = [r for r in contrasts if int(r["horizon"]) == h
                        and r["group"] == g and r["contrast"] == cname]
                for m in METRICS:
                    vals = [float(r[m]) for r in recs
                            if r[m] is not None and math.isfinite(float(r[m]))]
                    res[f"h{h}|state{g}|{cname}|{m}"] = {
                        "horizon": h, "state": g, "contrast": cname, "metric": m,
                        "mean_over_valid_roots": (sum(vals) / len(vals)) if vals else None,
                        "n_valid": len(vals), "n_roots": roots_per_h,
                        "n_positive": sum(v > 0 for v in vals),
                        "n_negative": sum(v < 0 for v in vals),
                        "n_zero": sum(v == 0 for v in vals)}
    return {
        "cells": res,
        "interpretation": (
            "Descriptive probability-level / transition-rate transfer within exact "
            "origin states across FIT, C and E3. Role composition differs (areas, "
            "label months, eras, calendar regimes), so no causal time-drift or pure "
            "overfitting claim. The Brier residual after bias is not called "
            "refinement; AUC is not treated as a Brier decomposition. Rare-state "
            "support is visible in role_cells.csv / month_cells.csv. Unweighted "
            "means over valid roots; signs from raw recorded deltas; no pooled "
            "cross-root table and no significance tests."),
    }


# ---------------------------------------------------------------- selftest
def selftest():
    def mk(rows):
        return pd.DataFrame(rows, columns=COLUMNS)
    # empty cell
    e = cell_metrics(mk([]))
    assert e["null_reason"] == "empty" and e["support_flag"] == "empty"
    assert all(math.isnan(e[k]) for k in ("rate", "mean_s_original", "brier_p_binary"))
    # single-class non-empty cell
    sc = mk([("a1", "2019-02", 4, 0, 1, 0.2, 0.4), ("a2", "2019-06", 4, 0, 1, 0.4, 0.0)])
    c = cell_metrics(sc)
    assert c["null_reason"] is None and c["support_flag"] == "no_positive"
    assert c["rate"] == 0.0 and abs(c["mean_s_original"] - 0.3) < TOL
    assert abs(c["bias_s_original"] - 0.3) < TOL
    assert abs(c["brier_s_original"] - (0.04 + 0.16) / 2) < TOL
    assert abs(c["brier_p_binary"] - 0.08) < TOL
    c1 = cell_metrics(mk([("a1", "2019-02", 4, 1, 3, 0.9, 0.7)]))
    assert c1["support_flag"] == "no_negative" and c1["null_reason"] is None
    # support counts
    sp = mk([("a1", "2019-02", 4, 1, 2, .5, .5), ("a1", "2019-06", 4, 0, 2, .5, .5),
             ("a2", "2019-02", 4, 1, 2, .5, .5)])
    s = cell_metrics(sp)
    assert (s["n"], s["positives"], s["negatives"], s["n_unique_areas"],
            s["n_unique_label_months"], s["min_label_month"], s["max_label_month"]) == \
        (3, 2, 1, 2, 2, "2019-02", "2019-06")
    # weighted-month recomposition: month A 4 rows rate .25, month B 1 row rate 1
    rows = [(f"a{i}", "2019-02", 4, int(i == 0), 0, 0.1 * (i + 1), 0.2) for i in range(4)]
    rows += [("b0", "2019-06", 4, 1, 0, 0.9, 0.8), ("m0", "2019-06", 4, 0, np.nan, .3, .3)]
    df = mk(rows)
    validate_rows(df, 4, "selftest")
    meta = {"root": "r1", "horizon": 4, "target": "2020-02"}
    rr, mr = role_tables(df, meta, "FIT")
    g0 = next(r for r in rr if r["group"] == "0")
    mm = [r for r in mr if r["group"] == "0"]
    assert len(mm) == 2 and {r["rate"] for r in mm} == {0.25, 1.0}
    tot = sum(r["n"] for r in mm)
    for k in ("rate", "mean_s_original", "mean_p_binary"):
        assert abs(g0[k] - sum(r["n"] * r[k] for r in mm) / tot) < TOL
    assert abs(g0["rate"] - 0.4) < TOL
    gmiss = next(r for r in rr if r["group"] == "missing")
    assert gmiss["n"] == 1
    empty_months = [r for r in mr if r["group"] == "2"]
    assert len(empty_months) == 2 and all(r["null_reason"] == "empty" for r in empty_months)
    assert len(rr) == 5 and len(mr) == 10
    # contrasts: identity and sign from raw tiny delta
    tiny = 1e-15
    fit = mk([("a", "2018-02", 4, 0, s_, 0.5, 0.5) for s_ in (0, 1, 2)])
    cc = mk([("a", "2018-06", 4, 0, s_, 0.5, 0.5 + tiny) for s_ in (0, 1)])
    e3 = mk([("a", "2019-02", 4, 1, 0, 0.6, 0.5), ("a", "2019-02", 4, 0, 1, 0.5, 0.5),
             ("b", "2019-02", 4, 0, 2, 0.5, 0.5)])
    allr = []
    for role, d in (("FIT", fit), ("C", cc), ("E3", e3)):
        allr += role_tables(d, meta, role)[0]
    con = contrasts_for(allr)
    assert len(con) == 4 * 2
    st1c = next(r for r in con if r["group"] == "1" and r["contrast"] == "E3_minus_C")
    assert st1c["delta_mean_p_binary"] < 0 and round(st1c["delta_mean_p_binary"], 12) == 0
    st2c = next(r for r in con if r["group"] == "2" and r["contrast"] == "E3_minus_C")
    assert st2c["null_reason"] == "empty_C" and math.isnan(st2c["delta_rate"])
    st3 = [r for r in con if r["group"] == "3"]
    assert {r["null_reason"] for r in st3} == {"empty_E3_FIT", "empty_E3_C"}
    for r in con:
        if r["null_reason"] is None:
            for sname in SCORES:
                assert abs(r[f"delta_bias_{sname}"] -
                           (r[f"delta_mean_{sname}"] - r["delta_rate"])) <= TOL
    summ = summarize(con, roots_per_h=1)
    cell = summ["cells"]["h4|state1|E3_minus_C|delta_mean_p_binary"]
    assert cell["n_negative"] == 1 and cell["n_zero"] == 0 and cell["n_valid"] == 1
    cell = summ["cells"]["h4|state2|E3_minus_C|delta_rate"]
    assert cell["n_valid"] == 0 and cell["mean_over_valid_roots"] is None
    # JSON allow_nan=False
    json.dumps(clean(summ), allow_nan=False)
    json.dumps(clean(con), allow_nan=False)
    try:
        json.dumps({"x": float("nan")}, allow_nan=False)
        raise AssertionError("allow_nan=False did not refuse NaN")
    except ValueError:
        pass
    # validation refusals
    for bad in (mk([("a", "2021-02", 4, 0, 0, .5, .5)]),
                mk([("a", "2019-02", 4, 2, 0, .5, .5)]),
                mk([("a", "2019-02", 4, 0, 4, .5, .5)]),
                mk([("a", "2019-02", 4, 0, 0, 1.5, .5)]),
                mk([("a", "2019-02", 8, 0, 0, .5, .5)]),
                mk([("a", "2019-02", 4, 0, 0, .5, np.nan)]),
                mk([("a", "2019-02", 4, 0, 0, .5, .5)] * 2)):
        try:
            validate_rows(bad, 4, "bad")
            raise AssertionError("validation did not refuse bad rows")
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


def verify_inputs(source):
    ext = source / "completion.json"
    rec_bytes, ext_bytes = RECORD.read_bytes(), ext.read_bytes()
    if rec_bytes != ext_bytes or hashlib.sha256(rec_bytes).hexdigest() != \
            hashlib.sha256(ext_bytes).hexdigest():
        raise ValueError("external completion.json differs from research/d52_completion.json")
    outputs = json.loads(rec_bytes)["outputs"]
    roots = sorted({k.split("/")[0] for k in outputs if "/rows_" in k})
    if len(roots) != 21:
        raise ValueError(f"expected 21 roots, found {len(roots)}")
    for h in HORIZONS:
        if sum(r.startswith(f"h{h}_") for r in roots) != 7:
            raise ValueError(f"expected 7 roots for h{h}")
    expected = {f"{r}/rows_{role}.csv.gz" for r in roots for role in ROLES}
    recorded = {k for k in outputs if "/rows_" in k}
    if expected != recorded or len(expected) != 63:
        raise ValueError("row-file inventory differs from the D52 record")
    hashes = {}
    for k in sorted(expected):
        got = sha256(source / k)
        if got != outputs[k]:
            raise ValueError(f"sha256 mismatch for {k}")
        hashes[k] = got
    return roots, hashes, hashlib.sha256(rec_bytes).hexdigest()


def real_run(source, out):
    roots, hashes, rec_sha = verify_inputs(source)
    if out.exists():
        raise FileExistsError(f"output directory exists: {out}")
    ident = script_identity()
    out.mkdir(parents=True)
    dump_json({"script": ident, "d52_record": {"path": "research/d52_completion.json",
                                               "sha256": rec_sha},
               "source": str(source), "inputs_sha256": hashes,
               "runtime": {"python": sys.version, "numpy": np.__version__,
                           "pandas": pd.__version__, "platform": platform.platform()}},
              out / "identity.json")
    role_rows, month_rows = [], []
    for root in roots:
        h = int(root.split("_")[0][1:])
        meta = {"root": root, "horizon": h, "target": root.split("_")[1]}
        for role in ROLES:
            name = f"{root}/rows_{role}.csv.gz"
            df = pd.read_csv(source / name, usecols=list(COLUMNS),
                             dtype={"area": str, "target_month": str},
                             float_precision="round_trip")
            validate_rows(df, h, name)
            rr, mr = role_tables(df, meta, role)
            role_rows += rr
            month_rows += mr
    if len(role_rows) != 315:
        raise ValueError(f"expected 315 role cells, got {len(role_rows)}")
    con = contrasts_for(role_rows)
    if len(con) != 168:
        raise ValueError(f"expected 168 contrasts, got {len(con)}")
    write_csv(pd.DataFrame(role_rows), out / "role_cells.csv")
    write_csv(pd.DataFrame(month_rows), out / "month_cells.csv")
    write_csv(pd.DataFrame(con), out / "contrasts.csv")
    dump_json(summarize(con), out / "summary.json")
    files = ("identity.json", "role_cells.csv", "month_cells.csv", "contrasts.csv",
             "summary.json")
    dump_json({"status": "complete", "outputs": {f: sha256(out / f) for f in files}},
              out / "completion.json")
    print(f"D53 complete: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--selftest", action="store_true", help="synthetic checks only")
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    real_run(local_path(a.source), local_path(a.out))


if __name__ == "__main__":
    main()
