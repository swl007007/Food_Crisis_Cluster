"""D50 zero-fit exact-origin-phase ranking diagnostic (d50-origin-phase-ranking-plan.md).

Within each exact origin phase (persistence_code 0..3), crisis AUC of the frozen
saved D38 E3 scores s = (p2+p3)/sum(p0..p3) in float64, plus fixed argmax
confusions. No model fits, calibration, thresholds, pooling or adoption.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path(r"C:\Users\swl00\geoxgb_runs")
SOURCE = BASE / "geoxgb-d38-persistence-margin-root-20261002"
OUT = BASE / "geoxgb-d50-origin-phase-ranking-20261002"
HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
ARMS = ("original", "anchored")
PCOLS = ("1", "2", "3", "4或5")  # class codes 0..3
CODES = (0, 1, 2, 3)
HORIZONS = (4, 8, 12)
EXPECTED_KNOWN = 112795
EXPECTED_EXCLUDED = 713
CF = ("tp", "fp", "fn", "tn")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def confusion(truth, pred):
    z = np.asarray(truth, bool); y = np.asarray(pred, bool)
    return {k: int(v.sum()) for k, v in zip(CF, (z & y, ~z & y, z & ~y, ~z & ~y))}


def auc(score, truth):
    """Crisis AUC via sklearn only; null with explicit reason when P*N == 0."""
    z = np.asarray(truth, bool); n = len(z); P = int(z.sum()); N = n - P
    if n == 0:
        return None, "empty"
    if P == 0:
        return None, "no_positive"
    if N == 0:
        return None, "no_negative"
    s = np.asarray(score, dtype=np.float64)
    assert np.isfinite(s).all()
    return float(roc_auc_score(z.astype(int), s)), None


def selftest():
    # tied scores: pos {.5,.9}, neg {.5,.1}: pairs (.5,.5)=tie,(.5,.1)=win,(.9,.5)=win,(.9,.1)=win -> 3.5/4
    v, r = auc([.5, .9, .5, .1], [1, 1, 0, 0])
    assert r is None and abs(v - 0.875) < 1e-15, v
    v, r = auc([.3, .3, .3, .3], [1, 0, 1, 0])
    assert r is None and v == 0.5, v
    assert auc([], []) == (None, "empty")
    assert auc([.1, .2], [0, 0]) == (None, "no_positive")
    assert auc([.1, .2], [1, 1]) == (None, "no_negative")
    # brute-force pair counts (2*wins+ties)/(2PN) on a small heavily tied case
    rng = np.random.default_rng(50)
    for _ in range(50):
        n = int(rng.integers(2, 30))
        s = rng.integers(0, 4, n) / 3.0; z = rng.random(n) < 0.4
        P = int(z.sum()); N = n - P
        v, r = auc(s, z)
        if P * N == 0:
            assert v is None and r in ("no_positive", "no_negative")
            continue
        w = t = 0
        for a in s[z]:
            for b in s[~z]:
                w += a > b; t += a == b
        assert abs(v - (2 * w + t) / (2 * P * N)) < 1e-12, (v, w, t, P, N)
    assert confusion([1, 1, 0, 0, 1], [1, 0, 1, 0, 1]) == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}
    json.dumps({"a": None, "b": 0.5, "c": [1, 2]}, allow_nan=False)
    try:
        json.dumps({"x": float("nan")}, allow_nan=False)
        raise AssertionError("allow_nan=False did not refuse NaN")
    except ValueError:
        pass
    print("SELFTEST OK")


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def identity(hashes):
    rel = Path(__file__).resolve().relative_to(REPO).as_posix()
    blob = git("hash-object", rel); head = git("rev-parse", f"HEAD:{rel}")
    if blob != head or git("status", "--porcelain", "--", rel):
        raise RuntimeError(f"Script bytes not committed at HEAD: {rel}")
    import sklearn
    return {"script": rel, "git_blob": blob, "head": git("rev-parse", "HEAD"),
            "script_sha256": sha(__file__), "source": str(SOURCE), "input_hashes": hashes,
            "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                        "sklearn": sklearn.__version__, "platform": platform.platform()}}


def real():
    assert not OUT.exists(), "Preserve existing evidence: output directory exists"
    d39 = json.loads((HERE / "d39_probability_diagnostic.json").read_text(encoding="utf-8"))
    d40 = json.loads((HERE / "d40_summary.json").read_text(encoding="utf-8"))
    d49i = json.loads((HERE / "d49_identity.json").read_text(encoding="utf-8"))
    d49 = json.loads((HERE / "d49_summary.json").read_text(encoding="utf-8"))
    hashes = d39["input_hashes"]
    assert hashes == d40["input_hashes"] == d49i["input_hashes"] and len(hashes) == 21
    assert set(d49["per_root"]) == set(hashes)
    ident = identity(hashes)
    for name, h in hashes.items():
        assert sha(SOURCE / name / "rows_E3.csv.gz") == h, f"hash drift {name}"
    frames = {name: pd.read_csv(SOURCE / name / "rows_E3.csv.gz", float_precision="round_trip") for name in hashes}
    dates = {h: set() for h in HORIZONS}; roots_h = {h: 0 for h in HORIZONS}
    for name, f in frames.items():
        hs = set(f.horizon.unique()); assert len(hs) == 1 and int(next(iter(hs))) in dates, (name, hs)
        assert f.target_month.astype(str).nunique() == 1, (name, "one target month per root")
        dates[int(next(iter(hs)))].update(f.target_month.astype(str).unique()); roots_h[int(next(iter(hs)))] += 1
    assert all(len(dates[h]) == 7 and roots_h[h] == 7 for h in HORIZONS), ({h: sorted(v) for h, v in dates.items()}, roots_h)
    OUT.mkdir(parents=True)
    (OUT / "identity.json").write_text(json.dumps(ident, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    per_root = {}; known_total = excl_total = 0
    for name, f in frames.items():
        assert not f.duplicated(["area", "target_month", "horizon"]).any() and (f.target_month.astype(str) <= "2020-12").all()
        for arm in ARMS:  # argmax gate on every row
            Pm = f[[f"p_{arm}_{c}" for c in PCOLS]].to_numpy(dtype=np.float64)
            assert np.isfinite(Pm).all()
            assert (np.argmax(Pm, axis=1) == f[f"y_{arm}"].to_numpy()).all(), f"argmax mismatch {name} {arm}"
        pc = f.persistence_code.to_numpy(dtype=float)
        k = np.isfinite(pc); g = f[k]; pcode = pc[k]
        assert (pcode == np.round(pcode)).all() and np.isin(pcode, CODES).all(), name
        assert (g.origin_phase.to_numpy(dtype=float) == pcode + 1).all(), f"origin_phase relation {name}"
        ref = d49["per_root"][name]
        n, nx = int(k.sum()), int((~k).sum())
        assert n == ref["n"] and nx == ref["excluded_missing_origin"], (name, n, nx)
        known_total += n; excl_total += nx
        z = g.truth.to_numpy() >= 2
        score = {}; crisis = {"persistence": pcode >= 2}
        for arm in ARMS:
            Pm = g[[f"p_{arm}_{c}" for c in PCOLS]].to_numpy(dtype=np.float64)
            score[arm] = (Pm[:, 2] + Pm[:, 3]) / Pm.sum(axis=1)
            crisis[arm] = g[f"y_{arm}"].to_numpy() >= 2
        cells = {}
        sums = {m: dict.fromkeys(CF, 0) for m in ("persistence",) + ARMS}
        for c in CODES:
            m = pcode == c; P = int(z[m].sum())
            cell = {"origin_phase": c + 1, "n": int(m.sum()), "P": P, "N": int(m.sum()) - P,
                    "auc": {}, "auc_null_reason": {}, "confusion": {}}
            for arm in ARMS:
                cell["auc"][arm], cell["auc_null_reason"][arm] = auc(score[arm][m], z[m])
            for mod in ("persistence",) + ARMS:
                cm = confusion(z[m], crisis[mod][m]); cell["confusion"][mod] = cm
                for kk in CF:
                    sums[mod][kk] += cm[kk]
            cells[str(c)] = cell
        assert all(sums["persistence"][kk] == ref["persistence"][kk] for kk in CF), (name, "persistence")
        for arm in ARMS:
            assert all(sums[arm][kk] == ref["arms"][arm]["argmax"][kk] for kk in CF), (name, arm)
        per_root[name] = {"horizon": int(g.horizon.iloc[0]), "target_month": str(g.target_month.iloc[0]),
                          "n": n, "excluded_missing_origin": nx, "cells": cells}
    assert known_total == EXPECTED_KNOWN and excl_total == EXPECTED_EXCLUDED, (known_total, excl_total)

    by_date = {}
    for name, r in sorted(per_root.items(), key=lambda kv: (kv[1]["target_month"], kv[1]["horizon"])):
        by_date.setdefault(r["target_month"], {})[name] = {"horizon": r["horizon"], "cells": r["cells"]}
    by_h = {}
    for h in HORIZONS:
        rs = sorted((r for r in per_root.values() if r["horizon"] == h), key=lambda r: r["target_month"])
        assert len(rs) == 7, (h, len(rs))
        e = {}
        for c in CODES:
            valid = [r for r in rs if r["cells"][str(c)]["P"] * r["cells"][str(c)]["N"] > 0]
            ent = {"origin_phase": c + 1, "n_valid": len(valid), "n_folds": len(rs),
                   "supports": [{"date": r["target_month"], "P": r["cells"][str(c)]["P"], "N": r["cells"][str(c)]["N"],
                                 "valid": r["cells"][str(c)]["P"] * r["cells"][str(c)]["N"] > 0} for r in rs],
                   "mean_fold_auc": {}}
            for arm in ARMS:
                vals = [r["cells"][str(c)]["auc"][arm] for r in valid]
                assert all(v is not None for v in vals)
                ent["mean_fold_auc"][arm] = float(np.mean(vals)) if vals else None
            e[str(c)] = ent
        by_h[str(h)] = e

    summary = {
        "plan": "d50-origin-phase-ranking-plan.md", "script_sha256": ident["script_sha256"],
        "score": "s=(p2+p3)/(p0+p1+p2+p3), float64; crisis truth code>=2; AUC via sklearn.metrics.roc_auc_score (ties half credit)",
        "cells": "persistence_code 0/1/2/3 = origin phase 1/2/3/4-or-5; all cells reported, no minimum sample",
        "known_total": known_total, "excluded_missing_origin_total": excl_total,
        "per_root": per_root, "by_date": by_date, "by_horizon_mean_fold_valid_only": by_h,
        "interpretation": ("Within-cell AUC above .5 is descriptive ranking beyond constant exact-phase information only, "
                           "on repeatedly exposed origin-known E3 folds; it can reflect other history, country or era "
                           "features. It is not deployability, validation, stable forward transfer or an overfitting "
                           "resolution. Low within-cell AUC alone does not show that global AUC or the D49 headroom is "
                           "mostly persistence; no decomposition or causal attribution is made. Per-H means use valid "
                           "folds only (P*N>0) with n_valid/7 and P/N supports; no pooled AUC/AP, thresholds or country "
                           "slicing. Cell persistence F1 is omitted as redundant to the counts. No numeric success "
                           "threshold and no adoption.")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"D50 done: {OUT}")


def main():
    ap = argparse.ArgumentParser(description="D50 zero-fit exact-origin-phase ranking diagnostic (no model fits).")
    ap.add_argument("--selftest", action="store_true", help="run synthetic checks only (no real data)")
    args = ap.parse_args()
    selftest()
    if not args.selftest:
        real()


if __name__ == "__main__":
    main()
