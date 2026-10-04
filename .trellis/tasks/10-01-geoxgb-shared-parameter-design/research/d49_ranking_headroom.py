"""D49 zero-fit ranking-headroom diagnostic (d49-ranking-headroom-plan.md).

Hindsight envelope of the deterministic scalar-cutoff family `s >= c` on frozen
saved D38 E3 scores, s = (p2+p3)/sum(p0..p3) in float64. No model fits, no
calibration, no threshold policy, no other score families.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\swl00\geoxgb_runs")
SOURCE = BASE / "geoxgb-d38-persistence-margin-root-20261002"
OUT = BASE / "geoxgb-d49-ranking-headroom-20261002"
HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
ARMS = ("original", "anchored")
PCOLS = ("1", "2", "3", "4或5")  # class codes 0..3
EXPECTED_EXCLUDED = 713


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return Fraction(2 * tp, d) if d else Fraction(0)


def confusion(truth, pred):
    z = np.asarray(truth, bool); y = np.asarray(pred, bool)
    tp, fp, fn, tn = (int(v.sum()) for v in (z & y, ~z & y, z & ~y, ~z & ~y))
    v = f1(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1_exact": str(v), "f1": float(v)}


def frontier(score, truth):
    """Predict-none first, then one endpoint per unique-score block (descending); last = predict-all."""
    score = np.asarray(score, dtype=np.float64); truth = np.asarray(truth, bool)
    assert len(score) == len(truth) and len(score) > 0 and np.isfinite(score).all()
    order = np.argsort(-score, kind="stable"); s = score[order]; z = truth[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    ct = np.cumsum(z, dtype=np.int64); P = int(z.sum())
    pts = [{"kind": "none", "cutoff": None, "cutoff_hex": None, "calls": 0, "tp": 0, "fp": 0, "fn": P}]
    for ix in ends:
        tp = int(ct[ix]); c = float(s[ix])
        pts.append({"kind": "cutoff", "cutoff": c, "cutoff_hex": c.hex(), "calls": int(ix + 1),
                    "tp": tp, "fp": int(ix + 1 - tp), "fn": P - tp})
    assert pts[-1]["calls"] == len(s)
    return pts


def point(p):
    v = f1(p["tp"], p["fp"], p["fn"])
    return {**p, "f1_exact": str(v), "f1": float(v)}


def analyse(pts, per):
    """pts ordered by decreasing cutoff (predict-none first); per = persistence confusion."""
    vals = [f1(p["tp"], p["fp"], p["fn"]) for p in pts]
    best = max(vals); win = vals.index(best)  # first = highest cutoff / fewest calls
    dom = [i for i, p in enumerate(pts) if p["tp"] >= per["tp"] and p["fp"] <= per["fp"]
           and (p["tp"] > per["tp"] or p["fp"] < per["fp"])]
    kp = per["tp"] + per["fp"]
    exact = [p for p in pts if p["calls"] == kp]
    if exact:
        budget = {"k_p": kp, "status": "exact", "point": point(exact[0])}
    else:
        below = [p for p in pts if p["calls"] < kp][-1]
        above = [p for p in pts if p["calls"] > kp][0]
        budget = {"k_p": kp, "status": "bracket", "below": point(below), "above": point(above)}
    return {"optimum": {"f1_exact": str(best), "f1": float(best), "n_optimal": vals.count(best),
                        "winner": point(pts[win])},
            "dominance": {"dominates": bool(dom), "count": len(dom),
                          "witness": point(pts[dom[0]]) if dom else None},
            "budget": budget}


def brute(score, truth, per):
    score = np.asarray(score, dtype=np.float64); truth = np.asarray(truth, bool); P = int(truth.sum())
    cands = [(None, np.zeros(len(score), bool))] + [(float(t), score >= t) for t in sorted(np.unique(score), reverse=True)]
    rows = []
    for t, pred in cands:
        tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
        rows.append({"cutoff": t, "calls": int(pred.sum()), "tp": tp, "fp": fp, "fn": P - tp})
    vals = [f1(r["tp"], r["fp"], r["fn"]) for r in rows]; best = max(vals)
    win = rows[vals.index(best)]["cutoff"]
    dom = [r for r in rows if r["tp"] >= per["tp"] and r["fp"] <= per["fp"]
           and (r["tp"] > per["tp"] or r["fp"] < per["fp"])]
    kp = per["tp"] + per["fp"]; ex = [r for r in rows if r["calls"] == kp]
    if ex:
        bud = ("exact", ex[0]["cutoff"], ex[0]["tp"])
    else:
        bud = ("bracket", max((r for r in rows if r["calls"] < kp), key=lambda r: r["calls"])["cutoff"],
               min((r for r in rows if r["calls"] > kp), key=lambda r: r["calls"])["cutoff"])
    return best, vals.count(best), win, len(dom), (dom[0]["cutoff"] if dom else "absent"), bud


def check_case(score, truth, per):
    a = analyse(frontier(score, truth), per)
    b = brute(score, truth, per)
    assert Fraction(a["optimum"]["f1_exact"]) == b[0] and a["optimum"]["n_optimal"] == b[1]
    assert a["optimum"]["winner"]["cutoff"] == b[2]
    assert a["dominance"]["count"] == b[3] and a["dominance"]["dominates"] == (b[3] > 0)
    assert (a["dominance"]["witness"]["cutoff"] if a["dominance"]["witness"] else "absent") == b[4]
    bud = a["budget"]
    if bud["status"] == "exact":
        assert b[5] == ("exact", bud["point"]["cutoff"], bud["point"]["tp"])
    else:
        assert b[5] == ("bracket", bud["below"]["cutoff"], bud["above"]["cutoff"])
    json.dumps(a, allow_nan=False)
    return a


def selftest():
    rng = np.random.default_rng(49)
    statuses = set()
    for _ in range(300):
        n = int(rng.integers(1, 40))
        score = rng.integers(0, 6, n) / 5.0  # heavy ties
        truth = rng.random(n) < rng.random()
        pp = rng.random(n) < 0.4
        per = confusion(truth, pp)
        a = check_case(score, truth, per)
        statuses.add(a["budget"]["status"])
        pts = frontier(score, truth)
        assert pts[0]["kind"] == "none" and pts[-1]["calls"] == n and pts[-1]["fn"] == 0
    assert statuses == {"exact", "bracket"}
    # explicit bracket: tie block of 3 straddles k_p=2
    s = np.array([.9, .5, .5, .5, .1]); z = np.array([1, 1, 0, 1, 0], bool)
    a = check_case(s, z, {"tp": 1, "fp": 1, "fn": 2})
    assert a["budget"]["status"] == "bracket" and a["budget"]["below"]["calls"] == 1 and a["budget"]["above"]["calls"] == 4
    # explicit exact
    a = check_case(s, z, {"tp": 1, "fp": 0, "fn": 2})
    assert a["budget"]["status"] == "exact" and a["budget"]["point"]["cutoff"] == .9
    # no-positive case: every F1 denominator for predict-none is 0 -> F1 0; predict-none wins the tie
    s = np.array([.7, .7, .2]); z = np.zeros(3, bool)
    a = check_case(s, z, {"tp": 0, "fp": 1, "fn": 0})
    assert a["optimum"]["f1"] == 0 and a["optimum"]["winner"]["kind"] == "none" and a["optimum"]["winner"]["cutoff"] is None
    assert a["optimum"]["n_optimal"] == 3
    # tied optimum between two finite cutoffs: highest cutoff wins
    s = np.array([.9, .8, .7, .6]); z = np.array([1, 0, 0, 1], bool)
    # cutoff .9: F1 2/3; cutoff .6 (predict-all): 4/6=2/3
    a = check_case(s, z, {"tp": 0, "fp": 0, "fn": 2})
    assert a["optimum"]["n_optimal"] == 2 and a["optimum"]["winner"]["cutoff"] == .9
    # predict-all optimum
    s = np.array([.3, .2, .1]); z = np.ones(3, bool)
    a = check_case(s, z, {"tp": 1, "fp": 0, "fn": 2})
    assert a["optimum"]["winner"]["calls"] == 3 and a["optimum"]["f1"] == 1.0
    assert a["dominance"]["witness"]["cutoff"] == .2  # .3 equals persistence (not strict)
    print("SELFTEST OK")


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def identity(hashes):
    rel = Path(__file__).resolve().relative_to(REPO).as_posix()
    blob = git("hash-object", rel); head = git("rev-parse", f"HEAD:{rel}")
    if blob != head or git("status", "--porcelain", "--", rel):
        raise RuntimeError(f"Script bytes not committed at HEAD: {rel}")
    return {"script": rel, "git_blob": blob, "head": git("rev-parse", "HEAD"),
            "script_sha256": sha(__file__), "source": str(SOURCE), "input_hashes": hashes,
            "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                        "platform": platform.platform()}}


def mean(xs):
    return float(np.mean(xs)) if xs else None


def real():
    assert not OUT.exists(), "Preserve existing evidence: output directory exists"
    d39 = json.loads((HERE / "d39_probability_diagnostic.json").read_text(encoding="utf-8"))
    d40 = json.loads((HERE / "d40_summary.json").read_text(encoding="utf-8"))
    hashes = d39["input_hashes"]
    assert hashes == d40["input_hashes"] and len(hashes) == 21
    ident = identity(hashes)
    for name, h in hashes.items():
        assert sha(SOURCE / name / "rows_E3.csv.gz") == h, f"hash drift {name}"
    OUT.mkdir(parents=True)
    (OUT / "identity.json").write_text(json.dumps(ident, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    per_root = {}; frontier_rows = []; excluded = 0
    for name in hashes:
        f = pd.read_csv(SOURCE / name / "rows_E3.csv.gz", float_precision="round_trip")
        assert not f.duplicated(["area", "target_month", "horizon"]).any() and (f.target_month <= "2020-12").all()
        for arm in ARMS:  # argmax gate on every row, before key filtering
            P = f[[f"p_{arm}_{c}" for c in PCOLS]].to_numpy(dtype=np.float64)
            assert np.isfinite(P).all()
            assert (np.argmax(P, axis=1) == f[f"y_{arm}"].to_numpy()).all(), f"argmax mismatch {name} {arm}"
        k = f.persistence_code.notna().to_numpy() & np.isfinite(f.persistence_code.to_numpy(dtype=float))
        g = f[k]; nx = int((~k).sum()); excluded += nx
        z = g.truth.to_numpy() >= 2
        per = confusion(z, g.persistence_code.to_numpy(dtype=float) >= 2)
        ref = d39["per_root"][name]["matched"]
        assert all(per[c] == ref["persistence"]["argmax_crisis"][c] for c in ("tp", "fp", "fn", "tn")), name
        rec = {"horizon": int(g.horizon.iloc[0]), "target_month": str(g.target_month.iloc[0]),
               "excluded_missing_origin": nx, "n": int(k.sum()), "positives": int(z.sum()),
               "persistence": per, "arms": {}}
        for arm in ARMS:
            am = confusion(z, g[f"y_{arm}"].to_numpy() >= 2)
            assert all(am[c] == ref[arm]["argmax_crisis"][c] for c in ("tp", "fp", "fn", "tn")), (name, arm)
            P = g[[f"p_{arm}_{c}" for c in PCOLS]].to_numpy(dtype=np.float64)
            s = (P[:, 2] + P[:, 3]) / P.sum(axis=1)
            pts = frontier(s, z)
            for i, p in enumerate(pts):
                frontier_rows.append({"root": name, "arm": arm, "idx": i, **p})
            rec["arms"][arm] = {"argmax": am, "n_endpoints": len(pts), **analyse(pts, per)}
        per_root[name] = rec
    assert excluded == EXPECTED_EXCLUDED, excluded

    by_h = {}
    for h in sorted({r["horizon"] for r in per_root.values()}):
        rs = [r for r in per_root.values() if r["horizon"] == h]
        e = {"n_folds": len(rs), "persistence_f1": mean([r["persistence"]["f1"] for r in rs]), "arms": {}}
        for arm in ARMS:
            a = [r["arms"][arm] for r in rs]; pf = [r["persistence"]["f1"] for r in rs]
            ex = [(x["budget"]["point"]["tp"] - r["persistence"]["tp"]) for x, r in zip(a, rs) if x["budget"]["status"] == "exact"]
            e["arms"][arm] = {
                "argmax_f1": mean([x["argmax"]["f1"] for x in a]),
                "hindsight_max_f1": mean([x["optimum"]["f1"] for x in a]),
                "gap_optimum_minus_persistence": mean([x["optimum"]["f1"] - p for x, p in zip(a, pf)]),
                "gap_optimum_minus_argmax": mean([x["optimum"]["f1"] - x["argmax"]["f1"] for x in a]),
                "gap_argmax_minus_persistence": mean([x["argmax"]["f1"] - p for x, p in zip(a, pf)]),
                "folds_optimum_gt_persistence": sum(Fraction(x["optimum"]["f1_exact"]) > Fraction(r["persistence"]["f1_exact"]) for x, r in zip(a, rs)),
                "folds_dominating_persistence": sum(x["dominance"]["dominates"] for x in a),
                "budget_exact": len(ex), "budget_bracket": len(a) - len(ex),
                "mean_budget_tp_minus_persistence_tp_exact": mean(ex)}
        by_h[str(h)] = e

    pd.DataFrame(frontier_rows).to_csv(OUT / "frontier.csv.gz", index=False, compression="gzip")
    summary = {
        "plan": "d49-ranking-headroom-plan.md", "script_sha256": ident["script_sha256"],
        "score": "s=(p2+p3)/(p0+p1+p2+p3), float64; family predict crisis iff s>=c",
        "excluded_missing_origin_total": excluded, "per_root": per_root, "by_horizon_mean_fold": by_h,
        "interpretation": ("Hindsight envelope within the deterministic scalar-cutoff family only, chosen with E3 truth "
                           "on repeatedly exposed origin-known folds. Not a bound on four-class, feature-based, randomised "
                           "or row-specific policies; not a deployable rule, validation or adoption evidence. Failure on a "
                           "root means only that this frozen checkpoint's cutoff family cannot beat persistence on that fold; "
                           "success is fold-specific headroom, not a transferable gain, and does not mean overfitting is solved. "
                           "k_p is fixed without target outcomes but evaluated on an exposed cohort. Argmax is generally not in "
                           "the family, so the optimum need not reach argmax. Both arms already use persistence information "
                           "(anchored is not independent ranking evidence). Mean-fold per H only; no pooled oracle, no "
                           "significance tests.")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"D49 done: {OUT}")


def main():
    ap = argparse.ArgumentParser(description="D49 zero-fit ranking-headroom diagnostic (no model fits).")
    ap.add_argument("--selftest", action="store_true", help="run synthetic brute-force checks only (no real data)")
    args = ap.parse_args()
    selftest()
    if not args.selftest:
        real()


if __name__ == "__main__":
    main()
