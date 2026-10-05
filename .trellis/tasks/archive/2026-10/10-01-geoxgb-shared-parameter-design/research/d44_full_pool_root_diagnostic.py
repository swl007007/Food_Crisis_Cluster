"""D44 / A18: zero-fit full-pool (v1 saved globals) vs r80-FIT (D34 saved roots) on 15 shared pairs.

Raw XGBoost predict of 30 saved boosters on exact E3 matrices (fresh DMatrix each); exact float32
equality with the saved probabilities after round-trip parsing; fail closed on any mismatch.
Only pure-constant/metric package modules are imported (src.experiment.plan, src.metrics.fourclass);
no fit or continuation API is called (static AST guard below). Snapshots are read with a pyarrow
filter target_month <= 2020-12. Output: fresh external directory with keyed rows, metrics, identity.
"""
import ast, gzip, hashlib, json, platform, sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

B = Path(r"C:\Users\swl00\geoxgb_runs")
PKG = Path(r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster\FEWSNETGeoXGBExperiment")
V1 = B / "geoxgb-v1-20261001"
D34 = B / "geoxgb-d34-e1-brier-20261002"
STAGE = D34 / "stage1_e1pair"
OUT = B / "geoxgb-d44-full-pool-root-20261002"
SCHEMA_SHA = "51b6f8b21b76a78510522c34e2d1f2a648b7aec768bbcac3dd2318669fa13349"
V1_GIT, V1_CODE = "268c17b960912b68b5a27c2877fbe05e6857906d", "bfe559236b415f27e1f9c69e316b4a3b498805c0596faa2758f6dfb87998afa0"
D34_GIT, D34_CODE = "7b2bf6fe482d0a77a664f3627e493934d976696d", "cad8439de1d1c838ddd8482c6782f4cfa205e5c28670e45dad2ddbe92e69421b"
G = {4: "G1", 8: "G4", 12: "G2"}
TARGETS = ("2019-02", "2019-06", "2019-10", "2020-02", "2020-06")
PAIRS = [(h, t) for h in (4, 8, 12) for t in TARGETS]
MAX_MONTH = 2020 * 12 + 11
LABELS = ("1", "2", "3", "4或5")
FORBIDDEN = {"train", "fit_global", "continue_booster", "run_candidate", "fit", "update", "boost"}

sys.path.insert(0, str(PKG))
from src.experiment import plan  # noqa: E402  (pure constants)
from src.metrics import fourclass  # noqa: E402  (pure metrics)


class Stop(RuntimeError):
    pass


def guard_no_fit_calls():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            if name in FORBIDDEN:
                raise Stop(f"forbidden fit/continuation call: {name}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            if any(m and ("native_xgb" in m or "main_model" in m or "stage3" in m or "GeoRF" in m) for m in mods):
                raise Stop(f"forbidden production import: {mods}")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mi(s):
    return int(s[:4]) * 12 + int(s[5:7]) - 1


def ml(i):
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


def keys_sha(area, month):
    k = np.column_stack([np.asarray(area, dtype=np.int64), np.asarray(month, dtype=np.int64)])
    return hashlib.sha256(np.ascontiguousarray(k).tobytes()).hexdigest()


def check(cond, msg):
    if not cond:
        raise Stop(msg)


def replay(booster, X):
    X = np.where(np.isinf(X), np.nan, X)
    out = booster.predict(xgb.DMatrix(X, missing=np.nan, nthread=4))   # fresh DMatrix
    check(out.ndim == 2 and out.shape[1] == 4, f"unexpected output shape {out.shape}")
    return out.astype(np.float32)


def crisis_counts(t, p):
    z, a = t >= 2, p >= 2
    return int((z & a).sum()), int((~z & a).sum()), int((z & ~a).sum()), int((~z & ~a).sum())


def score(t, pred, proba=None):
    tp, fp, fn, tn = crisis_counts(t, pred)
    f1 = Fraction(2 * tp, 2 * tp + fp + fn) if (2 * tp + fp + fn) else Fraction(0)
    out = {"n": int(len(t)), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "crisis_f1": float(f1), "crisis_f1_exact": str(f1),
           "macro_f1_fourclass": fourclass.macro_f1(t, pred), "confusion_fourclass": fourclass.confusion(t, pred).astype(int).tolist()}
    if proba is not None:
        pc = proba[:, 2].astype(np.float64) + proba[:, 3].astype(np.float64)
        out["crisis_brier"] = float(np.mean((pc - (t >= 2).astype(np.float64)) ** 2))
    return out


def decisions(t, a, b):
    za, zb, z = a >= 2, b >= 2, t >= 2
    ch = za != zb
    return {"changed": int(ch.sum()), "corrected": int((ch & (za == z)).sum()), "spoiled": int((ch & (zb == z)).sum()),
            "tp_delta": int((za & z).sum() - (zb & z).sum()), "fp_delta": int((za & ~z).sum() - (zb & ~z).sum())}


def pooled(blocks, arm):
    conf = np.sum([np.array(b[arm]["confusion_fourclass"]) for b in blocks], axis=0)
    tp = sum(b[arm]["tp"] for b in blocks); fp = sum(b[arm]["fp"] for b in blocks); fn = sum(b[arm]["fn"] for b in blocks)
    n = sum(b[arm]["n"] for b in blocks)
    f1 = Fraction(2 * tp, 2 * tp + fp + fn)
    out = {"n": n, "tp": tp, "fp": fp, "fn": fn, "crisis_f1": float(f1), "crisis_f1_exact": str(f1),
           "macro_f1_fourclass": fourclass.macro_f1_from_matrix(conf)}
    if all("crisis_brier" in b[arm] for b in blocks):
        out["crisis_brier"] = float(sum(b[arm]["crisis_brier"] * b[arm]["n"] for b in blocks) / n)
    return out


def main():
    guard_no_fit_calls()
    check(not OUT.exists(), f"{OUT} exists")
    schema_path = PKG / "feature-schema.json"
    check(sha(schema_path) == SCHEMA_SHA, "schema file sha differs from the pinned value")
    features = json.loads(schema_path.read_text(encoding="utf-8"))["ordered_features"]
    check(len(features) == 162, "schema does not list 162 features")
    for run, git, code in ((V1, V1_GIT, V1_CODE), (D34, D34_GIT, D34_CODE)):
        ident = json.loads((run / "prepared" / "manifests" / "identity.json").read_text(encoding="utf-8"))
        check(ident["git_head"] == git and ident["code"]["sha256"] == code, f"{run.name} producer identity differs")
    gpred = pd.read_csv(V1 / "gscreen" / "predictions.csv.gz", float_precision="round_trip", dtype={"area": np.int64})
    snaps, identity, rows_all, per_pair = {}, {"pairs": {}}, [], {}
    for h in (4, 8, 12):
        p1, p2 = V1 / "prepared" / f"snapshot_h{h}.parquet", D34 / "prepared" / f"snapshot_h{h}.parquet"
        s1, s2 = sha(p1), sha(p2)
        check(s1 == s2, f"h{h}: v1 and D34 snapshots differ")
        check("hist_phase_o00" in features, "hist_phase_o00 is not one of the 162 features")
        cols = ["area", "target_month", "horizon", "class_code"] + features     # hist_phase_o00 comes via features
        check(len(cols) == len(set(cols)), "duplicate snapshot column selection")
        snap = pd.read_parquet(p2, columns=cols, filters=[("target_month", "<=", MAX_MONTH)])
        check(list(snap.columns) == cols, "snapshot columns differ from the requested selection")
        check((snap["horizon"] == h).all(), f"h{h}: snapshot horizon mismatch")
        snaps[h] = (snap.sort_values(["area", "target_month"]).reset_index(drop=True), s2)
    for h, t in PAIRS:
        g = G[h]; o = mi(t) - h; snap, ssha = snaps[h]
        name = f"h{h}_{t}_{g}_r80_s42_e1pair"
        # ---- full-pool arm provenance
        rec = json.loads((V1 / "globals" / f"h{h}" / g / f"O{ml(o)}.json").read_text(encoding="utf-8"))
        ubj1 = V1 / "globals" / f"h{h}" / g / f"O{ml(o)}.ubj"
        params, rounds = plan.booster_params(plan.G_CONFIGS[g])
        check(rec["snapshot_sha256"] == ssha and rec["origin_month"] == ml(o) and rec["g_config"] == g, f"{name}: v1 record identity")
        check(rec["params"] == params and int(rec["rounds_total"]) == rounds, f"{name}: v1 params/rounds")
        check(sha(ubj1) == rec["booster_sha256"], f"{name}: v1 booster sha")
        win = snap[(snap.target_month >= o - plan.WINDOW) & (snap.target_month < o)]
        check(len(win) == rec["rows"] and keys_sha(win.area, win.target_month) == rec["fit_keys_sha256"], f"{name}: full-pool keys")
        # ---- r80-FIT arm provenance
        root = json.loads((STAGE / "roots" / name / "root.json").read_text(encoding="utf-8"))
        mem = pd.read_csv(STAGE / "roots" / name / "fold_membership.csv.gz")
        check(root["horizon"] == h and root["target_month"] == t and root["origin_month"] == ml(o) and root["g_config"] == g,
              f"{name}: root.json pair fields")
        check(not mem.duplicated(["area", "target_month"]).any(), f"{name}: duplicate membership keys")
        fit = mem[mem.role == "fitting"]; legal = mem[mem.role != "heldout_target"]; e3m = mem[mem.role == "heldout_target"]
        check((e3m.target_month == t).all(), f"{name}: heldout_target rows are not at T")
        check(keys_sha(fit.area, fit.target_month.map(mi)) == root["fitting_keys_sha256"], f"{name}: FIT keys")
        cand = f"h{h}_{t}_{g}_L1_r80_s42_e1brier_gt0"
        ubj2 = STAGE / "checkpoints" / cand / "xgb_root.ubj"
        check(sha(ubj2) == root["root_booster_sha256"] == root["root_fit"]["booster_sha256"], f"{name}: D34 root sha")
        check(root["root_fit"]["params"] == params and int(root["root_fit"]["rounds_total"]) == rounds, f"{name}: D34 params/rounds")
        wk = set(zip(win.area.tolist(), win.target_month.tolist()))
        lk = set(zip(legal.area.tolist(), legal.target_month.map(mi).tolist()))
        check(lk <= wk, f"{name}: D34 legal pool not a subset of the v1 window")
        extra = wk - lk; extra_areas = sorted({a for a, _ in extra})
        # ---- E3 keys / truth / origin
        e3_areas = e3m.area.to_numpy(np.int64)
        rt = pd.read_csv(STAGE / "roots" / name / "root_target_predictions.csv", float_precision="round_trip")
        gp = gpred[(gpred.horizon == h) & (gpred.g_config == g) & (gpred.target_month == t)].sort_values("area")
        check(gp.area.is_unique and rt.FEWSNET_admin_code.is_unique, f"{name}: duplicate E3 keys in saved predictions")
        check(len(set(e3_areas)) == len(e3_areas) and set(e3_areas) == set(rt.FEWSNET_admin_code) == set(gp.area), f"{name}: E3 area sets")
        check((gp.origin_month == ml(o)).all(), f"{name}: v1 origin month")
        check(not extra_areas or not (set(extra_areas) & set(e3_areas)), f"{name}: extra areas appear in E3")
        tgt = snap[snap.target_month == mi(t)].set_index("area").loc[np.sort(e3_areas)]
        areas = tgt.index.to_numpy(np.int64); truth = tgt.class_code.to_numpy(np.int64)
        rtx = rt.set_index("FEWSNET_admin_code").loc[areas]; gpx = gp.set_index("area").loc[areas]
        memx = e3m.set_index("area").loc[areas]
        check((rtx.y_true_code.to_numpy() == truth).all() and (gpx.y_true_code.to_numpy() == truth).all()
              and (memx.class_code.to_numpy() == truth).all(), f"{name}: E3 truth")
        X = tgt[features].to_numpy(dtype=float)
        check(X.shape == (len(areas), 162), f"{name}: E3 matrix is not n x 162 ({X.shape})")
        # ---- raw replay, exact float32
        b1 = xgb.Booster(); b1.load_model(bytearray(ubj1.read_bytes()))
        b2 = xgb.Booster(); b2.load_model(bytearray(ubj2.read_bytes()))
        p_full, p_r80 = replay(b1, X), replay(b2, X)
        saved_full = gpx[[f"p_{l}" for l in LABELS]].to_numpy(np.float64).astype(np.float32)
        saved_r80 = rtx[[f"p_pooled_{l}" for l in LABELS]].to_numpy(np.float64).astype(np.float32)
        check(np.array_equal(p_full, saved_full), f"{name}: full-pool replay not exactly equal (max diff {np.abs(p_full - saved_full).max()})")
        check(np.array_equal(p_r80, saved_r80), f"{name}: r80 replay not exactly equal (max diff {np.abs(p_r80 - saved_r80).max()})")
        y_full, y_r80 = p_full.argmax(1), p_r80.argmax(1)
        check((y_full == gpx.y_pred_code.to_numpy()).all() and (y_r80 == rtx.y_pred_pooled_code.to_numpy()).all(), f"{name}: labels")
        ph = tgt.hist_phase_o00.to_numpy(float)
        check(np.isin(ph[np.isfinite(ph)], [1, 2, 3, 4, 5]).all(), f"{name}: origin phase values")
        per = np.where(np.isfinite(ph), np.minimum(ph, 4) - 1, np.nan)
        known = np.isfinite(per)
        # ---- metrics
        blk = {"all": {"full_pool_root": score(truth, y_full, p_full), "r80_fit_root": score(truth, y_r80, p_r80)}}
        blk["all"]["delta"] = decisions(truth, y_full, y_r80)
        pk = per[known].astype(np.int64)
        blk["matched"] = {"full_pool_root": score(truth[known], y_full[known], p_full[known]),
                          "r80_fit_root": score(truth[known], y_r80[known], p_r80[known]),
                          "persistence": score(truth[known], pk, np.eye(4, dtype=np.float32)[pk]),
                          "delta": decisions(truth[known], y_full[known], y_r80[known])}
        per_pair[name] = {"horizon": h, "target_month": t, "origin_month": ml(o), "scores": blk,
                          "fit_fraction": {"fit_rows": int(len(fit)), "window_rows": int(len(win)), "legal_rows": int(len(legal)),
                                           "fit_over_window": len(fit) / len(win), "fit_over_legal": len(fit) / len(legal)},
                          "window_minus_legal": {"rows": len(extra), "areas": extra_areas}}
        identity["pairs"][name] = {"v1_record": str(ubj1.with_suffix(".json")), "v1_booster_sha256": rec["booster_sha256"],
                                   "v1_fit_keys_sha256": rec["fit_keys_sha256"], "d34_root_booster_sha256": root["root_booster_sha256"],
                                   "d34_fitting_keys_sha256": root["fitting_keys_sha256"], "snapshot_sha256": ssha,
                                   "membership_sha256": sha(STAGE / "roots" / name / "fold_membership.csv.gz"),
                                   "root_target_predictions_sha256": sha(STAGE / "roots" / name / "root_target_predictions.csv"),
                                   "d34_root_checkpoint": str(ubj2),
                                   "replay": "exact float32 equality for both boosters", "e3_rows": int(len(areas))}
        df = pd.DataFrame({"pair": name, "horizon": h, "target_month": t, "area": areas, "truth": truth,
                           "persistence_code": per, "y_full_pool": y_full, "y_r80_fit": y_r80})
        for k, l in enumerate(LABELS):
            df[f"p_full_pool_{l}"] = p_full[:, k]; df[f"p_r80_fit_{l}"] = p_r80[:, k]
        rows_all.append(df)
        print(f"{name}: replay exact (2 models); E3 {len(areas)}; persistence known {int(known.sum())}", flush=True)
    # ---- aggregates
    def agg(names):
        out = {}
        for cohort, arms in (("all", ("full_pool_root", "r80_fit_root")), ("matched", ("full_pool_root", "r80_fit_root", "persistence"))):
            bl = [per_pair[n]["scores"][cohort] for n in names]
            res = {a: pooled(bl, a) for a in arms}
            res["pooled_delta_full_minus_r80"] = {"crisis_f1": res["full_pool_root"]["crisis_f1"] - res["r80_fit_root"]["crisis_f1"],
                                                  "macro_f1_fourclass": res["full_pool_root"]["macro_f1_fourclass"] - res["r80_fit_root"]["macro_f1_fourclass"],
                                                  "crisis_brier": res["full_pool_root"]["crisis_brier"] - res["r80_fit_root"]["crisis_brier"],
                                                  "decisions": {k: int(sum(b["delta"][k] for b in bl)) for k in bl[0]["delta"]}}
            res["mean_of_pair_deltas_full_minus_r80"] = {k: float(np.mean([b["full_pool_root"][k] - b["r80_fit_root"][k] for b in bl]))
                                                         for k in ("crisis_f1", "macro_f1_fourclass", "crisis_brier")}
            out[cohort] = res
        return out
    names = list(per_pair)
    summary = {"rule": "D44 zero-fit: v1 full-pool global vs D34 r80-FIT root on identical E3 keys; 30 saved-model raw replays; "
                       "persistence only on the available cohort and its crisis_brier is a one-hot reference, not a calibrated probability; "
                       "pooled and mean-of-pair deltas are distinct quantities",
               "per_pair": per_pair, "by_horizon": {f"H{h}": agg([n for n in names if per_pair[n]["horizon"] == h]) for h in (4, 8, 12)},
               "overall_15": agg(names), "models_replayed": 2 * len(per_pair), "fits": 0}
    identity = {**identity, "script_sha256": sha(Path(__file__)), "schema_sha256": SCHEMA_SHA,
                "v1_gscreen_predictions_sha256": sha(V1 / "gscreen" / "predictions.csv.gz"),
                "d34_runtime": json.loads((D34 / "prepared" / "manifests" / "identity.json").read_text(encoding="utf-8"))["runtime"],
                     "producers": {"v1": {"git_head": V1_GIT, "code_sha256": V1_CODE}, "d34": {"git_head": D34_GIT, "code_sha256": D34_CODE}},
                     "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "xgboost": xgb.__version__},
                "v1_runtime": json.loads((V1 / "prepared" / "manifests" / "identity.json").read_text(encoding="utf-8"))["runtime"]}
    OUT.mkdir(parents=True)
    with gzip.open(OUT / "rows_E3.csv.gz", "wt", encoding="utf-8", newline="") as fh:
        pd.concat(rows_all, ignore_index=True).to_csv(fh, index=False, float_format="%.17g")
    identity["rows_E3_sha256"] = sha(OUT / "rows_E3.csv.gz")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (OUT / "identity.json").write_text(json.dumps(identity, indent=1), encoding="utf-8")
    o = summary["overall_15"]
    print(json.dumps({c: {a: {k: o[c][a][k] for k in ("n", "crisis_f1", "crisis_brier", "macro_f1_fourclass") if k in o[c][a]}
                          for a in o[c] if isinstance(o[c][a], dict) and "n" in o[c][a]} for c in o}, indent=1))
    print(f"D44 completed: {OUT}")


if __name__ == "__main__":
    try:
        main()
    except Stop as exc:
        print(f"STOP: {exc}")
        sys.exit(2)
