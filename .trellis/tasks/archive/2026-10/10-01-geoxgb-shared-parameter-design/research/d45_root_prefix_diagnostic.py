"""D45 / A19: saved D34 root boosting-prefix learning curve (FIT / C / E3), zero fits.

For each of the 21 saved D34 roots: full raw replay must equal the saved C (p_root_*) and E3
(p_pooled_*) probabilities exactly (float32, round-trip parsing); then predict(iteration_range=(0, r))
for r = T/4, T/2, T (never (0, 0)) on FIT, C and E3. Fresh DMatrix per predict. Only pure package
modules are imported (src.experiment.plan, src.metrics.fourclass); a static AST guard forbids fit or
continuation calls. Snapshots are read with a pyarrow filter target_month <= 2020-12.
FIT probabilities are scored and discarded; C/E3 prefix probabilities are kept externally.
"""
import ast, gzip, hashlib, json, platform, sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

B = Path(r"C:\Users\swl00\geoxgb_runs")
PKG = Path(r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster\FEWSNETGeoXGBExperiment")
D34 = B / "geoxgb-d34-e1-brier-20261002"
STAGE = D34 / "stage1_e1pair"
OUT = B / "geoxgb-d45-root-prefix-20261002"
SCHEMA_SHA = "51b6f8b21b76a78510522c34e2d1f2a648b7aec768bbcac3dd2318669fa13349"
D34_GIT, D34_CODE = "7b2bf6fe482d0a77a664f3627e493934d976696d", "cad8439de1d1c838ddd8482c6782f4cfa205e5c28670e45dad2ddbe92e69421b"
RUNTIME = {"python": "3.12.10", "numpy": "2.2.6", "pandas": "2.2.3", "xgboost": "3.0.0"}
G = {4: "G1", 8: "G4", 12: "G2"}
TARGETS = ("2018-06", "2018-10", "2019-02", "2019-06", "2019-10", "2020-02", "2020-06")
PAIRS = [(h, t) for h in (4, 8, 12) for t in TARGETS]
ROUNDS = {4: (50, 100, 200), 8: (100, 200, 400), 12: (100, 200, 400)}
PREFIX = ("quarter", "half", "full")
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
            if any(m and (any(x in m for x in ("native_xgb", "main_model", "stage3", "GeoRF", "scripts"))
                          or (m.startswith("src") and m not in ("src.experiment", "src.metrics"))) for m in mods):
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


def predict(booster, X, r=None):
    X = np.where(np.isinf(X), np.nan, X)
    dm = xgb.DMatrix(X, missing=np.nan, nthread=4)                  # fresh DMatrix per predict
    out = booster.predict(dm) if r is None else booster.predict(dm, iteration_range=(0, r))
    check(out.ndim == 2 and out.shape[1] == 4, f"unexpected output shape {out.shape}")
    return out.astype(np.float32)


def score(t, proba=None, pred=None, logloss=True):
    if pred is None:
        pred = proba.argmax(1)
    z, a = t >= 2, pred >= 2
    tp, fp, fn = int((z & a).sum()), int((~z & a).sum()), int((z & ~a).sum())
    f1 = Fraction(2 * tp, 2 * tp + fp + fn) if (2 * tp + fp + fn) else Fraction(0)
    out = {"n": int(len(t)), "tp": tp, "fp": fp, "fn": fn, "tn": int(len(t)) - tp - fp - fn,
           "crisis_f1": float(f1), "crisis_f1_exact": str(f1), "macro_f1_fourclass": fourclass.macro_f1(t, pred),
           "confusion_fourclass": fourclass.confusion(t, pred).astype(int).tolist()}
    if proba is not None:
        p64 = proba.astype(np.float64)
        pc = p64[:, 2] + p64[:, 3]
        out["crisis_brier"] = float(np.mean((pc - z.astype(np.float64)) ** 2))
        if logloss:
            pt = p64[np.arange(len(t)), t]
            check(np.all(pt > 0), "non-positive true-class probability (no clipping)")
            ll = float(-np.mean(np.log(pt)))
            check(np.isfinite(ll), "non-finite log loss")
            out["logloss_fourclass"] = ll
    return out


def decisions(t, a, b):
    za, zb, z = a >= 2, b >= 2, t >= 2
    ch = za != zb
    return {"changed": int(ch.sum()), "corrected": int((ch & (za == z)).sum()), "spoiled": int((ch & (zb == z)).sum()),
            "tp_delta": int((za & z).sum() - (zb & z).sum()), "fp_delta": int((za & ~z).sum() - (zb & ~z).sum())}


def describe(t, months):
    return {"n": int(len(t)), "label_dates": int(len(set(months))),
            "class_prevalence": (np.bincount(t, minlength=4) / max(len(t), 1)).tolist()}


def pooled(blocks):
    conf = np.sum([np.array(b["confusion_fourclass"]) for b in blocks], axis=0)
    tp = sum(b["tp"] for b in blocks); fp = sum(b["fp"] for b in blocks); fn = sum(b["fn"] for b in blocks)
    n = sum(b["n"] for b in blocks)
    f1 = Fraction(2 * tp, 2 * tp + fp + fn) if (2 * tp + fp + fn) else Fraction(0)
    out = {"n": n, "tp": tp, "fp": fp, "fn": fn, "crisis_f1": float(f1), "crisis_f1_exact": str(f1),
           "macro_f1_fourclass": fourclass.macro_f1_from_matrix(conf)}
    for k in ("crisis_brier", "logloss_fourclass"):
        if all(k in b for b in blocks):
            out[k] = float(sum(b[k] * b["n"] for b in blocks) / n)
    return out


def main():
    guard_no_fit_calls()
    check(not OUT.exists(), f"{OUT} exists")
    env = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "xgboost": xgb.__version__}
    check(env == RUNTIME, f"runtime differs from the pinned environment: {env}")
    schema_path = PKG / "feature-schema.json"
    check(sha(schema_path) == SCHEMA_SHA, "schema file sha differs from the pinned value")
    features = json.loads(schema_path.read_text(encoding="utf-8"))["ordered_features"]
    check(len(features) == 162 and len(set(features)) == 162 and "hist_phase_o00" in features, "schema feature list")
    ident = json.loads((D34 / "prepared" / "manifests" / "identity.json").read_text(encoding="utf-8"))
    check(ident["git_head"] == D34_GIT and ident["code"]["sha256"] == D34_CODE, "D34 producer identity differs")
    outputs_path = D34 / "prepared" / "manifests" / "outputs.json"
    check(sha(outputs_path) == ident["outputs_sha256"], "D34 outputs manifest sha differs")
    outputs = json.loads(outputs_path.read_text(encoding="utf-8"))      # hash record only; no ledger is opened
    cols = ["area", "target_month", "horizon", "class_code"] + features
    check(len(cols) == len(set(cols)), "duplicate snapshot column selection")
    per_pair, identity, keep = {}, {"pairs": {}}, []
    for h in (4, 8, 12):
        sp = D34 / "prepared" / f"snapshot_h{h}.parquet"
        snap = pd.read_parquet(sp, columns=cols, filters=[("target_month", "<=", MAX_MONTH)])
        check(list(snap.columns) == cols and (snap["horizon"] == h).all(), f"h{h}: snapshot columns/horizon")
        check(not snap.duplicated(["area", "target_month"]).any(), f"h{h}: duplicate snapshot keys")
        snap = snap.set_index(["area", "target_month"])
        ssha = sha(sp)
        check(ssha == outputs[f"snapshot_h{h}.parquet"], f"h{h}: snapshot sha differs from D34 prepared")
        for t in TARGETS:
            g = G[h]; o = mi(t) - h; name = f"h{h}_{t}_{g}_r80_s42_e1pair"
            cand = f"h{h}_{t}_{g}_L1_r80_s42_e1brier_gt0"
            root = json.loads((STAGE / "roots" / name / "root.json").read_text(encoding="utf-8"))
            check(root["horizon"] == h and root["target_month"] == t and root["origin_month"] == ml(o) and root["g_config"] == g,
                  f"{name}: root.json pair fields")
            params, rounds = plan.booster_params(plan.G_CONFIGS[g])
            check(rounds == ROUNDS[h][2] and ROUNDS[h][0] * 4 == rounds and ROUNDS[h][1] * 2 == rounds, f"{name}: prefix schedule")
            check(root["root_fit"]["params"] == params and int(root["root_fit"]["rounds_total"]) == rounds, f"{name}: params/rounds")
            ubj = STAGE / "checkpoints" / cand / "xgb_root.ubj"
            check(sha(ubj) == root["root_booster_sha256"] == root["root_fit"]["booster_sha256"], f"{name}: root booster sha")
            booster = xgb.Booster(); booster.load_model(bytearray(ubj.read_bytes()))
            check(booster.num_boosted_rounds() == rounds, f"{name}: num_boosted_rounds")
            mem = pd.read_csv(STAGE / "roots" / name / "fold_membership.csv.gz")
            check(not mem.duplicated(["area", "target_month"]).any(), f"{name}: duplicate membership keys")
            fit = mem[mem.role == "fitting"]; cmem = mem[mem.role == "confirmation"]; emem = mem[mem.role == "heldout_target"]
            smem = mem[mem.role == "validation"]
            check(cand in root["candidates"] and root["rows"] == {"fitting": len(fit), "validation": len(smem) + len(cmem),
                                                                  "heldout_target": len(emem)}, f"{name}: root.json rows/candidates")
            check(keys_sha(fit.area, fit.target_month.map(mi)) == root["fitting_keys_sha256"], f"{name}: FIT keys")
            check((emem.target_month == t).all(), f"{name}: E3 rows not at T")
            cp = pd.read_csv(STAGE / "candidates" / cand / "confirmation_predictions.csv.gz", float_precision="round_trip",
                             dtype={"target_month": str})
            rt = pd.read_csv(STAGE / "roots" / name / "root_target_predictions.csv", float_precision="round_trip")
            check(not cp.duplicated(["area", "target_month"]).any() and rt.FEWSNET_admin_code.is_unique, f"{name}: duplicate saved keys")
            check(set(zip(cp.area, cp.target_month)) == set(zip(cmem.area, cmem.target_month)), f"{name}: C keys")
            check(set(rt.FEWSNET_admin_code) == set(emem.area), f"{name}: E3 keys")
            parts = {}
            for part, a, m in (("FIT", fit.area.to_numpy(np.int64), fit.target_month.map(mi).to_numpy(np.int64)),
                               ("C", cp.area.to_numpy(np.int64), cp.target_month.map(mi).to_numpy(np.int64)),
                               ("E3", rt.FEWSNET_admin_code.to_numpy(np.int64), np.full(len(rt), mi(t), dtype=np.int64))):
                rows = snap.reindex(pd.MultiIndex.from_arrays([a, m]))
                check(rows["class_code"].notna().all(), f"{name}: {part} keys missing from the snapshot")
                X = rows[features].to_numpy(dtype=float)
                check(X.shape == (len(a), 162), f"{name}: {part} matrix is not n x 162")
                parts[part] = (a, m, rows["class_code"].to_numpy(np.int64), X, rows["hist_phase_o00"].to_numpy(float))
            memc = cmem.set_index(["area", "target_month"]).loc[list(zip(cp.area, cp.target_month)), "class_code"].to_numpy()
            check((parts["C"][2] == cp.y_true.to_numpy()).all() and (parts["C"][2] == memc).all(), f"{name}: C truth")
            meme = emem.set_index("area").loc[rt.FEWSNET_admin_code, "class_code"].to_numpy()
            check((parts["E3"][2] == rt.y_true_code.to_numpy()).all() and (parts["E3"][2] == meme).all(), f"{name}: E3 truth")
            check((parts["FIT"][2] == fit.class_code.to_numpy()).all(), f"{name}: FIT truth")
            # ---- full-root replay gate (exact float32) and prefix consistency
            for part, saved, ysaved in (("C", cp[[f"p_root_{l}" for l in LABELS]], cp.y_root),
                                        ("E3", rt[[f"p_pooled_{l}" for l in LABELS]], rt.y_pred_pooled_code)):
                full = predict(booster, parts[part][3])
                s32 = saved.to_numpy(np.float64).astype(np.float32)
                check(np.array_equal(full, s32), f"{name}: {part} full replay not exactly equal (max {np.abs(full - s32).max()})")
                check((full.argmax(1) == ysaved.to_numpy()).all(), f"{name}: {part} full replay labels")
                check(np.array_equal(predict(booster, parts[part][3], rounds), full), f"{name}: {part} prefix (0,T) != full")
            # ---- prefixes
            res = {"horizon": h, "target_month": t, "origin_month": ml(o), "rounds": dict(zip(PREFIX, ROUNDS[h])), "parts": {}}
            keep_parts = {}
            for part in ("FIT", "C", "E3"):
                a, m, y, X, ph = parts[part]
                probs = {pf: predict(booster, X, r) for pf, r in zip(PREFIX, ROUNDS[h])}
                blk = {"describe": describe(y, m), "prefix": {pf: score(y, probs[pf]) for pf in PREFIX}}
                blk["decisions_vs_full"] = {pf: decisions(y, probs[pf].argmax(1), probs["full"].argmax(1)) for pf in PREFIX[:2]}
                if part == "E3":
                    k = np.isfinite(ph)
                    check(np.isin(ph[k], [1, 2, 3, 4, 5]).all(), f"{name}: origin phase values")
                    per = np.minimum(ph[k], 4).astype(np.int64) - 1
                    blk["matched"] = {"n": int(k.sum()), "prefix": {pf: score(y[k], probs[pf][k]) for pf in PREFIX},
                                      "persistence_reference": score(y[k], np.eye(4, dtype=np.float32)[per], pred=per, logloss=False),
                                      "decisions_vs_full": {pf: decisions(y[k], probs[pf][k].argmax(1), probs["full"][k].argmax(1))
                                                            for pf in PREFIX[:2]}}
                    keep_parts[part] = (a, m, y, probs, np.where(np.isfinite(ph), np.minimum(ph, 4) - 1, np.nan))
                elif part == "C":
                    keep_parts[part] = (a, m, y, probs, None)
                res["parts"][part] = blk
            per_pair[name] = res
            identity["pairs"][name] = {"g_config": g, "rounds_total": rounds, "candidate": cand,
                                       "root_booster_sha256": root["root_booster_sha256"], "checkpoint": str(ubj),
                                       "fitting_keys_sha256": root["fitting_keys_sha256"], "snapshot_sha256": ssha,
                                       "membership_sha256": sha(STAGE / "roots" / name / "fold_membership.csv.gz"),
                                       "confirmation_predictions_sha256": sha(STAGE / "candidates" / cand / "confirmation_predictions.csv.gz"),
                                       "root_target_predictions_sha256": sha(STAGE / "roots" / name / "root_target_predictions.csv"),
                                       "full_replay": "exact float32 on C and E3; prefix (0,T) equals full",
                                       "rows": {p: int(len(parts[p][0])) for p in parts}}
            for part, (a, m, y, probs, per) in keep_parts.items():
                df = pd.DataFrame({"root": name, "part": part, "horizon": h, "area": a, "target_month": [ml(x) for x in m], "truth": y})
                if per is not None:
                    df["persistence_code"] = per
                for pf in PREFIX:
                    for k, l in enumerate(LABELS):
                        df[f"p_{pf}_{l}"] = probs[pf][:, k]
                keep.append(df)
            print(f"{name}: gates passed; FIT {len(parts['FIT'][0])} C {len(parts['C'][0])} E3 {len(parts['E3'][0])}", flush=True)
        del snap

    def agg(names, part, cohort=None):
        out = {}
        def blocks(pf):
            return [(per_pair[n]["parts"][part]["matched"] if cohort else per_pair[n]["parts"][part])["prefix"][pf] for n in names]
        for pf in PREFIX:
            out[pf] = {"pooled": pooled(blocks(pf)),
                       "mean_of_pairs": {k: float(np.mean([b[k] for b in blocks(pf)])) for k in ("crisis_f1", "macro_f1_fourclass", "crisis_brier", "logloss_fourclass")}}
        for pf in PREFIX[:2]:
            out[f"{pf}_minus_full"] = {
                "pooled": {k: out[pf]["pooled"][k] - out["full"]["pooled"][k] for k in ("crisis_f1", "macro_f1_fourclass", "crisis_brier", "logloss_fourclass")},
                "mean_of_pair_deltas": {k: float(np.mean([a[k] - b[k] for a, b in zip(blocks(pf), blocks("full"))]))
                                        for k in ("crisis_f1", "macro_f1_fourclass", "crisis_brier", "logloss_fourclass")},
                "decisions": {k: int(sum((per_pair[n]["parts"][part]["matched"] if cohort else per_pair[n]["parts"][part])["decisions_vs_full"][pf][k]
                                         for n in names)) for k in ("changed", "corrected", "spoiled", "tp_delta", "fp_delta")}}
        if cohort:
            out["persistence_reference"] = pooled([per_pair[n]["parts"][part]["matched"]["persistence_reference"] for n in names])
        return out

    by_h = {}
    for h in (4, 8, 12):
        names = [n for n in per_pair if per_pair[n]["horizon"] == h]
        by_h[f"H{h}"] = {"rounds": dict(zip(PREFIX, ROUNDS[h])), "FIT": agg(names, "FIT"), "C": agg(names, "C"),
                         "E3": agg(names, "E3"), "E3_persistence_matched": agg(names, "E3", cohort="matched")}
    summary = {"rule": "D45 zero-fit exact prefixes of each saved D34 root path (iteration_range (0,r), r=T/4,T/2,T; never (0,0)); "
                       "FIT in-sample, C in-window random holdout (historical interpolation), E3 forward; per-H only (absolute rounds "
                       "differ by H), no cross-H pooling, no best-round selection; persistence has no log loss (one-hot); decisions_vs_full: "
                       "'corrected' = prefix right where full wrong, 'spoiled' = prefix wrong where full right",
               "per_pair": per_pair, "by_horizon": by_h, "models_loaded": len(per_pair), "prefix_evaluations": 3 * len(per_pair),
               "part_predictions": 9 * len(per_pair), "fits": 0}
    OUT.mkdir(parents=True)
    with gzip.open(OUT / "rows_C_E3_prefix.csv.gz", "wt", encoding="utf-8", newline="") as fh:
        pd.concat(keep, ignore_index=True).to_csv(fh, index=False, float_format="%.17g")
    identity = {**identity, "script_sha256": sha(Path(__file__)), "schema_sha256": SCHEMA_SHA, "environment": env,
                "producer": {"git_head": D34_GIT, "code_sha256": D34_CODE}, "d34_runtime": ident["runtime"],
                "d34_outputs_manifest_sha256": ident["outputs_sha256"],
                "rows_C_E3_prefix_sha256": sha(OUT / "rows_C_E3_prefix.csv.gz")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (OUT / "identity.json").write_text(json.dumps(identity, indent=1), encoding="utf-8")
    for h in (4, 8, 12):
        v = by_h[f"H{h}"]
        print(f"H{h} rounds {v['rounds']}")
        for part in ("FIT", "C", "E3", "E3_persistence_matched"):
            print("  ", part, {pf: {k: round(v[part][pf]["pooled"][k], 6) for k in ("crisis_f1", "crisis_brier", "logloss_fourclass")} for pf in PREFIX})
    print(f"D45 completed: {OUT}")


if __name__ == "__main__":
    try:
        main()
    except Stop as exc:
        print(f"STOP: {exc}")
        sys.exit(2)
