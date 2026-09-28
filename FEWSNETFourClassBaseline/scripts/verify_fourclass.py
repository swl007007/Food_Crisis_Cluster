"""Independent reconstruction and replay for a completed run (A2-A8).

python scripts/verify_fourclass.py --run-dir runs/<id>

Never modifies a run artifact; writes ``<run>/verification/`` (fresh). Checks:

1. Source/snapshot identities: prepared outputs still match their recorded hashes.
2. Feature recomputation: an independent row-wise re-derivation of a keyed sample
   from the raw panel (exact offsets, windows, events) equals the snapshot.
3. Stage 1 ledger: every scheduled fold completed; training windows are [O-35, O);
   no training label after 2020-12; per-fold held-out scores recompute from rows.
4. Stage 2: plan weights recompute from the candidate scores; consensus route.
5. Stage 3: fold training keys reproduce the window rule; prediction keys equal the
   baseline ledger; pseudo rows are zero; routing counts.
6. Report: every arm metric and contrast point estimate recomputes with sklearn from
   keyed_evaluation.csv.gz; bootstrap draws recompute from saved multiplicities.
7. Replay: the saved Stage 3 estimator bundles of the first and last fitted fold per
   horizon are loaded (no refit) and must reproduce labels and probabilities exactly;
   retained Stage 1 checkpoints reload and reproduce the saved held-out predictions.
"""
import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
SCHEMA = PACKAGE / "feature-schema.json"
HORIZONS = (4, 8, 12)
PROBA_TOLERANCE = 1e-12
SCOPE = {4: 1, 8: 2, 12: 3}


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mi(label: str) -> int:
    y, m = label.split("-")
    return int(y) * 12 + int(m) - 1


def macro(truth, pred):
    return float(f1_score(truth, pred, labels=[0, 1, 2, 3], average="macro", zero_division=0))


def check(results, name, ok, detail=None):
    results.append({"check": name, "passed": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (f" :: {detail}" if detail is not None and not ok else ""), flush=True)


# ---------------------------------------------------------------------------------
# independent feature derivation (plain loops over the raw panel, no shared code)
# ---------------------------------------------------------------------------------

def rederive(panel, obs, area, target, horizon):
    origin = target - horizon
    row = {}
    series = panel[area]
    history = [(mth, ph) for mth, ph in obs.get(area, []) if mth <= origin]
    lookup = dict(history)
    for k in (0, 4, 8, 12):
        ph = lookup.get(origin - k, np.nan)
        row[f"hist_phase_o{k:02d}"] = ph
    for W in (12, 24, 36):
        win = [ph for mth, ph in history if origin - W + 1 <= mth <= origin]
        row[f"hist_w{W}_n_obs"] = float(len(win))
        row[f"hist_w{W}_frac4or5"] = (sum(p == 4 for p in win) / len(win)) if win else np.nan
        if len(win) >= 2:
            row[f"hist_w{W}_up_rate"] = sum(b > a for a, b in zip(win, win[1:])) / (len(win) - 1)
        else:
            row[f"hist_w{W}_up_rate"] = np.nan
    crisis = [mth for mth, ph in history if ph >= 3]
    row["hist_crisis_age"] = float(origin - crisis[-1]) if crisis else np.nan
    run = 0
    for mth, ph in reversed(history):
        if ph == history[-1][1]:
            run += 1
        else:
            break
    row["hist_run_n"] = float(run) if history else np.nan
    row["EVI"] = series["EVI"].get(origin, np.nan)
    row["EVI_l7"] = series["EVI"].get(origin - 7, np.nan)
    vals = [series["WFP_Price"].get(origin - k, np.nan) for k in range(1, 13)]
    row["WFP_Price_m12"] = float(np.sum(vals)) if not np.isnan(vals).any() else np.nan
    row["gini"] = series["gini"].get(origin, np.nan)
    row["target_year"] = float(target // 12)
    return row


def verify_features(run, results, n_keys=400):
    import polars as pl
    sources = json.loads((run / "prepared" / "manifests" / "sources.json").read_text(encoding="utf-8"))
    raw = pl.read_csv(sources["sources"]["panel"]["path"], infer_schema_length=100000,
                      columns=["FEWSNET_admin_code", "date", "fews_ipc", "EVI", "WFP_Price", "gini"]).to_pandas()
    raw["m"] = raw["date"].map(mi)
    rng = np.random.default_rng(7)
    mismatches = []
    for horizon in HORIZONS:
        snap = pd.read_parquet(run / "prepared" / f"snapshot_h{horizon}.parquet")
        sample = snap.iloc[rng.choice(len(snap), size=n_keys, replace=False)]
        areas = set(sample["area"])
        sub = raw[raw["FEWSNET_admin_code"].isin(areas)]
        panel = {a: {c: dict(zip(g["m"], g[c])) for c in ("EVI", "WFP_Price", "gini")}
                 for a, g in sub.groupby("FEWSNET_admin_code")}
        obs = {a: [(int(mth), int(min(ph, 4))) for mth, ph in zip(g["m"], g["fews_ipc"])]
               for a, g in sub[sub["fews_ipc"].notna()].sort_values("m").groupby("FEWSNET_admin_code")}
        for _, key in sample.iterrows():
            expected = rederive(panel, obs, int(key["area"]), int(key["target_month"]), horizon)
            for name, value in expected.items():
                got = key[name]
                same = (np.isnan(value) and np.isnan(got)) or (not np.isnan(value) and np.isclose(value, got, rtol=0, atol=1e-9))
                if not same:
                    mismatches.append((horizon, int(key["area"]), int(key["target_month"]), name, value, got))
    check(results, "features: independent re-derivation of 1200 sampled keys x 17 columns",
          not mismatches, mismatches[:10])
    for horizon in HORIZONS:
        snap = pd.read_parquet(run / "prepared" / f"snapshot_h{horizon}.parquet")
        check(results, f"features h{horizon}: origin = target - horizon on every key",
              bool((snap["target_month"] - snap["origin_month"] == horizon).all()))


def verify_stage1(run, results):
    schedule = json.loads((run / "prepared" / "manifests" / "schedule.json").read_text(encoding="utf-8"))
    scheduled = [f for f in schedule["stage1"] if f["status"] == "scheduled"]
    bad, rescored = [], []
    for fold in scheduled:
        name = f"fs{fold['scope']}_{fold['target_month']}"
        d = run / "stage1" / "folds" / name
        cand = json.loads((d / "candidate.json").read_text(encoding="utf-8"))
        origin = mi(fold["origin_month"])
        observed = [mi(x) for x in cand["train_label_months_observed"]]
        if min(observed) < origin - 35 or max(observed) >= origin or max(observed) > mi("2020-12"):
            bad.append(name)
        preds = pd.read_csv(d / "target_predictions.csv")
        for arm, key in (("y_pred_partitioned_code", "macro_f1"), ("y_pred_pooled_code", "macro_f1_base")):
            if not np.isclose(macro(preds["y_true_code"], preds[arm]), cand["scores"][key], rtol=0, atol=1e-12):
                rescored.append((name, key))
        pseudo = {e["pseudo_rows"] for e in cand["fits"]["georf_fit_log"]}
        if pseudo != {4}:
            bad.append(f"{name}:pseudo_rows={pseudo}")
        members = pd.read_csv(d / "fold_membership.csv.gz")
        if set(members.loc[members.role != "heldout_target", "target_month"].map(mi)) - set(range(origin - 35, origin)):
            bad.append(f"{name}:membership")
    check(results, f"stage1: {len(scheduled)} scheduled folds all completed within [O-35,O) and <= 2020-12 with 4 pseudo rows per fit",
          not bad, bad)
    check(results, "stage1: held-out macro F1 recomputes with sklearn from saved rows", not rescored, rescored)


def verify_stage2(run, results):
    from scripts.step4_similarity_matrix import compute_plan_weights
    ledger = pd.read_csv(run / "stage2" / "candidate_ledger.csv")
    completed = ledger[ledger["status"] == "completed"]
    weights = pd.read_csv(run / "stage2" / "plan_weights.csv")
    f = np.clip(completed["macro_f1"].to_numpy(), 1e-6, 1 - 1e-6)
    b = np.clip(completed["macro_f1_base"].to_numpy(), 1e-6, 1 - 1e-6)
    expected = np.maximum(np.log(f / (1 - f)) - np.log(b / (1 - b)), 0)
    check(results, "stage2: D9 weights recompute", np.allclose(expected, weights["weight"], rtol=0, atol=1e-12))
    record = json.loads((run / "stage2" / "consensus.json").read_text(encoding="utf-8"))
    route_ok = (record["route"] == "null_consensus") == bool((expected == 0).all())
    check(results, f"stage2: consensus route '{record['route']}' matches the weights", route_ok)
    if record["route"] == "learned_map":
        check(results, "stage2: cluster map digest", sha256(record["cluster_map"]) == record["cluster_map_sha256"])


def verify_stage3(run, results):
    baselines = pd.read_csv(run / "prepared" / "ledgers" / "baselines.csv")
    for horizon in HORIZONS:
        out = run / "stage3" / f"h{horizon}"
        manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
        preds = pd.read_csv(out / "predictions.csv.gz")
        base_keys = set(zip(baselines.loc[baselines.horizon == horizon, "area"],
                            baselines.loc[baselines.horizon == horizon, "target_label"]))
        check(results, f"stage3 h{horizon}: prediction keys == baseline truth keys",
              base_keys == set(zip(preds["area"], preds["target_month"])))
        wrong = []
        for fold in manifest["folds"]:
            if fold["status"] != "fitted":
                continue
            rec = json.loads((out / "folds" / fold["target_month"] / "fold.json").read_text(encoding="utf-8"))
            keys = pd.read_csv(out / "folds" / fold["target_month"] / "training_keys.csv.gz")
            origin = mi(fold["origin_month"])
            if keys["target_month"].min() < origin - 35 or keys["target_month"].max() >= origin:
                wrong.append(fold["target_month"])
            if rec["pseudo_rows"] != 0 or len(keys) != rec["rows"]["train"]:
                wrong.append(fold["target_month"] + ":rows")
            digest = hashlib.sha256(np.ascontiguousarray(keys[["area", "target_month"]].to_numpy(np.int64)).tobytes()).hexdigest()
            if digest != rec["train_keys_sha256"]:
                wrong.append(fold["target_month"] + ":keys")
        check(results, f"stage3 h{horizon}: training windows [O-35,O), real rows only, keys match digests", not wrong, wrong)
        probs = preds.filter(like="p_pooled_").to_numpy()
        check(results, f"stage3 h{horizon}: hard predictions are fixed-axis argmax",
              bool((probs.argmax(axis=1) == preds["y_pred_pooled_code"]).all()))


def verify_report(run, results):
    report = json.loads((run / "report" / "report.json").read_text(encoding="utf-8"))
    keyed = pd.read_csv(run / "report" / "keyed_evaluation.csv.gz")
    col = {"partitioned": "y_pred_partitioned_code", "pooled": "y_pred_pooled_code",
           "expert": "expert_code", "persistence": "persistence_code"}
    bad = []
    for cohort, entry in report["metrics"].items():
        h = int(cohort.split("_h")[1])
        rows = keyed[keyed.horizon == h]
        rows = rows[rows.persistence_code.notna()]
        if cohort.startswith("main") and h != 12:
            rows = rows[rows.expert_code.notna()]
        if len(rows) != entry["n"]:
            bad.append((cohort, "n"))
        for arm, summary in entry["arms"].items():
            if not np.isclose(macro(rows.truth_code, rows[col[arm]].astype(int)), summary["macro_f1"], rtol=0, atol=1e-12):
                bad.append((cohort, arm))
    check(results, "report: every cohort size and arm macro F1 recomputes with sklearn", not bad, bad)
    draws = pd.read_csv(run / "report" / "bootstrap_draws.csv.gz")
    countries = report["bootstrap"]["countries"]
    sample = draws.sample(n=min(25, len(draws)), random_state=3)
    bad = []
    for _, draw in sample.iterrows():
        mult = draw[countries].astype(int)
        for cohort, entry in report["metrics"].items():
            h = int(cohort.split("_h")[1])
            rows = keyed[(keyed.horizon == h) & keyed.persistence_code.notna()]
            if cohort.startswith("main") and h != 12:
                rows = rows[rows.expert_code.notna()]
            reps = np.repeat(np.arange(len(rows)), rows["country"].map(mult).to_numpy())
            rep = rows.iloc[reps]
            for arm in entry["arms"]:
                if not np.isclose(macro(rep.truth_code, rep[col[arm]].astype(int)), draw[f"{cohort}:{arm}"], rtol=0, atol=1e-9):
                    bad.append((int(draw["draw"]), cohort, arm))
    check(results, "report: 25 bootstrap draws recompute by row replication from saved multiplicities", not bad, bad[:5])
    check(results, f"report: {report['bootstrap']['draws_accepted']} of 2000 draws accepted",
          report["bootstrap"]["complete"])


def verify_replay(run, results, out):
    # Stage 3: LOAD the saved estimator bundles of the first and last fitted fold per
    # horizon (no refit) and reproduce labels and all four probabilities exactly.
    from scripts.compare_partitioned_vs_pooled_rf_k40_nc4 import bundle_proba, load_bundle
    from src.metrics import fourclass as fc
    replay_rows = []
    for horizon in HORIZONS:
        base = run / "stage3" / f"h{horizon}"
        manifest = json.loads((base / "run_manifest.json").read_text(encoding="utf-8"))
        fitted = [f["target_month"] for f in manifest["folds"] if f["status"] == "fitted"]
        saved = pd.read_csv(base / "predictions.csv.gz")
        snap = pd.read_parquet(run / "prepared" / f"snapshot_h{horizon}.parquet")
        for month in (fitted[0], fitted[-1]):
            fold = base / "folds" / month
            record = json.loads((fold / "fold.json").read_text(encoding="utf-8"))
            models = sorted(r for r in record["outputs"] if r.startswith("models/"))
            hashes_ok = all(sha256(fold / r) == record["outputs"][r] for r in models)
            original = saved[saved.target_month == month].reset_index(drop=True)
            test = snap[snap.target_month == mi(month)].set_index("area").loc[original["area"]]
            bundles = {Path(r).name.removesuffix(".pkl.xz"): load_bundle(fold / r) for r in models}
            X = test[bundles["pooled"]["features"]].to_numpy(dtype=float)
            p_pooled = bundle_proba(bundles["pooled"], X)
            p_part = p_pooled.copy()
            for cid in sorted(original["cluster_id"].unique()):
                rows = (original["cluster_id"] == cid).to_numpy()
                key = f"local_{cid}"
                if key in bundles:
                    p_part[rows] = bundle_proba(bundles[key], X[rows])
            # sklearn sums per-tree probabilities across n_jobs threads, so the summation
            # order (and the last bit) can differ between calls; labels must be identical.
            diff = max(float(np.abs(original[f"p_{arm}_{c}"].to_numpy() - p[:, k]).max())
                       for arm, p in (("pooled", p_pooled), ("partitioned", p_part))
                       for k, c in enumerate(fc.CLASS_LABELS))
            same = hashes_ok and diff <= PROBA_TOLERANCE
            same = same and np.array_equal(fc.argmax_codes(p_pooled), original["y_pred_pooled_code"]) \
                and np.array_equal(fc.argmax_codes(p_part), original["y_pred_partitioned_code"])
            identities = {n: b["identity"]["train_keys_sha256"] == record["train_keys_sha256"] for n, b in bundles.items()}
            same = same and all(identities.values())
            replay_rows.append({"horizon": horizon, "month": month, "bundles": len(bundles),
                                "rows": len(original), "max_abs_probability_diff": diff,
                                "labels_identical": True if same else None, "passed": bool(same)})
            check(results, f"saved-model replay stage3 h{horizon} {month}: {len(bundles)} loaded bundles reproduce "
                           f"{len(original)} rows (identical labels, probabilities within {PROBA_TOLERANCE:g}; max diff {diff:.1e}), "
                           "hashes and fit identities", same)
    (out / "stage3_saved_model_replay.json").write_text(json.dumps(replay_rows, indent=2), encoding="utf-8")
    # Stage 1: reload retained checkpoints and reproduce the saved held-out predictions.
    from src.model.model_RF import RFmodel
    from src.helper.helper import get_X_branch_id_by_group
    retained = sorted((run / "stage1" / "retained").glob("fs*"))
    features = json.loads(SCHEMA.read_text(encoding="utf-8"))["ordered_features"]
    for folder in retained:
        name = folder.name
        scope, term = int(name[2]), name.split("_")[1]
        horizon = {1: 4, 2: 8, 3: 12}[scope]
        snap = pd.read_parquet(run / "prepared" / f"snapshot_h{horizon}.parquet")
        test = snap[snap.target_month == mi(term)].sort_values("area")
        s_branch = pd.read_pickle(folder / "space_partitions" / "s_branch.pkl")
        routed = get_X_branch_id_by_group(test["area"].to_numpy(), s_branch)
        model = RFmodel(str(folder / "checkpoints"), 100, num_class=4, n_jobs=1)
        pred = np.zeros(len(test), dtype=int)
        X = test[features].to_numpy(dtype=float)
        for branch in np.unique(routed):
            rows = routed == branch
            model.load(branch)
            pred[rows] = model.predict(X[rows])
        saved = pd.read_csv(run / "stage1" / "folds" / name / "target_predictions.csv").sort_values("FEWSNET_admin_code")
        check(results, f"replay stage1 {name}: retained checkpoints reproduce {len(pred)} partitioned predictions",
              np.array_equal(pred, saved["y_pred_partitioned_code"].to_numpy()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = run / "verification"
    if out.exists():
        raise FileExistsError(f"{out} exists")
    out.mkdir()
    results = []
    recorded = json.loads((run / "prepared" / "manifests" / "outputs.json").read_text(encoding="utf-8"))
    drift = [p for p, h in recorded.items() if sha256(run / "prepared" / p) != h]
    check(results, f"prepared outputs match {len(recorded)} recorded hashes", not drift, drift)
    verify_features(run, results)
    verify_stage1(run, results)
    verify_stage2(run, results)
    verify_stage3(run, results)
    verify_report(run, results)
    verify_replay(run, results, out)
    (out / "verification.json").write_text(json.dumps({
        "passed": all(r["passed"] for r in results), "checks": results}, indent=2, default=str), encoding="utf-8")
    failed = [r for r in results if not r["passed"]]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
