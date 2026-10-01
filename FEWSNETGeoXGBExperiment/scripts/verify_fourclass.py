"""Independent reconstruction and replay for a completed GeoXGBoost run.

python scripts/verify_fourclass.py --run-dir RUN

Never modifies a run artifact; writes ``RUN/verification/`` (fresh). Checks:

1. Identity: producer code and verifier equal the committed blobs at the run's git_head
   and HEAD; prepared outputs still match their recorded hashes.
2. Features: an independent row-wise re-derivation of a keyed sample from the raw panel.
3. Stage 1: the 162 roots / 648 candidates are accepted; fitting rows lie in [O-59, O)
   and <= 2020-12, fitting/validation keys are disjoint; held-out E3 scores recompute
   with sklearn; every recorded E2 decision equals its own exact threshold rule; for a
   sample of candidates every saved child checkpoint carries the root's exact structural
   prefix and the terminal checkpoints reproduce the saved held-out probabilities.
4. G screening and development: G choice and the 24-scheme ranking recompute with
   sklearn from saved keyed predictions; every map's weights recompute.
5. Gates: every development and final gate decision recomputes from its saved pairs,
   and every prediction row's route equals its region decision.
6. Final: keys == truth keys for every arm; saved global/local boosters of the first and
   last fold per horizon and arm replay the saved probabilities exactly (no refit); shared
   locals carry the global's exact prefix; no fit window reaches the target.
7. Report: arm metrics, bootstrap draws (sample) and the D3 decision recompute.
"""
import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
SCHEMA = PACKAGE / "feature-schema.json"
HORIZONS = (4, 8, 12)


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mi(label: str) -> int:
    y, m = str(label).split("-")
    return int(y) * 12 + int(m) - 1


def macro(truth, pred):
    return float(f1_score(np.asarray(truth, int), np.asarray(pred, int), labels=[0, 1, 2, 3], average="macro",
                          zero_division=0))


def exact_macro(truth, pred):
    truth, pred = np.asarray(truth, int), np.asarray(pred, int)
    total = Fraction(0)
    for k in range(4):
        tp = int(np.sum((truth == k) & (pred == k)))
        fp = int(np.sum((truth != k) & (pred == k)))
        fn = int(np.sum((truth == k) & (pred != k)))
        if 2 * tp + fp + fn:
            total += Fraction(2 * tp, 2 * tp + fp + fn)
    return total / 4


def check(results, name, ok, detail=None):
    results.append({"check": name, "passed": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (f" :: {str(detail)[:400]}" if detail and not ok else ""), flush=True)


def read(path):
    return pd.read_csv(path, float_precision="round_trip", low_memory=False)


def main_cohort(frame, h):
    keep = frame["persistence_code"].notna()
    if h in (4, 8):
        keep &= frame["expert_code"].notna()
    return frame[keep]


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
    from src.utils import acceptance as acc
    from src.model import native_xgb as nx
    cands = acc.accept_stage1(run)
    check(results, f"stage1: {len(cands)} scheduled candidates accepted "
                   f"({sum(c['status'] == 'completed' for c in cands.values())} completed)", len(cands) == 648)
    stage1 = run / "stage1"
    bad_window, rescored, e2, keys = [], [], [], []
    roots = sorted({c["root"] for c in cands.values()})
    for root in roots:
        r = json.loads((stage1 / "roots" / root / "root.json").read_text(encoding="utf-8"))
        if r["status"] != "completed":
            continue
        origin = mi(r["origin_month"])
        months = [mi(x) for x in r["train_label_months_observed"]]
        if min(months) < origin - 59 or max(months) >= origin or max(months) > mi("2020-12"):
            bad_window.append(root)
        members = read(stage1 / "roots" / root / "fold_membership.csv.gz")
        inner = members[members.role != "heldout_target"]
        if set(inner["target_month"].map(mi)) - set(range(origin - 59, origin)):
            bad_window.append(f"{root}:membership")
        fit = set(zip(inner.loc[inner.role == "fitting", "area"], inner.loc[inner.role == "fitting", "target_month"]))
        val = set(zip(inner.loc[inner.role == "validation", "area"], inner.loc[inner.role == "validation", "target_month"]))
        if fit & val or len(fit) != r["rows"]["fitting"] or len(val) != r["rows"]["validation"]:
            keys.append(root)
        pooled = read(stage1 / "roots" / root / "root_target_predictions.csv")
        for cand in r["candidates"]:
            c = json.loads((stage1 / "candidates" / cand / "candidate.json").read_text(encoding="utf-8"))
            preds = read(stage1 / "candidates" / cand / "target_predictions.csv")
            if not (np.isclose(macro(preds.y_true_code, preds.y_pred_partitioned_code), c["scores"]["macro_f1"], rtol=0, atol=1e-12)
                    and np.isclose(macro(pooled.y_true_code, pooled.y_pred_pooled_code), c["scores"]["macro_f1_base"], rtol=0, atol=1e-12)
                    and (preds.y_pred_pooled_code.to_numpy() == pooled.y_pred_pooled_code.to_numpy()).all()):
                rescored.append(cand)
            threshold = Fraction(c["threshold"])
            for d in c["partition"]["decisions"]:
                if d["outcome"] in ("accepted", "rejected_gate"):
                    accept = Fraction(d["gain"]) > threshold
                    if accept != (d["outcome"] == "accepted"):
                        e2.append((cand, d["branch_id"]))
    check(results, "stage1: fitting/validation rows in [O-59,O) and <= 2020-12", not bad_window, bad_window)
    check(results, "stage1: fitting and validation keys disjoint and complete", not keys, keys)
    check(results, "stage1: E3 partitioned/pooled macro F1 recompute; pooled E3 = the candidate's own root", not rescored, rescored)
    check(results, "stage1: every E2 decision equals 'exact gain > its family threshold'", not e2, e2[:5])
    replay_stage1(run, results, cands)


def replay_stage1(run, results, cands, per_h=2):
    """Structural prefix of every saved checkpoint and terminal replay, sampled candidates."""
    from src.model import native_xgb as nx
    from src.feature.fourclass_features import load_schema
    features = load_schema(SCHEMA)["ordered_features"]
    stage1 = run / "stage1"
    for h in HORIZONS:
        names = sorted(n for n, c in cands.items() if c["horizon"] == h and c["status"] == "completed")
        sample = [names[0], names[-1]][:per_h]
        snap = pd.read_parquet(run / "prepared" / f"snapshot_h{h}.parquet")
        for name in sample:
            c = json.loads((stage1 / "candidates" / name / "candidate.json").read_text(encoding="utf-8"))
            ckpt = stage1 / "checkpoints" / name
            changed = [f for f, s in c["checkpoints"]["sha256"].items() if sha256(ckpt / f) != s]
            root = nx.from_raw((ckpt / "xgb_root.ubj").read_bytes())
            root_sha = nx.prefix_identity(root)["sha256"]
            n0 = root.num_boosted_rounds()
            prefix_bad = [f for f in sorted(ckpt.glob("*.ubj"))
                          if nx.prefix_identity(nx.from_raw(f.read_bytes()), n0)["sha256"] != root_sha]
            preds = read(stage1 / "candidates" / name / "target_predictions.csv")
            test = snap[snap.target_month == mi(c["candidate"].split("_")[1])].set_index("area").loc[preds.FEWSNET_admin_code]
            got = np.zeros((len(preds), 4))
            for branch in preds.branch_id.unique():
                rows = (preds.branch_id == branch).to_numpy()
                booster = nx.from_raw((ckpt / f"xgb_{'root' if branch == 'root' else branch}.ubj").read_bytes())
                got[rows] = nx.proba(booster, test[features].to_numpy(float)[rows])
            want = preds.filter(like="p_partitioned_").to_numpy()
            check(results, f"replay stage1 {name}: {len(c['checkpoints']['sha256'])} checkpoint files unchanged, "
                           f"every booster carries the root prefix, held-out probabilities exact",
                  not changed and not prefix_bad and np.array_equal(got, want),
                  {"changed": changed, "prefix_bad": [p.name for p in prefix_bad],
                   "max_abs": float(np.abs(got - want).max())})


def verify_gscreen(run, results):
    from src.experiment import plan
    sel = json.loads((run / "gscreen" / "selection.json").read_text(encoding="utf-8"))
    preds = read(run / "gscreen" / "predictions.csv.gz")
    base = read(run / "prepared" / "ledgers" / "dev_baselines.csv")
    base["target_month"] = base["target_label"]
    bad = []
    for h in HORIZONS:
        scores = {}
        for g in plan.G_CONFIGS:
            p = preds[(preds.horizon == h) & (preds.g_config == g)]
            k = main_cohort(base[base.horizon == h].merge(p, on=["area", "target_month", "horizon"], validate="one_to_one"), h)
            if len(p) != int((base.horizon == h).sum()):
                bad.append((h, g, "keys"))
            scores[g] = (exact_macro(k.truth_code, k.y_pred_code), tuple(-x for x in plan.g_tiebreak_key(g)))
        if max(scores, key=scores.get) != sel["selected"][str(h)]:
            bad.append((h, "selection"))
    check(results, "gscreen: every G scored on all development truth keys; selection recomputes", not bad, bad)


def verify_maps(run, results):
    bad = []
    count = 0
    for d in sorted((run / "maps").iterdir()):
        record = json.loads((d / "consensus.json").read_text(encoding="utf-8"))
        count += 1
        if record["route"] == "no_prior_candidates":
            continue
        w = read(d / "plan_weights.csv")
        f = np.clip(w["macro_f1"].to_numpy(), 1e-6, 1 - 1e-6)
        b = np.clip(w["macro_f1_base"].to_numpy(), 1e-6, 1 - 1e-6)
        expect = np.maximum(np.log(f / (1 - f)) - np.log(b / (1 - b)), 0)
        if not np.allclose(expect, w["weight"], rtol=0, atol=1e-12) or \
                (record["route"] == "null_consensus") != (expect.max() <= 0):
            bad.append(d.name)
        if record["route"] == "learned_map" and sha256(d / record["cluster_map"]) != record["cluster_map_sha256"]:
            bad.append(f"{d.name}:map")
    check(results, f"maps: D9 weights and routes recompute for {count} consensus builds", not bad, bad)


def verify_gate_dir(fold_dir, problems):
    gate = json.loads((fold_dir / "gate.json").read_text(encoding="utf-8"))
    preds = read(fold_dir / "predictions.csv.gz")
    if not gate["regions"]:
        if preds["route"].str.contains("local").any():
            problems.append(f"{fold_dir}: local rows without regions")
        return 0
    pairs = read(fold_dir / "gate_pairs.csv.gz")
    for d in gate["regions"]:
        sub = pairs[pairs.cluster_id == d["cluster_id"]]
        rows = len(sub)
        dates = sub.groupby("validation_month")["local_fit_ok"].any() if rows else pd.Series(dtype=bool)
        support = (rows >= 100 and sub.area.nunique() >= 20 and len(dates) >= 3 and int(dates.sum()) >= 3)
        gain = exact_macro(sub.y_true, sub.y_local_routed) - exact_macro(sub.y_true, sub.y_global) if rows else None
        enabled = bool(support and gain > Fraction(1, 100))
        if enabled != d["enabled"] or (rows and Fraction(d["gain"]) != gain):
            problems.append(f"{fold_dir.name} c{d['cluster_id']}: gate decision does not recompute")
        bad_fit = sub[~sub.local_fit_ok & (sub.y_local_routed != sub.y_global)]
        if len(bad_fit):
            problems.append(f"{fold_dir.name} c{d['cluster_id']}: failed-fit rows not on the global prediction")
        cur = d["current_fit_support"]
        fit_ok = cur["rows"] >= 500 and cur["areas"] >= 50 and cur["dates"] >= 6 and cur["classes"] >= 2
        want = "local_model" if (enabled and fit_ok) else "global_fallback"
        routes = set(preds.loc[preds.cluster_id == d["cluster_id"], "route"].str.split(":").str[0])
        if d["route"] != want or routes != {want}:
            problems.append(f"{fold_dir.name} c{d['cluster_id']}: route {d['route']} / {routes} != {want}")
    return len(gate["regions"])


def verify_development(run, results):
    from src.experiment import plan
    problems, regions, folds = [], 0, 0
    for fold_dir in sorted((run / "development").glob("*/h*/*/*")):
        if (fold_dir / "fold.json").is_file():
            folds += 1
            regions += verify_gate_dir(fold_dir, problems)
    check(results, f"development: {regions} region gate decisions over {folds} arm-folds recompute from saved pairs; "
                   "rows follow their decision", not problems, problems[:5])
    table = read(run / "development" / "selection_table.csv")
    sel = json.loads((run / "development" / "selection.json").read_text(encoding="utf-8"))
    index = json.loads((run / "development" / "scheme_fold_index.json").read_text(encoding="utf-8"))
    base = read(run / "prepared" / "ledgers" / "dev_baselines.csv")
    base["target_month"] = base["target_label"]
    keys, bad = {}, []
    for scheme in plan.schemes():
        deltas, expert = {}, {}
        for h in HORIZONS:
            frames = [read(run / "development" / "folds" / f"h{h}" / t / index[f"h{h}_{t}"][scheme["scheme"]] /
                           "predictions.csv.gz") for t in plan.DEV_TARGETS]
            p = pd.concat(frames)
            k = base[base.horizon == h].merge(p, on=["area", "target_month", "horizon"], validate="one_to_one")
            if len(k) != int((base.horizon == h).sum()):
                bad.append((scheme["scheme"], h, "keys"))
            k = main_cohort(k, h)
            deltas[h] = exact_macro(k.truth_code, k.y_pred_code) - exact_macro(k.truth_code, k.persistence_code)
            if h in (4, 8):
                expert[h] = exact_macro(k.truth_code, k.y_pred_code) - exact_macro(k.truth_code, k.expert_code)
        keys[scheme["scheme"]] = (min(deltas.values()), sum(deltas.values()) / 3, (expert[4] + expert[8]) / 2,
                                  sum(v == "L1" for v in scheme["l_vector"].values()),
                                  -plan.STRATEGIES.index(scheme["strategy"]), -scheme["l_vector_id"])
        row = table[table.scheme == scheme["scheme"]].iloc[0]
        if Fraction(row["delta_persistence_exact_h4"]) != deltas[4]:
            bad.append((scheme["scheme"], "table"))
    winner = max(keys, key=keys.get)
    check(results, "development: 24-scheme deltas recompute on all truth keys and the lexicographic winner matches",
          not bad and winner == sel["selected_scheme"], {"bad": bad[:5], "winner": winner})


def verify_final(run, results):
    from src.model import native_xgb as nx
    from src.feature.fourclass_features import load_schema
    from src.utils import acceptance as acc
    features = load_schema(SCHEMA)["ordered_features"]
    frozen = json.loads((run / "frozen" / "frozen.json").read_text(encoding="utf-8"))
    sched = acc.schedule(run)
    problems, regions = [], 0
    arms = ("pooled", "rfmap_independent", "rfmap_shared", "xgbmap_shared")
    base = read(run / "prepared" / "ledgers" / "baselines.csv")
    base["target_month"] = base["target_label"]
    for h in HORIZONS:
        folds = [f for f in sched["stage3"] if f["horizon"] == h and f["status"] == "scheduled"]
        snap = pd.read_parquet(run / "prepared" / f"snapshot_h{h}.parquet")
        for arm in arms:
            frames = []
            for i, f in enumerate(folds):
                d = run / "final" / f"h{h}" / f["target_month"] / arm
                regions += verify_gate_dir(d, problems)
                p = read(d / "predictions.csv.gz")
                frames.append(p)
                gate = json.loads((d / "gate.json").read_text(encoding="utf-8"))
                g_path = run / "globals" / f"h{h}" / frozen["g_selection"][str(h)] / f"O{f['origin_month']}"
                g_rec = json.loads(g_path.with_suffix(".json").read_text(encoding="utf-8"))
                if mi(g_rec["fit_label_months"][1]) >= mi(f["origin_month"]):
                    problems.append(f"{d}: global window reaches the origin")
                if g_rec["booster_sha256"] != gate["global"]["booster_sha256"]:
                    problems.append(f"{d}: fold global differs from the stored global")
                if i not in (0, len(folds) - 1):
                    continue
                g = nx.from_raw(g_path.with_suffix(".ubj").read_bytes())
                g_struct = nx.prefix_identity(g)["sha256"]
                test = snap[snap.target_month == mi(f["target_month"])].set_index("area").loc[p.area]
                X = test[features].to_numpy(float)
                got = nx.proba(g, X)
                for c, record in gate["locals"].items():
                    rows = (p.cluster_id == int(c)).to_numpy()
                    booster = nx.from_raw((d / "models" / f"local_{c}.ubj").read_bytes())
                    if nx.sha(booster) != record["booster_sha256"]:
                        problems.append(f"{d}: local_{c} bytes differ from the record")
                    if arm != "rfmap_independent" and \
                            nx.prefix_identity(booster, g.num_boosted_rounds())["sha256"] != g_struct:
                        problems.append(f"{d}: shared local_{c} lacks the global prefix")
                    got[rows] = nx.proba(booster, X[rows])
                if not np.array_equal(got, p.filter(like="p_").to_numpy()):
                    problems.append(f"{d}: saved boosters do not replay the saved probabilities")
            pred = pd.concat(frames)
            want = base[base.horizon == h]
            if set(zip(pred.area, pred.target_month)) != set(zip(want.area, want.target_month)) or len(pred) != len(want):
                problems.append(f"h{h} {arm}: prediction keys differ from the truth keys")
    check(results, f"final: {regions} gates recompute; keys == truth keys for every arm; first/last fold per H and arm "
                   "replay exactly from saved boosters; shared locals carry the global prefix", not problems, problems[:8])


def verify_report(run, results):
    report = json.loads((run / "report" / "report.json").read_text(encoding="utf-8"))
    keyed = read(run / "report" / "keyed_evaluation.csv.gz")
    col = {"main": "y_pred_xgbmap_shared", "pooled": "y_pred_pooled", "rfmap_shared": "y_pred_rfmap_shared",
           "rfmap_independent": "y_pred_rfmap_independent", "expert": "expert_code", "persistence": "persistence_code",
           "v7_partitioned_rf": "y_pred_v7_partitioned_rf", "v7_pooled_rf": "y_pred_v7_pooled_rf"}

    def cohort_rows(cohort):
        h = int(cohort.split("_h")[1])
        rows = keyed[(keyed.horizon == h) & keyed.persistence_code.notna()]
        return rows[rows.expert_code.notna()] if cohort.startswith("main") and h != 12 else rows
    bad = []
    for cohort, entry in report["metrics"].items():
        rows = cohort_rows(cohort)
        if len(rows) != entry["n"]:
            bad.append((cohort, "n"))
        for arm, summary in entry["arms"].items():
            if not np.isclose(macro(rows.truth_code, rows[col[arm]].astype(int)), summary["macro_f1"], rtol=0, atol=1e-12):
                bad.append((cohort, arm))
    check(results, "report: every cohort size and arm macro F1 recomputes with sklearn", not bad, bad)
    draws = read(run / "report" / "bootstrap_draws.csv.gz")
    countries = report["bootstrap"]["countries"]
    bad = []
    for _, draw in draws.sample(n=min(20, len(draws)), random_state=3).iterrows():
        mult = draw[countries].astype(int)
        for cohort, entry in report["metrics"].items():
            rows = cohort_rows(cohort)
            rep = rows.iloc[np.repeat(np.arange(len(rows)), rows["country"].map(mult).to_numpy())]
            for arm in entry["arms"]:
                if not np.isclose(macro(rep.truth_code, rep[col[arm]].astype(int)), draw[f"{cohort}:{arm}"], rtol=0, atol=1e-9):
                    bad.append((int(draw["draw"]), cohort, arm))
    check(results, "report: 20 bootstrap draws recompute by row replication from saved multiplicities", not bad, bad[:5])
    check(results, f"report: {report['bootstrap']['draws_accepted']} of 2000 draws accepted", report["bootstrap"]["complete"])
    bad = []
    for h, entry in report["scientific_target_D3"]["per_horizon"].items():
        name = f"main_h{h}:main"
        deltas = draws[name] - draws[f"main_h{h}:persistence"]
        lo = float(np.percentile(deltas, 2.5, method="linear"))
        point = report["metrics"][f"main_h{h}"]["arms"]["main"]["macro_f1"] - \
            report["metrics"][f"main_h{h}"]["arms"]["persistence"]["macro_f1"]
        status = "pass" if point > 0 and lo > 0 else "fail"
        if not np.isclose(lo, entry["ci95"][0], rtol=0, atol=1e-12) or status != entry["status"]:
            bad.append(h)
    check(results, "report: D3 per-H decisions recompute from saved draws", not bad, bad)


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
    from src.utils.acceptance import identity_problems
    from src.utils.run_identity import git_head
    identity = json.loads((run / "prepared" / "manifests" / "identity.json").read_text(encoding="utf-8"))
    problems = identity_problems(run)
    check(results, "producer code and verifier == committed blobs at the run's git_head and HEAD == working tree",
          not problems, {"problems": problems, "run_git_head": identity.get("git_head"), "current_head": git_head()})
    recorded = json.loads((run / "prepared" / "manifests" / "outputs.json").read_text(encoding="utf-8"))
    drift = [p for p, h in recorded.items() if sha256(run / "prepared" / p) != h]
    check(results, f"prepared outputs match {len(recorded)} recorded hashes", not drift, drift)
    for step in (verify_features, verify_stage1, verify_gscreen, verify_maps, verify_development, verify_final,
                 verify_report):
        try:
            step(run, results)
        except Exception as exc:  # a crashed check is a failed check, never a skipped one
            check(results, f"{step.__name__} completed", False, repr(exc))
    (out / "verification.json").write_text(json.dumps({
        "passed": all(r["passed"] for r in results), "checks": results}, indent=2, default=str), encoding="utf-8")
    failed = [r for r in results if not r["passed"]]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
