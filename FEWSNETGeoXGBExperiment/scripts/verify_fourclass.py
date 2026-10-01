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




# ---------------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------------

def stage1_training_rows(snap, target, horizon):
    """Independent rebuild of a root's training rows: labels in [O-59, O) of the areas
    present at the target month, in (area, target_month) order."""
    snap = snap.sort_values(["area", "target_month"]).reset_index(drop=True)
    origin = target - horizon
    rows = snap[(snap.target_month >= origin - 59) & (snap.target_month < origin)]
    return rows[rows.area.isin(set(snap.loc[snap.target_month == target, "area"]))]


def keys_digest(areas, months):
    keys = np.column_stack([np.asarray(areas, dtype=np.int64), np.asarray(months, dtype=np.int64)])
    return hashlib.sha256(np.ascontiguousarray(keys).tobytes()).hexdigest()


def e2_problems(c, e2, val_branch, val_keys, family_threshold):
    """Recompute every fitted E2 decision from its keyed rows on the complete parent keys."""
    problems = []
    decisions = c["partition"]["decisions"]
    for i, d in enumerate(decisions):
        if d["outcome"] not in ("accepted", "rejected_gate"):
            continue
        rows = e2[e2.decision == i]
        b = d["branch_id"]
        expected = {k for k, br in zip(val_keys, val_branch) if br.startswith(b)}
        got = set(zip(rows.area, rows.target_month))
        if got != expected or len(rows) != len(expected):
            problems.append(f"decision {i} ({b!r}): E2 keys are not the complete parent validation keys")
            continue
        s0, s1 = rows[rows.side == 0], rows[rows.side == 1]
        truth = np.r_[s0.y_true, s1.y_true]
        base = exact_macro(truth, np.r_[s0.y_parent, s1.y_parent])
        best, choice = base, (False, False)
        scores = {"parent_parent": base}
        for u0, u1 in ((True, False), (False, True), (True, True)):
            if (u0 and not s0.child_eligible.all()) or (u1 and not s1.child_eligible.all()):
                continue
            score = exact_macro(truth, np.r_[s0.y_child if u0 else s0.y_parent, s1.y_child if u1 else s1.y_parent])
            scores[f"{'child' if u0 else 'parent'}_{'child' if u1 else 'parent'}"] = score
            if score > best:
                best, choice = score, (u0, u1)
        accepted = best - base > family_threshold
        if ({k: Fraction(v) for k, v in d["scores"].items()} != scores or Fraction(d["gain"]) != best - base
                or accepted != (d["outcome"] == "accepted")
                or (accepted and d["selected_children"] != [bool(x) for x in choice])):
            problems.append(f"decision {i} ({b!r}): routes/gain/outcome do not recompute")
    return problems


def verify_stage1(run, results):
    from src.experiment import plan
    from src.utils import acceptance as acc
    from src.utils.split import group_aware_train_val_split
    cands = acc.accept_stage1(run)
    check(results, f"stage1: {len(cands)} scheduled candidates accepted "
                   f"({sum(c['status'] == 'completed' for c in cands.values())} completed)", len(cands) == 648)
    stage1 = run / "stage1"
    sched = acc.schedule(run)
    by_root = {}
    for c in cands.values():
        by_root.setdefault(c["root"], c)
    identity, split_bad, rescored, e2_bad, windows = [], [], [], [], []
    snaps = {h: pd.read_parquet(run / "prepared" / f"snapshot_h{h}.parquet") for h in HORIZONS}
    for root, entry in sorted(by_root.items()):
        r = json.loads((stage1 / "roots" / root / "root.json").read_text(encoding="utf-8"))
        h, target = entry["horizon"], mi(entry["target_month"])
        if (r["horizon"], r["target_month"], r["origin_month"], r["ratio"], r["split_seed"]) != \
                (h, entry["target_month"], entry["origin_month"], entry["ratio"], entry["split_seed"]) \
                or mi(r["origin_month"]) != target - h or r["val_ratio"] != plan.SPLIT_RATIOS[entry["ratio"]]:
            identity.append(root)
        if r["status"] != "completed":
            continue
        train = stage1_training_rows(snaps[h], target, h)
        if train.target_month.max() > mi("2020-12") or train.target_month.min() < target - h - 59:
            windows.append(root)
        x_set = group_aware_train_val_split(train.area.to_numpy(), plan.SPLIT_RATIOS[entry["ratio"]], 1,
                                            entry["split_seed"], True)["X_set"]
        fit, val = train[x_set == 0], train[x_set == 1]
        members = read(stage1 / "roots" / root / "fold_membership.csv.gz")
        inner = members[members.role != "heldout_target"]
        if (keys_digest(fit.area, fit.target_month) != r["fitting_keys_sha256"]
                or keys_digest(val.area, val.target_month) != r["validation_keys_sha256"]
                or list(inner.role) != list(np.where(x_set == 1, "validation", "fitting"))
                or list(inner.area) != list(train.area)):
            split_bad.append(root)
        pooled = read(stage1 / "roots" / root / "root_target_predictions.csv")
        truth_t = snaps[h].loc[snaps[h].target_month == target].set_index("area").loc[pooled.FEWSNET_admin_code, "class_code"]
        if not (pooled.y_true_code.to_numpy() == truth_t.to_numpy()).all():
            rescored.append(f"{root}: target truth")
        val_keys = list(zip(val.area, [f"{m // 12:04d}-{m % 12 + 1:02d}" for m in val.target_month]))
        for cand in r["candidates"]:
            c = json.loads((stage1 / "candidates" / cand / "candidate.json").read_text(encoding="utf-8"))
            preds = read(stage1 / "candidates" / cand / "target_predictions.csv")
            if not (np.isclose(macro(preds.y_true_code, preds.y_pred_partitioned_code), c["scores"]["macro_f1"], rtol=0, atol=1e-12)
                    and np.isclose(macro(pooled.y_true_code, pooled.y_pred_pooled_code), c["scores"]["macro_f1_base"], rtol=0, atol=1e-12)
                    and (preds.y_pred_pooled_code.to_numpy() == pooled.y_pred_pooled_code.to_numpy()).all()
                    and (preds.y_pred_partitioned_code.to_numpy() == preds.filter(like="p_partitioned_").to_numpy().argmax(axis=1)).all()):
                rescored.append(cand)
            family = cand.split("_")[-1]
            threshold = plan.THRESHOLD_FAMILIES[family]
            if Fraction(c["threshold"]) != threshold or c["threshold_family"] != family:
                e2_bad.append(f"{cand}: threshold is not its family's")
            branch = np.load(stage1 / "candidates" / cand / "X_branch_id.npy", allow_pickle=False)
            e2 = read(stage1 / "candidates" / cand / "e2_predictions.csv.gz")
            e2["branch_id"] = e2["branch_id"].fillna("").astype(str)
            e2_bad += [f"{cand}: {p}" for p in e2_problems(c, e2, list(branch[x_set == 1]), val_keys, threshold)]
    check(results, "stage1: every root's H/T/O/ratio/seed equals its scheduled identity", not identity, identity[:5])
    check(results, "stage1: training rows lie in [O-59,O) and <= 2020-12", not windows, windows[:5])
    check(results, "stage1: the within-area random split recomputes; fitting/validation key digests and "
                   "membership roles match", not split_bad, split_bad[:5])
    check(results, "stage1: E3 scores, truth and argmax recompute; pooled E3 = the candidate's own root",
          not rescored, rescored[:5])
    check(results, "stage1: every fitted E2 decision recomputes from keyed parent/child predictions on the "
                   "complete parent validation keys with its family threshold", not e2_bad, e2_bad[:5])
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
            for branch in preds.branch_id.astype(str).unique():
                rows = (preds.branch_id.astype(str) == branch).to_numpy()
                booster = nx.from_raw((ckpt / f"xgb_{branch}.ubj").read_bytes())
                got[rows] = nx.proba(booster, test[features].to_numpy(float)[rows])
            want = preds.filter(like="p_partitioned_").to_numpy()
            check(results, f"replay stage1 {name}: {len(c['checkpoints']['sha256'])} checkpoint files unchanged, "
                           f"every booster carries the root prefix, held-out probabilities exact",
                  not changed and not prefix_bad and np.array_equal(got, want),
                  {"changed": changed, "prefix_bad": [p.name for p in prefix_bad],
                   "max_abs": float(np.abs(got - want).max())})


# ---------------------------------------------------------------------------------
# G screening, maps
# ---------------------------------------------------------------------------------

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
            k = base[base.horizon == h].merge(p, on=["area", "target_month", "horizon"], validate="one_to_one")
            if len(p) != int((base.horizon == h).sum()) or len(k) != len(p) or not (k.truth_code == k.y_true_code).all():
                bad.append((h, g, "keys/truth"))
            if not (p.y_pred_code.to_numpy() == p.filter(like="p_").to_numpy().argmax(axis=1)).all():
                bad.append((h, g, "argmax"))
            k = main_cohort(k, h)
            scores[g] = (exact_macro(k.truth_code, k.y_pred_code), tuple(-x for x in plan.g_tiebreak_key(g)))
        if max(scores, key=scores.get) != sel["selected"][str(h)]:
            bad.append((h, "selection"))
    check(results, "gscreen: every G scored on all development truth keys; selection recomputes", not bad, bad)


def verify_maps(run, results):
    """Every map a fold used is the map of its EXPECTED pool (accept_consensus with the pool
    rebuilt from accepted candidates by the approved rule), incl. no-prior and frozen maps."""
    from scripts import run_experiment as rexp
    from scripts.run_stage2 import pool_identity
    from src.experiment import plan
    from src.utils import acceptance as acc
    g_of, _ = acc.accept_g_selection(run)
    xgb, paths = acc.candidate_frame(acc.accept_stage1(run))
    v7, _ = acc.candidate_frame(acc.v7_candidates())
    bad, used = [], set()
    index = json.loads((run / "development" / "scheme_fold_index.json").read_text(encoding="utf-8"))
    for fold in acc.schedule(run)["development"]:
        h, t = fold["horizon"], fold["target_month"]
        origin = mi(fold["origin_month"])
        try:
            specs, identities, dirs, scheme_maps = rexp.dev_specs(run, fold, xgb, paths, g_of)
        except Exception as exc:
            bad.append(f"h{h} {t}: {exc}")
            continue
        if scheme_maps != index[f"h{h}_{t}"]:
            bad.append(f"h{h} {t}: scheme index differs from the expected maps")
        for scheme in plan.schemes():
            sub = rexp.scheme_pool(xgb, scheme, origin)
            if (sub.target_month.map(mi) >= origin).any():
                bad.append(f"h{h} {t} {scheme['scheme']}: candidate scored at/after O")
            used.add(pool_identity(sub))
        for label, ident in identities.items():
            record = json.loads((dirs[label] / "fold.json").read_text(encoding="utf-8"))
            if record.get("map_id") != ident.get("map_id") or record.get("local_config") != ident.get("local_config"):
                bad.append(f"h{h} {t} {label}: fold map/local differs from its expected pool")
        ident, record, _ = rexp.map_for(run, rexp.v7_pool(v7, origin))
        used.add(ident)
        for arm in ("independent", "shared"):
            rec = json.loads((run / "development" / "oldmap" / f"h{h}" / t / f"rfmap_{arm}" / "fold.json").read_text(encoding="utf-8"))
            if rec.get("map_id") != ident:
                bad.append(f"h{h} {t} rfmap_{arm}: old map differs from the truncated v7 pool")
    frozen = json.loads((run / "frozen" / "frozen.json").read_text(encoding="utf-8"))
    scheme = next(s for s in plan.schemes() if s["scheme"] == frozen["selected_scheme"])
    sub = rexp.scheme_pool(xgb, scheme, mi("2020-12") + 1)
    try:
        ident, record, _ = rexp.map_for(run, sub)
        if ident != frozen["final_map"]["map_id"] or len(sub) != record["candidates"]:
            bad.append("frozen map is not the selected scheme's full 2018-2020 pool")
    except Exception as exc:
        bad.append(f"frozen map: {exc}")
    used.add(frozen["final_map"]["map_id"])
    stray = sorted(p.name for p in (run / "maps").iterdir() if p.name not in used)
    check(results, f"maps: {len(used)} expected pools (dev schemes x origins, truncated v7, frozen) accepted "
                   "against their rebuilt candidate pools; no candidate scored at/after O; no stray map",
          not bad and not stray, {"bad": bad[:5], "stray": stray[:5]})
    w_bad = []
    for d in sorted((run / "maps").iterdir()):
        record = json.loads((d / "consensus.json").read_text(encoding="utf-8"))
        if record["route"] == "no_prior_candidates":
            continue
        w = read(d / "plan_weights.csv")
        f = np.clip(w["macro_f1"].to_numpy(), 1e-6, 1 - 1e-6)
        b = np.clip(w["macro_f1_base"].to_numpy(), 1e-6, 1 - 1e-6)
        expect = np.maximum(np.log(f / (1 - f)) - np.log(b / (1 - b)), 0)
        if not np.allclose(expect, w["weight"], rtol=0, atol=1e-12) or \
                (record["route"] == "null_consensus") != (expect.max() <= 0):
            w_bad.append(d.name)
    check(results, "maps: E4 weights and routes recompute independently", not w_bad, w_bad[:5])


# ---------------------------------------------------------------------------------
# gates and folds
# ---------------------------------------------------------------------------------

class Expected:
    """Independent window/key/support arithmetic over one horizon's snapshot."""

    def __init__(self, run, h):
        snap = pd.read_parquet(run / "prepared" / f"snapshot_h{h}.parquet").sort_values(["area", "target_month"])
        self.h, self.area, self.month, self.y = h, snap.area.to_numpy(), snap.target_month.to_numpy(), snap.class_code.to_numpy()
        self.snap = snap.reset_index(drop=True)

    def window(self, origin):
        return np.where((self.month >= origin - 59) & (self.month < origin))[0]

    def support(self, rows):
        counts = [int(np.sum(self.y[rows] == k)) for k in range(4)]
        return {"rows": int(len(rows)), "areas": int(np.unique(self.area[rows]).size),
                "dates": int(np.unique(self.month[rows]).size), "classes": int(sum(c > 0 for c in counts))}


def fit_ok(sup):
    return sup["rows"] >= 500 and sup["areas"] >= 50 and sup["dates"] >= 6 and sup["classes"] >= 2


def verify_gate_dir(fold_dir, fold, cluster_of, ex, run, g, problems):
    """Expected gate population rebuilt from the schedule, snapshot and actual map."""
    gate = json.loads((fold_dir / "gate.json").read_text(encoding="utf-8"))
    preds = read(fold_dir / "predictions.csv.gz")
    target, origin = mi(fold["target_month"]), mi(fold["origin_month"])
    if origin != target - ex.h:
        problems.append(f"{fold_dir}: O != T - H")
    test = np.where(ex.month == target)[0]
    test_areas = ex.area[test]
    want_cluster = np.array([cluster_of.get(int(a), -1) for a in preds.area]) if cluster_of else np.full(len(preds), -1)
    if not (preds.cluster_id.to_numpy() == want_cluster).all():
        problems.append(f"{fold_dir}: cluster_id differs from the map")
    if not cluster_of:
        if gate["regions"] or preds.route.str.contains("local").any():
            problems.append(f"{fold_dir}: map-less arm has regions or local routes")
        return 0
    clusters = sorted({cluster_of[int(a)] for a in test_areas if int(a) in cluster_of})
    if sorted(d["cluster_id"] for d in gate["regions"]) != clusters:
        problems.append(f"{fold_dir}: regions differ from the clusters present at T")
    pairs = read(fold_dir / "gate_pairs.csv.gz") if (fold_dir / "gate_pairs.csv.gz").is_file() else pd.DataFrame()
    store = run / "globals" / f"h{ex.h}" / g
    members = {c: {a for a, k in cluster_of.items() if k == c} for c in clusters}
    for spec in fold["gate"]:
        u, v = mi(spec["validation_month"]), mi(spec["internal_origin"])
        if not (u < origin and v == u - ex.h):
            problems.append(f"{fold_dir}: gate date {spec} not U<O, V=U-H")
        rec = json.loads((store / f"O{spec['internal_origin']}.json").read_text(encoding="utf-8"))
        win = ex.window(v)
        if rec["fit_label_months"] != [f"{(v - 59) // 12:04d}-{(v - 59) % 12 + 1:02d}", f"{(v - 1) // 12:04d}-{(v - 1) % 12 + 1:02d}"] \
                or rec["fit_keys_sha256"] != keys_digest(ex.area[win], ex.month[win]):
            problems.append(f"{fold_dir}: internal global at {spec['internal_origin']} not fitted on its own window")
        date_pairs = pairs[pairs.validation_month == spec["validation_month"]] if len(pairs) else pairs
        if len(date_pairs) and set(date_pairs.global_sha256) != {rec["booster_sha256"]}:
            problems.append(f"{fold_dir}: {spec['validation_month']} pairs not scored by that internal global")
        at_u = np.where(ex.month == u)[0]
        for c in clusters:
            expected = {int(a) for a in ex.area[at_u] if int(a) in members[c]}
            sub = date_pairs[date_pairs.cluster_id == c] if len(date_pairs) else date_pairs
            if set(sub.area.astype(int)) != expected or len(sub) != len(expected) or \
                    (len(sub) and not (sub.internal_origin == spec["internal_origin"]).all()):
                problems.append(f"{fold_dir} c{c} {spec['validation_month']}: gate keys differ from the expected population")
                continue
            if len(sub):
                rows = win[np.isin(ex.area[win], list(members[c]))]
                ok = fit_ok(ex.support(rows))
                if set(sub.local_fit_ok) != {ok}:
                    problems.append(f"{fold_dir} c{c} {spec['validation_month']}: local_fit_ok != recomputed support")
                truth = ex.snap.iloc[at_u].set_index("area").loc[sub.area, "class_code"].to_numpy()
                if not (sub.y_true.to_numpy() == truth).all():
                    problems.append(f"{fold_dir} c{c}: gate truth differs from the snapshot")
    win_o = ex.window(origin)
    for d in gate["regions"]:
        c = d["cluster_id"]
        sub = pairs[pairs.cluster_id == c] if len(pairs) else pairs
        rows = len(sub)
        dates = sub.groupby("validation_month")["local_fit_ok"].any() if rows else pd.Series(dtype=bool)
        support = rows >= 100 and sub.area.nunique() >= 20 and len(dates) >= 3 and int(dates.sum()) >= 3
        gain = exact_macro(sub.y_true, sub.y_local_routed) - exact_macro(sub.y_true, sub.y_global) if rows else None
        enabled = bool(support and gain > Fraction(1, 100))
        if enabled != d["enabled"] or (rows and Fraction(d["gain"]) != gain):
            problems.append(f"{fold_dir.name} c{c}: gate decision does not recompute")
        if len(sub[~sub.local_fit_ok & (sub.y_local_routed != sub.y_global)]):
            problems.append(f"{fold_dir.name} c{c}: failed-fit rows not on the global prediction")
        cur = ex.support(win_o[np.isin(ex.area[win_o], list(members[c]))])
        want = "local_model" if (enabled and fit_ok(cur)) else "global_fallback"
        routes = set(preds.loc[preds.cluster_id == c, "route"].str.split(":").str[0])
        if d["route"] != want or routes != {want}:
            problems.append(f"{fold_dir.name} c{c}: route {d['route']} / {routes} != {want}")
    unmapped = preds.cluster_id == -1
    if not (preds.loc[unmapped, "route"] == "unmapped_area_global").all():
        problems.append(f"{fold_dir}: unmapped rows not on the global")
    return len(gate["regions"])


def fold_label_problems(fold_dir, fold, ex, problems):
    preds = read(fold_dir / "predictions.csv.gz")
    target = mi(fold["target_month"])
    test = ex.snap[ex.snap.target_month == target]
    if sorted(preds.area) != sorted(test.area) or len(preds) != len(test):
        problems.append(f"{fold_dir}: prediction keys differ from the target month's truth keys")
        return preds
    truth = test.set_index("area").loc[preds.area, "class_code"].to_numpy()
    if not (preds.y_true_code.to_numpy() == truth).all():
        problems.append(f"{fold_dir}: y_true differs from the prepared truth")
    if not (preds.y_pred_code.to_numpy() == preds.filter(like="p_").to_numpy().argmax(axis=1)).all():
        problems.append(f"{fold_dir}: hard label is not the argmax of the saved probabilities")
    return preds


def map_of(run, map_id):
    from scripts.run_stage2 import accept_consensus, cluster_map
    from src.utils import acceptance as acc
    if map_id in (None, "v7_final"):
        return acc.v7_final_map() if map_id == "v7_final" else None
    record = accept_consensus(run / "maps" / map_id)
    return cluster_map(run / "maps" / map_id, record)


def verify_development(run, results):
    from src.experiment import plan
    from src.utils import acceptance as acc
    g_of, _ = acc.accept_g_selection(run)
    problems, regions, folds = [], 0, 0
    maps = {}
    for fold in acc.schedule(run)["development"]:
        h = fold["horizon"]
        ex = Expected(run, h)
        for fold_dir in sorted((run / "development" / "folds" / f"h{h}" / fold["target_month"]).iterdir()) + \
                sorted((run / "development" / "oldmap" / f"h{h}" / fold["target_month"]).iterdir()):
            record = json.loads((fold_dir / "fold.json").read_text(encoding="utf-8"))
            if record.get("status") != "fitted":
                continue
            folds += 1
            mid = record.get("map_id")
            if mid not in maps:
                maps[mid] = map_of(run, mid)
            fold_label_problems(fold_dir, fold, ex, problems)
            regions += verify_gate_dir(fold_dir, fold, maps[mid] if record["arm"] != "pooled" else None,
                                       ex, run, g_of[str(h)], problems)
    check(results, f"development: {regions} region gates over {folds} arm-folds recompute from the expected gate "
                   "population (U<O, V=U-H, own windows, real keys, map clusters); labels/truth/routes recompute",
          not problems, problems[:6])
    table = read(run / "development" / "selection_table.csv")
    sel = json.loads((run / "development" / "selection.json").read_text(encoding="utf-8"))
    index = json.loads((run / "development" / "scheme_fold_index.json").read_text(encoding="utf-8"))
    base = read(run / "prepared" / "ledgers" / "dev_baselines.csv")
    base["target_month"] = base["target_label"]
    keys, bad = {}, []
    for scheme in plan.schemes():
        deltas, expert = {}, {}
        for h in HORIZONS:
            p = pd.concat([read(run / "development" / "folds" / f"h{h}" / t / index[f"h{h}_{t}"][scheme["scheme"]] /
                                "predictions.csv.gz") for t in plan.DEV_TARGETS])
            k = base[base.horizon == h].merge(p, on=["area", "target_month", "horizon"], validate="one_to_one")
            if len(k) != int((base.horizon == h).sum()):
                bad.append((scheme["scheme"], h, "keys"))
            k = main_cohort(k, h)
            deltas[h] = exact_macro(k.truth_code, k.y_pred_code) - exact_macro(k.truth_code, k.persistence_code)
            if h in (4, 8):
                expert[h] = exact_macro(k.truth_code, k.y_pred_code) - exact_macro(k.truth_code, k.expert_code)
        keys[scheme["scheme"]] = (1, min(deltas.values()), sum(deltas.values()) / 3, (expert[4] + expert[8]) / 2,
                                  sum(v == "L1" for v in scheme["l_vector"].values()),
                                  -plan.STRATEGIES.index(scheme["strategy"]), -scheme["l_vector_id"])
        row = table[table.scheme == scheme["scheme"]].iloc[0]
        if any(Fraction(row[f"delta_persistence_exact_h{h}"]) != deltas[h] for h in HORIZONS):
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
    arms = {"pooled": None, "rfmap_independent": "v7_final", "rfmap_shared": "v7_final",
            "xgbmap_shared": frozen["final_map"]["map_id"]}
    maps = {m: map_of(run, m) for m in set(arms.values())}
    for h in HORIZONS:
        g = frozen["g_selection"][str(h)]
        folds = [f for f in sched["stage3"] if f["horizon"] == h and f["status"] == "scheduled"]
        ex = Expected(run, h)
        for arm, mid in arms.items():
            for i, f in enumerate(folds):
                d = run / "final" / f"h{h}" / f["target_month"] / arm
                record = json.loads((d / "fold.json").read_text(encoding="utf-8"))
                if record.get("status") != "fitted":
                    problems.append(f"{d}: status {record.get('status')}")
                    continue
                if record.get("map_id") != mid or record.get("g_config") != g or \
                        record.get("local_config") != (None if arm == "pooled" else frozen["l_vector"][str(h)]):
                    problems.append(f"{d}: arm identity differs from the frozen record")
                p = fold_label_problems(d, f, ex, problems)
                regions += verify_gate_dir(d, f, maps[mid], ex, run, g, problems)
                gate = json.loads((d / "gate.json").read_text(encoding="utf-8"))
                g_path = run / "globals" / f"h{h}" / g / f"O{f['origin_month']}"
                g_rec = json.loads(g_path.with_suffix(".json").read_text(encoding="utf-8"))
                win = ex.window(mi(f["origin_month"]))
                if g_rec["fit_keys_sha256"] != keys_digest(ex.area[win], ex.month[win]) or \
                        g_rec["booster_sha256"] != gate["global"]["booster_sha256"]:
                    problems.append(f"{d}: fold global is not the stored global on [O-59, O)")
                if i not in (0, len(folds) - 1):
                    continue
                gb = nx.from_raw(g_path.with_suffix(".ubj").read_bytes())
                g_struct = nx.prefix_identity(gb)["sha256"]
                X = ex.snap[ex.snap.target_month == mi(f["target_month"])].set_index("area").loc[p.area, features].to_numpy(float)
                got = nx.proba(gb, X)
                for c, rec in gate["locals"].items():
                    rows = (p.cluster_id == int(c)).to_numpy()
                    booster = nx.from_raw((d / "models" / f"local_{c}.ubj").read_bytes())
                    if nx.sha(booster) != rec["booster_sha256"]:
                        problems.append(f"{d}: local_{c} bytes differ from the record")
                    if arm != "rfmap_independent" and nx.prefix_identity(booster, gb.num_boosted_rounds())["sha256"] != g_struct:
                        problems.append(f"{d}: shared local_{c} lacks the global prefix")
                    got[rows] = nx.proba(booster, X[rows])
                if not np.array_equal(got, p.filter(like="p_").to_numpy()):
                    problems.append(f"{d}: saved boosters do not replay the saved probabilities")
    check(results, f"final: every arm-fold fitted with its frozen identity; keys/truth/argmax/map clusters recompute; "
                   f"{regions} gates recompute from the expected population; first/last fold per H and arm replay "
                   "exactly from saved boosters; shared locals carry the global prefix", not problems, problems[:8])


# ---------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------

COHORT_ARMS = {
    "main_h4": ("main", "pooled", "rfmap_shared", "rfmap_independent", "expert", "persistence", "v7_partitioned_rf", "v7_pooled_rf"),
    "supp_h4": ("main", "pooled", "rfmap_shared", "rfmap_independent", "persistence", "v7_partitioned_rf", "v7_pooled_rf"),
    "main_h8": ("main", "pooled", "rfmap_shared", "rfmap_independent", "expert", "persistence", "v7_partitioned_rf", "v7_pooled_rf"),
    "supp_h8": ("main", "pooled", "rfmap_shared", "rfmap_independent", "persistence", "v7_partitioned_rf", "v7_pooled_rf"),
    "main_h12": ("main", "pooled", "rfmap_shared", "rfmap_independent", "persistence", "v7_partitioned_rf", "v7_pooled_rf"),
}
COL = {"main": "y_pred_xgbmap_shared", "pooled": "y_pred_pooled", "rfmap_shared": "y_pred_rfmap_shared",
       "rfmap_independent": "y_pred_rfmap_independent", "expert": "expert_code", "persistence": "persistence_code",
       "v7_partitioned_rf": "y_pred_v7_partitioned_rf", "v7_pooled_rf": "y_pred_v7_pooled_rf"}


def cohort_rows(keyed, cohort):
    h = int(cohort.split("_h")[1])
    rows = keyed[(keyed.horizon == h) & keyed.persistence_code.notna()]
    return rows[rows.expert_code.notna()] if cohort.startswith("main") and h != 12 else rows


def verify_report(run, results):
    from scripts import report_fourclass as rep
    report = json.loads((run / "report" / "report.json").read_text(encoding="utf-8"))
    keyed = read(run / "report" / "keyed_evaluation.csv.gz")
    accepted = rep.load_keyed(run)  # through prepared/frozen/fold acceptance
    cols = list(keyed.columns)
    a = accepted[cols].sort_values(["area", "target_month", "horizon"]).reset_index(drop=True)
    k = keyed.sort_values(["area", "target_month", "horizon"]).reset_index(drop=True)
    same = a.shape == k.shape and all(((a[c] == k[c]) | (a[c].isna() & k[c].isna())).all() for c in cols)
    check(results, "report: keyed_evaluation equals the accepted final predictions joined to prepared truth", same)
    bad = []
    if set(report["metrics"]) != set(COHORT_ARMS):
        bad.append(("cohorts", sorted(report["metrics"])))
    for cohort, arms in COHORT_ARMS.items():
        entry = report["metrics"].get(cohort, {"arms": {}, "n": None})
        rows = cohort_rows(keyed, cohort)
        if tuple(entry["arms"]) != arms or len(rows) != entry["n"]:
            bad.append((cohort, "arms/n"))
        for arm in arms:
            if rows[COL[arm]].isna().any() or not np.isclose(
                    macro(rows.truth_code, rows[COL[arm]].astype(int)), entry["arms"].get(arm, {}).get("macro_f1", np.nan),
                    rtol=0, atol=1e-12):
                bad.append((cohort, arm))
    check(results, "report: the five required cohorts with every required arm; sizes and macro F1 recompute", not bad, bad)

    # Bootstrap: regenerate the seed-42 sequence and its rejections independently.
    countries = sorted(keyed.country.unique().tolist())
    mats = {}
    for cohort, arms in COHORT_ARMS.items():
        rows = cohort_rows(keyed, cohort)
        pos = rows.country.map({c: i for i, c in enumerate(countries)}).to_numpy()
        m = np.zeros((len(countries), len(arms), 4, 4))
        for j, arm in enumerate(arms):
            np.add.at(m[:, j], (pos, rows.truth_code.to_numpy(int), rows[COL[arm]].to_numpy(float).astype(int)), 1)
        mats[cohort] = m
    rng = np.random.default_rng(42)
    mults, attempts = [], 0
    while len(mults) < 2000 and attempts < 20000:
        attempts += 1
        mult = np.bincount(rng.integers(0, len(countries), size=len(countries)), minlength=len(countries)).astype(float)
        if any(np.tensordot(mult, m[:, 0], axes=1).sum() == 0 for m in mats.values()):
            continue
        mults.append(mult)
    draws = read(run / "report" / "bootstrap_draws.csv.gz")
    saved = draws[report["bootstrap"]["countries"]].to_numpy(float) if report["bootstrap"]["countries"] == countries else None
    seq_ok = (len(mults) == 2000 and report["bootstrap"]["draws_accepted"] == 2000 and report["bootstrap"]["complete"]
              and saved is not None and saved.shape == (2000, len(countries)) and np.array_equal(saved, np.array(mults))
              and report["bootstrap"]["attempts"] == attempts)
    check(results, f"report: 2000 of {attempts} seed-42 draws regenerate exactly (same country universe, "
                   "multiplicities and rejection rule)", seq_ok)
    tp = lambda m: np.diagonal(m, axis1=-2, axis2=-1)  # noqa: E731
    stats_bad, d3_bad = [], []
    stats = {}
    M = np.array(mults)
    for cohort, arms in COHORT_ARMS.items():
        t = np.tensordot(M, mats[cohort], axes=1)          # (draws, arms, 4, 4)
        d = tp(t); fp = t.sum(axis=-2) - d; fn = t.sum(axis=-1) - d
        den = 2 * d + fp + fn
        f1 = np.divide(2 * d, den, out=np.zeros_like(d), where=den > 0).mean(axis=-1)
        stats[cohort] = f1
        for j, arm in enumerate(arms):
            if not np.allclose(draws[f"{cohort}:{arm}"].to_numpy(), f1[:, j], rtol=0, atol=1e-12):
                stats_bad.append((cohort, arm))
    for h in HORIZONS:
        cohort = f"main_h{h}"
        arms = COHORT_ARMS[cohort]
        delta = stats[cohort][:, arms.index("main")] - stats[cohort][:, arms.index("persistence")]
        lo, hi = np.percentile(delta, [2.5, 97.5], method="linear")
        point = report["metrics"][cohort]["arms"]["main"]["macro_f1"] - report["metrics"][cohort]["arms"]["persistence"]["macro_f1"]
        entry = report["scientific_target_D3"]["per_horizon"][str(h)]
        status = "pass" if point > 0 and lo > 0 else "fail"
        if not (np.isclose(lo, entry["ci95"][0], rtol=0, atol=1e-12) and np.isclose(hi, entry["ci95"][1], rtol=0, atol=1e-12)
                and status == entry["status"]):
            d3_bad.append(h)
    overall = "pass" if all(report["scientific_target_D3"]["per_horizon"][str(h)]["status"] == "pass" for h in HORIZONS) else "fail"
    check(results, "report: every saved draw statistic recomputes from the regenerated multiplicities", not stats_bad, stats_bad[:5])
    check(results, "report: both CI endpoints and the D3 per-H and overall decisions recompute",
          not d3_bad and overall == report["scientific_target_D3"]["overall"], d3_bad)
    sample = draws.sample(n=min(10, len(draws)), random_state=3)
    rep_bad = []
    for _, draw in sample.iterrows():
        mult = draw[countries].astype(int)
        for cohort, arms in COHORT_ARMS.items():
            rows = cohort_rows(keyed, cohort)
            rr = rows.iloc[np.repeat(np.arange(len(rows)), rows.country.map(mult).to_numpy())]
            for arm in arms:
                if not np.isclose(macro(rr.truth_code, rr[COL[arm]].astype(int)), draw[f"{cohort}:{arm}"], rtol=0, atol=1e-9):
                    rep_bad.append((int(draw["draw"]), cohort, arm))
    check(results, "report: 10 sampled draws recompute by row replication (sklearn)", not rep_bad, rep_bad[:5])


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
            import traceback
            check(results, f"{step.__name__} completed", False, traceback.format_exc()[-1500:])
    (out / "verification.json").write_text(json.dumps({
        "passed": all(r["passed"] for r in results), "checks": results}, indent=2, default=str), encoding="utf-8")
    failed = [r for r in results if not r["passed"]]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
