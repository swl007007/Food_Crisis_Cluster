#!/usr/bin/env python3
"""D37 / A11: fixed 24-month half-life recency-weighted ROOT for the 21 saved D34 roots.

python scripts/stage1_recency_root.py --d34-run D34_RUN --out NEW_DIR [--producer-rev 7b2bf6f]

Per root (sequential): rebuild the producer's r80 fitting / S / C / E3 rows from the frozen
snapshot (Parquet filter target_month <= 2020-12) with the D35 key / role / fitting-hash / window
guards, then GATE on the saved original root (S/C y_root, C p_root_*, E3 y_pred_pooled_code and
root_target p_pooled_* exact, both paired candidates). Only then ONE fresh
``fit_global(X_fit, y_fit, G, sample_weight=w)`` with w_i = u_i / mean_fit(u),
u_i = 2^(-((O-1)-m_i)/24) computed on the fitting rows only (float64, passed as float32).
The weighted root is saved, reloaded and scored on the same E3 (primary) and C (diagnostic) keys
next to the replayed original root and exact-origin persistence (hist_phase_o00 - 1, phase 5 -> 3,
missing kept NaN). No E4 weights, maps or Stage 2 inputs; no final-period ledger is read.
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from scripts import stage1_global_increment as gi  # noqa: E402
from scripts import stage1_shallow_replay as sr  # noqa: E402
from scripts.stage1_rootconf_compare import accept_mode  # noqa: E402
from src.experiment import plan  # noqa: E402
from src.feature.fourclass_features import load_schema, month_label  # noqa: E402
from src.metrics import fourclass  # noqa: E402
from src.model import native_xgb as nx  # noqa: E402
from src.utils import run_identity as rid  # noqa: E402

MAX_MONTH = gi.MAX_MONTH
HALF_LIFE = 24
LABELS = fourclass.CLASS_LABELS
PARTS = ("C", "E3")
MODELS = ("original", "weighted")
GROUPS = ("00", "01", "10", "11", "missing")
GateError = sr.GateError


# ---------------------------------------------------------------- pure helpers (tested)

def recency_weights(months, o_index: int, half_life: int = HALF_LIFE) -> dict:
    """Fitting-only mean-1 recency weights; months must lie in [O-59, O)."""
    m = np.asarray(months, dtype=np.int64)
    if len(m) == 0 or m.min() < o_index - plan.WINDOW or m.max() >= o_index:
        raise GateError("fitting months outside [O-59, O)")
    u = np.power(2.0, -((o_index - 1) - m).astype(np.float64) / half_life)
    w64 = u / u.mean()
    w32 = w64.astype(np.float32)
    return {"w64": w64, "w32": w32, "record": nx.weight_record(w32)}


def persistence_codes(phase) -> np.ndarray:
    """Exact-origin hist_phase_o00 -> code (phase 5 -> 3); missing stays NaN (never backfilled)."""
    p = np.asarray(phase, dtype=float)
    known = np.isfinite(p)
    if not np.all(np.isin(p[known], (1.0, 2.0, 3.0, 4.0, 5.0))):
        raise GateError("hist_phase_o00 holds values outside the IPC phases 1..5")
    return np.where(known, np.minimum(p, 4.0) - 1.0, np.nan)


def crisis(codes) -> np.ndarray:
    return np.asarray(codes) >= 2


def score_block(truth, pred, p=None) -> dict:
    out = sr.metrics(truth, pred) | {"n": int(len(truth))}
    c = out["crisis"]
    out["crisis"] = c | {"tn": int(len(truth) - c["tp"] - c["fp"] - c["fn"])}
    if p is not None:
        pc = np.asarray(p, dtype=float)[:, 2:].sum(axis=1)
        out["crisis_brier"] = float(np.mean((pc - crisis(truth).astype(float)) ** 2)) if len(pc) else None
    return out


def delta(a: dict, b: dict) -> dict:
    d = Fraction(a["crisis_f1_exact"]) - Fraction(b["crisis_f1_exact"])
    out = {"crisis_f1_delta_exact": str(d), "crisis_f1_delta": float(d),
           "macro_f1_fourclass_delta": a["macro_f1_fourclass"] - b["macro_f1_fourclass"]}
    if a.get("crisis_brier") is not None and b.get("crisis_brier") is not None:
        out["crisis_brier_delta"] = a["crisis_brier"] - b["crisis_brier"]
    return out


def transition_groups(frame: pd.DataFrame) -> dict:
    """Post-hoc D36 layers: origin persistence crisis x target truth crisis (+ missing origin)."""
    pers = frame["persistence_code"].to_numpy(float)
    t = crisis(frame["truth"].to_numpy())
    grp = np.where(~np.isfinite(pers), "missing",
                   np.where(pers >= 2, "1", "0").astype(object) + np.where(t, "1", "0").astype(object))
    yo, yw = crisis(frame["y_original"].to_numpy()), crisis(frame["y_weighted"].to_numpy())
    out = {}
    for g in GROUPS:
        k = grp == g
        okw, oko = yw[k] == t[k], yo[k] == t[k]
        out[g] = {"n": int(k.sum()), "corrected": int(np.sum(okw & ~oko)), "spoiled": int(np.sum(~okw & oko)),
                  "tp_change": int(np.sum(yw[k] & t[k]) - np.sum(yo[k] & t[k])),
                  "fp_change": int(np.sum(yw[k] & ~t[k]) - np.sum(yo[k] & ~t[k]))}
    return out


def score_frame(frame: pd.DataFrame) -> dict:
    """All keys: original vs weighted; matched non-missing-persistence keys: plus persistence."""
    if len(frame) == 0:
        return {"n": 0, "status": "no_data"}
    t = frame["truth"].to_numpy()
    prob = {m: frame[[f"p_{m}_{l}" for l in LABELS]].to_numpy(float) for m in MODELS}
    out = {"n": int(len(t)), "all": {m: score_block(t, frame[f"y_{m}"].to_numpy(), prob[m]) for m in MODELS}}
    out["all"]["weighted_minus_original"] = delta(out["all"]["weighted"], out["all"]["original"])
    k = np.isfinite(frame["persistence_code"].to_numpy(float))
    matched = {"n": int(k.sum()), "coverage": float(k.mean())}
    if k.any():
        for m in MODELS:
            matched[m] = score_block(t[k], frame[f"y_{m}"].to_numpy()[k], prob[m][k])
        matched["persistence"] = score_block(t[k], frame["persistence_code"].to_numpy(float)[k].astype(np.int64))
        for m in MODELS:
            matched[f"{m}_minus_persistence"] = delta(matched[m], matched["persistence"])
        matched["weighted_minus_original"] = delta(matched["weighted"], matched["original"])
    out["matched_persistence"] = matched
    out["transition_groups_post_hoc"] = transition_groups(frame)
    return out


def pooled(mats: dict) -> dict:
    out = {}
    for m, mat in mats.items():
        mat = np.asarray(mat, dtype=np.int64)
        tp, fp, fn = int(mat[2:, 2:].sum()), int(mat[:2, 2:].sum()), int(mat[2:, :2].sum())
        tn = int(mat.sum()) - tp - fp - fn
        f = Fraction(2 * tp, 2 * tp + fp + fn) if (2 * tp + fp + fn) else Fraction(0)
        out[m] = {"confusion_fourclass": mat.tolist(), "crisis": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                  "crisis_f1_exact": str(f), "crisis_f1": float(f),
                  "macro_f1_fourclass": fourclass.macro_f1_from_matrix(mat)}
    return out


def aggregate(per_root: dict, select) -> dict:
    """Mean-fold deltas and pooled confusions kept separate; 21 folds are not independent."""
    out = {}
    for part in PARTS:
        have = [r["scores"][part] for r in per_root.values() if select(r) and r["scores"][part].get("n", 0)]
        if not have:
            out[part] = {"folds_with_data": 0, "status": "no_data"}
            continue
        allp = pooled({m: np.sum([h["all"][m]["confusion_fourclass"] for h in have], axis=0) for m in MODELS})
        d = Fraction(allp["weighted"]["crisis_f1_exact"]) - Fraction(allp["original"]["crisis_f1_exact"])
        res = {"folds_with_data": len(have), "rows": int(sum(h["n"] for h in have)),
               "mean_fold_weighted_minus_original_crisis_f1": float(np.mean(
                   [h["all"]["weighted_minus_original"]["crisis_f1_delta"] for h in have])),
               "pooled_all": allp | {"weighted_minus_original": {"crisis_f1_delta_exact": str(d),
                                                                  "crisis_f1_delta": float(d)}}}
        mh = [h["matched_persistence"] for h in have if h["matched_persistence"]["n"]]
        if mh:
            mp = pooled({m: np.sum([h[m]["confusion_fourclass"] for h in mh], axis=0)
                         for m in MODELS + ("persistence",)})
            for m in MODELS:
                dd = Fraction(mp[m]["crisis_f1_exact"]) - Fraction(mp["persistence"]["crisis_f1_exact"])
                mp[f"{m}_minus_persistence"] = {"crisis_f1_delta_exact": str(dd), "crisis_f1_delta": float(dd)}
            res["pooled_matched_persistence"] = mp | {
                "rows": int(sum(h["n"] for h in mh)),
                "mean_fold": {f"{m}_minus_persistence": float(np.mean(
                    [h[f"{m}_minus_persistence"]["crisis_f1_delta"] for h in mh])) for m in MODELS}}
        res["transition_groups_post_hoc"] = {g: {k: int(sum(h["transition_groups_post_hoc"][g][k] for h in have))
                                                 for k in ("n", "corrected", "spoiled", "tp_change", "fp_change")}
                                             for g in GROUPS}
        out[part] = res
    return out


def dev_baseline_check(frame: pd.DataFrame, base: pd.DataFrame, horizon: int) -> dict:
    """Persistence codes on E3 keys vs prepared dev_baselines.csv (covers 2019-2020 targets only)."""
    rows = base[(base["horizon"] == horizon) & base["target_label"].isin(frame["target_month"].unique())]
    if len(rows) == 0:
        return {"status": "not_covered"}
    merged = frame[["area", "target_month", "truth", "persistence_code"]].merge(
        rows.rename(columns={"target_label": "target_month_l"})[["area", "target_month_l", "truth_code",
                                                                  "persistence_code"]],
        left_on=["area", "target_month"], right_on=["area", "target_month_l"], how="inner",
        suffixes=("", "_dev"), validate="one_to_one")
    a, b = merged["persistence_code"].to_numpy(float), merged["persistence_code_dev"].to_numpy(float)
    same = (np.isnan(a) & np.isnan(b)) | (a == b)
    return {"status": "checked", "e3_rows": int(len(frame)), "joined": int(len(merged)),
            "persistence_mismatches": int(np.sum(~same)),
            "truth_mismatches": int(np.sum(merged["truth"].to_numpy() != merged["truth_code"].to_numpy()))}


# ---------------------------------------------------------------- per root

def gate_root(run: Path, stage: Path, name: str, pair: dict, root: dict):
    """D35 guards + original-root replay gate. Returns (gate, data, booster, replay, root_ubj)."""
    gate = {"root": name, "checks": {}}

    def check(key, n_bad, n):
        gate["checks"][key] = {"mismatches": int(n_bad), "n": int(n)}

    data = sr.rebuild(run, root, max_month=MAX_MONTH, with_fitting=True)
    _, _, gfit, mfit = data["FIT"]
    check("fitting_keys_sha256", int(nx.keys_sha(gfit, mfit) != root.get("fitting_keys_sha256")), 1)
    saved_m = gi.read_csv(stage / "roots" / name / "fold_membership.csv.gz")
    mine = data["membership"]
    same = len(saved_m) == len(mine) and all(
        (saved_m[c].astype(str).to_numpy() == mine[c].astype(str).to_numpy()).all()
        for c in ("area", "target_month", "role", "class_code"))
    check("membership_keys_truth_roles", int(not same), len(mine))
    held = set()
    for part in ("S",) + PARTS:
        held |= gi.key_set(data[part][2], data[part][3])
    fit_keys = gi.key_set(gfit, month_label(mfit))
    check("fitting_excludes_S_C_target", len(fit_keys & held), len(fit_keys))
    lo, hi, o_index = data["window"]
    check("window_[O-59,O)", int(lo < o_index - plan.WINDOW or hi >= o_index), 1)
    months = np.concatenate([mfit] + [np.array([int(m[:4]) * 12 + int(m[5:]) - 1 for m in data[p][3]],
                                               dtype=np.int64) for p in ("S",) + PARTS])
    check("all_keys_le_2020-12", int(np.sum(months > MAX_MONTH)), len(months))

    hard, brier = pair["hard_f1"], pair["brier_crisis"]
    ubj = {c: stage / "checkpoints" / c / "xgb_root.ubj" for c in (hard, brier)}
    check("root_booster_sha", sum(rid.file_sha256(p) != root["root_booster_sha256"] for p in ubj.values()), 2)
    booster = nx.from_raw(ubj[hard].read_bytes())
    replay = {p: nx.proba(booster, data[p][0]) for p in ("S",) + PARTS}
    y_rep = {p: fourclass.argmax_codes(replay[p]) for p in replay}
    for e1, cand in (("hard", hard), ("brier", brier)):
        cdir = stage / "candidates" / cand
        for part, fname in (("S", "validation_predictions.csv.gz"), ("C", "confirmation_predictions.csv.gz")):
            f = gi.read_csv(cdir / fname)
            _, y, g, m = data[part]
            n = len(y)
            ok = len(f) == n and (f["area"].to_numpy() == g).all() and (f["target_month"].to_numpy() == m).all()
            check(f"{e1}_{part}_keys", int(not ok), n)
            if not ok:
                continue
            check(f"{e1}_{part}_truth", np.sum(f["y_true"].to_numpy() != y), n)
            check(f"{e1}_{part}_y_root", np.sum(f["y_root"].to_numpy() != y_rep[part]), n)
            if part == "C":
                check(f"{e1}_C_p_root", sr.exact_mismatches(f[[f"p_root_{l}" for l in LABELS]], replay["C"]), 4 * n)
        tgt = gi.read_csv(cdir / "target_predictions.csv")
        _, y, g, _ = data["E3"]
        n = len(y)
        ok = len(tgt) == n and (tgt["FEWSNET_admin_code"].to_numpy() == g).all()
        check(f"{e1}_E3_keys", int(not ok), n)
        if ok:
            check(f"{e1}_E3_truth", np.sum(tgt["y_true_code"].to_numpy() != y), n)
            check(f"{e1}_E3_y_pred_pooled_code", np.sum(tgt["y_pred_pooled_code"].to_numpy() != y_rep["E3"]), n)
    saved_root = gi.read_csv(stage / "roots" / name / "root_target_predictions.csv")
    n = len(data["E3"][1])
    ok = len(saved_root) == n and (saved_root["FEWSNET_admin_code"].to_numpy() == data["E3"][2]).all()
    check("E3_root_keys", int(not ok), n)
    if ok:
        check("E3_y_pred_pooled_code", np.sum(saved_root["y_pred_pooled_code"].to_numpy() != y_rep["E3"]), n)
        check("E3_p_pooled", sr.exact_mismatches(saved_root[[f"p_pooled_{l}" for l in LABELS]], replay["E3"]), 4 * n)
    gate["passed"] = all(c["mismatches"] == 0 for c in gate["checks"].values())
    return gate, data, replay, ubj[hard]


def fit_weighted_root(data: dict, g_config: str):
    """The single D37 fit: fresh G root on the original fitting rows with recency weights."""
    X_fit, y_fit, _, m_fit = data["FIT"]
    w = recency_weights(m_fit, data["window"][2])
    booster, record = nx.fit_global(X_fit, y_fit, plan.G_CONFIGS[g_config], sample_weight=w["w32"])
    return booster, record, w


def run_root(run, stage, out, name, pair, producer_rev, phase_col, base):
    root = json.loads((stage / "roots" / name / "root.json").read_text(encoding="utf-8"))
    gate, data, replay, root_ubj = gate_root(run, stage, name, pair, root)
    if not gate["passed"]:
        return gate, None, None
    g_config = root["g_config"]
    booster, record, w = fit_weighted_root(data, g_config)
    rdir = out / name
    rdir.mkdir()
    (rdir / "weighted_root.ubj").write_bytes(nx.raw(booster))
    frozen = nx.from_raw((rdir / "weighted_root.ubj").read_bytes())
    _, y_fit, gfit, mfit = data["FIT"]
    with gzip.open(rdir / "fitting_weights.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        pd.DataFrame({"area": gfit, "target_month": month_label(mfit), "class_code": y_fit,
                      "weight_float64": w["w64"], "weight_float32": w["w32"]}).to_csv(
            handle, index=False, float_format="%.17g")
    o_index = data["window"][2]
    meta = {"root": name, "horizon": int(root["horizon"]), "target_month": root["target_month"],
            "g_config": g_config, "origin_month": month_label(np.array([o_index]))[0],
            "half_life_months": HALF_LIFE, "fitting_rows": int(len(gfit)),
            "fitting_label_months": sorted(set(month_label(np.unique(mfit)))),
            "fitting_keys_sha256": nx.keys_sha(gfit, mfit),
            "original_root_source": str(root_ubj), "original_root_sha256": root["root_booster_sha256"],
            "weighted_root_sha256": rid.file_sha256(rdir / "weighted_root.ubj"),
            "fit_record": record, "producer_rev": producer_rev}
    rid.write_json_atomic(rdir / "weighted_root.json", meta)
    rows, checks = {}, {}
    for part in PARTS:
        X, y, g, m = data[part]
        pw = nx.proba(frozen, X)
        frame = pd.DataFrame({"area": g, "target_month": m, "horizon": int(root["horizon"]), "truth": y,
                              "y_original": fourclass.argmax_codes(replay[part]),
                              "y_weighted": fourclass.argmax_codes(pw),
                              "persistence_code": persistence_codes(X[:, phase_col])})
        for k, lab in enumerate(LABELS):
            frame[f"p_original_{lab}"] = replay[part][:, k]
        for k, lab in enumerate(LABELS):
            frame[f"p_weighted_{lab}"] = pw[:, k]
        rows[part] = frame
        with gzip.open(rdir / f"rows_{part}.csv.gz", "wt", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False, float_format="%.17g")
    checks["dev_baselines_E3"] = dev_baseline_check(rows["E3"], base, int(root["horizon"]))
    dc = checks["dev_baselines_E3"]
    if dc["status"] == "checked" and (dc["joined"] != dc["e3_rows"] or dc["persistence_mismatches"]
                                      or dc["truth_mismatches"]):
        gate = {**gate, "passed": False, "error": f"dev_baselines cross-check failed: {dc}"}
    return gate, rows, meta | {"dev_baselines_check": checks["dev_baselines_E3"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d34-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--producer-rev", default="7b2bf6f")
    args = parser.parse_args()
    run, out = args.d34_run.resolve(), args.out.resolve()
    if out.is_relative_to(run):
        raise ValueError("--out must not be inside the read-only D34 run")
    rid.refuse_existing(out, "D37 recency root")
    producer = rid.code_identity_at(args.producer_rev)
    if rid.code_identity() != rid.code_identity_at("HEAD"):
        raise RuntimeError("D37 must run from committed package code (working tree differs from HEAD)")
    script_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE, capture_output=True, text=True,
                                   check=True).stdout.strip()
    roots, cands, ident = accept_mode(run, plan.E1PAIR, producer, args.producer_rev)
    stage = Path(ident["stage"])
    phase_col = load_schema(sr.SCHEMA)["ordered_features"].index("hist_phase_o00")
    base_path = run / "prepared" / "ledgers" / "dev_baselines.csv"   # dev targets only; never baselines.csv
    base = pd.read_csv(base_path)
    out.mkdir(parents=True)
    identity = {"stage": "d37_recency_root", "d34_run": str(run), "producer_rev": args.producer_rev,
                "producer_code": producer, "script_commit": script_commit, "script_code": rid.code_identity(),
                "runtime": rid.runtime_identity(), "max_month": "2020-12", "half_life_months": HALF_LIFE,
                "dev_baselines": {"path": str(base_path), "sha256": rid.file_sha256(base_path)},
                "acceptance": {k: (str(v) if isinstance(v, Path) else v) for k, v in ident.items()},
                "inputs": {}}
    gates, per_root = {}, {}
    for name in roots:
        entry = roots[name]
        pair = {cands[c]["e1"]: c for c in cands if cands[c]["root"] == name}
        h = entry["horizon"]
        identity["inputs"][name] = {
            "snapshot": rid.file_sha256(run / "prepared" / f"snapshot_h{h}.parquet"),
            **{f: rid.file_sha256(stage / "roots" / name / f)
               for f in ("root.json", "fold_membership.csv.gz", "root_target_predictions.csv")},
            **{f"{c}/{f}": rid.file_sha256(stage / "candidates" / c / f) for c in pair.values()
               for f in ("candidate.json", "validation_predictions.csv.gz", "confirmation_predictions.csv.gz",
                         "target_predictions.csv")},
            **{f"{c}/xgb_root.ubj": rid.file_sha256(stage / "checkpoints" / c / "xgb_root.ubj") for c in pair.values()}}
        try:
            gate, rows, meta = run_root(run, stage, out, name, pair, args.producer_rev, phase_col, base)
        except GateError as exc:
            gate, rows, meta = {"root": name, "passed": False, "error": str(exc)}, None, None
        gates[name] = gate
        print(f"{name}: gate {'passed' if gate['passed'] else 'FAILED'}", flush=True)
        if rows is not None:
            per_root[name] = {**meta, "scores": {p: score_frame(rows[p]) for p in PARTS}}
    passed = all(g["passed"] for g in gates.values())
    rid.write_json_atomic(out / "gate.json", {"passed": passed, "rule": "exact equality, round-trip parse; "
                                              "failed roots are not fitted", "roots": gates})
    rid.write_json_atomic(out / "identity.json", identity)
    if not passed:
        print("GATE FAILED for at least one root: no summary computed; see gate.json")
        return 2
    summary = {
        "per_root": per_root,
        "by_horizon": {f"H{h}": aggregate(per_root, lambda r, h=h: r["horizon"] == h) for h in (4, 8, 12)},
        "by_target": {t: aggregate(per_root, lambda r, t=t: r["target_month"] == t) for t in plan.E1PAIR_TARGETS},
        "overall": aggregate(per_root, lambda r: True),
        "interpretation": ("Primary: same-key E3, original vs recency-weighted root, and versus persistence on "
                           "matched non-missing-persistence keys (coverage reported). C is diagnostic only. "
                           "mean_fold averages per-root deltas; pooled sums confusions. Transition groups are "
                           "post-hoc error layers, not routing rules. 21 overlapping, repeatedly developed "
                           "folds: no significance test; one fixed half-life only."),
    }
    rid.write_json_atomic(out / "summary.json", summary)
    rid.write_json_atomic(out / "completion.json", {"status": "completed", "outputs": rid.output_hashes(out)})
    print(f"D37 recency root completed: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
