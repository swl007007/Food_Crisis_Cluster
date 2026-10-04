#!/usr/bin/env python3
"""D46 / A20: fixed 2:1 crisis-class weighted ROOT for the 21 saved D34 roots.

python scripts/stage1_class_weight_root.py --d34-run D34_RUN --out NEW_DIR [--producer-rev 7b2bf6f]

Per root (sequential, stop at the first failure): the D37 rebuild (Parquet filter target_month <=
2020-12, fitting / S / C / E3 key, role, fitting-hash and window guards) and original-root replay
GATE (stage1_recency_root.gate_root) before any fit. Only then ONE fresh
``fit_global(X_fit, y_fit, G, sample_weight=w)`` with w = u / mean(u), u = 1 + I[y_fit >= 2]
computed in float64 from the FIT labels only and passed as float32. The weighted root is saved,
reloaded from UBJ (probabilities must reproduce exactly) and scored on the same FIT (in-sample),
C (in-window interpolation) and E3 (forward, primary) keys next to the replayed original root, a
zero-fit post-hoc control (original probabilities with classes 2 and 3 multiplied by 2, then
renormalised) and exact-origin persistence on matched non-missing keys. No local trees, E4 weights,
maps or Stage 2 inputs; no final-period ledger is read.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from scripts import stage1_recency_root as rr  # noqa: E402
from scripts import stage1_shallow_replay as sr  # noqa: E402
from scripts.stage1_rootconf_compare import accept_mode  # noqa: E402
from src.experiment import plan  # noqa: E402
from src.feature.fourclass_features import load_schema, month_label  # noqa: E402
from src.metrics import fourclass  # noqa: E402
from src.model import native_xgb as nx  # noqa: E402
from src.utils import run_identity as rid  # noqa: E402

CRISIS_COST = 2.0
G_BY_H = {4: "G1", 8: "G4", 12: "G2"}
LABELS = fourclass.CLASS_LABELS
PARTS = ("FIT", "C", "E3")
ROLE = {"FIT": "in-sample fitting rows", "C": "in-window historical interpolation (diagnostic)",
        "E3": "forward target month (primary)"}
ARMS = ("original", "weighted", "posthoc2x")
ARM_DELTAS = (("weighted", "original"), ("weighted", "posthoc2x"), ("posthoc2x", "original"))
RANK_TOL = 1e-12
RULE = ("exact equality, round-trip parse; a root failing the pre-fit gate is not fitted; the first gate, "
        "fit, reload, scoring or dev_baselines failure stops the run before later roots")
GateError = rr.GateError


# ---------------------------------------------------------------- pure helpers (tested)

def crisis_weights(y_fit) -> dict:
    """Mean-1 2:1 crisis-class weights from the FIT labels only (float64, then float32)."""
    y = np.asarray(y_fit, dtype=np.int64)
    crisis = y >= 2
    # Guard, not a formula failure: with a single crisis group u / mean(u) is simply all ones, so the
    # 2:1 crisis vs non-crisis contrast does not exist. All 21 real D34 FIT pools hold both groups.
    if not crisis.any() or crisis.all():
        raise GateError("FIT rows have no crisis or no non-crisis rows; the 2:1 class contrast is undefined "
                        "there (u / mean(u) would be all ones)")
    u = 1.0 + crisis.astype(np.float64) * (CRISIS_COST - 1.0)
    w64 = u / u.mean()
    w32 = nx.check_sample_weight(w64, len(y))
    values = {name: np.unique(w32[mask]) for name, mask in (("crisis", crisis), ("noncrisis", ~crisis))}
    if any(len(v) != 1 for v in values.values()):
        raise GateError("class weights are not constant within a class")
    return {"w64": w64, "w32": w32, "record": nx.weight_record(w32),
            "class_values_float32": {k: float(v[0]) for k, v in values.items()},
            "class_counts": {"crisis": int(crisis.sum()), "noncrisis": int((~crisis).sum())},
            "float64_mean": float(w64.mean()),
            "kish_ess_note": "stored by weight_record; not interpreted as an independent sample size"}


def posthoc2x(p_original) -> np.ndarray:
    """Zero-fit control: classes 2 and 3 times 2, renormalised (float64)."""
    q = np.asarray(p_original, dtype=np.float32).astype(np.float64)
    q[:, 2:] *= CRISIS_COST
    return q / q.sum(axis=1, keepdims=True)


def crisis_score(p) -> np.ndarray:
    """Ranking score s = (p2 + p3) / sum(p0..p3) in float64."""
    p = np.asarray(p, dtype=np.float64)
    return p[:, 2:].sum(axis=1) / p.sum(axis=1)


def log_loss_fourclass(truth, p) -> float:
    """Mean -log p[i, y_i] (float64, no clipping); a non-positive true-class probability stops."""
    p = np.asarray(p, dtype=np.float64)
    pt = p[np.arange(len(p)), np.asarray(truth, dtype=np.int64)]
    if np.any(~(pt > 0)):
        raise GateError(f"non-positive true-class probability in {int(np.sum(~(pt > 0)))} rows")
    return float(np.mean(-np.log(pt)))


def ranking(truth, p) -> dict:
    """Within-root crisis AUC / AP on s; ineligible when only one crisis class is present."""
    z = rr.crisis(truth).astype(int)
    if len(z) == 0 or z.min() == z.max():
        return {"eligible": False, "auc": None, "ap": None}
    s = crisis_score(p)
    return {"eligible": True, "auc": float(roc_auc_score(z, s)), "ap": float(average_precision_score(z, s))}


def posthoc_rank_check(p_original, p_post) -> dict:
    """Fail closed: the control's crisis score must equal 2s/(1+s) within RANK_TOL and keep the original order.

    Only exact float ties in s (or in the control score) are labelled ties. A strict inversion (s_i < s_j but
    t_i > t_j) is never labelled a tie: inversions no larger than RANK_TOL are counted and reported as
    floating-point rounding; any larger inversion, or a larger map error, raises GateError and stops the run."""
    s, t = crisis_score(p_original), crisis_score(p_post)
    if len(s) == 0:
        return {"n": 0}
    map_error = float(np.max(np.abs(t - 2 * s / (1 + s))))
    order = np.lexsort((t, s))                       # by s, exact s ties ordered by t
    drops = -np.diff(t[order])                       # every positive drop is a strict inversion in s
    violation = float(max(0.0, drops.max())) if len(drops) else 0.0
    out = {"n": int(len(s)), "max_abs_map_error": map_error, "max_order_violation": violation,
           "rounding_inversions": int(np.sum(drops > 0)),
           "rank_preserved": bool(violation <= RANK_TOL and map_error <= RANK_TOL),
           "exact_tied_rows_original": int(len(s) - len(np.unique(s))),
           "exact_tied_rows_posthoc": int(len(t) - len(np.unique(t))),
           "max_abs_average_rank_diff": float(np.max(np.abs(rankdata(s) - rankdata(t)))),
           "tolerance": RANK_TOL,
           "note": ("average-rank differences come only from exact float ties and counted rounding "
                    "inversions <= tolerance; anything larger stops the run")}
    if not out["rank_preserved"]:
        raise GateError(f"post-hoc x2 control breaks 2s/(1+s) or the original crisis ranking beyond "
                        f"{RANK_TOL}: map error {map_error}, max inversion {violation}")
    return out


def block(truth, pred, p=None, logloss=True) -> dict:
    out = rr.score_block(truth, pred, p)
    if p is not None and logloss:
        out["logloss_fourclass"] = log_loss_fourclass(truth, p)
    return out


def delta(a: dict, b: dict) -> dict:
    out = rr.delta(a, b)
    if a.get("logloss_fourclass") is not None and b.get("logloss_fourclass") is not None:
        out["logloss_fourclass_delta"] = a["logloss_fourclass"] - b["logloss_fourclass"]
    return out


def score_part(frame: pd.DataFrame) -> dict:
    """All keys: three arms; matched non-missing-persistence keys: plus persistence (no log loss)."""
    if len(frame) == 0:
        return {"n": 0, "status": "no_data"}
    t = frame["truth"].to_numpy()
    prob = {a: frame[[f"p_{a}_{lab}" for lab in LABELS]].to_numpy(np.float64) for a in ARMS}
    out = {"n": int(len(t)), "crisis_prevalence": float(rr.crisis(t).mean()),
           "all": {a: block(t, frame[f"y_{a}"].to_numpy(), prob[a]) for a in ARMS}}
    for a, b in ARM_DELTAS:
        out["all"][f"{a}_minus_{b}"] = delta(out["all"][a], out["all"][b])
    out["ranking"] = {a: ranking(t, prob[a]) for a in ARMS}
    out["posthoc_rank_check"] = posthoc_rank_check(prob["original"], prob["posthoc2x"])
    k = np.isfinite(frame["persistence_code"].to_numpy(float))
    matched = {"n": int(k.sum()), "coverage": float(k.mean())}
    if k.any():
        pers = frame["persistence_code"].to_numpy(float)[k].astype(np.int64)
        for a in ARMS:
            matched[a] = block(t[k], frame[f"y_{a}"].to_numpy()[k], prob[a][k])
        matched["persistence"] = block(t[k], pers, np.eye(nx.N_CLASSES)[pers], logloss=False)
        for a in ARMS:
            matched[f"{a}_minus_persistence"] = delta(matched[a], matched["persistence"])
        for a, b in ARM_DELTAS:
            matched[f"{a}_minus_{b}"] = delta(matched[a], matched[b])
    out["matched_persistence"] = matched
    return out


def _pool(blocks: dict) -> dict:
    """Summed confusions (exact crisis F1 / macro-F1) and row-weighted Brier / log loss."""
    out = rr.pooled({m: np.sum([b["confusion_fourclass"] for b in bl], axis=0) for m, bl in blocks.items()})
    for m, bl in blocks.items():
        n = sum(b["n"] for b in bl)
        for key in ("crisis_brier", "logloss_fourclass"):
            if all(b.get(key) is not None for b in bl):
                out[m][key] = float(sum(b[key] * b["n"] for b in bl) / n)
    return out


def _pooled_deltas(p: dict, pairs) -> dict:
    out = {}
    for a, b in pairs:
        d = Fraction(p[a]["crisis_f1_exact"]) - Fraction(p[b]["crisis_f1_exact"])
        out[f"{a}_minus_{b}"] = {"crisis_f1_delta_exact": str(d), "crisis_f1_delta": float(d),
                                 "macro_f1_fourclass_delta": p[a]["macro_f1_fourclass"] - p[b]["macro_f1_fourclass"],
                                 **{f"{k}_delta": p[a][k] - p[b][k] for k in ("crisis_brier", "logloss_fourclass")
                                    if k in p[a] and k in p[b]}}
    return out


def _mean_fold(entries: list, models, pairs) -> dict:
    keys = ("crisis_f1", "macro_f1_fourclass", "crisis_brier", "logloss_fourclass")
    out = {m: {k: float(np.mean([e[m][k] for e in entries])) for k in keys
               if all(e[m].get(k) is not None for e in entries)} for m in models}
    for a, b in pairs:
        name = f"{a}_minus_{b}"
        out[name] = {k: float(np.mean([e[name][k] for e in entries])) for k in
                     ("crisis_f1_delta", "macro_f1_fourclass_delta", "crisis_brier_delta", "logloss_fourclass_delta")
                     if all(k in e[name] for e in entries)}
    return out


def aggregate(per_root: dict, select) -> dict:
    """Pooled (summed confusions, row-weighted losses) and mean-fold kept separate; AUC/AP mean-fold only."""
    out = {}
    pers_pairs = tuple((a, "persistence") for a in ARMS)
    for part in PARTS:
        have = [r["scores"][part] for r in per_root.values() if select(r) and r["scores"][part].get("n", 0)]
        if not have:
            out[part] = {"folds_with_data": 0, "status": "no_data"}
            continue
        rows = int(sum(h["n"] for h in have))
        pooled_all = _pool({a: [h["all"][a] for h in have] for a in ARMS})
        res = {"role": ROLE[part], "folds_with_data": len(have), "rows": rows,
               "crisis_prevalence_pooled": float(sum(h["crisis_prevalence"] * h["n"] for h in have) / rows),
               "pooled_all": pooled_all | _pooled_deltas(pooled_all, ARM_DELTAS),
               "mean_fold_all": _mean_fold([h["all"] for h in have], ARMS, ARM_DELTAS),
               "ranking_mean_fold": {}}
        for a in ARMS:
            el = [h["ranking"][a] for h in have if h["ranking"][a]["eligible"]]
            res["ranking_mean_fold"][a] = {"eligible_folds": len(el),
                                           "auc": float(np.mean([e["auc"] for e in el])) if el else None,
                                           "ap": float(np.mean([e["ap"] for e in el])) if el else None}
        checks = [h["posthoc_rank_check"] for h in have]
        res["posthoc_rank_check"] = {"all_preserved": all(c["rank_preserved"] for c in checks),
                                     "max_order_violation": max(c["max_order_violation"] for c in checks),
                                     "max_abs_map_error": max(c["max_abs_map_error"] for c in checks),
                                     "rounding_inversions": sum(c["rounding_inversions"] for c in checks)}
        mh = [h["matched_persistence"] for h in have if h["matched_persistence"]["n"]]
        if mh:
            mp = _pool({m: [h[m] for h in mh] for m in ARMS + ("persistence",)})
            res["matched_persistence"] = {
                "folds_with_data": len(mh), "rows": int(sum(h["n"] for h in mh)),
                "pooled": mp | _pooled_deltas(mp, pers_pairs + ARM_DELTAS),
                "mean_fold": _mean_fold(mh, ARMS + ("persistence",), pers_pairs + ARM_DELTAS)}
        out[part] = res
    return out


# ---------------------------------------------------------------- per root

def fit_weighted_root(data: dict, g_config: str):
    """The single D46 fit: fresh G root on the original fitting rows with the 2:1 class weights."""
    X_fit, y_fit, _, _ = data["FIT"]
    w = crisis_weights(y_fit)
    booster, record = nx.fit_global(X_fit, y_fit, plan.G_CONFIGS[g_config], sample_weight=w["w32"])
    return booster, record, w


def check_fit(booster, record: dict, w: dict, g_config: str, n_rows: int) -> None:
    """Fail closed: scalar base score .5, fixed four classes, configured rounds, recorded weight bytes."""
    _, rounds = nx.booster_params(plan.G_CONFIGS[g_config])
    problems = []
    if record.get("base_score") != "5E-1" or nx.base_score(booster) != "5E-1":
        problems.append(f"base_score {record.get('base_score')!r} != '5E-1'")
    if nx.num_class(booster) != nx.N_CLASSES:
        problems.append("class axis is not four")
    if booster.num_boosted_rounds() != rounds or record.get("rounds_total") != rounds:
        problems.append(f"rounds {booster.num_boosted_rounds()} != configured {rounds}")
    if record.get("sample_weight", {}).get("sha256") != hashlib.sha256(
            np.ascontiguousarray(w["w32"]).tobytes()).hexdigest():
        problems.append("recorded sample_weight sha differs from the float32 weight bytes")
    if record.get("rows") != n_rows:
        problems.append("fit record rows differ from the FIT rows")
    if problems:
        raise RuntimeError("weighted root fit check failed: " + "; ".join(problems))


def part_frame(X, y, g, m, horizon, p_original, p_weighted, phase_col) -> pd.DataFrame:
    probs = {"original": p_original, "weighted": p_weighted, "posthoc2x": posthoc2x(p_original)}
    frame = pd.DataFrame({"area": g, "target_month": m, "horizon": int(horizon), "truth": y,
                          "persistence_code": rr.persistence_codes(X[:, phase_col])})
    for arm, p in probs.items():
        frame[f"y_{arm}"] = fourclass.argmax_codes(p)
        for k, lab in enumerate(LABELS):
            frame[f"p_{arm}_{lab}"] = p[:, k]
    return frame


def run_root(run, stage, out, name, pair, producer_rev, phase_col, base):
    """Returns (gate, rows, meta); rows is None when the gate did not pass (no fit happened)."""
    root = json.loads((stage / "roots" / name / "root.json").read_text(encoding="utf-8"))
    gate, data, replay, root_ubj = rr.gate_root(run, stage, name, pair, root)
    h = int(root["horizon"])
    g_config = root["g_config"]
    gate["checks"]["g_config_locked"] = {"mismatches": int(G_BY_H.get(h) != g_config), "n": 1}
    gate["passed"] = gate["passed"] and G_BY_H.get(h) == g_config
    if not gate["passed"]:
        return gate, None, None
    booster, record, w = fit_weighted_root(data, g_config)
    X_fit, y_fit, gfit, mfit = data["FIT"]
    check_fit(booster, record, w, g_config, len(y_fit))
    rdir = out / name
    rdir.mkdir()
    (rdir / "weighted_root.ubj").write_bytes(nx.raw(booster))
    frozen = nx.from_raw((rdir / "weighted_root.ubj").read_bytes())
    original = nx.from_raw(root_ubj.read_bytes())          # sha gated in gate_root
    inputs = {"FIT": data["FIT"], "C": data["C"], "E3": data["E3"]}
    p_orig = {"FIT": nx.proba(original, X_fit), "C": replay["C"], "E3": replay["E3"]}
    rows, reload_checks = {}, {}
    for part in PARTS:
        X, y, g, m = inputs[part]
        if part == "FIT":
            m = month_label(m)
        pw = nx.proba(frozen, X)
        mism = sr.exact_mismatches(nx.proba(booster, X), pw)
        reload_checks[part] = {"mismatches": int(mism), "n": int(pw.size)}
        if mism:
            raise RuntimeError(f"reloaded weighted root does not reproduce its {part} probabilities ({mism} cells)")
        rows[part] = part_frame(X, y, g, m, h, p_orig[part], pw, phase_col)
        with gzip.open(rdir / f"rows_{part}.csv.gz", "wt", encoding="utf-8", newline="") as handle:
            rows[part].to_csv(handle, index=False, float_format="%.17g")
    with gzip.open(rdir / "fitting_weights.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        pd.DataFrame({"area": gfit, "target_month": month_label(mfit), "class_code": y_fit,
                      "weight_float64": w["w64"], "weight_float32": w["w32"]}).to_csv(
            handle, index=False, float_format="%.17g")
    o_index = data["window"][2]
    meta = {"root": name, "horizon": h, "target_month": root["target_month"], "g_config": g_config,
            "origin_month": month_label(np.array([o_index]))[0], "crisis_cost": CRISIS_COST,
            "weights": {k: v for k, v in w.items() if k not in ("w64", "w32")},
            "fitting_rows": int(len(gfit)), "fitting_keys_sha256": nx.keys_sha(gfit, mfit),
            "original_root_source": str(root_ubj), "original_root_sha256": root["root_booster_sha256"],
            "weighted_root_sha256": rid.file_sha256(rdir / "weighted_root.ubj"),
            "reload_exact": reload_checks, "fit_record": record, "producer_rev": producer_rev}
    rid.write_json_atomic(rdir / "weighted_root.json", meta)
    dc = rr.dev_baseline_check(rows["E3"], base, h)
    if dc["status"] == "checked" and (dc["joined"] != dc["e3_rows"] or dc["persistence_mismatches"]
                                      or dc["truth_mismatches"]):
        gate = {**gate, "passed": False, "error": f"dev_baselines cross-check failed: {dc}"}
    return gate, rows, meta | {"dev_baselines_check": dc}


def run_all(run, stage, out, roots, cands, producer_rev, phase_col, base) -> tuple[int, dict, dict]:
    """Sequential roots; the first gate / fit / scoring failure writes evidence and stops."""
    gates, per_root = {}, {}
    for name in roots:
        pair = {cands[c]["e1"]: c for c in cands if cands[c]["root"] == name}
        try:
            gate, rows, meta = run_root(run, stage, out, name, pair, producer_rev, phase_col, base)
            if rows is not None and gate["passed"]:
                per_root[name] = {**meta, "scores": {p: score_part(rows[p]) for p in PARTS}}
        except Exception as exc:   # noqa: BLE001 - any failure stops the run with evidence
            gate = {"root": name, "passed": False, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()}
        gates[name] = gate
        print(f"{name}: {'passed' if gate['passed'] else 'FAILED'}", flush=True)
        if not gate["passed"]:
            rid.write_json_atomic(out / "gate.json", {"passed": False, "roots": gates, "rule": RULE})
            rid.write_json_atomic(out / "failure.json", {
                "status": "stopped", "failed_root": name, "error": gate.get("error", "gate mismatches"),
                "completed_roots": sorted(per_root), "not_attempted": [r for r in roots if r not in gates]})
            return 2, gates, per_root
    rid.write_json_atomic(out / "gate.json", {"passed": True, "roots": gates, "rule": RULE})
    return 0, gates, per_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d34-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--producer-rev", default="7b2bf6f")
    args = parser.parse_args()
    run, out = args.d34_run.resolve(), args.out.resolve()
    if out.is_relative_to(run):
        raise ValueError("--out must not be inside the read-only D34 run")
    rid.refuse_existing(out, "D46 class-weight root")
    producer = rid.code_identity_at(args.producer_rev)
    if rid.code_identity() != rid.code_identity_at("HEAD"):
        raise RuntimeError("D46 must run from committed package code (working tree differs from HEAD)")
    script_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE, capture_output=True, text=True,
                                   check=True).stdout.strip()
    roots, cands, ident = accept_mode(run, plan.E1PAIR, producer, args.producer_rev)
    stage = Path(ident["stage"])
    phase_col = load_schema(sr.SCHEMA)["ordered_features"].index("hist_phase_o00")
    base_path = run / "prepared" / "ledgers" / "dev_baselines.csv"   # dev targets only; never baselines.csv
    base = pd.read_csv(base_path)
    identity = {"stage": "d46_class_weight_root", "d34_run": str(run), "producer_rev": args.producer_rev,
                "producer_code": producer, "script_commit": script_commit, "script_code": rid.code_identity(),
                "runtime": rid.runtime_identity(), "max_month": "2020-12", "crisis_cost": CRISIS_COST,
                "weight_rule": "u = 1 + I[y_fit >= 2]; w = u / mean(u) float64 -> float32 (FIT labels only)",
                "g_by_horizon": G_BY_H,
                "dev_baselines": {"path": str(base_path), "sha256": rid.file_sha256(base_path)},
                "acceptance": {k: (str(v) if isinstance(v, Path) else v) for k, v in ident.items()},
                "inputs": {}}
    for name, entry in roots.items():
        pair = [c for c in cands if cands[c]["root"] == name]
        identity["inputs"][name] = {
            "snapshot": rid.file_sha256(run / "prepared" / f"snapshot_h{entry['horizon']}.parquet"),
            **{f: rid.file_sha256(stage / "roots" / name / f)
               for f in ("root.json", "fold_membership.csv.gz", "root_target_predictions.csv")},
            **{f"{c}/{f}": rid.file_sha256(stage / "candidates" / c / f) for c in pair
               for f in ("candidate.json", "validation_predictions.csv.gz", "confirmation_predictions.csv.gz",
                         "target_predictions.csv")},
            **{f"{c}/xgb_root.ubj": rid.file_sha256(stage / "checkpoints" / c / "xgb_root.ubj") for c in pair}}
    out.mkdir(parents=True)
    rid.write_json_atomic(out / "identity.json", identity)
    code, gates, per_root = run_all(run, stage, out, roots, cands, args.producer_rev, phase_col, base)
    if code:
        print("STOPPED at the first failure: no summary computed; see gate.json / failure.json")
        return code
    summary = {
        "per_root": per_root,
        "by_horizon": {f"H{h}": aggregate(per_root, lambda r, h=h: r["horizon"] == h) for h in (4, 8, 12)},
        "by_target": {t: aggregate(per_root, lambda r, t=t: r["target_month"] == t) for t in plan.E1PAIR_TARGETS},
        "overall": aggregate(per_root, lambda r: True),
        "interpretation": ("Primary: same-key E3 per H, weighted - original, weighted - posthoc2x, weighted - "
                           "persistence (matched non-missing exact-origin keys, coverage reported) and "
                           "posthoc2x - original. FIT is in-sample, C in-window interpolation (diagnostic). "
                           "Weighted outputs are cost-sensitive scores, not calibrated posteriors; unweighted "
                           "Brier / log loss are reported with no promised direction. AUC/AP are within-root "
                           "and mean-fold only. Kish ESS is not an independent sample size. 21 overlapping, "
                           "repeatedly developed folds: no significance test, no adoption rule; one fixed 2:1 "
                           "cost."),
    }
    rid.write_json_atomic(out / "summary.json", summary)
    rid.write_json_atomic(out / "completion.json", {"status": "completed", "outputs": rid.output_hashes(out)})
    print(f"D46 class-weight root completed: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
