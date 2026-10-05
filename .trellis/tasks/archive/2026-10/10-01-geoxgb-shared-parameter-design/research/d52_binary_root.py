#!/usr/bin/env python3
"""D52 / A26: independent binary-objective root diagnostic for the 21 saved D34 roots (one-off research).

python .trellis/tasks/10-01-geoxgb-shared-parameter-design/research/d52_binary_root.py \
    --d34-run D34_RUN --out C:\\Users\\swl00\\geoxgb_runs\\geoxgb-d52-binary-root-20261002 \
    [--producer-rev 7b2bf6f]
python .../d52_binary_root.py --selftest      # synthetic fits only, no real data

Two passes over the accepted 21-root inventory, in inventory order. Pass 1 (gate pass, zero fits): for
EVERY root, D34 pinned acceptance, the D37 rebuild and original-root replay GATE
(stage1_recency_root.gate_root) on the full-162 matrices, the G lock, the original-arm D49/D50 consistency
gate (matched n/excluded, original argmax and persistence confusions, original D50 cells) on the replayed E3
probabilities, and the FIT-key sha against the root's fitting_keys_sha256; matrices are discarded after each
root. Any pass-1 failure writes gate.json / failure.json and stops with no fit at all. Pass 2 (fit pass),
only after all 21 roots passed pass 1, root by root: rebuild and re-assert the replay gate and an identical
FIT-key sha, then ONE standalone ``xgb.train`` run (``fit_binary_root``):
G params copied from ``booster_params(G)`` with ``objective`` -> ``binary:logistic``, ``num_class`` and
``multi_strategy`` removed and ``base_score=0.5`` added; configured rounds; all original FIT rows, order,
162 features; target I[code >= 2]; no weight, no margin. The booster is saved, reloaded from UBJ (exact on
FIT/C/E3) and scored next to the original root (normalised mass s >= .5 and pipeline argmax -> code >= 2)
and exact-origin persistence. The package (native_xgb, plan, schema) is not edited. No local trees, maps,
Stage 2/3 inputs, thresholds or adoption.
"""
from __future__ import annotations

import argparse
import ast
import copy
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
RESEARCH = SCRIPT.parent
PACKAGE = REPO / "FEWSNETGeoXGBExperiment"
sys.path.insert(0, str(PACKAGE))
from scripts import stage1_class_weight_root as cw  # noqa: E402
from scripts import stage1_recency_root as rr  # noqa: E402
from scripts.stage1_rootconf_compare import accept_mode  # noqa: E402
from src.experiment import plan  # noqa: E402
from src.feature.fourclass_features import month_label  # noqa: E402
from src.metrics import fourclass  # noqa: E402
from src.model import native_xgb as nx  # noqa: E402
from src.utils import run_identity as rid  # noqa: E402

G_BY_H = {4: "G1", 8: "G4", 12: "G2"}
HORIZONS = (4, 8, 12)
LABELS = fourclass.CLASS_LABELS
PARTS = ("FIT", "C", "E3")
ROLE = {"FIT": "in-sample fitting rows", "C": "in-window historical interpolation",
        "E3": "forward target month (primary on matched exact-origin keys)"}
MODEL_ARMS = ("binary", "original_mass", "original_argmax")
ARMS = MODEL_ARMS + ("persistence",)
SCORE_OF = {"binary": "binary", "original_mass": "original", "original_argmax": "original"}
SCORES = ("binary", "original")
CONTRAST_A = (("binary", "original_mass"),)
CONTRAST_B = (("binary", "original_argmax"), ("binary", "persistence"), ("original_argmax", "persistence"))
ALL_PAIRS = CONTRAST_A + CONTRAST_B
CODES = (0, 1, 2, 3)
CF = ("tp", "fp", "fn", "tn")
N_FEATURES = 162
PHASE_FULL_INDEX = 87
THRESHOLD = 0.5
BASE_SCORE = 0.5
EPS = float(np.finfo(np.float64).eps)
D49_SUMMARY = RESEARCH / "d49_summary.json"
D50_SUMMARY = RESEARCH / "d50_summary.json"
AUC_TOL = 1e-12
FIT_RULE = ("fresh binary:logistic root on ALL original rebuilt FIT rows, order and full 162 features "
            "(incl. missing-origin rows); target I[four-class code >= 2]; G params copied with objective "
            "replaced, num_class/multi_strategy removed, base_score=0.5 added; G rounds/seed unchanged; "
            "no weights or margins")
RULE = ("exact equality, round-trip parse; pass 1 gates all 21 roots (replay, G lock, D49/D50 consistency, FIT "
        "keys) before any fit; a pass-1 failure stops with zero fits; in pass 2 the first re-gate, fit, reload, "
        "scoring or dev_baselines failure stops the run")
GateError = rr.GateError


# ---------------------------------------------------------------- config and binary fit

def plan_defaults() -> dict:
    return {"G_CONFIGS": copy.deepcopy(plan.G_CONFIGS), "XGB_BASE": dict(plan.XGB_BASE)}


def check_defaults(snapshot: dict) -> None:
    if plan_defaults() != snapshot:
        raise GateError("plan.G_CONFIGS or plan.XGB_BASE changed during the run")


def binary_params(g_config: str) -> tuple[dict, int, dict, dict]:
    """(binary params, rounds, four-class params, diff); the diff must be exactly the approved one."""
    four, rounds = nx.booster_params(plan.G_CONFIGS[g_config])
    four = copy.deepcopy(four)
    if four.get("objective") != "multi:softprob" or "num_class" not in four or "multi_strategy" not in four:
        raise GateError("four-class G params lack objective multi:softprob / num_class / multi_strategy")
    if "base_score" in four:
        raise GateError("four-class G params already set base_score")
    params = {k: v for k, v in four.items() if k not in ("num_class", "multi_strategy")}
    params["objective"] = "binary:logistic"
    params["base_score"] = BASE_SCORE
    diff = param_diff(four, params)
    if diff != expected_diff(four):
        raise GateError(f"binary parameter diff {diff} is not the approved diff")
    return params, rounds, four, diff


def expected_diff(four: dict) -> dict:
    return {"replaced": {"objective": [four["objective"], "binary:logistic"]},
            "removed": {"multi_strategy": four["multi_strategy"], "num_class": four["num_class"]},
            "added": {"base_score": BASE_SCORE}}


def param_diff(old: dict, new: dict) -> dict:
    return {"replaced": {k: [old[k], new[k]] for k in sorted(old) if k in new and old[k] != new[k]},
            "removed": {k: old[k] for k in sorted(old) if k not in new},
            "added": {k: new[k] for k in sorted(new) if k not in old}}


def binary_target(y) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    if y.size and (y.min() < 0 or y.max() > 3):
        raise GateError("four-class labels outside 0..3")
    return (y >= 2).astype(np.int64)


def labels_sha(y) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(y, dtype=np.int64)).tobytes()).hexdigest()


def num_feature(booster) -> str:
    return json.loads(bytes(booster.save_raw("json")))["learner"]["learner_model_param"]["num_feature"]


def objective_name(booster) -> str:
    return nx.resolved_config(booster)["learner"]["objective"]["name"]


def proba_binary(booster, X) -> np.ndarray:
    """(n,) float64 crisis probabilities from a binary booster; shape and range fail closed."""
    if len(X) == 0:
        return np.zeros(0)
    out = booster.predict(nx.dmatrix(X))
    if out.ndim != 1 or out.shape[0] != len(X):
        raise GateError(f"binary booster returned shape {out.shape}, not ({len(X)},)")
    out = out.astype(np.float64)
    if not (np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0):
        raise GateError("binary probabilities outside [0, 1] or non-finite")
    return out


def fit_binary_root(X_fit, y_fit, g_config: str):
    """The single D52 fit: standalone binary:logistic xgb.train on all original FIT rows (no weight/margin)."""
    params, rounds, four, diff = binary_params(g_config)
    y_bin = binary_target(y_fit)
    if len(y_bin) == 0:
        raise ValueError("empty fitting pool")
    booster = xgb.train(params, nx.dmatrix(X_fit, y_bin), num_boost_round=rounds)
    record = {"kind": "fresh_binary", "params": params, "four_class_params": four, "param_diff": diff,
              "rounds_total": rounds, "resolved_config": nx.resolved_config(booster),
              "base_score": nx.base_score(booster), "objective": objective_name(booster),
              "num_feature": num_feature(booster), "rows": int(len(y_bin)),
              "four_class_counts": [int(np.sum(np.asarray(y_fit) == k)) for k in range(4)],
              "binary_counts": [int(np.sum(y_bin == 0)), int(np.sum(y_bin == 1))],
              "four_class_labels_sha256": labels_sha(y_fit), "binary_labels_sha256": labels_sha(y_bin),
              "booster_sha256": nx.sha(booster)}
    return booster, record


def check_fit(booster, record: dict, g_config: str, X_fit, y_fit) -> np.ndarray:
    params, rounds, four, diff = binary_params(g_config)
    problems = []
    if record.get("params") != params or record.get("param_diff") != diff:
        problems.append("recorded params / diff differ from the approved binary params")
    if record.get("base_score") != "5E-1" or nx.base_score(booster) != "5E-1":
        problems.append(f"base_score {record.get('base_score')!r} != '5E-1'")
    if objective_name(booster) != "binary:logistic":
        problems.append(f"objective {objective_name(booster)!r} != 'binary:logistic'")
    if booster.num_boosted_rounds() != rounds or record.get("rounds_total") != rounds:
        problems.append(f"rounds {booster.num_boosted_rounds()} != configured {rounds}")
    if num_feature(booster) != str(N_FEATURES) or booster.num_features() != N_FEATURES:
        problems.append(f"recorded num_feature {num_feature(booster)!r} != '{N_FEATURES}'")
    if "sample_weight" in record or "base_margin" in record or nx.is_margin_marked(booster):
        problems.append("a sample weight or base margin is present")
    if record.get("rows") != len(y_fit) or record.get("binary_labels_sha256") != labels_sha(binary_target(y_fit)):
        problems.append("fit record rows / binary labels differ from the original FIT labels")
    if problems:
        raise GateError("binary root fit check failed: " + "; ".join(problems))
    return proba_binary(booster, X_fit)      # shape (n,) and range check on FIT


# ---------------------------------------------------------------- scoring

def mass(p4) -> np.ndarray:
    p4 = np.asarray(p4, dtype=np.float64)
    return (p4[:, 2] + p4[:, 3]) / p4.sum(axis=1)


def part_frame(X_full, y, g, m, horizon, p_original, p_binary=None) -> pd.DataFrame:
    """All keys; persistence from the full matrix column 87 (NaN = missing origin)."""
    frame = pd.DataFrame({"area": g, "target_month": m, "horizon": int(horizon), "truth": y,
                          "truth_crisis": binary_target(y),
                          "persistence_code": rr.persistence_codes(np.asarray(X_full)[:, PHASE_FULL_INDEX])})
    for k, lab in enumerate(LABELS):
        frame[f"p_original_{lab}"] = np.asarray(p_original, dtype=np.float64)[:, k]
    frame["s_original"] = mass(p_original)
    frame["y_original_argmax"] = fourclass.argmax_codes(p_original)
    frame["call_original_argmax"] = (frame["y_original_argmax"] >= 2).astype(int)
    frame["call_original_mass"] = (frame["s_original"] >= THRESHOLD).astype(int)
    if p_binary is not None:
        frame["p_binary"] = p_binary
        frame["call_binary"] = (frame["p_binary"] >= THRESHOLD).astype(int)
    return frame


def crisis_confusion(z, c) -> dict:
    z, c = np.asarray(z).astype(bool), np.asarray(c).astype(bool)
    return {k: int(v.sum()) for k, v in zip(CF, (z & c, ~z & c, z & ~c, ~z & ~c))}


def f1_exact(cf: dict) -> Fraction:
    d = 2 * cf["tp"] + cf["fp"] + cf["fn"]
    return Fraction(2 * cf["tp"], d) if d else Fraction(0)


def from_confusion(cf: dict) -> dict:
    n = sum(cf[c] for c in CF)
    f = f1_exact(cf)
    return {"n": n, **cf, "crisis_f1_exact": str(f), "crisis_f1": float(f),
            "crisis_call_share": (cf["tp"] + cf["fp"]) / n if n else None}


def brier(score, z) -> float | None:
    s = np.asarray(score, dtype=np.float64)
    return float(np.mean((s - np.asarray(z, dtype=np.float64)) ** 2)) if len(s) else None


def log_loss_binary(score, z) -> dict:
    """Binary log loss with clip [eps, 1-eps] applied to the log loss only; clipped counts disclosed."""
    s = np.asarray(score, dtype=np.float64)
    if not len(s):
        return {"logloss_binary": None, "clipped_low": 0, "clipped_high": 0, "eps": EPS}
    low, high = int(np.sum(s < EPS)), int(np.sum(s > 1.0 - EPS))
    c = np.clip(s, EPS, 1.0 - EPS)
    zz = np.asarray(z, dtype=np.float64)
    ll = float(np.mean(-(zz * np.log(c) + (1.0 - zz) * np.log1p(-c))))
    return {"logloss_binary": ll, "clipped_low": low, "clipped_high": high, "eps": EPS}


def arm_block(frame: pd.DataFrame, arm: str) -> dict:
    z = frame["truth_crisis"].to_numpy()
    if arm == "persistence":
        call = frame["persistence_code"].to_numpy(float).astype(np.int64) >= 2
        out = from_confusion(crisis_confusion(z, call))
        out["crisis_brier"] = brier(call.astype(float), z)          # one-hot, no log loss
        return out
    score = frame["p_binary" if arm == "binary" else "s_original"].to_numpy(np.float64)
    out = from_confusion(crisis_confusion(z, frame[f"call_{arm}"].to_numpy()))
    out["crisis_brier"] = brier(score, z)
    out.update(log_loss_binary(score, z))
    if arm == "binary":
        out["macro_f1_fourclass"] = None
        out["macro_f1_fourclass_null_reason"] = "not applicable: binary arm has no four-class prediction"
    return out


def delta(a: dict, b: dict) -> dict:
    d = Fraction(a["crisis_f1_exact"]) - Fraction(b["crisis_f1_exact"])
    out = {"crisis_f1_delta_exact": str(d), "crisis_f1_delta": float(d),
           "crisis_call_share_delta": (a["crisis_call_share"] - b["crisis_call_share"]
                                       if a["crisis_call_share"] is not None and b["crisis_call_share"] is not None
                                       else None),
           "crisis_brier_delta": (a["crisis_brier"] - b["crisis_brier"]
                                  if a["crisis_brier"] is not None and b["crisis_brier"] is not None else None)}
    if a.get("logloss_binary") is not None and b.get("logloss_binary") is not None:
        out["logloss_binary_delta"] = a["logloss_binary"] - b["logloss_binary"]
    return out


def ranking(score, z) -> dict:
    z = np.asarray(z, dtype=int)
    if len(z) == 0 or z.min() == z.max():
        return {"eligible": False, "auc": None, "ap": None}
    s = np.asarray(score, dtype=np.float64)
    return {"eligible": True, "auc": float(roc_auc_score(z, s)), "ap": float(average_precision_score(z, s))}


def original_reference(frame: pd.DataFrame) -> dict:
    """Original-only four-class reference: macro-F1 and four-class log loss (cw helpers)."""
    t = frame["truth"].to_numpy()
    p4 = frame[[f"p_original_{lab}" for lab in LABELS]].to_numpy(np.float64)
    b = cw.block(t, frame["y_original_argmax"].to_numpy(), p4)
    return {"macro_f1_fourclass": b["macro_f1_fourclass"], "logloss_fourclass": b["logloss_fourclass"]}


def score_part(frame: pd.DataFrame) -> dict:
    """All keys: model arms separately; matched exact-origin keys: plus persistence."""
    if len(frame) == 0:
        return {"n": 0, "status": "no_data"}
    k = np.isfinite(frame["persistence_code"].to_numpy(float))
    out = {"n_all": int(len(frame)), "excluded_missing_origin": int((~k).sum()),
           "all_keys": {a: arm_block(frame, a) for a in MODEL_ARMS},
           "all_keys_original_reference": original_reference(frame)}
    out["n"] = int(k.sum())
    if not k.any():
        out["status"] = "no_matched_keys"
        return out
    f = frame[k]
    out["crisis_prevalence"] = float(f["truth_crisis"].mean())
    for a in ARMS:
        out[a] = arm_block(f, a)
    out["original_reference"] = original_reference(f)
    for a, b in ALL_PAIRS:
        out[f"{a}_minus_{b}"] = delta(out[a], out[b])
    z = f["truth_crisis"].to_numpy()
    out["ranking"] = {"binary": ranking(f["p_binary"], z), "original": ranking(f["s_original"], z)}
    return out


def cell_auc(score, truth_bool) -> tuple:
    z = np.asarray(truth_bool, bool)
    n, P = len(z), int(z.sum())
    if n == 0:
        return None, "empty"
    if P == 0:
        return None, "no_positive"
    if P == n:
        return None, "no_negative"
    s = np.asarray(score, dtype=np.float64)
    if not np.isfinite(s).all():
        raise GateError("non-finite crisis score in a D50 cell")
    return float(roc_auc_score(z.astype(int), s)), None


def d50_cells(frame: pd.DataFrame, scores=SCORES) -> dict:
    """Per exact origin phase (persistence_code 0..3) on matched keys: n, P, N, AUC per score, null reasons."""
    k = np.isfinite(frame["persistence_code"].to_numpy(float))
    f = frame[k]
    pc = f["persistence_code"].to_numpy(float)
    z = f["truth_crisis"].to_numpy().astype(bool)
    col = {"binary": "p_binary", "original": "s_original"}
    cells = {}
    for c in CODES:
        m = pc == c
        P = int(z[m].sum())
        cell = {"origin_phase": c + 1, "n": int(m.sum()), "P": P, "N": int(m.sum()) - P,
                "auc": {}, "auc_null_reason": {}}
        for s in scores:
            cell["auc"][s], cell["auc_null_reason"][s] = cell_auc(f[col[s]].to_numpy()[m], z[m])
        cells[str(c)] = cell
    return cells


def consistency(name: str, frame: pd.DataFrame, d49: dict, d50: dict) -> dict:
    """Pre-fit: original arm on matched E3 keys must equal D49 confusions / counts and D50 original cells."""
    cells = d50_cells(frame, ("original",))
    r49, r50 = d49["per_root"][name], d50["per_root"][name]
    k = np.isfinite(frame["persistence_code"].to_numpy(float))
    problems = []
    if int(k.sum()) != r49["n"] or int((~k).sum()) != r49["excluded_missing_origin"]:
        problems.append(f"matched n/excluded {int(k.sum())}/{int((~k).sum())} != D49 "
                        f"{r49['n']}/{r49['excluded_missing_origin']}")
    z = frame["truth_crisis"].to_numpy()[k]
    orig = crisis_confusion(z, frame["call_original_argmax"].to_numpy()[k])
    pers = crisis_confusion(z, frame["persistence_code"].to_numpy(float)[k].astype(np.int64) >= 2)
    if any(orig[c] != r49["arms"]["original"]["argmax"][c] for c in CF):
        problems.append(f"original argmax confusion {orig} != D49")
    if any(pers[c] != r49["persistence"][c] for c in CF):
        problems.append(f"persistence confusion {pers} != D49")
    for c in map(str, CODES):
        mine, ref = cells[c], r50["cells"][c]
        if (mine["n"], mine["P"], mine["N"]) != (ref["n"], ref["P"], ref["N"]):
            problems.append(f"D50 cell {c} n/P/N differ")
        a, b = mine["auc"]["original"], ref["auc"]["original"]
        if mine["auc_null_reason"]["original"] != ref["auc_null_reason"]["original"]:
            problems.append(f"D50 cell {c} null reason differs")
        elif (a is None) != (b is None) or (a is not None and abs(a - b) > AUC_TOL):
            problems.append(f"D50 cell {c} original AUC {a} != {b}")
    if problems:
        raise GateError(f"{name}: original-arm D49/D50 consistency failed: " + "; ".join(problems))
    return {"n": int(k.sum()), "excluded_missing_origin": int((~k).sum()), "original_confusion": orig,
            "persistence_confusion": pers, "d50_cells_equal": True, "auc_tolerance": AUC_TOL}


def _sum_cf(blocks: list) -> dict:
    return from_confusion({c: int(sum(b[c] for b in blocks)) for c in CF})


def aggregate(per_root: dict, select) -> dict:
    """Per H: pooled (summed confusions) and mean-fold separate; fold wins; AUC/AP and D50 mean-fold only."""
    out = {}
    for part in PARTS:
        sel = [r["scores"][part] for r in per_root.values() if select(r)]
        have = [s for s in sel if s.get("n", 0)]
        cov = {"excluded_missing_origin_pooled": int(sum(s.get("excluded_missing_origin", 0) for s in sel)),
               "n_all_pooled": int(sum(s.get("n_all", 0) for s in sel))}
        if not have:
            out[part] = {"folds_with_data": 0, "status": "no_data", "coverage": cov}
            continue
        pooled = {a: _sum_cf([h[a] for h in have]) for a in ARMS}
        res = {"role": ROLE[part], "coverage": cov, "folds_with_data": len(have),
               "rows": int(sum(h["n"] for h in have)),
               "pooled_crisis_confusion": pooled,
               "pooled_deltas": {f"{a}_minus_{b}": {"crisis_f1_delta": float(Fraction(pooled[a]["crisis_f1_exact"])
                                                                            - Fraction(pooled[b]["crisis_f1_exact"]))}
                                 for a, b in ALL_PAIRS},
               "all_keys_pooled": {a: _sum_cf([s["all_keys"][a] for s in sel if s.get("n_all", 0)])
                                   for a in MODEL_ARMS},
               "mean_fold": {}, "mean_fold_deltas": {}, "fold_wins": {}, "ranking_mean_fold": {}}
        for a in ARMS:
            res["mean_fold"][a] = {m: (float(np.mean([h[a][m] for h in have]))
                                       if all(h[a].get(m) is not None for h in have) else None)
                                   for m in ("crisis_f1", "crisis_call_share", "crisis_brier", "logloss_binary")}
        res["mean_fold"]["original_reference"] = {
            m: float(np.mean([h["original_reference"][m] for h in have]))
            for m in ("macro_f1_fourclass", "logloss_fourclass")}
        for a, b in ALL_PAIRS:
            d = [Fraction(h[f"{a}_minus_{b}"]["crisis_f1_delta_exact"]) for h in have]
            res["mean_fold_deltas"][f"{a}_minus_{b}"] = {"crisis_f1_delta": float(sum(d) / len(d))}
            res["fold_wins"][f"{a}_minus_{b}"] = {"wins": sum(x > 0 for x in d), "ties": sum(x == 0 for x in d),
                                                  "losses": sum(x < 0 for x in d), "folds": len(d)}
        for s in SCORES:
            el = [h["ranking"][s] for h in have if h["ranking"][s]["eligible"]]
            res["ranking_mean_fold"][s] = {"eligible_folds": len(el),
                                           "auc": float(np.mean([e["auc"] for e in el])) if el else None,
                                           "ap": float(np.mean([e["ap"] for e in el])) if el else None}
        out[part] = res
    roots = [r for r in per_root.values() if select(r)]
    cells = {}
    for c in map(str, CODES):
        entry = {"origin_phase": int(c) + 1, "n_folds": len(roots), "mean_fold_auc": {}, "n_valid": {}}
        for s in SCORES:
            v = [r["d50_cells"][c]["auc"][s] for r in roots if r["d50_cells"][c]["auc"][s] is not None]
            entry["n_valid"][s] = f"{len(v)}/{len(roots)}"
            entry["mean_fold_auc"][s] = float(np.mean(v)) if v else None
        entry["supports"] = [{"target_month": r["target_month"], "n": r["d50_cells"][c]["n"],
                              "P": r["d50_cells"][c]["P"], "N": r["d50_cells"][c]["N"]}
                             for r in sorted(roots, key=lambda r: r["target_month"])]
        cells[c] = entry
    out["d50_cells_valid_fold_mean"] = cells
    return out


# ---------------------------------------------------------------- per root

def write_rows(path: Path, frame: pd.DataFrame) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, float_format="%.17g")


def gate_one(run, stage, name, pair, d49, d50) -> tuple:
    """Replay gate, G lock, D49/D50 consistency and FIT-key sha for one root; no fit.

    Returns (gate, data, replay, root_ubj, root, cons, fit_keys_sha); data is None when the replay gate or the
    G lock did not pass. A consistency or FIT-key mismatch raises."""
    root = json.loads((stage / "roots" / name / "root.json").read_text(encoding="utf-8"))
    gate, data, replay, root_ubj = rr.gate_root(run, stage, name, pair, root)
    h = int(root["horizon"])
    gate["checks"]["g_config_locked"] = {"mismatches": int(G_BY_H.get(h) != root["g_config"]), "n": 1}
    gate["passed"] = gate["passed"] and G_BY_H.get(h) == root["g_config"]
    if not gate["passed"]:
        return gate, None, None, None, root, None, None
    X_e3, y_e3, g_e3, m_e3 = data["E3"]
    cons = consistency(name, part_frame(X_e3, y_e3, g_e3, m_e3, h, replay["E3"]), d49, d50)
    gate["checks"]["d49_d50_consistency"] = {"mismatches": 0, "n": 1}
    _, _, gfit, mfit = data["FIT"]
    fit_keys_sha = nx.keys_sha(gfit, mfit)
    if fit_keys_sha != root.get("fitting_keys_sha256"):
        raise GateError("FIT keys differ from the original FIT keys")
    gate["checks"]["fit_keys_sha"] = {"mismatches": 0, "n": 1}
    return gate, data, replay, root_ubj, root, cons, fit_keys_sha


def fit_one(out, name, root, data, replay, root_ubj, cons, fit_keys_sha, gate, producer_rev, base, defaults):
    """Pass 2 for one already-gated root: the binary fit, check, save/reload, rows and dev_baselines check."""
    h = int(root["horizon"])
    g_config = root["g_config"]
    X_fit, y_fit, gfit, mfit = data["FIT"]
    check_defaults(defaults)
    booster, record = fit_binary_root(X_fit, y_fit, g_config)
    p_fit = check_fit(booster, record, g_config, X_fit, y_fit)
    rdir = out / name
    rdir.mkdir()
    (rdir / "binary_root.ubj").write_bytes(nx.raw(booster))
    frozen = nx.from_raw((rdir / "binary_root.ubj").read_bytes())
    if num_feature(frozen) != str(N_FEATURES) or nx.base_score(frozen) != "5E-1":
        raise GateError("reloaded binary root does not record 162 features / base_score 5E-1")
    original = nx.from_raw(root_ubj.read_bytes())          # sha gated in gate_root
    rows, reload_checks = {}, {}
    for part in PARTS:
        X, y, g, m = data[part]
        if part == "FIT":
            m = month_label(m)
        p_orig = nx.proba(original, X) if part == "FIT" else replay[part]
        p_bin = proba_binary(frozen, X)
        live = p_fit if part == "FIT" else proba_binary(booster, X)
        mism = int(np.sum(live != p_bin))
        reload_checks[part] = {"mismatches": mism, "n": int(p_bin.size)}
        if mism:
            raise GateError(f"reloaded binary root does not reproduce its {part} probabilities ({mism} rows)")
        rows[part] = part_frame(X, y, g, m, h, p_orig, p_bin)
        write_rows(rdir / f"rows_{part}.csv.gz", rows[part])
    o_index = data["window"][2]
    meta = {"root": name, "horizon": h, "target_month": root["target_month"], "g_config": g_config,
            "origin_month": month_label(np.array([o_index]))[0], "fit_rule": FIT_RULE,
            "fitting_rows": int(len(gfit)), "fitting_keys_sha256": fit_keys_sha,
            "fitting_four_class_labels_sha256": labels_sha(y_fit),
            "fitting_binary_labels_sha256": labels_sha(binary_target(y_fit)),
            "prediction_shape_fit": list(p_fit.shape), "original_root_source": str(root_ubj),
            "original_root_sha256": root["root_booster_sha256"],
            "binary_root_sha256": rid.file_sha256(rdir / "binary_root.ubj"),
            "num_feature": num_feature(frozen), "reload_exact": reload_checks, "fit_record": record,
            "producer_rev": producer_rev, "d49_d50_consistency": cons}
    rid.write_json_atomic(rdir / "binary_root.json", meta)
    dc = rr.dev_baseline_check(rows["E3"], base, h)
    if dc["status"] == "checked" and (dc["joined"] != dc["e3_rows"] or dc["persistence_mismatches"]
                                      or dc["truth_mismatches"]):
        gate = {**gate, "passed": False, "error": f"dev_baselines cross-check failed: {dc}"}
    return gate, rows, meta | {"dev_baselines_check": dc, "d50_cells": d50_cells(rows["E3"])}


def _stop(out, gates, phase, name, gate, roots, done, extra) -> int:
    rid.write_json_atomic(out / "gate.json", {"passed": False, "phase": phase, "roots": gates, "rule": RULE})
    rid.write_json_atomic(out / "failure.json", {
        "status": "stopped", "phase": phase, "failed_root": name, "error": gate.get("error", "gate mismatches"),
        **extra, "not_attempted": [r for r in roots if r not in done]})
    return 2


def run_all(run, stage, out, roots, cands, producer_rev, base, d49, d50) -> tuple[int, dict, dict]:
    """Pass 1 gates every root (zero fits); pass 2 fits only after all passed. First failure stops."""
    defaults = plan_defaults()
    pairs = {name: {cands[c]["e1"]: c for c in cands if cands[c]["root"] == name} for name in roots}
    gates, fit_sha, per_root = {}, {}, {}
    for name in roots:                                            # pass 1: gate + consistency, no fit
        try:
            gate, data, *_rest, sha = gate_one(run, stage, name, pairs[name], d49, d50)
            del data, _rest                                       # matrices are not kept across roots
            fit_sha[name] = sha
        except Exception as exc:   # noqa: BLE001 - any failure stops the run with evidence
            gate = {"root": name, "passed": False, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()}
        gates[name] = {"gate_pass": gate}
        print(f"{name}: gate {'passed' if gate['passed'] else 'FAILED'}", flush=True)
        if not gate["passed"]:
            return _stop(out, gates, "gate_pass", name, gate, roots, gates,
                         {"gated_passed_roots": [r for r in gates if r != name], "fitted_roots": []}), gates, per_root
    for name in roots:                                            # pass 2: re-gate, then the single fit
        try:
            gate, data, replay, root_ubj, root, cons, sha = gate_one(run, stage, name, pairs[name], d49, d50)
            if not gate["passed"] or sha != fit_sha[name]:
                gate = {**gate, "passed": False,
                        "error": "pass-2 re-gate failed or FIT-key sha differs from pass 1"}
            else:
                gate, rows, meta = fit_one(out, name, root, data, replay, root_ubj, cons, sha, gate,
                                           producer_rev, base, defaults)
                if gate["passed"]:
                    per_root[name] = {**meta, "scores": {p: score_part(rows[p]) for p in PARTS}}
            del data
        except Exception as exc:   # noqa: BLE001
            gate = {"root": name, "passed": False, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()}
        gates[name]["fit_pass"] = gate
        print(f"{name}: fit {'passed' if gate['passed'] else 'FAILED'}", flush=True)
        if not gate["passed"]:
            done = [r for r in roots if "fit_pass" in gates[r]]
            return _stop(out, gates, "fit_pass", name, gate, roots, done,
                         {"gated_passed_roots": list(roots), "completed_roots": sorted(per_root)}), gates, per_root
    check_defaults(defaults)
    rid.write_json_atomic(out / "gate.json", {"passed": True, "roots": gates, "rule": RULE})
    return 0, gates, per_root


def check_inventory(roots, d49: dict, d50: dict) -> None:
    mine = set(roots)
    if len(mine) != 21 or mine != set(d49["per_root"]) or mine != set(d50["per_root"]):
        raise GateError("accepted 21-root inventory differs from the D49 / D50 per_root inventories")


# ---------------------------------------------------------------- identity

def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def script_identity() -> dict:
    """Fail closed unless the script bytes are exactly the committed HEAD blob and the file is clean."""
    rel = SCRIPT.relative_to(REPO).as_posix()
    blob = _git("hash-object", rel)
    try:
        head_blob = _git("rev-parse", f"HEAD:{rel}")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"D52 script {rel} is not committed at HEAD") from exc
    dirty = _git("status", "--porcelain", "--", rel)
    if blob != head_blob or dirty:
        raise RuntimeError(f"D52 script {rel} differs from HEAD (blob {blob} vs {head_blob}; status {dirty!r})")
    return {"path": rel, "git_blob": blob, "sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest()}


# ---------------------------------------------------------------- selftest (synthetic fits only)

def fit_call_sites() -> list:
    """Functions calling ``xgb.train`` or ``nx.fit_global``/``nx.continue_booster`` (AST, not text)."""
    tree, sites = ast.parse(SCRIPT.read_text(encoding="utf-8")), []
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and (node.func.value.id, node.func.attr) in
                        {("xgb", "train"), ("nx", "fit_global"), ("nx", "continue_booster")}):
                    sites.append(fn.name)
    return sites


def selftest() -> int:
    rng = np.random.default_rng(0)
    assert fit_call_sites() == ["fit_binary_root"], fit_call_sites()
    # (1) parameter diff exactly as approved, on every locked G
    for g in sorted(set(G_BY_H.values())):
        params, rounds, four, diff = binary_params(g)
        assert diff == {"replaced": {"objective": ["multi:softprob", "binary:logistic"]},
                        "removed": {"multi_strategy": four["multi_strategy"], "num_class": 4},
                        "added": {"base_score": 0.5}}, diff
        assert rounds == plan.G_CONFIGS[g]["rounds"] and "num_class" not in params
        assert {k: v for k, v in params.items() if k not in ("objective", "base_score")} == \
            {k: v for k, v in four.items() if k not in ("objective", "num_class", "multi_strategy")}
    # (2) target mapping code >= 2 -> 1
    assert binary_target([0, 1, 2, 3]).tolist() == [0, 0, 1, 1]
    try:
        binary_target([4])
        raise AssertionError("label 4 accepted")
    except GateError:
        pass
    # (3) synthetic binary fit: shape (n,), range, base_score 5E-1, reload exact (not a real fit)
    Xs = rng.normal(size=(400, 162))
    Xs[rng.random(Xs.shape) < 0.1] = np.nan
    Xs[:3, 5] = np.inf
    ys = rng.integers(0, 4, size=400)
    plan.G_CONFIGS["_SELFTEST"] = {**plan.G_CONFIGS["G1"], "rounds": 3}
    try:
        booster, record = fit_binary_root(Xs, ys, "_SELFTEST")
        p_fit = check_fit(booster, record, "_SELFTEST", Xs, ys)
    finally:
        del plan.G_CONFIGS["_SELFTEST"]
    assert p_fit.shape == (400,) and 0 <= p_fit.min() and p_fit.max() <= 1
    assert record["base_score"] == "5E-1" and record["objective"] == "binary:logistic"
    assert record["num_feature"] == "162" and booster.num_boosted_rounds() == 3
    assert record["binary_counts"][1] == int(np.sum(ys >= 2))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "binary_root.ubj"
        path.write_bytes(nx.raw(booster))
        frozen = nx.from_raw(path.read_bytes())
    assert np.array_equal(proba_binary(frozen, Xs), p_fit) and nx.base_score(frozen) == "5E-1"
    # (4) log-loss clipping only; Brier and AUC unclipped
    z = np.array([1, 0, 1, 0, 1])
    s = np.array([0.0, 1.0, 1.0, 0.0, 0.5])
    ll = log_loss_binary(s, z)
    assert ll["clipped_low"] == 2 and ll["clipped_high"] == 2 and np.isfinite(ll["logloss_binary"])
    assert ll["logloss_binary"] > 10
    assert brier(s, z) == float(np.mean((s - z) ** 2)) == (1 + 1 + 0 + 0 + 0.25) / 5
    assert ranking(s, z)["auc"] == roc_auc_score(z, s)
    # (5) frame, scoring, cells, consistency, aggregate on synthetic data
    n = 60
    Xf = np.zeros((n, 162))
    Xf[:, 87] = np.tile([1.0, 2.0, 3.0, 4.0, 5.0], 12)
    Xf[:5, 87] = np.nan
    yv = np.tile([0, 1, 2, 3, 2, 0], 10)
    p = rng.dirichlet(np.ones(4), size=n)
    pb = rng.random(n)
    fr = part_frame(Xf, yv, np.arange(n), np.array(["2019-06"] * n), 4, p, pb)
    sc = score_part(fr)
    assert sc["n"] == 55 and sc["excluded_missing_origin"] == 5 and sc["n_all"] == 60
    assert "logloss_binary" not in sc["persistence"] and "logloss_binary" in sc["binary"]
    assert sc["binary"]["macro_f1_fourclass"] is None and "logloss_fourclass" in sc["original_reference"]
    assert sc["original_mass"]["crisis_brier"] == sc["original_argmax"]["crisis_brier"]
    assert set(sc) >= {f"{a}_minus_{b}" for a, b in ALL_PAIRS}
    cells = d50_cells(fr)
    assert sum(cells[c]["n"] for c in cells) == 55
    k = np.isfinite(fr["persistence_code"].to_numpy(float))
    zt = fr["truth_crisis"].to_numpy()[k]
    oc = crisis_confusion(zt, fr["call_original_argmax"].to_numpy()[k])
    pc = crisis_confusion(zt, fr["persistence_code"].to_numpy(float)[k] >= 2)
    ref50 = {"cells": {c: {"n": v["n"], "P": v["P"], "N": v["N"], "auc": {"original": v["auc"]["original"]},
                           "auc_null_reason": {"original": v["auc_null_reason"]["original"]}}
                       for c, v in cells.items()}}
    d49 = {"per_root": {"r": {"n": 55, "excluded_missing_origin": 5, "persistence": pc,
                              "arms": {"original": {"argmax": oc}}}}}
    d50 = {"per_root": {"r": ref50}}
    consistency("r", fr, d49, d50)
    bad = copy.deepcopy(d49)
    bad["per_root"]["r"]["arms"]["original"]["argmax"]["tp"] += 1
    try:
        consistency("r", fr, bad, d50)
        raise AssertionError("a D49 confusion mismatch was accepted")
    except GateError:
        pass
    agg = aggregate({"r": {"horizon": 4, "target_month": "2019-06", "scores": {q: sc for q in PARTS},
                           "d50_cells": cells}}, lambda r: True)
    assert agg["E3"]["rows"] == 55 and agg["E3"]["coverage"]["excluded_missing_origin_pooled"] == 5
    # (6) failed replay gate and failed consistency both stop before any fit
    calls = []
    real_gate, real_train = rr.gate_root, xgb.train

    def sentinel_train(*a, **kw):
        calls.append(1)
        raise AssertionError("xgb.train called after a failed gate")

    def failing_gate(run, stage, name, pair, root):
        return {"root": name, "checks": {"injected": {"mismatches": 1, "n": 1}}, "passed": False}, None, None, None

    def passing_gate(run, stage, name, pair, root):
        part = (Xf, yv, np.arange(n), np.array(["2019-06"] * n))
        return ({"root": name, "checks": {}, "passed": True},
                {"FIT": part, "C": part, "E3": part, "window": (0, 0, 0)}, {"C": p, "E3": p}, None)

    xgb.train = sentinel_train
    try:
        for gate_fn, refs in ((failing_gate, ({}, {})), (passing_gate, (bad, d50))):
            rr.gate_root = gate_fn
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                stage, out = tmp / "stage", tmp / "out"
                for name in ("r", "r2"):
                    (stage / "roots" / name).mkdir(parents=True)
                    (stage / "roots" / name / "root.json").write_text(
                        json.dumps({"horizon": 4, "g_config": "G1", "target_month": "2019-06"}), encoding="utf-8")
                out.mkdir()
                cands = {"c1": {"root": "r", "e1": "hard_f1"}, "c2": {"root": "r2", "e1": "hard_f1"}}
                code, gates, per_root = run_all(tmp, stage, out, ["r", "r2"], cands, "7b2bf6f", None, *refs)
                failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
                assert code == 2 and not calls and not per_root
                assert failure["failed_root"] == "r" and failure["not_attempted"] == ["r2"]
                assert not (out / "r").exists()
                if gate_fn is passing_gate:
                    assert "consistency failed" in failure["error"], failure["error"]
        # (7) the first root passes gate + consistency, a LATER root fails its gate: still zero fits
        m_int = np.full(n, 113)
        part_int = (Xf, yv, np.arange(n), m_int)
        good49 = {"per_root": {"r": d49["per_root"]["r"]}}
        good50 = {"per_root": {"r": ref50}}

        def mixed_gate(run, stage, name, pair, root):
            if name == "r2":
                return failing_gate(run, stage, name, pair, root)
            return ({"root": name, "checks": {}, "passed": True},
                    {"FIT": part_int, "C": part_int, "E3": part_int, "window": (0, 0, 0)}, {"C": p, "E3": p}, None)

        rr.gate_root = mixed_gate
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            stage, out = tmp / "stage", tmp / "out"
            for name in ("r", "r2", "r3"):
                (stage / "roots" / name).mkdir(parents=True)
                (stage / "roots" / name / "root.json").write_text(json.dumps(
                    {"horizon": 4, "g_config": "G1", "target_month": "2019-06",
                     "fitting_keys_sha256": nx.keys_sha(np.arange(n), m_int)}), encoding="utf-8")
            out.mkdir()
            cands = {f"c{i}": {"root": r, "e1": "hard_f1"} for i, r in enumerate(("r", "r2", "r3"))}
            code, gates, per_root = run_all(tmp, stage, out, ["r", "r2", "r3"], cands, "7b2bf6f", None,
                                            good49, good50)
            failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
            assert code == 2 and not calls and not per_root, (code, calls)
            assert gates["r"]["gate_pass"]["passed"] and "fit_keys_sha" in gates["r"]["gate_pass"]["checks"]
            assert failure["phase"] == "gate_pass" and failure["failed_root"] == "r2"
            assert failure["gated_passed_roots"] == ["r"] and failure["not_attempted"] == ["r3"]
            assert failure["fitted_roots"] == []
            assert not (out / "r").exists() and not list(out.rglob("*.ubj"))
        # (8) all roots pass: every gate-pass call precedes the first fit, one fit per root, pass-2 re-gate
        order = []
        real_gate_one, real_fit_one = globals()["gate_one"], globals()["fit_one"]

        def rec_gate_one(run, stage, name, pair, d49_, d50_):
            order.append(("gate", name))
            return {"root": name, "checks": {}, "passed": True}, {}, None, None, {}, None, "sha-" + name

        def rec_fit_one(out_, name, root, data, replay, root_ubj, cons, sha, gate, *rest):
            assert sha == "sha-" + name
            order.append(("fit", name))
            return gate, {q: fr for q in PARTS}, {"horizon": 4, "target_month": "2019-06"}

        globals()["gate_one"], globals()["fit_one"] = rec_gate_one, rec_fit_one
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                code, gates, per_root = run_all(None, None, out, ["r", "r2", "r3"], cands, "7b2bf6f", None,
                                                None, None)
        finally:
            globals()["gate_one"], globals()["fit_one"] = real_gate_one, real_fit_one
        assert code == 0 and sorted(per_root) == ["r", "r2", "r3"]
        assert order[:3] == [("gate", r) for r in ("r", "r2", "r3")], order
        assert order[3:] == [x for r in ("r", "r2", "r3") for x in (("gate", r), ("fit", r))], order
        assert [x for x in order if x[0] == "fit"] == [("fit", r) for r in ("r", "r2", "r3")]
    finally:
        rr.gate_root, xgb.train = real_gate, real_train
    print("SELFTEST OK")
    return 0


# ---------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d34-run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--producer-rev", default="7b2bf6f")
    parser.add_argument("--selftest", action="store_true", help="synthetic checks only; no real data or fit")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if args.d34_run is None or args.out is None:
        parser.error("--d34-run and --out are required unless --selftest")
    run, out = args.d34_run.resolve(), args.out.resolve()
    if out.is_relative_to(run):
        raise ValueError("--out must not be inside the read-only D34 run")
    rid.refuse_existing(out, "D52 binary root")
    defaults = plan_defaults()
    script = script_identity()
    producer = rid.code_identity_at(args.producer_rev)
    if rid.code_identity() != rid.code_identity_at("HEAD"):
        raise RuntimeError("D52 must run from committed package code (working tree differs from HEAD)")
    roots, cands, ident = accept_mode(run, plan.E1PAIR, producer, args.producer_rev)
    stage = Path(ident["stage"])
    d49 = json.loads(D49_SUMMARY.read_text(encoding="utf-8"))
    d50 = json.loads(D50_SUMMARY.read_text(encoding="utf-8"))
    check_inventory(roots, d49, d50)
    base_path = run / "prepared" / "ledgers" / "dev_baselines.csv"   # dev targets only; never baselines.csv
    base = pd.read_csv(base_path)
    identity = {"stage": "d52_binary_root", "d34_run": str(run), "producer_rev": args.producer_rev,
                "producer_code": producer, "repo_head": _git("rev-parse", "HEAD"), "script": script,
                "package_code": rid.code_identity(), "runtime": rid.runtime_identity(), "max_month": "2020-12",
                "xgboost_version": xgb.__version__, "g_by_horizon": G_BY_H,
                "g_configs": {g: plan.G_CONFIGS[g] for g in sorted(set(G_BY_H.values()))},
                "binary_params": {g: {"params": binary_params(g)[0], "rounds": binary_params(g)[1],
                                      "diff": binary_params(g)[3]} for g in sorted(set(G_BY_H.values()))},
                "fit_rule": FIT_RULE, "threshold": THRESHOLD, "logloss_eps": EPS, "plan_defaults": defaults,
                "consistency_sources": {"d49": rid.file_sha256(D49_SUMMARY), "d50": rid.file_sha256(D50_SUMMARY)},
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
    code, gates, per_root = run_all(run, stage, out, roots, cands, args.producer_rev, base, d49, d50)
    if code:
        print("STOPPED at the first failure: no summary computed; see gate.json / failure.json")
        return code
    summary = {
        "per_root": per_root,
        "by_horizon": {f"H{h}": aggregate(per_root, lambda r, h=h: r["horizon"] == h) for h in HORIZONS},
        "interpretation": ("Primary: E3 crisis F1 on matched exact-origin keys per H. Contrast A (same .5 rule): "
                           "binary p >= .5 vs original normalised mass s >= .5. Contrast B (endpoint): binary vs "
                           "the pipeline four-class argmax -> code >= 2, and each vs persistence; reported "
                           "separately. Shared binary Brier and binary log loss on the score (binary p; original "
                           "s for both original arms); eps clip on log loss only, clipped counts disclosed; "
                           "persistence one-hot Brier only. Four-class macro-F1 / log loss are original-only "
                           "references (not applicable to the binary arm). All-key E3 per model arm only, with "
                           "missing-origin counts. FIT in-sample, C in-window interpolation. AUC/AP within-root "
                           "mean-fold; D50 cells valid-fold means with n_valid/7. Objective, per-round capacity "
                           "and Hessian scale change together (G was selected for four-class): no objective "
                           "causality. 21 overlapping, repeatedly developed folds: no significance test, "
                           "threshold or adoption rule."),
    }
    check_defaults(defaults)
    rid.write_json_atomic(out / "summary.json", summary)
    rid.write_json_atomic(out / "completion.json", {"status": "completed", "outputs": rid.output_hashes(out)})
    print(f"D52 binary root completed: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
