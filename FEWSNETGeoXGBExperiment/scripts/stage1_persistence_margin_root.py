#!/usr/bin/env python3
"""D38 / A12: fixed weak exact-origin persistence base margin for the 21 saved D34 roots.

python scripts/stage1_persistence_margin_root.py --d34-run D34_RUN --out NEW_DIR [--producer-rev 7b2bf6f]

Per root (sequential): the D37 rebuild (Parquet filter target_month <= 2020-12, fitting / S / C / E3
key, role, fitting-hash and window guards) and original-root replay GATE (stage1_recency_root.gate_root)
before any fit. Only then ONE fresh ``fit_global(X_fit, y_fit, G, base_margin=m_fit)``. Each row's
margin uses only its own exact-origin feature hist_phase_o00 (label month - H for fitting rows):
q = 0.5 * one_hot(k) + 0.5 * Uniform(4), m = 0.5 + log(q) - mean_class(log(q)) (float64, passed as
float32); a missing origin gets exactly 0.5 in every class (the default base score). Margins never
read labels. The anchored root is saved, reloaded from UBJ (marker checked) and scored with the same
per-row margins on E3 (primary) and C (diagnostic), next to the replayed original root, a prior-only
control (q itself), a no-fit post-hoc control (p_original * q renormalised; missing origin keeps
p_original exactly) and exact-origin persistence on matched non-missing keys. No local trees, E4
weights, maps or Stage 2 inputs; no final-period ledger is read.
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
from scripts import stage1_recency_root as rr  # noqa: E402
from scripts import stage1_shallow_replay as sr  # noqa: E402
from scripts.stage1_rootconf_compare import accept_mode  # noqa: E402
from src.experiment import plan  # noqa: E402
from src.feature.fourclass_features import load_schema, month_label  # noqa: E402
from src.metrics import fourclass  # noqa: E402
from src.model import native_xgb as nx  # noqa: E402
from src.utils import run_identity as rid  # noqa: E402

LAMBDA = 0.5
DEFAULT_MARGIN = 0.5
LABELS = fourclass.CLASS_LABELS
PARTS = rr.PARTS
MODELS = ("original", "anchored", "prior_only", "posthoc")
GROUPS = rr.GROUPS
GateError = rr.GateError
RULE = {"marker": nx.MARGIN_MARKER, "lambda": LAMBDA, "known_q": {"own_class": 0.625, "other": 0.125},
        "missing_q": "uniform 0.25", "margin": "0.5 + log(q) - mean_class(log(q)), float64 -> float32",
        "missing_margin": "exactly float32 0.5", "class_order": list(LABELS),
        "origin_feature": "hist_phase_o00 (each row's own exact origin)"}


# ---------------------------------------------------------------- pure helpers (tested)

def persistence_prior(phase) -> np.ndarray:
    """Fixed lambda=0.5 prior q (n, 4) float64; missing origin -> uniform. Invalid phases raise."""
    codes = rr.persistence_codes(phase)
    known = np.isfinite(codes)
    q = np.full((len(codes), nx.N_CLASSES), 0.25)
    k = codes[known].astype(np.int64)
    q[known] = (1.0 - LAMBDA) / nx.N_CLASSES
    q[np.flatnonzero(known), k] += LAMBDA
    return q


def persistence_margin(phase) -> np.ndarray:
    """(n, 4) float32 centred log-prior margin around 0.5; missing origin rows exactly 0.5."""
    q = persistence_prior(phase)
    known = np.isfinite(rr.persistence_codes(phase))
    lq = np.log(q[known])
    m = np.full(q.shape, DEFAULT_MARGIN, dtype=np.float32)
    m[known] = (DEFAULT_MARGIN + lq - lq.mean(axis=1, keepdims=True)).astype(np.float32)
    return m


def posthoc(p_original, phase) -> np.ndarray:
    """No-fit control: known rows p*q / sum(p*q) (float64); missing-origin rows p_original exactly."""
    p = np.asarray(p_original, dtype=np.float64)
    q = persistence_prior(phase)
    known = np.isfinite(rr.persistence_codes(phase))
    out = p.copy()
    pq = p[known] * q[known]
    out[known] = pq / pq.sum(axis=1, keepdims=True)
    return out


def transition_groups(frame: pd.DataFrame, model: str) -> dict:
    """Post-hoc D36 layers for ``model`` vs the original root (no routing use)."""
    pers = frame["persistence_code"].to_numpy(float)
    t = rr.crisis(frame["truth"].to_numpy())
    grp = np.where(~np.isfinite(pers), "missing",
                   np.where(pers >= 2, "1", "0").astype(object) + np.where(t, "1", "0").astype(object))
    yo, ym = rr.crisis(frame["y_original"].to_numpy()), rr.crisis(frame[f"y_{model}"].to_numpy())
    out = {}
    for g in GROUPS:
        k = grp == g
        okm, oko = ym[k] == t[k], yo[k] == t[k]
        out[g] = {"n": int(k.sum()), "corrected": int(np.sum(okm & ~oko)), "spoiled": int(np.sum(~okm & oko)),
                  "tp_change": int(np.sum(ym[k] & t[k]) - np.sum(yo[k] & t[k])),
                  "fp_change": int(np.sum(ym[k] & ~t[k]) - np.sum(yo[k] & ~t[k]))}
    return out


def departure(frame: pd.DataFrame, model: str) -> dict:
    """Crisis disagreement of ``model`` with exact-origin persistence on matched keys; who is right."""
    pers = frame["persistence_code"].to_numpy(float)
    k = np.isfinite(pers)
    t = rr.crisis(frame["truth"].to_numpy()[k])
    yp, ym = pers[k] >= 2, rr.crisis(frame[f"y_{model}"].to_numpy()[k])
    d = yp != ym
    return {"matched": int(k.sum()), "disagree": int(d.sum()), "rate": float(d.mean()) if k.any() else None,
            "model_right": int(np.sum(d & (ym == t))), "persistence_right": int(np.sum(d & (yp == t))),
            "model_crisis_persistence_not": int(np.sum(d & ym)), "persistence_crisis_model_not": int(np.sum(d & yp))}


def score_frame(frame: pd.DataFrame) -> dict:
    """All keys: four models; matched non-missing-persistence keys: plus persistence."""
    if len(frame) == 0:
        return {"n": 0, "status": "no_data"}
    t = frame["truth"].to_numpy()
    prob = {m: frame[[f"p_{m}_{l}" for l in LABELS]].to_numpy(float) for m in MODELS}
    out = {"n": int(len(t)), "all": {m: rr.score_block(t, frame[f"y_{m}"].to_numpy(), prob[m]) for m in MODELS}}
    for m in MODELS[1:]:
        out["all"][f"{m}_minus_original"] = rr.delta(out["all"][m], out["all"]["original"])
    out["all"]["anchored_minus_posthoc"] = rr.delta(out["all"]["anchored"], out["all"]["posthoc"])
    out["all"]["anchored_minus_prior_only"] = rr.delta(out["all"]["anchored"], out["all"]["prior_only"])
    k = np.isfinite(frame["persistence_code"].to_numpy(float))
    matched = {"n": int(k.sum()), "coverage": float(k.mean())}
    if k.any():
        pers = frame["persistence_code"].to_numpy(float)[k].astype(np.int64)
        for m in MODELS:
            matched[m] = rr.score_block(t[k], frame[f"y_{m}"].to_numpy()[k], prob[m][k])
        matched["persistence"] = rr.score_block(t[k], pers, np.eye(nx.N_CLASSES)[pers])
        for m in MODELS:
            matched[f"{m}_minus_persistence"] = rr.delta(matched[m], matched["persistence"])
        matched["anchored_minus_original"] = rr.delta(matched["anchored"], matched["original"])
        matched["anchored_minus_posthoc"] = rr.delta(matched["anchored"], matched["posthoc"])
        matched["prior_only_argmax_equals_persistence"] = bool(
            np.array_equal(frame["y_prior_only"].to_numpy()[k], pers))
    out["matched_persistence"] = matched
    out["departure_from_persistence"] = {m: departure(frame, m) for m in ("original", "anchored", "posthoc")}
    out["transition_groups_post_hoc"] = {m: transition_groups(frame, m) for m in ("anchored", "posthoc")}
    return out


def _fdelta(a: dict, b: dict) -> dict:
    d = Fraction(a["crisis_f1_exact"]) - Fraction(b["crisis_f1_exact"])
    return {"crisis_f1_delta_exact": str(d), "crisis_f1_delta": float(d)}


def aggregate(per_root: dict, select) -> dict:
    """Mean-fold deltas and pooled confusions kept separate; 21 folds are not independent."""
    out = {}
    for part in PARTS:
        have = [r["scores"][part] for r in per_root.values() if select(r) and r["scores"][part].get("n", 0)]
        if not have:
            out[part] = {"folds_with_data": 0, "status": "no_data"}
            continue
        allp = rr.pooled({m: np.sum([h["all"][m]["confusion_fourclass"] for h in have], axis=0) for m in MODELS})
        for m in MODELS[1:]:
            allp[f"{m}_minus_original"] = _fdelta(allp[m], allp["original"])
        allp["anchored_minus_posthoc"] = _fdelta(allp["anchored"], allp["posthoc"])
        res = {"folds_with_data": len(have), "rows": int(sum(h["n"] for h in have)), "pooled_all": allp,
               "mean_fold_all": {f"{m}_minus_original": {
                   "crisis_f1": float(np.mean([h["all"][f"{m}_minus_original"]["crisis_f1_delta"] for h in have])),
                   "crisis_brier": float(np.mean([h["all"][f"{m}_minus_original"]["crisis_brier_delta"]
                                                  for h in have]))} for m in MODELS[1:]},
               "mean_crisis_brier_all": {m: float(np.mean([h["all"][m]["crisis_brier"] for h in have]))
                                         for m in MODELS}}
        mh = [h["matched_persistence"] for h in have if h["matched_persistence"]["n"]]
        if mh:
            mp = rr.pooled({m: np.sum([h[m]["confusion_fourclass"] for h in mh], axis=0)
                            for m in MODELS + ("persistence",)})
            for m in MODELS:
                mp[f"{m}_minus_persistence"] = _fdelta(mp[m], mp["persistence"])
            res["pooled_matched_persistence"] = mp | {
                "rows": int(sum(h["n"] for h in mh)),
                "mean_fold": {f"{m}_minus_persistence": float(np.mean(
                    [h[f"{m}_minus_persistence"]["crisis_f1_delta"] for h in mh])) for m in MODELS}}
        res["departure_from_persistence"] = {
            m: {k: int(sum(h["departure_from_persistence"][m][k] for h in have))
                for k in ("matched", "disagree", "model_right", "persistence_right",
                          "model_crisis_persistence_not", "persistence_crisis_model_not")}
            for m in ("original", "anchored", "posthoc")}
        res["transition_groups_post_hoc"] = {
            m: {g: {k: int(sum(h["transition_groups_post_hoc"][m][g][k] for h in have))
                    for k in ("n", "corrected", "spoiled", "tp_change", "fp_change")} for g in GROUPS}
            for m in ("anchored", "posthoc")}
        out[part] = res
    return out


# ---------------------------------------------------------------- per root

def fit_anchored_root(data: dict, g_config: str, phase_col: int):
    """The single D38 fit: fresh G root on the original fitting rows with the fixed persistence margin."""
    X_fit, y_fit, _, _ = data["FIT"]
    m_fit = persistence_margin(X_fit[:, phase_col])
    booster, record = nx.fit_global(X_fit, y_fit, plan.G_CONFIGS[g_config], base_margin=m_fit)
    return booster, record, m_fit


def part_frame(X, y, g, m, horizon, p_original, frozen, phase_col) -> pd.DataFrame:
    phase = X[:, phase_col]
    probs = {"original": p_original,
             "anchored": nx.proba(frozen, X, base_margin=persistence_margin(phase)),
             "prior_only": persistence_prior(phase),
             "posthoc": posthoc(p_original, phase)}
    frame = pd.DataFrame({"area": g, "target_month": m, "horizon": int(horizon), "truth": y,
                          "origin_phase": phase, "persistence_code": rr.persistence_codes(phase)})
    for name, p in probs.items():
        frame[f"y_{name}"] = fourclass.argmax_codes(p) if name != "prior_only" else np.argmax(p, axis=1)
        for k, lab in enumerate(LABELS):
            frame[f"p_{name}_{lab}"] = p[:, k]
    return frame


def run_root(run, stage, out, name, pair, producer_rev, phase_col, base):
    root = json.loads((stage / "roots" / name / "root.json").read_text(encoding="utf-8"))
    gate, data, replay, root_ubj = rr.gate_root(run, stage, name, pair, root)
    if not gate["passed"]:
        return gate, None, None
    g_config = root["g_config"]
    booster, record, m_fit = fit_anchored_root(data, g_config, phase_col)
    rdir = out / name
    rdir.mkdir()
    (rdir / "anchored_root.ubj").write_bytes(nx.raw(booster))
    frozen = nx.from_raw((rdir / "anchored_root.ubj").read_bytes())
    if frozen.attr(nx.MARGIN_ATTR) != nx.MARGIN_MARKER:
        raise RuntimeError("reloaded anchored root lost its base-margin marker")
    X_fit, y_fit, gfit, mfit = data["FIT"]
    fit_codes = rr.persistence_codes(X_fit[:, phase_col])
    with gzip.open(rdir / "fitting_origin.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        pd.DataFrame({"area": gfit, "target_month": month_label(mfit), "class_code": y_fit,
                      "origin_code": fit_codes,
                      **{f"margin_{lab}": m_fit[:, k] for k, lab in enumerate(LABELS)}}).to_csv(
            handle, index=False, float_format="%.9g")
    o_index = data["window"][2]
    rows = {}
    for part in PARTS:
        X, y, g, m = data[part]
        rows[part] = part_frame(X, y, g, m, root["horizon"], replay[part], frozen, phase_col)
        with gzip.open(rdir / f"rows_{part}.csv.gz", "wt", encoding="utf-8", newline="") as handle:
            rows[part].to_csv(handle, index=False, float_format="%.17g")
    meta = {"root": name, "horizon": int(root["horizon"]), "target_month": root["target_month"],
            "g_config": g_config, "origin_month": month_label(np.array([o_index]))[0], "margin_rule": RULE,
            "fitting_rows": int(len(gfit)), "fitting_label_months": sorted(set(month_label(np.unique(mfit)))),
            "fitting_keys_sha256": nx.keys_sha(gfit, mfit),
            "origin_available_fraction": {"fitting": float(np.isfinite(fit_codes).mean()),
                                          **{p: float(np.isfinite(rows[p]["persistence_code"]).mean())
                                             for p in PARTS}},
            "original_root_source": str(root_ubj), "original_root_sha256": root["root_booster_sha256"],
            "anchored_root_sha256": rid.file_sha256(rdir / "anchored_root.ubj"),
            "anchored_root_marker": frozen.attr(nx.MARGIN_ATTR),
            "fit_record": record, "producer_rev": producer_rev}
    rid.write_json_atomic(rdir / "anchored_root.json", meta)
    dc = rr.dev_baseline_check(rows["E3"], base, int(root["horizon"]))
    if dc["status"] == "checked" and (dc["joined"] != dc["e3_rows"] or dc["persistence_mismatches"]
                                      or dc["truth_mismatches"]):
        gate = {**gate, "passed": False, "error": f"dev_baselines cross-check failed: {dc}"}
    return gate, rows, meta | {"dev_baselines_check": dc}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d34-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--producer-rev", default="7b2bf6f")
    args = parser.parse_args()
    run, out = args.d34_run.resolve(), args.out.resolve()
    if out.is_relative_to(run):
        raise ValueError("--out must not be inside the read-only D34 run")
    rid.refuse_existing(out, "D38 persistence-margin root")
    producer = rid.code_identity_at(args.producer_rev)
    if rid.code_identity() != rid.code_identity_at("HEAD"):
        raise RuntimeError("D38 must run from committed package code (working tree differs from HEAD)")
    script_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE, capture_output=True, text=True,
                                   check=True).stdout.strip()
    roots, cands, ident = accept_mode(run, plan.E1PAIR, producer, args.producer_rev)
    stage = Path(ident["stage"])
    phase_col = load_schema(sr.SCHEMA)["ordered_features"].index("hist_phase_o00")
    base_path = run / "prepared" / "ledgers" / "dev_baselines.csv"   # dev targets only; never baselines.csv
    base = pd.read_csv(base_path)
    out.mkdir(parents=True)
    identity = {"stage": "d38_persistence_margin_root", "d34_run": str(run), "producer_rev": args.producer_rev,
                "producer_code": producer, "script_commit": script_commit, "script_code": rid.code_identity(),
                "runtime": rid.runtime_identity(), "max_month": "2020-12", "margin_rule": RULE,
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
    rid.write_json_atomic(out / "gate.json", {"passed": passed, "rule": "exact equality, round-trip parse; a root failing a "
                                              "pre-fit gate is not fitted; the dev_baselines check runs after "
                                              "the fit, so its failure blocks the summary but leaves that "
                                              "root's files", "roots": gates})
    rid.write_json_atomic(out / "identity.json", identity)
    if not passed:
        print("GATE FAILED for at least one root: no summary computed; see gate.json")
        return 2
    summary = {
        "per_root": per_root,
        "by_horizon": {f"H{h}": aggregate(per_root, lambda r, h=h: r["horizon"] == h) for h in (4, 8, 12)},
        "by_target": {t: aggregate(per_root, lambda r, t=t: r["target_month"] == t) for t in plan.E1PAIR_TARGETS},
        "overall": aggregate(per_root, lambda r: True),
        "interpretation": ("Primary: same-key E3 for original / anchored / prior-only / post-hoc, and versus "
                           "exact-origin persistence on matched non-missing keys (coverage reported). C is "
                           "diagnostic only. prior-only argmax equals persistence on known-origin keys by "
                           "construction (mechanical check, not learning). anchored must beat post-hoc to "
                           "credit the training-time parameterization. F1 up with crisis Brier worse is a "
                           "decision trade-off only. mean_fold averages per-root deltas; pooled sums confusions. "
                           "Transition groups and departures are post-hoc layers, not routing rules. 21 "
                           "overlapping, repeatedly developed folds: no significance test; one fixed lambda."),
    }
    rid.write_json_atomic(out / "summary.json", summary)
    rid.write_json_atomic(out / "completion.json", {"status": "completed", "outputs": rid.output_hashes(out)})
    print(f"D38 persistence-margin root completed: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
