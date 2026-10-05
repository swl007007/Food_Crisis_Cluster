#!/usr/bin/env python3
"""D51 / A25: history + known-calendar (78-feature) root ablation for the 21 saved D34 roots (one-off research).

python .trellis/tasks/10-01-geoxgb-shared-parameter-design/research/d51_history_calendar_root.py \
    --d34-run D34_RUN --out C:\\Users\\swl00\\geoxgb_runs\\geoxgb-d51-history-calendar-root-20261002 \
    [--producer-rev 7b2bf6f]
python .../d51_history_calendar_root.py --selftest      # synthetic fits only, no real data

Per root (sequential, stop at the first failure): D34 pinned acceptance, then the D37 rebuild and
original-root replay GATE (stage1_recency_root.gate_root) on the FULL-162 matrices before any fit. Only then
the FIT/C/E3 matrices are projected (pure column take) onto the 78 schema columns (history_blocks +
known_calendar, global schema order) and ONE fresh ``fit_global(X_fit[:, IDX78], y_fit, plan.G_CONFIGS[G])``
is run on ALL original FIT rows, order and labels (including missing-origin rows): same G (H4 G1, H8 G4,
H12 G2), rounds, seed; no sample weight, no base margin. Fit checks fail closed (params/rounds =
booster_params(G), base score .5, four classes, recorded num_feature 78, no weight/margin block). The root
is saved, reloaded from UBJ (exact on projected FIT/C/E3) and scored next to the replayed original root
(full matrices) and exact-origin persistence (full-schema hist_phase_o00, index 87, never on the projected
matrix). Original-arm consistency with D49/D50 summaries is a gate. No local trees, maps, Stage 2/3 inputs.
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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
RESEARCH = SCRIPT.parent
PACKAGE = REPO / "FEWSNETGeoXGBExperiment"
sys.path.insert(0, str(PACKAGE))
from scripts import stage1_class_weight_root as cw  # noqa: E402
from scripts import stage1_recency_root as rr  # noqa: E402
from scripts import stage1_shallow_replay as sr  # noqa: E402
from scripts import stage1_stump_root as st  # noqa: E402  (arm-parameterised helpers only)
from scripts.stage1_rootconf_compare import accept_mode  # noqa: E402
from src.experiment import plan  # noqa: E402
from src.feature.fourclass_features import load_schema, month_label  # noqa: E402
from src.metrics import fourclass  # noqa: E402
from src.model import native_xgb as nx  # noqa: E402
from src.utils import run_identity as rid  # noqa: E402

G_BY_H = {4: "G1", 8: "G4", 12: "G2"}
HORIZONS = (4, 8, 12)
LABELS = fourclass.CLASS_LABELS
PARTS = ("FIT", "C", "E3")
ROLE = {"FIT": "in-sample fitting rows", "C": "in-window historical interpolation",
        "E3": "forward target month (primary on matched exact-origin keys)"}
ARMS = ("original", "h78")
ARM_DELTAS = (("h78", "original"),)
PERS_DELTAS = (("h78", "persistence"), ("original", "persistence"))
ALL_PAIRS = PERS_DELTAS + ARM_DELTAS
CODES = (0, 1, 2, 3)
CF = ("tp", "fp", "fn", "tn")
N_FEATURES = 78
PHASE_FULL_INDEX = 87
D49_SUMMARY = RESEARCH / "d49_summary.json"
D50_SUMMARY = RESEARCH / "d50_summary.json"
AUC_TOL = 1e-12
FIT_RULE = ("fresh root on ALL original rebuilt FIT rows, order and labels (incl. missing-origin rows), "
            "projected onto the 78 history_blocks + known_calendar columns in global schema order; "
            "no weights or margins; G/rounds/seed unchanged")
RULE = ("exact equality, round-trip parse; a root failing the pre-fit gate is not fitted; the first gate, "
        "projection, fit, reload, D49/D50 consistency, scoring or dev_baselines failure stops the run")
GateError = rr.GateError


# ---------------------------------------------------------------- feature projection

def feature_map(schema: dict) -> dict:
    """78 selected positions (sorted, global order) and the 84-name complement check."""
    ordered = list(schema["ordered_features"])
    names = [n for block in schema["history_blocks"].values() for n in block] + list(schema["known_calendar"])
    if len(set(names)) != len(names) or any(n not in ordered for n in names):
        raise GateError("history_blocks / known_calendar names are duplicated or not in ordered_features")
    idx = sorted(ordered.index(n) for n in names)
    if len(idx) != N_FEATURES:
        raise GateError(f"selected feature count {len(idx)} != {N_FEATURES}")
    removed_expected = (list(schema["static_sources"]) + list(schema["dynamic_sources_at_origin"])
                        + list(schema["legacy_covariate_derived"]))
    removed = [n for i, n in enumerate(ordered) if i not in set(idx)]
    if len(removed) != 84 or removed != removed_expected or set(removed) != set(removed_expected):
        raise GateError("removed names differ from static + dynamic_sources_at_origin + legacy_covariate_derived")
    phase = ordered.index("hist_phase_o00")
    if phase != PHASE_FULL_INDEX:
        raise GateError(f"hist_phase_o00 full-schema index {phase} != {PHASE_FULL_INDEX}")
    return {"selected_names": [ordered[i] for i in idx], "selected_index": idx,
            "index_map": {ordered[i]: {"full": i, "projected": j} for j, i in enumerate(idx)},
            "removed_names": removed, "removed_count": len(removed), "phase_full_index": phase,
            "phase_projected_index": idx.index(phase)}


def project(X, idx) -> tuple[np.ndarray, dict]:
    """Pure column take; gated by exact equality (equal_nan), dtype and row count."""
    X = np.asarray(X)
    P = X[:, idx]
    ok = (P.dtype == X.dtype and P.shape == (X.shape[0], len(idx))
          and np.array_equal(X[:, idx], P, equal_nan=True))
    for j, i in enumerate(idx):           # column-by-column equality against the full matrix
        ok = ok and np.array_equal(X[:, i], P[:, j], equal_nan=True)
    if not ok:
        raise GateError("projected matrix is not the exact selected-column take of the full matrix")
    return P, {"rows": int(P.shape[0]), "columns": int(P.shape[1]), "dtype": str(P.dtype), "equal": True}


# ---------------------------------------------------------------- config and fit checks

def plan_defaults() -> dict:
    return {"G_CONFIGS": copy.deepcopy(plan.G_CONFIGS), "XGB_BASE": dict(plan.XGB_BASE)}


def check_defaults(snapshot: dict) -> None:
    if plan_defaults() != snapshot:
        raise GateError("plan.G_CONFIGS or plan.XGB_BASE changed during the run")


def num_feature(booster) -> str:
    return json.loads(bytes(booster.save_raw("json")))["learner"]["learner_model_param"]["num_feature"]


def check_fit(booster, record: dict, g_config: dict, n_rows: int) -> None:
    params, rounds = nx.booster_params(g_config)
    problems = []
    if record.get("params") != params:
        problems.append("recorded params differ from booster_params(G)")
    if record.get("base_score") != "5E-1" or nx.base_score(booster) != "5E-1":
        problems.append(f"base_score {record.get('base_score')!r} != '5E-1'")
    if nx.num_class(booster) != nx.N_CLASSES:
        problems.append("class axis is not four")
    if booster.num_boosted_rounds() != rounds or record.get("rounds_total") != rounds:
        problems.append(f"rounds {booster.num_boosted_rounds()} != configured {rounds}")
    if num_feature(booster) != str(N_FEATURES) or booster.num_features() != N_FEATURES:
        problems.append(f"recorded num_feature {num_feature(booster)!r} != '{N_FEATURES}'")
    if "sample_weight" in record or "base_margin" in record or nx.is_margin_marked(booster):
        problems.append("a sample weight or base margin is present")
    if record.get("rows") != n_rows:
        problems.append("fit record rows differ from the original FIT rows")
    if problems:
        raise GateError("78-feature root fit check failed: " + "; ".join(problems))


def fit_h78_root(X_fit_full, y_fit, idx, g_config: str):
    """The single D51 fit: fresh root on all original FIT rows, 78 projected columns (no weight, no margin)."""
    return nx.fit_global(np.asarray(X_fit_full)[:, idx], y_fit, plan.G_CONFIGS[g_config])


# ---------------------------------------------------------------- scoring

def part_frame(X_full, y, g, m, horizon, p_original, p_h78) -> pd.DataFrame:
    """All keys; persistence from the FULL matrix column 87 (NaN = missing origin)."""
    frame = pd.DataFrame({"area": g, "target_month": m, "horizon": int(horizon), "truth": y,
                          "persistence_code": rr.persistence_codes(np.asarray(X_full)[:, PHASE_FULL_INDEX])})
    for arm, p in (("original", p_original), ("h78", p_h78)):
        frame[f"y_{arm}"] = fourclass.argmax_codes(p)
        for k, lab in enumerate(LABELS):
            frame[f"p_{arm}_{lab}"] = p[:, k]
    return frame


def crisis_confusion(truth, pred) -> dict:
    z, c = rr.crisis(truth).astype(bool), rr.crisis(pred).astype(bool)
    return {k: int(v.sum()) for k, v in zip(CF, (z & c, ~z & c, z & ~c, ~z & ~c))}


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


def d50_cells(frame: pd.DataFrame) -> dict:
    """Per exact origin phase (persistence_code 0..3) on matched keys: n, P, N, AUC per arm, null reasons."""
    k = np.isfinite(frame["persistence_code"].to_numpy(float))
    f = frame[k]
    pc = f["persistence_code"].to_numpy(float)
    z = rr.crisis(f["truth"].to_numpy()).astype(bool)
    score = {}
    for arm in ARMS:
        P4 = f[[f"p_{arm}_{lab}" for lab in LABELS]].to_numpy(np.float64)
        score[arm] = (P4[:, 2] + P4[:, 3]) / P4.sum(axis=1)
    cells = {}
    for c in CODES:
        m = pc == c
        P = int(z[m].sum())
        cell = {"origin_phase": c + 1, "n": int(m.sum()), "P": P, "N": int(m.sum()) - P,
                "auc": {}, "auc_null_reason": {}}
        for arm in ARMS:
            cell["auc"][arm], cell["auc_null_reason"][arm] = cell_auc(score[arm][m], z[m])
        cells[str(c)] = cell
    return cells


def score_part(frame: pd.DataFrame) -> dict:
    """All keys: original and h78 separately; matched exact-origin keys: plus persistence (no log loss)."""
    if len(frame) == 0:
        return {"n": 0, "status": "no_data"}
    t = frame["truth"].to_numpy()
    prob = {a: frame[[f"p_{a}_{lab}" for lab in LABELS]].to_numpy(np.float64) for a in ARMS}
    k = np.isfinite(frame["persistence_code"].to_numpy(float))
    out = {"n_all": int(len(t)), "excluded_missing_origin": int((~k).sum()),
           "all_keys": {a: st._share(cw.block(t, frame[f"y_{a}"].to_numpy(), prob[a])) for a in ARMS}}
    out["n"] = int(k.sum())
    if not k.any():
        out["status"] = "no_matched_keys"
        return out
    pers = frame["persistence_code"].to_numpy(float)[k].astype(np.int64)
    out["crisis_prevalence"] = float(rr.crisis(t[k]).mean())
    for a in ARMS:
        out[a] = st._share(cw.block(t[k], frame[f"y_{a}"].to_numpy()[k], prob[a][k]))
    out["persistence"] = st._share(cw.block(t[k], pers, np.eye(nx.N_CLASSES)[pers], logloss=False))
    for a, b in ALL_PAIRS:
        out[f"{a}_minus_{b}"] = cw.delta(out[a], out[b])
    out["ranking"] = {a: cw.ranking(t[k], prob[a][k]) for a in ARMS}
    return out


def consistency(name: str, frame: pd.DataFrame, cells: dict, d49: dict, d50: dict) -> dict:
    """Original arm on matched E3 keys must equal the D49 confusions / counts and the D50 original cells."""
    r49, r50 = d49["per_root"][name], d50["per_root"][name]
    k = np.isfinite(frame["persistence_code"].to_numpy(float))
    problems = []
    if int(k.sum()) != r49["n"] or int((~k).sum()) != r49["excluded_missing_origin"]:
        problems.append(f"matched n/excluded {int(k.sum())}/{int((~k).sum())} != D49 "
                        f"{r49['n']}/{r49['excluded_missing_origin']}")
    t = frame["truth"].to_numpy()[k]
    orig = crisis_confusion(t, frame["y_original"].to_numpy()[k])
    pers = crisis_confusion(t, frame["persistence_code"].to_numpy(float)[k].astype(np.int64))
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


def aggregate(per_root: dict, select) -> dict:
    """Per H: pooled (summed confusions) and mean-fold separate; AUC/AP mean-fold only; D50 valid-fold means."""
    out = {}
    for part in PARTS:
        sel = [r["scores"][part] for r in per_root.values() if select(r)]
        have = [s for s in sel if s.get("n", 0)]
        miss = {"excluded_missing_origin_pooled": int(sum(s.get("excluded_missing_origin", 0) for s in sel)),
                "n_all_pooled": int(sum(s.get("n_all", 0) for s in sel))}
        if not have:
            out[part] = {"folds_with_data": 0, "status": "no_data", "coverage": miss}
            continue
        rows = int(sum(h["n"] for h in have))
        models = ARMS + ("persistence",)
        allk = [{a: s["all_keys"][a] for a in ARMS} for s in sel if s.get("n_all", 0)]
        res = {"role": ROLE[part], "coverage": miss, "folds_with_data": len(have), "rows": rows,
               "crisis_prevalence_pooled": float(sum(h["crisis_prevalence"] * h["n"] for h in have) / rows),
               "pooled": st._pooled({m: [h[m] for h in have] for m in models}, ALL_PAIRS),
               "mean_fold": st._mean_fold(have, models, ALL_PAIRS),
               "fold_wins": st._wins(have, ALL_PAIRS),
               "all_keys_pooled": st._pooled({a: [e[a] for e in allk] for a in ARMS}, ()),
               "ranking_mean_fold": {}}
        for a in ARMS:
            el = [h["ranking"][a] for h in have if h["ranking"][a]["eligible"]]
            res["ranking_mean_fold"][a] = {"eligible_folds": len(el),
                                           "auc": float(np.mean([e["auc"] for e in el])) if el else None,
                                           "ap": float(np.mean([e["ap"] for e in el])) if el else None}
        out[part] = res
    roots = [r for r in per_root.values() if select(r)]
    cells = {}
    for c in map(str, CODES):
        entry = {"origin_phase": int(c) + 1, "n_folds": len(roots), "mean_fold_auc": {}, "n_valid": {}}
        for a in ARMS:
            v = [r["d50_cells"][c]["auc"][a] for r in roots if r["d50_cells"][c]["auc"][a] is not None]
            entry["n_valid"][a] = f"{len(v)}/{len(roots)}"
            entry["mean_fold_auc"][a] = float(np.mean(v)) if v else None
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


def run_root(run, stage, out, name, pair, producer_rev, fmap, base, defaults, d49, d50):
    """Returns (gate, rows, meta); rows is None when the gate did not pass (no fit happened)."""
    root = json.loads((stage / "roots" / name / "root.json").read_text(encoding="utf-8"))
    gate, data, replay, root_ubj = rr.gate_root(run, stage, name, pair, root)
    h = int(root["horizon"])
    g_config = root["g_config"]
    gate["checks"]["g_config_locked"] = {"mismatches": int(G_BY_H.get(h) != g_config), "n": 1}
    gate["passed"] = gate["passed"] and G_BY_H.get(h) == g_config
    if not gate["passed"]:
        return gate, None, None
    idx = fmap["selected_index"]
    inputs = {"FIT": data["FIT"], "C": data["C"], "E3": data["E3"]}
    projected, proj_checks = {}, {}
    for part in PARTS:
        projected[part], proj_checks[part] = project(inputs[part][0], idx)
    X_fit, y_fit, gfit, mfit = data["FIT"]
    check_defaults(defaults)
    booster, record = fit_h78_root(X_fit, y_fit, idx, g_config)
    check_fit(booster, record, plan.G_CONFIGS[g_config], int(len(y_fit)))
    fit_keys_sha, fit_labels_sha = nx.keys_sha(gfit, mfit), hashlib.sha256(
        np.ascontiguousarray(np.asarray(y_fit, dtype=np.int64)).tobytes()).hexdigest()
    if fit_keys_sha != root.get("fitting_keys_sha256"):
        raise GateError("FIT keys used for the fit differ from the original FIT keys")
    if record.get("class_counts") != [int(np.sum(np.asarray(y_fit) == k)) for k in range(nx.N_CLASSES)]:
        raise GateError("fit record class counts differ from the original FIT labels")
    rdir = out / name
    rdir.mkdir()
    (rdir / "h78_root.ubj").write_bytes(nx.raw(booster))
    frozen = nx.from_raw((rdir / "h78_root.ubj").read_bytes())
    if num_feature(frozen) != str(N_FEATURES):
        raise GateError("reloaded 78-feature root does not record 78 features")
    original = nx.from_raw(root_ubj.read_bytes())          # sha gated in gate_root
    rows, reload_checks = {}, {}
    for part in PARTS:
        X, y, g, m = inputs[part]
        if part == "FIT":
            m = month_label(m)
        p_orig = nx.proba(original, X) if part == "FIT" else replay[part]   # FULL-162 matrices
        p78 = nx.proba(frozen, projected[part])
        mism = sr.exact_mismatches(nx.proba(booster, projected[part]), p78)
        reload_checks[part] = {"mismatches": int(mism), "n": int(p78.size)}
        if mism:
            raise GateError(f"reloaded 78-feature root does not reproduce its {part} probabilities ({mism} cells)")
        rows[part] = part_frame(X, y, g, m, h, p_orig, p78)
        write_rows(rdir / f"rows_{part}.csv.gz", rows[part])
    cells = d50_cells(rows["E3"])
    cons = consistency(name, rows["E3"], cells, d49, d50)
    o_index = data["window"][2]
    meta = {"root": name, "horizon": h, "target_month": root["target_month"], "g_config": g_config,
            "origin_month": month_label(np.array([o_index]))[0], "fit_rule": FIT_RULE,
            "fitting_rows": int(len(gfit)), "fitting_keys_sha256": fit_keys_sha,
            "fitting_labels_sha256": fit_labels_sha, "g_config_values": plan.G_CONFIGS[g_config],
            "projection": proj_checks, "original_root_source": str(root_ubj),
            "original_root_sha256": root["root_booster_sha256"],
            "h78_root_sha256": rid.file_sha256(rdir / "h78_root.ubj"), "num_feature": num_feature(frozen),
            "reload_exact": reload_checks, "fit_record": record, "producer_rev": producer_rev,
            "d49_d50_consistency": cons}
    rid.write_json_atomic(rdir / "h78_root.json", meta)
    dc = rr.dev_baseline_check(rows["E3"], base, h)
    if dc["status"] == "checked" and (dc["joined"] != dc["e3_rows"] or dc["persistence_mismatches"]
                                      or dc["truth_mismatches"]):
        gate = {**gate, "passed": False, "error": f"dev_baselines cross-check failed: {dc}"}
    return gate, rows, meta | {"dev_baselines_check": dc, "d50_cells": cells}


def run_all(run, stage, out, roots, cands, producer_rev, fmap, base, d49, d50) -> tuple[int, dict, dict]:
    """Sequential roots; the first gate / projection / fit / consistency failure writes evidence and stops."""
    defaults = plan_defaults()
    gates, per_root = {}, {}
    for name in roots:
        pair = {cands[c]["e1"]: c for c in cands if cands[c]["root"] == name}
        try:
            gate, rows, meta = run_root(run, stage, out, name, pair, producer_rev, fmap, base, defaults, d49, d50)
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
        raise RuntimeError(f"D51 script {rel} is not committed at HEAD") from exc
    dirty = _git("status", "--porcelain", "--", rel)
    if blob != head_blob or dirty:
        raise RuntimeError(f"D51 script {rel} differs from HEAD (blob {blob} vs {head_blob}; status {dirty!r})")
    return {"path": rel, "git_blob": blob, "sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest()}


# ---------------------------------------------------------------- selftest (synthetic fits only)

def fit_call_sites() -> list:
    """Names of the functions that call ``nx.fit_global`` in this script (AST, not text)."""
    tree, sites = ast.parse(SCRIPT.read_text(encoding="utf-8")), []
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "fit_global" and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "nx"):
                    sites.append(fn.name)
    return sites


def selftest() -> int:
    rng = np.random.default_rng(0)
    # structural guard: exactly one fit call, inside fit_h78_root
    assert fit_call_sites() == ["fit_h78_root"], fit_call_sites()
    # (1) projection on the real schema file
    schema = load_schema(sr.SCHEMA)
    fmap = feature_map(schema)
    idx = fmap["selected_index"]
    assert len(idx) == 78 and idx == sorted(idx) and fmap["removed_count"] == 84
    assert fmap["selected_names"] == [n for n in schema["ordered_features"]
                                      if n in set(sum(schema["history_blocks"].values(), []))
                                      or n in set(schema["known_calendar"])]
    assert set(fmap["selected_names"]) >= set(schema["known_calendar"]) and fmap["phase_full_index"] == 87
    assert fmap["selected_names"][fmap["phase_projected_index"]] == "hist_phase_o00"
    assert fmap["phase_projected_index"] != 87
    X = rng.normal(size=(50, 162)).astype(np.float32)
    X[rng.random(X.shape) < 0.2] = np.nan
    P, rec = project(X, idx)
    assert rec["dtype"] == "float32" and P.shape == (50, 78)
    assert np.array_equal(P, X[:, idx], equal_nan=True) and np.isnan(P).any()
    X64 = X.astype(np.float64)
    assert project(X64, idx)[1]["dtype"] == "float64"
    try:
        feature_map({**schema, "known_calendar": schema["known_calendar"][:2]})
        raise AssertionError("a 77-feature schema was accepted")
    except GateError:
        pass
    # (2) D50 cells, score_part and consistency on synthetic frames
    n = 60
    Xf = np.zeros((n, 162))
    Xf[:, 87] = np.tile([1.0, 2.0, 3.0, 4.0, 5.0], 12)
    Xf[:5, 87] = np.nan
    yv = np.tile([0, 1, 2, 3, 2, 0], 10)
    p = rng.dirichlet(np.ones(4), size=n)
    fr = part_frame(Xf, yv, np.arange(n), np.array(["2019-06"] * n), 4, p, p[::-1].copy())
    sc = score_part(fr)
    assert sc["n"] == 55 and sc["excluded_missing_origin"] == 5 and sc["n_all"] == 60
    assert "logloss_fourclass" not in sc["persistence"] and "logloss_fourclass" in sc["h78"]
    cells = d50_cells(fr)
    assert sum(cells[c]["n"] for c in cells) == 55
    k = np.isfinite(fr["persistence_code"].to_numpy(float))
    oc = crisis_confusion(fr["truth"].to_numpy()[k], fr["y_original"].to_numpy()[k])
    pc = crisis_confusion(fr["truth"].to_numpy()[k], fr["persistence_code"].to_numpy(float)[k].astype(int))
    ref50 = {"cells": {c: {"n": v["n"], "P": v["P"], "N": v["N"], "auc": {"original": v["auc"]["original"]},
                           "auc_null_reason": {"original": v["auc_null_reason"]["original"]}}
                       for c, v in cells.items()}}
    d49 = {"per_root": {"r": {"n": 55, "excluded_missing_origin": 5, "persistence": pc,
                              "arms": {"original": {"argmax": oc}}}}}
    consistency("r", fr, cells, d49, {"per_root": {"r": ref50}})
    bad = copy.deepcopy(d49)
    bad["per_root"]["r"]["arms"]["original"]["argmax"]["tp"] += 1
    try:
        consistency("r", fr, cells, bad, {"per_root": {"r": ref50}})
        raise AssertionError("a D49 confusion mismatch was accepted")
    except GateError:
        pass
    agg = aggregate({"r": {"horizon": 4, "target_month": "2019-06", "scores": {q: sc for q in PARTS},
                           "d50_cells": cells}}, lambda r: True)
    assert agg["E3"]["rows"] == 55 and agg["E3"]["coverage"]["excluded_missing_origin_pooled"] == 5
    # (3) synthetic four-class fit-and-reload round trip on 78 projected columns (not a real fit)
    Xs = rng.normal(size=(400, 162))
    Xs[rng.random(Xs.shape) < 0.1] = np.nan
    ys = rng.integers(0, 4, size=400)
    plan.G_CONFIGS["_SELFTEST"] = {**plan.G_CONFIGS["G1"], "rounds": 3}
    try:
        booster, record = fit_h78_root(Xs, ys, idx, "_SELFTEST")
        check_fit(booster, record, plan.G_CONFIGS["_SELFTEST"], 400)
    finally:
        del plan.G_CONFIGS["_SELFTEST"]
    assert num_feature(booster) == "78"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "h78_root.ubj"
        path.write_bytes(nx.raw(booster))
        frozen = nx.from_raw(path.read_bytes())
    Ps = Xs[:, idx]
    assert sr.exact_mismatches(nx.proba(booster, Ps), nx.proba(frozen, Ps)) == 0 and num_feature(frozen) == "78"
    # (4) a failed gate stops the run with evidence and never fits
    calls = []
    real_gate, real_fit = rr.gate_root, nx.fit_global

    def failing_gate(run, stage, name, pair, root):
        return {"root": name, "checks": {"injected": {"mismatches": 1, "n": 1}}, "passed": False}, None, None, None

    def sentinel_fit(*a, **kw):
        calls.append(1)
        raise AssertionError("fit_global called after a failed gate")

    rr.gate_root, nx.fit_global = failing_gate, sentinel_fit
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            stage, out = tmp / "stage", tmp / "out"
            for name in ("r1", "r2"):
                (stage / "roots" / name).mkdir(parents=True)
                (stage / "roots" / name / "root.json").write_text(
                    json.dumps({"horizon": 4, "g_config": "G1", "target_month": "2019-06"}), encoding="utf-8")
            out.mkdir()
            cands = {"c1": {"root": "r1", "e1": "hard_f1"}, "c2": {"root": "r2", "e1": "hard_f1"}}
            code, gates, per_root = run_all(tmp, stage, out, ["r1", "r2"], cands, "7b2bf6f", fmap, None, {}, {})
            failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
            assert code == 2 and not calls and not per_root
            assert failure["failed_root"] == "r1" and failure["not_attempted"] == ["r2"]
            assert not (out / "r1").exists()
    finally:
        rr.gate_root, nx.fit_global = real_gate, real_fit
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
    rid.refuse_existing(out, "D51 history+calendar root")
    defaults = plan_defaults()
    script = script_identity()
    producer = rid.code_identity_at(args.producer_rev)
    if rid.code_identity() != rid.code_identity_at("HEAD"):
        raise RuntimeError("D51 must run from committed package code (working tree differs from HEAD)")
    roots, cands, ident = accept_mode(run, plan.E1PAIR, producer, args.producer_rev)
    stage = Path(ident["stage"])
    d49 = json.loads(D49_SUMMARY.read_text(encoding="utf-8"))
    d50 = json.loads(D50_SUMMARY.read_text(encoding="utf-8"))
    check_inventory(roots, d49, d50)
    fmap = feature_map(load_schema(sr.SCHEMA))
    base_path = run / "prepared" / "ledgers" / "dev_baselines.csv"   # dev targets only; never baselines.csv
    base = pd.read_csv(base_path)
    identity = {"stage": "d51_history_calendar_root", "d34_run": str(run), "producer_rev": args.producer_rev,
                "producer_code": producer, "repo_head": _git("rev-parse", "HEAD"), "script": script,
                "package_code": rid.code_identity(), "runtime": rid.runtime_identity(), "max_month": "2020-12",
                "g_by_horizon": G_BY_H, "g_configs": {g: plan.G_CONFIGS[g] for g in sorted(set(G_BY_H.values()))},
                "fit_rule": FIT_RULE, "plan_defaults": defaults,
                "schema": {"path": str(sr.SCHEMA), "sha256": rid.file_sha256(sr.SCHEMA)},
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
    rid.write_json_atomic(out / "feature_map.json", {**fmap, "schema_sha256": rid.file_sha256(sr.SCHEMA)})
    code, gates, per_root = run_all(run, stage, out, roots, cands, args.producer_rev, fmap, base, d49, d50)
    if code:
        print("STOPPED at the first failure: no summary computed; see gate.json / failure.json")
        return code
    summary = {
        "per_root": per_root,
        "by_horizon": {f"H{h}": aggregate(per_root, lambda r, h=h: r["horizon"] == h) for h in HORIZONS},
        "interpretation": ("Primary: E3 crisis F1 (argmax -> code >= 2) on matched exact-origin keys per H, "
                           "h78 - original and each - persistence on the same keys; crisis-call share, "
                           "macro-F1, unweighted Brier and log loss alongside (persistence: no log loss). "
                           "All-key E3 per arm only, with missing-origin counts; nothing vs persistence on all "
                           "keys. FIT is in-sample, C in-window interpolation; a smaller FIT/C gap is not "
                           "evidence. Joint removal of 84 features with G/rounds/colsample frozen at 162-feature "
                           "values: no attribution to a variable, block or mechanism. AUC/AP within-root, "
                           "mean-fold only; D50 cells valid-fold means with n_valid/7, no pooled AUC. 21 "
                           "overlapping, repeatedly developed folds: no significance test, no adoption rule."),
    }
    check_defaults(defaults)
    rid.write_json_atomic(out / "summary.json", summary)
    rid.write_json_atomic(out / "completion.json", {"status": "completed", "outputs": rid.output_hashes(out)})
    print(f"D51 history+calendar root completed: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
