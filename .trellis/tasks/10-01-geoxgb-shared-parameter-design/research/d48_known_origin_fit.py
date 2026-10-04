#!/usr/bin/env python3
"""D48 / A22: known-origin FIT restriction contrast for the 21 saved D34 roots (one-off research script).

python .trellis/tasks/10-01-geoxgb-shared-parameter-design/research/d48_known_origin_fit.py \
    --d34-run D34_RUN --out NEW_DIR [--producer-rev 7b2bf6f]
python .../d48_known_origin_fit.py --selftest      # synthetic, no real data, no real fit

Per root (sequential, stop at the first failure): D34 pinned acceptance, then the D37 rebuild and
original-root replay GATE (stage1_recency_root.gate_root) before any fit. Only then the eligible FIT mask
``np.isfinite(X_fit[:, hist_phase_o00])`` on the ORIGINAL rebuilt FIT matrix (target values are never
read to choose rows) and ONE fresh ``fit_global(X_fit[known], y_fit[known], plan.G_CONFIGS[G])`` -- the
same G (H4 G1, H8 G4, H12 G2), rounds, seed and features; no sample weight, no base margin. Fit checks fail
closed (params/rounds = booster_params(G), base score .5, four classes, no weight/margin block). The root
is saved, reloaded from UBJ (exact on FIT/C/E3-known) and scored next to the replayed original root and
exact-origin persistence on EXACTLY the origin-known keys of each part: FIT (in-sample), C (in-window
interpolation), E3 (forward, primary). Missing-origin rows are only counted (coverage); they are never
predicted for reporting or scored. No local trees, maps, Stage 2/3 inputs; no final-period ledger is read.
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

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
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
LABELS = fourclass.CLASS_LABELS
PARTS = ("FIT", "C", "E3")
ROLE = {"FIT": "in-sample fitting rows (origin-known)", "C": "in-window historical interpolation (origin-known)",
        "E3": "forward target month, origin-known keys (primary)"}
ARMS = ("original", "known")
ARM_DELTAS = (("known", "original"),)
PERS_DELTAS = (("known", "persistence"), ("original", "persistence"))
ALL_PAIRS = PERS_DELTAS + ARM_DELTAS
MASK_RULE = ("eligible FIT rows = original rebuilt FIT rows with finite hist_phase_o00 (exact-origin label "
             "feature); target values never read; no S/C/E3 rows; no weights or margins")
RULE = ("exact equality, round-trip parse; a root failing the pre-fit gate is not fitted; the first gate, "
        "mask, fit, reload, scoring or dev_baselines failure stops the run before later roots")
GateError = rr.GateError


# ---------------------------------------------------------------- config, mask and fit checks

def plan_defaults() -> dict:
    return {"G_CONFIGS": copy.deepcopy(plan.G_CONFIGS), "XGB_BASE": dict(plan.XGB_BASE)}


def check_defaults(snapshot: dict) -> None:
    if plan_defaults() != snapshot:
        raise GateError("plan.G_CONFIGS or plan.XGB_BASE changed during the run")


def known_mask(X, phase_col: int) -> np.ndarray:
    """Origin-known rows: finite exact-origin hist_phase_o00. Depends on the feature column only."""
    return np.isfinite(np.asarray(X, dtype=float)[:, phase_col])


def sorted_keys_sha(areas, months) -> str:
    """sha256 of the (area, month-index) keys sorted by area then month."""
    a, m = np.asarray(areas, dtype=np.int64), np.asarray(months, dtype=np.int64)
    order = np.lexsort((m, a))
    return nx.keys_sha(a[order], m[order])


def eligibility(X_fit, y_fit, gfit, mfit, phase_col: int) -> tuple[np.ndarray, dict]:
    """Mask (from X only) plus its record; y is used only for the class-count description afterwards."""
    known = known_mask(X_fit, phase_col)
    if not known.any():
        raise GateError("no origin-known FIT rows")
    y_k = np.asarray(y_fit, dtype=np.int64)[known]
    return known, {"rule": MASK_RULE, "known": int(known.sum()), "total": int(len(known)),
                   "fraction": float(known.mean()),
                   "eligible_keys_sha256_sorted": sorted_keys_sha(np.asarray(gfit)[known], np.asarray(mfit)[known]),
                   "label_dates": int(len(np.unique(np.asarray(mfit)[known]))),
                   "areas": int(len(np.unique(np.asarray(gfit)[known]))),
                   "class_counts": {lab: int(np.sum(y_k == k)) for k, lab in enumerate(LABELS)}}


def check_disjoint(data: dict, known: np.ndarray) -> dict:
    """No S / C / E3 key may be an eligible FIT key."""
    _, _, gfit, mfit = data["FIT"]
    fit_keys = rr.gi.key_set(np.asarray(gfit)[known], month_label(np.asarray(mfit)[known]))
    out = {}
    for part in ("S", "C", "E3"):
        if part in data:
            n = len(fit_keys & rr.gi.key_set(data[part][2], data[part][3]))
            out[part] = {"overlap": int(n), "n": int(len(data[part][1]))}
            if n:
                raise GateError(f"{n} {part} keys are in the eligible FIT key set")
    return out


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
    if "sample_weight" in record or "base_margin" in record or nx.is_margin_marked(booster):
        problems.append("a sample weight or base margin is present")
    if record.get("rows") != n_rows:
        problems.append("fit record rows differ from the origin-known FIT rows")
    if problems:
        raise GateError("known-origin root fit check failed: " + "; ".join(problems))


def fit_known_root(data: dict, known: np.ndarray, g_config: str):
    """The single D48 fit: fresh root on the origin-known original FIT rows (no weight, no margin)."""
    X_fit, y_fit, _, _ = data["FIT"]
    return nx.fit_global(X_fit[known], y_fit[known], plan.G_CONFIGS[g_config])


# ---------------------------------------------------------------- scoring (origin-known keys only)

def part_frame(X, y, g, m, horizon, p_original, p_known, phase_col) -> pd.DataFrame:
    """Rows must already be origin-known; persistence is then defined on every key."""
    frame = pd.DataFrame({"area": g, "target_month": m, "horizon": int(horizon), "truth": y,
                          "persistence_code": rr.persistence_codes(X[:, phase_col])})
    if not np.isfinite(frame["persistence_code"].to_numpy(float)).all():
        raise GateError("part_frame received a missing-origin row")
    for arm, p in (("original", p_original), ("known", p_known)):
        frame[f"y_{arm}"] = fourclass.argmax_codes(p)
        for k, lab in enumerate(LABELS):
            frame[f"p_{arm}_{lab}"] = p[:, k]
    return frame


def score_part(frame: pd.DataFrame, total: int) -> dict:
    """original, known and persistence on the same origin-known keys; persistence has no log loss."""
    cov = {"known": int(len(frame)), "total": int(total), "excluded_missing_origin": int(total - len(frame)),
           "coverage": float(len(frame) / total) if total else None}
    if len(frame) == 0:
        return {"n": 0, "status": "no_data", "coverage": cov}
    t = frame["truth"].to_numpy()
    prob = {a: frame[[f"p_{a}_{lab}" for lab in LABELS]].to_numpy(np.float64) for a in ARMS}
    pers = frame["persistence_code"].to_numpy(float).astype(np.int64)
    out = {"n": int(len(t)), "coverage": cov, "crisis_prevalence": float(rr.crisis(t).mean())}
    for a in ARMS:
        out[a] = st._share(cw.block(t, frame[f"y_{a}"].to_numpy(), prob[a]))
    out["persistence"] = st._share(cw.block(t, pers, np.eye(nx.N_CLASSES)[pers], logloss=False))
    for a, b in ALL_PAIRS:
        out[f"{a}_minus_{b}"] = cw.delta(out[a], out[b])
    out["ranking"] = {a: cw.ranking(t, prob[a]) for a in ARMS}
    return out


def aggregate(per_root: dict, select) -> dict:
    """Per H: coverage, pooled (summed confusions) and mean-fold kept separate; AUC/AP mean-fold only."""
    out = {}
    for part in PARTS:
        sel = [r["scores"][part] for r in per_root.values() if select(r)]
        cov = [s["coverage"] for s in sel]
        coverage = {"known_pooled": int(sum(c["known"] for c in cov)), "total_pooled": int(sum(c["total"] for c in cov)),
                    "excluded_pooled": int(sum(c["excluded_missing_origin"] for c in cov))}
        coverage["coverage_pooled"] = (coverage["known_pooled"] / coverage["total_pooled"]
                                       if coverage["total_pooled"] else None)
        fr = [c["coverage"] for c in cov if c["coverage"] is not None]
        coverage["coverage_mean_fold"] = float(np.mean(fr)) if fr else None
        have = [s for s in sel if s.get("n", 0)]
        if not have:
            out[part] = {"folds_with_data": 0, "status": "no_data", "coverage": coverage}
            continue
        rows = int(sum(h["n"] for h in have))
        models = ARMS + ("persistence",)
        res = {"role": ROLE[part], "coverage": coverage, "folds_with_data": len(have), "rows": rows,
               "crisis_prevalence_pooled": float(sum(h["crisis_prevalence"] * h["n"] for h in have) / rows),
               "pooled": st._pooled({m: [h[m] for h in have] for m in models}, ALL_PAIRS),
               "mean_fold": st._mean_fold(have, models, ALL_PAIRS),
               "fold_wins": st._wins(have, ALL_PAIRS), "ranking_mean_fold": {}}
        for a in ARMS:
            el = [h["ranking"][a] for h in have if h["ranking"][a]["eligible"]]
            res["ranking_mean_fold"][a] = {"eligible_folds": len(el),
                                           "auc": float(np.mean([e["auc"] for e in el])) if el else None,
                                           "ap": float(np.mean([e["ap"] for e in el])) if el else None}
        out[part] = res
    return out


# ---------------------------------------------------------------- per root

def run_root(run, stage, out, name, pair, producer_rev, phase_col, base, defaults):
    """Returns (gate, rows, meta); rows is None when the gate did not pass (no fit happened)."""
    root = json.loads((stage / "roots" / name / "root.json").read_text(encoding="utf-8"))
    gate, data, replay, root_ubj = rr.gate_root(run, stage, name, pair, root)
    h = int(root["horizon"])
    g_config = root["g_config"]
    gate["checks"]["g_config_locked"] = {"mismatches": int(G_BY_H.get(h) != g_config), "n": 1}
    gate["passed"] = gate["passed"] and G_BY_H.get(h) == g_config
    if not gate["passed"]:
        return gate, None, None
    X_fit, y_fit, gfit, mfit = data["FIT"]
    known, mask_record = eligibility(X_fit, y_fit, gfit, mfit, phase_col)
    mask_record["disjoint"] = check_disjoint(data, known)
    check_defaults(defaults)
    booster, record = fit_known_root(data, known, g_config)
    check_fit(booster, record, plan.G_CONFIGS[g_config], int(known.sum()))
    rdir = out / name
    rdir.mkdir()
    (rdir / "known_root.ubj").write_bytes(nx.raw(booster))
    frozen = nx.from_raw((rdir / "known_root.ubj").read_bytes())
    original = nx.from_raw(root_ubj.read_bytes())          # sha gated in gate_root
    inputs = {"FIT": data["FIT"], "C": data["C"], "E3": data["E3"]}
    rows, totals, reload_checks = {}, {}, {}
    for part in PARTS:
        X, y, g, m = inputs[part]
        if part == "FIT":
            m = month_label(m)
        k = known if part == "FIT" else known_mask(X, phase_col)
        totals[part] = int(len(y))
        Xk = X[k]
        p_orig = nx.proba(original, Xk) if part == "FIT" else replay[part][k]
        pk = nx.proba(frozen, Xk)
        mism = sr.exact_mismatches(nx.proba(booster, Xk), pk)
        reload_checks[part] = {"mismatches": int(mism), "n": int(pk.size)}
        if mism:
            raise GateError(f"reloaded known-origin root does not reproduce its {part} probabilities ({mism} cells)")
        rows[part] = part_frame(Xk, y[k], g[k], m[k], h, p_orig, pk, phase_col)
        with gzip.open(rdir / f"rows_{part}_known.csv.gz", "wt", encoding="utf-8", newline="") as handle:
            rows[part].to_csv(handle, index=False, float_format="%.17g")
    o_index = data["window"][2]
    meta = {"root": name, "horizon": h, "target_month": root["target_month"], "g_config": g_config,
            "origin_month": month_label(np.array([o_index]))[0], "fitting_rows": int(len(gfit)),
            "fitting_keys_sha256": nx.keys_sha(gfit, mfit), "eligibility": mask_record, "part_totals": totals,
            "original_root_source": str(root_ubj), "original_root_sha256": root["root_booster_sha256"],
            "known_root_sha256": rid.file_sha256(rdir / "known_root.ubj"), "reload_exact": reload_checks,
            "fit_record": record, "producer_rev": producer_rev}
    rid.write_json_atomic(rdir / "known_root.json", meta)
    dc = rr.dev_baseline_check(rows["E3"], base, h)
    if dc["status"] == "checked" and (dc["joined"] != dc["e3_rows"] or dc["persistence_mismatches"]
                                      or dc["truth_mismatches"]):
        gate = {**gate, "passed": False, "error": f"dev_baselines cross-check failed: {dc}"}
    return gate, rows, meta | {"dev_baselines_check": dc}


def run_all(run, stage, out, roots, cands, producer_rev, phase_col, base) -> tuple[int, dict, dict]:
    """Sequential roots; the first gate / mask / fit / scoring failure writes evidence and stops."""
    defaults = plan_defaults()
    gates, per_root = {}, {}
    for name in roots:
        pair = {cands[c]["e1"]: c for c in cands if cands[c]["root"] == name}
        try:
            gate, rows, meta = run_root(run, stage, out, name, pair, producer_rev, phase_col, base, defaults)
            if rows is not None and gate["passed"]:
                per_root[name] = {**meta, "scores": {p: score_part(rows[p], meta["part_totals"][p])
                                                     for p in PARTS}}
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
        raise RuntimeError(f"D48 script {rel} is not committed at HEAD") from exc
    dirty = _git("status", "--porcelain", "--", rel)
    if blob != head_blob or dirty:
        raise RuntimeError(f"D48 script {rel} differs from HEAD (blob {blob} vs {head_blob}; status {dirty!r})")
    return {"path": rel, "git_blob": blob, "sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest()}


# ---------------------------------------------------------------- selftest (synthetic only)

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
    # (d) structural guard: exactly one fit call, inside fit_known_root
    assert fit_call_sites() == ["fit_known_root"], fit_call_sites()
    # (b) mask independence from y
    n, phase_col = 40, 2
    X = rng.normal(size=(n, 4))
    X[:, phase_col] = rng.integers(1, 6, size=n).astype(float)
    X[rng.random(n) < 0.4, phase_col] = np.nan
    g = rng.integers(100, 110, size=n)
    m = np.arange(n, dtype=np.int64) + 24000
    y1, y2 = rng.integers(0, 4, size=n), rng.integers(0, 4, size=n)
    k1, r1 = eligibility(X, y1, g, m, phase_col)
    k2, r2 = eligibility(X, y2, g, m, phase_col)
    assert np.array_equal(k1, k2) and np.array_equal(k1, np.isfinite(X[:, phase_col]))
    assert r1["eligible_keys_sha256_sorted"] == r2["eligible_keys_sha256_sorted"]
    perm = rng.permutation(n)
    _, r3 = eligibility(X[perm], y1[perm], g[perm], m[perm], phase_col)
    assert r3["eligible_keys_sha256_sorted"] == r1["eligible_keys_sha256_sorted"]

    # (c) scoring on origin-known keys only; missing-origin rows counted, not scored
    total = 30
    Xs = np.full((total, 4), 0.0)
    Xs[:, phase_col] = np.tile([1.0, 2.0, 3.0, 4.0, 5.0], 6)
    Xs[:7, phase_col] = np.nan
    ys = np.tile([0, 1, 2, 3, 1], 6)
    p = rng.dirichlet(np.ones(4), size=total)
    p[np.arange(total), ys] += 0.01                     # keep true-class probabilities positive
    p /= p.sum(axis=1, keepdims=True)
    k = known_mask(Xs, phase_col)
    frame = part_frame(Xs[k], ys[k], np.arange(total)[k], np.array(["2019-06"] * total)[k], 4, p[k], p[k][::-1].copy(),
                       phase_col)
    sc = score_part(frame, total)
    assert sc["n"] == 23 and sc["coverage"]["excluded_missing_origin"] == 7 and sc["coverage"]["total"] == 30
    assert "logloss_fourclass" not in sc["persistence"] and "logloss_fourclass" in sc["known"]
    try:
        part_frame(Xs, ys, np.arange(total), np.array(["2019-06"] * total), 4, p, p, phase_col)
        raise AssertionError("missing-origin row was accepted for scoring")
    except GateError:
        pass
    agg = aggregate({"r": {"horizon": 4, "scores": {q: sc for q in PARTS}}}, lambda r: True)
    assert agg["E3"]["coverage"]["excluded_pooled"] == 7 and agg["E3"]["rows"] == 23

    # (a) a failed gate stops the run with evidence and never fits
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
            code, gates, per_root = run_all(tmp, stage, out, ["r1", "r2"], cands, "7b2bf6f", phase_col, None)
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
    rid.refuse_existing(out, "D48 known-origin root")
    defaults = plan_defaults()
    script = script_identity()
    producer = rid.code_identity_at(args.producer_rev)
    if rid.code_identity() != rid.code_identity_at("HEAD"):
        raise RuntimeError("D48 must run from committed package code (working tree differs from HEAD)")
    roots, cands, ident = accept_mode(run, plan.E1PAIR, producer, args.producer_rev)
    stage = Path(ident["stage"])
    phase_col = load_schema(sr.SCHEMA)["ordered_features"].index("hist_phase_o00")
    base_path = run / "prepared" / "ledgers" / "dev_baselines.csv"   # dev targets only; never baselines.csv
    base = pd.read_csv(base_path)
    identity = {"stage": "d48_known_origin_fit", "d34_run": str(run), "producer_rev": args.producer_rev,
                "producer_code": producer, "repo_head": _git("rev-parse", "HEAD"), "script": script,
                "package_code": rid.code_identity(), "runtime": rid.runtime_identity(), "max_month": "2020-12",
                "g_by_horizon": G_BY_H, "g_configs": {g: plan.G_CONFIGS[g] for g in sorted(set(G_BY_H.values()))},
                "mask_rule": MASK_RULE, "plan_defaults": defaults,
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
        "interpretation": ("Primary: E3 crisis F1 (argmax -> code >= 2) on origin-known keys per H, known - "
                           "original and each - persistence on the same keys; crisis-call share, macro-F1, "
                           "unweighted Brier and log loss alongside. FIT-known is in-sample, C-known in-window "
                           "interpolation. Missing-origin E3 rows are counted only (no scores, no extrapolation) "
                           "and nothing is compared with all-key scores. The restriction changes sample size, "
                           "calendar regime, era, composition, missingness selection and optimisation path "
                           "together; no causal or covariate-shift claim. AUC/AP within-root, mean-fold only. "
                           "21 overlapping, repeatedly developed folds: no significance test, no adoption rule."),
    }
    check_defaults(defaults)
    rid.write_json_atomic(out / "summary.json", summary)
    rid.write_json_atomic(out / "completion.json", {"status": "completed", "outputs": rid.output_hashes(out)})
    print(f"D48 known-origin root completed: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
