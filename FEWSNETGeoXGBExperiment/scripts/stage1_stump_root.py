#!/usr/bin/env python3
"""D47 / A21: depth-1 (stump) ROOT contrast for the 21 saved D34 roots.

python scripts/stage1_stump_root.py --d34-run D34_RUN --out NEW_DIR [--producer-rev 7b2bf6f]

Per root (sequential, stop at the first failure): D34 pinned acceptance, then the D37 rebuild
(Parquet filter target_month <= 2020-12, fitting / S / C / E3 key, role, fitting-hash and window
guards) and original-root replay GATE (stage1_recency_root.gate_root) before any fit. Only then ONE
fresh ``fit_global(X_fit, y_fit, dict(G, max_depth=1))`` -- the same G (H4 G1, H8 G4, H12 G2), rounds,
eta, sampling, regularisers and seed; no sample weight, no base margin. Fit checks (fail closed):
recorded params equal booster_params(G) except max_depth 1; the fit-time resolved config shows
max_depth 1; every saved tree has depth <= 1 (the UBJ reload does not keep training parameters, so the
tree structure is the depth evidence); base score .5, four classes, configured rounds; no weight or
margin. The stump is saved, reloaded from UBJ (FIT/C/E3 probabilities must reproduce exactly) and
scored on the same FIT (in-sample), C (in-window interpolation) and E3 (forward, primary) keys next to
the replayed original root and exact-origin persistence (matched non-missing keys, no log loss).
No local trees, E4 weights, maps or Stage 2 inputs; no final-period ledger is read.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import subprocess
import sys
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from scripts import stage1_class_weight_root as cw  # noqa: E402
from scripts import stage1_recency_root as rr  # noqa: E402
from scripts import stage1_shallow_replay as sr  # noqa: E402
from scripts.stage1_rootconf_compare import accept_mode  # noqa: E402
from src.experiment import plan  # noqa: E402
from src.feature.fourclass_features import load_schema, month_label  # noqa: E402
from src.metrics import fourclass  # noqa: E402
from src.model import native_xgb as nx  # noqa: E402
from src.utils import run_identity as rid  # noqa: E402

STUMP_DEPTH = 1
G_BY_H = {4: "G1", 8: "G4", 12: "G2"}
LABELS = fourclass.CLASS_LABELS
PARTS = ("FIT", "C", "E3")
ROLE = {"FIT": "in-sample fitting rows", "C": "in-window historical interpolation (diagnostic)",
        "E3": "forward target month (primary)"}
ARMS = ("original", "stump")
ARM_DELTAS = (("stump", "original"),)
PERS_DELTAS = (("stump", "persistence"), ("original", "persistence"))
RULE = ("exact equality, round-trip parse; a root failing the pre-fit gate is not fitted; the first gate, "
        "fit, depth, reload, scoring or dev_baselines failure stops the run before later roots")
GateError = rr.GateError


# ---------------------------------------------------------------- config and fit checks (tested)

def plan_defaults() -> dict:
    """Snapshot of the shared plan defaults; must be equal before every fit and at the end."""
    return {"G_CONFIGS": copy.deepcopy(plan.G_CONFIGS), "XGB_BASE": dict(plan.XGB_BASE)}


def check_defaults(snapshot: dict) -> None:
    if plan_defaults() != snapshot:
        raise GateError("plan.G_CONFIGS or plan.XGB_BASE changed during the run")


def stump_config(g: str) -> dict:
    """A copy of the G configuration with max_depth 1; the plan table is not touched."""
    return dict(plan.G_CONFIGS[g], max_depth=STUMP_DEPTH)


def tree_structure(booster) -> dict:
    """Depth, split and leaf counts of every saved tree (from the model JSON, not the config)."""
    trees = json.loads(bytes(booster.save_raw("json")))["learner"]["gradient_booster"]["model"]["trees"]
    depths, splits, leaves = [], [], []
    for t in trees:
        left, right = t["left_children"], t["right_children"]
        depth = {0: 0}
        stack = [0]
        while stack:
            node = stack.pop()
            for child in (left[node], right[node]):
                if child != -1:
                    depth[child] = depth[node] + 1
                    stack.append(child)
        if len(depth) != len(left):
            raise GateError("tree has nodes unreachable from its root")
        depths.append(max(depth.values()))
        splits.append(int(sum(c != -1 for c in left)))
        leaves.append(int(sum(c == -1 for c in left)))
    return {"trees": len(trees), "max_depth": int(max(depths)) if depths else 0,
            "trees_depth_gt_1": int(sum(d > STUMP_DEPTH for d in depths)),
            "split_nodes_total": int(sum(splits)), "leaf_nodes_total": int(sum(leaves)),
            "split_count_histogram": {str(k): int(v) for k, v in zip(*np.unique(splits, return_counts=True))},
            "zero_split_trees": int(sum(s == 0 for s in splits))}


def check_fit(booster, record: dict, g_config: dict, n_rows: int) -> dict:
    """Fail closed. ``g_config`` is the ORIGINAL G configuration (depth 3/4); returns the tree structure."""
    params, rounds = nx.booster_params(g_config)
    want = dict(params, max_depth=STUMP_DEPTH)
    problems = []
    if record.get("params") != want:
        problems.append("recorded params differ from booster_params(G) with only max_depth=1")
    tparam = record.get("resolved_config", {}).get("learner", {}).get("gradient_booster", {}).get(
        "tree_train_param", {})
    if tparam.get("max_depth") != str(STUMP_DEPTH):
        problems.append(f"fit-time resolved max_depth {tparam.get('max_depth')!r} != '1'")
    if record.get("base_score") != "5E-1" or nx.base_score(booster) != "5E-1":
        problems.append(f"base_score {record.get('base_score')!r} != '5E-1'")
    if nx.num_class(booster) != nx.N_CLASSES:
        problems.append("class axis is not four")
    if booster.num_boosted_rounds() != rounds or record.get("rounds_total") != rounds:
        problems.append(f"rounds {booster.num_boosted_rounds()} != configured {rounds}")
    if "sample_weight" in record or "base_margin" in record or nx.is_margin_marked(booster):
        problems.append("a sample weight or base margin is present")
    if record.get("rows") != n_rows:
        problems.append("fit record rows differ from the FIT rows")
    structure = tree_structure(booster)
    if structure["trees"] != rounds * nx.N_CLASSES:
        problems.append(f"{structure['trees']} saved trees != {rounds} rounds x 4")
    if structure["trees_depth_gt_1"]:
        problems.append(f"{structure['trees_depth_gt_1']} saved trees have depth > 1 "
                        f"(max {structure['max_depth']})")
    if problems:
        raise GateError("stump root fit check failed: " + "; ".join(problems))
    return structure


def fit_stump_root(data: dict, g_config: str):
    """The single D47 fit: fresh stump root on the original fitting rows (no weight, no margin)."""
    X_fit, y_fit, _, _ = data["FIT"]
    return nx.fit_global(X_fit, y_fit, stump_config(g_config))


# ---------------------------------------------------------------- scoring (D46 helpers, local arms)

def _share(b: dict) -> dict:
    c = b["crisis"]
    n = c["tp"] + c["fp"] + c["fn"] + c["tn"]
    return b | {"crisis_call_share": (c["tp"] + c["fp"]) / n if n else None}


def score_part(frame: pd.DataFrame) -> dict:
    """All keys: original vs stump; matched non-missing-persistence keys: plus persistence (no log loss)."""
    if len(frame) == 0:
        return {"n": 0, "status": "no_data"}
    t = frame["truth"].to_numpy()
    prob = {a: frame[[f"p_{a}_{lab}" for lab in LABELS]].to_numpy(np.float64) for a in ARMS}
    out = {"n": int(len(t)), "crisis_prevalence": float(rr.crisis(t).mean()),
           "all": {a: _share(cw.block(t, frame[f"y_{a}"].to_numpy(), prob[a])) for a in ARMS}}
    for a, b in ARM_DELTAS:
        out["all"][f"{a}_minus_{b}"] = cw.delta(out["all"][a], out["all"][b])
    out["ranking"] = {a: cw.ranking(t, prob[a]) for a in ARMS}
    k = np.isfinite(frame["persistence_code"].to_numpy(float))
    matched = {"n": int(k.sum()), "coverage": float(k.mean())}
    if k.any():
        pers = frame["persistence_code"].to_numpy(float)[k].astype(np.int64)
        for a in ARMS:
            matched[a] = _share(cw.block(t[k], frame[f"y_{a}"].to_numpy()[k], prob[a][k]))
        matched["persistence"] = _share(cw.block(t[k], pers, np.eye(nx.N_CLASSES)[pers], logloss=False))
        for a, b in PERS_DELTAS + ARM_DELTAS:
            matched[f"{a}_minus_{b}"] = cw.delta(matched[a], matched[b])
    out["matched_persistence"] = matched
    return out


def _wins(entries: list, pairs) -> dict:
    """Descriptive per-fold sign counts of exact crisis-F1 deltas (no test)."""
    out = {}
    for a, b in pairs:
        d = [Fraction(e[f"{a}_minus_{b}"]["crisis_f1_delta_exact"]) for e in entries]
        out[f"{a}_minus_{b}"] = {"positive": sum(x > 0 for x in d), "negative": sum(x < 0 for x in d),
                                 "zero": sum(x == 0 for x in d), "folds": len(d)}
    return out


def _pooled(blocks: dict, pairs) -> dict:
    p = cw._pool(blocks)
    for m in blocks:
        p[m] = _share(p[m])
    return p | cw._pooled_deltas(p, pairs)


def _mean_fold(entries: list, models, pairs) -> dict:
    out = cw._mean_fold(entries, models, pairs)
    for m in models:
        out[m]["crisis_call_share"] = float(np.mean([e[m]["crisis_call_share"] for e in entries]))
    return out


def aggregate(per_root: dict, select) -> dict:
    """Pooled (summed confusions, row-weighted losses) and mean-fold kept separate; AUC/AP mean-fold only."""
    out = {}
    for part in PARTS:
        have = [r["scores"][part] for r in per_root.values() if select(r) and r["scores"][part].get("n", 0)]
        if not have:
            out[part] = {"folds_with_data": 0, "status": "no_data"}
            continue
        rows = int(sum(h["n"] for h in have))
        res = {"role": ROLE[part], "folds_with_data": len(have), "rows": rows,
               "crisis_prevalence_pooled": float(sum(h["crisis_prevalence"] * h["n"] for h in have) / rows),
               "pooled_all": _pooled({a: [h["all"][a] for h in have] for a in ARMS}, ARM_DELTAS),
               "mean_fold_all": _mean_fold([h["all"] for h in have], ARMS, ARM_DELTAS),
               "fold_wins_all": _wins([h["all"] for h in have], ARM_DELTAS),
               "ranking_mean_fold": {}}
        for a in ARMS:
            el = [h["ranking"][a] for h in have if h["ranking"][a]["eligible"]]
            res["ranking_mean_fold"][a] = {"eligible_folds": len(el),
                                           "auc": float(np.mean([e["auc"] for e in el])) if el else None,
                                           "ap": float(np.mean([e["ap"] for e in el])) if el else None}
        mh = [h["matched_persistence"] for h in have if h["matched_persistence"]["n"]]
        if mh:
            models = ARMS + ("persistence",)
            res["matched_persistence"] = {
                "folds_with_data": len(mh), "rows": int(sum(h["n"] for h in mh)),
                "pooled": _pooled({m: [h[m] for h in mh] for m in models}, PERS_DELTAS + ARM_DELTAS),
                "mean_fold": _mean_fold(mh, models, PERS_DELTAS + ARM_DELTAS),
                "fold_wins": _wins(mh, PERS_DELTAS + ARM_DELTAS)}
        out[part] = res
    return out


# ---------------------------------------------------------------- per root

def part_frame(X, y, g, m, horizon, p_original, p_stump, phase_col) -> pd.DataFrame:
    frame = pd.DataFrame({"area": g, "target_month": m, "horizon": int(horizon), "truth": y,
                          "persistence_code": rr.persistence_codes(X[:, phase_col])})
    for arm, p in (("original", p_original), ("stump", p_stump)):
        frame[f"y_{arm}"] = fourclass.argmax_codes(p)
        for k, lab in enumerate(LABELS):
            frame[f"p_{arm}_{lab}"] = p[:, k]
    return frame


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
    check_defaults(defaults)
    booster, record = fit_stump_root(data, g_config)
    X_fit, y_fit, gfit, mfit = data["FIT"]
    structure = check_fit(booster, record, plan.G_CONFIGS[g_config], len(y_fit))
    rdir = out / name
    rdir.mkdir()
    (rdir / "stump_root.ubj").write_bytes(nx.raw(booster))
    frozen = nx.from_raw((rdir / "stump_root.ubj").read_bytes())
    if tree_structure(frozen) != structure:
        raise GateError("reloaded stump root tree structure differs from the fitted booster")
    original = nx.from_raw(root_ubj.read_bytes())          # sha gated in gate_root
    inputs = {"FIT": data["FIT"], "C": data["C"], "E3": data["E3"]}
    p_orig = {"FIT": nx.proba(original, X_fit), "C": replay["C"], "E3": replay["E3"]}
    rows, reload_checks = {}, {}
    for part in PARTS:
        X, y, g, m = inputs[part]
        if part == "FIT":
            m = month_label(m)
        ps = nx.proba(frozen, X)
        mism = sr.exact_mismatches(nx.proba(booster, X), ps)
        reload_checks[part] = {"mismatches": int(mism), "n": int(ps.size)}
        if mism:
            raise GateError(f"reloaded stump root does not reproduce its {part} probabilities ({mism} cells)")
        rows[part] = part_frame(X, y, g, m, h, p_orig[part], ps, phase_col)
        with gzip.open(rdir / f"rows_{part}.csv.gz", "wt", encoding="utf-8", newline="") as handle:
            rows[part].to_csv(handle, index=False, float_format="%.17g")
    o_index = data["window"][2]
    meta = {"root": name, "horizon": h, "target_month": root["target_month"], "g_config": g_config,
            "stump_config": stump_config(g_config), "origin_month": month_label(np.array([o_index]))[0],
            "fitting_rows": int(len(gfit)), "fitting_keys_sha256": nx.keys_sha(gfit, mfit),
            "original_root_source": str(root_ubj), "original_root_sha256": root["root_booster_sha256"],
            "stump_root_sha256": rid.file_sha256(rdir / "stump_root.ubj"), "tree_structure": structure,
            "reload_exact": reload_checks, "fit_record": record, "producer_rev": producer_rev}
    rid.write_json_atomic(rdir / "stump_root.json", meta)
    dc = rr.dev_baseline_check(rows["E3"], base, h)
    if dc["status"] == "checked" and (dc["joined"] != dc["e3_rows"] or dc["persistence_mismatches"]
                                      or dc["truth_mismatches"]):
        gate = {**gate, "passed": False, "error": f"dev_baselines cross-check failed: {dc}"}
    return gate, rows, meta | {"dev_baselines_check": dc}


def run_all(run, stage, out, roots, cands, producer_rev, phase_col, base) -> tuple[int, dict, dict]:
    """Sequential roots; the first gate / fit / scoring failure writes evidence and stops."""
    defaults = plan_defaults()
    gates, per_root = {}, {}
    for name in roots:
        pair = {cands[c]["e1"]: c for c in cands if cands[c]["root"] == name}
        try:
            gate, rows, meta = run_root(run, stage, out, name, pair, producer_rev, phase_col, base, defaults)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d34-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--producer-rev", default="7b2bf6f")
    args = parser.parse_args()
    run, out = args.d34_run.resolve(), args.out.resolve()
    if out.is_relative_to(run):
        raise ValueError("--out must not be inside the read-only D34 run")
    rid.refuse_existing(out, "D47 stump root")
    defaults = plan_defaults()
    producer = rid.code_identity_at(args.producer_rev)
    if rid.code_identity() != rid.code_identity_at("HEAD"):
        raise RuntimeError("D47 must run from committed package code (working tree differs from HEAD)")
    script_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE, capture_output=True, text=True,
                                   check=True).stdout.strip()
    roots, cands, ident = accept_mode(run, plan.E1PAIR, producer, args.producer_rev)
    stage = Path(ident["stage"])
    phase_col = load_schema(sr.SCHEMA)["ordered_features"].index("hist_phase_o00")
    base_path = run / "prepared" / "ledgers" / "dev_baselines.csv"   # dev targets only; never baselines.csv
    base = pd.read_csv(base_path)
    identity = {"stage": "d47_stump_root", "d34_run": str(run), "producer_rev": args.producer_rev,
                "producer_code": producer, "script_commit": script_commit, "script_code": rid.code_identity(),
                "runtime": rid.runtime_identity(), "max_month": "2020-12", "g_by_horizon": G_BY_H,
                "stump_rule": "dict(plan.G_CONFIGS[G], max_depth=1); no sample weight, no base margin",
                "stump_configs": {g: stump_config(g) for g in sorted(set(G_BY_H.values()))},
                "plan_defaults": defaults,
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
        "interpretation": ("Primary: same-key E3 crisis F1 (argmax -> code >= 2) per H, stump - original, and "
                           "stump / original - persistence on matched non-missing exact-origin keys (coverage "
                           "reported); crisis-call share, macro-F1, unweighted Brier and log loss alongside. "
                           "FIT is in-sample, C in-window interpolation (diagnostic). Depth changes capacity, "
                           "optimisation path and effective regularisation together; a smaller FIT-to-E3 gap is "
                           "not overfitting evidence. AUC/AP are within-root, mean-fold only, secondary. 21 "
                           "overlapping, repeatedly developed folds: fold wins are descriptive; no significance "
                           "test, no adoption rule."),
    }
    check_defaults(defaults)
    rid.write_json_atomic(out / "summary.json", summary)
    rid.write_json_atomic(out / "completion.json", {"status": "completed", "outputs": rid.output_hashes(out)})
    print(f"D47 stump root completed: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
