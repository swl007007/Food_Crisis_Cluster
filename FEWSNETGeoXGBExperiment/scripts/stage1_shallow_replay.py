#!/usr/bin/env python3
"""D33 / A8: predict-only depth-1 truncation replay of the six frozen D29 rootconf candidates.

python scripts/stage1_shallow_replay.py --d29-run D29_RUN --out NEW_DIR [--producer-rev ab1ac83]

Never fits. Rebuilds each root's S / C / E3 rows with the producer's data semantics
(app/main_model_GF.main, rootconf path), loads the frozen boosters, and compares three fixed
comparators per row: root, depth1 (s_branch truncated to "", "0", "1"; areas outside both
columns -> root, D32 semantics) and full (frozen terminal routing).

MANDATORY GATE first (gate.json): the reconstructed root/full probabilities equal every saved
value exactly (round-trip parse, float64, no tolerance) and all saved hard predictions and
routes agree on S, C and E3. If any check fails, only gate.json (+ identity) is written and no
summary is computed. The D29 run is read only; no E4 weights, maps or Stage 2 inputs are written.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from app.main_model_GF import assignment_evidence, stage1_split  # noqa: E402
from config import TRAIN_WINDOW_MONTHS  # noqa: E402
from scripts.stage1_rootconf_compare import accept_mode  # noqa: E402
from src.customize.customize import train_test_split_rolling_window  # noqa: E402
from src.experiment import plan  # noqa: E402
from src.feature.fourclass_features import load_schema, month_label  # noqa: E402
from src.helper.helper import get_X_branch_id_by_group  # noqa: E402
from src.metrics import fourclass  # noqa: E402
from src.model import native_xgb as nx  # noqa: E402
from src.utils import run_identity as rid  # noqa: E402
from src.utils.split import confirmation_split  # noqa: E402

STAGE = "stage1_rootconf"
DEPTH1_COLUMNS = ("", "0", "1")
LABELS = fourclass.CLASS_LABELS
SCHEMA = PACKAGE / "feature-schema.json"


class GateError(RuntimeError):
    pass


# ---------------------------------------------------------------- pure helpers (tested)

def truncate_s_branch(s_branch: pd.DataFrame) -> pd.DataFrame:
    """Keep only the root decision's columns ("", "0", "1"); deeper columns are dropped."""
    keep = [c for c in s_branch.columns if str(c) in DEPTH1_COLUMNS]
    return s_branch[keep]


def depth1_routes(groups, s_branch) -> np.ndarray:
    """Branch id per row under the truncated tree: "0"/"1" or "" (root) for any other area."""
    routes = get_X_branch_id_by_group(np.asarray(groups), truncate_s_branch(s_branch))
    if not set(np.unique(routes).tolist()) <= {"", "0", "1"}:
        raise RuntimeError("depth-1 routing produced a deeper branch")
    return routes


def check_root_decision(saved_log, decisions, root_sha) -> dict:
    """Every saved "0"/"1" entry belongs to the root decision (precedes any depth>=2 save),
    the root decision was accepted; returns final SHAs and the retained-parent (root copy) side."""
    root_dec = [d for d in decisions if (d.get("branch_id") or "") == ""]
    if len(root_dec) != 1 or root_dec[0].get("outcome") != "accepted":
        raise GateError(f"root decision not exactly one accepted decision: {root_dec}")
    names = [e["saved_as"] for e in saved_log]
    deep = [i for i, n in enumerate(names) if n != "root" and len(n) >= 2]
    first_deep = deep[0] if deep else len(names)
    last = {}
    for i, entry in enumerate(saved_log):
        if entry["saved_as"] in ("0", "1"):
            if i > first_deep:
                raise GateError(f"saved_log entry {i} ({entry['saved_as']}) is after a depth>=2 save")
            last[entry["saved_as"]] = entry.get("booster_sha256")
    if set(last) != {"0", "1"}:
        raise GateError(f"root decision did not save both sides: {sorted(last)}")
    copies = sorted(s for s, v in last.items() if v == root_sha)
    retained = sorted(str(i) for i, chosen in enumerate(root_dec[0].get("selected_children", [])) if not chosen)
    if copies != retained:
        raise GateError(f"root-copy sides {copies} differ from the root decision's retained-parent sides {retained}")
    return {"sha": last, "root_copy_sides": copies, "selected_children": root_dec[0].get("selected_children")}


def exact_mismatches(saved, rebuilt) -> int:
    """Count of unequal float64 cells (no tolerance; NaN never equal)."""
    a, b = np.asarray(saved, dtype=np.float64), np.asarray(rebuilt, dtype=np.float64)
    if a.shape != b.shape:
        return max(a.size, b.size, 1)
    return int(np.sum(~(a == b)))


def row_equal(saved, rebuilt) -> np.ndarray:
    """Per-row exact float64 equality of all four class probabilities (historical check only)."""
    return (np.asarray(saved, dtype=np.float64) == np.asarray(rebuilt, dtype=np.float64)).all(axis=1)


def flips(truth, a, b) -> dict:
    t, pa, pb = fourclass.crisis(truth), fourclass.crisis(a), fourclass.crisis(b)
    ra, rb = pa == t, pb == t
    return {"corrected": int(np.sum(~ra & rb)), "spoiled": int(np.sum(ra & ~rb)),
            "new_tp": int(np.sum((t == 1) & (pa == 0) & (pb == 1))),
            "lost_tp": int(np.sum((t == 1) & (pa == 1) & (pb == 0))),
            "new_fp": int(np.sum((t == 0) & (pa == 0) & (pb == 1))),
            "removed_fp": int(np.sum((t == 0) & (pa == 1) & (pb == 0))),
            "fourclass_changed": int(np.sum(np.asarray(a) != np.asarray(b)))}


def metrics(truth, pred) -> dict:
    f = fourclass.crisis_f1_exact(truth, pred)
    return {"confusion_fourclass": fourclass.confusion(truth, pred).astype(int).tolist(),
            "crisis": fourclass.crisis_counts(truth, pred), "crisis_f1_exact": str(f), "crisis_f1": float(f),
            "macro_f1_fourclass": fourclass.macro_f1(truth, pred)}


# ---------------------------------------------------------------- data reconstruction

def rebuild(run: Path, root: dict):
    """Producer data of one rootconf root (main_model_GF.main semantics)."""
    horizon = int(root["horizon"])
    term = pd.Period(root["target_month"], freq="M")
    features = load_schema(SCHEMA)["ordered_features"]
    snap = pd.read_parquet(run / "prepared" / f"snapshot_h{horizon}.parquet")
    if not (snap["horizon"] == horizon).all() or list(snap.columns[-len(features):]) != features:
        raise GateError("snapshot horizon / feature order differs from the producer contract")
    snap = snap.sort_values(["area", "target_month"]).reset_index(drop=True)
    X = snap[features].to_numpy(dtype=float)
    y = snap["class_code"].to_numpy(dtype=np.int64)
    groups = snap["area"].to_numpy(dtype=np.int64)
    months = snap["target_month"].to_numpy(dtype=np.int64)
    dates = pd.to_datetime(pd.Series(month_label(months)) + "-01")
    split = train_test_split_rolling_window(
        X, y, snap[["lat", "lon"]].to_numpy(dtype=float), groups, dates.dt.year.to_numpy(), dates,
        test_month=term, active_lag=horizon, train_window_months=TRAIN_WINDOW_MONTHS,
        admin_codes=np.arange(len(snap)))
    Xtrain, ytrain, _, gtrain, Xtest, ytest, _, gtest, idx_train, idx_test = split
    mtrain = months[idx_train]
    origin = term - horizon
    o_index = int(origin.year * 12 + origin.month - 1)
    if len(idx_test) == 0 or mtrain.min() < o_index - plan.WINDOW or mtrain.max() >= o_index:
        raise GateError("reconstructed fitting window / target rows violate the producer's [O-59, O) contract")
    x_set, _, _ = stage1_split(root["ratio"], gtrain, mtrain, o_index, int(root["split_seed"]), horizon, str(term))
    orig_val = np.flatnonzero(x_set == 1)
    conf = np.zeros(len(x_set), dtype=bool)
    conf[orig_val[confirmation_split(gtrain[orig_val], mtrain[orig_val], plan.CONFIRMATION_SEED) == 1]] = True
    s_rows = (x_set == 1) & ~conf
    keep = ~conf
    return {
        "S": (Xtrain[s_rows], ytrain[s_rows], gtrain[s_rows], month_label(mtrain[s_rows])),
        "C": (Xtrain[conf], ytrain[conf], gtrain[conf], month_label(mtrain[conf])),
        "E3": (Xtest, ytest, gtest, np.array([str(term)] * len(gtest))),
        "search": (gtrain[keep], x_set[keep]), "gtest": gtest, "conf_groups": gtrain[conf],
        "fitting_keys_sha256": nx.keys_sha(gtrain[x_set == 0], mtrain[x_set == 0]),
    }


class Boosters:
    def __init__(self, ckpt: Path):
        self.ckpt, self.cache = ckpt, {}

    def get(self, branch):
        key = branch or "root"
        if key not in self.cache:
            self.cache[key] = nx.from_raw((self.ckpt / f"xgb_{key}.ubj").read_bytes())
        return self.cache[key]

    def proba(self, X, routes):
        out = np.zeros((len(X), fourclass.N_CLASSES))
        for b in np.unique(routes):
            rows = np.flatnonzero(routes == b)
            out[rows] = nx.proba(self.get(b), X[rows])
        return out


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip", dtype={"branch_id": str}, keep_default_na=False)


def route_str(routes):
    return np.where(routes == "", "root", routes.astype(str))


def replay_candidate(run: Path, stage: Path, root_name: str, cand: str):
    root = json.loads((stage / "roots" / root_name / "root.json").read_text(encoding="utf-8"))
    cdir, ckpt = stage / "candidates" / cand, stage / "checkpoints" / cand
    record = json.loads((cdir / "candidate.json").read_text(encoding="utf-8"))
    with open(cdir / "s_branch.pkl", "rb") as handle:
        s_branch = pickle.load(handle)
    data = rebuild(run, root)
    gate = {"candidate": cand, "checks": {}}

    def check(name, n_bad, n):
        gate["checks"][name] = {"mismatches": int(n_bad), "n": int(n)}

    check("fitting_keys_sha256", int(data["fitting_keys_sha256"] != root.get("fitting_keys_sha256")), 1)
    saved_log = record["fits"]["saved_log"]
    last = {e["saved_as"]: e.get("booster_sha256") for e in saved_log}
    root_sha = root.get("root_booster_sha256") or last.get("root")
    used = {p.name: rid.file_sha256(p) for p in sorted(ckpt.glob("xgb_*.ubj"))}
    check("booster_file_sha_equals_last_saved",
          sum(used.get(f"xgb_{b}.ubj") != s for b, s in last.items()), len(last))
    check("root_booster_sha", int(used.get("xgb_root.ubj") != root_sha), 1)
    rootdec = check_root_decision(saved_log, record["partition"]["decisions"], root_sha)
    boosters = Boosters(ckpt)
    rows, saved_files = {}, {"S": "validation_predictions.csv.gz", "C": "confirmation_predictions.csv.gz",
                             "E3": "target_predictions.csv"}
    for part in ("S", "C", "E3"):
        X, y, g, m = data[part]
        full_r, d1_r = get_X_branch_id_by_group(g, s_branch), depth1_routes(g, s_branch)
        p_root = boosters.proba(X, np.full(len(X), "", dtype=full_r.dtype))
        p_full, p_d1 = boosters.proba(X, full_r), boosters.proba(X, d1_r)
        frame = pd.DataFrame({"area": g, "target_month": m, "horizon": int(root["horizon"]), "truth": y,
                              "y_root": fourclass.argmax_codes(p_root), "y_depth1": fourclass.argmax_codes(p_d1),
                              "y_full": fourclass.argmax_codes(p_full),
                              "route_depth1": route_str(d1_r), "route_full": route_str(full_r)})
        for name, p in (("root", p_root), ("depth1", p_d1), ("full", p_full)):
            for k, lab in enumerate(LABELS):
                frame[f"p_{name}_{lab}"] = p[:, k]
        frame["probabilities_source"] = "replay_derived"   # no historical depth1 probabilities exist
        frame["p_root_equals_saved"] = pd.Series([pd.NA] * len(frame), dtype="boolean")   # C/E3 only
        frame["p_full_equals_saved"] = pd.Series([pd.NA] * len(frame), dtype="boolean")
        saved = read_csv(cdir / saved_files[part])
        n = len(frame)
        if part == "E3":
            saved_root = read_csv(stage / "roots" / root_name / "root_target_predictions.csv")
            check("E3_keys", int(len(saved) != n) or int(np.sum(saved["FEWSNET_admin_code"].to_numpy() != g)), n)
            check("E3_truth", np.sum(saved["y_true_code"].to_numpy() != y), n)
            check("E3_y_full", np.sum(saved["y_pred_partitioned_code"].to_numpy() != frame["y_full"].to_numpy()), n)
            check("E3_y_root", np.sum(saved["y_pred_pooled_code"].to_numpy() != frame["y_root"].to_numpy()), n)
            check("E3_route_full", np.sum(saved["branch_id"].astype(str).to_numpy() != frame["route_full"].to_numpy()), n)
            check("E3_p_full", exact_mismatches(saved[[f"p_partitioned_{l}" for l in LABELS]], p_full), 4 * n)
            check("E3_p_root", exact_mismatches(saved_root[[f"p_pooled_{l}" for l in LABELS]], p_root), 4 * n)
            if len(saved) == n and len(saved_root) == n:
                frame["p_full_equals_saved"] = row_equal(saved[[f"p_partitioned_{l}" for l in LABELS]], p_full)
                frame["p_root_equals_saved"] = row_equal(saved_root[[f"p_pooled_{l}" for l in LABELS]], p_root)
            check("E3_root_keys", int(len(saved_root) != n) or
                  int(np.sum(saved_root["FEWSNET_admin_code"].to_numpy() != g)), n)
        else:
            ok_keys = len(saved) == n and (saved["area"].to_numpy() == g).all() and \
                (saved["target_month"].astype(str).to_numpy() == m).all()
            check(f"{part}_keys", int(not ok_keys), n)
            if ok_keys:
                check(f"{part}_truth", np.sum(saved["y_true"].to_numpy() != y), n)
                check(f"{part}_y_root", np.sum(saved["y_root"].to_numpy() != frame["y_root"].to_numpy()), n)
                check(f"{part}_y_full", np.sum(saved["y_final"].to_numpy() != frame["y_full"].to_numpy()), n)
                check(f"{part}_route_full",
                      np.sum(saved["branch_id"].astype(str).to_numpy() != frame["route_full"].to_numpy()), n)
                if part == "C":
                    check("C_p_root", exact_mismatches(saved[[f"p_root_{l}" for l in LABELS]], p_root), 4 * n)
                    check("C_p_full", exact_mismatches(saved[[f"p_final_{l}" for l in LABELS]], p_full), 4 * n)
                    frame["p_root_equals_saved"] = row_equal(saved[[f"p_root_{l}" for l in LABELS]], p_root)
                    frame["p_full_equals_saved"] = row_equal(saved[[f"p_final_{l}" for l in LABELS]], p_full)
        rows[part] = frame
    gate["passed"] = all(c["mismatches"] == 0 for c in gate["checks"].values())
    evidence = assignment_evidence(data["search"][0], data["search"][1], data["gtest"], s_branch,
                                   conf_groups=data["conf_groups"])
    d1_side = pd.Series(route_str(depth1_routes(evidence["FEWSNET_admin_code"].to_numpy(), s_branch)),
                        index=evidence.index)
    meta = {"root": root_name, "horizon": int(root["horizon"]), "target_month": root["target_month"],
            "booster_sha256": {"root": root_sha, **rootdec["sha"]},
            "retained_parent_root_copy_sides": rootdec["root_copy_sides"],
            "depth1_sides": {side: {"areas": int((d1_side == side).sum()),
                                    "rows": {p: int((rows[p]["route_depth1"] == side).sum()) for p in rows},
                                    "d32_status_areas": evidence.loc[d1_side == side, "assignment_status"]
                                    .value_counts().sort_index().astype(int).to_dict()}
                             for side in ("0", "1", "root")},
            "checkpoints_used": {k: v for k, v in used.items()
                                 if k in ("xgb_root.ubj", "xgb_0.ubj", "xgb_1.ubj") or
                                 k[4:-4] in set(np.unique(np.concatenate([rows[p]["route_full"] for p in rows])))}}
    return gate, rows, meta


def summarize(rows: dict) -> dict:
    out = {}
    for part, frame in rows.items():
        t = frame["truth"].to_numpy()
        preds = {c: frame[f"y_{c}"].to_numpy() for c in ("root", "depth1", "full")}
        block = {c: metrics(t, p) for c, p in preds.items()}
        for a, b in (("root", "depth1"), ("depth1", "full"), ("root", "full")):
            fa, fb = Fraction(block[a]["crisis_f1_exact"]), Fraction(block[b]["crisis_f1_exact"])
            block[f"{b}_minus_{a}"] = {"crisis_f1_delta_exact": str(fb - fa), "crisis_f1_delta": float(fb - fa),
                                       "macro_f1_fourclass_delta": block[b]["macro_f1_fourclass"] - block[a]["macro_f1_fourclass"],
                                       "row_flips_binary": flips(t, preds[a], preds[b])}
        block["n"] = int(len(t))
        out[part] = block
    out["interpretation"] = ("All probabilities are replay-derived; root/full C and E3 probabilities were checked "
                             "exactly against saved values (p_*_equals_saved), depth1 has no historical "
                             "probabilities and S has saved hard labels only. S: adaptive reuse, descriptive; C: diagnostic; "
                             "E3: exposed development target, descriptive. Six correlated candidates; no selection.")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d29-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--producer-rev", default="ab1ac83")
    args = parser.parse_args()
    run, out = args.d29_run.resolve(), args.out.resolve()
    if out.is_relative_to(run):
        raise ValueError("--out must not be inside the read-only D29 run")
    rid.refuse_existing(out, "D33 shallow replay")
    producer = rid.code_identity_at(args.producer_rev)
    if rid.code_identity() != rid.code_identity_at("HEAD"):
        raise RuntimeError("replay must run from committed package code (working tree differs from HEAD)")
    import subprocess
    replay_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE, capture_output=True, text=True,
                                   check=True).stdout.strip()
    roots, cands, ident = accept_mode(run, plan.ROOTCONF, producer, args.producer_rev)
    stage = run / STAGE
    out.mkdir(parents=True)
    identity = {"stage": "d33_shallow_replay", "d29_run": str(run), "producer_rev": args.producer_rev,
                "producer_code": producer, "replay_commit": replay_commit, "replay_code": rid.code_identity(),
                "runtime": rid.runtime_identity(),
                "reconstruction_source_equivalence": "research/d33-source-equivalence.md (ab1ac83 vs replay commit)",
                "acceptance": {k: v for k, v in ident.items() if k != "stage"}, "inputs": {}}
    gates, results = {}, {}
    for cand, entry in sorted(cands.items()):
        root_name = entry["root"]
        h = roots[root_name]["horizon"]
        identity["inputs"][cand] = {
            "snapshot": rid.file_sha256(run / "prepared" / f"snapshot_h{h}.parquet"),
            "membership": rid.file_sha256(stage / "roots" / root_name / "fold_membership.csv.gz"),
            "s_branch": rid.file_sha256(stage / "candidates" / cand / "s_branch.pkl"),
            "schema": rid.file_sha256(SCHEMA),
            **{f: rid.file_sha256(stage / "candidates" / cand / f) for f in
               ("candidate.json", "validation_predictions.csv.gz", "confirmation_predictions.csv.gz",
                "target_predictions.csv")},
            **{f: rid.file_sha256(stage / "roots" / root_name / f) for f in ("root.json", "root_target_predictions.csv")}}
        try:
            gate, rows, meta = replay_candidate(run, stage, root_name, cand)
        except GateError as exc:
            gate, rows, meta = {"candidate": cand, "passed": False, "error": str(exc)}, None, None
        identity["inputs"][cand]["checkpoints_used"] = (meta or {}).get("checkpoints_used")
        gates[cand] = gate
        results[cand] = (rows, meta)
    passed = all(g["passed"] for g in gates.values())
    rid.write_json_atomic(out / "gate.json", {"passed": passed, "rule": "exact float64 equality, round-trip parse",
                                              "candidates": gates})
    rid.write_json_atomic(out / "identity.json", identity)
    if not passed:
        print("GATE FAILED: no shallow output computed; see gate.json")
        return 2
    summary = {}
    for cand, (rows, meta) in results.items():
        cd = out / cand
        cd.mkdir()
        for part, frame in rows.items():
            with gzip.open(cd / f"rows_{part}.csv.gz", "wt", encoding="utf-8", newline="") as handle:
                frame.to_csv(handle, index=False, float_format="%.17g")
        summary[cand] = {"depth1_metadata": {k: v for k, v in meta.items() if k != "checkpoints_used"},
                         **summarize(rows)}
    rid.write_json_atomic(out / "summary.json", summary)
    rid.write_json_atomic(out / "completion.json", {"status": "completed", "outputs": rid.output_hashes(out)})
    print(f"D33 replay completed: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
