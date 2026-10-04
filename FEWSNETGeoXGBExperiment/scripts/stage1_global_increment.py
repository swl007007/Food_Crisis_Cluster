#!/usr/bin/env python3
"""D35 / A10: fixed global L1 +20-round continuation of the 21 saved D34 roots.

python scripts/stage1_global_increment.py --d34-run D34_RUN --out NEW_DIR [--producer-rev 7b2bf6f]

Per root (sequential): rebuild the producer's r80 fitting / S / C / E3 rows from the frozen
snapshot (Parquet filter target_month <= 2020-12), verify keys/truth/roles against D34
fold_membership.csv.gz and the fitting-key hash, load the saved root booster and GATE: the
replayed root reproduces the saved hard labels (S, C, E3) and C/E3 probabilities exactly for
both paired candidates. Only then ONE native continue_booster(root, X_fit, y_fit, L1) on the
original fitting rows (no S/C/target rows), saved as global_plus20.ubj with its continuation
record. After freezing, global+20 is scored on the same S/C/E3 keys next to the frozen D34
root / hard-local / Brier-local predictions (read from the saved candidate files, never
regenerated). Any gate failure stops that root and suppresses summary.json.

The D34 run is read only. No E4 weights, maps or Stage 2 inputs are written. C/E3 same-key
results are primary, S descriptive; search_rows strata describe D32 search support only.
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
from scripts import stage1_shallow_replay as sr  # noqa: E402
from scripts.stage1_rootconf_compare import accept_mode  # noqa: E402
from src.experiment import plan  # noqa: E402
from src.feature.fourclass_features import month_label  # noqa: E402
from src.metrics import fourclass  # noqa: E402
from src.model import native_xgb as nx  # noqa: E402
from src.utils import run_identity as rid  # noqa: E402

MAX_MONTH = 2020 * 12 + 11          # 2020-12 month index; no 2021+ rows are read
LABELS = fourclass.CLASS_LABELS
MODELS = ("root", "global20", "hard_local", "brier_local")
DELTAS = (("global20", "root"), ("hard_local", "global20"), ("brier_local", "global20"))
PARTS = ("S", "C", "E3")
STRATA = ("search_rows>0", "search_rows==0")
GateError = sr.GateError


# ---------------------------------------------------------------- pure helpers (tested)

def fit_global_plus20(root_booster, data: dict):
    """The single D35 fit: L1 continuation of the root on the original r80 fitting rows only."""
    X_fit, y_fit, _, _ = data["FIT"]
    child, record = nx.continue_booster(root_booster, X_fit, y_fit, plan.L_CONFIGS["L1"])
    if record["rounds_added"] != 20 or record["parent_rounds"] != root_booster.num_boosted_rounds():
        raise RuntimeError(f"continuation did not add exactly 20 rounds: {record['rounds_added']}")
    return child, record


def key_set(areas, months) -> set:
    return set(zip(np.asarray(areas, dtype=np.int64).tolist(), [str(m) for m in months]))


def block(truth, pred) -> dict:
    return sr.metrics(truth, pred) | {"n": int(len(truth))}


def score_frame(frame: pd.DataFrame) -> dict:
    """Metrics of the four comparators plus the three deltas on one same-key frame."""
    if len(frame) == 0:
        return {"n": 0, "status": "no_data"}
    t = frame["truth"].to_numpy()
    out = {m: block(t, frame[f"y_{m}"].to_numpy()) for m in MODELS}
    for a, b in DELTAS:
        fa, fb = Fraction(out[a]["crisis_f1_exact"]), Fraction(out[b]["crisis_f1_exact"])
        out[f"{a}_minus_{b}"] = {"crisis_f1_delta_exact": str(fa - fb), "crisis_f1_delta": float(fa - fb),
                                 "macro_f1_fourclass_delta": out[a]["macro_f1_fourclass"] - out[b]["macro_f1_fourclass"]}
    out["n"] = int(len(t))
    return out


def score_part(frame: pd.DataFrame) -> dict:
    """All rows plus the two search_rows strata (empty stratum reported as no data)."""
    return {"all": score_frame(frame),
            "strata": {s: score_frame(frame[frame["stratum"] == s]) for s in STRATA}}


def pooled_from_confusions(confs: dict) -> dict:
    """Pooled (summed four-class confusion) crisis/macro F1 per model and deltas."""
    out = {}
    for m, mat in confs.items():
        mat = np.asarray(mat, dtype=np.int64)
        tp, fp, fn = int(mat[2:, 2:].sum()), int(mat[:2, 2:].sum()), int(mat[2:, :2].sum())
        f = Fraction(2 * tp, 2 * tp + fp + fn) if (2 * tp + fp + fn) else Fraction(0)
        out[m] = {"confusion_fourclass": mat.tolist(), "crisis": {"tp": tp, "fp": fp, "fn": fn},
                  "crisis_f1_exact": str(f), "crisis_f1": float(f),
                  "macro_f1_fourclass": fourclass.macro_f1_from_matrix(mat)}
    for a, b in DELTAS:
        d = Fraction(out[a]["crisis_f1_exact"]) - Fraction(out[b]["crisis_f1_exact"])
        out[f"{a}_minus_{b}"] = {"crisis_f1_delta_exact": str(d), "crisis_f1_delta": float(d)}
    return out


def aggregate(per_root: dict, select) -> dict:
    """Mean-fold deltas (folds with data only) and pooled confusions, kept separate."""
    out = {}
    for part in PARTS:
        out[part] = {}
        for layer in ("all",) + STRATA:
            blocks = [(r["scores"][part]["all"] if layer == "all" else r["scores"][part]["strata"][layer])
                      for k, r in per_root.items() if select(r)]
            have = [b for b in blocks if b.get("n", 0) > 0]
            if not have:
                out[part][layer] = {"folds_with_data": 0, "folds": len(blocks), "status": "no_data"}
                continue
            mean = {f"{a}_minus_{b}": float(np.mean([h[f"{a}_minus_{b}"]["crisis_f1_delta"] for h in have]))
                    for a, b in DELTAS}
            confs = {m: np.sum([np.asarray(h[m]["confusion_fourclass"]) for h in have], axis=0) for m in MODELS}
            out[part][layer] = {"folds_with_data": len(have), "folds": len(blocks),
                                "rows": int(sum(h["n"] for h in have)),
                                "mean_fold_crisis_f1_delta": mean, "pooled": pooled_from_confusions(confs)}
    return out


# ---------------------------------------------------------------- per root

def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip", dtype={"branch_id": str, "target_month": str},
                       keep_default_na=False)


def run_root(run: Path, stage: Path, out: Path, name: str, pair: dict, producer_rev: str):
    """(gate, rows, meta). Rows/meta are None when the gate fails (no fit is made)."""
    root = json.loads((stage / "roots" / name / "root.json").read_text(encoding="utf-8"))
    gate = {"root": name, "checks": {}}

    def check(key, n_bad, n):
        gate["checks"][key] = {"mismatches": int(n_bad), "n": int(n)}

    data = sr.rebuild(run, root, max_month=MAX_MONTH, with_fitting=True)
    _, _, gfit, mfit = data["FIT"]
    check("fitting_keys_sha256", int(nx.keys_sha(gfit, mfit) != root.get("fitting_keys_sha256")), 1)
    saved_m = read_csv(stage / "roots" / name / "fold_membership.csv.gz")
    mine = data["membership"]
    same = len(saved_m) == len(mine) and all(
        (saved_m[c].astype(str).to_numpy() == mine[c].astype(str).to_numpy()).all()
        for c in ("area", "target_month", "role", "class_code"))
    check("membership_keys_truth_roles", int(not same), len(mine))
    fit_keys = key_set(gfit, month_label(mfit))
    held = set()
    for part in PARTS:
        held |= key_set(data[part][2], data[part][3])
    check("fitting_excludes_S_C_target", len(fit_keys & held), len(fit_keys))
    lo, hi, o_index = data["window"]
    check("window_[O-59,O)", int(lo < o_index - plan.WINDOW or hi >= o_index), 1)
    all_months = np.concatenate([mfit] + [np.array([int(m[:4]) * 12 + int(m[5:]) - 1 for m in data[p][3]],
                                                   dtype=np.int64) for p in PARTS])
    check("all_keys_le_2020-12", int(np.sum(all_months > MAX_MONTH)), len(all_months))

    root_sha = root["root_booster_sha256"]
    hard, brier = pair["hard_f1"], pair["brier_crisis"]
    ubj = {c: stage / "checkpoints" / c / "xgb_root.ubj" for c in (hard, brier)}
    check("root_booster_sha", sum(rid.file_sha256(p) != root_sha for p in ubj.values()), 2)
    root_bytes = ubj[hard].read_bytes()
    booster = nx.from_raw(root_bytes)
    replay = {p: nx.proba(booster, data[p][0]) for p in PARTS}
    y_rep = {p: fourclass.argmax_codes(replay[p]) for p in PARTS}

    local = {}
    for e1, cand in (("hard", hard), ("brier", brier)):
        cdir = stage / "candidates" / cand
        frames = {"S": read_csv(cdir / "validation_predictions.csv.gz"),
                  "C": read_csv(cdir / "confirmation_predictions.csv.gz")}
        for part in ("S", "C"):
            _, y, g, m = data[part]
            f, n = frames[part], len(y)
            ok = len(f) == n and (f["area"].to_numpy() == g).all() and (f["target_month"].to_numpy() == m).all()
            check(f"{e1}_{part}_keys", int(not ok), n)
            if not ok:
                continue
            check(f"{e1}_{part}_truth", np.sum(f["y_true"].to_numpy() != y), n)
            check(f"{e1}_{part}_y_root", np.sum(f["y_root"].to_numpy() != y_rep[part]), n)
            if part == "C":
                check(f"{e1}_C_p_root", sr.exact_mismatches(f[[f"p_root_{l}" for l in LABELS]], replay["C"]), 4 * n)
            local[(e1, part)] = f["y_final"].to_numpy()
        tgt = read_csv(cdir / "target_predictions.csv")
        _, y, g, _ = data["E3"]
        n = len(y)
        ok = len(tgt) == n and (tgt["FEWSNET_admin_code"].to_numpy() == g).all()
        check(f"{e1}_E3_keys", int(not ok), n)
        if ok:
            check(f"{e1}_E3_truth", np.sum(tgt["y_true_code"].to_numpy() != y), n)
            check(f"{e1}_E3_y_root", np.sum(tgt["y_pred_pooled_code"].to_numpy() != y_rep["E3"]), n)
            local[(e1, "E3")] = tgt["y_pred_partitioned_code"].to_numpy()
    saved_root = read_csv(stage / "roots" / name / "root_target_predictions.csv")
    n = len(data["E3"][1])
    ok = len(saved_root) == n and (saved_root["FEWSNET_admin_code"].to_numpy() == data["E3"][2]).all()
    check("E3_root_keys", int(not ok), n)
    if ok:
        check("E3_y_pred_pooled_code", np.sum(saved_root["y_pred_pooled_code"].to_numpy() != y_rep["E3"]), n)
        check("E3_p_pooled", sr.exact_mismatches(saved_root[[f"p_pooled_{l}" for l in LABELS]], replay["E3"]), 4 * n)
    evidence = pd.read_csv(stage / "candidates" / hard / "assignment_evidence.csv", dtype={"prediction_branch_id": str,
                                                                                          "spatial_partition_id": str})
    search = dict(zip(evidence["FEWSNET_admin_code"].astype(np.int64), evidence["search_rows"].astype(np.int64)))
    missing = sum(int(a) not in search for p in PARTS for a in np.unique(data[p][2]))
    check("search_rows_cover_all_areas", missing, 1)
    gate["passed"] = all(c["mismatches"] == 0 for c in gate["checks"].values())
    if not gate["passed"]:
        return gate, None, None

    # -------- single fit, freeze, then score
    child, record = fit_global_plus20(booster, data)
    rdir = out / name
    rdir.mkdir()
    (rdir / "global_plus20.ubj").write_bytes(nx.raw(child))
    frozen = nx.from_raw((rdir / "global_plus20.ubj").read_bytes())
    cont = {"root": name, "root_booster_source": str(ubj[hard]), "root_booster_sha256": root_sha,
            "fitting_keys_sha256": nx.keys_sha(gfit, mfit), "fitting_rows": int(len(gfit)),
            "local_config": "L1", "rounds_added": record["rounds_added"], "producer_rev": producer_rev,
            "global_plus20_sha256": rid.file_sha256(rdir / "global_plus20.ubj"), "continuation_record": record}
    rid.write_json_atomic(rdir / "continuation.json", cont)
    rows = {}
    for part in PARTS:
        X, y, g, m = data[part]
        p20 = nx.proba(frozen, X)
        sr_ = np.array([search[int(a)] for a in g], dtype=np.int64)
        frame = pd.DataFrame({"area": g, "target_month": m, "horizon": int(root["horizon"]), "truth": y,
                              "y_root": y_rep[part], "y_global20": fourclass.argmax_codes(p20),
                              "y_hard_local": local[("hard", part)], "y_brier_local": local[("brier", part)],
                              "search_rows": sr_, "stratum": np.where(sr_ > 0, STRATA[0], STRATA[1])})
        for k, lab in enumerate(LABELS):
            frame[f"p_root_{lab}"] = replay[part][:, k]
        for k, lab in enumerate(LABELS):
            frame[f"p_global20_{lab}"] = p20[:, k]
        frame["p_root_source"] = "replay_of_saved_root"
        frame["p_global20_source"] = "new_fit_d35"
        rows[part] = frame
        with gzip.open(rdir / f"rows_{part}.csv.gz", "wt", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False, float_format="%.17g")
    meta = {"horizon": int(root["horizon"]), "target_month": root["target_month"], "hard": hard, "brier": brier,
            "global_plus20_sha256": cont["global_plus20_sha256"], "fitting_rows": int(len(gfit))}
    return gate, rows, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d34-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--producer-rev", default="7b2bf6f")
    args = parser.parse_args()
    run, out = args.d34_run.resolve(), args.out.resolve()
    if out.is_relative_to(run):
        raise ValueError("--out must not be inside the read-only D34 run")
    rid.refuse_existing(out, "D35 global increment")
    producer = rid.code_identity_at(args.producer_rev)
    if rid.code_identity() != rid.code_identity_at("HEAD"):
        raise RuntimeError("D35 must run from committed package code (working tree differs from HEAD)")
    script_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE, capture_output=True, text=True,
                                   check=True).stdout.strip()
    roots, cands, ident = accept_mode(run, plan.E1PAIR, producer, args.producer_rev)
    stage = Path(ident["stage"])
    out.mkdir(parents=True)
    identity = {"stage": "d35_global_increment", "d34_run": str(run), "producer_rev": args.producer_rev,
                "producer_code": producer, "script_commit": script_commit, "script_code": rid.code_identity(),
                "runtime": rid.runtime_identity(), "max_month": "2020-12",
                "acceptance": {k: (str(v) if isinstance(v, Path) else v) for k, v in ident.items()},
                "inputs": {}}
    gates, per_root = {}, {}
    for name in roots:                                   # accept_mode order (canonical schedule)
        entry = roots[name]
        pair = {cands[c]["e1"]: c for c in cands if cands[c]["root"] == name}
        h = entry["horizon"]
        identity["inputs"][name] = {
            "snapshot": rid.file_sha256(run / "prepared" / f"snapshot_h{h}.parquet"),
            **{f: rid.file_sha256(stage / "roots" / name / f)
               for f in ("root.json", "fold_membership.csv.gz", "root_target_predictions.csv")},
            **{f"{c}/{f}": rid.file_sha256(stage / "candidates" / c / f) for c in pair.values()
               for f in ("candidate.json", "validation_predictions.csv.gz", "confirmation_predictions.csv.gz",
                         "target_predictions.csv", "assignment_evidence.csv")},
            **{f"{c}/xgb_root.ubj": rid.file_sha256(stage / "checkpoints" / c / "xgb_root.ubj") for c in pair.values()}}
        try:
            gate, rows, meta = run_root(run, stage, out, name, pair, args.producer_rev)
        except GateError as exc:
            gate, rows, meta = {"root": name, "passed": False, "error": str(exc)}, None, None
        gates[name] = gate
        print(f"{name}: gate {'passed' if gate['passed'] else 'FAILED'}", flush=True)
        if rows is not None:
            per_root[name] = {**meta, "scores": {p: score_part(rows[p]) for p in PARTS}}
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
        "interpretation": ("Primary: same-key C and E3; S was used by the local partition search and is "
                           "descriptive only. mean_fold deltas average per-root crisis F1 deltas over folds with "
                           "data; pooled sums four-class confusions. search_rows strata describe D32 search "
                           "support, not whether an increment model was used; empty strata are no data. 21 "
                           "overlapping, repeatedly developed folds: no significance test; no causal attribution."),
    }
    rid.write_json_atomic(out / "summary.json", summary)
    rid.write_json_atomic(out / "completion.json", {"status": "completed", "outputs": rid.output_hashes(out)})
    print(f"D35 global increment completed: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
