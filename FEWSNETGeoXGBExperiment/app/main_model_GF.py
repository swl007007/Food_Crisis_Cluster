#!/usr/bin/env python3
"""Stage 1: one shared-tree GeoXGBoost root and its four candidates (experiment-plan s.4).

Adapted from the four-class GeoRF entrypoint. One process = one root identity
(H, target T, selected G, split ratio, split seed): the global G booster is fitted ONCE
on the candidate's fitting rows, then the four candidate searches (L1/L2 x E2 threshold
family gt0/gt001) each partition from that same immutable root. Children only append
the chosen L rounds to their parent (D4); there is no RF, imputer or pseudo row (D14).

python app/main_model_GF.py --data SNAPSHOT --geometry-dir DIR --schema SCHEMA \
    --forecasting_scope N --desired_terms YYYY-MM --g-config G1 --ratio r80 --split-seed 42 \
    --checkpoint-dir SCRATCH_DIR

Run from a fresh working directory. Writes, in that directory:
  root.json, fold_membership.csv.gz, root_target_predictions.csv and, per candidate,
  <candidate>/{candidate.json, correspondence_table.csv, target_predictions.csv,
  heldout_scores.csv, s_branch.pkl, branch_table.npy, X_branch_id.npy}.
Booster checkpoints (root + every child) go to --checkpoint-dir (outside Dropbox).
"""
import argparse
import gzip
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

import numpy as np
import pandas as pd

import config
from config import GROUP_SPLIT, MAX_DEPTH, MIN_DEPTH, TRAIN_WINDOW_MONTHS, LAGS_MONTHS
from src.customize.customize import train_test_split_rolling_window
from src.experiment import plan
from src.feature.fourclass_features import load_schema, month_label
from src.helper.helper import get_X_branch_id_by_group
from src.metrics import fourclass
from src.model import native_xgb as nx
from src.model.GeoRF import GeoRF
from src.utils.lag_schedules import forecasting_scope_to_lag
from src.utils.split import group_aware_train_val_split

PARTITION_INFO_CUTOFF = plan.PARTITION_INFO_CUTOFF


def module_locations():
    """Every package module must resolve inside this package (R15)."""
    locations = {}
    for name, module in list(sys.modules.items()):
        if name in ("config", "config_visual") or name.split(".")[0] == "src":
            location = getattr(module, "__file__", None)
            if location:
                if not Path(location).resolve().is_relative_to(PACKAGE):
                    raise RuntimeError(f"{name} resolved outside the package: {location}")
                locations[name] = str(Path(location).resolve().relative_to(PACKAGE))
    return locations


def class_counts(y):
    return [int(np.sum(np.asarray(y) == k)) for k in range(fourclass.N_CLASSES)]


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run_candidate(name, local, family, root, data, work, checkpoint_dir, contiguity_info, features):
    """One partition search from the shared root; returns the candidate record."""
    Xtrain, ytrain, gtrain, mtrain, x_set, Xtest, ytest, gtest, y_pool = data
    cand_work = work / "georf" / name
    cand_work.mkdir(parents=True)
    here = os.getcwd()
    os.chdir(cand_work)
    try:
        model = GeoRF(min_model_depth=MIN_DEPTH, max_model_depth=MAX_DEPTH)
        fit_started = time.time()
        model.fit(Xtrain, ytrain, gtrain, X_set=x_set, split={"X_set": x_set},
                  contiguity_type="polygon", polygon_contiguity_info=contiguity_info,
                  feature_names=features, print_to_file=True, track_partition_metrics=False,
                  VIS_DEBUG_MODE=False, root=root, local_config=plan.L_CONFIGS[local],
                  threshold=plan.THRESHOLD_FAMILIES[family], X_month=mtrain)
        fit_seconds = time.time() - fit_started
        model_dir = Path(model.model_dir).resolve()
    finally:
        os.chdir(here)
    saved_branch = np.load(model_dir / "space_partitions" / "X_branch_id.npy", allow_pickle=False)
    if not np.array_equal(saved_branch, get_X_branch_id_by_group(gtrain, model.s_branch)):
        raise RuntimeError("saved X_branch_id disagrees with s_branch routing")
    routed_test = get_X_branch_id_by_group(gtest, model.s_branch)
    proba_part = model.model.predict_proba_georf(Xtest, gtest, model.s_branch, X_branch_id=routed_test)
    y_part = fourclass.argmax_codes(proba_part)
    part_summary, pool_summary = fourclass.summary(ytest, y_part), fourclass.summary(ytest, y_pool)

    branch_str = np.where(saved_branch == "", "root", saved_branch.astype(str))
    corr = pd.DataFrame({"FEWSNET_admin_code": gtrain, "partition_id": branch_str}).drop_duplicates()
    if corr["FEWSNET_admin_code"].duplicated().any():
        raise RuntimeError("an area received more than one terminal partition")
    corr = corr.sort_values("FEWSNET_admin_code")
    out = work / name
    out.mkdir()
    corr.to_csv(out / "correspondence_table.csv", index=False)
    lookup = dict(zip(corr["FEWSNET_admin_code"], corr["partition_id"]))
    routed_test_str = np.where(routed_test == "", "root", routed_test.astype(str))
    in_train = np.isin(gtest, list(lookup))
    if any(lookup[a] != b for a, b in zip(gtest[in_train], routed_test_str[in_train])):
        raise RuntimeError("test routing disagrees with the exported correspondence")
    preds = pd.DataFrame({"FEWSNET_admin_code": gtest, "y_true_code": ytest,
                          "y_pred_partitioned_code": y_part, "y_pred_pooled_code": y_pool,
                          "branch_id": routed_test_str,
                          "routing": np.where(in_train, "terminal_branch", "root_unassigned_test_area")})
    for k, label in enumerate(fourclass.CLASS_LABELS):
        preds[f"p_partitioned_{label}"] = proba_part[:, k]
    preds.to_csv(out / "target_predictions.csv", index=False, float_format="%.17g")
    pd.DataFrame([{"macro_f1": part_summary["macro_f1"], "macro_f1_base": pool_summary["macro_f1"],
                   "n": len(ytest)}]).to_csv(out / "heldout_scores.csv", index=False, float_format="%.17g")
    for fname in ("s_branch.pkl", "branch_table.npy", "X_branch_id.npy"):
        shutil.copy2(model_dir / "space_partitions" / fname, out / fname)

    # Checkpoints (root copy + every saved child) move to the scratch store.
    ckpt = Path(checkpoint_dir) / name
    shutil.copytree(model_dir / "checkpoints", ckpt)
    checkpoints = {p.name: nx_file_sha(p) for p in sorted(ckpt.iterdir())}
    terminal = sorted(set(branch_str.tolist()))
    last_save = {}
    for entry in model.model.saved_log:
        last_save[entry["saved_as"]] = entry
    record = {
        "candidate": name, "local_config": local, "threshold_family": family,
        "threshold": str(plan.THRESHOLD_FAMILIES[family]),
        "partition": {
            "terminal_partitions": terminal, "n_terminal": len(terminal),
            "accepted_splits": sum(d.get("outcome") == "accepted" for d in model.partition_decisions),
            "decisions": model.partition_decisions,
            "gate": (f"E2: strict fixed-four macro-F1 gain > {plan.THRESHOLD_FAMILIES[family]} over the "
                     "current parent on its complete validation rows; parent wins ties"),
            "terminal_checkpoint_records": {b: last_save.get(b) for b in terminal},
            "correspondence_source": "space_partitions/X_branch_id.npy == s_branch routing (checked)",
        },
        "scores": {"macro_f1": part_summary["macro_f1"], "macro_f1_base": pool_summary["macro_f1"],
                   "partitioned": part_summary, "pooled": pool_summary,
                   "pooled_source": "the candidate's own global root booster (same G, same fitting rows)",
                   "note": "E3 held-out target-month scores; NOT the E1/E2 validation scores"},
        "fits": {"fit_log": model.model.fit_log, "saved_log": model.model.saved_log,
                 "child_fits": sum(1 for e in model.model.fit_log if e.get("kind") == "continuation")},
        "checkpoints": {"dir": str(ckpt), "sha256": checkpoints},
        "timings": {"fit_seconds": round(fit_seconds, 2)},
    }
    write_json(out / "candidate.json", record)
    import logging
    for handler in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(handler)
        handler.close()
    shutil.rmtree(work / "georf" / name, ignore_errors=True)
    return record


def nx_file_sha(path):
    from src.utils.run_identity import file_sha256
    return file_sha256(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="snapshot_h{H}.parquet for this scope")
    parser.add_argument("--geometry-dir", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--forecasting_scope", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--desired_terms", required=True, help="single target month YYYY-MM")
    parser.add_argument("--g-config", required=True, choices=sorted(plan.G_CONFIGS))
    parser.add_argument("--ratio", required=True, choices=sorted(plan.SPLIT_RATIOS))
    parser.add_argument("--split-seed", type=int, required=True, choices=plan.SPLIT_SEEDS)
    parser.add_argument("--checkpoint-dir", required=True, help="scratch store for boosters (outside Dropbox)")
    args = parser.parse_args()
    started = time.time()
    work = Path.cwd()

    horizon = forecasting_scope_to_lag(args.forecasting_scope, LAGS_MONTHS)
    term = pd.Period(args.desired_terms, freq="M")
    if str(term) > PARTITION_INFO_CUTOFF:
        raise ValueError("Stage 1 targets must not exceed the 2020-12 partition information cutoff")
    if str(term) not in plan.STAGE1_TARGETS:
        raise ValueError(f"{term} is not a frozen Stage 1 candidate target")
    if TRAIN_WINDOW_MONTHS - 1 != plan.WINDOW:
        raise ValueError("config window is not the frozen 59-month label window")
    schema = load_schema(Path(args.schema))
    features = schema["ordered_features"]

    snap = pd.read_parquet(args.data)
    if not (snap["horizon"] == horizon).all():
        raise ValueError("snapshot horizon does not match the scope")
    if list(snap.columns[-len(features):]) != features:
        raise ValueError("snapshot feature order differs from the frozen schema")
    snap = snap.sort_values(["area", "target_month"]).reset_index(drop=True)
    X = snap[features].to_numpy(dtype=float)
    y = snap["class_code"].to_numpy(dtype=np.int64)
    groups = snap["area"].to_numpy(dtype=np.int64)
    months = snap["target_month"].to_numpy(dtype=np.int64)
    dates = pd.to_datetime(pd.Series(month_label(months)) + "-01")
    X_loc = snap[["lat", "lon"]].to_numpy(dtype=float)
    years = dates.dt.year.to_numpy()

    split = train_test_split_rolling_window(
        X, y, X_loc, groups, years, dates, test_month=term, active_lag=horizon,
        train_window_months=TRAIN_WINDOW_MONTHS, admin_codes=np.arange(len(snap)))
    Xtrain, ytrain, _, gtrain, Xtest, ytest, _, gtest, idx_train, idx_test = split
    origin = term - horizon
    o_index = int(origin.year * 12 + origin.month - 1)
    if len(idx_test) == 0:
        raise ValueError(f"{term} has no labelled target rows; it must be skipped at scheduling")
    mtrain = months[idx_train]
    window = (o_index - plan.WINDOW, o_index)
    if len(idx_train) == 0 or mtrain.min() < window[0] or mtrain.max() >= window[1]:
        raise ValueError("training rows fall outside [O-59, O)")
    if month_label([mtrain.max()])[0] > PARTITION_INFO_CUTOFF:
        raise ValueError("a Stage 1 training label is after the partition cutoff")

    val_ratio = plan.SPLIT_RATIOS[args.ratio]
    split_result = group_aware_train_val_split(
        groups=gtrain, val_ratio=val_ratio, min_val_per_group=int(GROUP_SPLIT["min_val_per_group"]),
        random_state=args.split_seed, skip_singleton_groups=bool(GROUP_SPLIT["skip_singleton_groups"]))
    x_set = np.asarray(split_result["X_set"], dtype=int)
    fit_rows = x_set == 0
    root_name = plan.root_name(horizon, str(term), args.g_config, args.ratio, args.split_seed)

    membership = pd.DataFrame({
        "area": np.concatenate([gtrain, gtest]),
        "target_month": month_label(np.concatenate([mtrain, months[idx_test]])),
        "role": np.concatenate([np.where(x_set == 1, "validation", "fitting"),
                                np.full(len(gtest), "heldout_target")]),
        "class_code": np.concatenate([ytrain, ytest]),
    })
    with gzip.open("fold_membership.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        membership.to_csv(handle, index=False)

    root_support = nx.support(ytrain[fit_rows], gtrain[fit_rows], mtrain[fit_rows])
    base = {
        "root": root_name, "scope": args.forecasting_scope, "horizon": horizon, "target_month": str(term),
        "origin_month": str(origin), "g_config": args.g_config, "ratio": args.ratio,
        "val_ratio": val_ratio, "split_seed": args.split_seed, "model_seed": plan.XGB_BASE["seed"],
        "train_label_months": [month_label([window[0]])[0], month_label([window[1] - 1])[0]],
        "train_label_months_observed": sorted(set(month_label(mtrain).tolist())),
        "rows": {"fitting": int(fit_rows.sum()), "validation": int((~fit_rows).sum()),
                 "heldout_target": int(len(ytest))},
        "class_counts": {"fitting": class_counts(ytrain[fit_rows]), "validation": class_counts(ytrain[~fit_rows]),
                         "heldout_target": class_counts(ytest)},
        "root_support": root_support,
        "fitting_keys_sha256": nx.keys_sha(gtrain[fit_rows], mtrain[fit_rows]),
        "validation_keys_sha256": nx.keys_sha(gtrain[~fit_rows], mtrain[~fit_rows]),
        "validation_split": {"rule": "within-area random: ceil(n*ratio) validation, >=1, keep >=1 fitting, "
                                     "singletons fitting-only (src/utils/split.py)",
                             "groups_with_validation": int((split_result["coverage"]["val_count"] > 0).sum()),
                             "singleton_groups_train_only": int((split_result["coverage"]["total_count"] == 1).sum())},
        "inherited_restrictions": "training areas restricted to areas present in the target month",
        "config": {"MIN_DEPTH": MIN_DEPTH, "MAX_DEPTH": MAX_DEPTH, "CONTIGUITY": config.CONTIGUITY,
                   "REFINE_TIMES": config.REFINE_TIMES, "MIN_BRANCH_SAMPLE_SIZE": config.MIN_BRANCH_SAMPLE_SIZE,
                   "MIN_SCAN_CLASS_SAMPLE": config.MIN_SCAN_CLASS_SAMPLE,
                   "G": plan.G_CONFIGS[args.g_config], "L": plan.L_CONFIGS,
                   "path_round_cap": plan.PATH_ROUND_CAP, "fit_support": plan.FIT_SUPPORT,
                   "val_support": plan.STAGE1_VAL_SUPPORT},
    }
    candidates = [plan.candidate_name(horizon, str(term), args.g_config, l, args.ratio, args.split_seed, f)
                  for l in plan.L_CONFIGS for f in plan.THRESHOLD_FAMILIES]
    if root_support["classes"] < 2:
        # Recorded, never a zero-weight candidate and never padded with fake labels.
        write_json("root.json", {**base, "status": "root_insufficient_support", "candidates": candidates,
                                 "module_locations": module_locations()})
        return 0

    with open(Path(args.geometry_dir) / "polygon_contiguity_info.pkl", "rb") as handle:
        contiguity_info = pickle.load(handle)
    root_started = time.time()
    booster, record = nx.fit_global(Xtrain[fit_rows], ytrain[fit_rows], plan.G_CONFIGS[args.g_config])
    record.update(fit_keys_sha256=base["fitting_keys_sha256"], fit_support=root_support)
    root_seconds = time.time() - root_started
    proba_pool = nx.proba(booster, Xtest)
    y_pool = fourclass.argmax_codes(proba_pool)
    pooled = pd.DataFrame({"FEWSNET_admin_code": gtest, "y_true_code": ytest, "y_pred_pooled_code": y_pool})
    for k, label in enumerate(fourclass.CLASS_LABELS):
        pooled[f"p_pooled_{label}"] = proba_pool[:, k]
    pooled.to_csv("root_target_predictions.csv", index=False, float_format="%.17g")

    data = (Xtrain, ytrain, gtrain, mtrain, x_set, Xtest, ytest, gtest, y_pool)
    records = {}
    for name in candidates:
        local, family = name.split("_")[3], name.split("_")[-1]
        records[name] = run_candidate(name, local, family, (booster, record), data, work,
                                      args.checkpoint_dir, contiguity_info, features)
    write_json("root.json", {**base, "status": "completed", "candidates": candidates,
                             "root_fit": record, "root_booster_sha256": record["booster_sha256"],
                             "module_locations": module_locations(),
                             "timings": {"root_fit_seconds": round(root_seconds, 2),
                                         "candidate_fit_seconds": {n: r["timings"]["fit_seconds"]
                                                                   for n, r in records.items()},
                                         "total_seconds": round(time.time() - started, 2)}})
    shutil.rmtree(work / "georf", ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
