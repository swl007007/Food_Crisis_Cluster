#!/usr/bin/env python3
"""Stage 1: one four-class GeoRF partition-learning fold (design.md "Stage 1").

Adapted from the release entrypoint. The release loaded the raw panel, imputed the
whole panel, shifted rows and dropped NaN rows; this version consumes the
origin-aligned snapshot for one horizon, keeps every row, and lets each RF fit its
own imputer on its real fitting rows (PRD R5-R8, R12).

python app/main_model_GF.py --data SNAPSHOT --geometry-dir DIR --forecasting_scope N \
    --desired_terms YYYY-MM --schema FEATURE_SCHEMA [--retain-checkpoints DIR]

Run from a fresh working directory. Writes, in that directory:
  results_df_gp_fs{N}_{Y}_{Y}.csv, y_pred_test_gp_fs{N}_{Y}_{Y}.csv,
  result_GeoRF*/correspondence_table_{YYYY-MM}.csv, candidate.json,
  fold_membership.csv.gz
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
from config import GROUP_SPLIT, MAX_DEPTH, MIN_DEPTH, TRAIN_WINDOW_MONTHS, VAL_RATIO, LAGS_MONTHS
from src.customize.customize import train_test_split_rolling_window
from src.feature.fourclass_features import load_schema, month_label
from src.helper.helper import get_X_branch_id_by_group
from src.metrics import fourclass
from src.model.GeoRF import GeoRF
from src.model.model_RF import RFmodel
from src.utils.lag_schedules import forecasting_scope_to_lag
from src.utils.split import group_aware_train_val_split

PARTITION_INFO_CUTOFF = "2020-12"


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="snapshot_h{H}.parquet for this scope")
    parser.add_argument("--geometry-dir", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--forecasting_scope", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--desired_terms", required=True, help="single target month YYYY-MM")
    parser.add_argument("--retain-checkpoints", default=None,
                        help="copy the fitted branch checkpoints here before cleanup (replay folds)")
    args = parser.parse_args()
    started = time.time()

    horizon = forecasting_scope_to_lag(args.forecasting_scope, LAGS_MONTHS)
    term = pd.Period(args.desired_terms, freq="M")
    if str(term) > PARTITION_INFO_CUTOFF:
        raise ValueError("Stage 1 targets must not exceed the 2020-12 partition information cutoff (D10)")
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
    origin = (term - horizon)
    if len(idx_test) == 0:
        raise ValueError(f"{term} has no labelled target rows; it must be skipped at scheduling")
    train_months = months[idx_train]
    window = (int(origin.year * 12 + origin.month - 1) - (TRAIN_WINDOW_MONTHS - 1),
              int(origin.year * 12 + origin.month - 1))
    if len(idx_train) == 0 or train_months.min() < window[0] or train_months.max() >= window[1]:
        raise ValueError("training rows fall outside [O-35, O)")
    if month_label([train_months.max()])[0] > PARTITION_INFO_CUTOFF:
        raise ValueError("a Stage 1 training label is after the partition cutoff")

    # The released within-area validation split, computed once and handed to fit.
    split_result = group_aware_train_val_split(
        groups=gtrain, val_ratio=float(VAL_RATIO),
        min_val_per_group=int(GROUP_SPLIT["min_val_per_group"]),
        random_state=GROUP_SPLIT["random_state"],
        skip_singleton_groups=bool(GROUP_SPLIT["skip_singleton_groups"]))
    x_set = np.asarray(split_result["X_set"], dtype=int)

    with open(Path(args.geometry_dir) / "polygon_contiguity_info.pkl", "rb") as handle:
        contiguity_info = pickle.load(handle)

    fit_started = time.time()
    model = GeoRF(min_model_depth=MIN_DEPTH, max_model_depth=MAX_DEPTH)
    model.fit(Xtrain, ytrain, gtrain, X_set=x_set, split={"X_set": x_set}, val_ratio=VAL_RATIO,
              contiguity_type="polygon", polygon_contiguity_info=contiguity_info,
              feature_names=features, print_to_file=True, track_partition_metrics=False,
              VIS_DEBUG_MODE=False)
    fit_seconds = time.time() - fit_started

    model_dir = Path(model.model_dir)
    saved_branch = np.load(model_dir / "space_partitions" / "X_branch_id.npy", allow_pickle=False)
    routed_train = get_X_branch_id_by_group(gtrain, model.s_branch)
    if not np.array_equal(saved_branch, routed_train):
        raise RuntimeError("saved X_branch_id disagrees with s_branch routing")

    # Held-out monthly scores (Stage 2 input). Partitioned = branch routing;
    # pooled = release eval RF retrained on the same real fitting rows.
    y_pred_part = model.predict(Xtest, gtest).astype(np.int64)
    routed_test = get_X_branch_id_by_group(gtest, model.s_branch)
    pooled = RFmodel(model.dir_ckpt, model.n_trees_unit, max_depth=model.max_depth,
                     num_class=model.num_class, random_state=model.random_state, n_jobs=model.n_jobs)
    pooled.train(model._base_training_X, model._base_training_y, branch_id="eval")
    y_pred_base = pooled.predict(Xtest).astype(np.int64)
    part_summary = fourclass.summary(ytest, y_pred_part)
    base_summary = fourclass.summary(ytest, y_pred_base)

    # Correspondence of training areas to terminal branches, from the routed array.
    branch_str = np.where(saved_branch == "", "root", saved_branch.astype(str))
    corr = pd.DataFrame({"FEWSNET_admin_code": gtrain, "partition_id": branch_str}).drop_duplicates()
    if corr["FEWSNET_admin_code"].duplicated().any():
        raise RuntimeError("an area received more than one terminal partition")
    corr = corr.sort_values("FEWSNET_admin_code")
    corr.to_csv(model_dir / f"correspondence_table_{term}.csv", index=False)
    train_areas = set(corr["FEWSNET_admin_code"].tolist())
    test_in_train = np.isin(gtest, list(train_areas))
    lookup = dict(zip(corr["FEWSNET_admin_code"], corr["partition_id"]))
    routed_test_str = np.where(routed_test == "", "root", routed_test.astype(str))
    if any(lookup[a] != b for a, b in zip(gtest[test_in_train], routed_test_str[test_in_train])):
        raise RuntimeError("test routing disagrees with the exported correspondence")

    row = {"year": term.year, "month": term.month, "macro_f1": part_summary["macro_f1"],
           "macro_f1_base": base_summary["macro_f1"], "n": len(ytest)}
    for k, label in enumerate(fourclass.CLASS_LABELS):
        row[f"f1_class{label}"] = part_summary["per_class"][label]["f1"]
        row[f"f1_base_class{label}"] = base_summary["per_class"][label]["f1"]
        row[f"support_class{label}"] = part_summary["per_class"][label]["support"]
    tag = f"fs{args.forecasting_scope}_{term.year}_{term.year}"
    pd.DataFrame([row]).to_csv(f"results_df_gp_{tag}.csv", index=False)
    pd.DataFrame({
        "FEWSNET_admin_code": gtest, "target_month": str(term), "horizon": horizon,
        "origin_month": str(origin), "y_true_code": ytest, "y_pred_partitioned_code": y_pred_part,
        "y_pred_pooled_code": y_pred_base, "branch_id": routed_test_str,
        "routing": np.where(test_in_train, "terminal_branch", "root_unassigned_test_area"),
    }).to_csv(f"y_pred_test_gp_{tag}.csv", index=False)

    membership = pd.DataFrame({
        "area": np.concatenate([gtrain, gtest]),
        "target_month": month_label(np.concatenate([months[idx_train], months[idx_test]])),
        "role": np.concatenate([np.where(x_set == 1, "validation", "fitting"),
                                np.full(len(gtest), "heldout_target")]),
        "branch_id": np.concatenate([branch_str, routed_test_str]),
    })
    with gzip.open("fold_membership.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        membership.to_csv(handle, index=False)

    rf = model.model
    terminal = sorted(set(branch_str.tolist()))
    # Inherited release behaviour: scan candidates are built from validation groups
    # only, so an area with no validation row (e.g. a singleton) is absent from the
    # accepted s0/s1 lists and keeps its parent branch, routing to the parent model.
    internal = {b for b in terminal if any(o != b and o.startswith(b if b != "root" else "") for o in terminal)}
    parent_routed = {
        "train_areas": int(corr["partition_id"].isin(internal).sum()),
        "train_rows": int(np.isin(branch_str, list(internal)).sum()),
        "heldout_rows": int(np.isin(routed_test_str, list(internal)).sum()),
        "labels": sorted(internal),
        "note": ("areas without validation rows are not in any accepted scan subset; they keep the "
                 "parent branch and use the parent's checkpoint (routing and correspondence agree)"),
    }
    last_save = {}
    for entry in rf.saved_log:
        last_save[entry["saved_as"] or "root"] = entry
    terminal_routes = {b: last_save.get(b if b != "root" else "root") for b in terminal}
    candidate = {
        "scope": args.forecasting_scope, "horizon": horizon, "target_month": str(term),
        "origin_month": str(origin),
        "train_label_months": [month_label([window[0]])[0], month_label([window[1] - 1])[0]],
        "train_label_months_observed": sorted(set(month_label(train_months).tolist())),
        "rows": {"fitting": int((x_set == 0).sum()), "validation": int((x_set == 1).sum()),
                 "heldout_target": int(len(ytest))},
        "class_counts": {"fitting": class_counts(ytrain[x_set == 0]),
                         "validation": class_counts(ytrain[x_set == 1]),
                         "heldout_target": class_counts(ytest)},
        "areas": {"training": len(train_areas), "heldout_target": int(np.unique(gtest).size),
                  "heldout_target_without_training_rows": int(np.unique(gtest[~test_in_train]).size)},
        "inherited_restrictions": {
            "training_restricted_to_target_month_areas": True,
            "note": ("customize.train_test_split_rolling_window keeps only training areas present "
                     "in the target month; held-out areas without training rows route to the root."),
        },
        "validation_split": {"val_ratio": float(VAL_RATIO), **{k: GROUP_SPLIT[k] for k in (
            "min_val_per_group", "skip_singleton_groups", "random_state")},
            "groups_with_validation": int((split_result["coverage"]["val_count"] > 0).sum()),
            "singleton_groups_train_only": int((split_result["coverage"]["total_count"] == 1).sum())},
        "partition": {
            "terminal_partitions": terminal, "n_terminal": len(terminal),
            "accepted_splits": len(terminal) - 1,
            "decisions": model.partition_decisions,
            "gate": "strict fixed-four macro-F1 gain > 0.01 on the parent's validation rows; parent wins ties",
            "terminal_checkpoint_routes": terminal_routes,
            "parent_routed_areas": parent_routed,
            "correspondence_source": "space_partitions/X_branch_id.npy == s_branch routing (checked)",
        },
        "scores": {"macro_f1": part_summary["macro_f1"], "macro_f1_base": base_summary["macro_f1"],
                   "partitioned": part_summary, "pooled": base_summary,
                   "note": "held-out target-month scores; NOT the within-window split-validation scores"},
        "fits": {"georf_fit_log": rf.fit_log, "georf_saved_log": rf.saved_log,
                 "pooled_eval_fit": pooled.fit_record,
                 "pseudo_rows_per_fit": int(rf.num_class),
                 "pseudo_row_rule": "one zero-feature row per class appended after real-row imputation"},
        "imputers": {"root": rf.fit_log[0]["imputer_sha256"] if rf.fit_log else None,
                     "pooled_eval": pooled.fit_record["imputer_sha256"]},
        "estimator_params": {k: v for k, v in pooled.model.get_params().items()
                             if k in ("n_estimators", "max_depth", "random_state", "class_weight",
                                      "max_features", "bootstrap", "criterion")},
        "config": {"MIN_DEPTH": MIN_DEPTH, "MAX_DEPTH": MAX_DEPTH, "NUM_CLASS": config.NUM_CLASS,
                   "GOVERNING_METRIC": config.GOVERNING_METRIC,
                   "MIN_MACRO_F1_IMPROVEMENT_THRESHOLD": config.MIN_MACRO_F1_IMPROVEMENT_THRESHOLD,
                   "MIN_BRANCH_SAMPLE_SIZE": config.MIN_BRANCH_SAMPLE_SIZE,
                   "MIN_SCAN_CLASS_SAMPLE": config.MIN_SCAN_CLASS_SAMPLE,
                   "CONTIGUITY": config.CONTIGUITY, "REFINE_TIMES": config.REFINE_TIMES,
                   "FEATURE_DROP": config.FEATURE_DROP, "N_JOBS": config.N_JOBS,
                   "RUN_PRE_PARTITION_DIAGNOSTIC": config.RUN_PRE_PARTITION_DIAGNOSTIC},
        "module_locations": module_locations(),
        "timings": {"fit_seconds": round(fit_seconds, 2), "total_seconds": round(time.time() - started, 2)},
    }
    Path("candidate.json").write_text(json.dumps(candidate, indent=2, default=str), encoding="utf-8")

    import logging
    for handler in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(handler)
        handler.close()
    if args.retain_checkpoints:
        target = Path(args.retain_checkpoints)
        target.mkdir(parents=True, exist_ok=False)
        shutil.copytree(model_dir / "checkpoints", target / "checkpoints")
        shutil.copytree(model_dir / "space_partitions", target / "space_partitions")
    shutil.rmtree(model_dir / "checkpoints")
    shutil.rmtree(model_dir / "vis", ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
