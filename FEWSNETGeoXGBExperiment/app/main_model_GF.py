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

``--ratio tb3`` (D27, experiment-plan A2) replaces the within-area random split by the
time block: the latest three observed label months of the root's pool are the common
E1/E2 validation rows, all earlier rows are fitting; it runs only the L1/gt0 candidate.

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
from src.utils.split import confirmation_split, group_aware_train_val_split, time_block_split

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


def run_candidate(name, local, family, root, data, work, checkpoint_dir, contiguity_info, features,
                  increment_source="parent", confirmation=None):
    """One partition search from the shared root; returns the candidate record.

    ``confirmation`` (D29/A4 only) = (X, y, groups, months) of the C rows. They are never
    part of ``data``, so GeoRF.fit (E1/q, E2, support, stopping) cannot see them; the
    candidate is frozen (digest recorded) before C is predicted with predict-only routing."""
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
                  threshold=plan.THRESHOLD_FAMILIES[family], X_month=mtrain,
                  increment_source=increment_source)
        fit_seconds = time.time() - fit_started
        model_dir = Path(model.model_dir).resolve()
    finally:
        os.chdir(here)
    frozen = frozen_digest(model_dir) if confirmation is not None else None
    saved_branch = np.load(model_dir / "space_partitions" / "X_branch_id.npy", allow_pickle=False)
    if not np.array_equal(saved_branch, get_X_branch_id_by_group(gtrain, model.s_branch)):
        raise RuntimeError("saved X_branch_id disagrees with s_branch routing")
    routed_test = get_X_branch_id_by_group(gtest, model.s_branch)
    proba_part = model.model.predict_proba_georf(Xtest, gtest, model.s_branch, X_branch_id=routed_test)
    y_part = fourclass.argmax_codes(proba_part)
    part_summary, pool_summary = fourclass.summary(ytest, y_part), fourclass.summary(ytest, y_pool)
    part_crisis, pool_crisis = fourclass.crisis_summary(ytest, y_part), fourclass.crisis_summary(ytest, y_pool)
    score = float(fourclass.endpoint_exact(ytest, y_part))
    score_base = float(fourclass.endpoint_exact(ytest, y_pool))
    # Generalisation evidence (D26): the final partition and the root on ALL of the
    # candidate's validation rows (E2 population) next to the E3 target-month scores.
    val = x_set == 1
    branch_val = get_X_branch_id_by_group(gtrain[val], model.s_branch)
    y_final_val = model.model.predict_georf(Xtrain[val], gtrain[val], model.s_branch, X_branch_id=branch_val)
    model.model.load("")
    y_root_val = model.model.predict(Xtrain[val])

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
    # Keyed E2 evidence: every scored parent validation row of every fitted decision.
    e2 = (pd.concat(model.partition_e2_rows, ignore_index=True) if model.partition_e2_rows else
          pd.DataFrame(columns=["decision", "branch_id", "side", "row_id", "y_true", "y_parent", "y_child",
                                "child_eligible"]))
    e2.insert(4, "area", gtrain[e2["row_id"].to_numpy(dtype=np.int64)])
    e2.insert(5, "target_month", month_label(mtrain[e2["row_id"].to_numpy(dtype=np.int64)]))
    with gzip.open(out / "e2_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        e2.to_csv(handle, index=False)
    validation = pd.DataFrame({"area": gtrain[val], "target_month": month_label(mtrain[val]),
                               "y_true": ytrain[val], "y_root": y_root_val, "y_final": y_final_val,
                               "branch_id": np.where(branch_val == "", "root", branch_val.astype(str))})
    with gzip.open(out / "validation_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        validation.to_csv(handle, index=False)
    pd.DataFrame([{"endpoint": plan.ENDPOINT, "score": score, "score_base": score_base,
                   "macro_f1_fourclass": part_summary["macro_f1"], "macro_f1_fourclass_base": pool_summary["macro_f1"],
                   "n": len(ytest)}]).to_csv(out / "heldout_scores.csv", index=False, float_format="%.17g")
    for fname in ("s_branch.pkl", "branch_table.npy", "X_branch_id.npy"):
        shutil.copy2(model_dir / "space_partitions" / fname, out / fname)
    confirmation_scores = None
    if confirmation is not None:
        confirmation_scores = score_confirmation(model, root[0], confirmation, lookup, out)
        if frozen_digest(model_dir) != frozen:
            raise RuntimeError("the candidate changed during confirmation scoring")

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
        "increment_source": increment_source,
        "threshold": str(plan.THRESHOLD_FAMILIES[family]),
        "partition": {
            "terminal_partitions": terminal, "n_terminal": len(terminal),
            "accepted_splits": sum(d.get("outcome") == "accepted" for d in model.partition_decisions),
            "decisions": model.partition_decisions,
            "gate": (f"E2: strict {plan.ENDPOINT} gain > {plan.THRESHOLD_FAMILIES[family]} over the "
                     "current parent on its complete validation rows; parent wins ties"),
            "terminal_checkpoint_records": {b: last_save.get(b) for b in terminal},
            "correspondence_source": "space_partitions/X_branch_id.npy == s_branch routing (checked)",
        },
        "scores": {"endpoint": plan.ENDPOINT, "score": score, "score_base": score_base,
                   "macro_f1_fourclass": part_summary["macro_f1"], "macro_f1_fourclass_base": pool_summary["macro_f1"],
                   "partitioned": part_summary, "pooled": pool_summary,
                   "partitioned_crisis": part_crisis, "pooled_crisis": pool_crisis,
                   "validation": {"n": int(val.sum()),
                                  "final": {"crisis_f1": fourclass.crisis_f1(ytrain[val], y_final_val),
                                            "macro_f1_fourclass": fourclass.macro_f1(ytrain[val], y_final_val)},
                                  "root": {"crisis_f1": fourclass.crisis_f1(ytrain[val], y_root_val),
                                           "macro_f1_fourclass": fourclass.macro_f1(ytrain[val], y_root_val)}},
                   "pooled_source": "the candidate's own global root booster (same G, same fitting rows)",
                   "note": "score/score_base = E3 target-month crisis-positive F1 (D26 primary, also the E4 weight "
                           "input); fixed-four macro F1 secondary; 'validation' = final vs root on all E2 rows"},
        "fits": {"fit_log": model.model.fit_log, "saved_log": model.model.saved_log,
                 "child_fits": sum(1 for e in model.model.fit_log if e.get("kind") == "continuation")},
        "checkpoints": {"dir": str(ckpt), "sha256": checkpoints},
        **({"confirmation": {**confirmation_scores, "frozen_digest_before_scoring": frozen,
                             "role": "D29 diagnostic only: no gate, no pruning, no root fallback, not an E4 input"}}
           if confirmation is not None else {}),
        "timings": {"fit_seconds": round(fit_seconds, 2)},
    }
    write_json(out / "candidate.json", record)
    import logging
    for handler in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(handler)
        handler.close()
    shutil.rmtree(work / "georf" / name, ignore_errors=True)
    return record


def frozen_digest(model_dir) -> str:
    """SHA-256 over every saved checkpoint plus s_branch/branch_table/X_branch_id."""
    import hashlib
    model_dir = Path(model_dir)
    files = sorted((model_dir / "checkpoints").iterdir()) + [
        model_dir / "space_partitions" / f for f in ("s_branch.pkl", "branch_table.npy", "X_branch_id.npy")]
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{path.parent.name}/{path.name}\0{nx_file_sha(path)}\n".encode())
    return digest.hexdigest()


def score_confirmation(model, root_booster, confirmation, lookup, out):
    """Predict every C row with the frozen candidate (predict-only routing; areas without
    a learned route use the existing root fallback) and the root; write the keyed file."""
    Xc, yc, gc, mc = confirmation
    branch_c = get_X_branch_id_by_group(gc, model.s_branch)
    proba_final = model.model.predict_proba_georf(Xc, gc, model.s_branch, X_branch_id=branch_c)
    proba_root = nx.proba(root_booster, Xc)
    y_final, y_root = fourclass.argmax_codes(proba_final), fourclass.argmax_codes(proba_root)
    routed = np.isin(gc, list(lookup))
    frame = pd.DataFrame({"area": gc, "target_month": month_label(mc), "y_true": yc, "y_root": y_root,
                          "y_final": y_final, "branch_id": np.where(branch_c == "", "root", branch_c.astype(str)),
                          "routing": np.where(routed, "terminal_branch", "root_unassigned_area")})
    for k, label in enumerate(fourclass.CLASS_LABELS):
        frame[f"p_root_{label}"] = proba_root[:, k]
    for k, label in enumerate(fourclass.CLASS_LABELS):
        frame[f"p_final_{label}"] = proba_final[:, k]
    with gzip.open(out / "confirmation_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, float_format="%.17g")
    return {"n": int(len(yc)), "unrouted_rows": int((~routed).sum()),
            "final": {"crisis_f1": fourclass.crisis_f1(yc, y_final), "macro_f1_fourclass": fourclass.macro_f1(yc, y_final)},
            "root": {"crisis_f1": fourclass.crisis_f1(yc, y_root), "macro_f1_fourclass": fourclass.macro_f1(yc, y_root)}}


def stage1_split(mode, groups, months, origin_index, seed, horizon, target):
    """(x_set, val_ratio, validation_split record) of one root.

    r80/r50: the inherited within-area random split, unchanged. tb3 (D27): the time block
    whose validation months must equal the frozen A2 table for (H, T)."""
    if mode == plan.TIME_BLOCK:
        if seed != plan.TB3_SEED:
            raise ValueError(f"tb3 uses split seed {plan.TB3_SEED} only")
        expected = plan.TB3_VALIDATION_MONTHS.get((horizon, target))
        if expected is None:
            raise ValueError(f"no frozen tb3 validation months for h{horizon} {target}")
        result = time_block_split(groups, months, origin_index, plan.TIME_BLOCK_MONTHS, expected)
        return np.asarray(result["X_set"], dtype=int), None, {
            "split_mode": plan.TIME_BLOCK,
            "rule": (f"D27 time block: the latest {plan.TIME_BLOCK_MONTHS} observed label months of the root's "
                     "legal pool (after the target-month area restriction) are the common E1/E2 validation "
                     "rows for every area; all earlier rows are fitting; no per-area reassignment, "
                     "validation-only areas stay validation (src/utils/split.py time_block_split)"),
            "validation_months": result["validation_months"], "fitting_months": result["fitting_months"],
            "groups_with_validation": result["groups_with_validation"],
            "validation_only_groups": result["validation_only_groups"],
            "fitting_only_groups": result["fitting_only_groups"]}
    val_ratio = plan.SPLIT_RATIOS[mode]
    split_result = group_aware_train_val_split(
        groups=groups, val_ratio=val_ratio, min_val_per_group=int(GROUP_SPLIT["min_val_per_group"]),
        random_state=seed, skip_singleton_groups=bool(GROUP_SPLIT["skip_singleton_groups"]))
    return np.asarray(split_result["X_set"], dtype=int), val_ratio, {
        "rule": "within-area random: ceil(n*ratio) validation, >=1, keep >=1 fitting, "
                "singletons fitting-only (src/utils/split.py)",
        "groups_with_validation": int((split_result["coverage"]["val_count"] > 0).sum()),
        "singleton_groups_train_only": int((split_result["coverage"]["total_count"] == 1).sum())}


def root_candidates(horizon, target, g, mode, seed):
    """The candidate names of one root: four for r80/r50, the single L1/gt0 for tb3."""
    if mode == plan.TIME_BLOCK:
        return [plan.candidate_name(horizon, target, g, plan.TB3_LOCAL, mode, seed, plan.TB3_FAMILY)]
    return [plan.candidate_name(horizon, target, g, l, mode, seed, f)
            for l in plan.L_CONFIGS for f in plan.THRESHOLD_FAMILIES]


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
    parser.add_argument("--ratio", required=True, choices=sorted(plan.SPLIT_RATIOS) + [plan.TIME_BLOCK],
                        help="r80/r50 within-area random split, or tb3 (D27 time block)")
    parser.add_argument("--split-seed", type=int, required=True, choices=plan.SPLIT_SEEDS)
    parser.add_argument("--checkpoint-dir", required=True, help="scratch store for boosters (outside Dropbox)")
    parser.add_argument("--increment-source", choices=nx.INCREMENT_SOURCES, default="parent",
                        help="parent: children continue the current parent (D4); root: D28/A3 shared-root "
                             "single L1 increment, r80/seed 42/L1/gt0 only")
    parser.add_argument("--confirmation-split", action="store_true",
                        help="D29/A4 rootconf: split the original r80 validation label-blind into search S "
                             "and frozen-candidate confirmation C (root increments only)")
    args = parser.parse_args()
    if args.confirmation_split and args.increment_source != "root":
        raise ValueError("--confirmation-split runs only with --increment-source root (A4)")
    if args.increment_source == "root" and (args.ratio != plan.ROOTINC_RATIO or args.split_seed != plan.ROOTINC_SEED
                                            or args.desired_terms not in plan.ROOTINC_TARGETS
                                            or args.g_config != plan.TB3_G[str(forecasting_scope_to_lag(args.forecasting_scope, LAGS_MONTHS))]):
        raise ValueError(f"root increments run only {plan.ROOTINC_RATIO}/seed {plan.ROOTINC_SEED} at "
                         f"{plan.ROOTINC_TARGETS} with the locked G {plan.TB3_G} (A3)")
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

    x_set, val_ratio, validation_split = stage1_split(args.ratio, gtrain, mtrain, o_index, args.split_seed,
                                                      horizon, str(term))
    fit_rows = x_set == 0
    if args.confirmation_split:
        root_name = plan.rootconf_root_name(horizon, str(term), args.g_config)
    elif args.increment_source == "root":
        root_name = plan.rootinc_root_name(horizon, str(term), args.g_config)
    else:
        root_name = plan.root_name(horizon, str(term), args.g_config, args.ratio, args.split_seed)
    # D29/A4: original validation -> S (search, role "validation") / C ("confirmation").
    conf_rows = np.zeros(len(x_set), dtype=bool)
    if args.confirmation_split:
        orig_val = np.flatnonzero(x_set == 1)
        conf_rows[orig_val[confirmation_split(gtrain[orig_val], mtrain[orig_val], plan.CONFIRMATION_SEED) == 1]] = True
    roles = np.where(conf_rows, "confirmation", np.where(x_set == 1, "validation", "fitting"))

    membership = pd.DataFrame({
        "area": np.concatenate([gtrain, gtest]),
        "target_month": month_label(np.concatenate([mtrain, months[idx_test]])),
        "role": np.concatenate([roles, np.full(len(gtest), "heldout_target")]),
        "class_code": np.concatenate([ytrain, ytest]),
    })
    with gzip.open("fold_membership.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        membership.to_csv(handle, index=False)

    root_support = nx.support(ytrain[fit_rows], gtrain[fit_rows], mtrain[fit_rows])
    base = {
        "root": root_name, "scope": args.forecasting_scope, "horizon": horizon, "target_month": str(term),
        "origin_month": str(origin), "g_config": args.g_config, "ratio": args.ratio,
        "val_ratio": val_ratio, "split_seed": args.split_seed, "increment_source": args.increment_source, "model_seed": plan.XGB_BASE["seed"],
        "train_label_months": [month_label([window[0]])[0], month_label([window[1] - 1])[0]],
        "train_label_months_observed": sorted(set(month_label(mtrain).tolist())),
        "rows": {"fitting": int(fit_rows.sum()), "validation": int((~fit_rows).sum()),
                 "heldout_target": int(len(ytest))},
        "class_counts": {"fitting": class_counts(ytrain[fit_rows]), "validation": class_counts(ytrain[~fit_rows]),
                         "heldout_target": class_counts(ytest)},
        "root_support": root_support,
        "fitting_keys_sha256": nx.keys_sha(gtrain[fit_rows], mtrain[fit_rows]),
        "validation_keys_sha256": nx.keys_sha(gtrain[~fit_rows], mtrain[~fit_rows]),
        "validation_split": validation_split,
        **({"confirmation_split": {
            "rule": ("D29/A4 label-blind: fresh random.Random(42); odd-count areas shuffled, first floor(n_odd/2) "
                     "give S the extra row; per area (ascending) shuffled month indices, first floor(n/2)+extra "
                     "are S, rest C (src/utils/split.py confirmation_split)"),
            "confirmation_seed": plan.CONFIRMATION_SEED,
            "rows": {"search_S": int(((x_set == 1) & ~conf_rows).sum()), "confirmation_C": int(conf_rows.sum())},
            "original_validation_keys_sha256": nx.keys_sha(gtrain[x_set == 1], mtrain[x_set == 1]),
            "search_keys_sha256": nx.keys_sha(gtrain[(x_set == 1) & ~conf_rows], mtrain[(x_set == 1) & ~conf_rows]),
            "confirmation_keys_sha256": nx.keys_sha(gtrain[conf_rows], mtrain[conf_rows]),
            "class_counts": {"search_S": class_counts(ytrain[(x_set == 1) & ~conf_rows]),
                             "confirmation_C": class_counts(ytrain[conf_rows])}}}
           if args.confirmation_split else {}),
        "inherited_restrictions": "training areas restricted to areas present in the target month",
        "config": {"MIN_DEPTH": MIN_DEPTH, "MAX_DEPTH": MAX_DEPTH, "CONTIGUITY": config.CONTIGUITY,
                   "REFINE_TIMES": config.REFINE_TIMES, "MIN_BRANCH_SAMPLE_SIZE": config.MIN_BRANCH_SAMPLE_SIZE,
                   "MIN_SCAN_CLASS_SAMPLE": config.MIN_SCAN_CLASS_SAMPLE,
                   "G": plan.G_CONFIGS[args.g_config], "L": plan.L_CONFIGS,
                   "path_round_cap": plan.PATH_ROUND_CAP, "fit_support": plan.FIT_SUPPORT,
                   "val_support": plan.STAGE1_VAL_SUPPORT},
    }
    if args.ratio == plan.TIME_BLOCK:
        base.update(split_mode=plan.TIME_BLOCK, validation_months=validation_split["validation_months"],
                    fitting_months=validation_split["fitting_months"])
    if args.confirmation_split:
        candidates = [plan.rootconf_candidate_name(horizon, str(term), args.g_config)]
        explicit = {candidates[0]: (plan.ROOTINC_LOCAL, plan.ROOTINC_FAMILY)}
    elif args.increment_source == "root":
        candidates = [plan.rootinc_candidate_name(horizon, str(term), args.g_config)]
        explicit = {candidates[0]: (plan.ROOTINC_LOCAL, plan.ROOTINC_FAMILY)}
    else:
        candidates = root_candidates(horizon, str(term), args.g_config, args.ratio, args.split_seed)
        explicit = {}
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

    confirmation = None
    if args.confirmation_split:
        # C rows leave the search data entirely; fitting rows and S are untouched.
        confirmation = (Xtrain[conf_rows], ytrain[conf_rows], gtrain[conf_rows], mtrain[conf_rows])
        keep = ~conf_rows
        Xtrain, ytrain, gtrain, mtrain, x_set = Xtrain[keep], ytrain[keep], gtrain[keep], mtrain[keep], x_set[keep]
    data = (Xtrain, ytrain, gtrain, mtrain, x_set, Xtest, ytest, gtest, y_pool)
    records = {}
    for name in candidates:
        local, family = explicit.get(name, (name.split("_")[3], name.split("_")[-1]))
        records[name] = run_candidate(name, local, family, (booster, record), data, work,
                                      args.checkpoint_dir, contiguity_info, features,
                                      increment_source=args.increment_source, confirmation=confirmation)
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
