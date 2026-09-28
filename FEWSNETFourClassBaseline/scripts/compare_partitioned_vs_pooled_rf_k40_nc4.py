#!/usr/bin/env python3
"""Stage 3: four-class partitioned versus pooled RF, one horizon (design.md "Stage 3").

Adapted in place from the release comparison script. Binary metrics, SMOTE and the
validation-threshold arms are removed; this package has none of them (PRD R12, R15).

python scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py --data SNAPSHOT \
    --schema FEATURE_SCHEMA --consensus CONSENSUS_JSON --observations LEDGER --out-dir DIR \
    --start-month YYYY-MM --end-month YYYY-MM --forecasting-scope N

``CONSENSUS_JSON`` is Stage 2's record. With ``route == "learned_map"`` it names the
cluster map; with ``route == "null_consensus"`` (D15) only the pooled RF is fitted
and the partitioned arm reuses its predictions exactly.

Per fold, in ``DIR/folds/<YYYY-MM>/``: fold.json, local_support.csv,
imputer_statistics.csv.gz, training_keys.csv.gz. For the horizon:
predictions.csv.gz and run_manifest.json. Existing output is never overwritten.
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ['PYTHONHASHSEED'] = '5'
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import LAGS_MONTHS, TRAIN_WINDOW_MONTHS
from src.feature.fourclass_features import load_schema, month_label
from src.metrics import fourclass
from src.model.model_RF import MaxPlusImputer
from src.utils.lag_schedules import forecasting_scope_to_lag

RANDOM_STATE = 5  # MUST match main pipeline (GeoRF.py default)
PARTITION_UNMAPPED_THRESHOLD_PCT = 2.0
PARTITION_INFO_CUTOFF = "2020-12"
RF_PARAMS = {'n_estimators': 100, 'max_depth': None, 'random_state': RANDOM_STATE, 'class_weight': None}
# n_jobs changes wall time only; sklearn forests are identical for any n_jobs at a fixed seed.
N_JOBS = int(os.environ.get("FOURCLASS_STAGE3_N_JOBS", "8"))
# Minimum samples per partition to train separate model (else fallback to pooled)
MIN_PARTITION_SAMPLES = 50


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class FittedRF:
    """One Stage 3 estimator: its own training-only imputer plus a real-rows-only forest."""

    def __init__(self, X, y, n_jobs=N_JOBS):
        self.imputer = MaxPlusImputer().fit(X)
        self.forest = RandomForestClassifier(n_jobs=n_jobs, **RF_PARAMS)
        self.forest.fit(self.imputer.transform(X), y)
        self.n_rows = int(len(y))
        self.class_counts = [int(np.sum(y == k)) for k in range(fourclass.N_CLASSES)]

    def proba(self, X):
        raw = self.forest.predict_proba(self.imputer.transform(X))
        return fourclass.align_probabilities(raw, self.forest.classes_)

    def record(self):
        return {"rows": self.n_rows, "class_counts": self.class_counts,
                "classes_": [int(c) for c in self.forest.classes_],
                "imputer_sha256": self.imputer.digest(),
                "params": {k: v for k, v in self.forest.get_params().items() if k in RF_PARAMS}}


def load_consensus(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("route") == "null_consensus":
        return record, None
    if record.get("route") != "learned_map":
        raise ValueError(f"unknown consensus route in {path}")
    mapping = pd.read_csv(record["cluster_map"])
    if file_sha256(record["cluster_map"]) != record["cluster_map_sha256"]:
        raise ValueError("cluster map differs from the digest Stage 2 recorded")
    if mapping["FEWSNET_admin_code"].duplicated().any():
        raise ValueError("cluster map assigns an area twice")
    return record, dict(zip(mapping["FEWSNET_admin_code"].astype(int), mapping["cluster_id"].astype(int)))


def fit_fold(snap, features, test_month, horizon, cluster_of):
    """Fit and predict one target month. Returns (predictions, fold record, extras)."""
    target = test_month.year * 12 + test_month.month - 1
    origin = target - horizon
    if month_label([origin])[0] <= PARTITION_INFO_CUTOFF:
        raise ValueError("Stage 3 origin must be after the partition information cutoff")
    lo = origin - (TRAIN_WINDOW_MONTHS - 1)
    months = snap["target_month"].to_numpy()
    train = snap[(months >= lo) & (months < origin)]
    test = snap[months == target]
    record = {"target_month": str(test_month), "origin_month": month_label([origin])[0],
              "horizon": horizon, "train_label_months": [month_label([lo])[0], month_label([origin - 1])[0]],
              "train_label_months_observed": sorted(set(month_label(train["target_month"]).tolist()))}
    if test.empty:
        record["status"] = "skipped_empty_target"
        return None, record, None
    if train.empty:
        raise RuntimeError(f"{test_month}: empty required fitting pool with a nonempty target")
    Xtr = train[features].to_numpy(dtype=float)
    ytr = train["class_code"].to_numpy(dtype=np.int64)
    Xte = test[features].to_numpy(dtype=float)
    started = time.time()
    pooled = FittedRF(Xtr, ytr)
    p_pooled = pooled.proba(Xte)
    estimators = {"pooled": pooled}
    local_rows = []

    if cluster_of is None:
        p_part = p_pooled.copy()
        route = np.full(len(test), "null_consensus_pooled_reuse", dtype=object)
        cluster_test = np.full(len(test), -1)
    else:
        cluster_train = np.array([cluster_of.get(int(a), -1) for a in train["area"]])
        cluster_test = np.array([cluster_of.get(int(a), -1) for a in test["area"]])
        p_part = np.zeros_like(p_pooled)
        route = np.empty(len(test), dtype=object)
        # Inherited restriction: local fits only for clusters present in the target month.
        for cid in sorted(set(cluster_test.tolist())):
            rows_te = cluster_test == cid
            if cid < 0:
                p_part[rows_te] = p_pooled[rows_te]
                route[rows_te] = "unmapped_area_pooled"
                continue
            rows_tr = cluster_train == cid
            n_rows, n_classes = int(rows_tr.sum()), int(np.unique(ytr[rows_tr]).size)
            entry = {"cluster_id": int(cid), "train_rows": n_rows, "observed_classes": n_classes,
                     "test_rows": int(rows_te.sum()),
                     **{f"train_class{fourclass.CLASS_LABELS[k]}": int(np.sum(ytr[rows_tr] == k))
                        for k in range(fourclass.N_CLASSES)}}
            if n_rows < MIN_PARTITION_SAMPLES or n_classes < 2:
                reason = ("no_local_training_rows" if n_rows == 0 else
                          "local_rows_below_50" if n_rows < MIN_PARTITION_SAMPLES else "single_class_local")
                p_part[rows_te] = p_pooled[rows_te]
                route[rows_te] = f"pooled_fallback:{reason}"
                entry.update(route="pooled_fallback", reason=reason)
            else:
                local = FittedRF(Xtr[rows_tr], ytr[rows_tr])
                estimators[f"local_{cid}"] = local
                p_part[rows_te] = local.proba(Xte[rows_te])
                route[rows_te] = "local_model"
                entry.update(route="local_model", reason="", imputer_sha256=local.imputer.digest(),
                             classes_=[int(c) for c in local.forest.classes_])
            local_rows.append(entry)

    y_pooled = fourclass.argmax_codes(p_pooled)
    y_part = fourclass.argmax_codes(p_part)
    ytest = test["class_code"].to_numpy(dtype=np.int64)
    preds = pd.DataFrame({
        "area": test["area"].to_numpy(), "target_month": str(test_month),
        "origin_month": month_label([origin])[0], "horizon": horizon,
        "y_true_code": ytest, "y_pred_pooled_code": y_pooled, "y_pred_partitioned_code": y_part,
        "cluster_id": cluster_test, "partitioned_route": route,
    })
    for k, label in enumerate(fourclass.CLASS_LABELS):
        preds[f"p_pooled_{label}"] = p_pooled[:, k]
        preds[f"p_partitioned_{label}"] = p_part[:, k]
    keys = train[["area", "target_month"]].to_numpy(dtype=np.int64)
    record.update({
        "status": "fitted",
        "route": "null_consensus" if cluster_of is None else "learned_map",
        "rows": {"train": int(len(train)), "test": int(len(test))},
        "train_class_counts": pooled.class_counts,
        "test_class_counts": [int(np.sum(ytest == k)) for k in range(fourclass.N_CLASSES)],
        "train_keys_sha256": sha256_bytes(np.ascontiguousarray(keys).tobytes()),
        "estimators": {name: est.record() for name, est in estimators.items()},
        "fits": {"pooled": 1, "local": len(estimators) - 1,
                 "partitioned_arm_reuses_pooled": cluster_of is None},
        "partitioned_route_counts": pd.Series(route).value_counts().to_dict(),
        "pseudo_rows": 0,
        "pooled_summary": fourclass.summary(ytest, y_pooled),
        "partitioned_summary": fourclass.summary(ytest, y_part),
        "seconds": round(time.time() - started, 2),
    })
    fills = pd.DataFrame({name: est.imputer.fill_ for name, est in estimators.items()}, index=features)
    fills.index.name = "feature"
    return preds, record, {"local_support": pd.DataFrame(local_rows), "imputer_fills": fills,
                           "training_keys": train[["area", "target_month", "class_code"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, help='snapshot_h{H}.parquet for this horizon')
    parser.add_argument('--schema', required=True)
    parser.add_argument('--consensus', required=True, help="Stage 2 consensus.json")
    parser.add_argument('--observations', required=True, help="prepared/ledgers/observations.csv (coverage gate)")
    parser.add_argument('--out-dir', required=True, help='Output directory')
    parser.add_argument('--start-month', required=True, help='Start month (YYYY-MM)')
    parser.add_argument('--end-month', required=True, help='End month (YYYY-MM)')
    parser.add_argument('--forecasting-scope', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--only-month', default=None, help='replay a single fold into a fresh --out-dir')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; Stage 3 output is never overwritten")
    out_dir.mkdir(parents=True)
    horizon = forecasting_scope_to_lag(args.forecasting_scope, LAGS_MONTHS)
    features = load_schema(Path(args.schema))["ordered_features"]
    snap = pd.read_parquet(args.data)
    if not (snap["horizon"] == horizon).all() or list(snap.columns[-len(features):]) != features:
        raise ValueError("snapshot horizon or feature order mismatch")
    consensus, cluster_of = load_consensus(args.consensus)

    coverage = None
    if cluster_of is not None:
        # Inherited gate (release create_partition_group_array): unmapped share of ALL
        # labelled panel rows. The evaluated-target share is disclosed, not gated.
        observed = pd.read_csv(args.observations, usecols=["area"])
        unmapped_gate = float(100 * (~observed["area"].isin(list(cluster_of))).mean())
        first = pd.Period(args.start_month, "M")
        last = pd.Period(args.end_month, "M")
        months = snap["target_month"]
        evaluated = snap[(months >= first.year * 12 + first.month - 1) & (months <= last.year * 12 + last.month - 1)]
        coverage = {"gate_population": "all labelled panel rows (release definition)",
                    "unmapped_pct_gate": unmapped_gate,
                    "unmapped_pct_evaluated_targets_disclosed": float(
                        100 * (~evaluated["area"].isin(list(cluster_of))).mean()),
                    "unmapped_areas_evaluated": int(evaluated.loc[~evaluated["area"].isin(list(cluster_of)), "area"].nunique()),
                    "threshold_pct": PARTITION_UNMAPPED_THRESHOLD_PCT,
                    "unmapped_route": "pooled RF of the same fold"}
        if unmapped_gate > PARTITION_UNMAPPED_THRESHOLD_PCT:
            raise ValueError(f"partition coverage insufficient: {coverage}")

    months = pd.period_range(args.start_month, args.end_month, freq="M")
    if args.only_month:
        months = [pd.Period(args.only_month, "M")]
    folds, predictions = [], []
    for month in months:
        preds, record, extras = fit_fold(snap, features, month, horizon, cluster_of)
        fold_dir = out_dir / "folds" / str(month)
        fold_dir.mkdir(parents=True)
        (fold_dir / "fold.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        folds.append({k: record[k] for k in ("target_month", "origin_month", "status")})
        if preds is None:
            print(f"{month}: skipped (no labelled target rows)", flush=True)
            continue
        extras["local_support"].to_csv(fold_dir / "local_support.csv", index=False)
        with gzip.open(fold_dir / "imputer_statistics.csv.gz", "wt", encoding="utf-8", newline="") as handle:
            extras["imputer_fills"].to_csv(handle)
        with gzip.open(fold_dir / "training_keys.csv.gz", "wt", encoding="utf-8", newline="") as handle:
            extras["training_keys"].to_csv(handle, index=False)
        predictions.append(preds)
        print(f"{month}: n={record['rows']['test']} pooled={record['pooled_summary']['macro_f1']:.4f} "
              f"partitioned={record['partitioned_summary']['macro_f1']:.4f} "
              f"local_fits={record['fits']['local']} ({record['seconds']}s)", flush=True)

    all_preds = pd.concat(predictions, ignore_index=True)
    with gzip.open(out_dir / "predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        all_preds.to_csv(handle, index=False, float_format="%.17g")
    manifest = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "scope": args.forecasting_scope, "horizon": horizon,
        "months": [args.start_month, args.end_month], "only_month": args.only_month,
        "data": args.data, "data_sha256": file_sha256(args.data),
        "consensus": consensus, "coverage": coverage,
        "rf_params": RF_PARAMS, "n_jobs": N_JOBS, "min_partition_samples": MIN_PARTITION_SAMPLES,
        "decision": "fixed-axis argmax; ties to the first class; no threshold calibration",
        "pseudo_rows": 0, "smote": False,
        "folds": folds,
        "fitted_folds": sum(f["status"] == "fitted" for f in folds),
        "skipped_empty": sum(f["status"] != "fitted" for f in folds),
        "prediction_rows": int(len(all_preds)),
        "python": sys.version.split()[0],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


if __name__ == '__main__':
    main()
