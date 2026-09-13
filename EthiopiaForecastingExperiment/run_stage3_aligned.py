#!/usr/bin/env python3
"""Evaluate one Ethiopia GeoRF fold from a pre-aligned 88-feature snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EthiopiaForecastingExperiment.aligned_refit import (
    KEY,
    MODEL_PREDICTORS,
    SCOPE_HORIZONS,
    TARGET,
    apply_fold_medians,
    fit_fold_medians,
    select_rolling_fold,
    split_latest_months,
    validate_aligned_frame,
)


def assign_partitions(frame: pd.DataFrame, mapping: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return row-aligned partition labels after exact cohort validation."""
    partition_column = next(
        (column for column in ("cluster_id", "cluster", "partition_id") if column in mapping),
        None,
    )
    if KEY not in mapping or partition_column is None:
        raise ValueError("Partition map needs FEWSNET_admin_code and a partition column")
    map_frame = mapping[[KEY, partition_column]].copy()
    map_frame[KEY] = pd.to_numeric(map_frame[KEY], errors="raise").astype("int64")
    frame_codes = pd.to_numeric(frame[KEY], errors="raise").astype("int64")
    if map_frame[KEY].duplicated().any() or set(map_frame[KEY]) != set(frame_codes):
        raise ValueError("Partition map does not cover the exact snapshot cohort")
    labels = frame_codes.map(map_frame.set_index(KEY)[partition_column])
    if labels.isna().any():
        raise ValueError("Partition map produced null assignments")
    return labels.astype(str).to_numpy(), partition_column


def prepare_stage3_fold(
    frame: pd.DataFrame,
    *,
    target_month: str,
    horizon: int,
    train_window: int,
    validation_months: int,
) -> dict[str, object]:
    """Build separate no-leak threshold and final imputation paths."""
    train_indices, test_indices = select_rolling_fold(
        frame,
        target_month=target_month,
        horizon=horizon,
        window_months=train_window,
    )
    test_groups = frame.iloc[test_indices][KEY].unique()
    train_indices = train_indices[
        np.isin(frame.iloc[train_indices][KEY].to_numpy(), test_groups)
    ]
    train_dates = pd.to_datetime(frame.iloc[train_indices]["target_month"])
    fit_mask, validation_mask = split_latest_months(
        train_dates,
        validation_months=validation_months,
    )
    X_raw = frame.iloc[train_indices][list(MODEL_PREDICTORS)].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=float)
    X_test_raw = frame.iloc[test_indices][list(MODEL_PREDICTORS)].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=float)
    threshold_medians = fit_fold_medians(X_raw[fit_mask])
    final_medians = fit_fold_medians(X_raw)
    return {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "fit_mask": fit_mask,
        "validation_mask": validation_mask,
        "X_fit": apply_fold_medians(X_raw[fit_mask], threshold_medians),
        "X_validation": apply_fold_medians(X_raw[validation_mask], threshold_medians),
        "X_train": apply_fold_medians(X_raw, final_medians),
        "X_test": apply_fold_medians(X_test_raw, final_medians),
        "y_fit": pd.to_numeric(frame.iloc[train_indices][TARGET], errors="raise").astype(int).to_numpy()[fit_mask],
        "y_validation": pd.to_numeric(frame.iloc[train_indices][TARGET], errors="raise").astype(int).to_numpy()[validation_mask],
        "y_train": pd.to_numeric(frame.iloc[train_indices][TARGET], errors="raise").astype(int).to_numpy(),
        "y_test": pd.to_numeric(frame.iloc[test_indices][TARGET], errors="coerce").to_numpy(dtype=float),
        "train_dates": train_dates.reset_index(drop=True),
        "threshold_medians": threshold_medians,
        "final_medians": final_medians,
    }


def _new_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=seed,
        n_jobs=1,
    )


def _fit_partitioned(
    X: np.ndarray,
    y: np.ndarray,
    partitions: np.ndarray,
    *,
    seed: int,
    min_samples: int = 50,
) -> tuple[dict[str, RandomForestClassifier], dict[str, int]]:
    models: dict[str, RandomForestClassifier] = {}
    smote_count = 0
    fallback_count = 0
    for partition_id in np.unique(partitions):
        mask = partitions == partition_id
        X_local = X[mask]
        y_local = y[mask]
        classes, counts = np.unique(y_local, return_counts=True)
        if len(y_local) < min_samples or len(classes) < 2:
            fallback_count += 1
            continue
        if counts.min() >= 2:
            k_neighbors = min(5, int(counts.min()) - 1)
            X_local, y_local = SMOTE(
                random_state=seed,
                k_neighbors=k_neighbors,
            ).fit_resample(X_local, y_local)
            smote_count += 1
        model = _new_rf(seed)
        model.fit(X_local, y_local)
        models[str(partition_id)] = model
    return models, {
        "local_model_count": len(models),
        "smote_partition_count": smote_count,
        "pooled_fallback_partition_count": fallback_count,
    }


def _class1_probability(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    match = np.flatnonzero(classes == 1)
    if match.size == 0:
        return np.zeros(len(X), dtype=float)
    return probabilities[:, match[0]]


def _predict_partitioned(
    X: np.ndarray,
    partitions: np.ndarray,
    pooled: RandomForestClassifier,
    local_models: dict[str, RandomForestClassifier],
) -> np.ndarray:
    output = np.empty(len(X), dtype=float)
    for partition_id in np.unique(partitions):
        mask = partitions == partition_id
        output[mask] = _class1_probability(
            local_models.get(str(partition_id), pooled),
            X[mask],
        )
    return output


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int | float]:
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    tp = int(((true == 1) & (pred == 1)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    tn = int(((true == 0) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _select_threshold(y_true: np.ndarray, probability: np.ndarray) -> dict[str, object]:
    base: dict[str, object] = {
        "selected_threshold": 0.5,
        "validation_precision": np.nan,
        "validation_recall": np.nan,
        "validation_f1": np.nan,
        "validation_support": int(len(y_true)),
        "validation_positive_cases": int((np.asarray(y_true) == 1).sum()),
        "fallback_reason": "",
    }
    if not len(y_true):
        return {**base, "fallback_reason": "no_validation_observations"}
    if not (np.asarray(y_true) == 1).any():
        return {**base, "fallback_reason": "no_validation_positive_cases"}
    candidates = np.unique(np.round(np.asarray(probability, dtype=float), 2))
    candidates = np.sort(candidates[(candidates >= 0.05) & (candidates <= 0.95)])[::-1]
    if not candidates.size:
        return {**base, "fallback_reason": "no_candidate_thresholds"}
    scored = [
        (float(threshold), _metrics(y_true, probability >= threshold))
        for threshold in candidates
    ]
    threshold, score = max(scored, key=lambda item: (item[1]["f1"], item[0]))
    return {
        **base,
        "selected_threshold": threshold,
        "validation_precision": score["precision"],
        "validation_recall": score["recall"],
        "validation_f1": score["f1"],
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run_stage3(args: argparse.Namespace) -> Path:
    """Run one scope-month evaluation fold and write its three artifacts."""
    if args.start_month != args.end_month:
        raise ValueError("Aligned Stage 3 accepts one target month per invocation")
    if not args.enable_symmetric_validation_threshold:
        raise ValueError("Symmetric validation-only thresholds are required")
    scope = args.forecasting_scope
    if scope not in SCOPE_HORIZONS:
        raise ValueError(f"Unsupported forecasting scope: {scope}")

    frame = pd.read_csv(args.data, low_memory=False)
    validate_aligned_frame(frame, f"fs{scope}", SCOPE_HORIZONS[scope], MODEL_PREDICTORS)
    mapping = pd.read_csv(args.partition_map, low_memory=False)
    partitions, partition_column = assign_partitions(frame, mapping)
    prepared = prepare_stage3_fold(
        frame,
        target_month=args.start_month,
        horizon=SCOPE_HORIZONS[scope],
        train_window=args.train_window,
        validation_months=args.threshold_validation_months,
    )
    train_idx = prepared["train_indices"]
    test_idx = prepared["test_indices"]
    fit_mask = prepared["fit_mask"]
    validation_mask = prepared["validation_mask"]
    train_parts = partitions[train_idx]
    fit_parts = train_parts[fit_mask]
    validation_parts = train_parts[validation_mask]
    test_parts = partitions[test_idx]

    threshold_pooled = _new_rf(args.random_seed)
    threshold_pooled.fit(prepared["X_fit"], prepared["y_fit"])
    threshold_local, threshold_model_audit = _fit_partitioned(
        prepared["X_fit"],
        prepared["y_fit"],
        fit_parts,
        seed=args.random_seed,
    )
    pooled_validation_probability = _class1_probability(
        threshold_pooled, prepared["X_validation"]
    )
    partitioned_validation_probability = _predict_partitioned(
        prepared["X_validation"],
        validation_parts,
        threshold_pooled,
        threshold_local,
    )
    thresholds = {
        "pooled": _select_threshold(
            prepared["y_validation"], pooled_validation_probability
        ),
        "partitioned": _select_threshold(
            prepared["y_validation"], partitioned_validation_probability
        ),
    }

    pooled = _new_rf(args.random_seed)
    pooled.fit(prepared["X_train"], prepared["y_train"])
    local_models, final_model_audit = _fit_partitioned(
        prepared["X_train"],
        prepared["y_train"],
        train_parts,
        seed=args.random_seed,
    )
    pooled_probability = _class1_probability(pooled, prepared["X_test"])
    partitioned_probability = _predict_partitioned(
        prepared["X_test"],
        test_parts,
        pooled,
        local_models,
    )
    pooled_threshold = float(thresholds["pooled"]["selected_threshold"])
    partitioned_threshold = float(thresholds["partitioned"]["selected_threshold"])
    predictions = pd.DataFrame(
        {
            KEY: frame.iloc[test_idx][KEY].to_numpy(),
            "month_start": pd.Period(args.start_month, freq="M").to_timestamp(),
            "partition_id": test_parts,
            "y_true": prepared["y_test"],
            "y_pred_pooled": (pooled_probability >= 0.5).astype(int),
            "y_pred_partitioned": (partitioned_probability >= 0.5).astype(int),
            "y_prob_pooled": pooled_probability,
            "y_prob_partitioned": partitioned_probability,
            "selected_threshold_pooled": pooled_threshold,
            "selected_threshold_partitioned": partitioned_threshold,
            "y_pred_pooled_thresholded": (pooled_probability >= pooled_threshold).astype(int),
            "y_pred_partitioned_thresholded": (
                partitioned_probability >= partitioned_threshold
            ).astype(int),
        }
    )

    target_observed = np.isfinite(prepared["y_test"])
    suppress_test_metrics = target_observed.sum() / len(mapping) < 0.9
    threshold_row: dict[str, object] = {
        "test_month": args.start_month,
        "forecasting_scope": scope,
        "active_lag_months": SCOPE_HORIZONS[scope],
        "threshold_enabled": True,
        "threshold_mode": "symmetric_validation_only",
        "fit_start": prepared["train_dates"][fit_mask].min(),
        "fit_end": prepared["train_dates"][fit_mask].max(),
        "validation_start": prepared["train_dates"][validation_mask].min(),
        "validation_end": prepared["train_dates"][validation_mask].max(),
    }
    for name in ("pooled", "partitioned"):
        threshold_row.update(
            {f"{name}_{key}": value for key, value in thresholds[name].items()}
        )
        if suppress_test_metrics:
            test_metrics = {
                key: np.nan for key in ("precision", "recall", "f1", "tp", "fp", "fn", "tn")
            }
        else:
            test_metrics = _metrics(
                prepared["y_test"][target_observed],
                predictions.loc[target_observed, f"y_pred_{name}_thresholded"].to_numpy(),
            )
        threshold_row.update(
            {f"{name}_test_{key}": value for key, value in test_metrics.items()}
        )

    args.out_dir.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(args.out_dir / "predictions_monthly.csv", index=False)
    pd.DataFrame([threshold_row]).to_csv(args.out_dir / "thresholds_by_fold.csv", index=False)
    all_null_threshold = [
        feature
        for feature, median in zip(MODEL_PREDICTORS, prepared["threshold_medians"])
        if median == 0
        and frame.iloc[train_idx[fit_mask]][feature].isna().all()
    ]
    all_null_final = [
        feature
        for feature, median in zip(MODEL_PREDICTORS, prepared["final_medians"])
        if median == 0 and frame.iloc[train_idx][feature].isna().all()
    ]
    manifest = {
        "data_path": str(args.data.resolve()),
        "data_sha256": _sha256(args.data),
        "partition_map_path": str(args.partition_map.resolve()),
        "partition_map_sha256": _sha256(args.partition_map),
        "partition_column": partition_column,
        "partition_admin_count": int(mapping[KEY].nunique()),
        "scope": f"fs{scope}",
        "horizon_months": SCOPE_HORIZONS[scope],
        "target_month": args.start_month,
        "train_window_months": args.train_window,
        "threshold_validation_months": args.threshold_validation_months,
        "random_seed": args.random_seed,
        "predictor_count": len(MODEL_PREDICTORS),
        "predictors": list(MODEL_PREDICTORS),
        "train_rows": len(train_idx),
        "fit_rows": int(fit_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "test_rows": len(test_idx),
        "test_target_observed": int(target_observed.sum()),
        "test_metrics_suppressed": suppress_test_metrics,
        "threshold_imputer_all_null_columns": all_null_threshold,
        "final_imputer_all_null_columns": all_null_final,
        "threshold_models": threshold_model_audit,
        "final_models": final_model_audit,
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return args.out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--partition-map", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start-month", required=True)
    parser.add_argument("--end-month", required=True)
    parser.add_argument("--train-window", type=int, default=36)
    parser.add_argument("--forecasting-scope", type=int, required=True)
    parser.add_argument("--enable-symmetric-validation-threshold", action="store_true")
    parser.add_argument("--threshold-validation-months", type=int, default=6)
    parser.add_argument("--random-seed", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    completed = run_stage3(parse_args())
    print(f"Stage 3 fold complete: {completed}")
