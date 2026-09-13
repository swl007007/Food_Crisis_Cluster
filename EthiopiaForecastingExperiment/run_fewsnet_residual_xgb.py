#!/usr/bin/env python3
"""Fit the isolated Ethiopia FEWS NET residual XGBoost comparison."""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBRegressor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EthiopiaForecastingExperiment.run_local_partition_experiment import (
    SCOPES,
    TARGET_MONTHS,
    _filter_fewsnet_ethiopia,
    _phase3_binary,
    binary_metrics,
    create_run_directory,
    normalize_admin_codes,
    sha256_file,
    write_json,
)
from EthiopiaForecastingExperiment.run_binary_xgb_comparison import class_weights
from EthiopiaForecastingExperiment.aligned_refit import validate_aligned_frame


KEY = "FEWSNET_admin_code"
TARGET = "fews_ipc_crisis"
TRAIN_MONTHS = 36
VALIDATION_MONTHS = 6
MIN_COVERAGE = 0.90
SEED = 5
EXPECTED_ADMINS = 1_040
SUPPRESSED_MONTH = pd.Period("2021-06", freq="M")
GRID = tuple(itertools.product((3, 6), (1, 5), (200, 400)))
COMPARISON_MODELS = {
    "fewsnet": "expert_anchor",
    "binary_xgboost": "binary_xgboost_prediction",
    "georf_v5": "georf_v5_prediction",
    "residual_xgboost": "residual_xgboost_prediction",
}
EXPERT_MAPPING = {
    0: (4, "fews_proj_near"),
    1: (4, "fews_proj_near"),
    2: (8, "fews_proj_med"),
    3: (12, "fews_proj_med"),
}


def split_residual_fold(
    frame: pd.DataFrame,
    *,
    target_month: str,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Build the outer fit/validation/test split on usable expert rows."""
    dates = pd.to_datetime(frame["target_month"], errors="raise").dt.to_period("M")
    target = pd.Period(target_month, freq="M")
    train_end = target - horizon
    train_start = train_end - TRAIN_MONTHS
    usable = (
        frame[TARGET].notna()
        & frame["expert_anchor"].notna()
        & dates.ne(SUPPRESSED_MONTH)
    )
    training = usable & dates.ge(train_start) & dates.lt(train_end)
    months = np.sort(dates.loc[training].unique())
    if len(months) <= VALIDATION_MONTHS:
        raise ValueError(f"Too few residual training months for {target_month}")
    validation_months = set(months[-VALIDATION_MONTHS:])
    fit = np.flatnonzero((training & ~dates.isin(validation_months)).to_numpy())
    validation = np.flatnonzero((training & dates.isin(validation_months)).to_numpy())
    test = np.flatnonzero(dates.eq(target).to_numpy())
    if not len(fit) or not len(validation) or not len(test):
        raise ValueError(f"Empty residual fold split for {target_month}")
    return fit, validation, test, {
        "target_month": str(target),
        "train_start": str(train_start),
        "train_end_exclusive": str(train_end),
        "eligible_training_months": int(len(months)),
        "validation_months": [str(value) for value in sorted(validation_months)],
        "fit_rows": int(len(fit)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "test_labels_used_in_selection": False,
    }


def new_residual_model(model_parameters: dict[str, int]) -> XGBRegressor:
    """Construct one frozen residual-regression candidate."""
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=1,
        **model_parameters,
    )


def temporal_oof_predictions(
    frame: pd.DataFrame,
    features: np.ndarray,
    desired_rows: np.ndarray,
    *,
    horizon: int,
    model_parameters: dict[str, int],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Generate first-residual predictions from month-grouped forward folds."""
    matrix = np.asarray(features, dtype=float)
    desired = np.asarray(desired_rows, dtype=int)
    if matrix.ndim != 2 or len(matrix) != len(frame) or np.isinf(matrix).any():
        raise ValueError("OOF feature matrix is misaligned or contains infinity")
    if desired.ndim != 1 or np.any((desired < 0) | (desired >= len(frame))):
        raise ValueError("OOF desired rows are invalid")

    predictions = np.full(len(frame), np.nan, dtype=float)
    dates = pd.to_datetime(frame["target_month"], errors="raise").dt.to_period("M")
    audits: list[dict[str, object]] = []
    for month in np.sort(dates.iloc[desired].unique()):
        month_name = str(month)
        try:
            train_rows, test_rows, audit = select_temporal_oof_fold(
                frame,
                target_month=month_name,
                horizon=horizon,
            )
            expected = desired[dates.iloc[desired].eq(month).to_numpy()]
            if not np.array_equal(np.sort(test_rows), np.sort(expected)):
                raise ValueError(f"OOF target-month rows are incomplete for {month_name}")
            y_train = frame.iloc[train_rows][TARGET].astype(int).to_numpy()
            weights, weight_audit = class_weights(y_train)
        except ValueError as error:
            audits.append({"target_month": month_name, "skipped": True, "detail": str(error)})
            continue

        residual = (
            y_train.astype(float)
            - frame.iloc[train_rows]["expert_anchor"].to_numpy(dtype=float)
        )
        model = new_residual_model(model_parameters)
        model.fit(matrix[train_rows], residual, sample_weight=weights)
        predicted = model.predict(matrix[test_rows]).astype(float)
        if not np.isfinite(predicted).all():
            raise ValueError(f"Non-finite OOF prediction for {month_name}")
        predictions[test_rows] = predicted
        audits.append({**audit, "skipped": False, "weight_audit": weight_audit})
    return predictions, audits


def candidate_order(row: dict[str, object]) -> tuple[float, int, int, int, int]:
    """Order validation candidates with the confirmed simplicity tie-break."""
    return (
        -float(row["validation_f1"]),
        int(row["layer_count"]),
        int(row["max_depth"]),
        int(row["n_estimators"]),
        -int(row["min_child_weight"]),
    )


def plot_summary(summary: pd.DataFrame, output_path: Path) -> int:
    """Plot equal-month binary metrics across the four forecasting scopes."""
    import matplotlib.pyplot as plt

    metrics = (
        ("precision", "Crisis precision"),
        ("recall", "Crisis recall"),
        ("f1", "Crisis F1"),
        ("balanced_accuracy", "Balanced accuracy"),
    )
    styles = {
        "fewsnet": ("FEWS NET", "#000000", "o", "-"),
        "binary_xgboost": ("Binary XGBoost", "#0072B2", "s", "--"),
        "georf_v5": ("GeoRF v5", "#E69F00", "^", "-."),
        "residual_xgboost": ("FEWS + residual XGBoost", "#009E73", "D", ":"),
    }
    horizons = {"fs0": 1, "fs1": 4, "fs2": 8, "fs3": 12}
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        for model, (label, color, marker, linestyle) in styles.items():
            subset = summary.loc[summary["model"].eq(model)].copy()
            subset["horizon"] = subset["scope"].map(horizons)
            subset = subset.dropna(subset=["horizon", metric]).sort_values("horizon")
            axis.plot(
                subset["horizon"],
                subset[metric],
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=5,
            )
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.set_xticks(tuple(horizons.values()), tuple(horizons))
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("Forecast horizon (months)")
    axes[1, 1].set_xlabel("Forecast horizon (months)")
    axes[0, 0].set_ylabel("Equal-month mean")
    axes[1, 0].set_ylabel("Equal-month mean")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Ethiopia crisis classification: expert residual comparison", y=0.99)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return int(axes.size)


def fold_coverage(
    test: pd.DataFrame,
    *,
    cohort_n: int,
    threshold: float = MIN_COVERAGE,
) -> dict[str, int | float | bool]:
    """Return target/expert coverage and the experiment's single suppression flag."""
    if cohort_n <= 0 or not 0 < threshold <= 1:
        raise ValueError("Cohort size and coverage threshold must be positive")
    target_available = test[TARGET].notna()
    expert_available = test["expert_anchor"].notna()
    target_n = int(target_available.sum())
    expert_n = int(expert_available.sum())
    common_n = int((target_available & expert_available).sum())
    target_coverage = target_n / cohort_n
    expert_coverage = expert_n / cohort_n
    return {
        "cohort_n": cohort_n,
        "target_n": target_n,
        "expert_n": expert_n,
        "common_n": common_n,
        "target_coverage": target_coverage,
        "expert_coverage": expert_coverage,
        "common_coverage": common_n / cohort_n,
        "suppressed": target_coverage < threshold or expert_coverage < threshold,
    }


def select_score_threshold(y_true: np.ndarray, score: np.ndarray) -> dict[str, object]:
    """Select a validation-only F1 threshold for an unbounded additive score."""
    y = np.asarray(y_true, dtype=int)
    values = np.asarray(score, dtype=float)
    base: dict[str, object] = {
        "selected_threshold": 0.5,
        "validation_precision": np.nan,
        "validation_recall": np.nan,
        "validation_f1": np.nan,
        "validation_support": int(len(y)),
        "validation_positive_cases": int((y == 1).sum()),
        "fallback_reason": "",
    }
    if y.ndim != 1 or values.ndim != 1 or y.shape != values.shape:
        raise ValueError("Validation labels and scores must be aligned vectors")
    if not len(y):
        return {**base, "fallback_reason": "no_validation_observations"}
    if not np.isfinite(values).all():
        raise ValueError("Validation scores must be finite")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Validation labels must be binary")
    if not (y == 1).any():
        return {**base, "fallback_reason": "no_validation_positive_cases"}

    candidates = np.unique(np.r_[np.round(values, 2), 0.5])
    threshold, metrics = max(
        (
            (float(candidate), binary_metrics(y, values >= candidate))
            for candidate in candidates
        ),
        key=lambda item: (float(item[1]["f1"]), item[0]),
    )
    return {
        **base,
        "selected_threshold": threshold,
        "validation_precision": metrics["precision"],
        "validation_recall": metrics["recall"],
        "validation_f1": metrics["f1"],
    }


def second_residual_target(
    *,
    y_true: np.ndarray,
    expert: np.ndarray,
    first_layer_oof: np.ndarray,
) -> np.ndarray:
    """Return the second-layer target from temporal OOF first-layer scores."""
    values = tuple(np.asarray(item, dtype=float) for item in (y_true, expert, first_layer_oof))
    if any(item.ndim != 1 for item in values) or len({item.shape for item in values}) != 1:
        raise ValueError("Residual target inputs must be aligned one-dimensional arrays")
    if not all(np.isfinite(item).all() for item in values):
        raise ValueError("Residual target inputs must be finite")
    return values[0] - values[1] - values[2]


def select_temporal_oof_fold(
    frame: pd.DataFrame,
    *,
    target_month: str,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Select one month-grouped, horizon-safe rolling OOF fold."""
    dates = pd.to_datetime(frame["target_month"], errors="raise").dt.to_period("M")
    target = pd.Period(target_month, freq="M")
    train_end = target - horizon
    train_start = train_end - TRAIN_MONTHS
    usable = (
        frame[TARGET].notna()
        & frame["expert_anchor"].notna()
        & dates.ne(SUPPRESSED_MONTH)
    )
    train = np.flatnonzero(
        (usable & dates.ge(train_start) & dates.lt(train_end)).to_numpy()
    )
    test = np.flatnonzero((usable & dates.eq(target)).to_numpy())
    if not len(train) or not len(test):
        raise ValueError(f"Insufficient temporal OOF support for {target_month}")
    return train, test, {
        "target_month": str(target),
        "train_start": str(train_start),
        "train_end_exclusive": str(train_end),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }


def attach_expert_anchor(
    snapshot: pd.DataFrame,
    fewsnet: pd.DataFrame,
    scope: int,
) -> pd.DataFrame:
    """Attach the scope-specific FEWS projection by explicit calendar month."""
    if scope not in EXPERT_MAPPING:
        raise ValueError(f"Unsupported forecasting scope: {scope}")
    lag_months, projection_column = EXPERT_MAPPING[scope]
    frame = snapshot.copy()
    frame[KEY] = normalize_admin_codes(frame[KEY]).to_numpy()
    target = pd.to_datetime(frame["target_month"], errors="raise").dt.to_period("M")
    frame["_target_period"] = target
    frame["_expert_source_period"] = target - lag_months
    frame["expert_source_month"] = frame["_expert_source_period"].astype(str)
    frame["_row_order"] = np.arange(len(frame))

    eth = _filter_fewsnet_ethiopia(fewsnet)
    source = eth[[KEY, "period", projection_column]].rename(
        columns={
            "period": "_expert_source_period",
            projection_column: "expert_projection_phase",
        }
    )
    actual = eth[[KEY, "period", "fews_ipc"]].rename(
        columns={"period": "_target_period", "fews_ipc": "fewsnet_target_phase"}
    )
    frame = frame.merge(
        source,
        on=[KEY, "_expert_source_period"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    frame = frame.merge(
        actual,
        on=[KEY, "_target_period"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    frame["expert_anchor"] = _phase3_binary(frame["expert_projection_phase"])
    frame["fewsnet_truth"] = _phase3_binary(frame["fewsnet_target_phase"])
    return (
        frame.sort_values("_row_order")
        .drop(columns=["_target_period", "_expert_source_period", "_row_order"])
        .reset_index(drop=True)
    )


def run_experiment(args: argparse.Namespace) -> Path:
    """Fit all eligible folds and write the isolated comparison bundle."""
    v5_dir = args.v5_run.resolve()
    binary_dir = args.binary_run.resolve()
    fewsnet_path = args.fewsnet.resolve()
    v5_manifest_path = v5_dir / "run_manifest.json"
    binary_metadata_path = binary_dir / "run_metadata.json"
    binary_predictions_path = binary_dir / "predictions.csv"
    required = (v5_manifest_path, binary_metadata_path, binary_predictions_path, fewsnet_path)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    v5_manifest = json.loads(v5_manifest_path.read_text(encoding="utf-8"))
    binary_metadata = json.loads(binary_metadata_path.read_text(encoding="utf-8"))
    predictors = list(v5_manifest["predictors"])
    expected_scopes = {f"fs{scope}": horizon for scope, horizon in SCOPES.items()}
    if (
        len(predictors) != 88
        or v5_manifest["scopes"] != expected_scopes
        or v5_manifest["target_months"] != list(TARGET_MONTHS)
        or binary_metadata["predictors"] != predictors
    ):
        raise ValueError("Frozen v5 or binary XGBoost contract mismatch")

    source_paths = {
        "script": Path(__file__).resolve(),
        "prd": REPO_ROOT / ".trellis" / "tasks" / "09-04-ethiopia-fewsnet-residual-xgb" / "prd.md",
        "aligned_contract_code": Path(__file__).resolve().parent / "aligned_refit.py",
        "binary_code": Path(__file__).resolve().parent / "run_binary_xgb_comparison.py",
        "comparison_code": Path(__file__).resolve().parent / "run_local_partition_experiment.py",
        "v5_manifest": v5_manifest_path,
        "binary_metadata": binary_metadata_path,
        "binary_predictions": binary_predictions_path,
        "fewsnet": fewsnet_path,
        **{
            f"snapshot_fs{scope}": v5_dir / "input" / f"ethiopia_panel_fs{scope}_88.csv"
            for scope in SCOPES
        },
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes_before = {name: sha256_file(path) for name, path in source_paths.items()}
    for scope in SCOPES:
        if hashes_before[f"snapshot_fs{scope}"] != v5_manifest["snapshot_hashes"][f"fs{scope}"]:
            raise ValueError(f"Frozen fs{scope} snapshot hash drift")
    frozen_hashes = binary_metadata["hashes"]
    expected_frozen = {
        "v5_manifest": frozen_hashes["input_v5_manifest"],
        "binary_predictions": frozen_hashes["artifact_predictions"],
        "fewsnet": frozen_hashes["input_fewsnet"],
    }
    for name, expected in expected_frozen.items():
        if hashes_before[name] != expected:
            raise ValueError(f"Frozen {name} hash drift")

    run_dir = create_run_directory(args.output_root.resolve(), args.run_id)
    fewsnet = pd.read_csv(fewsnet_path, low_memory=False)
    comparator = pd.read_csv(binary_predictions_path, low_memory=False)
    required_comparator = {
        "scope",
        "horizon_months",
        KEY,
        "target_month",
        "forecast_origin_month",
        TARGET,
        "xgboost_prediction",
        "y_true",
        "y_pred_partitioned_thresholded",
    }
    missing_comparator = required_comparator.difference(comparator.columns)
    if missing_comparator:
        raise ValueError(f"Binary comparator missing columns: {sorted(missing_comparator)}")
    comparator[KEY] = normalize_admin_codes(comparator[KEY]).to_numpy()
    for column in ("target_month", "forecast_origin_month"):
        comparator[column] = pd.to_datetime(comparator[column], errors="raise").dt.to_period("M").dt.to_timestamp()
    if comparator.duplicated(["scope", KEY, "target_month"]).any():
        raise ValueError("Duplicate frozen binary comparator keys")
    if not np.array_equal(comparator[TARGET].astype(int), comparator["y_true"].astype(int)):
        raise ValueError("Frozen binary and GeoRF truths disagree")
    comparator = comparator[
        ["scope", KEY, "target_month", TARGET, "xgboost_prediction", "y_pred_partitioned_thresholded"]
    ].rename(
        columns={
            TARGET: "comparator_truth",
            "xgboost_prediction": "binary_xgboost_prediction",
            "y_pred_partitioned_thresholded": "georf_v5_prediction",
        }
    )

    prediction_frames: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    fold_audits: list[dict[str, object]] = []
    model_count = len(COMPARISON_MODELS)

    for scope, horizon in SCOPES.items():
        scope_name = f"fs{scope}"
        snapshot = pd.read_csv(source_paths[f"snapshot_fs{scope}"], low_memory=False)
        validate_aligned_frame(snapshot, scope_name, horizon, predictors)
        snapshot[KEY] = normalize_admin_codes(snapshot[KEY]).to_numpy()
        for column in ("target_month", "forecast_origin_month"):
            snapshot[column] = pd.to_datetime(snapshot[column], errors="raise").dt.to_period("M").dt.to_timestamp()
        cohort_n = int(snapshot[KEY].nunique())
        if cohort_n != EXPECTED_ADMINS:
            raise ValueError(f"fs{scope} cohort has {cohort_n} admins, expected {EXPECTED_ADMINS}")
        snapshot = attach_expert_anchor(snapshot, fewsnet, scope)
        observed_truth = snapshot[TARGET].notna()
        if snapshot.loc[observed_truth, "fewsnet_truth"].isna().any():
            raise ValueError(f"Raw FEWS truth is missing for observed {scope_name} targets")
        if not np.array_equal(
            snapshot.loc[observed_truth, TARGET].astype(int),
            snapshot.loc[observed_truth, "fewsnet_truth"].astype(int),
        ):
            raise ValueError(f"Snapshot and raw FEWS truths disagree for {scope_name}")

        base_features = snapshot[predictors].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if np.isinf(base_features).any():
            raise ValueError(f"Infinite predictor value for {scope_name}")
        features = np.column_stack(
            [base_features, snapshot["expert_anchor"].to_numpy(dtype=float)]
        )
        dates = snapshot["target_month"].dt.to_period("M")
        scope_comparator = comparator.loc[comparator["scope"].eq(scope_name)]

        for target_month in TARGET_MONTHS:
            target = pd.Period(target_month, freq="M")
            test_rows = np.flatnonzero(dates.eq(target).to_numpy())
            if not len(test_rows):
                raise ValueError(f"No snapshot rows for {scope_name} {target_month}")
            test = snapshot.iloc[test_rows].copy()
            test["_snapshot_row"] = test_rows
            test = test.merge(
                scope_comparator,
                on=["scope", KEY, "target_month"],
                how="left",
                sort=False,
                validate="one_to_one",
            )
            coverage = fold_coverage(test, cohort_n=cohort_n)
            common = test[
                [
                    TARGET,
                    "fewsnet_truth",
                    "expert_anchor",
                    "comparator_truth",
                    "binary_xgboost_prediction",
                    "georf_v5_prediction",
                ]
            ].notna().all(axis=1)
            coverage["common_n"] = int(common.sum())
            coverage["common_coverage"] = float(common.sum() / cohort_n)
            coverage["suppressed"] = bool(
                coverage["suppressed"] or coverage["common_coverage"] < MIN_COVERAGE
            )
            if common.any():
                supported_truth = test.loc[common, TARGET].astype(int).to_numpy()
                if not np.array_equal(
                    supported_truth,
                    test.loc[common, "fewsnet_truth"].astype(int).to_numpy(),
                ) or not np.array_equal(
                    supported_truth,
                    test.loc[common, "comparator_truth"].astype(int).to_numpy(),
                ):
                    raise ValueError(f"Comparison truth mismatch for {scope_name} {target_month}")

            if coverage["suppressed"]:
                for model_name in COMPARISON_MODELS:
                    metric_rows.append(
                        {
                            "scope": scope_name,
                            "horizon_months": horizon,
                            "target_month": target_month,
                            "model": model_name,
                            "suppressed": True,
                            "coverage_n": coverage["common_n"],
                            "coverage_total": cohort_n,
                            "coverage": coverage["common_coverage"],
                            "precision": np.nan,
                            "recall": np.nan,
                            "f1": np.nan,
                            "balanced_accuracy": np.nan,
                            "n": np.nan,
                            "tp": np.nan,
                            "fp": np.nan,
                            "fn": np.nan,
                            "tn": np.nan,
                            "selected_layer_count": np.nan,
                        }
                    )
                fold_audits.append(
                    {
                        "scope": scope_name,
                        "horizon_months": horizon,
                        "target_month": target_month,
                        "coverage": coverage,
                        "suppressed": True,
                        "model_fitted": False,
                    }
                )
                continue

            fit_rows, validation_rows, fold_test_rows, fold_audit = split_residual_fold(
                snapshot,
                target_month=target_month,
                horizon=horizon,
            )
            if not np.array_equal(test_rows, fold_test_rows):
                raise ValueError(f"Test-row drift for {scope_name} {target_month}")
            y_fit = snapshot.iloc[fit_rows][TARGET].astype(int).to_numpy()
            y_validation = snapshot.iloc[validation_rows][TARGET].astype(int).to_numpy()
            expert_fit = snapshot.iloc[fit_rows]["expert_anchor"].to_numpy(dtype=float)
            expert_validation = snapshot.iloc[validation_rows]["expert_anchor"].to_numpy(dtype=float)
            fit_weights, fit_weight_audit = class_weights(y_fit)
            candidates: list[dict[str, object]] = []

            for max_depth, min_child_weight, n_estimators in GRID:
                parameters = {
                    "max_depth": max_depth,
                    "min_child_weight": min_child_weight,
                    "n_estimators": n_estimators,
                }
                first_model = new_residual_model(parameters)
                first_model.fit(
                    features[fit_rows],
                    y_fit.astype(float) - expert_fit,
                    sample_weight=fit_weights,
                )
                first_validation = first_model.predict(features[validation_rows]).astype(float)
                one_layer_score = expert_validation + first_validation
                one_layer_threshold = select_score_threshold(y_validation, one_layer_score)
                candidate_base = {
                    "scope": scope_name,
                    "horizon_months": horizon,
                    "target_month": target_month,
                    **parameters,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_seed": SEED,
                }
                candidates.append(
                    {
                        **candidate_base,
                        "layer_count": 1,
                        **one_layer_threshold,
                        "validation_score_min": float(one_layer_score.min()),
                        "validation_score_max": float(one_layer_score.max()),
                        "oof_target_months": 0,
                        "oof_rows": 0,
                        "oof_skipped_months": 0,
                    }
                )

                first_oof, oof_audits = temporal_oof_predictions(
                    snapshot,
                    features,
                    fit_rows,
                    horizon=horizon,
                    model_parameters=parameters,
                )
                oof_rows = fit_rows[np.isfinite(first_oof[fit_rows])]
                if not len(oof_rows):
                    raise ValueError(f"No temporal OOF residuals for {scope_name} {target_month}")
                y_oof = snapshot.iloc[oof_rows][TARGET].astype(int).to_numpy()
                second_target = second_residual_target(
                    y_true=y_oof,
                    expert=snapshot.iloc[oof_rows]["expert_anchor"].to_numpy(dtype=float),
                    first_layer_oof=first_oof[oof_rows],
                )
                second_weights, _ = class_weights(y_oof)
                second_model = new_residual_model(parameters)
                second_model.fit(
                    features[oof_rows],
                    second_target,
                    sample_weight=second_weights,
                )
                two_layer_score = (
                    one_layer_score
                    + second_model.predict(features[validation_rows]).astype(float)
                )
                two_layer_threshold = select_score_threshold(y_validation, two_layer_score)
                candidates.append(
                    {
                        **candidate_base,
                        "layer_count": 2,
                        **two_layer_threshold,
                        "validation_score_min": float(two_layer_score.min()),
                        "validation_score_max": float(two_layer_score.max()),
                        "oof_target_months": int(dates.iloc[oof_rows].nunique()),
                        "oof_rows": int(len(oof_rows)),
                        "oof_skipped_months": int(sum(audit["skipped"] for audit in oof_audits)),
                    }
                )

            selected = min(candidates, key=candidate_order)
            for candidate in candidates:
                candidate["selected"] = candidate is selected
            tuning_rows.extend(candidates)

            parameters = {
                name: int(selected[name])
                for name in ("max_depth", "min_child_weight", "n_estimators")
            }
            train_rows = np.sort(np.concatenate([fit_rows, validation_rows]))
            y_train = snapshot.iloc[train_rows][TARGET].astype(int).to_numpy()
            train_expert = snapshot.iloc[train_rows]["expert_anchor"].to_numpy(dtype=float)
            train_weights, train_weight_audit = class_weights(y_train)
            final_first = new_residual_model(parameters)
            final_first.fit(
                features[train_rows],
                y_train.astype(float) - train_expert,
                sample_weight=train_weights,
            )

            supported = test.loc[common].copy()
            model_test_rows = supported["_snapshot_row"].astype(int).to_numpy()
            first_test = final_first.predict(features[model_test_rows]).astype(float)
            second_test = np.zeros(len(model_test_rows), dtype=float)
            final_oof_audits: list[dict[str, object]] = []
            final_oof_rows = np.array([], dtype=int)
            if int(selected["layer_count"]) == 2:
                final_oof, final_oof_audits = temporal_oof_predictions(
                    snapshot,
                    features,
                    train_rows,
                    horizon=horizon,
                    model_parameters=parameters,
                )
                final_oof_rows = train_rows[np.isfinite(final_oof[train_rows])]
                y_final_oof = snapshot.iloc[final_oof_rows][TARGET].astype(int).to_numpy()
                final_second_target = second_residual_target(
                    y_true=y_final_oof,
                    expert=snapshot.iloc[final_oof_rows]["expert_anchor"].to_numpy(dtype=float),
                    first_layer_oof=final_oof[final_oof_rows],
                )
                final_second_weights, _ = class_weights(y_final_oof)
                final_second = new_residual_model(parameters)
                final_second.fit(
                    features[final_oof_rows],
                    final_second_target,
                    sample_weight=final_second_weights,
                )
                second_test = final_second.predict(features[model_test_rows]).astype(float)

            residual_score = (
                supported["expert_anchor"].to_numpy(dtype=float)
                + first_test
                + second_test
            )
            if not np.isfinite(residual_score).all():
                raise ValueError(f"Non-finite residual score for {scope_name} {target_month}")
            selected_threshold = float(selected["selected_threshold"])
            supported["residual_score"] = residual_score
            supported["selected_threshold"] = selected_threshold
            supported["selected_layer_count"] = int(selected["layer_count"])
            supported["residual_xgboost_prediction"] = (
                residual_score >= selected_threshold
            ).astype(int)
            output = supported[
                [
                    "scope",
                    "horizon_months",
                    KEY,
                    "target_month",
                    "forecast_origin_month",
                    TARGET,
                    "fewsnet_target_phase",
                    "expert_source_month",
                    "expert_projection_phase",
                    "expert_anchor",
                    "binary_xgboost_prediction",
                    "georf_v5_prediction",
                    "residual_score",
                    "selected_threshold",
                    "selected_layer_count",
                    "residual_xgboost_prediction",
                ]
            ].rename(columns={TARGET: "y_true"})
            prediction_frames.append(output)

            truth = supported[TARGET].astype(int).to_numpy()
            for model_name, prediction_column in COMPARISON_MODELS.items():
                predicted = supported[prediction_column].astype(int).to_numpy()
                metrics = binary_metrics(truth, predicted)
                metric_rows.append(
                    {
                        "scope": scope_name,
                        "horizon_months": horizon,
                        "target_month": target_month,
                        "model": model_name,
                        "suppressed": False,
                        "coverage_n": int(len(supported)),
                        "coverage_total": cohort_n,
                        "coverage": float(len(supported) / cohort_n),
                        **metrics,
                        "balanced_accuracy": float(
                            balanced_accuracy_score(truth, predicted)
                        ),
                        "selected_layer_count": (
                            int(selected["layer_count"])
                            if model_name == "residual_xgboost"
                            else np.nan
                        ),
                    }
                )
            fold_audits.append(
                {
                    "scope": scope_name,
                    "horizon_months": horizon,
                    **fold_audit,
                    "coverage": coverage,
                    "suppressed": False,
                    "model_fitted": True,
                    "candidate_count": len(candidates),
                    "selected_parameters": parameters,
                    "selected_layer_count": int(selected["layer_count"]),
                    "selected_threshold": selected_threshold,
                    "fit_weight_audit": fit_weight_audit,
                    "refit_weight_audit": train_weight_audit,
                    "final_layer2_oof_rows": int(len(final_oof_rows)),
                    "final_layer2_oof_audits": final_oof_audits,
                }
            )

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["scope", "target_month", KEY]
    )
    tuning = pd.DataFrame(tuning_rows).sort_values(
        ["scope", "target_month", "layer_count", "max_depth", "min_child_weight", "n_estimators"]
    )
    monthly = pd.DataFrame(metric_rows).sort_values(
        ["scope", "target_month", "model"]
    )
    available = monthly.loc[~monthly["suppressed"]]
    metric_columns = ["precision", "recall", "f1", "balanced_accuracy"]
    summary = available.groupby(
        ["scope", "horizon_months", "model"], as_index=False
    )[metric_columns].mean()
    eligible_months = (
        available.groupby(["scope", "horizon_months", "model"])
        .size()
        .rename("eligible_months")
        .reset_index()
    )
    summary = summary.merge(
        eligible_months,
        on=["scope", "horizon_months", "model"],
        validate="one_to_one",
    ).sort_values(["scope", "model"])

    selected_rows = tuning.loc[tuning["selected"]]
    if len(monthly) != len(SCOPES) * len(TARGET_MONTHS) * model_count:
        raise ValueError("Monthly metric row-count contract failed")
    if len(tuning) != len(selected_rows) * len(GRID) * 2:
        raise ValueError("Residual tuning row-count contract failed")
    if not tuning.groupby(["scope", "target_month"]).size().eq(len(GRID) * 2).all():
        raise ValueError("Residual candidate-count contract failed")
    if len(predictions) != 41_600 or predictions.duplicated(["scope", KEY, "target_month"]).any():
        raise ValueError("Residual prediction key/count contract failed")
    if len(selected_rows) != 40 or len(summary) != 16 or not summary["eligible_months"].eq(10).all():
        raise ValueError("Residual fold or summary contract failed")
    if not np.array_equal(
        predictions["residual_xgboost_prediction"].to_numpy(),
        (predictions["residual_score"] >= predictions["selected_threshold"]).astype(int).to_numpy(),
    ):
        raise ValueError("Exported residual threshold contract failed")

    outputs = {
        "predictions.csv": predictions,
        "tuning_results.csv": tuning,
        "metrics_monthly.csv": monthly,
        "metrics_summary.csv": summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(run_dir / name, index=False)
    plot_summary(summary, run_dir / "comparison_by_scope.png")

    hashes_after = {name: sha256_file(path) for name, path in source_paths.items()}
    if hashes_after != hashes_before:
        raise RuntimeError("Frozen input changed during the residual experiment")
    artifact_names = (*outputs, "comparison_by_scope.png")
    artifact_hashes = {
        name: sha256_file(run_dir / name)
        for name in artifact_names
    }
    write_json(
        run_dir / "run_metadata.json",
        {
            "run_id": args.run_id,
            "created_at": datetime.now().isoformat(),
            "command": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "xgboost_version": __import__("xgboost").__version__,
            "v5_run": str(v5_dir),
            "binary_run": str(binary_dir),
            "random_seed": SEED,
            "target": "fews_ipc >= 3",
            "predictor_count": len(predictors),
            "residual_feature_count": len(predictors) + 1,
            "predictors": predictors,
            "expert_mapping": {
                "fs0": "fews_proj_near at T-4 (latest release by origin T-1)",
                "fs1": "fews_proj_near at T-4",
                "fs2": "fews_proj_med at T-8",
                "fs3": "fews_proj_med issued at T-12 for T-4; residual bridges four months",
            },
            "residual_formula": "score = expert_anchor + residual_1 + optional residual_2",
            "objective": "reg:squarederror",
            "native_missing_handling": True,
            "median_imputation": False,
            "training_window_months": TRAIN_MONTHS,
            "validation_months": VALIDATION_MONTHS,
            "temporal_oof_policy": "group by target month; train labels strictly before pseudo-origin m-H; 36-month rolling window; exclude 2021-06; never use in-sample layer-1 predictions for layer 2",
            "threshold_policy": "validation scores rounded to 2 decimals plus 0.50; no [0,1] clipping; maximize crisis F1; higher threshold breaks ties",
            "sample_weight": "normalized sqrt(n / (2 * n_k)) from true binary class",
            "coverage_policy": {
                "threshold": MIN_COVERAGE,
                "denominator": EXPECTED_ADMINS,
                "single_suppression_flag": True,
                "xgb_only_fallback": False,
            },
            "grid": [
                {
                    "max_depth": depth,
                    "min_child_weight": child_weight,
                    "n_estimators": estimators,
                }
                for depth, child_weight, estimators in GRID
            ],
            "fold_audits": fold_audits,
            "rows": {name: len(frame) for name, frame in outputs.items()},
            "input_hashes_before": hashes_before,
            "input_hashes_after": hashes_after,
            "protected_inputs_unchanged": True,
            "artifact_hashes": artifact_hashes,
            "artifacts": [*artifact_names, "run_metadata.json"],
        },
    )
    print("\nEqual-month comparison metrics:")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    """Parse the isolated residual experiment command line."""
    experiment_root = Path(__file__).resolve().parent
    source_root = REPO_ROOT.parents[2] / "1.Source Data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v5-run",
        type=Path,
        default=experiment_root
        / "outputs"
        / "local_partition_experiment"
        / "eth_aligned_refit_20260904_seed5_v5",
    )
    parser.add_argument(
        "--binary-run",
        type=Path,
        default=experiment_root
        / "outputs"
        / "local_partition_experiment"
        / "eth_binary_xgb_20260904_seed5_v2",
    )
    parser.add_argument(
        "--fewsnet",
        type=Path,
        default=source_root / "Outcome" / "FEWSNET_IPC" / "FEWSNET.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=experiment_root / "outputs" / "local_partition_experiment",
    )
    parser.add_argument(
        "--run-id",
        default="eth_fewsnet_residual_xgb_20260904_seed5_v1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    completed = run_experiment(parse_args())
    print(f"Experiment complete: {completed}")
