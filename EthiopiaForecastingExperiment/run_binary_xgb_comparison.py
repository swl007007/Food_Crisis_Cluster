#!/usr/bin/env python3
"""Fit the isolated Ethiopia binary XGBoost comparison."""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EthiopiaForecastingExperiment.run_local_partition_experiment import (
    SCOPES,
    TARGET_MONTHS,
    binary_metrics,
    calendar_join_fewsnet,
    create_run_directory,
    normalize_admin_codes,
    sha256_file,
    write_json,
)
from EthiopiaForecastingExperiment.aligned_refit import validate_aligned_frame
from EthiopiaForecastingExperiment.run_stage3_aligned import _select_threshold


KEY = "FEWSNET_admin_code"
TARGET = "fews_ipc_crisis"
SEED = 5
TRAIN_MONTHS = 36
VALIDATION_MONTHS = 6
SUPPRESSED_MONTH = "2021-06"
MIN_FEWS_COVERAGE = 0.90
GRID = tuple(itertools.product((3, 6), (1, 5), (200, 400)))
METADATA_COLUMNS = (
    "scope",
    "horizon_months",
    KEY,
    "target_month",
    "forecast_origin_month",
    TARGET,
)


def class_weights(labels: Sequence[int]) -> tuple[np.ndarray, dict[str, object]]:
    """Return normalized binary square-root inverse-frequency row weights."""
    y = np.asarray(labels, dtype=int)
    if y.ndim != 1 or y.size == 0 or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Binary labels must be a non-empty zero/one vector")
    counts = {value: int(np.sum(y == value)) for value in (0, 1)}
    if not all(counts.values()):
        raise ValueError("Binary training subset must contain both classes")
    raw = {value: float(np.sqrt(len(y) / (2 * count))) for value, count in counts.items()}
    weights = np.asarray([raw[int(value)] for value in y], dtype=float)
    weights /= weights.mean()
    return weights, {
        "class_0_n": counts[0],
        "class_1_n": counts[1],
        "class_0_raw_weight": raw[0],
        "class_1_raw_weight": raw[1],
        "class_0_normalized_weight": float(weights[np.flatnonzero(y == 0)[0]]),
        "class_1_normalized_weight": float(weights[np.flatnonzero(y == 1)[0]]),
        "mean_weight": float(weights.mean()),
    }


def make_model(*, max_depth: int, min_child_weight: int, n_estimators: int) -> XGBClassifier:
    """Construct the frozen binary XGBoost candidate."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        n_estimators=n_estimators,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=1,
    )


def split_fold(
    frame: pd.DataFrame,
    *,
    target_month: str,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Build one validation-only selection fold and its audit."""
    dates = pd.to_datetime(frame["target_month"], errors="raise").dt.to_period("M")
    target = pd.Period(target_month, freq="M")
    train_end = target - horizon
    train_start = train_end - TRAIN_MONTHS
    training = dates.ge(train_start) & dates.lt(train_end) & frame[TARGET].notna()
    months = np.sort(dates.loc[training].unique())
    if len(months) <= VALIDATION_MONTHS:
        raise ValueError(f"Too few eligible training months for {target_month}")
    validation_months = set(months[-VALIDATION_MONTHS:])
    validation = training & dates.isin(validation_months)
    fit = training & ~validation
    test = dates.eq(target)
    fit_rows, validation_rows, test_rows = (
        np.flatnonzero(mask.to_numpy()) for mask in (fit, validation, test)
    )
    if not len(fit_rows) or not len(validation_rows) or not len(test_rows):
        raise ValueError(f"Empty fold split for {target_month}")
    return fit_rows, validation_rows, test_rows, {
        "target_month": str(target),
        "train_start": str(train_start),
        "train_end_exclusive": str(train_end),
        "eligible_training_months": int(len(months)),
        "validation_months": [str(value) for value in sorted(validation_months)],
        "fit_rows": int(len(fit_rows)),
        "validation_rows": int(len(validation_rows)),
        "test_rows": int(len(test_rows)),
        "test_labels_used_in_selection": False,
    }


def candidate_order(row: dict[str, object]) -> tuple[float, int, int, int]:
    """Apply validation F1 followed by the confirmed parameter tie-break."""
    return (
        -float(row["validation_f1"]),
        int(row["max_depth"]),
        -int(row["min_child_weight"]),
        int(row["n_estimators"]),
    )


def metric_record(
    *,
    track: str,
    scope: str,
    horizon: int,
    target_month: str,
    model: str,
    status: str,
    available: int,
    total: int,
    y_true: Sequence[int] | None = None,
    y_pred: Sequence[int] | None = None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Build one monthly metric row and its confusion-count record."""
    row: dict[str, object] = {
        "track": track,
        "scope": scope,
        "horizon_months": horizon,
        "target_month": target_month,
        "model": model,
        "status": status,
        "coverage_n": available,
        "coverage_total": total,
        "coverage": available / total if total else 0.0,
    }
    if y_true is None or y_pred is None:
        return row, None
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    row.update(binary_metrics(true, pred))
    row["balanced_accuracy"] = float(balanced_accuracy_score(true, pred))
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=(0, 1)).ravel()
    confusion = {
        "track": track,
        "scope": scope,
        "target_month": target_month,
        "model": model,
        "n": int(len(true)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return row, confusion


def add_comparison(
    metrics: list[dict[str, object]],
    confusions: list[dict[str, object]],
    *,
    track: str,
    scope: str,
    horizon: int,
    target_month: str,
    truth: Sequence[int],
    predictions: dict[str, Sequence[int]],
    total: int,
) -> None:
    """Append several models evaluated on one identical support."""
    for model, predicted in predictions.items():
        row, confusion = metric_record(
            track=track,
            scope=scope,
            horizon=horizon,
            target_month=target_month,
            model=model,
            status="available",
            available=len(truth),
            total=total,
            y_true=truth,
            y_pred=predicted,
        )
        metrics.append(row)
        if confusion is not None:
            confusions.append(confusion)


def summarize(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return primary horizon summaries with equal target-month weight."""
    metric_columns = ("precision", "recall", "f1", "balanced_accuracy")
    available = monthly.loc[monthly["status"].eq("available")]
    summary = available.groupby(["track", "scope", "model"], as_index=False)[list(metric_columns)].mean()
    months = (
        available.groupby(["track", "scope", "model"])
        .size()
        .rename("eligible_months")
        .reset_index()
    )
    return summary.merge(months, on=["track", "scope", "model"], validate="one_to_one")


def run_binary_xgb_comparison(args: argparse.Namespace) -> Path:
    """Fit all eligible folds and write the minimal comparison bundle."""
    v5_dir = args.v5_run.resolve()
    fewsnet_path = args.fewsnet.resolve()
    manifest_path = v5_dir / "run_manifest.json"
    if not manifest_path.is_file() or not fewsnet_path.is_file():
        raise FileNotFoundError("Required v5 run manifest or FEWS NET source is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictors = list(manifest["predictors"])
    if len(predictors) != 88 or manifest["scopes"] != {f"fs{k}": value for k, value in SCOPES.items()}:
        raise ValueError("Frozen v5 predictor or scope contract mismatch")

    sources = {
        "script": Path(__file__).resolve(),
        "aligned_contract_code": Path(__file__).resolve().parent / "aligned_refit.py",
        "comparison_code": Path(__file__).resolve().parent / "run_local_partition_experiment.py",
        "threshold_code": Path(__file__).resolve().parent / "run_stage3_aligned.py",
        "v5_manifest": manifest_path,
        "fewsnet": fewsnet_path,
        **{
            f"snapshot_fs{scope}": v5_dir / "input" / f"ethiopia_panel_fs{scope}_88.csv"
            for scope in SCOPES
        },
        **{
            f"georf_fs{scope}": v5_dir / "stage3" / f"fs{scope}" / "predictions_monthly.csv"
            for scope in SCOPES
        },
    }
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes_before = {name: sha256_file(path) for name, path in sources.items()}
    for scope in SCOPES:
        if hashes_before[f"snapshot_fs{scope}"] != manifest["snapshot_hashes"][f"fs{scope}"]:
            raise ValueError(f"Frozen fs{scope} snapshot hash drift")

    run_dir = create_run_directory(args.output_root.resolve(), args.run_id)
    fewsnet = pd.read_csv(fewsnet_path, low_memory=False)
    prediction_frames: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    fold_audits: list[dict[str, object]] = []

    for scope, horizon in SCOPES.items():
        scope_name = f"fs{scope}"
        snapshot = pd.read_csv(sources[f"snapshot_fs{scope}"], low_memory=False)
        validate_aligned_frame(snapshot, scope_name, horizon, predictors)
        snapshot[KEY] = normalize_admin_codes(snapshot[KEY]).to_numpy()
        for column in ("target_month", "forecast_origin_month"):
            snapshot[column] = pd.to_datetime(snapshot[column], errors="raise").dt.to_period("M").dt.to_timestamp()
        if snapshot.duplicated([KEY, "target_month"]).any():
            raise ValueError(f"Duplicate snapshot keys for {scope_name}")
        features = snapshot[predictors].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if np.isinf(features).any():
            raise ValueError(f"Infinite predictor value for {scope_name}")

        georf = pd.read_csv(sources[f"georf_fs{scope}"])
        georf[KEY] = normalize_admin_codes(georf[KEY]).to_numpy()
        georf["target_month"] = pd.to_datetime(georf.pop("month_start"), errors="raise").dt.to_period("M").dt.to_timestamp()
        georf = georf[[KEY, "target_month", "y_true", "y_pred_partitioned_thresholded"]]
        if georf.duplicated([KEY, "target_month"]).any():
            raise ValueError(f"Duplicate GeoRF keys for {scope_name}")

        for target_month in TARGET_MONTHS:
            fit_rows, validation_rows, test_rows, fold_audit = split_fold(
                snapshot,
                target_month=target_month,
                horizon=horizon,
            )
            observed_test = snapshot.iloc[test_rows][TARGET].notna()
            if target_month == SUPPRESSED_MONTH:
                if int(observed_test.sum()) != 1:
                    raise ValueError(
                        f"{scope_name} {SUPPRESSED_MONTH} expected one observed target, "
                        f"got {int(observed_test.sum())}"
                    )
                fold_audits.append(
                    {
                        "scope": scope_name,
                        "horizon_months": horizon,
                        **fold_audit,
                        "train_rows": int(len(fit_rows) + len(validation_rows)),
                        "observed_test_rows": int(observed_test.sum()),
                        "suppressed": True,
                        "suppression_reason": "only_one_observed_target",
                        "model_fitted": False,
                    }
                )
                tracks = [("georf_common", ("xgboost", "georf_v5"))]
                if scope in (1, 2):
                    tracks.append(("fews_common", ("xgboost", "georf_v5", "fewsnet")))
                for track, models in tracks:
                    for model_name in models:
                        row, _ = metric_record(
                            track=track,
                            scope=scope_name,
                            horizon=horizon,
                            target_month=target_month,
                            model=model_name,
                            status="suppressed_low_target_coverage",
                            available=int(observed_test.sum()),
                            total=len(test_rows),
                        )
                        metric_rows.append(row)
                continue

            y_fit = snapshot.iloc[fit_rows][TARGET].astype(int).to_numpy()
            y_validation = snapshot.iloc[validation_rows][TARGET].astype(int).to_numpy()
            fit_weights, fit_weight_audit = class_weights(y_fit)
            candidates: list[dict[str, object]] = []
            for max_depth, min_child_weight, n_estimators in GRID:
                model = make_model(
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    n_estimators=n_estimators,
                )
                model.fit(features[fit_rows], y_fit, sample_weight=fit_weights)
                probability = model.predict_proba(features[validation_rows])[:, 1]
                threshold = _select_threshold(y_validation, probability)
                candidates.append(
                    {
                        "scope": scope_name,
                        "target_month": target_month,
                        "max_depth": max_depth,
                        "min_child_weight": min_child_weight,
                        "n_estimators": n_estimators,
                        "learning_rate": 0.05,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "random_seed": SEED,
                        **threshold,
                    }
                )
            selected = min(candidates, key=candidate_order)
            for row in candidates:
                row["selected"] = all(
                    row[name] == selected[name]
                    for name in ("max_depth", "min_child_weight", "n_estimators")
                )
            tuning_rows.extend(candidates)

            train_rows = np.sort(np.concatenate([fit_rows, validation_rows]))
            y_train = snapshot.iloc[train_rows][TARGET].astype(int).to_numpy()
            train_weights, train_weight_audit = class_weights(y_train)
            final_model = make_model(
                max_depth=int(selected["max_depth"]),
                min_child_weight=int(selected["min_child_weight"]),
                n_estimators=int(selected["n_estimators"]),
            )
            final_model.fit(features[train_rows], y_train, sample_weight=train_weights)
            probability = final_model.predict_proba(features[test_rows])[:, 1]
            threshold_value = float(selected["selected_threshold"])
            predicted = (probability >= threshold_value).astype(int)

            prediction = snapshot.iloc[test_rows][
                [KEY, "target_month", "forecast_origin_month", TARGET]
            ].copy()
            prediction.insert(0, "scope", scope_name)
            prediction.insert(1, "horizon_months", horizon)
            prediction["xgboost_probability"] = probability
            prediction["selected_threshold"] = threshold_value
            prediction["xgboost_prediction"] = predicted
            prediction = prediction.merge(
                georf,
                on=[KEY, "target_month"],
                how="left",
                validate="one_to_one",
            )
            if prediction[[TARGET, "y_true", "y_pred_partitioned_thresholded"]].isna().any().any():
                raise ValueError(f"Missing target or GeoRF comparison value for {scope_name} {target_month}")
            if not np.array_equal(prediction[TARGET].astype(int), prediction["y_true"].astype(int)):
                raise ValueError(f"Snapshot and GeoRF truth disagree for {scope_name} {target_month}")

            coverage = {"available": 0, "total": len(prediction), "fraction": 0.0}
            if scope in (1, 2):
                baseline, coverage = calendar_join_fewsnet(
                    fewsnet,
                    target_month=target_month,
                    scope=scope,
                    cohort_codes=prediction[KEY],
                )
                prediction = prediction.merge(
                    baseline[[KEY, "y_true_fewsnet", "y_pred_fewsnet"]],
                    on=KEY,
                    how="left",
                    validate="one_to_one",
                )
            else:
                prediction["y_true_fewsnet"] = np.nan
                prediction["y_pred_fewsnet"] = np.nan
            prediction_frames.append(prediction)

            truth = prediction[TARGET].astype(int).to_numpy()
            add_comparison(
                metric_rows,
                confusion_rows,
                track="georf_common",
                scope=scope_name,
                horizon=horizon,
                target_month=target_month,
                truth=truth,
                predictions={
                    "xgboost": prediction["xgboost_prediction"].astype(int),
                    "georf_v5": prediction["y_pred_partitioned_thresholded"].astype(int),
                },
                total=len(prediction),
            )
            common = prediction[[TARGET, "y_true_fewsnet", "y_pred_fewsnet"]].notna().all(axis=1)
            common_coverage = {
                "available": int(common.sum()),
                "total": len(prediction),
                "fraction": float(common.mean()),
            }
            if scope in (1, 2) and common_coverage["fraction"] >= MIN_FEWS_COVERAGE:
                supported = prediction.loc[common]
                if not np.array_equal(supported[TARGET].astype(int), supported["y_true_fewsnet"].astype(int)):
                    raise ValueError(f"FEWS NET truth mismatch for {scope_name} {target_month}")
                add_comparison(
                    metric_rows,
                    confusion_rows,
                    track="fews_common",
                    scope=scope_name,
                    horizon=horizon,
                    target_month=target_month,
                    truth=supported[TARGET].astype(int),
                    predictions={
                        "xgboost": supported["xgboost_prediction"].astype(int),
                        "georf_v5": supported["y_pred_partitioned_thresholded"].astype(int),
                        "fewsnet": supported["y_pred_fewsnet"].astype(int),
                    },
                    total=len(prediction),
                )
            elif scope in (1, 2):
                for model_name in ("xgboost", "georf_v5", "fewsnet"):
                    row, _ = metric_record(
                        track="fews_common",
                        scope=scope_name,
                        horizon=horizon,
                        target_month=target_month,
                        model=model_name,
                        status="suppressed_low_fewsnet_coverage",
                        available=int(common_coverage["available"]),
                        total=int(common_coverage["total"]),
                    )
                    metric_rows.append(row)

            fold_audits.append(
                {
                    "scope": scope_name,
                    "horizon_months": horizon,
                    **fold_audit,
                    "train_rows": int(len(train_rows)),
                    "observed_test_rows": int(observed_test.sum()),
                    "suppressed": False,
                    "model_fitted": True,
                    "selected_parameters": {
                        name: selected[name]
                        for name in ("max_depth", "min_child_weight", "n_estimators")
                    },
                    "selected_threshold": threshold_value,
                    "fit_weight_audit": fit_weight_audit,
                    "refit_weight_audit": train_weight_audit,
                    "fewsnet_projection_coverage": coverage,
                    "fewsnet_common_coverage": common_coverage,
                }
            )

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["scope", "target_month", KEY]
    )
    tuning = pd.DataFrame(tuning_rows).sort_values(
        ["scope", "target_month", "max_depth", "min_child_weight", "n_estimators"]
    )
    selected = tuning.loc[tuning["selected"]].reset_index(drop=True)
    monthly = pd.DataFrame(metric_rows).sort_values(["track", "scope", "target_month", "model"])
    confusions = pd.DataFrame(confusion_rows).sort_values(["track", "scope", "target_month", "model"])
    summary = summarize(monthly)
    if len(predictions) != 45_760 or predictions.duplicated(["scope", KEY, "target_month"]).any():
        raise ValueError("Prediction key/count contract failed")
    if len(tuning) != 352 or len(selected) != 44:
        raise ValueError("Candidate or selected-parameter count contract failed")
    if not tuning.groupby(["scope", "target_month"]).size().eq(8).all():
        raise ValueError("Eight-candidate fold contract failed")

    outputs = {
        "predictions.csv": predictions,
        "tuning_results.csv": tuning,
        "selected_parameters.csv": selected,
        "metrics_monthly.csv": monthly,
        "metrics_summary.csv": summary,
        "confusion_counts.csv": confusions,
    }
    for name, frame in outputs.items():
        frame.to_csv(run_dir / name, index=False)

    hashes_after = {name: sha256_file(path) for name, path in sources.items()}
    if hashes_after != hashes_before:
        raise RuntimeError("Frozen input changed during the experiment")
    hashes = {
        **{f"input_{name}": digest for name, digest in hashes_before.items()},
        **{
            f"artifact_{Path(name).stem}": sha256_file(run_dir / name)
            for name in outputs
        },
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
            "random_seed": SEED,
            "predictor_count": len(predictors),
            "predictors": predictors,
            "objective": "binary:logistic",
            "native_missing_handling": True,
            "median_imputation": False,
            "smote": False,
            "training_window_months": TRAIN_MONTHS,
            "validation_months": VALIDATION_MONTHS,
            "threshold_policy": "Each XGBoost candidate's validation probabilities are rounded to 2 decimals; unique observed values within [0.05, 0.95] are scored; highest threshold breaks F1 ties; fallback 0.5",
            "sample_weight": "normalized sqrt(n / (2 * n_k)); fit/refit folds only",
            "suppressed_target_month": SUPPRESSED_MONTH,
            "fold_audits": fold_audits,
            "rows": {name: len(frame) for name, frame in outputs.items()},
            "hashes": hashes,
        },
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    """Parse the isolated experiment arguments."""
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
        "--fewsnet",
        type=Path,
        default=source_root / "Outcome" / "FEWSNET_IPC" / "FEWSNET.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=experiment_root / "outputs" / "local_partition_experiment",
    )
    parser.add_argument("--run-id", default="eth_binary_xgb_20260904_seed5_v2")
    return parser.parse_args()


if __name__ == "__main__":
    completed = run_binary_xgb_comparison(parse_args())
    print(f"Experiment complete: {completed}")
