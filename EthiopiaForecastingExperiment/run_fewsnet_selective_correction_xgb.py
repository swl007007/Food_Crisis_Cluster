#!/usr/bin/env python3
"""Fit the conservative Ethiopia FEWS NET selective-correction gate."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EthiopiaForecastingExperiment.aligned_refit import validate_aligned_frame
from EthiopiaForecastingExperiment.run_fewsnet_residual_xgb import (
    EXPECTED_ADMINS,
    GRID,
    KEY,
    MIN_COVERAGE,
    SEED,
    TARGET,
    attach_expert_anchor,
    fold_coverage,
    split_residual_fold,
)
from EthiopiaForecastingExperiment.run_local_partition_experiment import (
    SCOPES,
    TARGET_MONTHS,
    binary_metrics,
    create_run_directory,
    normalize_admin_codes,
    sha256_file,
    write_json,
)


RUN_ID = "eth_fewsnet_selective_correction_xgb_20260904_seed5_v1"
MIN_CORRECTION_PRECISION = 0.75
MIN_DIRECTION_FLIPS = 20
MIN_DIRECTION_MONTHS = 2
COMPARISON_MODELS = {
    "fewsnet": "expert_anchor",
    "binary_xgboost": "binary_xgboost_prediction",
    "georf_v5": "georf_v5_prediction",
    "selective_correction_xgboost": "selective_correction_prediction",
}


def candidate_order(row: dict[str, object]) -> tuple[float, int, int, int]:
    """Prefer validation F1, then expert-only, then fixed enumeration order."""
    return (
        -float(row["validation_f1"]),
        0 if bool(row["expert_only"]) else 1,
        int(row["grid_order"]),
        int(row["threshold_order"]),
    )


def apply_correction_rule(
    expert: np.ndarray,
    score: np.ndarray,
    *,
    threshold: float,
    enable_0_to_1: bool,
    enable_1_to_0: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a frozen threshold and direction mask without consulting outcomes."""
    calls = np.asarray(expert, dtype=int)
    values = np.asarray(score, dtype=float)
    if calls.ndim != 1 or values.ndim != 1 or calls.shape != values.shape:
        raise ValueError("Expert calls and wrong-call scores must be aligned vectors")
    if not set(np.unique(calls)).issubset({0, 1}) or not np.isfinite(values).all():
        raise ValueError("Expert calls must be binary and scores finite")
    proposed = values > float(threshold)
    applied = proposed & (
        ((calls == 0) & enable_0_to_1) | ((calls == 1) & enable_1_to_0)
    )
    prediction = calls.copy()
    prediction[applied] = 1 - prediction[applied]
    return prediction, proposed, applied


def evaluate_correction_rule(
    y_true: np.ndarray,
    expert: np.ndarray,
    score: np.ndarray,
    months: np.ndarray,
    *,
    threshold: float,
) -> dict[str, object]:
    """Evaluate one validation threshold with separate directional safety gates."""
    truth = np.asarray(y_true, dtype=int)
    calls = np.asarray(expert, dtype=int)
    values = np.asarray(score, dtype=float)
    dates = np.asarray(months).astype(str)
    if any(item.ndim != 1 for item in (truth, calls, values, dates)):
        raise ValueError("Correction-rule inputs must be one-dimensional")
    if len({item.shape for item in (truth, calls, values, dates)}) != 1:
        raise ValueError("Correction-rule inputs must be aligned")
    if not set(np.unique(truth)).issubset({0, 1}):
        raise ValueError("Validation truth must be binary")
    if not len(truth) or not np.isfinite(values).all():
        raise ValueError("Validation scores must be non-empty and finite")

    proposed = values > float(threshold)
    direction: dict[str, int | float | bool] = {}
    enabled: dict[int, bool] = {}
    for source, label in ((0, "0_to_1"), (1, "1_to_0")):
        mask = proposed & (calls == source)
        flips = int(mask.sum())
        corrections = int((mask & (truth != calls)).sum())
        precision = corrections / flips if flips else np.nan
        month_count = int(np.unique(dates[mask]).size)
        enabled[source] = bool(
            flips >= MIN_DIRECTION_FLIPS
            and month_count >= MIN_DIRECTION_MONTHS
            and precision >= MIN_CORRECTION_PRECISION
        )
        direction.update(
            {
                f"proposed_flips_{label}": flips,
                f"validation_corrections_{label}": corrections,
                f"correction_precision_{label}": precision,
                f"flip_months_{label}": month_count,
                f"enable_{label}": enabled[source],
            }
        )

    prediction, _, applied = apply_correction_rule(
        calls,
        values,
        threshold=threshold,
        enable_0_to_1=enabled[0],
        enable_1_to_0=enabled[1],
    )
    corrections = int((applied & (truth != calls)).sum())
    damage = int((applied & (truth == calls)).sum())
    metrics = binary_metrics(truth, prediction)
    return {
        "selected_threshold": float(threshold),
        **direction,
        "applied_flips_0_to_1": int((applied & (calls == 0)).sum()),
        "applied_flips_1_to_0": int((applied & (calls == 1)).sum()),
        "validation_corrections": corrections,
        "validation_damage": damage,
        "validation_net_corrections": corrections - damage,
        "validation_correction_precision": corrections / int(applied.sum())
        if applied.any()
        else np.nan,
        "validation_precision": metrics["precision"],
        "validation_recall": metrics["recall"],
        "validation_f1": metrics["f1"],
        "validation_balanced_accuracy": float(
            balanced_accuracy_score(truth, prediction)
        ),
        "prediction": prediction,
    }


def new_gate_model(parameters: dict[str, int]) -> XGBClassifier:
    """Construct one frozen, unweighted wrong-call classifier."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=1,
        **parameters,
    )


def correction_counts(
    truth: np.ndarray,
    expert: np.ndarray,
    proposed: np.ndarray,
    applied: np.ndarray,
) -> dict[str, int | float]:
    """Summarize proposed, beneficial, and harmful expert flips."""
    y = np.asarray(truth, dtype=int)
    e = np.asarray(expert, dtype=int)
    proposed_mask = np.asarray(proposed, dtype=bool)
    applied_mask = np.asarray(applied, dtype=bool)
    corrections = int((applied_mask & (y != e)).sum())
    damage = int((applied_mask & (y == e)).sum())
    flips = int(applied_mask.sum())
    return {
        "proposed_flips": int(proposed_mask.sum()),
        "applied_flips": flips,
        "corrections_of_expert_errors": corrections,
        "damage_to_correct_expert_calls": damage,
        "net_corrections": corrections - damage,
        "correction_precision": corrections / flips if flips else np.nan,
    }


def run_experiment(args: argparse.Namespace) -> Path:
    """Fit all eligible folds and write the isolated five-file bundle."""
    v5_dir = args.v5_run.resolve()
    binary_dir = args.binary_run.resolve()
    fewsnet_path = args.fewsnet.resolve()
    output_root = args.output_root.resolve()
    run_path = output_root / RUN_ID
    if run_path.exists():
        raise FileExistsError(run_path)

    v5_manifest_path = v5_dir / "run_manifest.json"
    binary_metadata_path = binary_dir / "run_metadata.json"
    binary_predictions_path = binary_dir / "predictions.csv"
    for path in (
        v5_manifest_path,
        binary_metadata_path,
        binary_predictions_path,
        fewsnet_path,
    ):
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

    predecessor_dir = (
        Path(__file__).resolve().parent
        / "outputs"
        / "local_partition_experiment"
        / "eth_fewsnet_residual_xgb_20260904_seed5_v2"
    )
    predecessor_artifacts = (
        "predictions.csv",
        "tuning_results.csv",
        "metrics_monthly.csv",
        "metrics_summary.csv",
        "comparison_by_scope.png",
        "run_metadata.json",
    )
    source_paths = {
        "script": Path(__file__).resolve(),
        "predecessor_code": Path(__file__).resolve().parent
        / "run_fewsnet_residual_xgb.py",
        "aligned_contract_code": Path(__file__).resolve().parent / "aligned_refit.py",
        "binary_code": Path(__file__).resolve().parent
        / "run_binary_xgb_comparison.py",
        "comparison_code": Path(__file__).resolve().parent
        / "run_local_partition_experiment.py",
        "v5_manifest": v5_manifest_path,
        "binary_metadata": binary_metadata_path,
        "binary_predictions": binary_predictions_path,
        "fewsnet": fewsnet_path,
        **{
            f"snapshot_fs{scope}": v5_dir
            / "input"
            / f"ethiopia_panel_fs{scope}_88.csv"
            for scope in SCOPES
        },
        **{
            f"predecessor_v2_{name}": predecessor_dir / name
            for name in predecessor_artifacts
        },
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes_before = {name: sha256_file(path) for name, path in source_paths.items()}
    for scope in SCOPES:
        if (
            hashes_before[f"snapshot_fs{scope}"]
            != v5_manifest["snapshot_hashes"][f"fs{scope}"]
        ):
            raise ValueError(f"Frozen fs{scope} snapshot hash drift")
    frozen_hashes = binary_metadata["hashes"]
    for name, expected in {
        "v5_manifest": frozen_hashes["input_v5_manifest"],
        "binary_predictions": frozen_hashes["artifact_predictions"],
        "fewsnet": frozen_hashes["input_fewsnet"],
    }.items():
        if hashes_before[name] != expected:
            raise ValueError(f"Frozen {name} hash drift")

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
    missing = required_comparator.difference(comparator.columns)
    if missing:
        raise ValueError(f"Binary comparator missing columns: {sorted(missing)}")
    comparator[KEY] = normalize_admin_codes(comparator[KEY]).to_numpy()
    for column in ("target_month", "forecast_origin_month"):
        comparator[column] = (
            pd.to_datetime(comparator[column], errors="raise")
            .dt.to_period("M")
            .dt.to_timestamp()
        )
    if comparator.duplicated(["scope", KEY, "target_month"]).any():
        raise ValueError("Duplicate frozen binary comparator keys")
    if not np.array_equal(
        comparator[TARGET].astype(int), comparator["y_true"].astype(int)
    ):
        raise ValueError("Frozen binary and GeoRF truths disagree")
    comparator = comparator[
        [
            "scope",
            KEY,
            "target_month",
            TARGET,
            "xgboost_prediction",
            "y_pred_partitioned_thresholded",
        ]
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

    for scope, horizon in SCOPES.items():
        scope_name = f"fs{scope}"
        snapshot = pd.read_csv(source_paths[f"snapshot_fs{scope}"], low_memory=False)
        validate_aligned_frame(snapshot, scope_name, horizon, predictors)
        snapshot[KEY] = normalize_admin_codes(snapshot[KEY]).to_numpy()
        for column in ("target_month", "forecast_origin_month"):
            snapshot[column] = (
                pd.to_datetime(snapshot[column], errors="raise")
                .dt.to_period("M")
                .dt.to_timestamp()
            )
        cohort_n = int(snapshot[KEY].nunique())
        if cohort_n != EXPECTED_ADMINS:
            raise ValueError(
                f"fs{scope} cohort has {cohort_n} admins, expected {EXPECTED_ADMINS}"
            )
        snapshot = attach_expert_anchor(snapshot, fewsnet, scope)
        observed_truth = snapshot[TARGET].notna()
        if snapshot.loc[observed_truth, "fewsnet_truth"].isna().any():
            raise ValueError(f"Raw FEWS truth is missing for observed {scope_name} targets")
        if not np.array_equal(
            snapshot.loc[observed_truth, TARGET].astype(int),
            snapshot.loc[observed_truth, "fewsnet_truth"].astype(int),
        ):
            raise ValueError(f"Snapshot and raw FEWS truths disagree for {scope_name}")

        numeric = snapshot[predictors].apply(pd.to_numeric, errors="coerce")
        coerced = snapshot[predictors].notna() & numeric.isna()
        if coerced.any().any():
            raise ValueError(f"Non-numeric predictor value for {scope_name}")
        base_features = numeric.to_numpy(dtype=float)
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
                coverage["suppressed"]
                or coverage["common_coverage"] < MIN_COVERAGE
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
                    raise ValueError(
                        f"Comparison truth mismatch for {scope_name} {target_month}"
                    )

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
                            "proposed_flips": np.nan,
                            "applied_flips": np.nan,
                            "corrections_of_expert_errors": np.nan,
                            "damage_to_correct_expert_calls": np.nan,
                            "net_corrections": np.nan,
                            "correction_precision": np.nan,
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

            fit_rows, validation_rows, fold_test_rows, fold_audit = (
                split_residual_fold(
                    snapshot,
                    target_month=target_month,
                    horizon=horizon,
                )
            )
            if not np.array_equal(test_rows, fold_test_rows):
                raise ValueError(f"Test-row drift for {scope_name} {target_month}")
            y_fit = snapshot.iloc[fit_rows][TARGET].astype(int).to_numpy()
            y_validation = snapshot.iloc[validation_rows][TARGET].astype(int).to_numpy()
            expert_fit = snapshot.iloc[fit_rows]["expert_anchor"].astype(int).to_numpy()
            expert_validation = (
                snapshot.iloc[validation_rows]["expert_anchor"].astype(int).to_numpy()
            )
            wrong_fit = (y_fit != expert_fit).astype(int)
            if np.unique(wrong_fit).size != 2:
                raise ValueError(
                    f"Wrong-call fit target has one class for {scope_name} {target_month}"
                )
            validation_months = dates.iloc[validation_rows].astype(str).to_numpy()
            expert_metrics = binary_metrics(y_validation, expert_validation)
            candidates: list[dict[str, object]] = [
                {
                    "scope": scope_name,
                    "horizon_months": horizon,
                    "target_month": target_month,
                    "candidate": "expert_only",
                    "expert_only": True,
                    "grid_order": -1,
                    "threshold_order": -1,
                    "max_depth": np.nan,
                    "min_child_weight": np.nan,
                    "n_estimators": np.nan,
                    "selected_threshold": np.nan,
                    "enable_0_to_1": False,
                    "enable_1_to_0": False,
                    "validation_precision": expert_metrics["precision"],
                    "validation_recall": expert_metrics["recall"],
                    "validation_f1": expert_metrics["f1"],
                    "validation_balanced_accuracy": float(
                        balanced_accuracy_score(y_validation, expert_validation)
                    ),
                    "selected": False,
                }
            ]

            for grid_order, (max_depth, min_child_weight, n_estimators) in enumerate(
                GRID
            ):
                parameters = {
                    "max_depth": max_depth,
                    "min_child_weight": min_child_weight,
                    "n_estimators": n_estimators,
                }
                model = new_gate_model(parameters)
                model.fit(features[fit_rows], wrong_fit)
                validation_score = model.predict_proba(features[validation_rows])[:, 1]
                if not np.isfinite(validation_score).all():
                    raise ValueError(
                        f"Non-finite validation score for {scope_name} {target_month}"
                    )
                for threshold_order, threshold in enumerate(
                    np.unique(np.round(validation_score, 2))
                ):
                    evaluated = evaluate_correction_rule(
                        y_validation,
                        expert_validation,
                        validation_score,
                        validation_months,
                        threshold=float(threshold),
                    )
                    evaluated.pop("prediction")
                    candidates.append(
                        {
                            "scope": scope_name,
                            "horizon_months": horizon,
                            "target_month": target_month,
                            "candidate": "selective_correction_xgboost",
                            "expert_only": False,
                            "grid_order": grid_order,
                            "threshold_order": threshold_order,
                            **parameters,
                            "learning_rate": 0.05,
                            "subsample": 0.8,
                            "colsample_bytree": 0.8,
                            "random_seed": SEED,
                            "validation_score_min": float(validation_score.min()),
                            "validation_score_max": float(validation_score.max()),
                            **evaluated,
                            "selected": False,
                        }
                    )

            selected = min(candidates, key=candidate_order)
            selected["selected"] = True
            tuning_rows.extend(candidates)

            supported = test.loc[common].copy()
            model_test_rows = supported["_snapshot_row"].astype(int).to_numpy()
            expert_test = supported["expert_anchor"].astype(int).to_numpy()
            if bool(selected["expert_only"]):
                wrong_score = np.full(len(supported), np.nan)
                threshold = np.nan
                enable_0_to_1 = False
                enable_1_to_0 = False
                selective_prediction = expert_test.copy()
                proposed = np.zeros(len(supported), dtype=bool)
                applied = np.zeros(len(supported), dtype=bool)
                selected_parameters = None
            else:
                selected_parameters = {
                    name: int(selected[name])
                    for name in ("max_depth", "min_child_weight", "n_estimators")
                }
                train_rows = np.sort(np.concatenate([fit_rows, validation_rows]))
                y_train = snapshot.iloc[train_rows][TARGET].astype(int).to_numpy()
                expert_train = (
                    snapshot.iloc[train_rows]["expert_anchor"].astype(int).to_numpy()
                )
                wrong_train = (y_train != expert_train).astype(int)
                if np.unique(wrong_train).size != 2:
                    raise ValueError(
                        f"Wrong-call refit target has one class for {scope_name} {target_month}"
                    )
                final_model = new_gate_model(selected_parameters)
                final_model.fit(features[train_rows], wrong_train)
                wrong_score = final_model.predict_proba(features[model_test_rows])[:, 1]
                threshold = float(selected["selected_threshold"])
                enable_0_to_1 = bool(selected["enable_0_to_1"])
                enable_1_to_0 = bool(selected["enable_1_to_0"])
                selective_prediction, proposed, applied = apply_correction_rule(
                    expert_test,
                    wrong_score,
                    threshold=threshold,
                    enable_0_to_1=enable_0_to_1,
                    enable_1_to_0=enable_1_to_0,
                )

            supported["wrong_call_score"] = wrong_score
            supported["selected_threshold"] = threshold
            supported["enable_0_to_1"] = enable_0_to_1
            supported["enable_1_to_0"] = enable_1_to_0
            supported["proposed_flip"] = proposed
            supported["applied_flip"] = applied
            supported["selective_correction_prediction"] = selective_prediction
            truth = supported[TARGET].astype(int).to_numpy()
            supported["flip_fixed_expert_error"] = applied & (truth != expert_test)
            supported["flip_damaged_correct_expert_call"] = applied & (
                truth == expert_test
            )
            prediction_frames.append(
                supported[
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
                        "wrong_call_score",
                        "selected_threshold",
                        "enable_0_to_1",
                        "enable_1_to_0",
                        "proposed_flip",
                        "applied_flip",
                        "flip_fixed_expert_error",
                        "flip_damaged_correct_expert_call",
                        "selective_correction_prediction",
                    ]
                ].rename(columns={TARGET: "y_true"})
            )

            flip_audit = correction_counts(truth, expert_test, proposed, applied)
            for model_name, prediction_column in COMPARISON_MODELS.items():
                predicted = supported[prediction_column].astype(int).to_numpy()
                metrics = binary_metrics(truth, predicted)
                correction_fields = (
                    flip_audit
                    if model_name == "selective_correction_xgboost"
                    else {
                        name: np.nan
                        for name in (
                            "proposed_flips",
                            "applied_flips",
                            "corrections_of_expert_errors",
                            "damage_to_correct_expert_calls",
                            "net_corrections",
                            "correction_precision",
                        )
                    }
                )
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
                        **correction_fields,
                    }
                )
            fold_audits.append(
                {
                    "scope": scope_name,
                    "horizon_months": horizon,
                    **fold_audit,
                    "coverage": coverage,
                    "suppressed": False,
                    "model_fitted": not bool(selected["expert_only"]),
                    "candidate_count": len(candidates),
                    "selected_candidate": str(selected["candidate"]),
                    "selected_parameters": selected_parameters,
                    "selected_threshold": None
                    if bool(selected["expert_only"])
                    else threshold,
                    "enable_0_to_1": enable_0_to_1,
                    "enable_1_to_0": enable_1_to_0,
                    "test_labels_used_in_selection": False,
                }
            )

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["scope", "target_month", KEY]
    )
    tuning = pd.DataFrame(tuning_rows).sort_values(
        ["scope", "target_month", "expert_only", "grid_order", "threshold_order"],
        ascending=[True, True, False, True, True],
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
    )
    correction_summary = (
        predictions.groupby("scope", as_index=False)
        .agg(
            proposed_flips=("proposed_flip", "sum"),
            applied_flips=("applied_flip", "sum"),
            corrections_of_expert_errors=("flip_fixed_expert_error", "sum"),
            damage_to_correct_expert_calls=(
                "flip_damaged_correct_expert_call",
                "sum",
            ),
        )
        .assign(model="selective_correction_xgboost")
    )
    correction_summary["net_corrections"] = (
        correction_summary["corrections_of_expert_errors"]
        - correction_summary["damage_to_correct_expert_calls"]
    )
    correction_summary["correction_precision"] = (
        correction_summary["corrections_of_expert_errors"]
        / correction_summary["applied_flips"].replace(0, np.nan)
    )
    summary = summary.merge(
        correction_summary,
        on=["scope", "model"],
        how="left",
        validate="one_to_one",
    ).sort_values(["scope", "model"])

    selected_rows = tuning.loc[tuning["selected"]]
    if len(monthly) != len(SCOPES) * len(TARGET_MONTHS) * len(COMPARISON_MODELS):
        raise ValueError("Monthly metric row-count contract failed")
    if len(predictions) != 41_600 or predictions.duplicated(
        ["scope", KEY, "target_month"]
    ).any():
        raise ValueError("Selective-correction prediction key/count contract failed")
    if len(selected_rows) != 40 or len(summary) != 16:
        raise ValueError("Selective-correction fold or summary contract failed")
    if not summary["eligible_months"].eq(10).all():
        raise ValueError("Selective-correction equal-month contract failed")
    if not tuning.groupby(["scope", "target_month"])["selected"].sum().eq(1).all():
        raise ValueError("Selective-correction candidate selection contract failed")
    selected_corrections = selected_rows.loc[~selected_rows["expert_only"]]
    if not selected_corrections.empty:
        expert_f1 = tuning.loc[tuning["expert_only"]].set_index(
            ["scope", "target_month"]
        )["validation_f1"]
        selected_f1 = selected_corrections.set_index(["scope", "target_month"])[
            "validation_f1"
        ]
        if not (selected_f1 > expert_f1.loc[selected_f1.index]).all():
            raise ValueError("Selected correction does not strictly improve validation F1")
    if (
        predictions["applied_flip"]
        & ~predictions["proposed_flip"]
    ).any():
        raise ValueError("Applied correction bypassed the frozen score threshold")
    expected_prediction = predictions["expert_anchor"].astype(int).to_numpy()
    applied_mask = predictions["applied_flip"].to_numpy(dtype=bool)
    expected_prediction[applied_mask] = 1 - expected_prediction[applied_mask]
    if not np.array_equal(
        expected_prediction,
        predictions["selective_correction_prediction"].astype(int).to_numpy(),
    ):
        raise ValueError("Exported selective-correction contract failed")

    hashes_after = {name: sha256_file(path) for name, path in source_paths.items()}
    if hashes_after != hashes_before:
        raise RuntimeError("Frozen input changed during the selective-correction run")

    run_dir = create_run_directory(output_root, RUN_ID)
    outputs = {
        "predictions.csv": predictions,
        "tuning_results.csv": tuning,
        "metrics_monthly.csv": monthly,
        "metrics_summary.csv": summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(run_dir / name, index=False)
    write_json(
        run_dir / "run_metadata.json",
        {
            "run_id": RUN_ID,
            "created_at": datetime.now().isoformat(),
            "command": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "xgboost_version": __import__("xgboost").__version__,
            "v5_run": str(v5_dir),
            "binary_run": str(binary_dir),
            "random_seed": SEED,
            "target": "expert_wrong = (fews_ipc >= 3) != expert_anchor",
            "predictor_count": len(predictors),
            "gate_feature_count": len(predictors) + 1,
            "predictors": predictors,
            "objective": "binary:logistic",
            "sample_weight": None,
            "native_missing_handling": True,
            "training_window_months": 36,
            "validation_months": 6,
            "threshold_policy": "ascending rounded unique validation wrong-call scores; raw values below 0.5 allowed",
            "direction_policy": {
                "minimum_correction_precision": MIN_CORRECTION_PRECISION,
                "minimum_flips": MIN_DIRECTION_FLIPS,
                "minimum_target_months": MIN_DIRECTION_MONTHS,
                "directions_evaluated_separately": True,
            },
            "selection_policy": "maximize validation crisis F1; expert-only wins exact ties; otherwise fixed grid then threshold enumeration order",
            "coverage_policy": {
                "threshold": MIN_COVERAGE,
                "denominator": EXPECTED_ADMINS,
                "single_suppression_flag": True,
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
            "artifacts": [*outputs, "run_metadata.json"],
        },
    )
    print("\nEqual-month comparison metrics:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return run_dir


def parse_args() -> argparse.Namespace:
    """Parse only the source and output paths; the run ID is fixed."""
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
    return parser.parse_args()


if __name__ == "__main__":
    completed = run_experiment(parse_args())
    print(f"Experiment complete: {completed}")
