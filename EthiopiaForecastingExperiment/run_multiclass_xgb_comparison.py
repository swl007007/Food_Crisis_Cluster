#!/usr/bin/env python3
"""Fit and compare the frozen Ethiopia four-class XGBoost experiment."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EthiopiaForecastingExperiment.run_local_partition_experiment import (
    SCOPES,
    TARGET_MONTHS,
    calendar_join_fewsnet,
    create_run_directory,
    normalize_admin_codes,
    sha256_file,
    write_json,
)


KEY = "FEWSNET_admin_code"
SEED = 5
TRAIN_MONTHS = 36
VALIDATION_MONTHS = 6
SUPPRESSED_MONTH = "2021-06"
MIN_COVERAGE = 0.90
GRID = tuple(itertools.product((3, 6), (1, 5), (200, 400)))
METADATA = (
    "scope",
    "horizon_months",
    KEY,
    "target_month",
    "forecast_origin_month",
    "fews_ipc_crisis",
)


def class_weights(labels: Sequence[int]) -> tuple[np.ndarray, dict[int, dict[str, float | int]]]:
    """Return normalized square-root inverse-frequency row weights."""
    y = np.asarray(labels, dtype=int)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("Class-weight labels must be a non-empty vector")
    counts = {phase: int(np.sum(y == phase)) for phase in range(4)}
    raw = {
        phase: float(np.sqrt(len(y) / (4 * count)))
        for phase, count in counts.items()
        if count
    }
    weights = np.asarray([raw[int(value)] for value in y], dtype=float)
    weights /= weights.mean()
    audit = {
        phase + 1: {
            "n": counts[phase],
            "raw_weight": raw.get(phase),
            "normalized_weight": (
                float(weights[np.flatnonzero(y == phase)[0]]) if counts[phase] else None
            ),
        }
        for phase in range(4)
    }
    return weights, audit


def candidate_order(row: dict[str, object]) -> tuple[float, int, int, int]:
    """Sort validation candidates by score and the frozen tie-break contract."""
    return (
        -float(row["validation_macro_f1"]),
        int(row["max_depth"]),
        -int(row["min_child_weight"]),
        int(row["n_estimators"]),
    )


def make_model(*, max_depth: int, min_child_weight: int, n_estimators: int) -> XGBClassifier:
    """Construct the single approved multiclass XGBoost model."""
    return XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
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
    """Return fit, validation, and test rows for one no-leak temporal fold."""
    dates = pd.to_datetime(frame["target_month"], errors="raise").dt.to_period("M")
    target = pd.Period(target_month, freq="M")
    train_end = target - horizon
    train_start = train_end - TRAIN_MONTHS
    train = dates.ge(train_start) & dates.lt(train_end) & frame["fews_ipc"].notna()
    eligible_months = np.sort(dates.loc[train].unique())
    if len(eligible_months) <= VALIDATION_MONTHS:
        raise ValueError(f"Too few eligible training months for {target_month}")
    validation_months = set(eligible_months[-VALIDATION_MONTHS:])
    validation = train & dates.isin(validation_months)
    fit = train & ~validation
    test = dates.eq(target)
    indices = tuple(np.flatnonzero(mask.to_numpy()) for mask in (fit, validation, test))
    if any(len(index) == 0 for index in indices):
        raise ValueError(f"Empty fit, validation, or test split for {target_month}")
    audit = {
        "target_month": str(target),
        "train_start": str(train_start),
        "train_end_exclusive": str(train_end),
        "window_calendar_months": TRAIN_MONTHS,
        "eligible_training_months": int(len(eligible_months)),
        "fit_rows": int(len(indices[0])),
        "validation_rows": int(len(indices[1])),
        "test_rows": int(len(indices[2])),
        "validation_months": [str(month) for month in sorted(validation_months)],
        "test_labels_used_in_selection": False,
    }
    return *indices, audit


def multiclass_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float | int]:
    """Compute the frozen four-class metric set on phases 1-4."""
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        pred,
        labels=(1, 2, 3, 4),
        zero_division=0,
    )
    result: dict[str, float | int] = {
        "n": int(len(true)),
        "macro_f1": float(f1_score(true, pred, labels=(1, 2, 3, 4), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(true, pred)),
        "ordinal_mae": float(mean_absolute_error(true, pred)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(true, pred, labels=(1, 2, 3, 4), weights="quadratic")),
    }
    for index, phase in enumerate((1, 2, 3, 4)):
        result.update(
            {
                f"phase{phase}_precision": float(precision[index]),
                f"phase{phase}_recall": float(recall[index]),
                f"phase{phase}_f1": float(f1[index]),
                f"phase{phase}_support": int(support[index]),
            }
        )
    return result


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float | int]:
    """Compute crisis-class metrics after collapsing phases at phase 3."""
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true,
        pred,
        labels=(0, 1),
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=(0, 1)).ravel()
    return {
        "n": int(len(true)),
        "crisis_precision": float(precision[1]),
        "crisis_recall": float(recall[1]),
        "crisis_f1": float(f1[1]),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def confusion_rows(
    *,
    track: str,
    scope: str,
    target_month: str,
    model: str,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int],
) -> list[dict[str, object]]:
    """Return a tidy confusion matrix for one model-month."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return [
        {
            "track": track,
            "scope": scope,
            "target_month": target_month,
            "model": model,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "count": int(matrix[row, column]),
        }
        for row, true_class in enumerate(labels)
        for column, predicted_class in enumerate(labels)
    ]


def metric_row(
    *,
    track: str,
    scope: str,
    horizon: int,
    target_month: str,
    model: str,
    support_status: str,
    coverage_n: int,
    coverage_total: int,
    metrics: dict[str, float | int] | None,
) -> dict[str, object]:
    """Build one monthly metric record."""
    row: dict[str, object] = {
        "track": track,
        "scope": scope,
        "horizon_months": horizon,
        "target_month": target_month,
        "model": model,
        "status": support_status,
        "coverage_n": coverage_n,
        "coverage_total": coverage_total,
        "coverage": coverage_n / coverage_total if coverage_total else 0.0,
    }
    if metrics:
        row.update(metrics)
    return row


def add_metrics(
    metric_rows: list[dict[str, object]],
    confusion_output: list[dict[str, object]],
    *,
    track: str,
    scope: str,
    horizon: int,
    target_month: str,
    model_predictions: dict[str, Sequence[int]],
    truth: Sequence[int],
    labels: Sequence[int],
    coverage_total: int,
) -> None:
    """Append identical-support metrics and confusion matrices for one track."""
    metric_function = multiclass_metrics if len(labels) == 4 else binary_metrics
    for model, prediction in model_predictions.items():
        metric_rows.append(
            metric_row(
                track=track,
                scope=scope,
                horizon=horizon,
                target_month=target_month,
                model=model,
                support_status="available",
                coverage_n=len(truth),
                coverage_total=coverage_total,
                metrics=metric_function(truth, prediction),
            )
        )
        confusion_output.extend(
            confusion_rows(
                track=track,
                scope=scope,
                target_month=target_month,
                model=model,
                y_true=truth,
                y_pred=prediction,
                labels=labels,
            )
        )


def directory_hashes(root: Path) -> dict[str, str]:
    """Hash every file in a protected directory using relative paths."""
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def hash_manifest(files: dict[str, str]) -> str:
    """Return one deterministic digest for a relative-path hash manifest."""
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def summarize_metrics(monthly: pd.DataFrame) -> pd.DataFrame:
    """Average eligible target months equally, then add secondary horizon means."""
    identifiers = {
        "horizon_months",
        "target_month",
        "status",
        "coverage_n",
        "coverage_total",
        "coverage",
        "n",
        "phase1_support",
        "phase2_support",
        "phase3_support",
        "phase4_support",
        "tn",
        "fp",
        "fn",
        "tp",
    }
    metrics = [
        column
        for column in monthly.columns
        if column not in identifiers | {"track", "scope", "model"}
        and pd.api.types.is_numeric_dtype(monthly[column])
    ]
    available = monthly.loc[monthly["status"].eq("available")].copy()
    horizon = available.groupby(["track", "scope", "model"], as_index=False)[metrics].mean()
    counts = available.groupby(["track", "scope", "model"]).size().rename("eligible_months").reset_index()
    horizon = horizon.merge(counts, on=["track", "scope", "model"], validate="one_to_one")
    horizon.insert(0, "summary_level", "horizon_equal_month")
    horizon["horizons"] = horizon["scope"]
    cross = horizon.groupby(["track", "model"], as_index=False)[metrics].mean()
    cross.insert(0, "summary_level", "cross_horizon_equal_scope_secondary")
    cross["scope"] = "all"
    cross["eligible_months"] = np.nan
    cross["horizons"] = (
        horizon.groupby(["track", "model"])["scope"]
        .apply(lambda values: ",".join(sorted(values)))
        .reindex(pd.MultiIndex.from_frame(cross[["track", "model"]]))
        .to_numpy()
    )
    return pd.concat([horizon, cross], ignore_index=True, sort=False)


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Render the single compact four-class and binary comparison figure."""
    import matplotlib.pyplot as plt

    horizon = summary.loc[summary["summary_level"].eq("horizon_equal_month")]
    panels = (
        ("multiclass_fews_common", "macro_f1", "Four-class macro-F1 (common support)"),
        ("binary_comparison", "crisis_f1", "Crisis-class F1 (common support)"),
    )
    colors = {"xgboost": "#4C78A8", "georf_v5": "#F58518", "fewsnet": "#54A24B"}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    scopes = ("fs0", "fs1", "fs2", "fs3")
    width = 0.24
    for axis, (track, metric, title) in zip(axes, panels):
        if track == "binary_comparison":
            subset = pd.concat(
                [
                    horizon.loc[
                        horizon["track"].eq("binary_georf_common")
                        & horizon["scope"].isin(("fs0", "fs3"))
                    ],
                    horizon.loc[horizon["track"].eq("binary_fews_common")],
                ],
                ignore_index=True,
            )
        else:
            subset = horizon.loc[horizon["track"].eq(track)]
        models = [model for model in ("xgboost", "georf_v5", "fewsnet") if model in set(subset["model"])]
        positions = np.arange(len(scopes))
        for index, model in enumerate(models):
            values = subset.loc[subset["model"].eq(model)].set_index("scope")[metric].reindex(scopes)
            offset = (index - (len(models) - 1) / 2) * width
            axis.bar(positions + offset, values, width=width, label=model, color=colors[model])
        axis.set_xticks(positions, scopes)
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.set_xlabel("Forecast scope")
        axis.grid(axis="y", alpha=0.25)
    for position in (0, 3):
        axes[0].text(position, 0.04, "N/A", ha="center", va="bottom", color="dimgray")
    axes[0].set_ylabel("Equal-month mean score")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.96))
    figure.suptitle("Ethiopia XGBoost, GeoRF v5, and FEWS NET comparison", y=1.02)
    figure.text(
        0.5,
        -0.01,
        "fs0/fs3: GeoRF-common, 11 months; fs1/fs2: FEWS-common, 10 months. N/A = no FEWS baseline.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.90))
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_experiment(args: argparse.Namespace) -> Path:
    """Run all four scopes and write the frozen comparison artifact bundle."""
    v5_dir = args.v5_run.resolve()
    working_panel_path = args.working_panel.resolve()
    fewsnet_path = args.fewsnet.resolve()
    required = (v5_dir / "run_manifest.json", working_panel_path, fewsnet_path)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    v5_manifest = json.loads((v5_dir / "run_manifest.json").read_text(encoding="utf-8"))
    predictors = list(v5_manifest["predictors"])
    if len(predictors) != 88 or v5_manifest["scopes"] != {f"fs{k}": v for k, v in SCOPES.items()}:
        raise ValueError("v5 predictor or scope contract mismatch")

    source_paths = {
        "experiment_script": Path(__file__).resolve(),
        "working_panel": working_panel_path,
        "fewsnet": fewsnet_path,
        "v5_manifest": v5_dir / "run_manifest.json",
        **{f"snapshot_fs{scope}": v5_dir / "input" / f"ethiopia_panel_fs{scope}_88.csv" for scope in SCOPES},
        **{f"georf_fs{scope}": v5_dir / "stage3" / f"fs{scope}" / "predictions_monthly.csv" for scope in SCOPES},
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    source_hashes_before = {name: sha256_file(path) for name, path in source_paths.items()}
    expected_snapshot_hashes = v5_manifest["snapshot_hashes"]
    for scope in SCOPES:
        if source_hashes_before[f"snapshot_fs{scope}"] != expected_snapshot_hashes[f"fs{scope}"]:
            raise ValueError(f"Frozen fs{scope} snapshot hash drift")
    protected_v5_before = directory_hashes(v5_dir)
    protected_0901_before = {
        path.name: directory_hashes(path)
        for path in sorted(v5_dir.parent.glob("eth_local_20260901_seed5_v*"))
        if path.is_dir()
    }

    run_dir = create_run_directory(args.output_root.resolve(), args.run_id)
    panel = pd.read_csv(working_panel_path, usecols=[KEY, "date", "fews_ipc"], low_memory=False)
    panel[KEY] = normalize_admin_codes(panel[KEY]).to_numpy()
    panel["target_month"] = pd.to_datetime(panel.pop("date"), errors="raise").dt.to_period("M").dt.to_timestamp()
    panel["fews_ipc"] = pd.to_numeric(panel["fews_ipc"], errors="coerce")
    invalid = panel["fews_ipc"].notna() & ~panel["fews_ipc"].isin((1, 2, 3, 4))
    if invalid.any() or panel.duplicated([KEY, "target_month"]).any():
        raise ValueError("Working-panel phase or key contract failed")
    fewsnet = pd.read_csv(fewsnet_path, low_memory=False)

    predictions_output: list[pd.DataFrame] = []
    tuning_output: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    confusion_output: list[dict[str, object]] = []
    fold_audits: list[dict[str, object]] = []
    truth_join_audits: list[dict[str, object]] = []

    for scope, horizon in SCOPES.items():
        scope_name = f"fs{scope}"
        snapshot = pd.read_csv(source_paths[f"snapshot_fs{scope}"], low_memory=False)
        if list(snapshot.columns) != [*METADATA, *predictors]:
            raise ValueError(f"{scope_name} ordered feature contract failed")
        snapshot[KEY] = normalize_admin_codes(snapshot[KEY]).to_numpy()
        for column in ("target_month", "forecast_origin_month"):
            snapshot[column] = pd.to_datetime(snapshot[column], errors="raise").dt.to_period("M").dt.to_timestamp()
        if snapshot.duplicated([KEY, "target_month"]).any():
            raise ValueError(f"Duplicate snapshot keys for {scope_name}")
        before_rows = len(snapshot)
        snapshot = snapshot.merge(
            panel,
            on=[KEY, "target_month"],
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        unmatched_keys = int(snapshot["_merge"].ne("both").sum())
        snapshot = snapshot.drop(columns="_merge")
        if unmatched_keys:
            raise ValueError(f"{scope_name} truth join has {unmatched_keys} unmatched keys")
        truth_join_audits.append(
            {
                "scope": scope_name,
                "before_rows": before_rows,
                "after_rows": len(snapshot),
                "unmatched_keys": unmatched_keys,
                "missing_target_rows": int(snapshot["fews_ipc"].isna().sum()),
                "duplicate_keys": int(snapshot.duplicated([KEY, "target_month"]).sum()),
            }
        )
        georf = pd.read_csv(source_paths[f"georf_fs{scope}"])
        georf[KEY] = normalize_admin_codes(georf[KEY]).to_numpy()
        georf["target_month"] = pd.to_datetime(georf.pop("month_start"), errors="raise").dt.to_period("M").dt.to_timestamp()
        georf = georf[[KEY, "target_month", "y_true", "y_pred_partitioned_thresholded"]]
        if georf.duplicated([KEY, "target_month"]).any():
            raise ValueError(f"Duplicate GeoRF prediction keys for {scope_name}")

        features = snapshot[predictors].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if np.isinf(features).any():
            raise ValueError(f"Infinite XGBoost input for {scope_name}")
        for target_month in TARGET_MONTHS:
            fit_rows, validation_rows, test_rows, fold_audit = split_fold(
                snapshot,
                target_month=target_month,
                horizon=horizon,
            )
            y = snapshot["fews_ipc"].to_numpy(dtype=float)
            if target_month == SUPPRESSED_MONTH:
                observed_test = np.isfinite(y[test_rows])
                fold_audits.append(
                    {
                        "scope": scope_name,
                        "horizon_months": horizon,
                        **fold_audit,
                        "train_rows": int(len(fit_rows) + len(validation_rows)),
                        "observed_test_rows": int(observed_test.sum()),
                        "observed_test_phases": sorted(y[test_rows][observed_test].astype(int).tolist()),
                        "suppressed": True,
                        "suppression_reason": "only_one_observed_target",
                        "model_fitted": False,
                    }
                )
                suppressed_tracks: list[tuple[str, Sequence[str]]] = [
                    ("multiclass_model_support", ("xgboost",)),
                    ("binary_georf_common", ("xgboost", "georf_v5")),
                ]
                if scope in (1, 2):
                    suppressed_tracks.extend(
                        [
                            ("multiclass_fews_common", ("xgboost", "fewsnet")),
                            ("binary_fews_common", ("xgboost", "georf_v5", "fewsnet")),
                        ]
                    )
                for track, models in suppressed_tracks:
                    for model_name in models:
                        metric_rows.append(
                            metric_row(
                                track=track,
                                scope=scope_name,
                                horizon=horizon,
                                target_month=target_month,
                                model=model_name,
                                support_status="suppressed_low_target_coverage",
                                coverage_n=int(observed_test.sum()),
                                coverage_total=len(test_rows),
                                metrics=None,
                            )
                        )
                continue
            y_fit = y[fit_rows].astype(int) - 1
            y_validation = y[validation_rows].astype(int) - 1
            fit_weights, fit_weight_audit = class_weights(y_fit)
            candidate_rows: list[dict[str, object]] = []
            for max_depth, min_child_weight, n_estimators in GRID:
                model = make_model(
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    n_estimators=n_estimators,
                )
                model.fit(features[fit_rows], y_fit, sample_weight=fit_weights)
                validation_prediction = model.predict(features[validation_rows]).astype(int)
                candidate_rows.append(
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
                        "validation_macro_f1": float(
                            f1_score(y_validation, validation_prediction, labels=(0, 1, 2, 3), average="macro", zero_division=0)
                        ),
                    }
                )
            selected = min(candidate_rows, key=candidate_order)
            for row in candidate_rows:
                row["selected"] = all(
                    row[name] == selected[name]
                    for name in ("max_depth", "min_child_weight", "n_estimators")
                )
            tuning_output.extend(candidate_rows)

            train_rows = np.sort(np.concatenate([fit_rows, validation_rows]))
            y_train = y[train_rows].astype(int) - 1
            train_weights, train_weight_audit = class_weights(y_train)
            final_model = make_model(
                max_depth=int(selected["max_depth"]),
                min_child_weight=int(selected["min_child_weight"]),
                n_estimators=int(selected["n_estimators"]),
            )
            final_model.fit(features[train_rows], y_train, sample_weight=train_weights)
            probabilities = final_model.predict_proba(features[test_rows])
            predicted_phase = probabilities.argmax(axis=1) + 1
            if probabilities.shape[1] != 4 or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError(f"Invalid probabilities for {scope_name} {target_month}")

            prediction = snapshot.iloc[test_rows][[KEY, "target_month", "forecast_origin_month", "fews_ipc"]].copy()
            prediction.insert(0, "scope", scope_name)
            prediction.insert(1, "horizon_months", horizon)
            prediction["xgboost_predicted_phase"] = predicted_phase
            for phase in (1, 2, 3, 4):
                prediction[f"xgboost_probability_phase{phase}"] = probabilities[:, phase - 1]
            prediction = prediction.merge(georf, on=[KEY, "target_month"], how="left", validate="one_to_one")
            if target_month != SUPPRESSED_MONTH and prediction["y_true"].isna().any():
                raise ValueError(f"Missing GeoRF truth for {scope_name} {target_month}")
            observed = prediction["fews_ipc"].notna()
            if observed.any() and not np.array_equal(
                (prediction.loc[observed, "fews_ipc"].astype(int) >= 3).astype(int).to_numpy(),
                prediction.loc[observed, "y_true"].astype(int).to_numpy(),
            ):
                raise ValueError(f"Working-panel and GeoRF truth disagree for {scope_name} {target_month}")

            baseline_coverage = {"available": 0, "total": len(prediction), "fraction": 0.0}
            if scope in (1, 2):
                baseline, baseline_coverage = calendar_join_fewsnet(
                    fewsnet,
                    target_month=target_month,
                    scope=scope,
                    cohort_codes=prediction[KEY],
                )
                prediction = prediction.merge(
                    baseline[[KEY, "actual_phase", "projection_phase"]],
                    on=KEY,
                    how="left",
                    validate="one_to_one",
                )
            else:
                prediction["actual_phase"] = np.nan
                prediction["projection_phase"] = np.nan
            predictions_output.append(prediction)

            fold_audits.append(
                {
                    "scope": scope_name,
                    "horizon_months": horizon,
                    **fold_audit,
                    "train_rows": int(len(train_rows)),
                    "observed_test_rows": int(observed.sum()),
                    "observed_test_phases": sorted(prediction.loc[observed, "fews_ipc"].astype(int).unique().tolist()),
                    "suppressed": target_month == SUPPRESSED_MONTH,
                    "suppression_reason": "only_one_observed_target" if target_month == SUPPRESSED_MONTH else None,
                    "selected_parameters": {
                        name: selected[name] for name in ("max_depth", "min_child_weight", "n_estimators")
                    },
                    "fit_weight_audit": fit_weight_audit,
                    "refit_weight_audit": train_weight_audit,
                    "fit_weight_mean": float(fit_weights.mean()),
                    "refit_weight_mean": float(train_weights.mean()),
                    "fewsnet_coverage": baseline_coverage,
                }
            )

            supported = prediction.loc[observed].copy()
            true_phase = supported["fews_ipc"].astype(int).to_numpy()
            add_metrics(
                metric_rows,
                confusion_output,
                track="multiclass_model_support",
                scope=scope_name,
                horizon=horizon,
                target_month=target_month,
                model_predictions={"xgboost": supported["xgboost_predicted_phase"].astype(int)},
                truth=true_phase,
                labels=(1, 2, 3, 4),
                coverage_total=len(prediction),
            )
            add_metrics(
                metric_rows,
                confusion_output,
                track="binary_georf_common",
                scope=scope_name,
                horizon=horizon,
                target_month=target_month,
                model_predictions={
                    "xgboost": (supported["xgboost_predicted_phase"].astype(int) >= 3).astype(int),
                    "georf_v5": supported["y_pred_partitioned_thresholded"].astype(int),
                },
                truth=(true_phase >= 3).astype(int),
                labels=(0, 1),
                coverage_total=len(prediction),
            )

            if scope in (1, 2) and baseline_coverage["fraction"] >= MIN_COVERAGE:
                common = prediction[["fews_ipc", "actual_phase", "projection_phase"]].notna().all(axis=1)
                comparison = prediction.loc[common].copy()
                if not np.array_equal(
                    comparison["fews_ipc"].astype(int).to_numpy(),
                    comparison["actual_phase"].astype(int).to_numpy(),
                ):
                    raise ValueError(f"FEWS NET truth mismatch for {scope_name} {target_month}")
                common_truth = comparison["fews_ipc"].astype(int).to_numpy()
                add_metrics(
                    metric_rows,
                    confusion_output,
                    track="multiclass_fews_common",
                    scope=scope_name,
                    horizon=horizon,
                    target_month=target_month,
                    model_predictions={
                        "xgboost": comparison["xgboost_predicted_phase"].astype(int),
                        "fewsnet": comparison["projection_phase"].astype(int),
                    },
                    truth=common_truth,
                    labels=(1, 2, 3, 4),
                    coverage_total=len(prediction),
                )
                add_metrics(
                    metric_rows,
                    confusion_output,
                    track="binary_fews_common",
                    scope=scope_name,
                    horizon=horizon,
                    target_month=target_month,
                    model_predictions={
                        "xgboost": (comparison["xgboost_predicted_phase"].astype(int) >= 3).astype(int),
                        "georf_v5": comparison["y_pred_partitioned_thresholded"].astype(int),
                        "fewsnet": (comparison["projection_phase"].astype(int) >= 3).astype(int),
                    },
                    truth=(common_truth >= 3).astype(int),
                    labels=(0, 1),
                    coverage_total=len(prediction),
                )
            elif scope in (1, 2):
                for track, models in (
                    ("multiclass_fews_common", ("xgboost", "fewsnet")),
                    ("binary_fews_common", ("xgboost", "georf_v5", "fewsnet")),
                ):
                    for model_name in models:
                        metric_rows.append(
                            metric_row(
                                track=track,
                                scope=scope_name,
                                horizon=horizon,
                                target_month=target_month,
                                model=model_name,
                                support_status="suppressed_low_fewsnet_coverage",
                                coverage_n=int(baseline_coverage["available"]),
                                coverage_total=int(baseline_coverage["total"]),
                                metrics=None,
                            )
                        )

            print(
                f"{scope_name} {target_month}: selected depth={selected['max_depth']}, "
                f"child={selected['min_child_weight']}, trees={selected['n_estimators']}",
                flush=True,
            )

    predictions = pd.concat(predictions_output, ignore_index=True).sort_values(["scope", "target_month", KEY])
    tuning = pd.DataFrame(tuning_output).sort_values(["scope", "target_month", "max_depth", "min_child_weight", "n_estimators"])
    monthly = pd.DataFrame(metric_rows).sort_values(["track", "scope", "target_month", "model"])
    confusions = pd.DataFrame(confusion_output).sort_values(["track", "scope", "target_month", "model", "true_class", "predicted_class"])
    if (
        tuning.groupby(["scope", "target_month"]).ngroups != len(SCOPES) * (len(TARGET_MONTHS) - 1)
        or tuning.groupby(["scope", "target_month"]).size().ne(8).any()
        or tuning.groupby(["scope", "target_month"])["selected"].sum().ne(1).any()
    ):
        raise ValueError("Eight-candidate tuning contract failed")
    if predictions.duplicated(["scope", KEY, "target_month"]).any():
        raise ValueError("Duplicate exported prediction keys")
    probability_columns = [f"xgboost_probability_phase{phase}" for phase in (1, 2, 3, 4)]
    if not np.allclose(predictions[probability_columns].sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Exported probability sums failed")
    summary = summarize_metrics(monthly)
    selected_parameters = tuning.loc[tuning["selected"]].reset_index(drop=True)

    predictions.to_csv(run_dir / "predictions.csv", index=False)
    selected_parameters.to_csv(run_dir / "selected_parameters.csv", index=False)
    monthly.to_csv(run_dir / "metrics_monthly.csv", index=False)
    summary.to_csv(run_dir / "metrics_summary.csv", index=False)
    confusions.to_csv(run_dir / "confusion_matrices.csv", index=False)
    plot_summary(summary, run_dir / "comparison.png")

    source_hashes_after = {name: sha256_file(path) for name, path in source_paths.items()}
    protected_v5_after = directory_hashes(v5_dir)
    protected_0901_after = {
        path.name: directory_hashes(path)
        for path in sorted(v5_dir.parent.glob("eth_local_20260901_seed5_v*"))
        if path.is_dir()
    }
    if source_hashes_after != source_hashes_before or protected_v5_after != protected_v5_before:
        raise RuntimeError("Protected v5 source changed during experiment")
    if protected_0901_after != protected_0901_before:
        raise RuntimeError("Protected 2026-09-01 run changed during experiment")
    write_json(run_dir / "source_hashes.json", {"before": source_hashes_before, "after": source_hashes_after})
    artifact_names = (
        "predictions.csv",
        "selected_parameters.csv",
        "metrics_monthly.csv",
        "metrics_summary.csv",
        "confusion_matrices.csv",
        "comparison.png",
        "source_hashes.json",
    )
    artifact_hashes = {name: sha256_file(run_dir / name) for name in artifact_names}
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    metadata = {
        "run_id": args.run_id,
        "created_at": datetime.now().isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "xgboost_version": __import__("xgboost").__version__,
        "git_head": git_head,
        "git_status_porcelain": git_status,
        "experiment_script_sha256": sha256_file(Path(__file__).resolve()),
        "random_seed": SEED,
        "n_jobs": 1,
        "v5_run": str(v5_dir),
        "predictor_count": len(predictors),
        "predictors": predictors,
        "native_missing_handling": True,
        "median_imputation": False,
        "smote": False,
        "training_window_calendar_months": TRAIN_MONTHS,
        "validation_latest_eligible_months": VALIDATION_MONTHS,
        "target_months": list(TARGET_MONTHS),
        "suppressed_target_month": SUPPRESSED_MONTH,
        "grid": [
            {"max_depth": depth, "min_child_weight": child, "n_estimators": trees}
            for depth, child, trees in GRID
        ],
        "fixed_parameters": {
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "num_class": 4,
        },
        "selection_metric": "validation_macro_f1",
        "tie_break": ["shallower_depth", "higher_min_child_weight", "fewer_trees"],
        "sample_weight": "normalized sqrt(n / (4 * n_k)) from fit rows only",
        "fewsnet_policy": {
            "fs1": "fews_proj_near at T-4",
            "fs2": "fews_proj_med at T-8",
            "fs0": "unavailable",
            "fs3": "unavailable",
            "minimum_projection_coverage": MIN_COVERAGE,
        },
        "truth_join_audits": truth_join_audits,
        "fold_audits": fold_audits,
        "tuning_results": tuning.to_dict(orient="records"),
        "rows": {
            "predictions": len(predictions),
            "selected_parameters": len(selected_parameters),
            "tuning_candidates_in_metadata": len(tuning),
            "metrics_monthly": len(monthly),
            "metrics_summary": len(summary),
            "confusion_cells": len(confusions),
        },
        "protected_v5_file_count": len(protected_v5_before),
        "protected_0901_runs": {name: len(files) for name, files in protected_0901_before.items()},
        "protected_directory_hashes": {
            "v5_before": hash_manifest(protected_v5_before),
            "v5_after": hash_manifest(protected_v5_after),
            "runs_0901_before": {name: hash_manifest(files) for name, files in protected_0901_before.items()},
            "runs_0901_after": {name: hash_manifest(files) for name, files in protected_0901_after.items()},
        },
        "protected_hashes_unchanged": True,
        "artifact_hashes": artifact_hashes,
        "artifacts": [
            "predictions.csv",
            "selected_parameters.csv",
            "metrics_monthly.csv",
            "metrics_summary.csv",
            "confusion_matrices.csv",
            "comparison.png",
            "source_hashes.json",
            "run_metadata.json",
        ],
    }
    write_json(run_dir / "run_metadata.json", metadata)
    headline = summary.loc[
        summary["summary_level"].eq("horizon_equal_month"),
        ["track", "scope", "model", "macro_f1", "crisis_f1", "eligible_months"],
    ]
    print("\nEqual-month horizon headline metrics:")
    print(headline.to_string(index=False))
    return run_dir


def parse_args() -> argparse.Namespace:
    """Parse the isolated experiment command line."""
    experiment_root = Path(__file__).resolve().parent
    source_root = REPO_ROOT.parents[2] / "1.Source Data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v5-run",
        type=Path,
        default=experiment_root / "outputs" / "local_partition_experiment" / "eth_aligned_refit_20260904_seed5_v5",
    )
    parser.add_argument("--working-panel", type=Path, default=experiment_root / "data" / "working" / "ethiopia_panel.csv")
    parser.add_argument("--fewsnet", type=Path, default=source_root / "Outcome" / "FEWSNET_IPC" / "FEWSNET.csv")
    parser.add_argument("--output-root", type=Path, default=experiment_root / "outputs" / "local_partition_experiment")
    parser.add_argument("--run-id", default="eth_multiclass_xgb_20260904_seed5_v2")
    return parser.parse_args()


if __name__ == "__main__":
    completed = run_experiment(parse_args())
    print(f"Experiment complete: {completed}")
