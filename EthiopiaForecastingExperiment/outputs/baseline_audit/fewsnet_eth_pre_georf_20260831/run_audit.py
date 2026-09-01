#!/usr/bin/env python3
"""Profile the exact ETH slice before GeoRF preprocessing and run risk probes.

This script is deliberately isolated from the production GeoRF entry points.
It reads the frozen assembled FEWS NET panel and frozen released predictions;
it never calls ``load_and_preprocess_data``, ``prepare_features``, or a batch
pipeline. The Random Forest fits below are lightweight diagnostic probes, not
GeoRF reproductions and not candidate-selection runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


OUTPUT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_ROOT.parents[3]
SOURCE_PATH = (
    REPO_ROOT.parents[2]
    / "1.Source Data"
    / "FEWSNET_forecast_unadjusted_bm.csv"
)
PROVIDER_ROOT = (
    REPO_ROOT
    / "archived"
    / "release_20260624_reproducibility_inputs"
)
EXPECTED_SOURCE_SHA256 = (
    "611f9e776380e28da3fc845888d66a117626f91b37868e236ce849d44bc8f651"
)
EXPECTED_SOURCE_SHAPE = (1_029_240, 88)
EXPECTED_ETH_SHAPE = (187_200, 88)
EXPECTED_ETH_ADMINS = 1_040
EXPECTED_ETH_MONTHS = 180

TARGET = "fews_ipc_crisis"
KEY = "FEWSNET_admin_code"
TARGET_RELATED = {
    "fews_ipc",
    "fews_ha",
    "fews_proj_near",
    "fews_proj_near_ha",
    "fews_proj_med",
    "fews_proj_med_ha",
    "fews_ipc_adjusted",
    "fews_proj_med_adjusted",
    TARGET,
}
TEXT_OR_DATE = {
    "unit_name",
    "ADMIN0",
    "ADMIN1",
    "ADMIN2",
    "ADMIN3",
    "ISO",
    "ISO3",
    "date",
}
CONFLICT_COLUMNS = {
    "distance_to_nearest_acled",
    "event_count_battles",
    "event_count_explosions",
    "event_count_violence",
    "sum_fatalities_battles",
    "sum_fatalities_explosions",
    "sum_fatalities_violence",
    "event_count_battles_w5",
    "event_count_explosions_w5",
    "event_count_violence_w5",
    "sum_fatalities_battles_w5",
    "sum_fatalities_explosions_w5",
    "sum_fatalities_violence_w5",
    "event_count_battles_w10",
    "event_count_explosions_w10",
    "event_count_violence_w10",
    "sum_fatalities_battles_w10",
    "sum_fatalities_explosions_w10",
    "sum_fatalities_violence_w10",
}
PRICE_COLUMNS = {"FAO_price", "WFP_Price", "WFP_Price_std"}
MACRO_COLUMNS = {"CPI", "GDP", "CC", "gini", "Food_CPI", "Food_food_inflation"}
WEATHER_COLUMNS = {
    "Rainf_f_tavg_mean",
    "Tair_f_tavg_mean",
    "Tair_zscore",
    "Rainf_zscore",
}
REMOTE_SENSING_COLUMNS = {"nightlight", "nightlight_sd", "EVI", "gpp_mean"}
TERRAIN_ACCESS_COLUMNS = {
    "lat",
    "lon",
    "distance_to_river",
    "elevation",
    "market_distance",
    "market_access",
    "ruggedness",
    "slope",
}
AGRICULTURE_COLUMNS = {"crop", "range", "pop"}


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_admin_code(series: pd.Series) -> pd.Series:
    """Normalize a FEWS NET key without changing cohort membership."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def feature_family(name: str) -> str:
    """Assign a transparent descriptive family to an assembled-panel field."""
    if name in {"unit_name", "ADMIN0", "ADMIN1", "ADMIN2", "ADMIN3", "ISO", "ISO3", KEY}:
        return "identifier_geography"
    if name in {"date", "month"}:
        return "time"
    if name in CONFLICT_COLUMNS:
        return "conflict"
    if name.startswith("AEZ_"):
        return "aez_dummy"
    if name in PRICE_COLUMNS:
        return "price"
    if name in MACRO_COLUMNS:
        return "macro"
    if name in WEATHER_COLUMNS:
        return "weather_derived"
    if name in REMOTE_SENSING_COLUMNS:
        return "remote_sensing"
    if name.startswith("sg_"):
        return "soil"
    if name in TERRAIN_ACCESS_COLUMNS:
        return "terrain_access"
    if name in AGRICULTURE_COLUMNS:
        return "agriculture_population"
    if name in TARGET_RELATED:
        return "ipc_target_projection"
    return "other_covariate"


def field_role(name: str) -> str:
    """Describe whether a field is metadata, covariate, or outcome-related."""
    if name == TARGET:
        return "binary_target"
    if name in TARGET_RELATED:
        return "target_or_provider_projection_related"
    if name in TEXT_OR_DATE or name == KEY:
        return "identifier_or_time_metadata"
    return "assembled_covariate"


def loader_action(name: str) -> str:
    """Record current production loader treatment from inspected source code."""
    initial_drop = {
        "ISO3",
        "fews_ipc_adjusted",
        "fews_proj_med_adjusted",
        "fews_proj_near",
        "fews_proj_near_ha",
        "fews_proj_med",
        "fews_proj_med_ha",
        "ADMIN0",
        "ADMIN1",
        "ADMIN2",
        "ADMIN3",
    }
    if name in initial_drop:
        return "dropped_immediately_by_loader"
    if name == TARGET:
        return "filters_to_non_null_then_used_as_target"
    if name == "ISO":
        return "encoded_to_ISO_encoded_then_dropped"
    if name == "unit_name":
        return "dropped_after_ISO_encoding"
    if name == "date":
        return "parsed_to_datetime_and_used_for_dummies_splits_lags"
    if name == "fews_ipc":
        return "used_to_create_4_8_12_lags_then_dropped"
    if name == "month":
        return "retained_by_loader_then_dropped_inside_model_fit"
    if name == "fews_ha":
        return "retained_by_loader_then_dropped_inside_model_fit"
    if name == KEY:
        return "retained_for_grouping_and_current_feature_matrix"
    return "retained_then_subject_to_feature_engineering_lagging_imputation"


def effective_missing_mask(frame: pd.DataFrame) -> pd.DataFrame:
    """Treat raw nulls and numeric infinities as pipeline-effective missing."""
    mask = frame.isna()
    numeric_columns = frame.select_dtypes(include=[np.number, "bool"]).columns
    if len(numeric_columns):
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        mask.loc[:, numeric_columns] |= ~np.isfinite(numeric.to_numpy(dtype=float))
    return mask


def finite_numeric(series: pd.Series) -> pd.Series:
    """Return numeric finite values only."""
    values = pd.to_numeric(series, errors="coerce")
    return values.loc[np.isfinite(values.to_numpy(dtype=float))]


def contiguous_runs(months: Iterable[pd.Period]) -> list[tuple[str, str, int]]:
    """Compress sorted monthly periods into inclusive contiguous runs."""
    ordered = sorted(set(months))
    if not ordered:
        return []
    runs: list[tuple[str, str, int]] = []
    start = previous = ordered[0]
    for current in ordered[1:]:
        if current != previous + 1:
            runs.append((str(start), str(previous), int(previous.ordinal - start.ordinal + 1)))
            start = current
        previous = current
    runs.append((str(start), str(previous), int(previous.ordinal - start.ordinal + 1)))
    return runs


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    """Calculate thresholded and threshold-free binary metrics."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "positive_n": int(y.sum()),
        "prevalence": float(y.mean()),
        "threshold": float(threshold),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def select_validation_threshold(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """Select a threshold on validation F1 only."""
    candidates = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scored = [
        (float(f1_score(y_true, probability >= threshold, zero_division=0)), float(threshold))
        for threshold in candidates
    ]
    best_f1, best_threshold = max(
        scored,
        key=lambda item: (item[0], -abs(item[1] - 0.5)),
    )
    return best_threshold, best_f1


def build_probe_frame(labeled: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build a conservative assembled-row numeric probe matrix."""
    excluded = TEXT_OR_DATE | TARGET_RELATED
    base_features = [
        column
        for column in labeled.columns
        if column not in excluded
        and column != "date_period"
        and (
            pd.api.types.is_numeric_dtype(labeled[column])
            or pd.api.types.is_bool_dtype(labeled[column])
        )
    ]
    frame = labeled[base_features].copy()
    for column in frame.columns:
        if pd.api.types.is_bool_dtype(frame[column]):
            frame[column] = frame[column].astype(int)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if "month" in frame.columns:
        month_dummies = pd.get_dummies(
            frame.pop("month").astype("Int64"),
            prefix="calendar_month",
            dtype=int,
        )
        frame = pd.concat([frame, month_dummies], axis=1)
    features_with_id = frame.columns.tolist()
    features_without_id = [column for column in features_with_id if column != KEY]
    return frame, features_with_id, features_without_id


def run_temporal_probe(
    labeled: pd.DataFrame,
    probe_frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit fixed RF diagnostics on blocked time splits."""
    periods = labeled["date_period"]
    split_masks = {
        "train": periods.dt.year <= 2018,
        "validation": periods.dt.year.between(2019, 2020),
        "test": periods.dt.year >= 2021,
    }
    model_specs = {
        "unrestricted_with_admin_id": {
            "feature_set": "with_admin_id",
            "params": {"max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
        },
        "unrestricted_without_admin_id": {
            "feature_set": "without_admin_id",
            "params": {"max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
        },
        "regularized_without_admin_id": {
            "feature_set": "without_admin_id",
            "params": {"max_depth": 12, "min_samples_leaf": 20, "max_features": "sqrt"},
        },
    }
    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []

    for candidate, spec in model_specs.items():
        feature_names = feature_sets[spec["feature_set"]]
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=5,
                        n_jobs=1,
                        **spec["params"],
                    ),
                ),
            ]
        )
        model.fit(
            probe_frame.loc[split_masks["train"], feature_names],
            labeled.loc[split_masks["train"], TARGET].astype(int),
        )
        probabilities = {
            split: model.predict_proba(probe_frame.loc[mask, feature_names])[:, 1]
            for split, mask in split_masks.items()
        }
        y_values = {
            split: labeled.loc[mask, TARGET].astype(int).to_numpy()
            for split, mask in split_masks.items()
        }
        threshold, validation_f1 = select_validation_threshold(
            y_values["validation"],
            probabilities["validation"],
        )
        for split, mask in split_masks.items():
            row = {
                "split_design": "blocked_time",
                "candidate": candidate,
                "feature_set": spec["feature_set"],
                "split": split,
                "date_min": str(periods.loc[mask].min()),
                "date_max": str(periods.loc[mask].max()),
                "feature_count_before_imputer": len(feature_names),
                "validation_selected_threshold": threshold,
                "validation_f1_at_selection": validation_f1,
            }
            row.update(binary_metrics(y_values[split], probabilities[split], threshold))
            metrics_rows.append(row)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "split_design": "blocked_time",
                        "candidate": candidate,
                        "split": split,
                        KEY: normalize_admin_code(labeled.loc[mask, KEY]).to_numpy(),
                        "date": labeled.loc[mask, "date"].astype(str).to_numpy(),
                        "y_true": y_values[split],
                        "probability": probabilities[split],
                        "threshold": threshold,
                        "y_pred": (probabilities[split] >= threshold).astype(int),
                    }
                )
            )
        imputer = model.named_steps["imputer"]
        rf = model.named_steps["rf"]
        transformed_names = imputer.get_feature_names_out(feature_names)
        for feature_name, importance in zip(transformed_names, rf.feature_importances_):
            importance_rows.append(
                {
                    "split_design": "blocked_time",
                    "candidate": candidate,
                    "feature_name": feature_name,
                    "importance": float(importance),
                }
            )
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            left_admins = set(normalize_admin_code(labeled.loc[split_masks[left], KEY]))
            right_admins = set(normalize_admin_code(labeled.loc[split_masks[right], KEY]))
            overlap_rows.append(
                {
                    "split_design": "blocked_time",
                    "candidate": candidate,
                    "left_split": left,
                    "right_split": right,
                    "left_admins": len(left_admins),
                    "right_admins": len(right_admins),
                    "admin_overlap": len(left_admins & right_admins),
                    "interpretation": "temporal_generalization_within_seen_admins",
                }
            )

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    importances = pd.DataFrame(importance_rows).sort_values(
        ["candidate", "importance"],
        ascending=[True, False],
    )
    overlaps = pd.DataFrame(overlap_rows)
    return metrics, predictions, importances, overlaps


def run_random_split_sensitivity(
    labeled: pd.DataFrame,
    probe_frame: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Demonstrate optimism from a random row split with repeated admins."""
    all_indices = np.arange(len(labeled))
    train_indices, remaining = train_test_split(
        all_indices,
        test_size=0.40,
        random_state=5,
        stratify=labeled[TARGET].astype(int),
    )
    validation_indices, test_indices = train_test_split(
        remaining,
        test_size=0.50,
        random_state=5,
        stratify=labeled.iloc[remaining][TARGET].astype(int),
    )
    index_sets = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=5,
                    n_jobs=1,
                    max_depth=None,
                    min_samples_leaf=1,
                    max_features="sqrt",
                ),
            ),
        ]
    )
    model.fit(
        probe_frame.iloc[train_indices][features],
        labeled.iloc[train_indices][TARGET].astype(int),
    )
    probabilities = {
        split: model.predict_proba(probe_frame.iloc[indices][features])[:, 1]
        for split, indices in index_sets.items()
    }
    y_values = {
        split: labeled.iloc[indices][TARGET].astype(int).to_numpy()
        for split, indices in index_sets.items()
    }
    threshold, validation_f1 = select_validation_threshold(
        y_values["validation"],
        probabilities["validation"],
    )
    metric_rows = []
    for split, indices in index_sets.items():
        row = {
            "split_design": "random_row_60_20_20",
            "candidate": "unrestricted_without_admin_id",
            "feature_set": "without_admin_id",
            "split": split,
            "date_min": str(labeled.iloc[indices]["date_period"].min()),
            "date_max": str(labeled.iloc[indices]["date_period"].max()),
            "feature_count_before_imputer": len(features),
            "validation_selected_threshold": threshold,
            "validation_f1_at_selection": validation_f1,
        }
        row.update(binary_metrics(y_values[split], probabilities[split], threshold))
        metric_rows.append(row)
    overlap_rows = []
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left_admins = set(normalize_admin_code(labeled.iloc[index_sets[left]][KEY]))
        right_admins = set(normalize_admin_code(labeled.iloc[index_sets[right]][KEY]))
        overlap_rows.append(
            {
                "split_design": "random_row_60_20_20",
                "candidate": "unrestricted_without_admin_id",
                "left_split": left,
                "right_split": right,
                "left_admins": len(left_admins),
                "right_admins": len(right_admins),
                "admin_overlap": len(left_admins & right_admins),
                "interpretation": "same_admins_and_all_years_cross_split_optimistic",
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(overlap_rows)


def hard_prediction_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    """Metrics for frozen hard predictions."""
    y = y_true.astype(int).to_numpy()
    predicted = y_pred.astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    true_negative_rate = tn / (tn + fp) if (tn + fp) else np.nan
    true_positive_rate = tp / (tp + fn) if (tp + fn) else np.nan
    balanced_accuracy = float(np.nanmean([true_negative_rate, true_positive_rate]))
    return {
        "n": int(len(y)),
        "positive_n": int(y.sum()),
        "prevalence": float(y.mean()),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "balanced_accuracy": balanced_accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def released_georf_diagnostics(
    eth_codes: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    """Characterize frozen ETH held-out GeoRF predictions without refitting."""
    summary_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    model_columns = {
        "pooled": ("y_pred_pooled", "y_prob_pooled"),
        "partitioned": ("y_pred_partitioned", "y_prob_partitioned"),
        "partitioned_thresholded": (
            "y_pred_partitioned_thresholded",
            "y_prob_partitioned",
        ),
    }
    for scope, horizon in ((1, 4), (2, 8), (3, 12)):
        provider = PROVIDER_ROOT / f"result_partition_k40_compare_GF_thresholded_fs{scope}"
        predictions_path = provider / "predictions_monthly.csv"
        manifest_path = provider / "run_manifest.json"
        source_paths.extend([predictions_path, manifest_path])
        predictions = pd.read_csv(
            predictions_path,
            dtype={KEY: "string"},
            low_memory=False,
        )
        predictions[KEY] = normalize_admin_code(predictions[KEY])
        eth = predictions.loc[predictions[KEY].isin(eth_codes)].copy()
        if len(eth) != 11_441 or eth[KEY].nunique() != EXPECTED_ETH_ADMINS:
            raise RuntimeError(f"Unexpected ETH released-prediction coverage for fs{scope}")
        if eth.duplicated([KEY, "month_start"]).any():
            raise RuntimeError(f"Duplicate released prediction keys for fs{scope}")
        for model, (prediction_column, probability_column) in model_columns.items():
            row = {
                "scope": scope,
                "forecasting_horizon": horizon,
                "model": model,
                "evaluation": "released_2021_2024_temporal_holdout",
            }
            row.update(hard_prediction_metrics(eth["y_true"], eth[prediction_column]))
            probability = eth[probability_column].astype(float)
            row["average_precision"] = float(average_precision_score(eth["y_true"], probability))
            row["roc_auc"] = float(roc_auc_score(eth["y_true"], probability))
            row["brier"] = float(brier_score_loss(eth["y_true"], probability))
            row["probability_source"] = probability_column
            summary_rows.append(row)
            for month, group in eth.groupby("month_start", sort=True):
                month_row = {
                    "scope": scope,
                    "forecasting_horizon": horizon,
                    "month_start": month,
                    "model": model,
                }
                month_row.update(hard_prediction_metrics(group["y_true"], group[prediction_column]))
                monthly_rows.append(month_row)
        for partition_id, group in eth.groupby("partition_id", dropna=False, sort=True):
            row = {
                "scope": scope,
                "forecasting_horizon": horizon,
                "partition_id": partition_id,
                "admin_codes": int(group[KEY].nunique()),
            }
            row.update(hard_prediction_metrics(group["y_true"], group["y_pred_partitioned"]))
            row["low_support_lt100"] = bool(len(group) < 100)
            row["low_positive_support_lt10"] = bool(group["y_true"].sum() < 10)
            partition_rows.append(row)
        monthly = pd.DataFrame(
            [row for row in monthly_rows if row["scope"] == scope]
        )
        stable = monthly.loc[monthly["n"] >= 30].pivot(
            index="month_start",
            columns="model",
            values="f1",
        )
        stability_rows.append(
            {
                "scope": scope,
                "forecasting_horizon": horizon,
                "stable_months_n_ge_30": int(len(stable)),
                "partitioned_beats_pooled_months": int((stable["partitioned"] > stable["pooled"]).sum()),
                "thresholded_beats_partitioned_months": int((stable["partitioned_thresholded"] > stable["partitioned"]).sum()),
                "excluded_low_support_months": int(12 - len(stable)),
            }
        )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(monthly_rows),
        pd.DataFrame(partition_rows),
        pd.DataFrame(stability_rows),
        source_paths,
    )


def write_pipeline_lineage() -> None:
    """Write the inspected code boundary in plain language."""
    text = """# Exact data boundary and current GeoRF lineage

## Boundary used in this audit

The characterized dataset is the parsed `ISO3 == \"ETH\"` slice of
`FEWSNET_forecast_unadjusted_bm.csv` before `load_and_preprocess_data()` is
called. It has 88 assembled fields. It is not upstream raw-source data: several
fields are already derived or aggregated, including z-scores, IPC binaries,
provider projections, conflict windows, AEZ indicators, prices, and macro data.

## What production currently does after this boundary

1. `load_and_preprocess_data()` drops 11 fields, filters to non-null
   `fews_ipc_crisis`, encodes `ISO`, parses dates, creates target/phase lags at
   4/8/12 months, drops `fews_ipc`, creates year/month dummies, applies three
   feature-engineering helpers, and creates AEZ groups.
2. Each feature-engineering helper currently returns inside its outer feature
   loop. Consequently only the first eligible feature in each list is handled.
3. `prepare_features()` detects time-varying columns, creates a scope-specific
   lagged copy, but retains the original contemporaneous column. It drops the
   target/date/group encodings, then performs max-plus out-of-range imputation.
4. The current feature-drop configuration later removes `month`, `fews_ha`,
   and `years` when present, but does not list `FEWSNET_admin_code`, `lat`, or
   `lon`.

These are observed code behaviors. They create forecast-availability and model
capacity questions but are not, by themselves, measured leakage or confirmed
GeoRF overfitting.
"""
    (OUTPUT_ROOT / "pipeline_lineage.md").write_text(text, encoding="utf-8")


def write_readme(
    column_profile: pd.DataFrame,
    temporal_metrics: pd.DataFrame,
    random_metrics: pd.DataFrame,
    georf_summary: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    """Write a concise Chinese interpretation with evidence boundaries."""
    high_missing = column_profile.loc[
        column_profile["effective_missing_rate"] >= 0.20,
        ["column", "feature_family", "effective_missing_rate"],
    ].sort_values("effective_missing_rate", ascending=False)
    temporal_display = temporal_metrics.loc[
        temporal_metrics["split"].isin(["train", "validation", "test"]),
        [
            "candidate",
            "split",
            "n",
            "prevalence",
            "threshold",
            "precision",
            "recall",
            "f1",
            "average_precision",
            "brier",
        ],
    ]
    random_display = random_metrics[
        ["split", "n", "threshold", "f1", "average_precision", "brier"]
    ]
    georf_display = georf_summary[
        [
            "forecasting_horizon",
            "model",
            "n",
            "precision",
            "recall",
            "f1",
            "average_precision",
            "brier",
        ]
    ]
    text = f"""# FEWS NET Ethiopia pre-GeoRF data audit

## 审计边界

本审计使用 checksum 固定的 assembled panel，精确筛选 `ISO3 == \"ETH\"`，
分析其进入 `load_and_preprocess_data()` 之前的状态。结果为 187,200 行、
88 列、1,040 个 admin、180 个月（2010-01 至 2024-12），且
`(FEWSNET_admin_code, date)` 无重复。未运行 GeoRF pipeline。

`fewsnet_eth_pre_georf.csv.gz` 是这一 parsed slice 的确定性压缩导出；
权威来源仍是外部完整 panel 和 `SOURCE_MANIFEST.csv` 中的源 SHA-256。

## 特征与缺失概览

88 个字段包含 ID/行政层级、时间和坐标、冲突事件及 5/10 月窗口、17 个
AEZ dummy、遥感与环境、土壤/地形/市场可达性、价格与宏观、IPC 标签和
provider projection、人口以及气温/降水 z-score。完整逐列解释见
`column_profile.csv` 和 `pipeline_column_lineage.csv`。

原始面板是完整的 1,040 × 180 平衡 admin-month 网格；缺失来自字段本身，
不是丢失整行。≥20% effective missing 的字段如下：

```csv
{high_missing.to_csv(index=False, float_format='%.6f').strip()}
```

主要 pattern 是制度性时间块：IPC 标签只在 51/180 个月发布；`gini` 从
2016 起缺失；Food CPI 两列从 2023-07 起缺失；2024 年 GDP 缺失；WFP
价格存在早期/中期整块缺失。`Tair_zscore` 与 `Rainf_zscore` 各有 2,880
个 infinity，集中于固定 16 个 admin，production imputation 会把它们按
non-finite/missing 处理。详细见 missingness 与 nonfinite 系列 CSV。

## 简单过拟合检查

诊断 RF 排除了 IPC 标签、IPC phase、humanitarian-area 与 provider
projection 字段；预处理仅在训练集拟合 median imputer，threshold 只用
2019-2020 validation 选择。它仍保留 contemporaneous assembled-row
covariates，因此不是 forecast-time-safe baseline，也不是 GeoRF reproduction。

时间阻断结果：

```csv
{temporal_display.to_csv(index=False, float_format='%.6f').strip()}
```

核心信号：unrestricted/no-admin-ID probe 的 train F1 接近 1，而 temporal
test F1 约 0.584、AP 约 0.616、Brier 约 0.164；固定 regularization 将
train F1 降至约 0.760，但 test F1 仍约 0.581。这个结果支持“无约束 RF 在
该 assembled-row 诊断上存在强 capacity/memorization gap”，但不是正式
GeoRF 过拟合裁决。

随机 row split sensitivity：

```csv
{random_display.to_csv(index=False, float_format='%.6f').strip()}
```

随机 test F1 约 0.831，显著高于 temporal test；且 train/test 都包含全部
1,040 个 admin。这说明随机行切分会严重乐观，不能作为时间或空间泛化证据。

冻结 GeoRF ETH temporal-test 结果（未重训）：

```csv
{georf_display.to_csv(index=False, float_format='%.6f').strip()}
```

稳定月份（排除只有 1 个标签的 2021-06）中，partitioned 相对 pooled 的
逐月 F1 胜场为：

```csv
{stability.to_csv(index=False).strip()}
```

因此现有 held-out GeoRF 结果没有呈现“partitioned 在 ETH 上整体崩溃”，
但没有对应 GeoRF train/OOF metrics，不能直接计算 GeoRF train-test gap；
小 partition、低 partition stability、forecast-origin availability 仍是风险。

## 结论边界

- 已支持：原始 assembled ETH slice 的缺失主要是时间/字段块结构；存在
  non-finite z-score、常量/近常量和长尾特征。
- 已支持：无约束 lightweight RF 有明显 in-sample 到 temporal holdout gap；
  random row split 明显乐观。
- 未证明：正式 GeoRF 已过拟合。要形成该 finding，仍需相同 GeoRF 的
  train/validation/OOF 与 fixed temporal test 对照，以及 forecast-time-safe
  feature contract。
"""
    (OUTPUT_ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Execute the isolated data audit and diagnostic probes."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(SOURCE_PATH)
    source_hash_before = sha256_file(SOURCE_PATH)
    if source_hash_before != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Authoritative source SHA-256 changed")
    expected_provider_paths = []
    for scope in (1, 2, 3):
        provider = PROVIDER_ROOT / f"result_partition_k40_compare_GF_thresholded_fs{scope}"
        expected_provider_paths.extend(
            [provider / "predictions_monthly.csv", provider / "run_manifest.json"]
        )
    for path in expected_provider_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes_before = {str(SOURCE_PATH): source_hash_before}
    hashes_before.update(
        {str(path): sha256_file(path) for path in expected_provider_paths}
    )

    source = pd.read_csv(SOURCE_PATH, low_memory=False)
    if source.shape != EXPECTED_SOURCE_SHAPE:
        raise RuntimeError(f"Unexpected source shape: {source.shape}")
    eth = source.loc[source["ISO3"].eq("ETH")].copy()
    del source
    if eth.shape != EXPECTED_ETH_SHAPE:
        raise RuntimeError(f"Unexpected ETH shape: {eth.shape}")
    eth["date_period"] = pd.PeriodIndex(eth["date"], freq="M")
    if eth[KEY].isna().any():
        raise RuntimeError("Null Ethiopia admin code")
    if eth.duplicated([KEY, "date"]).any():
        raise RuntimeError("Duplicate Ethiopia admin-month key")
    if eth[KEY].nunique() != EXPECTED_ETH_ADMINS:
        raise RuntimeError("Unexpected Ethiopia admin count")
    if eth["date_period"].nunique() != EXPECTED_ETH_MONTHS:
        raise RuntimeError("Unexpected Ethiopia month count")
    month_counts = eth.groupby("date_period").size()
    admin_counts = eth.groupby(KEY).size()
    if not (month_counts.eq(EXPECTED_ETH_ADMINS).all() and admin_counts.eq(EXPECTED_ETH_MONTHS).all()):
        raise RuntimeError("Ethiopia panel is not the expected balanced grid")

    raw_null = eth.drop(columns="date_period").isna()
    effective_missing = effective_missing_mask(eth.drop(columns="date_period"))
    profile_rows: list[dict[str, Any]] = []
    numeric_rows: list[dict[str, Any]] = []
    categorical_rows: list[dict[str, Any]] = []
    quality_flags: list[dict[str, Any]] = []
    pipeline_lineage_rows: list[dict[str, Any]] = []

    for column in eth.columns:
        if column == "date_period":
            continue
        series = eth[column]
        non_null = series.dropna()
        value_counts = non_null.value_counts(dropna=True)
        mode_value = value_counts.index[0] if not value_counts.empty else None
        mode_count = int(value_counts.iloc[0]) if not value_counts.empty else 0
        numeric_or_bool = pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
        inf_count = 0
        zero_rate = np.nan
        negative_rate = np.nan
        if numeric_or_bool:
            numeric = pd.to_numeric(series, errors="coerce")
            values = numeric.to_numpy(dtype=float)
            inf_count = int(np.isinf(values).sum())
            finite = numeric.loc[np.isfinite(values)].astype(float)
            if len(finite):
                zero_rate = float((finite == 0).mean())
                negative_rate = float((finite < 0).mean())
                quantiles = finite.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
                numeric_rows.append(
                    {
                        "column": column,
                        "feature_family": feature_family(column),
                        "finite_n": int(len(finite)),
                        "raw_null_n": int(series.isna().sum()),
                        "inf_n": inf_count,
                        "mean": float(finite.mean()),
                        "std": float(finite.std()),
                        "min": float(finite.min()),
                        "p01": float(quantiles.loc[0.01]),
                        "p05": float(quantiles.loc[0.05]),
                        "p25": float(quantiles.loc[0.25]),
                        "p50": float(quantiles.loc[0.50]),
                        "p75": float(quantiles.loc[0.75]),
                        "p95": float(quantiles.loc[0.95]),
                        "p99": float(quantiles.loc[0.99]),
                        "max": float(finite.max()),
                        "zero_rate_finite": zero_rate,
                        "negative_rate_finite": negative_rate,
                        "unique_finite": int(finite.nunique()),
                    }
                )
        else:
            categorical_rows.append(
                {
                    "column": column,
                    "feature_family": feature_family(column),
                    "non_null_n": int(series.notna().sum()),
                    "raw_null_n": int(series.isna().sum()),
                    "unique_non_null": int(series.nunique(dropna=True)),
                    "mode": mode_value,
                    "mode_n": mode_count,
                    "mode_rate_non_null": (
                        float(mode_count / len(non_null)) if len(non_null) else np.nan
                    ),
                }
            )
        effective_n = int(effective_missing[column].sum())
        unique_n = int(series.nunique(dropna=True))
        mode_rate = float(mode_count / len(non_null)) if len(non_null) else np.nan
        included_probe = (
            column not in TEXT_OR_DATE
            and column not in TARGET_RELATED
            and numeric_or_bool
        )
        profile_rows.append(
            {
                "column": column,
                "position_1based": int(eth.columns.get_loc(column) + 1),
                "raw_dtype": str(series.dtype),
                "feature_family": feature_family(column),
                "field_role": field_role(column),
                "loader_action": loader_action(column),
                "raw_null_n": int(raw_null[column].sum()),
                "raw_null_rate": float(raw_null[column].mean()),
                "inf_n": inf_count,
                "inf_rate": float(inf_count / len(eth)),
                "effective_missing_n": effective_n,
                "effective_missing_rate": float(effective_n / len(eth)),
                "unique_non_null": unique_n,
                "mode": mode_value,
                "mode_rate_non_null": mode_rate,
                "constant_non_null": bool(unique_n <= 1),
                "near_constant_ge_95pct": bool(mode_rate >= 0.95) if not np.isnan(mode_rate) else False,
                "zero_rate_finite": zero_rate,
                "negative_rate_finite": negative_rate,
                "included_in_lightweight_probe": bool(included_probe),
                "forecast_origin_availability": "undeclared_not_audited",
            }
        )
        pipeline_lineage_rows.append(
            {
                "column": column,
                "feature_family": feature_family(column),
                "field_role": field_role(column),
                "production_loader_action": loader_action(column),
                "lightweight_probe_action": (
                    "excluded_to_avoid_obvious_target_or_identifier_leakage"
                    if not included_probe
                    else "included_as_assembled_row_covariate"
                ),
            }
        )
        if unique_n <= 1:
            quality_flags.append(
                {"column": column, "flag": "constant_non_null", "value": str(mode_value), "severity": "warning"}
            )
        elif mode_rate >= 0.95:
            quality_flags.append(
                {"column": column, "flag": "near_constant_ge_95pct", "value": str(mode_value), "severity": "warning"}
            )
        if inf_count:
            quality_flags.append(
                {"column": column, "flag": "contains_infinity", "value": str(inf_count), "severity": "high"}
            )
        if effective_n / len(eth) >= 0.20:
            quality_flags.append(
                {"column": column, "flag": "effective_missing_ge_20pct", "value": f"{effective_n / len(eth):.6f}", "severity": "high"}
            )

    column_profile = pd.DataFrame(profile_rows)
    column_profile.to_csv(OUTPUT_ROOT / "column_profile.csv", index=False)
    pd.DataFrame(pipeline_lineage_rows).to_csv(
        OUTPUT_ROOT / "pipeline_column_lineage.csv",
        index=False,
    )
    pd.DataFrame(numeric_rows).to_csv(
        OUTPUT_ROOT / "numeric_descriptive_statistics.csv",
        index=False,
    )
    pd.DataFrame(categorical_rows).to_csv(
        OUTPUT_ROOT / "categorical_descriptive_statistics.csv",
        index=False,
    )
    pd.DataFrame(quality_flags).to_csv(
        OUTPUT_ROOT / "data_quality_flags.csv",
        index=False,
    )

    month_feature_rows = []
    month_summary_rows = []
    for month, indices in eth.groupby("date_period").groups.items():
        month_mask = effective_missing.loc[indices]
        month_summary_rows.append(
            {
                "month": str(month),
                "rows": int(len(indices)),
                "raw_missing_cells": int(raw_null.loc[indices].to_numpy().sum()),
                "raw_missing_cell_rate": float(raw_null.loc[indices].to_numpy().mean()),
                "effective_missing_cells": int(month_mask.to_numpy().sum()),
                "effective_missing_cell_rate": float(month_mask.to_numpy().mean()),
                "target_non_null_rows": int(eth.loc[indices, TARGET].notna().sum()),
                "target_positive_rows": int(eth.loc[indices, TARGET].fillna(0).sum()),
            }
        )
        rates = month_mask.mean()
        for column, rate in rates.items():
            month_feature_rows.append(
                {
                    "month": str(month),
                    "column": column,
                    "effective_missing_n": int(month_mask[column].sum()),
                    "effective_missing_rate": float(rate),
                }
            )
    month_feature = pd.DataFrame(month_feature_rows)
    month_feature.to_csv(OUTPUT_ROOT / "missingness_by_month_feature.csv", index=False)
    pd.DataFrame(month_summary_rows).to_csv(
        OUTPUT_ROOT / "missingness_by_month.csv",
        index=False,
    )

    admin_rows = []
    for admin, indices in eth.groupby(KEY).groups.items():
        admin_mask = effective_missing.loc[indices]
        admin_rows.append(
            {
                KEY: admin,
                "rows": int(len(indices)),
                "effective_missing_cells": int(admin_mask.to_numpy().sum()),
                "effective_missing_cell_rate": float(admin_mask.to_numpy().mean()),
                "target_non_null_rows": int(eth.loc[indices, TARGET].notna().sum()),
                "target_missing_rate": float(eth.loc[indices, TARGET].isna().mean()),
            }
        )
    pd.DataFrame(admin_rows).to_csv(
        OUTPUT_ROOT / "missingness_by_admin.csv",
        index=False,
    )

    missing_columns = [column for column in effective_missing if effective_missing[column].any()]
    pair_rows = []
    for left_index, left in enumerate(missing_columns):
        left_mask = effective_missing[left].to_numpy()
        for right in missing_columns[left_index + 1 :]:
            right_mask = effective_missing[right].to_numpy()
            union = int(np.logical_or(left_mask, right_mask).sum())
            intersection = int(np.logical_and(left_mask, right_mask).sum())
            pair_rows.append(
                {
                    "left_column": left,
                    "right_column": right,
                    "both_missing_n": intersection,
                    "either_missing_n": union,
                    "missingness_jaccard": float(intersection / union) if union else np.nan,
                }
            )
    pd.DataFrame(pair_rows).sort_values(
        "missingness_jaccard",
        ascending=False,
    ).to_csv(OUTPUT_ROOT / "missingness_pairwise_jaccard.csv", index=False)

    mask_array = effective_missing.to_numpy(dtype=np.uint8)
    packed = np.packbits(mask_array, axis=1)
    signatures = pd.Series(
        [row.tobytes().hex() for row in packed],
        index=eth.index,
        name="pattern_signature",
    )
    pattern_counts = signatures.value_counts().head(100)
    pattern_rows = []
    cumulative = 0
    for rank, (signature, count) in enumerate(pattern_counts.items(), start=1):
        first_index = signatures.index[signatures.eq(signature)][0]
        columns = effective_missing.columns[effective_missing.loc[first_index].to_numpy()].tolist()
        cumulative += int(count)
        pattern_rows.append(
            {
                "rank": rank,
                "pattern_signature": signature,
                "row_count": int(count),
                "row_rate": float(count / len(eth)),
                "cumulative_rate": float(cumulative / len(eth)),
                "missing_column_count": len(columns),
                "missing_columns": "|".join(columns),
            }
        )
    pd.DataFrame(pattern_rows).to_csv(
        OUTPUT_ROOT / "missing_pattern_top100.csv",
        index=False,
    )

    run_rows = []
    for column in effective_missing.columns:
        all_missing_months = month_feature.loc[
            month_feature["column"].eq(column)
            & month_feature["effective_missing_rate"].eq(1.0),
            "month",
        ]
        for start, end, count in contiguous_runs(pd.PeriodIndex(all_missing_months, freq="M")):
            run_rows.append(
                {
                    "column": column,
                    "all_missing_start": start,
                    "all_missing_end": end,
                    "consecutive_months": count,
                }
            )
    pd.DataFrame(run_rows).sort_values(
        ["consecutive_months", "column"],
        ascending=[False, True],
    ).to_csv(OUTPUT_ROOT / "all_missing_month_runs.csv", index=False)

    nonfinite_rows = []
    for column in eth.select_dtypes(include=[np.number]).columns:
        values = pd.to_numeric(eth[column], errors="coerce").to_numpy(dtype=float)
        indices = np.flatnonzero(np.isinf(values))
        for position in indices:
            nonfinite_rows.append(
                {
                    KEY: eth.iloc[position][KEY],
                    "date": eth.iloc[position]["date"],
                    "column": column,
                    "value": str(eth.iloc[position][column]),
                }
            )
    nonfinite = pd.DataFrame(nonfinite_rows)
    nonfinite.to_csv(OUTPUT_ROOT / "nonfinite_values.csv", index=False)
    if not nonfinite.empty:
        nonfinite.groupby("column").agg(
            rows=(KEY, "size"),
            admin_codes=(KEY, "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        ).reset_index().to_csv(
            OUTPUT_ROOT / "nonfinite_summary.csv",
            index=False,
        )

    target_available = eth.loc[eth[TARGET].notna()].copy()
    target_available[TARGET] = target_available[TARGET].astype(int)
    pd.DataFrame(
        [
            {
                "target": TARGET,
                "rows_total": len(eth),
                "rows_labeled": len(target_available),
                "rows_missing": int(eth[TARGET].isna().sum()),
                "label_availability_rate": float(eth[TARGET].notna().mean()),
                "negative_n": int((target_available[TARGET] == 0).sum()),
                "positive_n": int((target_available[TARGET] == 1).sum()),
                "positive_rate_labeled": float(target_available[TARGET].mean()),
            }
        ]
    ).to_csv(OUTPUT_ROOT / "target_distribution.csv", index=False)
    target_available.assign(year=target_available["date_period"].dt.year).groupby("year")[TARGET].agg(
        labeled_rows="size",
        positive_n="sum",
        positive_rate="mean",
    ).reset_index().to_csv(OUTPUT_ROOT / "target_by_year.csv", index=False)
    target_available.assign(calendar_month=target_available["date_period"].dt.month).groupby("calendar_month")[TARGET].agg(
        labeled_rows="size",
        positive_n="sum",
        positive_rate="mean",
    ).reset_index().to_csv(OUTPUT_ROOT / "target_by_calendar_month.csv", index=False)

    probe_frame, features_with_id, features_without_id = build_probe_frame(target_available)
    feature_sets = {
        "with_admin_id": features_with_id,
        "without_admin_id": features_without_id,
    }
    temporal_metrics, temporal_predictions, importances, temporal_overlaps = run_temporal_probe(
        target_available,
        probe_frame,
        feature_sets,
    )
    random_metrics, random_overlaps = run_random_split_sensitivity(
        target_available,
        probe_frame,
        features_without_id,
    )
    all_probe_metrics = pd.concat([temporal_metrics, random_metrics], ignore_index=True)
    all_probe_metrics.to_csv(OUTPUT_ROOT / "overfitting_probe_metrics.csv", index=False)
    temporal_predictions.to_csv(
        OUTPUT_ROOT / "overfitting_probe_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    importances.to_csv(OUTPUT_ROOT / "overfitting_probe_feature_importance.csv", index=False)
    pd.concat([temporal_overlaps, random_overlaps], ignore_index=True).to_csv(
        OUTPUT_ROOT / "split_overlap_audit.csv",
        index=False,
    )
    gap_rows = []
    for (design, candidate), group in all_probe_metrics.groupby(["split_design", "candidate"]):
        indexed = group.set_index("split")
        if {"train", "validation", "test"}.issubset(indexed.index):
            gap_rows.append(
                {
                    "split_design": design,
                    "candidate": candidate,
                    "train_minus_validation_f1": float(indexed.loc["train", "f1"] - indexed.loc["validation", "f1"]),
                    "train_minus_test_f1": float(indexed.loc["train", "f1"] - indexed.loc["test", "f1"]),
                    "validation_minus_test_f1": float(indexed.loc["validation", "f1"] - indexed.loc["test", "f1"]),
                    "train_minus_test_average_precision": float(indexed.loc["train", "average_precision"] - indexed.loc["test", "average_precision"]),
                    "test_minus_train_brier": float(indexed.loc["test", "brier"] - indexed.loc["train", "brier"]),
                }
            )
    pd.DataFrame(gap_rows).to_csv(
        OUTPUT_ROOT / "overfitting_generalization_gaps.csv",
        index=False,
    )

    eth_codes = set(normalize_admin_code(eth[KEY]).dropna())
    georf_summary, georf_monthly, partition_support, stability, provider_paths = released_georf_diagnostics(eth_codes)
    if set(provider_paths) != set(expected_provider_paths):
        raise RuntimeError("Released-provider path inventory changed")
    georf_summary.to_csv(OUTPUT_ROOT / "released_georf_eth_summary.csv", index=False)
    georf_monthly.to_csv(OUTPUT_ROOT / "released_georf_eth_monthly.csv", index=False)
    partition_support.to_csv(OUTPUT_ROOT / "released_georf_partition_support.csv", index=False)
    stability.to_csv(OUTPUT_ROOT / "released_georf_stability_summary.csv", index=False)

    export = eth.drop(columns="date_period")
    export.to_csv(
        OUTPUT_ROOT / "fewsnet_eth_pre_georf.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    cohort_manifest = {
        "iso3_filter": "ETH",
        "canonical_key": KEY,
        "source_path": str(SOURCE_PATH),
        "source_sha256": source_hash_before,
        "source_shape": list(EXPECTED_SOURCE_SHAPE),
        "eth_shape": list(EXPECTED_ETH_SHAPE),
        "eth_admin_codes": int(eth[KEY].nunique()),
        "eth_months": int(eth["date_period"].nunique()),
        "date_min": str(eth["date_period"].min()),
        "date_max": str(eth["date_period"].max()),
        "balanced_admin_month_grid": True,
        "duplicate_admin_month_keys": 0,
        "target": TARGET,
        "pipeline_run": False,
    }
    (OUTPUT_ROOT / "cohort_manifest.json").write_text(
        json.dumps(cohort_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reproduction = {
        "analysis_type": "pre_georf_assembled_panel_profile_and_lightweight_risk_probe",
        "pipeline_run": False,
        "source_sha256": source_hash_before,
        "eth_filter": "ISO3 == 'ETH'",
        "temporal_probe_split": {
            "train": "2010-2018 labeled rows",
            "validation": "2019-2020 labeled rows",
            "test": "2021-2024 labeled rows",
        },
        "threshold_selection": "validation F1 only; grid 0.05 to 0.95 by 0.01",
        "probe_exclusions": sorted(TEXT_OR_DATE | TARGET_RELATED),
        "forecast_origin_availability": "not established; probe is not forecast-time-safe",
        "python_version": sys.version,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "command": "python3 EthiopiaForecastingExperiment/outputs/baseline_audit/fewsnet_eth_pre_georf_20260831/run_audit.py",
    }
    (OUTPUT_ROOT / "reproduction.json").write_text(
        json.dumps(reproduction, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_pipeline_lineage()

    source_paths = [SOURCE_PATH, *provider_paths]
    hashes_after = {str(path): sha256_file(path) for path in source_paths}
    integrity_rows = []
    for path in source_paths:
        before = hashes_before[str(path)]
        after = hashes_after[str(path)]
        integrity_rows.append(
            {
                "source_path": str(path),
                "sha256_before": before,
                "sha256_after": after,
                "unchanged": before == after,
            }
        )
    integrity = pd.DataFrame(integrity_rows)
    if not integrity["unchanged"].all():
        raise RuntimeError("A source file changed during the audit")
    integrity.to_csv(OUTPUT_ROOT / "source_integrity_audit.csv", index=False)

    source_manifest_rows = []
    for path in source_paths:
        source_manifest_rows.append(
            {
                "role": "authoritative_assembled_panel" if path == SOURCE_PATH else "frozen_released_georf_provider",
                "source_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": hashes_after[str(path)],
                "read_only": True,
            }
        )
    pd.DataFrame(source_manifest_rows).to_csv(OUTPUT_ROOT / "SOURCE_MANIFEST.csv", index=False)

    grill = """# Docs-based reverse pressure test

| Failure condition | Result | Evidence |
|---|---|---|
| Wrong Ethiopia boundary | PASS | Exact `ISO3 == \"ETH\"`; 187,200 rows, 1,040 admins, 180 months. |
| Wrong input boundary | PASS | Profiles the assembled 88-column panel before loader; excludes Stage 2/3 artifacts from feature statistics. |
| Source drift | PASS | Authoritative SHA-256 matches the frozen evidence and is unchanged after execution. |
| Duplicate or incomplete panel keys | PASS | `(FEWSNET_admin_code,date)` is unique and the 1,040 x 180 grid is complete. |
| Missing labels treated as negatives | PASS | Model probes use only non-null `fews_ipc_crisis`; missing label months remain missing. |
| Infinity overlooked | PASS | Raw null and pipeline-effective missingness are separate; z-score infinities are explicitly inventoried. |
| Obvious outcome/projection leakage in probe | PASS | IPC phase, HA, adjusted IPC and near/medium provider projections are excluded. |
| Forecast-time safety assumed | WARN | Contemporaneous assembled covariates remain; availability classes are undeclared. Probe is diagnostic only. |
| Random row validation presented as generalization | PASS | Random split is labeled sensitivity only and admin overlap is recorded. |
| Test labels used for threshold/model selection | PASS | Threshold is selected on 2019-2020 validation only; fixed candidate definitions are not chosen on test. |
| Lightweight RF called GeoRF | PASS | Reports distinguish diagnostic pooled RF fits from frozen released GeoRF predictions. |
| GeoRF overfitting declared without train metrics | WARN | Frozen providers contain test predictions but no comparable GeoRF train/OOF metrics; conclusion remains risk, not proof. |
| Low-support 2021-06 concealed | PASS | Monthly and released-result tables retain the one-row ETH month and stability summaries exclude it explicitly. |

The WARN items are unresolved evidence limitations, not execution failures.
"""
    (OUTPUT_ROOT / "grill_reverse_stress_test.md").write_text(grill, encoding="utf-8")
    write_readme(column_profile, temporal_metrics, random_metrics, georf_summary, stability)

    checksum_lines = []
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(OUTPUT_ROOT).as_posix()}")
    (OUTPUT_ROOT / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )

    print(f"OUTPUT_ROOT={OUTPUT_ROOT}")
    print(f"ETH_SHAPE={eth.shape[0]}x{eth.shape[1] - 1}")
    print(f"LABELED_ROWS={len(target_available)}")
    print(f"OUTPUT_FILES_HASHED={len(checksum_lines)}")
    print("TEMPORAL_PROBE")
    print(
        temporal_metrics[
            ["candidate", "split", "n", "threshold", "f1", "average_precision", "brier"]
        ].to_csv(index=False, float_format="%.6f").strip()
    )
    print("RELEASED_GEORF")
    print(
        georf_summary[
            ["forecasting_horizon", "model", "n", "precision", "recall", "f1", "average_precision", "brier"]
        ].to_csv(index=False, float_format="%.6f").strip()
    )


if __name__ == "__main__":
    main()
