#!/usr/bin/env python3
"""Build GeoRF partitioned SHAP feature-group heatmap artifacts."""

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

RANDOM_STATE = 5
TARGET_MONTHS = (2, 6, 10)
SCOPE_TO_HORIZON_MONTHS = {"fs1": 4, "fs2": 8, "fs3": 12}
SCOPE_TO_INT = {"fs1": 1, "fs2": 2, "fs3": 3}
HORIZON_LABELS = {
    "fs1": "4-month horizon",
    "fs2": "8-month horizon",
    "fs3": "12-month horizon",
}
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final_artifacts_in_paper_updated" / "01_main_results"
DEFAULT_STAGE3_ROOT = REPO_ROOT
WSL_MOUNT_ROOT = Path("/mnt")

FEATURE_GROUPS: dict[str, dict[str, Any]] = {
    "weather": {
        "display": "Weather",
        "base_columns": [
            "Rainf_f_tavg_mean",
            "Rainf_zscore",
            "Tair_f_tavg_mean",
            "Tair_zscore",
        ],
    },
    "agri": {
        "display": "Agri",
        "base_columns": [
            "AEZ_10000",
            "AEZ_12000",
            "AEZ_15000",
            "AEZ_17000",
            "AEZ_19000",
            "AEZ_25000",
            "AEZ_31000",
            "AEZ_32000",
            "AEZ_33000",
            "AEZ_34000",
            "AEZ_36000",
            "AEZ_38000",
            "AEZ_4000",
            "AEZ_40000",
            "AEZ_43000",
            "AEZ_7000",
            "AEZ_9000",
            "EVI",
            "crop",
            "distance_to_river",
            "gpp_mean",
            "range",
        ],
    },
    "conflict": {
        "display": "Conflict",
        "base_columns": [
            "distance_to_nearest_acled",
            "event_count_battles",
            "event_count_battles_w10",
            "event_count_battles_w5",
            "event_count_explosions",
            "event_count_explosions_w10",
            "event_count_explosions_w5",
            "event_count_violence",
            "event_count_violence_w10",
            "event_count_violence_w5",
            "sum_fatalities_battles",
            "sum_fatalities_battles_w10",
            "sum_fatalities_battles_w5",
            "sum_fatalities_explosions",
            "sum_fatalities_explosions_w10",
            "sum_fatalities_explosions_w5",
            "sum_fatalities_violence",
            "sum_fatalities_violence_w10",
            "sum_fatalities_violence_w5",
        ],
    },
    "econ": {
        "display": "Econ",
        "base_columns": [
            "CC",
            "CPI",
            "Food_CPI",
            "Food_food_inflation",
            "GDP",
            "gini",
            "market_access",
            "market_distance",
            "nightlight",
            "nightlight_sd",
            "pop",
        ],
    },
    "food_prices": {
        "display": "Food Prices",
        "base_columns": [
            "FAO_price",
            "WFP_Price",
            "WFP_Price_std",
        ],
    },
    "geographic": {
        "display": "Geographic",
        "base_columns": [
            "elevation",
            "ruggedness",
            "sg_cec_5-15cm",
            "sg_cfvo_5-15cm",
            "sg_nitrogen_5-15cm",
            "sg_phh2o_5-15cm",
            "sg_soc_5-15cm",
            "slope",
        ],
    },
    "lag": {
        "display": "Lag",
        "base_columns": [
            "fews_ipc_lag",
            "fews_ipc_crisis_lag",
        ],
    },
}

BASE_COLUMN_TO_GROUP = {
    column: group
    for group, meta in FEATURE_GROUPS.items()
    if group != "lag"
    for column in meta["base_columns"]
}
TARGET_LAG_RE = re.compile(r"^fews_ipc(?:_crisis)?_lag_?\d+m?$")
LAG_SUFFIX_RE = re.compile(r"_lag_?\d+m?$")


@dataclass(frozen=True)
class FeatureGroupResolution:
    feature_to_group: dict[str, str]
    matched_features: dict[str, list[str]]
    missing_base_columns: dict[str, list[str]]
    unmatched_features: list[str]


def strip_lag_suffix(feature_name: str) -> str:
    return LAG_SUFFIX_RE.sub("", str(feature_name))


def assign_feature_group(feature_name: str) -> str | None:
    name = str(feature_name)
    if TARGET_LAG_RE.match(name):
        return "lag"
    base_name = strip_lag_suffix(name)
    return BASE_COLUMN_TO_GROUP.get(base_name)


def resolve_feature_group_matches(feature_names: list[str]) -> FeatureGroupResolution:
    feature_to_group = {}
    matched_features = {group: [] for group in FEATURE_GROUPS}
    unmatched_features = []
    for feature in [str(name) for name in feature_names]:
        group = assign_feature_group(feature)
        if group is None:
            unmatched_features.append(feature)
            continue
        feature_to_group[feature] = group
        matched_features[group].append(feature)
    missing_base_columns = {}
    stripped_by_group = {
        group: {strip_lag_suffix(feature) for feature in features}
        for group, features in matched_features.items()
    }
    for group, meta in FEATURE_GROUPS.items():
        if group == "lag":
            missing_base_columns[group] = []
            continue
        expected = [str(column) for column in meta["base_columns"]]
        observed = stripped_by_group[group]
        missing_base_columns[group] = [
            column for column in expected if column not in observed
        ]
    return FeatureGroupResolution(
        feature_to_group,
        matched_features,
        missing_base_columns,
        unmatched_features,
    )


def validate_feature_group_resolution(resolved: FeatureGroupResolution) -> None:
    empty_groups = [
        FEATURE_GROUPS[group]["display"]
        for group, features in resolved.matched_features.items()
        if not features
    ]
    if empty_groups:
        raise ValueError(
            f"Feature groups have no matched prepared features: {empty_groups}"
        )


def collapse_shap_values(raw_values: Any, n_samples: int, n_features: int) -> np.ndarray:
    if isinstance(raw_values, list):
        matrices = []
        for raw_matrix in raw_values:
            matrix = np.asarray(raw_matrix)
            if matrix.ndim != 2:
                matrix = matrix.reshape(matrix.shape[0], -1)
            matrices.append(matrix)
        if not matrices:
            raise ValueError("No SHAP matrices returned")
        values = np.mean(np.stack(matrices, axis=0), axis=0)
    else:
        values = np.asarray(raw_values)
        if values.ndim == 3:
            if values.shape[0] == n_samples and values.shape[1] == n_features:
                values = values.mean(axis=2)
            elif values.shape[1] == n_samples and values.shape[2] == n_features:
                values = values.mean(axis=0)
            else:
                values = values.reshape(values.shape[0], -1)
        elif (
            values.ndim == 2
            and values.shape[0] == n_samples
            and values.shape[1] == n_features
        ):
            pass
        elif (
            values.ndim == 2
            and values.shape[0] == n_samples
            and values.shape[1] % n_features == 0
        ):
            class_count = values.shape[1] // n_features
            values = values.reshape(n_samples, class_count, n_features).mean(axis=1)
        else:
            values = values.reshape(n_samples, -1)
    if values.shape != (n_samples, n_features):
        raise ValueError(
            f"Collapsed SHAP shape {values.shape} does not match expected "
            f"{(n_samples, n_features)}"
        )
    return values.astype(float, copy=False)


def build_monthly_group_rows(
    *,
    scope: str,
    horizon_months: int,
    target_month: str,
    shap_values: np.ndarray,
    feature_names: list[str],
    resolved: FeatureGroupResolution,
    fallback_samples: int,
    evaluated_samples: int,
) -> list[dict[str, Any]]:
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP feature count {shap_values.shape[1]} does not match "
            f"feature names {len(feature_names)}"
        )
    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=[str(name) for name in feature_names],
    )
    raw_by_group = {}
    matched_count_by_group = {}
    for group in FEATURE_GROUPS:
        group_features = [
            feature
            for feature in resolved.matched_features[group]
            if feature in mean_abs.index
        ]
        matched_count_by_group[group] = len(group_features)
        raw_by_group[group] = (
            float(mean_abs.loc[group_features].sum()) if group_features else 0.0
        )
    denominator = float(sum(raw_by_group.values()))
    if denominator <= 0:
        raise ValueError(
            f"Non-positive SHAP denominator for {scope} {target_month}: "
            f"{denominator}"
        )
    rows = []
    for group, meta in FEATURE_GROUPS.items():
        raw_value = raw_by_group[group]
        rows.append(
            {
                "scope": scope,
                "horizon_months": int(horizon_months),
                "forecasting_horizon": HORIZON_LABELS[scope],
                "target_month": str(target_month),
                "group": group,
                "display_group": meta["display"],
                "raw_mean_abs_shap": raw_value,
                "group_share": raw_value / denominator,
                "matched_feature_count": int(matched_count_by_group[group]),
                "normalization_denominator": denominator,
                "fallback_samples": int(fallback_samples),
                "evaluated_samples": int(evaluated_samples),
            }
        )
    return rows


def summarize_group_shares(
    monthly: pd.DataFrame,
    expected_month_count: int = 12,
) -> pd.DataFrame:
    monthly = monthly.copy()
    if "forecasting_horizon" not in monthly.columns and "horizon_months" in monthly.columns:
        monthly["forecasting_horizon"] = (
            monthly["horizon_months"].astype(int).astype(str) + "-month horizon"
        )
    for optional_column in (
        "raw_mean_abs_shap",
        "fallback_samples",
        "evaluated_samples",
    ):
        if optional_column not in monthly.columns:
            monthly[optional_column] = 0.0
    required = {
        "scope",
        "horizon_months",
        "forecasting_horizon",
        "group",
        "display_group",
        "target_month",
        "group_share",
    }
    missing = required - set(monthly.columns)
    if missing:
        raise ValueError(f"Monthly SHAP data missing columns: {sorted(missing)}")
    grouped = monthly.groupby(
        ["scope", "horizon_months", "forecasting_horizon", "group", "display_group"],
        as_index=False,
    ).agg(
        mean_share=("group_share", "mean"),
        sd_share=("group_share", "std"),
        n_months=("target_month", "nunique"),
        mean_raw_mean_abs_shap=("raw_mean_abs_shap", "mean"),
        mean_fallback_samples=("fallback_samples", "mean"),
        mean_evaluated_samples=("evaluated_samples", "mean"),
    )
    grouped["sd_share"] = grouped["sd_share"].fillna(0.0)
    wrong_count = grouped[grouped["n_months"] != expected_month_count]
    if not wrong_count.empty:
        details = wrong_count[["scope", "group", "n_months"]].to_dict("records")
        raise ValueError(
            f"Expected {expected_month_count} months for every scope/group, "
            f"got {details}"
        )
    order = {
        (group, scope): (group_index, scope_index)
        for group_index, group in enumerate(FEATURE_GROUPS)
        for scope_index, scope in enumerate(SCOPE_TO_HORIZON_MONTHS)
    }
    grouped["_order"] = grouped.apply(
        lambda row: order[(row["group"], row["scope"])],
        axis=1,
    )
    return (
        grouped.sort_values("_order", kind="mergesort")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def build_heatmap_matrices(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_order = [meta["display"] for meta in FEATURE_GROUPS.values()]
    column_order = [HORIZON_LABELS[scope] for scope in SCOPE_TO_HORIZON_MONTHS]
    values = summary.pivot(
        index="display_group",
        columns="forecasting_horizon",
        values="mean_share",
    ).reindex(index=row_order, columns=column_order)
    sd_values = summary.pivot(
        index="display_group",
        columns="forecasting_horizon",
        values="sd_share",
    ).reindex(index=row_order, columns=column_order)
    if values.isna().any().any() or sd_values.isna().any().any():
        raise ValueError("Heatmap matrix contains missing group-horizon cells")
    annotations = values.copy().astype(object)
    for row in row_order:
        for column in column_order:
            annotations.loc[row, column] = (
                f"{values.loc[row, column] * 100:.1f}%\n"
                f"+/- {sd_values.loc[row, column] * 100:.1f}"
            )
    return values, annotations


def resolve_path(path: Path | str) -> Path:
    """Resolve Windows-style and local paths from WSL-friendly scripts."""
    candidate = Path(path)
    if candidate.exists():
        return candidate
    raw = str(path)
    if re.match(r"^[A-Za-z]:[\\/]", raw):
        drive = raw[0].lower()
        converted = WSL_MOUNT_ROOT / drive / raw[3:].replace("\\", "/")
        if converted.exists():
            return converted
    return candidate


def sort_panel_for_feature_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """Sort panel rows to match prepare_features row order."""
    required = {"FEWSNET_admin_code", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset missing columns required for sorting: {sorted(missing)}"
        )

    sorted_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(sorted_df["date"]):
        sorted_df["date"] = pd.to_datetime(sorted_df["date"])
    return (
        sorted_df.sort_values(
            by=["FEWSNET_admin_code", "date"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def evaluation_months(start_month: str, end_month: str) -> list[pd.Period]:
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    months = pd.period_range(start=start, end=end, freq="M")
    return [month for month in months if month.month in TARGET_MONTHS]


def default_partition_maps_for_scope(stage3_root: Path, scope: str) -> dict[str, Path]:
    """Return refined general and month-specific partition-map paths for one scope."""
    refined = stage3_root / f"result_partition_k40_compare_GF_{scope}" / "refined"
    return {
        "general": refined / "cluster_mapping_k40_nc17_general_refined_contig3.csv",
        "m2": refined / "cluster_mapping_k40_nc13_m2_refined_contig3.csv",
        "m6": refined / "cluster_mapping_k40_nc11_m6_refined_contig3.csv",
        "m10": refined / "cluster_mapping_k40_nc16_m10_refined_contig3.csv",
    }


def select_partition_map(target_month: pd.Period, maps: dict[str, Path]) -> Path:
    month_key = f"m{pd.Period(target_month, freq='M').month}"
    return maps.get(month_key, maps["general"])


def validate_partition_maps(partition_maps_by_scope: dict[str, dict[str, Path]]) -> None:
    """Fail if any selected partition-map file is missing."""
    missing = []
    for scope, maps in partition_maps_by_scope.items():
        for label, path in maps.items():
            if not resolve_path(path).is_file():
                missing.append(f"{scope}:{label}:{path}")
    if missing:
        raise FileNotFoundError(f"Missing required partition maps: {missing}")


def write_tabular_outputs(
    *,
    monthly: pd.DataFrame,
    summary: pd.DataFrame,
    manifest: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Write CSV, manifest, and explanatory note outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = output_dir / "georf_partitioned_shap_monthly.csv"
    summary_path = output_dir / "georf_partitioned_shap_group_summary.csv"
    manifest_path = output_dir / "georf_partitioned_shap_manifest.json"
    note_path = output_dir / "georf_partitioned_shap_note.md"

    monthly.to_csv(monthly_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        **manifest,
        "output_paths": {
            "monthly_csv": str(monthly_path),
            "summary_csv": str(summary_path),
            "manifest_json": str(manifest_path),
            "note_md": str(note_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    note_path.write_text(
        "\n".join(
            [
                "# GeoRF Partitioned SHAP Group Heatmap",
                "",
                "Values are relative SHAP attribution shares for the partitioned GeoRF model.",
                "They are not causal effects and are not retraining ablation deltas.",
                "Each heatmap cell reports the mean group share of mean absolute SHAP values across the 12 February, June, and October target months in 2021-2024, plus one standard deviation.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "monthly_csv": monthly_path,
        "summary_csv": summary_path,
        "manifest_json": manifest_path,
        "note_md": note_path,
    }


def write_heatmap(summary: pd.DataFrame, output_dir: Path, dpi: int = 300) -> dict[str, Path]:
    """Write paper-facing heatmap PNG and PDF."""
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except ModuleNotFoundError:
        sns = None

    output_dir.mkdir(parents=True, exist_ok=True)
    values, annotations = build_heatmap_matrices(summary)

    png_path = output_dir / "georf_partitioned_shap_group_heatmap.png"
    pdf_path = output_dir / "georf_partitioned_shap_group_heatmap.pdf"
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=dpi)
    try:
        if sns is not None:
            sns.heatmap(
                values * 100.0,
                annot=annotations,
                fmt="",
                cmap="YlGnBu",
                linewidths=0.7,
                linecolor="white",
                cbar_kws={"label": "Group share of mean |SHAP| (%)"},
                ax=ax,
            )
        else:
            image = ax.imshow(values.to_numpy() * 100.0, cmap="YlGnBu")
            ax.set_xticks(range(len(values.columns)))
            ax.set_xticklabels(values.columns, rotation=0)
            ax.set_yticks(range(len(values.index)))
            ax.set_yticklabels(values.index)
            for row_index, row_label in enumerate(values.index):
                for column_index, column_label in enumerate(values.columns):
                    ax.text(
                        column_index,
                        row_index,
                        annotations.loc[row_label, column_label],
                        ha="center",
                        va="center",
                    )
            fig.colorbar(
                image,
                ax=ax,
                label="Group share of mean |SHAP| (%)",
            )
        ax.set_xlabel("Forecasting horizon")
        ax.set_ylabel("Feature group")
        ax.set_title("GeoRF Partitioned SHAP Attribution Share")
        fig.tight_layout()
        fig.savefig(png_path, dpi=dpi)
        fig.savefig(pdf_path)
    finally:
        plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def sampled_indices(n_rows: int, max_samples: int, random_state: int) -> np.ndarray:
    """Return deterministic row indices for optional SHAP sampling."""
    if max_samples <= 0 or n_rows <= max_samples:
        return np.arange(n_rows)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_rows, size=max_samples, replace=False))


def shap_values_for_partitioned_models(
    *,
    models: dict[int, Any],
    X_test: np.ndarray,
    X_group_test: np.ndarray,
    feature_names: list[str],
    max_samples_per_month: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Compute SHAP values using each sample's local partition RF model."""
    import shap

    n_features = len(feature_names)
    shap_blocks: list[np.ndarray] = []
    evaluated_samples = 0
    fallback_samples = 0
    missing_model_samples = 0
    unmapped_samples = 0

    selected = sampled_indices(len(X_test), max_samples_per_month, random_state)
    X_selected = X_test[selected]
    group_selected = X_group_test[selected]

    for partition_id in sorted(np.unique(group_selected)):
        partition_mask = group_selected == partition_id
        sample_count = int(partition_mask.sum())
        if int(partition_id) < 0:
            fallback_samples += sample_count
            unmapped_samples += sample_count
            continue

        model = models.get(int(partition_id))
        if model is None:
            fallback_samples += sample_count
            missing_model_samples += sample_count
            continue

        X_partition = X_selected[partition_mask]
        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(X_partition)
        shap_matrix = collapse_shap_values(
            raw_values,
            n_samples=X_partition.shape[0],
            n_features=n_features,
        )
        shap_blocks.append(shap_matrix)
        evaluated_samples += int(X_partition.shape[0])

    if not shap_blocks:
        raise ValueError(
            "No local partition SHAP values were computed; all samples used fallback"
        )

    return np.vstack(shap_blocks), {
        "selected_samples": int(len(selected)),
        "evaluated_samples": int(evaluated_samples),
        "fallback_samples": int(fallback_samples),
        "missing_model_samples": int(missing_model_samples),
        "unmapped_samples": int(unmapped_samples),
    }


def prepare_scope_context(
    data_path: Path | str,
    forecasting_scope: int,
    *,
    load_data_fn: Any | None = None,
    prepare_features_fn: Any | None = None,
) -> dict[str, Any]:
    """Load data and prepare features for one forecasting scope."""
    if load_data_fn is None:
        from src.preprocess.preprocess import load_and_preprocess_data as load_data_fn

    if prepare_features_fn is None:
        from src.feature.feature import prepare_features as prepare_features_fn

    df = sort_panel_for_feature_alignment(
        load_data_fn(str(resolve_path(data_path)))
    )

    if "latitude" in df.columns and "longitude" in df.columns:
        X_loc = df[["latitude", "longitude"]].values
    elif "lat" in df.columns and "lon" in df.columns:
        X_loc = df[["lat", "lon"]].values
    else:
        raise ValueError("Dataset must have latitude/longitude or lat/lon columns")

    temp_group = np.zeros(len(df), dtype=int)
    (
        X,
        y,
        _l1_index,
        _l2_index,
        years,
        terms,
        dates,
        feature_columns,
    ) = prepare_features_fn(
        df,
        temp_group,
        X_loc,
        forecasting_scope=forecasting_scope,
    )
    if "FEWSNET_admin_code" not in df.columns:
        raise ValueError("Dataset must include FEWSNET_admin_code")

    return {
        "df": df,
        "X": X,
        "y": y.astype(int),
        "X_loc": X_loc,
        "years": years,
        "terms": terms,
        "dates": dates,
        "feature_columns": [str(column) for column in feature_columns],
        "admin_codes": df["FEWSNET_admin_code"].values,
    }


def run_scope_month(
    *,
    context: dict[str, Any],
    scope: str,
    test_month: pd.Period,
    partition_map: Path,
    train_window_months: int,
    max_samples_per_month: int,
    random_state: int,
    resolved: FeatureGroupResolution,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train partitioned RF models and return SHAP group rows for one scope-month."""
    from config import LAGS_MONTHS
    from scripts.compare_partitioned_vs_pooled_rf_k40_nc4 import (
        MIN_PARTITION_SAMPLES,
        create_partition_group_array,
        load_partition_mapping,
        train_partitioned_model,
    )
    from src.customize.customize import train_test_split_rolling_window
    from src.utils.lag_schedules import forecasting_scope_to_lag

    forecasting_scope = SCOPE_TO_INT[scope]
    active_lag = forecasting_scope_to_lag(forecasting_scope, LAGS_MONTHS)

    partition_df = load_partition_mapping(str(resolve_path(partition_map)))
    X_group, _df_with_partition = create_partition_group_array(
        context["df"],
        partition_df,
    )

    split = train_test_split_rolling_window(
        context["X"],
        context["y"],
        context["X_loc"],
        X_group,
        context["years"],
        context["dates"],
        test_month=test_month,
        active_lag=active_lag,
        train_window_months=train_window_months,
        admin_codes=context["admin_codes"],
    )
    if len(split) == 10:
        (
            X_train,
            y_train,
            _xloc_train,
            X_group_train,
            X_test,
            _y_test,
            _xloc_test,
            X_group_test,
            _train_admin,
            _test_admin,
        ) = split
    else:
        (
            X_train,
            y_train,
            _xloc_train,
            X_group_train,
            X_test,
            _y_test,
            _xloc_test,
            X_group_test,
        ) = split

    if len(X_test) == 0:
        raise ValueError(f"No test samples for {scope} {test_month}")

    models = train_partitioned_model(
        X_train,
        y_train.astype(int),
        X_group_train,
        lower_model="rf",
        min_samples=MIN_PARTITION_SAMPLES,
    )
    shap_matrix, diagnostics = shap_values_for_partitioned_models(
        models=models,
        X_test=X_test,
        X_group_test=X_group_test,
        feature_names=context["feature_columns"],
        max_samples_per_month=max_samples_per_month,
        random_state=random_state,
    )
    rows = build_monthly_group_rows(
        scope=scope,
        horizon_months=SCOPE_TO_HORIZON_MONTHS[scope],
        target_month=str(test_month),
        shap_values=shap_matrix,
        feature_names=context["feature_columns"],
        resolved=resolved,
        fallback_samples=diagnostics["fallback_samples"],
        evaluated_samples=diagnostics["evaluated_samples"],
    )
    return rows, {
        **diagnostics,
        "scope": scope,
        "target_month": str(test_month),
        "partition_map": str(partition_map),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(X_test)),
        "active_lag_months": int(active_lag),
    }


def run_analysis(args: argparse.Namespace) -> dict[str, Path]:
    """Run all scope-month SHAP jobs and write artifacts."""
    output_dir = Path(args.output_dir)
    stage3_root = Path(args.stage3_root)
    months = evaluation_months(args.start_month, args.end_month)
    if len(months) != 12:
        raise ValueError(
            f"Expected 12 target months, got {len(months)} from "
            f"{args.start_month} to {args.end_month}"
        )

    partition_maps_by_scope = {
        scope: default_partition_maps_for_scope(stage3_root, scope)
        for scope in SCOPE_TO_HORIZON_MONTHS
    }
    validate_partition_maps(partition_maps_by_scope)

    all_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    feature_group_resolution: FeatureGroupResolution | None = None
    feature_group_resolution_by_scope: dict[str, dict[str, Any]] = {}

    for scope in SCOPE_TO_HORIZON_MONTHS:
        print(f"\nPreparing scope {scope} ({HORIZON_LABELS[scope]})")
        context = prepare_scope_context(args.data, SCOPE_TO_INT[scope])
        resolved = resolve_feature_group_matches(context["feature_columns"])
        validate_feature_group_resolution(resolved)
        feature_group_resolution_by_scope[scope] = {
            "feature_group_matches": resolved.matched_features,
            "feature_group_missing_base_columns": resolved.missing_base_columns,
            "unmatched_features": resolved.unmatched_features,
        }
        if feature_group_resolution is None:
            feature_group_resolution = resolved

        for test_month in months:
            partition_map = select_partition_map(
                test_month,
                partition_maps_by_scope[scope],
            )
            print(f"Running {scope} {test_month} with {partition_map}")
            rows, diag = run_scope_month(
                context=context,
                scope=scope,
                test_month=test_month,
                partition_map=partition_map,
                train_window_months=args.train_window,
                max_samples_per_month=args.max_shap_samples,
                random_state=args.random_state,
                resolved=resolved,
            )
            all_rows.extend(rows)
            diagnostics.append(diag)

    monthly = pd.DataFrame(all_rows)
    summary = summarize_group_shares(monthly, expected_month_count=12)
    plot_outputs = write_heatmap(summary, output_dir=output_dir, dpi=args.dpi)

    assert feature_group_resolution is not None
    from scripts.compare_partitioned_vs_pooled_rf_k40_nc4 import RF_PARAMS

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "source_csv": str(args.data),
        "scope_to_horizon_months": SCOPE_TO_HORIZON_MONTHS,
        "evaluated_months": [str(month) for month in months],
        "partition_maps": {
            scope: {label: str(path) for label, path in maps.items()}
            for scope, maps in partition_maps_by_scope.items()
        },
        "rf_params": RF_PARAMS,
        "train_window_months": int(args.train_window),
        "max_shap_samples_per_month": int(args.max_shap_samples),
        "feature_group_matches": feature_group_resolution.matched_features,
        "feature_group_missing_base_columns": (
            feature_group_resolution.missing_base_columns
        ),
        "unmatched_features": feature_group_resolution.unmatched_features,
        "feature_group_resolution_by_scope": feature_group_resolution_by_scope,
        "fallback_sample_counts": diagnostics,
    }
    table_outputs = write_tabular_outputs(
        monthly=monthly,
        summary=summary,
        manifest={
            **manifest,
            "plot_outputs": {key: str(value) for key, value in plot_outputs.items()},
        },
        output_dir=output_dir,
    )
    return {**plot_outputs, **table_outputs}
