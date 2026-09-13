"""Minimal data contract shared by the Ethiopia aligned GeoRF stages."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
import json
from pathlib import Path

import numpy as np
import pandas as pd

from EthiopiaForecastingExperiment.prepare_horizon_aligned_data import (
    REFERENCE_DYNAMIC_FEATURES,
    REFERENCE_STATIC_FEATURES,
    SEASON_FEATURES,
)


KEY = "FEWSNET_admin_code"
TARGET = "fews_ipc_crisis"
METADATA_COLUMNS = (
    "scope",
    "horizon_months",
    KEY,
    "target_month",
    "forecast_origin_month",
    TARGET,
)
ALIGNED_PREDICTORS = (
    *REFERENCE_STATIC_FEATURES,
    *REFERENCE_DYNAMIC_FEATURES,
    *SEASON_FEATURES,
)
RELEASE_FEATURES = (
    "fews_ipc_release_lag1",
    "fews_ipc_release_lag2",
    "fews_ipc_release_lag3",
)
MODEL_PREDICTORS = (*ALIGNED_PREDICTORS, *RELEASE_FEATURES)
SCOPE_HORIZONS = {0: 1, 1: 4, 2: 8, 3: 12}


def _admin_codes(values: Iterable[object]) -> pd.Series:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="raise")
    if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("FEWSNET admin codes must be non-null integers")
    return numeric.astype("int64")


def qualify_release_months(
    panel: pd.DataFrame,
    cohort_codes: Iterable[object],
    *,
    threshold: float = 0.9,
) -> pd.DataFrame:
    """Audit FEWS IPC release months against the frozen national cohort."""
    if not 0 < threshold <= 1:
        raise ValueError("Release coverage threshold must be in (0, 1]")
    required = {KEY, "date", "fews_ipc"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Working panel missing release columns: {sorted(missing)}")

    cohort = set(_admin_codes(cohort_codes).tolist())
    if not cohort:
        raise ValueError("Frozen cohort is empty")
    frame = panel[[KEY, "date", "fews_ipc"]].copy()
    frame[KEY] = _admin_codes(frame[KEY]).to_numpy()
    frame["release_month"] = pd.to_datetime(frame.pop("date"), errors="raise").dt.to_period("M").dt.to_timestamp()
    frame = frame.loc[frame[KEY].isin(cohort)]
    if frame.duplicated([KEY, "release_month"]).any():
        raise ValueError("Duplicate working-panel admin-month keys")

    observed = (
        frame.assign(_observed=frame["fews_ipc"].notna())
        .groupby("release_month", sort=True)["_observed"]
        .sum()
        .astype(int)
    )
    audit = observed.rename("observed_n").reset_index()
    audit["cohort_n"] = len(cohort)
    audit["coverage"] = audit["observed_n"] / audit["cohort_n"]
    audit["qualifying"] = audit["coverage"].ge(threshold)
    return audit


def attach_release_history(
    aligned: pd.DataFrame,
    panel: pd.DataFrame,
    release_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Attach phases from the latest three qualifying releases before origin."""
    required_audit = {"release_month", "qualifying"}
    if not required_audit.issubset(release_audit.columns):
        raise ValueError("Release audit is missing required columns")
    qualifying = np.sort(
        pd.to_datetime(
            release_audit.loc[release_audit["qualifying"], "release_month"],
            errors="raise",
        )
        .drop_duplicates()
        .to_numpy(dtype="datetime64[ns]")
    )
    if qualifying.size == 0:
        raise ValueError("No qualifying FEWS IPC release months")

    output = aligned.copy()
    output[KEY] = _admin_codes(output[KEY]).to_numpy()
    origins = pd.to_datetime(output["forecast_origin_month"], errors="raise").dt.to_period("M").dt.to_timestamp()
    phase = panel[[KEY, "date", "fews_ipc"]].copy()
    phase[KEY] = _admin_codes(phase[KEY]).to_numpy()
    phase["release_month"] = pd.to_datetime(phase.pop("date"), errors="raise").dt.to_period("M").dt.to_timestamp()
    if phase.duplicated([KEY, "release_month"]).any():
        raise ValueError("Duplicate working-panel admin-month keys")
    values = pd.to_numeric(phase["fews_ipc"], errors="coerce")
    invalid = values.notna() & ~values.isin([1, 2, 3, 4])
    if invalid.any():
        raise ValueError("Observed fews_ipc phases must be in 1..4")
    phase["fews_ipc"] = values

    positions = np.searchsorted(qualifying, origins.to_numpy(dtype="datetime64[ns]"), side="left")
    output["_row_order"] = np.arange(len(output))
    for offset, feature in enumerate(RELEASE_FEATURES, start=1):
        selected = np.full(len(output), np.datetime64("NaT"), dtype="datetime64[ns]")
        available = positions >= offset
        selected[available] = qualifying[positions[available] - offset]
        release_column = f"_{feature}_month"
        output[release_column] = selected
        source = phase.rename(
            columns={"release_month": release_column, "fews_ipc": feature}
        )[[KEY, release_column, feature]]
        output = output.merge(
            source,
            on=[KEY, release_column],
            how="left",
            sort=False,
            validate="many_to_one",
        ).drop(columns=release_column)
    return output.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)


def fit_fold_medians(train: np.ndarray) -> np.ndarray:
    """Fit column medians on training rows, using zero for all-null columns."""
    values = np.asarray(train, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Training features must be a non-empty 2D matrix")
    if np.isinf(values).any():
        raise ValueError("Training features contain infinite values")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(values, axis=0)
    return np.where(np.isnan(medians), 0.0, medians)


def apply_fold_medians(values: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Apply frozen training medians and require finite model input."""
    matrix = np.asarray(values, dtype=float)
    statistics = np.asarray(medians, dtype=float)
    if matrix.ndim != 2 or statistics.ndim != 1 or matrix.shape[1] != len(statistics):
        raise ValueError("Feature matrix and imputation statistics do not align")
    if np.isinf(matrix).any() or not np.isfinite(statistics).all():
        raise ValueError("Imputation input contains infinite values")
    transformed = np.where(np.isnan(matrix), statistics, matrix)
    if not np.isfinite(transformed).all():
        raise ValueError("Imputation did not produce finite model input")
    return transformed


def select_rolling_fold(
    frame: pd.DataFrame,
    *,
    target_month: str,
    horizon: int,
    window_months: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed-training and complete-test row indices for one exact fold."""
    if horizon <= 0 or window_months <= 0:
        raise ValueError("Horizon and training window must be positive")
    dates = pd.to_datetime(frame["target_month"], errors="raise").dt.to_period("M").dt.to_timestamp()
    target = pd.Period(target_month, freq="M").to_timestamp()
    train_end = target - pd.DateOffset(months=horizon)
    train_start = train_end - pd.DateOffset(months=window_months)
    train_mask = dates.ge(train_start) & dates.lt(train_end) & frame[TARGET].notna()
    test_mask = dates.eq(target)
    train = np.flatnonzero(train_mask.to_numpy())
    test = np.flatnonzero(test_mask.to_numpy())
    if train.size == 0 or test.size == 0:
        raise ValueError(f"Empty rolling fold for target month {target_month}")
    return train, test


def split_latest_months(
    dates: Sequence[object],
    *,
    validation_months: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows so the latest eligible calendar months form validation."""
    if validation_months <= 0:
        raise ValueError("Validation months must be positive")
    periods = pd.to_datetime(pd.Series(dates), errors="raise").dt.to_period("M")
    unique = np.sort(periods.unique())
    if len(unique) <= validation_months:
        raise ValueError("Training fold has too few months for validation")
    validation_periods = set(unique[-validation_months:])
    validation = periods.isin(validation_periods).to_numpy()
    return ~validation, validation


def validate_aligned_frame(
    frame: pd.DataFrame,
    scope: str,
    horizon: int,
    predictor_columns: Sequence[str],
) -> None:
    """Validate the exact scope, calendar origin, key, and predictor order."""
    expected = [*METADATA_COLUMNS, *predictor_columns]
    if list(frame.columns) != expected:
        raise ValueError("Aligned frame predictor order does not match the contract")
    if not frame["scope"].eq(scope).all() or not frame["horizon_months"].eq(horizon).all():
        raise ValueError("Aligned frame scope or horizon does not match the contract")
    if frame[KEY].isna().any() or frame.duplicated([KEY, "target_month"]).any():
        raise ValueError("Aligned frame has null or duplicate keys")
    target = pd.to_datetime(frame["target_month"], errors="raise")
    origin = pd.to_datetime(frame["forecast_origin_month"], errors="raise")
    gap = (target.dt.year - origin.dt.year) * 12 + target.dt.month - origin.dt.month
    if not gap.eq(horizon).all():
        raise ValueError("Aligned frame does not use the exact forecast origin")


def build_run_local_inputs(
    aligned_paths: dict[int, Path],
    working_panel_path: Path,
    season_lookup_path: Path,
    output_dir: Path,
    *,
    expected_admins: int = 1_040,
) -> tuple[dict[int, Path], dict[str, object]]:
    """Write the four validated 88-predictor snapshots used by both ETH stages."""
    if set(aligned_paths) != set(SCOPE_HORIZONS):
        raise ValueError("Exactly one aligned input is required for each fs0-fs3 scope")

    panel = pd.read_csv(working_panel_path, low_memory=False)
    panel_codes = set(_admin_codes(panel[KEY]).tolist())
    if len(panel_codes) != expected_admins:
        raise ValueError(f"Working-panel cohort has {len(panel_codes)} admins, expected {expected_admins}")
    release_audit = qualify_release_months(panel, panel_codes)

    season = pd.read_csv(season_lookup_path, low_memory=False)
    required_season = {KEY, "year", "month", "previous_season_end", *SEASON_FEATURES}
    missing_season = required_season.difference(season.columns)
    if missing_season:
        raise ValueError(f"Season lookup missing columns: {sorted(missing_season)}")
    season[KEY] = _admin_codes(season[KEY]).to_numpy()
    season["forecast_origin_month"] = pd.to_datetime(
        {"year": season["year"], "month": season["month"], "day": 1},
        errors="raise",
    )
    if season.duplicated([KEY, "forecast_origin_month"]).any():
        raise ValueError("Duplicate season lookup admin-origin keys")

    qualifying = np.sort(
        pd.to_datetime(
            release_audit.loc[release_audit["qualifying"], "release_month"],
            errors="raise",
        ).to_numpy(dtype="datetime64[ns]")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    release_audit.to_csv(output_dir / "release_coverage.csv", index=False)
    feature_manifest = {
        "aligned_predictor_count": len(ALIGNED_PREDICTORS),
        "predictor_count": len(MODEL_PREDICTORS),
        "predictors": list(MODEL_PREDICTORS),
        "release_predictors": list(RELEASE_FEATURES),
    }
    (output_dir / "feature_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    snapshots: dict[int, Path] = {}
    scope_audit: dict[str, object] = {}
    frozen_cohort: set[int] | None = None
    for scope, horizon in SCOPE_HORIZONS.items():
        frame = pd.read_csv(aligned_paths[scope], low_memory=False)
        for column in ("target_month", "forecast_origin_month"):
            frame[column] = pd.to_datetime(frame[column], errors="raise").dt.to_period("M").dt.to_timestamp()
        validate_aligned_frame(frame, f"fs{scope}", horizon, ALIGNED_PREDICTORS)
        frame[KEY] = _admin_codes(frame[KEY]).to_numpy()
        cohort = set(frame[KEY].unique())
        if len(cohort) != expected_admins or cohort != panel_codes:
            raise ValueError(f"fs{scope} aligned cohort does not match the working panel")
        if frozen_cohort is not None and cohort != frozen_cohort:
            raise ValueError("Aligned cohorts differ across scopes")
        frozen_cohort = cohort

        seasonal_check = frame[[KEY, "forecast_origin_month", *SEASON_FEATURES]].merge(
            season[[KEY, "forecast_origin_month", "previous_season_end", *SEASON_FEATURES]],
            on=[KEY, "forecast_origin_month"],
            how="left",
            validate="many_to_one",
            suffixes=("", "_lookup"),
        )
        modeling = pd.to_datetime(frame["target_month"]).dt.year.between(2018, 2024)
        season_end = pd.to_datetime(seasonal_check["previous_season_end"], errors="coerce")
        origins = pd.to_datetime(seasonal_check["forecast_origin_month"], errors="raise")
        if season_end.loc[modeling].isna().any() or not season_end.loc[modeling].lt(origins.loc[modeling]).all():
            raise ValueError(f"fs{scope} previous growing season does not end before forecast origin")
        for feature in SEASON_FEATURES:
            left = pd.to_numeric(seasonal_check[feature], errors="coerce").to_numpy()
            right = pd.to_numeric(seasonal_check[f"{feature}_lookup"], errors="coerce").to_numpy()
            if not np.allclose(left, right, equal_nan=True):
                raise ValueError(f"fs{scope} seasonal feature mismatch: {feature}")

        positions = np.searchsorted(
            qualifying,
            pd.to_datetime(frame["forecast_origin_month"]).to_numpy(dtype="datetime64[ns]"),
            side="left",
        )
        if (positions[modeling.to_numpy()] < len(RELEASE_FEATURES)).any():
            raise ValueError(f"fs{scope} has fewer than three qualifying releases before a modeling origin")
        snapshot = attach_release_history(frame, panel, release_audit)
        snapshot = snapshot[[*METADATA_COLUMNS, *MODEL_PREDICTORS]]
        validate_aligned_frame(snapshot, f"fs{scope}", horizon, MODEL_PREDICTORS)
        path = output_dir / f"ethiopia_panel_fs{scope}_88.csv"
        snapshot.to_csv(path, index=False)
        snapshots[scope] = path
        scope_audit[f"fs{scope}"] = {
            "rows": len(snapshot),
            "admins": len(cohort),
            "release_missing": {
                feature: int(snapshot[feature].isna().sum()) for feature in RELEASE_FEATURES
            },
            "season_rows_checked": int(modeling.sum()),
        }

    return snapshots, {**feature_manifest, "scopes": scope_audit}
