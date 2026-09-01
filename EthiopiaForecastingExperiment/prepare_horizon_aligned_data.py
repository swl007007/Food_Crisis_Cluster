"""Build Ethiopia-only, horizon-aligned panels without feature engineering."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EXPERIMENT_DIR / "data" / "working" / "ethiopia_panel.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "data" / "aligned"
HORIZONS = {"fs0": 1, "fs1": 4, "fs2": 8, "fs3": 12}

# Current reference set. Later approved variables can be added explicitly with
# --add-static-column or --add-dynamic-column.
REFERENCE_STATIC_FEATURES = (
    "lat",
    "lon",
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
    "crop",
    "range",
    "distance_to_river",
    "elevation",
    "sg_cec_5-15cm",
    "sg_cfvo_5-15cm",
    "sg_nitrogen_5-15cm",
    "sg_phh2o_5-15cm",
    "sg_soc_5-15cm",
    "market_access",
    "ruggedness",
    "slope",
)

REFERENCE_DYNAMIC_FEATURES = (
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
    "nightlight",
    "nightlight_sd",
    "EVI",
    "Rainf_f_tavg_mean",
    "Tair_f_tavg_mean",
    "gpp_mean",
    "gpp_mean_MA1",
    "gpp_mean_MA3",
    "gpp_mean_MA6",
    "gpp_mean_MA12",
    "Tair_zscore",
    "CPI",
    "GDP",
    "CC",
    "gini",
    "WFP_Price",
    "WFP_Price_std",
    "pop",
    "Food_CPI",
    "Food_food_inflation",
    "WB_RTFP_price_index",
    "WB_RTFP_price_index_lag1",
    "WB_RTFP_price_index_MA4",
)

KEY = "FEWSNET_admin_code"
DATE = "date"
TARGET = "fews_ipc_crisis"
# ponytail: this deny-list covers the current panel; extend it when a new
# protected metadata, outcome, or provider-projection field is introduced.
FORBIDDEN_PREDICTORS = {
    "unit_name",
    "ADMIN0",
    "ADMIN1",
    "ADMIN2",
    "ADMIN3",
    KEY,
    "ISO",
    "ISO3",
    DATE,
    "month",
    TARGET,
    "fews_ipc",
    "fews_ha",
    "fews_proj_near",
    "fews_proj_near_ha",
    "fews_proj_med",
    "fews_proj_med_ha",
    "fews_ipc_adjusted",
    "fews_proj_med_adjusted",
}


def unique_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """Return column names in first-seen order."""
    result: list[str] = []
    for column in columns:
        name = column.strip()
        if not name:
            raise ValueError("Feature column names cannot be empty")
        if name not in result:
            result.append(name)
    return tuple(result)


def load_and_validate_panel(
    input_path: Path,
    static_columns: Sequence[str],
    dynamic_columns: Sequence[str],
) -> pd.DataFrame:
    """Load required columns and validate the complete ETH monthly grid."""
    required = [
        "ISO3",
        KEY,
        DATE,
        TARGET,
        *static_columns,
        *dynamic_columns,
    ]
    header = pd.read_csv(input_path, nrows=0).columns
    missing = set(required).difference(header)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    panel = pd.read_csv(input_path, usecols=required, low_memory=False)
    if set(panel["ISO3"].dropna().unique()) != {"ETH"}:
        raise ValueError("Input panel is not an exact ISO3 == 'ETH' cohort")
    panel[DATE] = pd.to_datetime(panel[DATE], errors="raise")
    if not panel[DATE].dt.day.eq(1).all():
        raise ValueError("All dates must be calendar month starts")
    if panel.duplicated([KEY, DATE]).any():
        raise ValueError("Duplicate FEWSNET_admin_code/date keys found")

    panel = panel.sort_values([KEY, DATE], kind="stable").reset_index(drop=True)
    admins = pd.Index(panel[KEY].unique(), name=KEY)
    months = pd.date_range(panel[DATE].min(), panel[DATE].max(), freq="MS", name=DATE)
    expected = pd.MultiIndex.from_product([admins, months])
    actual = pd.MultiIndex.from_frame(panel[[KEY, DATE]])
    if len(actual) != len(expected) or not expected.difference(actual).empty:
        raise ValueError("Input panel is not a complete common admin-month grid")

    varying_static = panel.groupby(KEY, sort=False)[list(static_columns)].nunique(
        dropna=False
    ).max()
    varying_static = varying_static[varying_static > 1].index.tolist()
    if varying_static:
        raise ValueError(
            f"Declared static columns vary within admin: {varying_static}"
        )

    predictor_columns = [*static_columns, *dynamic_columns]
    panel[predictor_columns] = panel[predictor_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    labels = set(panel[TARGET].dropna().unique())
    if not labels.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Target is not binary: {sorted(labels)}")
    return panel


def align_horizon(
    panel: pd.DataFrame,
    scope: str,
    horizon: int,
    static_columns: Sequence[str],
    dynamic_columns: Sequence[str],
) -> pd.DataFrame:
    """Join target T to dynamic predictors from exact month T-H."""
    target_rows = panel[[KEY, DATE, TARGET, *static_columns]].rename(
        columns={DATE: "target_month"}
    )
    origin_rows = panel[[KEY, DATE, *dynamic_columns]].rename(
        columns={DATE: "forecast_origin_month"}
    )
    origin_rows["target_month"] = origin_rows["forecast_origin_month"] + pd.DateOffset(
        months=horizon
    )

    aligned = target_rows.merge(
        origin_rows,
        on=[KEY, "target_month"],
        how="inner",
        validate="one_to_one",
    )
    aligned = aligned.loc[aligned[TARGET].notna()].reset_index(drop=True)
    month_gap = (
        (aligned["target_month"].dt.year - aligned["forecast_origin_month"].dt.year)
        * 12
        + aligned["target_month"].dt.month
        - aligned["forecast_origin_month"].dt.month
    )
    if not month_gap.eq(horizon).all():
        raise ValueError(f"Calendar alignment failed for {scope}")
    if aligned.duplicated([KEY, "target_month"]).any():
        raise ValueError(f"Duplicate aligned keys found for {scope}")

    aligned.insert(0, "horizon_months", horizon)
    aligned.insert(0, "scope", scope)
    return aligned[
        [
            "scope",
            "horizon_months",
            KEY,
            "target_month",
            "forecast_origin_month",
            TARGET,
            *static_columns,
            *dynamic_columns,
        ]
    ]


def write_and_verify(frame: pd.DataFrame, output_path: Path) -> None:
    """Atomically write a CSV and verify its full round trip."""
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.unlink(missing_ok=True)
    try:
        frame.to_csv(temp_path, index=False)
        written = pd.read_csv(
            temp_path,
            parse_dates=["target_month", "forecast_origin_month"],
            low_memory=False,
        )
        pd.testing.assert_frame_equal(
            written,
            frame,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        os.replace(temp_path, output_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def build_aligned_panels(
    input_path: Path,
    output_dir: Path,
    static_columns: Sequence[str],
    dynamic_columns: Sequence[str],
    overwrite: bool = False,
) -> dict[str, tuple[Path, int]]:
    """Build and validate one aligned panel for each forecast horizon."""
    static_columns = unique_columns(static_columns)
    dynamic_columns = unique_columns(dynamic_columns)
    overlap = set(static_columns).intersection(dynamic_columns)
    if overlap:
        raise ValueError(f"Columns cannot be both static and dynamic: {sorted(overlap)}")
    forbidden = set((*static_columns, *dynamic_columns)).intersection(
        FORBIDDEN_PREDICTORS
    )
    if forbidden:
        raise ValueError(f"Forbidden predictor columns requested: {sorted(forbidden)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {scope: output_dir / f"ethiopia_panel_{scope}.csv" for scope in HORIZONS}
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            + ", ".join(str(path) for path in existing)
        )

    panel = load_and_validate_panel(input_path, static_columns, dynamic_columns)
    results: dict[str, tuple[Path, int]] = {}
    for scope, horizon in HORIZONS.items():
        aligned = align_horizon(
            panel, scope, horizon, static_columns, dynamic_columns
        )
        write_and_verify(aligned, paths[scope])
        results[scope] = (paths[scope], len(aligned))
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--add-static-column", action="append", default=[])
    parser.add_argument("--add-dynamic-column", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the horizon-only Ethiopia alignment."""
    args = parse_args()
    static_columns = (*REFERENCE_STATIC_FEATURES, *args.add_static_column)
    dynamic_columns = (*REFERENCE_DYNAMIC_FEATURES, *args.add_dynamic_column)
    outputs = build_aligned_panels(
        args.input.resolve(),
        args.output_dir.resolve(),
        static_columns,
        dynamic_columns,
        overwrite=args.overwrite,
    )
    print(
        f"Aligned {len(unique_columns(static_columns))} static and "
        f"{len(unique_columns(dynamic_columns))} dynamic reference features."
    )
    for scope, horizon in HORIZONS.items():
        path, rows = outputs[scope]
        print(f"{scope}: horizon={horizon}, rows={rows}, output={path}")


if __name__ == "__main__":
    main()
