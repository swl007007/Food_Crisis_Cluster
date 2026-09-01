"""Rebuild the Ethiopia working panel with approved pre-alignment features."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE = (
    EXPERIMENT_DIR
    / "outputs"
    / "baseline_audit"
    / "fewsnet_eth_pre_georf_20260831"
    / "fewsnet_eth_pre_georf.csv.gz"
)
DEFAULT_OUTPUT = EXPERIMENT_DIR / "data" / "working" / "ethiopia_panel.csv"

KEY = "FEWSNET_admin_code"
DATE = "date"
DROP_COLUMNS = ("Rainf_zscore", "FAO_price", "market_distance")
MARKET_GEO_ID = "WB_RTFP_market_geo_id"
MARKET_NAME = "WB_RTFP_market_name"
MARKET_DISTANCE = "WB_RTFP_market_distance_km"
PRICE_INDEX = "WB_RTFP_price_index"
PRICE_LAG1 = "WB_RTFP_price_index_lag1"
PRICE_MA4 = "WB_RTFP_price_index_MA4"
GPP = "gpp_mean"
GPP_MA_WINDOWS = (1, 3, 6, 12)
GPP_MA_COLUMNS = tuple(f"{GPP}_MA{window}" for window in GPP_MA_WINDOWS)
TEMPERATURE_ZSCORE = "Tair_zscore"
TEMPERATURE_MA_WINDOWS = (3, 6, 12)
TEMPERATURE_MA_COLUMNS = tuple(
    f"{TEMPERATURE_ZSCORE}_MA{window}" for window in TEMPERATURE_MA_WINDOWS
)
DERIVED_MA_COLUMNS = (*GPP_MA_COLUMNS, *TEMPERATURE_MA_COLUMNS)
WB_APPENDED_COLUMNS = (
    MARKET_GEO_ID,
    MARKET_NAME,
    MARKET_DISTANCE,
    PRICE_INDEX,
    PRICE_LAG1,
    PRICE_MA4,
)
APPENDED_COLUMNS = (*WB_APPENDED_COLUMNS, *DERIVED_MA_COLUMNS)
EARTH_RADIUS_KM = 6371.0088


def load_baseline_panel(path: Path) -> pd.DataFrame:
    """Load and validate the frozen Ethiopia baseline subset."""
    panel = pd.read_csv(path, low_memory=False)
    missing = {
        "ISO3",
        KEY,
        DATE,
        "lat",
        "lon",
        GPP,
        TEMPERATURE_ZSCORE,
        *DROP_COLUMNS,
    }.difference(panel.columns)
    if missing:
        raise ValueError(f"Baseline panel is missing columns: {sorted(missing)}")
    if set(panel["ISO3"].dropna().unique()) != {"ETH"}:
        raise ValueError("Baseline panel is not an exact ISO3 == 'ETH' cohort")
    panel[DATE] = pd.to_datetime(panel[DATE], errors="raise")
    if panel.duplicated([KEY, DATE]).any():
        raise ValueError("Baseline panel has duplicate admin-month keys")
    if panel.groupby(KEY)[["lat", "lon"]].nunique(dropna=False).max().max() != 1:
        raise ValueError("Admin coordinates vary over time")
    return panel.sort_values([KEY, DATE], kind="stable").reset_index(drop=True)


def load_wb_markets(path: Path) -> pd.DataFrame:
    """Load Ethiopia entity markets and construct market-level price features."""
    columns = [
        "ISO3",
        "geo_id",
        "mkt_name",
        "lat",
        "lon",
        "DATES",
        "o_food_price_index",
        "c_food_price_index",
    ]
    markets = pd.read_csv(path, usecols=columns, low_memory=False)
    markets = markets.loc[
        markets["ISO3"].eq("ETH")
        & markets["geo_id"].ne("gid_eth_national_average")
        & markets["mkt_name"].ne("Market Average")
        & markets["lat"].notna()
        & markets["lon"].notna()
    ].copy()
    markets[DATE] = pd.to_datetime(markets.pop("DATES"), errors="raise")
    if markets.duplicated(["geo_id", DATE]).any():
        raise ValueError("WB RTFP has duplicate Ethiopia market-month keys")
    if markets.groupby("geo_id")[["lat", "lon", "mkt_name"]].nunique(
        dropna=False
    ).max().max() != 1:
        raise ValueError("WB RTFP market identity or coordinates vary over time")

    markets = markets.sort_values(["geo_id", DATE], kind="stable").reset_index(
        drop=True
    )
    expected_months = pd.date_range(markets[DATE].min(), markets[DATE].max(), freq="MS")
    if not markets.groupby("geo_id")[DATE].apply(
        lambda values: pd.Index(values).equals(expected_months)
    ).all():
        raise ValueError("WB RTFP Ethiopia markets do not share a complete monthly grid")

    markets[PRICE_INDEX] = (
        markets["o_food_price_index"] + markets["c_food_price_index"]
    ) / 2
    grouped = markets.groupby("geo_id", sort=False)[PRICE_INDEX]
    markets[PRICE_LAG1] = grouped.shift(1)
    markets[PRICE_MA4] = grouped.transform(
        lambda values: values.shift(1).rolling(4, min_periods=4).mean()
    )
    return markets


def nearest_market_mapping(
    panel: pd.DataFrame, markets: pd.DataFrame
) -> pd.DataFrame:
    """Map each admin centroid to its nearest Ethiopia entity market."""
    admins = panel[[KEY, "lat", "lon"]].drop_duplicates(KEY).sort_values(KEY)
    locations = (
        markets[["geo_id", "mkt_name", "lat", "lon"]]
        .drop_duplicates("geo_id")
        .sort_values("geo_id")
        .reset_index(drop=True)
    )

    admin_lat = np.radians(admins["lat"].to_numpy())[:, None]
    admin_lon = np.radians(admins["lon"].to_numpy())[:, None]
    market_lat = np.radians(locations["lat"].to_numpy())[None, :]
    market_lon = np.radians(locations["lon"].to_numpy())[None, :]
    delta_lat = market_lat - admin_lat
    delta_lon = market_lon - admin_lon
    haversine = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(admin_lat) * np.cos(market_lat) * np.sin(delta_lon / 2) ** 2
    )
    distances = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(haversine))
    nearest = distances.argmin(axis=1)

    mapping = admins[[KEY]].reset_index(drop=True)
    mapping[MARKET_GEO_ID] = locations.iloc[nearest]["geo_id"].to_numpy()
    mapping[MARKET_NAME] = locations.iloc[nearest]["mkt_name"].to_numpy()
    mapping[MARKET_DISTANCE] = distances[np.arange(len(mapping)), nearest]
    return mapping


def build_working_panel(baseline_path: Path, wb_path: Path) -> pd.DataFrame:
    """Drop retired columns and append approved pre-alignment features."""
    raw = load_baseline_panel(baseline_path)
    markets = load_wb_markets(wb_path)
    mapping = nearest_market_mapping(raw, markets)
    market_features = markets[
        ["geo_id", DATE, PRICE_INDEX, PRICE_LAG1, PRICE_MA4]
    ]

    base_columns = [column for column in raw.columns if column not in DROP_COLUMNS]
    working = raw[base_columns].copy()
    grouped_gpp = working.groupby(KEY, sort=False)[GPP]
    for window, column in zip(GPP_MA_WINDOWS, GPP_MA_COLUMNS):
        working[column] = grouped_gpp.transform(
            lambda values, window=window: values.rolling(
                window, min_periods=window
            ).mean()
        )
    grouped_temperature = working[TEMPERATURE_ZSCORE].replace(
        [np.inf, -np.inf], np.nan
    ).groupby(working[KEY], sort=False)
    for window, column in zip(TEMPERATURE_MA_WINDOWS, TEMPERATURE_MA_COLUMNS):
        working[column] = grouped_temperature.transform(
            lambda values, window=window: values.rolling(
                window, min_periods=window
            ).mean()
        )

    working = working.merge(mapping, on=KEY, how="left", validate="many_to_one")
    working = working.merge(
        market_features,
        left_on=[MARKET_GEO_ID, DATE],
        right_on=["geo_id", DATE],
        how="left",
        validate="many_to_one",
    ).drop(columns="geo_id")

    if len(working) != len(raw):
        raise ValueError("WB RTFP merge changed the panel row count")
    if working.duplicated([KEY, DATE]).any():
        raise ValueError("WB RTFP merge created duplicate admin-month keys")
    if working[list(WB_APPENDED_COLUMNS)].isna().any().any():
        missing = working[list(WB_APPENDED_COLUMNS)].isna().sum()
        raise ValueError(f"WB RTFP merge left missing values: {missing[missing > 0].to_dict()}")
    if not np.allclose(
        working[GPP_MA_COLUMNS[0]], working[GPP], equal_nan=True
    ):
        raise ValueError("GPP MA1 does not equal contemporaneous gpp_mean")
    if np.isinf(working[list(TEMPERATURE_MA_COLUMNS)].to_numpy()).any():
        raise ValueError("Temperature moving averages contain infinite values")

    pd.testing.assert_frame_equal(
        working[base_columns], raw[base_columns], check_dtype=False
    )
    return working


def write_and_verify(frame: pd.DataFrame, output_path: Path, overwrite: bool) -> None:
    """Atomically replace the working panel after a focused round-trip check."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.unlink(missing_ok=True)
    try:
        frame.to_csv(temp_path, index=False, date_format="%Y-%m-%d")
        written = pd.read_csv(temp_path, usecols=[KEY, DATE, *APPENDED_COLUMNS])
        if len(written) != len(frame) or written.duplicated([KEY, DATE]).any():
            raise ValueError("Written working panel failed key validation")
        if written[list(WB_APPENDED_COLUMNS)].isna().any().any():
            raise ValueError("Written working panel lost WB RTFP values")
        if not np.allclose(
            written[list(DERIVED_MA_COLUMNS)],
            frame[list(DERIVED_MA_COLUMNS)],
            equal_nan=True,
        ):
            raise ValueError("Written working panel changed derived moving averages")
        os.replace(temp_path, output_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--wb-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Rebuild the Ethiopia working panel."""
    args = parse_args()
    working = build_working_panel(args.baseline.resolve(), args.wb_source.resolve())
    write_and_verify(working, args.output.resolve(), args.overwrite)
    distances = working[MARKET_DISTANCE]
    print(f"rows={len(working)}, columns={len(working.columns)}")
    print(
        f"nearest_market_distance_km: median={distances.median():.3f}, "
        f"p95={distances.quantile(0.95):.3f}, max={distances.max():.3f}"
    )
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
