"""Build the fixed Ethiopia previous-growing-season monthly lookup."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from era5_drought_spi import file_sha256


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE = (
    EXPERIMENT_DIR
    / "outputs"
    / "baseline_audit"
    / "fewsnet_eth_pre_georf_20260831"
    / "fewsnet_eth_pre_georf.csv.gz"
)
DEFAULT_SPI = (
    EXPERIMENT_DIR
    / "data"
    / "interim"
    / "era5_drought_spi"
    / "ethiopia_spi_monthly.csv"
)
DEFAULT_OUTPUT = (
    EXPERIMENT_DIR
    / "data"
    / "interim"
    / "growing_season"
    / "ethiopia_previous_growing_season_monthly.csv"
)

BASELINE_SHA256 = "25e458ac6fbdb27c9b264ada111bfe8fc38c6f2c376bd023dc8fb1c4d15eb855"
SPI_SHA256 = "d18e311aba521c680975a39ea8193c3b8f29d47f4ed5bb18fd0a6a0d6e40b547"
KEY = "FEWSNET_admin_code"
DATE = "date"
START_MONTH = pd.Timestamp("2010-01-01")
END_MONTH = pd.Timestamp("2024-12-01")

MODEL_FEATURES = (
    "previous_season_avg_SPI_1",
    "previous_season_avg_SPI_3",
    "previous_season_avg_SPI_6",
    "previous_season_avg_SPI_12",
    "previous_season_avg_gpp_mean",
    "previous_season_avg_Tair_f_tavg_mean",
    "previous_season_sum_EVI",
)
MEAN_FEATURES = {
    "SPI_1": "previous_season_avg_SPI_1",
    "SPI_3": "previous_season_avg_SPI_3",
    "SPI_6": "previous_season_avg_SPI_6",
    "SPI_12": "previous_season_avg_SPI_12",
    "gpp_mean": "previous_season_avg_gpp_mean",
    "Tair_f_tavg_mean": "previous_season_avg_Tair_f_tavg_mean",
}
OBSERVED_COLUMNS = {
    source: f"observed_{source}_months"
    for source in (*MEAN_FEATURES, "EVI")
}
SEASONS = {
    "meher_only": (("meher", 6, 12),),
    "belg_meher_bimodal": (("belg", 2, 7), ("meher", 6, 12)),
    "pastoral_bimodal": (("gu_genna", 3, 5), ("deyr_hageya", 10, 12)),
}
PASTORAL_ADMIN2 = frozenset(
    {
        ("Oromia", "Borena"),
        ("Oromia", "Guji"),
        ("Oromia", "West Guji"),
        ("Somali", "Afder"),
        ("Somali", "Daawa"),
        ("Somali", "Doolo"),
        ("Somali", "Korahe"),
        ("Somali", "Liban"),
        ("Somali", "Shabelle"),
    }
)
BELG_MEHER_ADMIN2 = frozenset(
    {
        ("Tigray", "Southern"),
        ("Tigray", "South Eastern"),
        ("Amhara", "North Wello"),
        ("Amhara", "South Wello"),
        ("Amhara", "North Shewa (AM)"),
        ("Amhara", "Oromia"),
        ("Oromia", "East Hararge"),
        ("Oromia", "West Hararge"),
        ("Oromia", "East Shewa"),
        ("Oromia", "Arsi"),
        ("Oromia", "Bale"),
        ("Oromia", "East Bale"),
        ("SNNP", "Guraghe"),
        ("SNNP", "Hadiya"),
        ("SNNP", "Halaba"),
        ("SNNP", "Kembata Tibaro"),
        ("SNNP", "Siltie"),
        ("SNNP", "Yem Special"),
    }
)
AUDIT_COLUMNS = (
    "calendar_group",
    "previous_season_start",
    "previous_season_end",
    "expected_months",
    *OBSERVED_COLUMNS.values(),
)
OUTPUT_COLUMNS = (KEY, "year", "month", *MODEL_FEATURES, *AUDIT_COLUMNS)


def main() -> None:
    """Validate frozen inputs, build the lookup, and write the fixed artifact."""
    hashes = {
        "baseline": (DEFAULT_BASELINE, BASELINE_SHA256),
        "SPI": (DEFAULT_SPI, SPI_SHA256),
    }
    for label, (path, expected_hash) in hashes.items():
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{label} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
            )

    baseline_columns = [
        KEY,
        DATE,
        "ISO3",
        "ADMIN1",
        "ADMIN2",
        "gpp_mean",
        "Tair_f_tavg_mean",
        "EVI",
    ]
    spi_columns = [KEY, DATE, "SPI_1", "SPI_3", "SPI_6", "SPI_12"]
    baseline = pd.read_csv(
        DEFAULT_BASELINE, usecols=baseline_columns, low_memory=False
    )
    spi = pd.read_csv(DEFAULT_SPI, usecols=spi_columns, low_memory=False)

    if set(baseline["ISO3"].dropna().unique()) != {"ETH"}:
        raise ValueError("Baseline is not an exact ISO3 == 'ETH' cohort")
    for label, frame in (("baseline", baseline), ("SPI", spi)):
        frame[DATE] = pd.to_datetime(frame[DATE], errors="raise")
        if not frame[DATE].dt.day.eq(1).all():
            raise ValueError(f"{label} dates must be calendar month starts")
        if frame.duplicated([KEY, DATE]).any():
            raise ValueError(f"{label} has duplicate admin-month keys")
        if frame[DATE].min() != START_MONTH or frame[DATE].max() != END_MONTH:
            raise ValueError(f"{label} must cover exactly 2010-01 through 2024-12")

    baseline_keys = pd.MultiIndex.from_frame(
        baseline[[KEY, DATE]].sort_values([KEY, DATE], kind="stable")
    )
    spi_keys = pd.MultiIndex.from_frame(
        spi[[KEY, DATE]].sort_values([KEY, DATE], kind="stable")
    )
    if not baseline_keys.equals(spi_keys):
        raise ValueError("Baseline and SPI admin-month key sets differ")

    admins = baseline[[KEY, "ADMIN1", "ADMIN2"]].drop_duplicates()
    if admins[[KEY, "ADMIN1", "ADMIN2"]].isna().any().any():
        raise ValueError("Admin identity fields cannot be null")
    if admins[KEY].duplicated().any() or len(admins) != 1_040:
        raise ValueError("Each frozen admin must have one ADMIN1/ADMIN2 assignment")

    pairs = list(zip(admins["ADMIN1"], admins["ADMIN2"]))
    admins["calendar_group"] = np.select(
        [
            [pair in PASTORAL_ADMIN2 for pair in pairs],
            [
                pair in BELG_MEHER_ADMIN2 or pair[0] == "Sidama"
                for pair in pairs
            ],
        ],
        ["pastoral_bimodal", "belg_meher_bimodal"],
        default="meher_only",
    )
    group_counts = admins["calendar_group"].value_counts().to_dict()
    expected_group_counts = {
        "meher_only": 650,
        "belg_meher_bimodal": 301,
        "pastoral_bimodal": 89,
    }
    if group_counts != expected_group_counts:
        raise ValueError(
            f"Calendar group counts changed: expected {expected_group_counts}, "
            f"got {group_counts}"
        )

    monthly = baseline.drop(columns="ISO3").merge(
        spi, on=[KEY, DATE], how="inner", validate="one_to_one"
    )
    monthly = monthly.merge(
        admins[[KEY, "calendar_group"]], on=KEY, how="left", validate="many_to_one"
    )

    season_frames: list[pd.DataFrame] = []
    source_columns = [*MEAN_FEATURES, "EVI"]
    for calendar_group, seasons in SEASONS.items():
        group_monthly = monthly.loc[
            monthly["calendar_group"].eq(calendar_group)
        ]
        admin_count = group_monthly[KEY].nunique()
        for _season_name, start_month, end_month in seasons:
            expected_months = end_month - start_month + 1
            minimum_mean_months = math.ceil(2 * expected_months / 3)
            for year in range(2010, 2025):
                window = group_monthly.loc[
                    group_monthly[DATE].dt.year.eq(year)
                    & group_monthly[DATE].dt.month.between(
                        start_month, end_month
                    )
                ]
                if len(window) != admin_count * expected_months:
                    raise ValueError(
                        f"Incomplete monthly keys for {calendar_group} in {year}"
                    )
                grouped = window.groupby(KEY, sort=False, observed=True)
                observed = grouped[source_columns].count()
                means = grouped[list(MEAN_FEATURES)].mean().rename(
                    columns=MEAN_FEATURES
                )
                for source, output in MEAN_FEATURES.items():
                    means[output] = means[output].where(
                        observed[source].ge(minimum_mean_months)
                    )
                means["previous_season_sum_EVI"] = grouped["EVI"].sum().where(
                    observed["EVI"].eq(expected_months)
                )
                means["previous_season_start"] = pd.Timestamp(
                    year=year, month=start_month, day=1
                ).as_unit("ns")
                means["previous_season_end"] = pd.Timestamp(
                    year=year, month=end_month, day=1
                ).as_unit("ns")
                means["expected_months"] = expected_months
                for source, output in OBSERVED_COLUMNS.items():
                    means[output] = observed[source]
                season_frames.append(means.reset_index())

    seasonal = pd.concat(season_frames, ignore_index=True)
    if seasonal.duplicated([KEY, "previous_season_end"]).any():
        raise ValueError("Duplicate completed seasons found for an admin")

    lookup_months = pd.DataFrame(
        {"lookup_month": pd.date_range(START_MONTH, END_MONTH, freq="MS")}
    )
    lookup = admins[[KEY, "calendar_group"]].merge(
        lookup_months, how="cross", sort=False
    )
    lookup = pd.merge_asof(
        lookup.sort_values(["lookup_month", KEY], kind="stable"),
        seasonal.drop(columns="calendar_group", errors="ignore").sort_values(
            ["previous_season_end", KEY], kind="stable"
        ),
        left_on="lookup_month",
        right_on="previous_season_end",
        by=KEY,
        direction="backward",
        allow_exact_matches=False,
    )
    lookup["year"] = lookup["lookup_month"].dt.year
    lookup["month"] = lookup["lookup_month"].dt.month
    lookup = lookup.sort_values([KEY, "lookup_month"], kind="stable").reset_index(
        drop=True
    )
    lookup = lookup[list(OUTPUT_COLUMNS)]

    if len(lookup) != 187_200 or lookup.duplicated([KEY, "year", "month"]).any():
        raise ValueError("Lookup must contain 187,200 unique admin-month rows")
    if lookup[["year", "month"]].drop_duplicates().shape[0] != 180:
        raise ValueError("Lookup must contain exactly 180 months")
    lookup_month = pd.to_datetime(
        {"year": lookup["year"], "month": lookup["month"], "day": 1}
    )
    has_season = lookup["previous_season_end"].notna()
    if not (
        lookup.loc[has_season, "previous_season_end"]
        < lookup_month.loc[has_season]
    ).all():
        raise ValueError("A lookup row uses a season that had not yet ended")
    evaluation = lookup.loc[lookup["year"].between(2018, 2024)]
    if len(evaluation) != 87_360 or evaluation[list(MODEL_FEATURES)].isna().any().any():
        raise ValueError("2018-2024 seasonal feature availability is incomplete")
    null_rows = lookup[list(MODEL_FEATURES)].isna().all(axis=1).sum()
    if null_rows != 10_352:
        raise ValueError(f"Expected 10,352 left-boundary null rows, got {null_rows}")

    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(DEFAULT_OUTPUT, index=False)
    print(
        f"Wrote {len(lookup):,} rows to {DEFAULT_OUTPUT}; "
        f"groups={group_counts}; sha256={file_sha256(DEFAULT_OUTPUT)}"
    )


if __name__ == "__main__":
    main()
