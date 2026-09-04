"""Focused checks for Ethiopia previous-growing-season features."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

import prepare_horizon_aligned_data as aligner  # noqa: E402


KEY = "FEWSNET_admin_code"
MODEL_FEATURES = (
    "previous_season_avg_SPI_1",
    "previous_season_avg_SPI_3",
    "previous_season_avg_SPI_6",
    "previous_season_avg_SPI_12",
    "previous_season_avg_gpp_mean",
    "previous_season_avg_Tair_f_tavg_mean",
    "previous_season_sum_EVI",
)
OBSERVED_COLUMNS = (
    "observed_SPI_1_months",
    "observed_SPI_3_months",
    "observed_SPI_6_months",
    "observed_SPI_12_months",
    "observed_gpp_mean_months",
    "observed_Tair_f_tavg_mean_months",
    "observed_EVI_months",
)
LOOKUP_COLUMNS = (
    KEY,
    "year",
    "month",
    *MODEL_FEATURES,
    "calendar_group",
    "previous_season_start",
    "previous_season_end",
    "expected_months",
    *OBSERVED_COLUMNS,
)
LOOKUP_PATH = (
    EXPERIMENT_DIR
    / "data"
    / "interim"
    / "growing_season"
    / "ethiopia_previous_growing_season_monthly.csv"
)


class GrowingSeasonFeatureTests(unittest.TestCase):
    def test_fixed_lookup_contract(self) -> None:
        lookup = pd.read_csv(
            LOOKUP_PATH,
            parse_dates=["previous_season_start", "previous_season_end"],
            low_memory=False,
        )
        self.assertEqual(tuple(lookup.columns), LOOKUP_COLUMNS)
        self.assertEqual(len(lookup), 187_200)
        self.assertEqual(lookup[KEY].nunique(), 1_040)
        self.assertFalse(lookup.duplicated([KEY, "year", "month"]).any())
        self.assertEqual(lookup[["year", "month"]].drop_duplicates().shape[0], 180)
        self.assertEqual(
            lookup.groupby("calendar_group")[KEY].nunique().to_dict(),
            {
                "belg_meher_bimodal": 301,
                "meher_only": 650,
                "pastoral_bimodal": 89,
            },
        )

        lookup_month = pd.to_datetime(
            {"year": lookup["year"], "month": lookup["month"], "day": 1}
        )
        valid = lookup["previous_season_end"].notna()
        self.assertTrue(
            (
                lookup.loc[valid, "previous_season_end"]
                < lookup_month.loc[valid]
            ).all()
        )
        self.assertTrue(
            (
                lookup.loc[valid, "previous_season_start"]
                <= lookup.loc[valid, "previous_season_end"]
            ).all()
        )

        evaluation = lookup.loc[lookup["year"].between(2018, 2024)]
        self.assertEqual(len(evaluation), 87_360)
        self.assertFalse(evaluation[list(MODEL_FEATURES)].isna().any().any())
        self.assertEqual(
            lookup[list(MODEL_FEATURES)].isna().all(axis=1).sum(), 10_352
        )

        minimum_mean_months = lookup["expected_months"].map(
            lambda value: math.ceil(2 * value / 3) if pd.notna(value) else np.nan
        )
        for feature, observed in zip(
            MODEL_FEATURES[:-1], OBSERVED_COLUMNS[:-1], strict=True
        ):
            published = lookup[feature].notna()
            self.assertTrue(
                lookup.loc[published, observed]
                .ge(minimum_mean_months.loc[published])
                .all()
            )
        published_evi = lookup[MODEL_FEATURES[-1]].notna()
        self.assertTrue(
            lookup.loc[published_evi, OBSERVED_COLUMNS[-1]]
            .eq(lookup.loc[published_evi, "expected_months"])
            .all()
        )

    def test_all_horizons_join_at_exact_forecast_origin(self) -> None:
        dates = pd.date_range("2019-01-01", "2020-12-01", freq="MS")
        panel = pd.DataFrame(
            {
                "ISO3": "ETH",
                KEY: 1,
                "date": dates,
                "fews_ipc_crisis": 1,
            }
        )
        lookup = pd.DataFrame(
            {
                KEY: 1,
                "year": dates.year,
                "month": dates.month,
                "calendar_group": "meher_only",
            }
        )
        for offset, feature in enumerate(MODEL_FEATURES):
            lookup[feature] = np.arange(len(dates), dtype=float) + offset * 100

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            panel_path = root / "panel.csv"
            lookup_path = root / "lookup.csv"
            output_dir = root / "aligned"
            panel.to_csv(panel_path, index=False)
            lookup.to_csv(lookup_path, index=False)
            with mock.patch.object(aligner, "DEFAULT_SEASON_LOOKUP", lookup_path):
                outputs = aligner.build_aligned_panels(
                    panel_path,
                    output_dir,
                    static_columns=(),
                    dynamic_columns=(),
                )

            lookup_expected = lookup.drop(columns="calendar_group").copy()
            lookup_expected["forecast_origin_month"] = pd.to_datetime(
                {
                    "year": lookup_expected.pop("year"),
                    "month": lookup_expected.pop("month"),
                    "day": 1,
                }
            )
            for scope, horizon in aligner.HORIZONS.items():
                aligned = pd.read_csv(
                    outputs[scope][0],
                    parse_dates=["target_month", "forecast_origin_month"],
                )
                self.assertTrue(
                    aligned["forecast_origin_month"].eq(
                        aligned["target_month"] - pd.DateOffset(months=horizon)
                    ).all()
                )
                expected = aligned[[KEY, "forecast_origin_month"]].merge(
                    lookup_expected,
                    on=[KEY, "forecast_origin_month"],
                    validate="many_to_one",
                )
                for feature in MODEL_FEATURES:
                    np.testing.assert_allclose(aligned[feature], expected[feature])
                self.assertNotIn("calendar_group", aligned.columns)


if __name__ == "__main__":
    unittest.main()
