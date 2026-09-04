"""Focused checks for Ethiopia MA12 conflict-intensity features."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from prepare_horizon_aligned_data import HORIZONS, align_horizon  # noqa: E402
from prepare_working_panel import (  # noqa: E402
    CONFLICT_EVENT_COLUMNS,
    CONFLICT_FATALITY_COLUMNS,
    rolling_conflict_intensity,
)


class ConflictIntensityTests(unittest.TestCase):
    def test_strict_ma12_and_exact_origin_alignment(self) -> None:
        dates = pd.date_range("2020-01-01", periods=13, freq="MS")
        panel = pd.DataFrame(
            {
                "FEWSNET_admin_code": ["a"] * 13 + ["b"] * 13,
                "date": list(dates) * 2,
                "event_count_battles": list(range(1, 14)) + [100] * 13,
                "event_count_explosions": [0] * 26,
                "event_count_violence": [0] * 26,
                "sum_fatalities_battles": [0] * 26,
                "sum_fatalities_explosions": [0] * 26,
                "sum_fatalities_violence": [1] * 26,
                "fews_ipc_crisis": [1] * 26,
            }
        )
        event = rolling_conflict_intensity(panel, CONFLICT_EVENT_COLUMNS)
        fatality = rolling_conflict_intensity(panel, CONFLICT_FATALITY_COLUMNS)
        self.assertTrue(event.iloc[:11].isna().all())
        self.assertEqual(event.iloc[11], 6.5)
        self.assertEqual(event.iloc[24], 100.0)
        self.assertEqual(fatality.iloc[11], 1.0)
        panel["conflict_event_intensity_MA12"] = event
        panel["conflict_fatality_intensity_MA12"] = fatality
        dynamic = ["conflict_event_intensity_MA12", "conflict_fatality_intensity_MA12"]
        for scope, horizon in HORIZONS.items():
            aligned = align_horizon(panel, scope, horizon, (), dynamic)
            row = aligned.loc[
                (aligned["FEWSNET_admin_code"] == "a")
                & (aligned["target_month"] == dates[-1])
            ].iloc[0]
            self.assertEqual(row["forecast_origin_month"], dates[-1] - pd.DateOffset(months=horizon))
            expected = panel.loc[
                (panel["FEWSNET_admin_code"] == "a")
                & (panel["date"] == row["forecast_origin_month"]),
                dynamic,
            ].iloc[0]
            for column in dynamic:
                if pd.isna(expected[column]):
                    self.assertTrue(pd.isna(row[column]))
                else:
                    self.assertEqual(row[column], expected[column])


if __name__ == "__main__":
    unittest.main()
