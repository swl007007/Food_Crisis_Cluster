import importlib.util
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "aligned_refit.py"


def _load_module():
    if not SCRIPT.is_file():
        raise AssertionError("aligned Ethiopia modeling helpers are not implemented")
    spec = importlib.util.spec_from_file_location("eth_aligned_refit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseHistoryTests(unittest.TestCase):
    @staticmethod
    def _panel() -> pd.DataFrame:
        records = []
        values = {
            "2020-10-01": {1: 2, 2: 1},
            "2021-02-01": {1: 3, 2: 2},
            "2021-04-01": {1: 4, 2: np.nan},
            "2021-06-01": {1: 1, 2: 4},
        }
        for date, special in values.items():
            for admin in range(1, 11):
                records.append(
                    {
                        "FEWSNET_admin_code": admin,
                        "date": date,
                        "fews_ipc": special.get(admin, 1),
                    }
                )
        return pd.DataFrame(records)

    def test_release_month_requires_ninety_percent_of_frozen_cohort(self):
        module = _load_module()
        panel = self._panel()
        panel.loc[
            panel["date"].eq("2021-06-01")
            & panel["FEWSNET_admin_code"].isin([2, 3]),
            "fews_ipc",
        ] = np.nan

        audit = module.qualify_release_months(panel, range(1, 11), threshold=0.9)
        audit = audit.set_index("release_month")

        self.assertTrue(audit.loc[pd.Timestamp("2021-04-01"), "qualifying"])
        self.assertEqual(audit.loc[pd.Timestamp("2021-04-01"), "observed_n"], 9)
        self.assertFalse(audit.loc[pd.Timestamp("2021-06-01"), "qualifying"])
        self.assertEqual(audit.loc[pd.Timestamp("2021-06-01"), "observed_n"], 8)

    def test_histories_use_three_qualifying_releases_strictly_before_origin(self):
        module = _load_module()
        panel = self._panel()
        audit = module.qualify_release_months(panel, range(1, 11), threshold=0.9)
        aligned = pd.DataFrame(
            {
                "FEWSNET_admin_code": [1, 2],
                "forecast_origin_month": ["2021-06-01", "2021-06-01"],
            }
        )

        augmented = module.attach_release_history(aligned, panel, audit)

        self.assertEqual(
            augmented.loc[0, list(module.RELEASE_FEATURES)].tolist(),
            [4.0, 3.0, 2.0],
        )
        self.assertTrue(pd.isna(augmented.loc[1, "fews_ipc_release_lag1"]))
        self.assertEqual(
            augmented.loc[1, ["fews_ipc_release_lag2", "fews_ipc_release_lag3"]].tolist(),
            [2.0, 1.0],
        )


class FoldImputationTests(unittest.TestCase):
    def test_training_medians_keep_all_columns_and_ignore_validation_values(self):
        module = _load_module()
        train = np.array([[1.0, np.nan], [3.0, np.nan], [np.nan, np.nan]])
        validation = np.array([[999.0, np.nan]])

        medians = module.fit_fold_medians(train)
        transformed_train = module.apply_fold_medians(train, medians)
        transformed_validation = module.apply_fold_medians(validation, medians)

        np.testing.assert_array_equal(medians, np.array([2.0, 0.0]))
        self.assertEqual(transformed_train.shape, train.shape)
        np.testing.assert_array_equal(transformed_validation, np.array([[999.0, 0.0]]))
        self.assertTrue(np.isfinite(transformed_train).all())

    def test_rolling_window_is_exactly_36_months_and_validation_is_latest_six(self):
        module = _load_module()
        dates = pd.date_range("2017-01-01", "2021-02-01", freq="MS")
        frame = pd.DataFrame(
            {
                "target_month": dates,
                "fews_ipc_crisis": 0,
            }
        )

        train, test = module.select_rolling_fold(
            frame,
            target_month="2021-02",
            horizon=1,
            window_months=36,
        )
        fit, validation = module.split_latest_months(
            frame.loc[train, "target_month"],
            validation_months=6,
        )

        self.assertEqual(len(train), 36)
        self.assertEqual(frame.loc[train, "target_month"].min(), pd.Timestamp("2018-01-01"))
        self.assertEqual(frame.loc[train, "target_month"].max(), pd.Timestamp("2020-12-01"))
        self.assertEqual(frame.loc[test, "target_month"].tolist(), [pd.Timestamp("2021-02-01")])
        self.assertEqual(validation.sum(), 6)
        self.assertEqual(fit.sum(), 30)


class AlignedContractTests(unittest.TestCase):
    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "scope": ["fs0"],
                "horizon_months": [1],
                "FEWSNET_admin_code": [1],
                "target_month": ["2021-06-01"],
                "forecast_origin_month": ["2021-05-01"],
                "fews_ipc_crisis": [1],
                "p1": [0.1],
                "p2": [0.2],
            }
        )

    def test_validation_requires_exact_origin_and_predictor_order(self):
        module = _load_module()
        frame = self._frame()

        module.validate_aligned_frame(frame, "fs0", 1, ("p1", "p2"))

        broken = frame.copy()
        broken["forecast_origin_month"] = "2021-04-01"
        with self.assertRaisesRegex(ValueError, "exact forecast origin"):
            module.validate_aligned_frame(broken, "fs0", 1, ("p1", "p2"))

        with self.assertRaisesRegex(ValueError, "predictor order"):
            module.validate_aligned_frame(frame[[*frame.columns[:-2], "p2", "p1"]], "fs0", 1, ("p1", "p2"))

    def test_run_local_snapshots_preserve_sources_and_append_release_features(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aligned_paths = {}
            origins = {0: "2021-05-01", 1: "2021-02-01", 2: "2020-10-01", 3: "2020-06-01"}
            targets = {0: "2021-06-01", 1: "2021-06-01", 2: "2021-06-01", 3: "2021-06-01"}
            seasonal_rows = []
            for scope, horizon in {0: 1, 1: 4, 2: 8, 3: 12}.items():
                row = {
                    "scope": f"fs{scope}",
                    "horizon_months": horizon,
                    "FEWSNET_admin_code": 1,
                    "target_month": targets[scope],
                    "forecast_origin_month": origins[scope],
                    "fews_ipc_crisis": 1,
                    **{feature: 0.0 for feature in module.ALIGNED_PREDICTORS},
                }
                for index, feature in enumerate(module.SEASON_FEATURES, start=1):
                    row[feature] = float(index)
                path = root / f"fs{scope}.csv"
                pd.DataFrame([row]).to_csv(path, index=False)
                aligned_paths[scope] = path
                origin = pd.Timestamp(origins[scope])
                seasonal_rows.append(
                    {
                        "FEWSNET_admin_code": 1,
                        "year": origin.year,
                        "month": origin.month,
                        "previous_season_end": origin - pd.offsets.MonthBegin(1),
                        **{feature: float(index) for index, feature in enumerate(module.SEASON_FEATURES, start=1)},
                    }
                )

            panel = pd.DataFrame(
                {
                    "FEWSNET_admin_code": [1] * 4,
                    "date": ["2019-06-01", "2019-10-01", "2020-02-01", "2020-04-01"],
                    "fews_ipc": [1, 2, 3, 4],
                }
            )
            panel_path = root / "working.csv"
            panel.to_csv(panel_path, index=False)
            season_path = root / "season.csv"
            pd.DataFrame(seasonal_rows).to_csv(season_path, index=False)
            before = {
                scope: hashlib.sha256(path.read_bytes()).hexdigest()
                for scope, path in aligned_paths.items()
            }

            snapshots, audit = module.build_run_local_inputs(
                aligned_paths,
                panel_path,
                season_path,
                root / "input",
                expected_admins=1,
            )

            self.assertEqual(set(snapshots), {0, 1, 2, 3})
            for scope, path in snapshots.items():
                snapshot = pd.read_csv(path)
                self.assertEqual(
                    list(snapshot.columns),
                    [*module.METADATA_COLUMNS, *module.MODEL_PREDICTORS],
                )
                self.assertEqual(len(module.MODEL_PREDICTORS), 88)
                self.assertEqual(
                    hashlib.sha256(aligned_paths[scope].read_bytes()).hexdigest(),
                    before[scope],
                )
            self.assertEqual(audit["predictor_count"], 88)
            self.assertTrue((root / "input" / "release_coverage.csv").is_file())
            self.assertTrue((root / "input" / "feature_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
