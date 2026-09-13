import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "run_stage3_aligned.py"


def _load_module():
    if not SCRIPT.is_file():
        raise AssertionError("aligned Ethiopia Stage 3 is not implemented")
    spec = importlib.util.spec_from_file_location("eth_stage3_aligned", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stage3FoldTests(unittest.TestCase):
    @staticmethod
    def _frame(module):
        records = []
        for date in pd.date_range("2020-01-01", "2021-02-01", freq="MS"):
            for admin in (1, 2):
                row = {
                    "target_month": date,
                    "fews_ipc_crisis": admin - 1,
                    "FEWSNET_admin_code": admin,
                    **{feature: 1.0 for feature in module.MODEL_PREDICTORS},
                }
                row["gini"] = np.nan
                if pd.Timestamp("2020-07-01") <= date <= pd.Timestamp("2020-12-01"):
                    row["lat"] = 999.0
                if date == pd.Timestamp("2021-02-01"):
                    row["lat"] = np.nan
                records.append(row)
        return pd.DataFrame(records)

    def test_threshold_and_final_imputers_never_use_later_rows(self):
        module = _load_module()
        prepared = module.prepare_stage3_fold(
            self._frame(module),
            target_month="2021-02",
            horizon=1,
            train_window=36,
            validation_months=6,
        )

        self.assertEqual(prepared["threshold_medians"][0], 1.0)
        self.assertEqual(prepared["final_medians"][0], 500.0)
        self.assertEqual(prepared["X_test"][0, 0], 500.0)
        gini = module.MODEL_PREDICTORS.index("gini")
        self.assertEqual(prepared["threshold_medians"][gini], 0.0)
        self.assertEqual(prepared["final_medians"][gini], 0.0)
        self.assertTrue(np.isfinite(prepared["X_fit"]).all())
        self.assertTrue(np.isfinite(prepared["X_validation"]).all())
        self.assertTrue(np.isfinite(prepared["X_test"]).all())

    def test_partition_map_must_cover_the_exact_snapshot_cohort(self):
        module = _load_module()
        frame = self._frame(module)
        valid = pd.DataFrame(
            {"FEWSNET_admin_code": [1, 2], "cluster": [10, 20]}
        )

        assigned, column = module.assign_partitions(frame, valid)
        self.assertEqual(column, "cluster")
        self.assertEqual(set(assigned), {"10", "20"})

        with self.assertRaisesRegex(ValueError, "exact snapshot cohort"):
            module.assign_partitions(frame, valid.iloc[:1])


if __name__ == "__main__":
    unittest.main()
