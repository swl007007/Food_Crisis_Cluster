import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "stage1_aligned_georf.py"


def _load_module():
    if not SCRIPT.is_file():
        raise AssertionError("aligned Ethiopia Stage 1 is not implemented")
    spec = importlib.util.spec_from_file_location("eth_stage1_aligned", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stage1FoldTests(unittest.TestCase):
    def test_partition_guard_rejects_empty_candidates_only_within_the_call(self):
        module = _load_module()
        observed = []
        fake = SimpleNamespace(MIN_BRANCH_SAMPLE_SIZE=0)
        fake.partition = lambda: observed.append(fake.MIN_BRANCH_SAMPLE_SIZE) or "ok"

        self.assertEqual(module.partition_with_nonempty_guard(fake), "ok")
        self.assertEqual(observed, [1])
        self.assertEqual(fake.MIN_BRANCH_SAMPLE_SIZE, 0)

    def test_validation_values_do_not_enter_imputation_statistics(self):
        module = _load_module()
        records = []
        for date in pd.date_range("2020-01-01", "2020-12-01", freq="MS"):
            for admin in (1, 2):
                row = {
                    "target_month": date,
                    "fews_ipc_crisis": admin - 1,
                    "FEWSNET_admin_code": admin,
                    **{feature: 1.0 for feature in module.MODEL_PREDICTORS},
                }
                row["gini"] = np.nan
                if date == pd.Timestamp("2020-12-01"):
                    row["lat"] = np.nan
                records.append(row)
        frame = pd.DataFrame(records)
        train, _ = module.select_rolling_fold(
            frame,
            target_month="2020-12",
            horizon=1,
            window_months=36,
        )
        split = module.group_aware_train_val_split(
            frame.iloc[train]["FEWSNET_admin_code"].to_numpy(),
            val_ratio=0.2,
            random_state=5,
        )
        frame.loc[train[split["X_set"].astype(bool)], "lat"] = 999.0

        fold = module.prepare_stage1_fold(
            frame,
            target_month="2020-12",
            horizon=1,
            validation_fraction=0.2,
        )

        self.assertEqual(fold["fit_rows"], 16)
        self.assertEqual(fold["validation_rows"], 4)
        self.assertEqual(set(fold["groups_train"][fold["X_set"] == 1]), {1, 2})
        self.assertEqual(fold["medians"][0], 1.0)
        self.assertEqual(fold["X_test"][0, 0], 1.0)
        self.assertEqual(fold["medians"][module.MODEL_PREDICTORS.index("gini")], 0.0)
        self.assertTrue(np.isfinite(fold["X_train"]).all())


if __name__ == "__main__":
    unittest.main()
