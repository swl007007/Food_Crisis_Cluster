import importlib.util
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "paper_horizon_labels.py"
spec = importlib.util.spec_from_file_location("paper_horizon_labels", SCRIPT_PATH)
labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels)


RELABELER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "relabel_final_artifact_horizons.py"


def load_relabeler_module():
    spec = importlib.util.spec_from_file_location("relabel_final_artifact_horizons", RELABELER_PATH)
    relabeler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(relabeler)
    return relabeler


class PaperHorizonLabelTests(unittest.TestCase):
    def test_label_for_scope_uses_horizon_wording(self):
        self.assertEqual(labels.label_for_scope("fs1"), "4-month horizon")
        self.assertEqual(labels.label_for_scope("fs2"), "8-month horizon")
        self.assertEqual(labels.label_for_scope("fs3"), "12-month horizon")

    def test_label_for_scope_preserves_unknown_scope(self):
        self.assertEqual(labels.label_for_scope("fs0"), "fs0")

    def test_replace_paper_horizon_terms_updates_only_display_phrases(self):
        old_axis_label = "Forecasting horizon" + " / " + "lag"
        old_header = "Forecasting horizon" + " (month " + "lag)"
        old_four = "4-month " + "lag"
        old_eight = "8-month " + "lag"
        old_twelve = "12-month " + "lag"
        text = (
            f"{old_axis_label}: {old_four}, {old_eight}, {old_twelve}. "
            f"{old_header}. Lag Exclude and lagged outcomes remain."
        )

        updated = labels.replace_paper_horizon_terms(text)

        self.assertIn("Forecasting horizon: 4-month horizon, 8-month horizon, 12-month horizon", updated)
        self.assertIn("Forecasting horizon.", updated)
        self.assertIn("Lag Exclude", updated)
        self.assertIn("lagged outcomes", updated)
        self.assertNotIn(old_four, updated)
        self.assertNotIn(old_eight, updated)
        self.assertNotIn(old_twelve, updated)
        self.assertNotIn("horizon" + " / " + "lag", updated.lower())
        self.assertNotIn("month " + "lag)", updated.lower())

    def test_replace_paper_horizon_terms_updates_plural_horizon_list(self):
        old_phrase = "4-month, 8-month, and 12-month " + "lags"
        text = f"Rows are reported across the {old_phrase}."

        updated = labels.replace_paper_horizon_terms(text)

        self.assertEqual(
            updated,
            "Rows are reported across the 4-month, 8-month, and 12-month horizons.",
        )
        self.assertEqual(labels.forbidden_paper_lag_terms(updated), [])

    def test_replace_paper_horizon_terms_preserves_lagged_mechanics(self):
        text = (
            "4-month lagged outcomes, 8-month lagged non-crisis states, "
            "and internal lag mechanics should remain unchanged."
        )

        updated = labels.replace_paper_horizon_terms(text)

        self.assertEqual(updated, text)

    def test_is_allowed_remaining_lag_line_distinguishes_technical_lag_terms(self):
        self.assertTrue(labels.is_allowed_remaining_lag_line("Lag Exclude"))
        self.assertTrue(labels.is_allowed_remaining_lag_line("Lagged outcomes are covariates."))
        self.assertTrue(labels.is_allowed_remaining_lag_line("mean_food_price_lag4m"))
        self.assertFalse(labels.is_allowed_remaining_lag_line("Forecasting horizon: 4-month lag"))

    def test_forbidden_paper_lag_terms_reports_only_display_terms(self):
        old_four = "4-month " + "lag"
        old_eight = "8-month-" + "lag"
        old_axis = "Forecasting horizon" + " / " + "lag"
        text = (
            f"{old_four}; {old_eight}; {old_axis}; "
            "Lag Exclude; lagged outcomes; mean_food_price_lag4m."
        )

        self.assertEqual(
            labels.forbidden_paper_lag_terms(text),
            [old_four, old_eight, old_axis],
        )

    def test_relabeler_csv_dry_run_preserves_file_and_numeric_signature(self):
        relabeler = load_relabeler_module()
        old_axis = "Forecasting horizon" + " / " + "lag"
        old_four = "4-month " + "lag"

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "table.csv"
            original = f"{old_axis},score\n{old_four},1.25\nLag Exclude,2\n"
            csv_path.write_text(original, encoding="utf-8")

            result = relabeler.relabel_file(csv_path, dry_run=True)

            self.assertTrue(result.updated)
            self.assertEqual(csv_path.read_text(encoding="utf-8"), original)
            self.assertEqual(result.remaining_forbidden_terms, [])

    def test_relabeler_preserves_crlf_newlines_when_writing_text_file(self):
        relabeler = load_relabeler_module()
        old_axis = "Forecasting horizon" + " / " + "lag"

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "table.csv"
            original = f"{old_axis},score\r\n4-month horizon,1.25\r\n"
            csv_path.write_bytes(original.encode("utf-8"))

            result = relabeler.relabel_file(csv_path, dry_run=False)

            self.assertTrue(result.updated)
            written = csv_path.read_bytes()
            self.assertIn(b"\r\n", written)
            self.assertNotIn(b"\n4-month", written.replace(b"\r\n", b""))
            self.assertIn(b"Forecasting horizon,score\r\n", written)

    def test_relabeler_requires_lag_exclude_only_for_ablation_workbook(self):
        relabeler = load_relabeler_module()
        old_four = "4-month " + "lag"

        with tempfile.TemporaryDirectory() as tmp:
            main_path = Path(tmp) / "main_month_ind_cont3.xlsx"
            wb = Workbook()
            wb.active["A1"] = old_four
            wb.save(main_path)

            main_result = relabeler.relabel_file(main_path, dry_run=True)

            self.assertTrue(main_result.updated)
            self.assertEqual(main_result.remaining_forbidden_terms, [])

            ablation_path = Path(tmp) / "ablation_feature_exclude.xlsx"
            wb = Workbook()
            wb.active["A1"] = old_four
            wb.save(ablation_path)

            with self.assertRaisesRegex(ValueError, "Lag Exclude"):
                relabeler.relabel_file(ablation_path, dry_run=True)


if __name__ == "__main__":
    unittest.main()
