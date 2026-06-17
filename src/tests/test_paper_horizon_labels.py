import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "paper_horizon_labels.py"
spec = importlib.util.spec_from_file_location("paper_horizon_labels", SCRIPT_PATH)
labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels)


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


if __name__ == "__main__":
    unittest.main()
