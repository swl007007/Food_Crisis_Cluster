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
        text = (
            "Forecasting horizon / lag: 4-month lag, 8-month lag, 12-month lag. "
            "Forecasting horizon (month lag). Lag Exclude and lagged outcomes remain."
        )

        updated = labels.replace_paper_horizon_terms(text)

        self.assertIn("Forecasting horizon: 4-month horizon, 8-month horizon, 12-month horizon", updated)
        self.assertIn("Forecasting horizon.", updated)
        self.assertIn("Lag Exclude", updated)
        self.assertIn("lagged outcomes", updated)
        self.assertNotIn("4-month lag", updated)
        self.assertNotIn("8-month lag", updated)
        self.assertNotIn("12-month lag", updated)
        self.assertNotIn("horizon / lag", updated.lower())
        self.assertNotIn("month lag)", updated.lower())


if __name__ == "__main__":
    unittest.main()
