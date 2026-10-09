"""rebuild_check.compare on small synthetic old/new stores.

Run: /home/swl007007/.venvs/ipcch-mlflow/bin/python -m unittest discover -s IPCCHMLflow/tests -v
"""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rebuild_check as rc  # noqa: E402

KEY = "p6_geoxgb/run/H1/geo/seed42"


def stores():
    old = {"experiments": {
        rc.OLD_DETAIL: {"o1": {"name": "x", "tags": {"source_key": KEY, "family": "p6_geoxgb", "arm": "geo", "horizon": "1"},
                               "metrics": {"main.E_persist.binary.f1": 0.5, "main.E_persist.n": 4.0,
                                           "main.E_persist.contrast.geo_vs_persistence.point_delta": 0.1}}},
        rc.OLD_SUMMARY: {"s1": {"name": "y", "tags": {"projection_key": f"{KEY}#main.E_persist", "family": "p6_geoxgb"},
                                "metrics": {"binary.f1": 0.5, "n": 4.0}}}}, "datasets": {}}
    new = {"experiments": {
        rc.NEW_DETAIL: {"n1": {"name": "x", "tags": {"zz_prov.source_key": KEY, "family": "geoxgb_reference",
                                                     "arm": "partitioned_gated", "lead_months": "01"},
                               "metrics": {"primary.persistence_available.binary.f1": 0.5,
                                           "primary.persistence_available.n_rows": 4.0,
                                           "primary.persistence_available.bootstrap.partitioned_gated_minus_persistence"
                                           ".binary.f1.delta": 0.1}}},
        rc.NEW_DASHBOARD: {"d1": {"name": "z", "tags": {"zz_prov.original_source_key": KEY, "seed": "42",
                                                        "family": "geoxgb_reference", "arm": "partitioned_gated",
                                                        "lead_months": "01", "zz_prov.projection_key": "k"},
                                  "metrics": {"primary.persistence_available.binary.f1": 0.5,
                                              "primary.persistence_available.n_rows": 4.0,
                                              "primary.persistence_available.binary.f1.minus_persistence": 0.1}}}},
        "datasets": {"IPCCH eval | 1-month | 2023-02..2025-10 | persistence_available": ["abc"]}}
    return old, new


class RebuildCheckTest(unittest.TestCase):
    def test_equal_values_pass(self):
        rep = rc.compare(*stores())
        self.assertTrue(rep["passed"], rep["problems"])
        self.assertEqual((rep["detailed"]["values_equal"], rep["summary"]["values_equal"],
                          rep["dashboard_extra"]["contrast_values"]), (3, 2, 1))

    def test_panel_value_new_in_dashboard_checked_against_detailed(self):
        old, new = stores()
        new["experiments"][rc.NEW_DETAIL]["n1"]["metrics"]["combined.all_scored.binary.f1"] = 0.4
        dash = new["experiments"][rc.NEW_DASHBOARD]["d1"]["metrics"]
        dash["combined.all_scored.binary.f1"] = 0.4
        rep = rc.compare(old, new)
        self.assertFalse(any("combined" in p for p in rep["problems"] if "dashboard" in p), rep["problems"])
        self.assertEqual(rep["dashboard_extra"]["panel_values_new_in_dashboard"], 1)
        dash["combined.all_scored.binary.f1"] = 0.41
        self.assertTrue(any("combined.all_scored.binary.f1" in p for p in rc.compare(old, new)["problems"]))

    def test_changed_detailed_value_reported(self):
        old, new = stores()
        new["experiments"][rc.NEW_DETAIL]["n1"]["metrics"]["primary.persistence_available.binary.f1"] = 0.51
        rep = rc.compare(old, new)
        self.assertFalse(rep["passed"])
        self.assertTrue(any("main.E_persist.binary.f1" in p for p in rep["problems"]))

    def test_unexplained_dashboard_value_and_split_dataset_reported(self):
        old, new = stores()
        new2 = copy.deepcopy(new)
        new2["experiments"][rc.NEW_DASHBOARD]["d1"]["metrics"]["primary.persistence_available.binary.f1.minus_persistence"] = 0.2
        new2["datasets"]["IPCCH eval | 1-month | 2023-02..2025-10 | persistence_available"] = ["abc", "def"]
        rep = rc.compare(old, new2)
        self.assertEqual(len(rep["problems"]), 2, rep["problems"])
        new3 = copy.deepcopy(new)
        new3["experiments"][rc.NEW_DETAIL]["n1"]["metrics"]["primary.all_scored.binary.f1"] = 0.7
        self.assertTrue(any("without an old source" in p for p in rc.compare(old, new3)["problems"]))


if __name__ == "__main__":
    unittest.main()
