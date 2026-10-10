"""Synthetic end-to-end: develop -> freeze-map -> predict -> report -> zero-fit replay on the real calendar.

Case A has a genuine region-specific slope (expect at least one selected recipe with a frozen split map); case B
makes every split unsupported (exact B2 everywhere: G == B2 and zero deltas). Scaled-down XGB/support only.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from helpers import small_contract
from ipcch_alphaearth import backbone, pipeline, replay, report, synthetic


def run_all(tmp: Path, contract: dict, inputs):
    ws = pipeline.Workspace(tmp)
    store = backbone.ModelStore(tmp / "models")
    env = {"test": True}
    dev = pipeline.run_develop(inputs, contract, ws, store, env)
    freeze = pipeline.run_freeze_map(inputs, contract, ws)
    pred = pipeline.run_predict(inputs, contract, ws, store, env)
    rep = report.run_report(tmp, contract, sorted(inputs.horizons))
    rp = replay.run_replay(inputs, contract, tmp, tmp / "models")
    return dev, freeze, pred, rep, rp, store


class SyntheticEndToEnd(unittest.TestCase):
    def test_regional_signal_selects_a_split_and_replays(self):
        contract = small_contract(**{"correction.support": {"n_eff": 40, "admins": 15, "target_months": 3}})
        inputs = synthetic.make_inputs(contract, regional=True, seed=11)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dev, freeze, pred, rep, rp, store = run_all(tmp, contract, inputs)
            fits = [r for r in store.ledger if r["action"] == "fit"]
            self.assertEqual(len(fits), 61)  # 38 B1 + 23 B0 quartets, each once
            selected = {h: s for h, s in dev["selections"].items() if s != "B2"}
            self.assertTrue(selected, dev["selections"])
            maps = [r for r in freeze["horizons"].values() if r["map"] is not None and r["map"]["n_leaves"] >= 2]
            self.assertTrue(maps)
            self.assertTrue(rp["pass"])
            for h in inputs.horizons:
                p = pd.read_csv(tmp / "predict" / f"predictions_h{h:02d}.csv.gz")
                cal = contract["calendar"]
                self.assertEqual(int((p["period"] == "supplement").sum()), int(np.isin(inputs.horizons[h].keys["target_month"], cal["supplement_target_months"]).sum()))
                main = p[p["period"] == "main"]
                self.assertTrue((main["origin_month"] >= cal["main_first_origin"]).all())
                self.assertTrue((main["target_month"] <= cal["main_last_target"]).all())
            self.assertEqual(rep["family_size"], 4)
            import json
            for h in inputs.horizons:
                sel = json.loads((tmp / "develop" / f"selection_h{h:02d}.json").read_text())
                self.assertEqual(len(sel["table"]), 9)
                self.assertEqual(sorted(r["recipe_id"] for r in sel["table"]),
                                 sorted(f"K{k}_lam{lam:g}" for k in (2, 4, 8) for lam in (0.01, 0.1, 1.0)))
                for r in sel["table"]:
                    self.assertEqual(sorted(r["years"]), ["2021", "2022"])
                    for y in ("2021", "2022"):
                        self.assertIn("guards_pass", r["years"][y])
                        self.assertIn("guards_failed", r["years"][y])

    def test_unsupported_splits_give_exact_b2(self):
        contract = small_contract(**{"correction.support": {"n_eff": 10 ** 9, "admins": 50, "target_months": 6}})
        inputs = synthetic.make_inputs(contract, regional=True, seed=11)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dev, freeze, pred, rep, rp, _ = run_all(tmp, contract, inputs)
            self.assertEqual(set(dev["selections"].values()), {"B2"})
            for h, r in rep["horizons"].items():
                self.assertEqual(r["decision"]["f1_gain"], "0")
                self.assertEqual(r["bootstrap"]["lower_bound"], 0.0)
                self.assertEqual(r["decision"]["status"], "no_gain_or_reversal")
                p = pd.read_csv(tmp / "predict" / f"predictions_h{int(h):02d}.csv.gz", float_precision="round_trip")
                np.testing.assert_array_equal(p[[f"G_{q}_raw" for q in ("q2", "q3", "q4", "q5")]].to_numpy(),
                                              p[[f"B2_{q}_raw" for q in ("q2", "q3", "q4", "q5")]].to_numpy())
                self.assertEqual(set(p["G_route"]), {"b2_reuse"})
            self.assertTrue(rp["pass"])


if __name__ == "__main__":
    unittest.main()
