"""Pipeline boundaries from the supervisor connection sample: empty OOT pools are a declared identity correction;
the MAP_FREEZE hash chain is verified before any frozen map is used."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from helpers import small_contract
from ipcch_alphaearth import backbone, correction as cr, pipeline, replay, report, synthetic
from ipcch_alphaearth.errors import ContractError
from ipcch_alphaearth.months import parse_month


def drop_targets(inputs, lo: str, hi: str):
    """Remove every key with target in [lo, hi] (all H), so earlier development pools become empty."""
    a, b = parse_month(lo), parse_month(hi)
    for h, hz in list(inputs.horizons.items()):
        keep = ~((hz.target >= a) & (hz.target <= b))
        inputs.horizons[h] = pipeline.HorizonInputs(h, hz.keys[keep].reset_index(drop=True), hz.X296[keep], hz.D[keep],
                                                     hz.routable[keep], hz.core_idx)
    return inputs


class EmptyPoolIdentity(unittest.TestCase):
    def test_empty_development_pool_is_declared_identity_end_to_end(self):
        contract = small_contract(**{"correction.support": {"n_eff": 30, "admins": 10, "target_months": 3}})
        inputs = drop_targets(synthetic.make_inputs(contract, n_admin=150, seed=5), "2018-01", "2020-12")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ws, store = pipeline.Workspace(tmp), backbone.ModelStore(tmp / "models")
            pipeline.run_develop(inputs, contract, ws, store, {})
            for h in inputs.horizons:
                ctx = json.loads((tmp / "develop" / f"context_h{h:02d}_dev_2021.json").read_text())
                self.assertTrue(ctx["identity_empty_pool"])
                self.assertEqual(ctx["pool_rows"], 0)
                self.assertEqual(ctx["lambdas"], {})
                dev = pd.read_csv(tmp / "develop" / f"development_predictions_h{h:02d}.csv.gz", float_precision="round_trip")
                part = dev[dev["context"] == f"h{h:02d}_dev_2021"]
                np.testing.assert_array_equal(part[[f"B2_{q}_raw" for q in ("q2", "q3", "q4", "q5")]].to_numpy(),
                                              part[[f"B1_{q}_raw" for q in ("q2", "q3", "q4", "q5")]].to_numpy())
                sel = json.loads((tmp / "develop" / f"selection_h{h:02d}.json").read_text())
                for row in sel["table"]:
                    self.assertEqual(row["years"]["2021"]["gain"], "0")
            pipeline.run_freeze_map(inputs, contract, ws)
            pipeline.run_predict(inputs, contract, ws, store, {})
            report.run_report(tmp, contract, sorted(inputs.horizons))
            self.assertTrue(replay.run_replay(inputs, contract, tmp, tmp / "models")["pass"])

    def test_annual_context_with_empty_pool_returns_b1(self):
        contract = small_contract()
        inputs = synthetic.make_inputs(contract, n_admin=40, seed=2)
        hi = inputs.horizons[1]
        empty = {"rows": np.array([], dtype=np.int64)}
        fit, support = pipeline.context_fit(hi, empty, None, None, contract)
        self.assertEqual(fit.g_kind, "identity_empty_pool")
        P = np.random.default_rng(0).uniform(size=(5, 4))
        out = fit.predict(P, np.zeros((5, 3)), np.full(5, None, dtype=object))
        np.testing.assert_array_equal(out["b2"], P)
        np.testing.assert_array_equal(out["g"], P)
        self.assertEqual(cr.CorrectionFit.from_record(fit.record()).g_kind, "identity_empty_pool")


class FreezeChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = small_contract(**{"correction.support": {"n_eff": 30, "admins": 10, "target_months": 3}})
        cls.inputs = synthetic.make_inputs(cls.contract, n_admin=150, seed=5)
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name) / "base"
        ws, store = pipeline.Workspace(root), backbone.ModelStore(root / "models")
        pipeline.run_develop(cls.inputs, cls.contract, ws, store, {})
        pipeline.run_freeze_map(cls.inputs, cls.contract, ws)
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def copy(self, name):
        import shutil
        dst = Path(self.tmp.name) / name
        shutil.copytree(self.root, dst)
        return dst

    def assert_refused(self, run):
        with self.assertRaises(ContractError):
            pipeline.run_predict(self.inputs, self.contract, pipeline.Workspace(run), backbone.ModelStore(run / "models"), {})
        self.assertFalse((run / "predict" / "COMPLETE.json").exists())

    def test_changed_final_map_refused(self):
        run = self.copy("map")
        path = run / "maps" / "final_map_h01.json"
        rec = json.loads(path.read_text())
        rec["map"] = {"splits": {"r": {"descriptor_index": 0, "descriptor": None, "threshold": 0.0, "threshold_hex": (0.0).hex()}},
                      "leaves": ["r0", "r1"], "n_leaves": 2, "depth": 1}
        rec["lambda_region"] = 0.1
        path.write_text(json.dumps(rec))
        self.assert_refused(run)

    def test_changed_freeze_manifest_refused(self):
        run = self.copy("freeze")
        path = run / "maps" / "MAP_FREEZE.json"
        rec = json.loads(path.read_text())
        rec["cutoff"] = "2023-06"
        path.write_text(json.dumps(rec))
        self.assert_refused(run)

    def test_changed_selection_refused(self):
        run = self.copy("selection")
        path = run / "develop" / "selection_h03.json"
        rec = json.loads(path.read_text())
        rec["selected"] = "K8_lam1"
        path.write_text(json.dumps(rec))
        self.assert_refused(run)


if __name__ == "__main__":
    unittest.main()
