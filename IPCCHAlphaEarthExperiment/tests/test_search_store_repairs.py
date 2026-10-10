"""Regressions for the supervisor P0 model/search sample (strict tie floor, all-unroutable B2, store integrity)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import intercept_active, intercept_pool, small_contract
from ipcch_alphaearth import backbone, correction as cr, partition as pt
from ipcch_alphaearth.errors import TechnicalError


class StrictGainFloor(unittest.TestCase):
    def test_tied_candidate_below_floor_never_selected(self):
        # supervisor reproduction: gains 4e-13 (threshold 0) and 1.333e-12 (threshold 1), tolerance 1e-12
        contract = small_contract(**{"correction.support": {"n_eff": 1, "admins": 1, "target_months": 1}})
        r = np.sqrt(1.6e-12) * np.array([1.0, 1.0, -1.0, -1.0])
        pool = intercept_pool(r, admins=np.arange(4), months=np.zeros(4))
        search = pt.Search(pool, np.arange(4.0)[:, None], np.arange(4), intercept_active(), contract, 0.1)
        states = search.run(2, 3)
        ledger = search.ledger_frame()
        step1 = ledger[ledger["step"] == 1].set_index("threshold_order")
        self.assertAlmostEqual(states[0]["objective"], 1.6e-12, delta=1e-24)
        self.assertLess(step1.loc[0, "gain"], 1e-12)
        self.assertGreater(step1.loc[1, "gain"], 1e-12)
        self.assertEqual(len(states), 2)
        self.assertEqual(states[1]["map"].splits["r"][1], 1.0)  # threshold 1 (value of rank 2), never threshold 0
        self.assertTrue(bool(step1.loc[1, "accepted"]))


class AllUnroutable(unittest.TestCase):
    def test_nonempty_pool_without_routable_rows_is_exact_b2(self):
        contract = small_contract()
        rng = np.random.default_rng(1)
        pool = intercept_pool(rng.normal(size=20), admins=np.arange(20) % 7, months=np.arange(20) % 9, routable=np.zeros(20, bool))
        search = pt.Search(pool, np.full((7, 3), np.nan), np.arange(7), intercept_active(), contract, 0.1)
        states = search.run(8, 3)
        self.assertEqual(len(states), 1)
        b2 = cr.solve_b2(pool.total, intercept_active(), 0.01, 1e-9)
        self.assertEqual(states[0]["fit"]["kind"], "b2")
        np.testing.assert_array_equal(states[0]["fit"]["beta"], b2["beta"])
        fitres = pt.fit_frozen_map(pool, intercept_active(), pt.PartitionMap({"r": (0, 0.0)}, ["r0", "r1"]),
                                   np.full(20, None, dtype=object), contract, 0.1)
        self.assertEqual(fitres["g_kind"], "b2_reuse")


def _store_case(tmp: Path):
    contract = small_contract()
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 4))
    Y = rng.uniform(size=(60, 4))
    w = np.linspace(0.5, 1.0, 60)
    rows = np.arange(60)
    ident = backbone.fit_identity(arm="B1", horizon=1, block_id="b", role="historical", fit_origin=100, rows=rows,
                                  keys=np.column_stack([rows, rows]), X=X, Y=Y, w=w, features=["a", "b", "c", "d"],
                                  contract=contract, env={}, source={})
    return contract, ident, X, Y, w, backbone.ModelStore(tmp / "models")


class StoreIntegrity(unittest.TestCase):
    def test_partial_entry_refused_before_any_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract, ident, X, Y, w, store = _store_case(Path(tmp))
            from ipcch_alphaearth.artifacts import digest
            d = digest(ident)
            (store._dir(d).with_name(d + ".partial")).mkdir(parents=True)
            calls = []
            original = backbone.fit_scalar
            backbone.fit_scalar = lambda *a, **k: calls.append(1) or original(*a, **k)
            try:
                with self.assertRaises(TechnicalError):
                    store.get_or_fit(ident, X, Y, w, contract, "test")
            finally:
                backbone.fit_scalar = original
            self.assertEqual(calls, [])

    def test_slices_must_match_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract, ident, X, Y, w, store = _store_case(Path(tmp))
            for bad in ((X, Y + 0.5, w), (X + 1e-9, Y, w), (X, Y, w * 2)):
                with self.assertRaises(TechnicalError):
                    store.get_or_fit(ident, *bad, contract, "test")
            other = small_contract(**{"backbone.recipes.q3.max_depth": 4})
            with self.assertRaises(TechnicalError):
                store.get_or_fit(ident, X, Y, w, other, "test")

    def test_reuse_verifies_record_and_digest_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract, ident, X, Y, w, store = _store_case(Path(tmp))
            _, rec = store.get_or_fit(ident, X, Y, w, contract, "fit")
            d = rec["identity_digest"]
            _, again = store.get_or_fit(ident, X, Y, w, contract, "reuse")
            self.assertEqual(store.ledger[-1]["action"], "reuse")
            self.assertEqual(store.load_by_digest(d)[1]["identity_digest"], d)
            path = store._dir(d) / "record.json"
            pristine = path.read_text()
            # base score changed while keeping the old config hash
            r = json.loads(pristine)
            r["targets"]["q2"]["resolved_config"]["learner"]["learner_model_param"]["base_score"] = "999"
            path.write_text(json.dumps(r))
            with self.assertRaises(TechnicalError):
                store.get_or_fit(ident, X, Y, w, contract, "reuse")
            # identity edited while keeping the directory and stored digest
            r = json.loads(pristine)
            r["identity"]["fit_origin"] = 99
            path.write_text(json.dumps(r))
            with self.assertRaises(TechnicalError):
                store.load_by_digest(d)
            # recorded rounds no longer recoverable from the booster
            r = json.loads(pristine)
            r["targets"]["q4"]["rounds"] = 6
            path.write_text(json.dumps(r))
            with self.assertRaises(TechnicalError):
                store.load_by_digest(d)
            path.write_text(pristine)
            store.load_by_digest(d)

    def test_read_only_store_never_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract, ident, X, Y, w, _ = _store_case(Path(tmp))
            with self.assertRaises(TechnicalError):
                backbone.ModelStore(Path(tmp) / "models", read_only=True).get_or_fit(ident, X, Y, w, contract, "replay")


if __name__ == "__main__":
    unittest.main()
