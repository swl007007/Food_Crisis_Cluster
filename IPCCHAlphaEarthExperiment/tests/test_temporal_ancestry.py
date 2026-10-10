"""Future-label perturbation cannot alter an earlier fit, prediction, correction pool or map (D29 inclusive cutoffs)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from helpers import small_contract
from ipcch_alphaearth import backbone, partition as pt, pipeline, schedule, synthetic
from ipcch_alphaearth.months import parse_month


def perturbed(hi, after: int):
    keys = hi.keys.copy()
    late = keys["target_ord"] > after
    for q in ("q2", "q3", "q4", "q5"):
        keys.loc[late, q] = 1.0 - keys.loc[late, q]
    keys.loc[late, "phase_truth"] = 6 - keys.loc[late, "phase_truth"]
    return pipeline.HorizonInputs(hi.h, keys, hi.X296, hi.D, hi.routable, hi.core_idx)


class FutureLabels(unittest.TestCase):
    def test_labels_after_origin_change_nothing_at_or_before_it(self):
        contract = small_contract(**{"correction.support": {"n_eff": 30, "admins": 10, "target_months": 3}})
        inputs = synthetic.make_inputs(contract, n_admin=150, seed=8)
        O = parse_month("2020-12")  # H1 development-2021 block fit origin
        base, pert = inputs.horizons[1], perturbed(inputs.horizons[1], O)
        blocks = [b for b in schedule.horizon_blocks(contract, 1) if b.role == "historical"]
        with tempfile.TemporaryDirectory() as tmp:
            out = {}
            for name, hi in (("base", base), ("pert", pert)):
                store = backbone.ModelStore(Path(tmp) / name)
                provs, idents = [], {}
                for b in blocks:
                    r = pipeline.fit_block(hi, b, "B1", store, contract, {}, {}, inputs.features424, "test")
                    idents[b.block_id] = r["identity"]
                    provs.append(pd.DataFrame({"row": r["rows"], "target_ord": hi.target[r["rows"]], "provider_block": b.block_id,
                                               "provider_fit_origin": b.fit_origin, "provider_identity": r["identity"],
                                               **{f"p_{q}": r["raw"][:, k] for k, q in enumerate(("q2", "q3", "q4", "q5"))}}))
                prov = pd.concat(provs, ignore_index=True)
                pool = pipeline.correction_pool(hi, prov, O, contract)
                transform, ppool = pipeline.make_pool(hi, pool)
                states = pt.Search(ppool, inputs.D_admin, inputs.admin_codes, transform.active(), contract, 0.01).run(8, 3)
                out[name] = {"idents": idents, "prov": prov, "pool": pool, "states": states}
            early = [b.block_id for b in blocks if b.fit_origin <= O]
            late = [b.block_id for b in blocks if b.fit_origin > O]
            for bid in early:
                self.assertEqual(out["base"]["idents"][bid], out["pert"]["idents"][bid])
                a = out["base"]["prov"][out["base"]["prov"]["provider_block"] == bid]
                b = out["pert"]["prov"][out["pert"]["prov"]["provider_block"] == bid]
                np.testing.assert_array_equal(a.filter(like="p_").to_numpy(), b.filter(like="p_").to_numpy())
            self.assertTrue(late and all(out["base"]["idents"][x] != out["pert"]["idents"][x] for x in late))  # power check
            np.testing.assert_array_equal(out["base"]["pool"]["rows"], out["pert"]["pool"]["rows"])
            np.testing.assert_array_equal(out["base"]["pool"]["R"], out["pert"]["pool"]["R"])
            self.assertEqual([s["map"].record() for s in out["base"]["states"]], [s["map"].record() for s in out["pert"]["states"]])
            self.assertTrue((base.target[out["base"]["pool"]["rows"]] <= O).all())
            self.assertIn(O, set(base.target[out["base"]["pool"]["rows"]]))  # U == O is eligible (inclusive)


if __name__ == "__main__":
    unittest.main()
