"""D30 analysis-grid extraction: semantics fixture, grid-aligned windows, plan binding and EE builder structure."""

import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_alphaearth import aemoments, descriptors, extraction
from ipcch_alphaearth.contract import load_contract
from ipcch_alphaearth.errors import TechnicalError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ee_pilot_c as epc  # noqa: E402


# --------------------------------------------------------------------------- independent semantics fixture (numpy)


class Source:
    """A native 10 m lattice (offset + rotation) with 64-band vectors and per-band masks."""

    def __init__(self, sid, ox, oy, theta, n, seed):
        rng = np.random.default_rng(seed)
        self.id, self.ox, self.oy, self.t, self.n = sid, ox, oy, theta, n
        v = rng.normal(size=(n, n, 64))
        self.v = v / np.linalg.norm(v, axis=2, keepdims=True)
        self.m = np.ones((n, n, 64), dtype=bool)

    def sample(self, x, y):
        """Nearest native pixel (the one containing the point) -> (vector, all-band-valid) or (None, False) outside."""
        c, s = math.cos(-self.t), math.sin(-self.t)
        u, w = c * (x - self.ox) - s * (y - self.oy), s * (x - self.ox) + c * (y - self.oy)
        i, j = math.floor(u / 10), math.floor(w / 10)
        if not (0 <= i < self.n and 0 <= j < self.n):
            return None, False
        vec = self.v[i, j]
        return vec, bool(self.m[i, j].all() and np.isfinite(vec).all())


def reference_c(sources, n_cells, inside):
    """D30 on an analysis lattice of n_cells x n_cells 10 m cells: owner = first valid source in ascending id."""
    ordered = sorted(sources, key=lambda s: s.id)
    owner = np.full((n_cells, n_cells), "", dtype=object)
    vec = np.zeros((n_cells, n_cells, 64))
    for a in range(n_cells):
        for b in range(n_cells):
            if not inside(a, b):
                continue
            for s in ordered:
                v, ok = s.sample(a * 10 + 5, b * 10 + 5)
                if ok:
                    owner[a, b], vec[a, b] = s.id, v
                    break
    return owner, vec


def sums(owner, vec, cells):
    A, S, Q = 0.0, np.zeros(64), np.zeros(64)
    for a, b in cells:
        if owner[a, b]:
            A += 100.0
            S += 100.0 * vec[a, b]
            Q += 100.0 * vec[a, b] ** 2
    return A, S, Q


class SemanticsFixture(unittest.TestCase):
    def setUp(self):
        self.n = 30
        self.a = Source("aa1", ox=-3.0, oy=4.0, theta=0.07, n=22, seed=1)   # covers the west part
        self.b = Source("bb2", ox=120.0, oy=-6.0, theta=-0.05, n=22, seed=2)  # overlaps a, covers the east part
        self.a.m[5, 5, 17] = False  # one masked band invalidates the whole vector of that native pixel
        self.a.v[6, 6, 3] = np.inf  # nonfinite band invalidates the whole vector
        self.a.m[15:, :, :] = False  # a fully masked on part of its footprint (where b is valid)
        self.inside = lambda i, j: (i - 15) ** 2 + (j - 15) ** 2 <= 13 ** 2

    def test_first_valid_whole_vector_no_band_mixing_and_exclusive_cells(self):
        owner, vec = reference_c([self.a, self.b], self.n, self.inside)
        cells = [(i, j) for i in range(self.n) for j in range(self.n) if self.inside(i, j)]
        for i, j in cells:
            if not owner[i, j]:
                continue
            src = self.a if owner[i, j] == "aa1" else self.b
            v, ok = src.sample(i * 10 + 5, j * 10 + 5)
            self.assertTrue(ok)
            np.testing.assert_array_equal(vec[i, j], v)  # whole vector from the single owner
            if owner[i, j] == "bb2":
                self.assertFalse(self.a.sample(i * 10 + 5, j * 10 + 5)[1])  # b only where a is invalid/outside
        both_valid = sum(1 for i, j in cells if self.a.sample(i * 10 + 5, j * 10 + 5)[1] and self.b.sample(i * 10 + 5, j * 10 + 5)[1])
        self.assertGreater(both_valid, 0)  # the fixture exercises genuine overlap
        self.assertTrue(set(np.unique(owner[[c[0] for c in cells], [c[1] for c in cells]])) <= {"", "aa1", "bb2"})

    def test_input_order_is_irrelevant_ids_decide(self):
        o1, v1 = reference_c([self.a, self.b], self.n, self.inside)
        o2, v2 = reference_c([self.b, self.a], self.n, self.inside)
        np.testing.assert_array_equal(o1, o2)
        np.testing.assert_array_equal(v1, v2)

    def test_windows_additive_on_the_same_lattice_and_moments_without_renormalization(self):
        owner, vec = reference_c([self.a, self.b], self.n, self.inside)
        cells = [(i, j) for i in range(self.n) for j in range(self.n) if self.inside(i, j)]
        whole = sums(owner, vec, cells)
        windows = [[c for c in cells if c[0] < 12 and c[1] < 17], [c for c in cells if c[0] >= 12 and c[1] < 17], [c for c in cells if c[1] >= 17]]
        parts = [sums(owner, vec, w) for w in windows]
        self.assertEqual(sum(len(w) for w in windows), len(cells))  # every cell in exactly one window
        self.assertAlmostEqual(sum(p[0] for p in parts), whole[0], places=6)
        np.testing.assert_allclose(sum(p[1] for p in parts), whole[1], rtol=1e-12, atol=1e-12)
        pieces = [{"piece_id": f"w{k}", "status": "complete", "valid_area": p[0], "S": list(p[1]), "Q": list(p[2])} for k, p in enumerate(parts)]
        merged = aemoments.merge_pieces(pieces, ["w0", "w1", "w2"])
        low = aemoments.descriptor(merged, polygon_area=len(cells) * 100.0)
        self.assertEqual(low["usable"], merged["A"] / (len(cells) * 100.0) >= 0.8)  # coverage rule still applies
        d = aemoments.descriptor(merged, polygon_area=merged["A"])  # full coverage to check the moments themselves
        owned = np.array([vec[c] for c in cells if owner[c]])
        np.testing.assert_allclose(d["mean"], owned.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(d["sd"], owned.std(axis=0), atol=1e-9)  # population SD
        self.assertLess(np.linalg.norm(d["mean"]), 1.0)  # mean vector is not renormalized

    def test_all_masked_and_outside_footprint_give_genuine_zeros(self):
        self.a.m[:] = False
        far = Source("cc3", ox=10_000.0, oy=10_000.0, theta=0.0, n=5, seed=3)  # footprint outside the area
        owner, vec = reference_c([self.a, far], self.n, self.inside)
        cells = [(i, j) for i in range(self.n) for j in range(self.n) if self.inside(i, j)]
        A, S, Q = sums(owner, vec, cells)
        self.assertEqual((A, float(np.abs(S).sum()), float(np.abs(Q).sum())), (0.0, 0.0, 0.0))
        d = aemoments.descriptor(aemoments.merge_pieces([{"piece_id": "w", "status": "complete", "valid_area": A, "S": list(S),
                                                           "Q": list(Q)}], ["w"]), polygon_area=1000.0)
        self.assertEqual(d["route_reason"], "no_coverage")


# --------------------------------------------------------------------------- lattice windows and plan


class LatticeWindows(unittest.TestCase):
    def setUp(self):
        from shapely.geometry import Polygon

        self.contract = load_contract()
        self.lattice = extraction.Lattice.from_contract(self.contract)
        self.geom = Polygon([(45.00, 13.00), (45.30, 13.02), (45.27, 13.25), (45.02, 13.22)],
                            [[(45.10, 13.08), (45.15, 13.08), (45.15, 13.12), (45.10, 13.12)]])

    def test_windows_tile_the_geometry_without_overlap(self):
        wins = extraction.lattice_windows(self.geom, self.lattice, max_cells=200_000)
        self.assertGreater(len(wins), 4)
        total = sum(w["geometry"].area for w in wins)
        self.assertAlmostEqual(total / self.geom.area, 1.0, places=12)
        for k, w in enumerate(wins):
            self.assertLessEqual(w["lattice_cells_bbox"], 200_000)
            for v in wins[k + 1:]:
                self.assertLess(w["geometry"].intersection(v["geometry"]).area, 1e-15)

    def test_cut_lines_are_lattice_lines(self):
        wins = extraction.lattice_windows(self.geom, self.lattice, max_cells=200_000)
        cols = sorted({c for w in wins for c in w["cols"]})
        for c in cols:
            x, _ = self.lattice.fw.transform(self.lattice.lon(c), 0.0)
            self.assertAlmostEqual((x - self.lattice.x0) / self.lattice.dx, c, places=6)
        x_a, _ = self.lattice.fw.transform(self.lattice.lon(cols[1]), -20.0)
        x_b, _ = self.lattice.fw.transform(self.lattice.lon(cols[1]), 30.0)
        self.assertAlmostEqual(x_a, x_b, places=6)  # a cut meridian is one lattice column line at every latitude

    def test_non_cylindrical_lattice_refused(self):
        with self.assertRaises(Exception):
            extraction.Lattice("EPSG:32638", [10, 0, 0, 0, -10, 0])

    def test_plan_binds_ascending_candidates_recipe_and_zero_coverage_admins(self):
        import geopandas as gpd
        from shapely.geometry import box

        g = gpd.GeoDataFrame({"admin_code": [7, 8]}, geometry=[self.geom, box(10, 10, 10.01, 10.01)], crs="EPSG:4326")
        fps = {"zz9": box(45.10, 12.9, 45.5, 13.5), "aa1": box(44.9, 12.9, 45.2, 13.5)}
        plan, geoms = extraction.plan_c_requests(g, fps, self.contract, max_cells=500_000)
        rid = extraction.recipe_id(self.contract)
        self.assertEqual(set(plan["recipe"]), {rid})
        seven = plan[plan.admin_code == 7]
        for r in seven.itertuples():
            ids = r.asset_id.split(",")
            self.assertEqual(ids, sorted(ids))
            self.assertTrue(set(ids) <= {"aa1", "zz9"})
        self.assertIn("aa1,zz9", set(seven.asset_id))  # windows covering the overlap keep both, lowest id first
        eight = plan[plan.admin_code == 8]
        self.assertEqual(len(eight), 1)
        self.assertTrue(pd.isna(eight.iloc[0].piece_id))  # no footprint: genuine zero-coverage admin kept


class RecipeBinding(unittest.TestCase):
    def setUp(self):
        self.contract = load_contract()
        self.rid = extraction.recipe_id(self.contract)
        cols = {"admin_code": [1], "piece_id": ["1|w000_000"], "asset_id": ["aa1,zz9"], "status": ["complete"], "valid_area": [90.0],
                "recipe": [self.rid]}
        for b in aemoments.BANDS:
            cols[f"S_{b}"], cols[f"Q_{b}"] = [9.0], [1.4]
        self.pieces = pd.DataFrame(cols)
        self.plan = pd.DataFrame({"admin_code": [1], "piece_id": ["1|w000_000"], "asset_id": ["aa1,zz9"], "recipe": [self.rid]})
        self.polys = pd.DataFrame({"admin_code": [1], "polygon_area_m2": [100.0]})

    def test_c_plan_accepted(self):
        _, ledger = descriptors.descriptor_table(self.pieces, self.plan, self.polys, self.contract)
        self.assertEqual(ledger["route_reason"].tolist(), ["usable"])

    def test_old_rule_plan_or_export_refused(self):
        with self.assertRaises(TechnicalError):
            descriptors.descriptor_table(self.pieces, self.plan.drop(columns="recipe"), self.polys, self.contract)
        with self.assertRaises(TechnicalError):
            descriptors.descriptor_table(self.pieces, self.plan.assign(recipe="native-first-footprint"), self.polys, self.contract)
        with self.assertRaises(TechnicalError):
            descriptors.descriptor_table(self.pieces.assign(recipe="other"), self.plan, self.polys, self.contract)
        with self.assertRaises(TechnicalError):
            descriptors.descriptor_table(self.pieces.assign(asset_id="zz9,aa1"), self.plan, self.polys, self.contract)


# --------------------------------------------------------------------------- EE builder structure (fake ee)


class Fake:
    log = []

    def __init__(self, tag, origin=None):
        self.tag, self.origin = tag, origin

    def __getattr__(self, name):
        def op(*args, **kwargs):
            Fake.log.append((self.tag, name, args, kwargs))
            return Fake(f"{self.tag}.{name}", self.origin)
        return op


class FakeEE:
    def __init__(self):
        Fake.log = []
        self.mosaics = []
        ee = self

        class Image:
            def __new__(cls, ident=None):
                return Fake(f"img[{ident}]", origin=ident)

            @staticmethod
            def constant(v):
                return Fake(f"const[{v}]")

            @staticmethod
            def pixelArea():
                return Fake("pixelArea")

            @staticmethod
            def cat(items):
                return Fake("cat")

        class ImageCollection:
            def __init__(self, items):
                ee.mosaics.append([i.origin for i in items])
                self.items = items

            def mosaic(self):
                return Fake("mosaic")

        self.Image, self.ImageCollection = Image, ImageCollection
        self.Reducer = types.SimpleNamespace(min=lambda: "min", sum=lambda: Fake("sum"), count=lambda: Fake("count"))
        self.Geometry = lambda *a, **k: Fake("geom")
        self.Dictionary = lambda d: types.SimpleNamespace(getInfo=lambda: {"keys": sorted(d)})


class BuilderStructure(unittest.TestCase):
    def test_mosaic_order_mask_before_mosaic_and_grid(self):
        ee = FakeEE()
        r = {"request_id": "t", "candidates": ["aa1", "mm5", "zz9"], "geometry": {"type": "Point", "coordinates": [0, 0]}}
        out = epc.reduction(ee, r, {"max_pixels": 1e9, "tile_scale": 4})
        self.assertEqual(ee.mosaics, [[f"{epc.COLLECTION}/zz9", f"{epc.COLLECTION}/mm5", f"{epc.COLLECTION}/aa1"]])
        for aid in r["candidates"]:  # per asset: rank band added, then updateMask(valid) applied to all bands before mosaic
            ops = [(t, n) for t, n, _a, _k in Fake.log if t.startswith(f"img[{epc.COLLECTION}/{aid}]")]
            names = [n for _t, n in ops]
            self.assertIn("addBands", names)
            self.assertIn("updateMask", names)
            self.assertLess(names.index("addBands"), len(names) - 1 - names[::-1].index("updateMask") + 1)
        reductions = [k for _t, n, _a, k in Fake.log if n == "reduceRegion"]
        self.assertEqual(len(reductions), 3)
        for k in reductions:
            self.assertEqual((k["crs"], k["crsTransform"], k["bestEffort"]), (epc.EE_CRS_WKT, [10, 0, 0, 0, -10, 0], False))
        self.assertEqual(out["keys"], ["area_by_rank", "count", "sums"])

    def test_sums_only_request_issues_one_aggregation_with_the_same_composite_and_grid(self):
        ee = FakeEE()
        r = {"request_id": "t", "candidates": ["aa1", "zz9"], "geometry": {"type": "Point", "coordinates": [0, 0]},
             "aggregations": "sums"}
        out = epc.reduction(ee, r, {"max_pixels": 1e9, "tile_scale": 4})
        self.assertEqual(ee.mosaics, [[f"{epc.COLLECTION}/zz9", f"{epc.COLLECTION}/aa1"]])  # same descending mosaic
        reductions = [k for _t, n, _a, k in Fake.log if n == "reduceRegion"]
        self.assertEqual(len(reductions), 1)
        self.assertEqual((reductions[0]["crs"], reductions[0]["crsTransform"], reductions[0]["bestEffort"]),
                         (epc.EE_CRS_WKT, [10, 0, 0, 0, -10, 0], False))
        self.assertEqual(out["keys"], ["sums"])

    def test_manifest_guards(self):
        good = {"recipe": epc.RECIPE, "grid": epc.GRID, "best_effort": False, "project": "p", "limits": {},
                "grid_crs_wkt": epc.EE_CRS_WKT, "projection_check": [[45.0, 13.0, 4341882.611290343, 1644631.1629230133]],
                "requests": [{"request_id": "r", "candidates": ["aa1", "bb2"]}]}
        with tempfile.TemporaryDirectory() as tmp:
            def write(body, release="rel"):
                path = Path(tmp) / "m.json"
                path.write_text(json.dumps({**body, "request_sha256": epc.digest(body), "release_id": release}))
                return path
            self.assertEqual(epc.load_released(write(good))["release_id"], "rel")
            for bad in ({**good, "release_id": None}, {**good, "grid": {"crs": "EPSG:4326", "crsTransform": [10, 0, 0, 0, -10, 0]}},
                        {**good, "recipe": "native"}, {**good, "requests": [{"request_id": "r", "candidates": ["bb2", "aa1"]}]},
                        {**good, "grid_crs_wkt": "EPSG:6933"}, {**good, "projection_check": []}):
                rel = bad.pop("release_id", "rel")
                with self.assertRaises(SystemExit):
                    epc.load_released(write(bad, rel))


if __name__ == "__main__":
    unittest.main()


class ConnectedIntegration(unittest.TestCase):
    def test_export_csv_round_trip_keeps_recipe_and_binding(self):
        contract = load_contract()
        rid = extraction.recipe_id(contract)
        cols = {"admin_code": [1], "piece_id": ["1|w000_000"], "asset_id": ["aa1,zz9"], "status": ["complete"], "valid_area": [90.0], "recipe": [rid]}
        for b in aemoments.BANDS:
            cols[f"S_{b}"], cols[f"Q_{b}"] = [9.0], [1.4]
        plan = pd.DataFrame({"admin_code": [1], "piece_id": ["1|w000_000"], "asset_id": ["aa1,zz9"], "recipe": [rid]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "export.csv"
            pd.DataFrame(cols).to_csv(path, index=False)
            parsed = extraction.parse_export_csv(path)
            self.assertEqual(parsed["recipe"].tolist(), [rid])
            self.assertEqual(parsed["asset_id"].tolist(), ["aa1,zz9"])
            got, _ = extraction.validate_export([parsed], plan)
            _, ledger = descriptors.descriptor_table(got, plan, pd.DataFrame({"admin_code": [1], "polygon_area_m2": [100.0]}), contract)
            self.assertEqual(ledger["route_reason"].tolist(), ["usable"])

    def test_partial_footprint_admin_has_no_empty_candidate_pieces(self):
        import geopandas as gpd
        from shapely.geometry import box

        contract = load_contract()
        g = gpd.GeoDataFrame({"admin_code": [5]}, geometry=[box(40.0, 5.0, 40.30, 5.20)], crs="EPSG:4326")
        fps = {"aa1": box(39.9, 4.9, 40.06, 5.3)}  # covers only the western strip
        plan, geoms = extraction.plan_c_requests(g, fps, contract, max_cells=4_000_000)
        real = plan[plan["piece_id"].notna()]
        self.assertGreater(len(real), 0)
        self.assertTrue((real["candidates"] > 0).all())
        self.assertTrue(all(ids for ids in real["asset_id"]))
        self.assertGreater(int(real["no_source_windows"].iloc[0]), 0)
        sourced_area = real["window_area_m2"].sum() + real["no_source_area_m2"].iloc[0]
        # every window accounted; the ~1e-6 residual is the recorded edge-convention effect (window cuts follow parallels,
        # which are not geodesics, while geodesic_area measures geodesic edges)
        self.assertAlmostEqual(sourced_area / real["polygon_area_m2"].iloc[0], 1.0, delta=1e-5)
        body = {"recipe": epc.RECIPE, "grid": epc.GRID, "best_effort": False, "project": "p", "limits": {},
                "grid_crs_wkt": epc.EE_CRS_WKT, "projection_check": [[0.0, 0.0, 0.0, 0.0]],
                "requests": [{"request_id": r.piece_id, "candidates": r.asset_id.split(",")} for r in real.itertuples()]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            path.write_text(json.dumps({**body, "request_sha256": epc.digest(body), "release_id": "rel"}))
            epc.load_released(path)  # every planned piece is a valid EE request


class TransportCrs(unittest.TestCase):
    def test_wkt_equals_epsg6933_and_parse_error_is_nonretryable(self):
        from pyproj import CRS, Transformer

        c = CRS.from_wkt(epc.EE_CRS_WKT)
        self.assertTrue(c.equals(CRS.from_epsg(6933), ignore_axis_order=True))
        self.assertNotIn('AUTHORITY["EPSG","6933"]', epc.EE_CRS_WKT)
        a, b = Transformer.from_crs(4326, 6933, always_xy=True), Transformer.from_crs(CRS.from_epsg(4326), c, always_xy=True)
        for p in ((0, 0), (45.0, 13.0), (-92.2, 38.4), (127.3, -34.8)):
            self.assertEqual(a.transform(*p), b.transform(*p))
        import ee_metadata as em
        with tempfile.TemporaryDirectory() as tmp:
            budget, calls = em.Budget(4, Path(tmp) / "l.jsonl"), []

            def fail():
                calls.append(1)
                raise RuntimeError("Projection: The CRS of a map projection could not be parsed.")
            with self.assertRaises(em.Stop):
                budget.call("p", fail)
            self.assertEqual(len(calls), 1)  # deterministic parse failure is not retried
