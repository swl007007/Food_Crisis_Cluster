"""Offline extraction contracts (no EE contact): piece binding, admin coverage, orientation-free area, finite mask,
pre-start concurrency."""

import json
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
import extract_alphaearth as xa  # noqa: E402


def export_rows(spec):
    cols = {"admin_code": [], "piece_id": [], "asset_id": [], "status": [], "valid_area": []}
    for b in aemoments.BANDS:
        cols[f"S_{b}"], cols[f"Q_{b}"] = [], []
    for admin, pid, asset, area in spec:
        cols["admin_code"].append(admin), cols["piece_id"].append(pid), cols["asset_id"].append(asset)
        cols["status"].append("complete"), cols["valid_area"].append(area)
        for b in aemoments.BANDS:
            cols[f"S_{b}"].append(0.1 * area), cols[f"Q_{b}"].append(0.02 * area)
    return pd.DataFrame(cols)


class PieceBindingAndCoverage(unittest.TestCase):
    plan = pd.DataFrame({"admin_code": [1, 2, 3], "piece_id": ["1|A", "2|A", None], "asset_id": ["A", "A", None]})
    polys = pd.DataFrame({"admin_code": [1, 2, 3], "polygon_area_m2": [100.0, 100.0, 50.0]})

    def test_wrong_asset_for_a_valid_piece_id_refused(self):
        bad = export_rows([(1, "1|A", "OTHER", 100.0), (2, "2|A", "A", 90.0)])
        with self.assertRaises(TechnicalError):
            extraction.validate_export([bad], self.plan)
        with self.assertRaises(TechnicalError):
            descriptors.descriptor_table(bad, self.plan, self.polys, load_contract())
        wrong_admin = export_rows([(2, "1|A", "A", 100.0), (2, "2|A", "A", 90.0)])
        with self.assertRaises(TechnicalError):
            extraction.validate_export([wrong_admin], self.plan)

    def test_complete_admin_set_including_zero_piece_admins(self):
        good = export_rows([(1, "1|A", "A", 100.0), (2, "2|A", "A", 70.0)])
        table, ledger = descriptors.descriptor_table(good, self.plan, self.polys, load_contract(), expected_admins=[1, 2, 3])
        self.assertEqual(ledger["admin_code"].tolist(), [1, 2, 3])
        self.assertEqual(ledger["route_reason"].tolist(), ["usable", "low_coverage", "no_coverage"])
        with self.assertRaises(TechnicalError):  # a declared geometry missing from the polygon table
            descriptors.descriptor_table(good, self.plan, self.polys.iloc[:2], load_contract())
        with self.assertRaises(TechnicalError):  # plan narrower than the declared geometry set
            descriptors.descriptor_table(good, self.plan, self.polys, load_contract(), expected_admins=[1, 2, 3, 4])


class OrientationFreeArea(unittest.TestCase):
    def test_hole_and_multipolygon_winding_invariance(self):
        from shapely.geometry import MultiPolygon, Polygon

        outer = [(0, 0), (2, 0), (2, 2), (0, 2)]
        hole = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]
        a = extraction.geodesic_area(Polygon(outer, [hole]))
        b = extraction.geodesic_area(Polygon(outer, [hole[::-1]]))
        c = extraction.geodesic_area(Polygon(outer[::-1], [hole]))
        full, inner = extraction.geodesic_area(Polygon(outer)), extraction.geodesic_area(Polygon(hole))
        self.assertAlmostEqual(a, b, places=3)
        self.assertAlmostEqual(a, c, places=3)
        self.assertAlmostEqual(a, full - inner, delta=1e-6 * full)
        sq1 = [(0, 0), (1, 0), (1, 1), (0, 1)]
        sq2 = [(3, 0), (4, 0), (4, 1), (3, 1)]
        aligned = extraction.geodesic_area(MultiPolygon([Polygon(sq1), Polygon(sq2)]))
        opposite = extraction.geodesic_area(MultiPolygon([Polygon(sq1), Polygon(sq2[::-1])]))
        self.assertAlmostEqual(aligned, opposite, places=3)
        self.assertAlmostEqual(aligned, extraction.geodesic_area(Polygon(sq1)) + extraction.geodesic_area(Polygon(sq2)), places=3)
        self.assertGreater(aligned, 0)

    def test_degenerate_sliver_is_numerical_zero_but_real_negatives_stop(self):
        from shapely.geometry import Polygon

        sliver = Polygon([(25.0, -18.0), (25.0 + 1e-9, -18.0), (25.0 + 2e-9, -18.0 + 1e-18), (25.0, -18.0)])
        self.assertGreaterEqual(extraction.geodesic_area(sliver), 0.0)
        self.assertLess(extraction.geodesic_area(sliver), extraction.SLIVER_TOLERANCE_M2)


class MockImage:
    """Numeric stand-in for the ee.Image algebra used by the mask: arrays shaped (bands, pixels)."""

    def __init__(self, values, mask=None):
        self.v = np.asarray(values, dtype=float)
        self.m = np.ones_like(self.v) if mask is None else np.asarray(mask, dtype=float)

    def mask(self):
        return MockImage(self.m)

    def abs(self):
        return MockImage(np.abs(self.v), self.m)

    def lt(self, x):
        with np.errstate(invalid="ignore"):
            return MockImage((self.v < x).astype(float), self.m)

    def reduce(self, _reducer):
        return MockImage(self.v.min(axis=0, keepdims=True), self.m.min(axis=0, keepdims=True))

    def And(self, other):
        return MockImage(((self.v != 0) & (other.v != 0)).astype(float))


class FiniteMask(unittest.TestCase):
    def test_nan_and_infinities_are_not_valid_area(self):
        vals = np.full((64, 5), 0.1)
        vals[3, 1] = np.inf
        vals[10, 2] = -np.inf
        vals[63, 3] = np.nan
        mask = np.ones((64, 5))
        mask[0, 4] = 0
        ee = types.SimpleNamespace(Reducer=types.SimpleNamespace(min=lambda: "min"))
        valid = xa.common_valid_mask(ee, MockImage(vals, mask))
        self.assertEqual(valid.v.ravel().tolist(), [1.0, 0.0, 0.0, 0.0, 0.0])


class FakeTask:
    def __init__(self, server, events, desc):
        self.server, self.events, self.id = server, events, f"T{len(server) + 1}"
        self.desc = desc

    def start(self):
        active = sum(s in xa.ACTIVE for s in self.server.values())
        self.events.append(("start", self.id, active))
        self.server[self.id] = "RUNNING"


class Concurrency(unittest.TestCase):
    def test_resumed_running_task_blocks_new_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pieces = [{"piece_id": f"{i}|A", "admin_code": i, "asset_id": "A", "geometry": {"type": "Point", "coordinates": [0, 0]}}
                      for i in range(4)]
            body = {"kind": "pilot", "collection": xa.COLLECTION, "year": 2017, "bands": xa.BANDS, "best_effort": False, "scale": "native",
                    "project": "p", "destination": {"type": "drive", "folder": "f"},
                    "limits": {"pieces_per_task": 2, "max_tasks": 10, "max_concurrent": 1, "max_pixels": 1e9},
                    "export_columns": extraction.EXPORT_COLUMNS, "pieces": pieces}
            req = {**body, "request_sha256": xa.canonical_digest(body), "release_id": "test-release"}
            (tmp / "req.json").write_text(json.dumps(req))
            server = {"T0": "RUNNING"}
            ledger = {"request_sha256": req["request_sha256"],
                      "tasks": [{"description": "old", "task_id": "T0", "state": "RUNNING", "piece_ids": ["0|A", "1|A"]}]}
            (tmp / "ledger.json").write_text(json.dumps(ledger))
            events = []

            def sleep(_):
                events.append(("sleep",))
                for k, v in server.items():  # time passes: every running task finishes
                    if v == "RUNNING":
                        server[k] = "COMPLETED"

            stub = lambda *a, **k: types.SimpleNamespace(map=lambda f: None)  # noqa: E731
            ee = types.SimpleNamespace(
                data=types.SimpleNamespace(getTaskList=lambda: [{"id": k, "state": v} for k, v in server.items()]),
                FeatureCollection=stub, Feature=lambda *a, **k: None, Geometry=lambda *a, **k: None,
                batch=types.SimpleNamespace(Export=types.SimpleNamespace(table=types.SimpleNamespace(
                    toDrive=lambda **kw: FakeTask(server, events, kw["description"])))))
            original = xa.reduce_piece_factory
            xa.reduce_piece_factory = lambda *a, **k: None
            try:
                xa.export(types.SimpleNamespace(request=str(tmp / "req.json"), ledger=str(tmp / "ledger.json")), ee=ee, sleep=sleep)
            finally:
                xa.reduce_piece_factory = original
            starts = [e for e in events if e[0] == "start"]
            self.assertEqual(len(starts), 1)  # pieces 0,1 belonged to the resumed task; only 2,3 are new
            self.assertEqual(events[0], ("sleep",))  # waited for the resumed RUNNING task before starting
            self.assertTrue(all(e[2] < 1 for e in starts))

    def test_unreleased_request_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            body = {"kind": "pilot", "collection": xa.COLLECTION, "year": 2017, "best_effort": False, "pieces": []}
            req = {**body, "request_sha256": xa.canonical_digest(body), "release_id": None}
            path = Path(tmp) / "r.json"
            path.write_text(json.dumps(req))
            with self.assertRaises(SystemExit):
                xa.load_released_request(path)


if __name__ == "__main__":
    unittest.main()


class BudgetClassification(unittest.TestCase):
    def run_failing(self, message):
        import ee_metadata as em

        with tempfile.TemporaryDirectory() as tmp:
            b = em.Budget(6, Path(tmp) / "log.jsonl")
            calls = []

            def fail():
                calls.append(1)
                raise RuntimeError(message)
            with self.assertRaises(em.Stop) as ctx:
                b.call("probe", fail)
            return str(ctx.exception), len(calls), b.used

    def test_too_many_pixels_is_nonretryable_not_fatal_service(self):
        msg, calls, used = self.run_failing("Image.reduceRegion: Too many pixels in the region. Found 66040811, but maxPixels allows only 2000000.")
        self.assertIn("non-retryable request-size rejection", msg)
        self.assertEqual((calls, used), (1, 1))

    def test_rate_limit_is_fatal_and_memory_retries_once(self):
        msg, calls, _ = self.run_failing("429 Too Many Requests")
        self.assertIn("fatal service error", msg)
        self.assertEqual(calls, 1)
        msg, calls, used = self.run_failing("User memory limit exceeded.")
        self.assertIn("failed twice", msg)
        self.assertEqual((calls, used), (2, 2))
