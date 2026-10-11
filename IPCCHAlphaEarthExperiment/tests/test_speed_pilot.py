"""D30 batch speed pilot (offline, fake ee): graph, pre-start body guard, ambiguous start HOLD, grouped status reads and
the 60 min cancel, multi-object download, exactly-once CSV ingest, deterministic selection."""

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ipcch_alphaearth import aemoments
from ipcch_alphaearth.errors import TechnicalError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_speed_pilot as bsp  # noqa: E402
import ee_batch_c as ebc  # noqa: E402
import ee_metadata  # noqa: E402
import ee_pilot_c as epc  # noqa: E402
import validate_speed_pilot as vsp  # noqa: E402

LOG = []


class Fx:
    """Chainable recording stand-in for any server-side ee object."""

    def __init__(self, tag, origin=None):
        self.tag, self.origin = tag, origin

    def __getattr__(self, name):
        def op(*args, **kwargs):
            LOG.append((self.tag, name, args, kwargs))
            return Fx(f"{self.tag}.{name}", self.origin)
        return op


class FakeEE:
    def __init__(self, body_len=1000, start_error=None, statuses=None):
        LOG.clear()
        self.mosaics, self.exports, self.started, self.cancelled, self.status_calls = [], [], [], [], []
        self.statuses = statuses or []
        outer = self

        class Image:
            def __new__(cls, ident=None):
                return Fx(f"img[{ident}]", ident)
            constant = staticmethod(lambda v: Fx(f"const[{v}]"))
            pixelArea = staticmethod(lambda: Fx("pixelArea"))
            cat = staticmethod(lambda items: Fx("stack"))

        class ImageCollection:
            def __init__(self, items):
                outer.mosaics.append([i.origin for i in items])

            def mosaic(self):
                return Fx("mosaic")

        class Feature(Fx):
            def __init__(self, geom, props=None):
                super().__init__("feature")
                self.props = props

            def geometry(self):
                return Fx("featgeom")

        class FeatureCollection:
            def __init__(self, items):
                self.items = items

            def map(self, fn):
                return FeatureCollection([fn(f) for f in self.items])

            def flatten(self):
                return FeatureCollection([f for c in self.items for f in c.items])

        class Task:
            def __init__(self, kw):
                self.kw, self.id = kw, f"T{len(outer.exports)}"

            def start(self):
                if start_error:
                    raise RuntimeError(start_error)
                outer.started.append(self.kw["description"])

        def export(**kw):
            outer.exports.append(kw)
            return Task(kw)

        class Data:
            setMaxRetries = staticmethod(lambda n: None)
            setDeadline = staticmethod(lambda ms: None)

            @staticmethod
            def getTaskStatus(ids):
                outer.status_calls.append(list(ids))
                return outer.statuses.pop(0) if outer.statuses else [{"id": i, "state": "RUNNING"} for i in ids]

            @staticmethod
            def cancelTask(i):
                outer.cancelled.append(i)

            getTaskList = staticmethod(lambda: [{"id": "T9", "description": "x"}])

        self.Image, self.ImageCollection, self.Feature, self.FeatureCollection = Image, ImageCollection, Feature, FeatureCollection
        self.Reducer = type("R", (), {"min": staticmethod(lambda: "min"), "sum": staticmethod(lambda: Fx("sum"))})
        self.Geometry = lambda *a, **k: Fx("geom")
        self.batch = type("B", (), {"Export": type("E", (), {"table": type("T", (), {"toCloudStorage": staticmethod(export)})})})
        self.serializer = type("S", (), {"toJSON": staticmethod(lambda obj: "x" * body_len)})
        self.data = Data()
        self.Initialize = lambda project: None


def manifest(tmp: Path) -> dict:
    groups = lambda t: [{"candidates": ["aa1", "zz9"], "max_pixels_band_px": 1000,
                         "items": [{"piece_id": f"{t}1|w000_000", "admin_code": 1 if t == "A" else 3, "role": "fresh", "expected_cells": 10.0,
                                    "bbox_cells": 7, "geometry": {"type": "Point", "coordinates": [0, 0]}},
                                   {"piece_id": f"{t}2|w000_000", "admin_code": 2 if t == "A" else 4, "role": "fresh", "expected_cells": 5.0,
                                    "bbox_cells": 7, "geometry": {"type": "Point", "coordinates": [0, 0]}}]},
                        {"candidates": ["mm5"], "max_pixels_band_px": 500,
                         "items": [{"piece_id": f"{t}3|w000_000", "admin_code": 5 if t == "A" else 6, "role": "reference", "expected_cells": 3.0,
                                    "bbox_cells": 3, "geometry": {"type": "Point", "coordinates": [0, 0]}}]}]
    body = {"kind": ebc.KIND, "recipe": epc.RECIPE, "grid": epc.GRID, "grid_crs_wkt": epc.EE_CRS_WKT, "best_effort": False,
            "projection_check": [[45.0, 13.0, 4341882.611290343, 1644631.1629230133]], "project": "p",
            "destination": {"type": "gcs", "bucket": "bkt", "prefix": "ipcch-alphaearth/speed-pilot-test"},
            "selectors": bsp.SELECTORS, "tasks": [{"name": "A", "groups": groups("A")}, {"name": "B", "groups": groups("B")}],
            "limits": {"tile_scale": 4, "deadline_ms": 120000, "max_body_bytes": 8_000_000, "cancel_after_s": 3600,
                       "status_read_interval_s": 60, "max_status_reads": 75, "max_cancels": 2, "max_gcs_reads": 20,
                       "max_ee_calls": 84, "max_pixels": 1000}}
    m = {**body, "request_sha256": epc.digest(body), "release_id": "rel"}
    path = tmp / "m.json"
    path.write_text(json.dumps(m))
    return ebc.load_released(path)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._pc = epc.projection_check
        epc.projection_check = lambda ee, pts: {str(k): [p[2], p[3]] for k, p in enumerate(pts)}
        self.req = manifest(self.tmp)
        self.out = self.tmp / "out"

    def tearDown(self):
        epc.projection_check = self._pc
        shutil.rmtree(self.tmp, ignore_errors=True)


class Graph(Base):
    def test_one_sums_reduceregion_per_item_same_composite_grid_constant_maxpixels_and_export(self):
        ee = FakeEE()
        s = ebc.submit(self.req, self.out, ee)
        self.assertEqual(ee.mosaics.count([f"{epc.COLLECTION}/zz9", f"{epc.COLLECTION}/aa1"]), 2)  # descending mosaic, per group per task
        red = [k for _t, n, _a, k in LOG if n == "reduceRegion"]
        self.assertEqual(len(red), 6)  # one per item (3 per task)
        self.assertEqual({(k["crs"], tuple(k["crsTransform"]), k["bestEffort"], k["tileScale"]) for k in red},
                         {(epc.EE_CRS_WKT, (10, 0, 0, 0, -10, 0), False, 4)})
        self.assertEqual(sorted({k["maxPixels"] for k in red}), [500, 1000])  # one constant per candidate group
        self.assertEqual([e["selectors"] for e in ee.exports], [bsp.SELECTORS] * 2)
        self.assertEqual([e["fileNamePrefix"] for e in ee.exports],
                         ["ipcch-alphaearth/speed-pilot-test/A/attempt-1/rows", "ipcch-alphaearth/speed-pilot-test/B/attempt-1/rows"])
        self.assertEqual({e["fileFormat"] for e in ee.exports}, {"CSV"})
        self.assertEqual(len(ee.started), 2)
        self.assertEqual((s["body_bytes_A"], s["body_bytes_B"]), (1000, 1000))
        states = [r["state"] for r in ebc.read_ledger(self.out)]
        self.assertEqual(states, ["INTENT", "SUBMITTED", "INTENT", "SUBMITTED"])  # intent recorded BEFORE each start

    def test_oversized_body_refuses_before_any_start_or_intent(self):
        ee = FakeEE(body_len=8_000_001)
        with self.assertRaisesRegex(ee_metadata.Stop, "exceeds"):
            ebc.submit(self.req, self.out, ee)
        self.assertEqual(ee.started, [])
        self.assertEqual(ebc.read_ledger(self.out), [])

    def test_ambiguous_start_holds_never_duplicates(self):
        ee = FakeEE(start_error="HTTPSConnectionPool: Read timed out.")
        with self.assertRaisesRegex(ee_metadata.Stop, "HOLD"):
            ebc.submit(self.req, self.out, ee)
        self.assertEqual([r["state"] for r in ebc.read_ledger(self.out)], ["INTENT", "AMBIGUOUS"])
        self.assertEqual(len(ee.exports), 1)  # task B never built into a start
        with self.assertRaisesRegex(ee_metadata.Stop, "no task is started twice"):
            ebc.submit(self.req, self.out, FakeEE())
        with self.assertRaisesRegex(ee_metadata.Stop, "AMBIGUOUS"):
            ebc.monitor(self.req, self.out, FakeEE())
        self.assertEqual([s["description"] for s in ebc.lookup(self.req, self.out, FakeEE())], [])  # read-only search

    def test_load_refuses_unreleased_or_duplicate_items(self):
        m = json.loads((self.tmp / "m.json").read_text())
        for bad in ({**m, "release_id": None},):
            p = self.tmp / "bad.json"
            p.write_text(json.dumps(bad))
            with self.assertRaises(SystemExit):
                ebc.load_released(p)
        body = {k: v for k, v in m.items() if k not in ("request_sha256", "release_id")}
        body["tasks"][1]["groups"][0]["items"][0]["piece_id"] = "A1|w000_000"
        p = self.tmp / "dup.json"
        p.write_text(json.dumps({**body, "request_sha256": epc.digest(body), "release_id": "r"}))
        with self.assertRaisesRegex(SystemExit, "more than once"):
            ebc.load_released(p)


class Monitor(Base):
    def submitted(self, epoch=1000.0):
        self.out.mkdir()
        for t, i in (("A", "T0"), ("B", "T1")):
            ebc.append_ledger(self.out, {"task": t, "state": "SUBMITTED", "task_id": i, "submitted_epoch_s": epoch})

    def test_grouped_reads_cancel_only_own_nonterminal_after_60_min(self):
        self.submitted()
        t = [1000.0]  # time advances only when monitor sleeps (60 s per grouped read)
        statuses = [[{"id": "T0", "state": "RUNNING"}, {"id": "T1", "state": "RUNNING"}]] * 59 + \
                   [[{"id": "T0", "state": "COMPLETED"}, {"id": "T1", "state": "RUNNING"}]] * 3 + \
                   [[{"id": "T0", "state": "COMPLETED"}, {"id": "T1", "state": "CANCELLED"}]]
        ee = FakeEE(statuses=list(statuses))
        r = ebc.monitor(self.req, self.out, ee, sleep=lambda s: t.__setitem__(0, t[0] + s), now=lambda: t[0])
        self.assertTrue(all(c == ["T0", "T1"] for c in ee.status_calls))  # one grouped read per interval
        self.assertEqual(ee.cancelled, ["T1"])  # only the own nonterminal task, once
        self.assertEqual(r["final_states"], {"T0": "COMPLETED", "T1": "CANCELLED"})
        cancel = [x for x in ebc.read_ledger(self.out) if x["state"] == "CANCEL_REQUESTED"]
        self.assertEqual(len(cancel), 1)
        self.assertGreaterEqual(cancel[0]["after_s"], 3600)
        self.assertEqual(len(ee.status_calls), 63)  # reads continue after the cancel until terminal

    def test_both_running_at_deadline_cancels_both_once(self):
        self.submitted()
        t = [1000.0]
        ee = FakeEE()
        ebc.monitor(self.req, self.out, ee, sleep=lambda s: t.__setitem__(0, t[0] + s), now=lambda: t[0])
        self.assertEqual(ee.cancelled, ["T0", "T1"])  # each own nonterminal task once, never more than max_cancels

    def test_read_cap_bounds_monitoring(self):
        self.submitted()
        clock = iter([1000 + k for k in range(500)])
        ee = FakeEE()
        r = ebc.monitor(self.req, self.out, ee, sleep=lambda s: None, now=lambda: next(clock))
        self.assertEqual(r["reads"], 75)
        self.assertEqual(len(ee.status_calls), 75)
        self.assertEqual(ee.cancelled, [])  # never reached 60 min on this clock


class Download(Base):
    def test_every_object_under_each_prefix_with_sha_and_read_cap(self):
        objs = {"ipcch-alphaearth/speed-pilot-test/A/attempt-1/": {"rowsA0.csv": b"a0", "rowsA1.csv": b"a1"},
                "ipcch-alphaearth/speed-pilot-test/B/attempt-1/": {"rows.csv": b"b0"}}

        class Req:
            def __init__(self, v):
                self.v = v

            def execute(self):
                return self.v

        class Objects:
            def list(self, bucket, prefix, pageToken=None):
                return Req({"items": [{"name": prefix + n, "generation": "1"} for n in objs.get(prefix, {})]})

            def get_media(self, bucket, object):
                pre, name = object.rsplit("/", 1)
                return Req(objs[pre + "/"][name])

        storage = type("St", (), {"objects": lambda self: Objects()})()
        self.out.mkdir()
        rows = ebc.download(self.req, self.out, storage)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["sha256"], hashlib.sha256(b"a0").hexdigest())
        self.assertEqual(json.loads((self.out / "downloads.json").read_text())["reads"], 5)
        with self.assertRaisesRegex(ee_metadata.Stop, "already downloaded"):
            ebc.download(self.req, self.out, storage)
        self.req["limits"]["max_gcs_reads"] = 2
        with self.assertRaisesRegex(ee_metadata.Stop, "read cap"):
            ebc.download(self.req, self.tmp / "o2", storage)


class Ingest(Base):
    def write(self, name, rows):
        cols = {c: [] for c in bsp.SELECTORS}
        for pid, admin, cands, area in rows:
            vals = {"admin_code": admin, "piece_id": pid, "asset_id": cands, "status": "complete", "valid_area": area,
                    "recipe": epc.RECIPE, "batch_id": name[0]}
            for b in aemoments.BANDS:
                vals[f"S_{b}"], vals[f"Q_{b}"] = 0.1 * (area or 0), 0.02 * (area or 0)
            for c in cols:
                cols[c].append(vals[c])
        p = self.tmp / name
        pd.DataFrame(cols).to_csv(p, index=False)
        return {"task": name[0], "local": str(p)}

    def rows(self, t):
        return [(f"{t}1|w000_000", 1, "aa1,zz9", 9.0), (f"{t}2|w000_000", 2, "aa1,zz9", 4.0), (f"{t}3|w000_000", 5, "mm5", 2.0)]

    def test_multi_object_exactly_once_and_refusals(self):
        a, b = self.rows("A"), self.rows("B")
        ok = [self.write("A0.csv", a[:2]), self.write("A1.csv", a[2:]), self.write("B0.csv", b)]
        self.assertEqual(len(vsp.read_rows(self.req, ok)), 6)
        with self.assertRaisesRegex(TechnicalError, "duplicated"):
            vsp.read_rows(self.req, ok + [self.write("A2.csv", a[:1])])
        with self.assertRaisesRegex(TechnicalError, "rows differ"):
            vsp.read_rows(self.req, ok[:1] + ok[2:])
        nul = [self.write("A3.csv", a[:2] + [(a[2][0], 5, "mm5", None)]), self.write("B1.csv", b)]
        with self.assertRaisesRegex(TechnicalError, "null/nonfinite"):
            vsp.read_rows(self.req, nul)


class Selection(unittest.TestCase):
    def test_systematic_balanced_deterministic_and_group_maxpixels(self):
        items = pd.DataFrame({"piece_id": [f"{i}|w000_000" for i in range(40)], "admin_code": range(40),
                              "cells": [float(i + 1) for i in range(40)], "asset_id": ["aa1"] * 20 + ["aa1,zz9"] * 20,
                              "bbox_cells": list(range(40))})
        s1, s2 = bsp.select_fresh(items, 100.0), bsp.select_fresh(items.sample(frac=1, random_state=3), 100.0)
        self.assertEqual(list(s1.piece_id), list(s2.piece_id))  # input order does not matter
        self.assertEqual(len(s1), round(100 / items.cells.mean()))
        self.assertEqual(sorted(set(s1.task)), ["A", "B"])
        groups = bsp.group_task(s1.assign(role="fresh"), {p: {"type": "Point"} for p in s1.piece_id})
        for g in groups:
            self.assertEqual(g["candidates"], sorted(g["candidates"]))
            self.assertEqual(g["max_pixels_band_px"], int(-(-129 * 1.1 * max(it["bbox_cells"] for it in g["items"]) // 1)))


if __name__ == "__main__":
    unittest.main()
