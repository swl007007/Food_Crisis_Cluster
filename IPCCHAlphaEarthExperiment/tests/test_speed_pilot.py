"""D30 batch speed pilot (offline, fake ee): graph, pre-start DAG guard, ambiguous/partial start HOLD with known-id
cancellation, per-ID counted status reads with cancel reserve and failure handoff, resumable multi-object download,
manifest/plan/task-state bound validation with exactly-once CSV ingest and fixed gates, deterministic selection."""

import gzip
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
NO_TRANSPORT = lambda req: "socket-timeout-transport"  # noqa: E731


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
    def __init__(self, body_len=1000, start_error_for=(), status=None, status_error=None):
        LOG.clear()
        self.mosaics, self.exports, self.started, self.cancelled, self.status_calls, self.inits = [], [], [], [], [], []
        self.status = status or (lambda i, n: "RUNNING")
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
                self.kw, self.id = kw, f"T{len(outer.exports) - 1}"

            def start(self):
                if any(self.kw["description"].endswith(f"_{t}_a1") for t in start_error_for):
                    raise RuntimeError("HTTPSConnectionPool: Read timed out.")
                outer.started.append(self.kw["description"])

        def export(**kw):
            outer.exports.append(kw)
            return Task(kw)

        class Data:
            setMaxRetries = staticmethod(lambda n: None)
            setDeadline = staticmethod(lambda ms: None)

            @staticmethod
            def getTaskStatus(i):
                outer.status_calls.append(i)
                if status_error:
                    raise RuntimeError(status_error)
                n = sum(1 for c in outer.status_calls if c == i)
                return [{"id": i, "state": outer.status(i, n)}]

            @staticmethod
            def cancelTask(i):
                outer.cancelled.append(i)

        self.Image, self.ImageCollection, self.Feature, self.FeatureCollection = Image, ImageCollection, Feature, FeatureCollection
        self.Reducer = type("R", (), {"min": staticmethod(lambda: "min"), "sum": staticmethod(lambda: Fx("sum"))})
        self.Geometry = lambda *a, **k: Fx("geom")
        self.batch = type("B", (), {"Export": type("E", (), {"table": type("T", (), {"toCloudStorage": staticmethod(export)})})})
        self.serializer = type("S", (), {"toJSON": staticmethod(lambda obj: "x" * body_len)})
        self.data = Data()
        self.Initialize = lambda project, http_transport=None: self.inits.append(http_transport)


def item(pid, admin, role="fresh", cells=10.0):
    return {"piece_id": pid, "admin_code": admin, "role": role, "expected_cells": cells, "bbox_cells": 7,
            "geometry": {"type": "Point", "coordinates": [0, 0]}}


def write_manifest(tmp: Path, plan_sha="p" * 64, pins=None, tasks=None, limits=None, release="rel") -> Path:
    tasks = tasks or [{"name": t, "groups": [{"candidates": ["aa1", "zz9"], "max_pixels_band_px": 1000,
                                              "items": [item(f"{t}1|w000_000", 1 if t == "A" else 3), item(f"{t}2|w000_000", 2 if t == "A" else 4, cells=5.0)]},
                                             {"candidates": ["mm5"], "max_pixels_band_px": 500,
                                              "items": [item(f"{t}3|w000_000", 5 if t == "A" else 6, role="reference", cells=3.0)]}]}
                      for t in ("A", "B")]
    lim = {"tile_scale": 4, "deadline_ms": 120000, "http_timeout_s": 60, "max_tasks": 2, "max_body_bytes": 8_000_000,
           "cancel_after_s": 3600, "status_read_interval_s": 120, "max_status_reads": 75, "max_cancels": 2,
           "max_failed_rounds": 3, "max_gcs_reads": 20, "max_ee_calls": 84, "max_pixels": 1000, **(limits or {})}
    body = {"kind": ebc.KIND, "recipe": epc.RECIPE, "grid": epc.GRID, "grid_crs_wkt": epc.EE_CRS_WKT, "best_effort": False,
            "projection_check": [[45.0, 13.0, 4341882.611290343, 1644631.1629230133]], "project": "p",
            "destination": {"type": "gcs", "bucket": "bkt", "prefix": "ipcch-alphaearth/speed-pilot-test"},
            "selectors": bsp.SELECTORS, "adaptive_plan_sha256": plan_sha,
            "remaining_workload": {"cells": 1e9, "adaptive_plan_sha256": plan_sha},
            "reference_pins": pins or {"interactive_responses": {}, "old_windows": {}}, "tasks": tasks, "limits": lim}
    path = tmp / "m.json"
    path.write_text(json.dumps({**body, "request_sha256": epc.digest(body), "release_id": release}))
    return path


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._pc = epc.projection_check
        epc.projection_check = lambda ee, pts: {str(k): [p[2], p[3]] for k, p in enumerate(pts)}
        self.req = ebc.load_released(write_manifest(self.tmp))
        self.out = self.tmp / "out"

    def tearDown(self):
        epc.projection_check = self._pc
        shutil.rmtree(self.tmp, ignore_errors=True)

    def submitted(self, epoch=1000.0, tasks=(("A", "T0"), ("B", "T1"))):
        self.out.mkdir(exist_ok=True)
        (self.out / "MANIFEST.sha256").write_text(self.req["request_sha256"] + "\n")
        for t, i in tasks:
            ebc.append_ledger(self.out, {"task": t, "state": "SUBMITTED", "task_id": i, "submitted_epoch_s": epoch})


class Submit(Base):
    def test_graph_one_sums_reduceregion_per_item_constant_group_maxpixels_and_timeout_transport(self):
        ee = FakeEE()
        s = ebc.submit(self.req, self.out, ee, make_transport=NO_TRANSPORT)
        self.assertEqual(ee.inits, ["socket-timeout-transport"])  # Initialize gets the socket-timeout transport
        self.assertEqual(ee.mosaics.count([f"{epc.COLLECTION}/zz9", f"{epc.COLLECTION}/aa1"]), 2)
        red = [k for _t, n, _a, k in LOG if n == "reduceRegion"]
        self.assertEqual(len(red), 6)
        self.assertEqual({(k["crs"], tuple(k["crsTransform"]), k["bestEffort"], k["tileScale"]) for k in red},
                         {(epc.EE_CRS_WKT, (10, 0, 0, 0, -10, 0), False, 4)})
        self.assertEqual(sorted({k["maxPixels"] for k in red}), [500, 1000])
        self.assertEqual([e["fileNamePrefix"] for e in ee.exports],
                         ["ipcch-alphaearth/speed-pilot-test/A/attempt-1/rows", "ipcch-alphaearth/speed-pilot-test/B/attempt-1/rows"])
        self.assertEqual({(e["fileFormat"], tuple(e["selectors"])) for e in ee.exports}, {("CSV", tuple(bsp.SELECTORS))})
        self.assertEqual((s["collection_dag_bytes_A"], s["collection_dag_bytes_B"]), (1000, 1000))
        self.assertEqual([r["state"] for r in ebc.read_ledger(self.out)], ["INTENT", "SUBMITTED", "INTENT", "SUBMITTED"])

    def test_oversized_dag_refuses_before_any_start_or_intent(self):
        ee = FakeEE(body_len=8_000_001)
        with self.assertRaisesRegex(ee_metadata.Stop, "exceeds"):
            ebc.submit(self.req, self.out, ee, make_transport=NO_TRANSPORT)
        self.assertEqual((ee.started, ebc.read_ledger(self.out)), ([], []))

    def test_partial_ambiguous_start_holds_and_known_id_is_still_cancelled_at_60_min(self):
        ee = FakeEE(start_error_for=("B",))
        with self.assertRaisesRegex(ee_metadata.Stop, "HOLD"):
            ebc.submit(self.req, self.out, ee, make_transport=NO_TRANSPORT)
        self.assertEqual([r["state"] for r in ebc.read_ledger(self.out)], ["INTENT", "SUBMITTED", "INTENT", "AMBIGUOUS"])
        with self.assertRaisesRegex(ee_metadata.Stop, "no task is started twice"):
            ebc.submit(self.req, self.out, FakeEE(), make_transport=NO_TRANSPORT)
        sub = next(r for r in ebc.read_ledger(self.out) if r["state"] == "SUBMITTED")
        t = [sub["submitted_epoch_s"]]
        mon = FakeEE()
        r = ebc.monitor(self.req, self.out, mon, sleep=lambda s: t.__setitem__(0, t[0] + s), now=lambda: t[0], make_transport=NO_TRANSPORT)
        self.assertEqual(set(mon.status_calls), {"T0"})  # the known id is monitored
        self.assertEqual(mon.cancelled, ["T0"])  # and cancelled at the cap
        self.assertEqual(r["ambiguous"], ["B"])
        self.assertEqual(ebc.read_ledger(self.out)[-1]["state"], "HOLD_HANDOFF")

    def test_loader_refuses_unreleased_too_many_tasks_missing_limits_duplicates(self):
        for kw in ({"release": None}, {"limits": {"max_tasks": 1}}):
            with self.assertRaises(SystemExit):
                ebc.load_released(write_manifest(self.tmp, **kw))
        m = json.loads(write_manifest(self.tmp).read_text())
        body = {k: v for k, v in m.items() if k not in ("request_sha256", "release_id")}
        del body["limits"]["http_timeout_s"]
        p = self.tmp / "x.json"
        p.write_text(json.dumps({**body, "request_sha256": epc.digest(body), "release_id": "r"}))
        with self.assertRaisesRegex(SystemExit, "lack"):
            ebc.load_released(p)


class Monitor(Base):
    def run_monitor(self, ee, **lim):
        self.req["limits"].update(lim)
        t = [1000.0]
        return ebc.monitor(self.req, self.out, ee, sleep=lambda s: t.__setitem__(0, t[0] + s), now=lambda: t[0],
                           make_transport=NO_TRANSPORT)

    def test_per_id_counted_reads_and_cancel_only_own_nonterminal_after_60_min(self):
        self.submitted()
        ee = FakeEE(status=lambda i, n: "COMPLETED" if (i == "T0" and n >= 10) or (i == "T1" and n >= 33) else "RUNNING")
        r = self.run_monitor(ee)
        self.assertTrue(all(isinstance(c, str) for c in ee.status_calls))  # one id per counted read (one HTTP GET)
        self.assertEqual(ee.cancelled, ["T1"])  # T0 already COMPLETED at the cap
        self.assertEqual(r["reads"], len(ee.status_calls))
        self.assertLessEqual(r["calls_used"], 84)
        cancel = [x for x in ebc.read_ledger(self.out) if x["state"] == "CANCEL_REQUESTED"]
        self.assertGreaterEqual(cancel[0]["after_s"], 3600)

    def test_unknown_state_counts_as_nonterminal_and_both_cancelled_once(self):
        self.submitted()
        ee = FakeEE(status=lambda i, n: "UNKNOWN" if i == "T1" else ("CANCELLED" if n > 31 else "RUNNING"))
        r = self.run_monitor(ee, max_status_reads=70)
        self.assertEqual(sorted(ee.cancelled), ["T0", "T1"])
        self.assertTrue(r["handoff"])  # T1 never terminal: budget handoff, recorded
        self.assertEqual(ebc.read_ledger(self.out)[-1]["state"], "HOLD_HANDOFF")

    def test_cancel_reserve_kept_when_call_budget_is_tight(self):
        self.submitted()
        ee = FakeEE()
        r = self.run_monitor(ee, max_ee_calls=11)
        self.assertEqual(sorted(ee.cancelled), ["T0", "T1"])  # reserved cancels still possible
        self.assertLessEqual(r["calls_used"], 11)
        self.assertIn("budget", r["handoff"])

    def test_failing_status_rounds_trigger_bounded_cancel_handoff(self):
        self.submitted()
        ee = FakeEE(status_error="HTTPSConnectionPool: Read timed out.")
        r = self.run_monitor(ee)
        self.assertEqual(len(ee.status_calls), 6)  # 3 failed rounds x 2 ids, then stop polling
        self.assertEqual(sorted(ee.cancelled), ["T0", "T1"])
        self.assertIn("failing", r["handoff"])


class Download(Base):
    objs = {"ipcch-alphaearth/speed-pilot-test/A/attempt-1/": {"rowsA0.csv": b"a0", "rowsA1.csv": b"a1"},
            "ipcch-alphaearth/speed-pilot-test/B/attempt-1/": {"rows.csv": b"b0"}}

    def storage(self, gets):
        objs = self.objs

        class Req:
            def __init__(self, v):
                self.v = v

            def execute(self):
                return self.v

        class Objects:
            def list(self, bucket, prefix, pageToken=None):
                return Req({"items": [{"name": prefix + n, "generation": "1"} for n in objs.get(prefix, {})]})

            def get_media(self, bucket, object):
                gets.append(object)
                pre, name = object.rsplit("/", 1)
                return Req(objs[pre + "/"][name])

        return type("St", (), {"objects": lambda self: Objects()})()

    def test_partial_failure_keeps_provenance_and_resume_skips_verified_objects(self):
        self.out.mkdir()
        gets = []
        self.req["limits"]["max_gcs_reads"] = 2  # list + first object, then the cap
        with self.assertRaisesRegex(ee_metadata.Stop, "read cap"):
            ebc.download(self.req, self.out, self.storage(gets))
        recs = [json.loads(x) for x in (self.out / "downloads.jsonl").read_text().splitlines()]
        self.assertEqual([r["object"].rsplit("/", 1)[1] for r in recs], ["rowsA0.csv"])
        self.assertEqual(recs[0]["sha256"], hashlib.sha256(b"a0").hexdigest())
        self.req["limits"]["max_gcs_reads"] = 20
        gets.clear()
        rows = ebc.download(self.req, self.out, self.storage(gets))
        self.assertEqual(len(rows), 3)
        self.assertNotIn("ipcch-alphaearth/speed-pilot-test/A/attempt-1/rowsA0.csv", gets)  # verified, not re-read
        self.assertEqual(len((self.out / "gcs_reads.log").read_text().splitlines()), 2 + 4)  # cumulative across runs
        (self.out / "downloads" / "A" / "rowsA0.csv").write_bytes(b"tampered")
        with self.assertRaisesRegex(ee_metadata.Stop, "altered"):
            ebc.download(self.req, self.out, self.storage([]))


class Validate(Base):
    def setUp(self):
        super().setUp()
        plan = pd.DataFrame({"admin_code": [1, 2, 5, 3, 4, 6], "piece_id": ["A1|w000_000", "A2|w000_000", "A3|w000_000", "B1|w000_000", "B2|w000_000", "B3|w000_000"],
                             "asset_id": ["aa1,zz9", "aa1,zz9", "mm5", "aa1,zz9", "aa1,zz9", "mm5"], "recipe": [epc.RECIPE] * 6,
                             "polygon_area_m2": [10.0, 10.0, 2.2, 10.0, 10.0, 10.0]})
        self.plan = self.tmp / "plan.csv.gz"
        with gzip.open(self.plan, "wt") as h:
            plan.to_csv(h, index=False)
        self.ref = self.tmp / "irun" / "out" / "responses" / "WA-REF-5.json"
        self.ref.parent.mkdir(parents=True)
        pins = {"interactive_responses": {"5": {"file": self.ref.name, "sha256": None}}, "old_windows": {}}
        self.write_ref(2.0)
        pins["interactive_responses"]["5"]["sha256"] = hashlib.sha256(self.ref.read_bytes()).hexdigest()
        self.mpath = write_manifest(self.tmp, plan_sha=hashlib.sha256(self.plan.read_bytes()).hexdigest(), pins=pins)
        self.req = ebc.load_released(self.mpath)

    def write_ref(self, area):
        sums = {"valid_area": area, **{f"S_{b}": 0.1 * area for b in aemoments.BANDS}, **{f"Q_{b}": 0.02 * area for b in aemoments.BANDS}}
        self.ref.write_text(json.dumps({"response": {"sums": sums}}))

    def finished_run(self, ref_area=2.0, final="COMPLETED", ts=(1, 2, 3)):
        self.submitted(epoch=0.0)
        with open(self.out / "status.jsonl", "w") as h:
            for i in ("T0", "T1"):
                h.write(json.dumps({"id": i, "status": {"id": i, "state": final, "creation_timestamp_ms": ts[0] * 1000,
                                                        "start_timestamp_ms": ts[1] * 1000, "update_timestamp_ms": ts[2] * 1000}}) + "\n")
        recs = []
        for t in ("A", "B"):
            rows = [(f"{t}1|w000_000", 1 if t == "A" else 3, "aa1,zz9", 9.0), (f"{t}2|w000_000", 2 if t == "A" else 4, "aa1,zz9", 4.0),
                    (f"{t}3|w000_000", 5 if t == "A" else 6, "mm5", ref_area if t == "A" else 2.0)]
            cols = {c: [] for c in bsp.SELECTORS}
            for pid, admin, cands, area in rows:
                vals = {"admin_code": admin, "piece_id": pid, "asset_id": cands, "status": "complete", "valid_area": area,
                        "recipe": epc.RECIPE, "batch_id": t, **{f"S_{b}": 0.1 * area for b in aemoments.BANDS},
                        **{f"Q_{b}": 0.02 * area for b in aemoments.BANDS}}
                for c in cols:
                    cols[c].append(vals[c])
            local = self.out / "downloads" / t / "rows.csv"
            local.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(cols).to_csv(local, index=False)
            recs.append({"task": t, "object": f"ipcch-alphaearth/speed-pilot-test/{t}/attempt-1/rows.csv", "generation": "1",
                         "sha256": hashlib.sha256(local.read_bytes()).hexdigest(), "local": str(local)})
        (self.out / "downloads.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))

    def main(self):
        return vsp.main(["--request", str(self.mpath), "--out", str(self.out), "--full-run", str(self.tmp),
                         "--interactive-run", str(self.tmp / "irun"), "--plan", str(self.plan)])

    def report(self):
        return json.loads((self.out / "validation.json").read_text())

    def test_success_passes_with_pinned_workload_and_exit_zero(self):
        self.finished_run()
        self.assertEqual(self.main(), 0)
        rep = self.report()
        self.assertEqual((rep["integrity"], rep["all_gates_pass"]), ("PASS", True))
        self.assertEqual(rep["timing"]["remaining_cells"], 1e9)  # from the manifest, not the command line

    def test_failed_gate_exits_nonzero_and_keeps_report(self):
        self.finished_run(ref_area=2.0 * (1 + 1e-6))
        self.assertEqual(self.main(), 1)
        self.assertFalse(self.report()["gates"]["5"]["pass"])

    def test_cancelled_or_nonmonotonic_task_is_not_a_pass(self):
        for kw in ({"final": "CANCELLED"}, {"ts": (3, 2, 4)}):
            shutil.rmtree(self.out, ignore_errors=True)
            self.finished_run(**kw)
            self.assertEqual(self.main(), 1)
            self.assertEqual(self.report()["integrity"], "FAIL")

    def test_plan_or_manifest_tamper_refused(self):
        self.finished_run()
        with gzip.open(self.plan, "at") as h:
            h.write("7,X|w000_000,aa1,x,1\n")
        self.assertEqual(self.main(), 1)
        self.assertIn("plan bytes", self.report()["error"])
        m = json.loads(self.mpath.read_text())
        m["tasks"][0]["groups"][0]["items"][0]["expected_cells"] *= 100  # without updating the hash
        self.mpath.write_text(json.dumps(m))
        with self.assertRaises(SystemExit):
            self.main()

    def test_rows_exactly_once_and_null_refused(self):
        self.finished_run()
        down = [json.loads(x) for x in (self.out / "downloads.jsonl").read_text().splitlines()]
        with self.assertRaisesRegex(TechnicalError, "duplicated"):
            vsp.read_rows(self.req, down + down[:1])
        with self.assertRaisesRegex(TechnicalError, "rows differ"):
            vsp.read_rows(self.req, down[:1])
        f = Path(down[0]["local"])
        df = pd.read_csv(f)
        df.loc[0, "Q_A10"] = None
        df.to_csv(f, index=False)
        with self.assertRaisesRegex(TechnicalError, "null/nonfinite"):
            vsp.read_rows(self.req, down)


class Selection(unittest.TestCase):
    def test_systematic_balanced_deterministic_and_group_maxpixels(self):
        items = pd.DataFrame({"piece_id": [f"{i}|w000_000" for i in range(40)], "admin_code": range(40),
                              "cells": [float(i + 1) for i in range(40)], "asset_id": ["aa1"] * 20 + ["aa1,zz9"] * 20,
                              "bbox_cells": list(range(40))})
        s1, s2 = bsp.select_fresh(items, 100.0), bsp.select_fresh(items.sample(frac=1, random_state=3), 100.0)
        self.assertEqual(list(s1.piece_id), list(s2.piece_id))
        self.assertEqual(len(s1), round(100 / items.cells.mean()))
        self.assertEqual(sorted(set(s1.task)), ["A", "B"])
        for g in bsp.group_task(s1.assign(role="fresh"), {p: {"type": "Point"} for p in s1.piece_id}):
            self.assertEqual(g["candidates"], sorted(g["candidates"]))
            self.assertEqual(g["max_pixels_band_px"], int(-(-129 * 1.1 * max(it["bbox_cells"] for it in g["items"]) // 1)))


if __name__ == "__main__":
    unittest.main()
