"""D30 production connection (offline, fake ee): durable per-piece outputs, hash-bound resume, batching, concurrency,
stop rules, cumulative call budget, and plan -> manifest -> run -> parse -> validate -> ingest -> descriptors."""

import gzip
import json
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_alphaearth import aemoments, descriptors, extraction, runinputs
from ipcch_alphaearth.artifacts import sha256_file, write_csv_gz, write_json
from ipcch_alphaearth.contract import config_identity, load_contract
from ipcch_alphaearth.errors import ContractError, TechnicalError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_c_outputs  # noqa: E402
import ee_metadata  # noqa: E402
import ee_pilot_c as epc  # noqa: E402

PROJ = [[45.0, 13.0, 4341882.611290343, 1644631.1629230133]]
LIMITS = {"tile_scale": 4, "deadline_ms": 120000, "max_high_level_calls": 1000, "concurrency": 1, "batch_size": 1,
          "max_consecutive_failures": 3, "max_failure_fraction": 0.5, "min_attempts_for_fraction": 4, "wall_budget_s": 600,
          "outer_timeout_s": 900, "max_invocations": 50}


def fake_response(r):
    """Deterministic per-piece sums from the piece id (so batching/concurrency must not change any value)."""
    seed = int.from_bytes(r["request_id"].encode()[-4:].rjust(4, b"0"), "big")
    rng = np.random.default_rng(seed)
    area = 0.9 * r["expected_cells"] * 100.0
    mu = rng.uniform(-0.2, 0.2, 64)
    sums = {"valid_area": area}
    for j, b in enumerate(aemoments.BANDS):
        sums[f"S_{b}"] = area * mu[j]
        sums[f"Q_{b}"] = area * (mu[j] ** 2 + 0.01)
    return {"sums": sums, "count": {"valid_cells": int(0.9 * r["expected_cells"])}, "area_by_rank": {"groups": [{"source_rank": 1, "sum": area}]}}


class FakeEE:
    """``Dictionary({...}).getInfo()`` evaluates the batch; ``fail`` = request ids whose batch raises ``error``."""

    def __init__(self, fail=(), error="Computation timed out.", script=None, init_delay=0.0, barrier_ids=()):
        """``script``: request id -> per-attempt list of (error message or None for success, delay seconds)."""
        self.fail, self.error, self.calls, self.lock = set(fail), error, [], threading.Lock()
        self.script, self.init_delay, self.starts = {k: list(v) for k, v in (script or {}).items()}, init_delay, []
        # first attempts of ``barrier_ids`` wait for each other: a deterministic all-in-flight start
        self.barrier = threading.Barrier(len(barrier_ids)) if barrier_ids else None
        self.barrier_ids, self.barrier_done = set(barrier_ids), set()
        outer = self

        class Dictionary:
            def __init__(self, d):
                self.d = d

            def getInfo(self):
                import time as _t

                ids = sorted(r["request_id"] for r in self.d.values())
                with outer.lock:
                    outer.calls.append(ids)
                    outer.starts.append((ids[0], _t.time()))
                    step = outer.script[ids[0]].pop(0) if ids[0] in outer.script else None
                    wait = ids[0] in outer.barrier_ids and ids[0] not in outer.barrier_done
                    outer.barrier_done.add(ids[0])
                if wait:
                    outer.barrier.wait(timeout=10)
                if step is not None:
                    _t.sleep(step[1])
                    if step[0]:
                        raise RuntimeError(step[0])
                if any(r["request_id"] in outer.fail for r in self.d.values()):
                    raise RuntimeError(outer.error)
                return {k: fake_response(r) for k, r in self.d.items()}

        class Data:
            def setMaxRetries(self, n):
                pass

            def setDeadline(self, ms):
                pass

        self.Dictionary, self.data = Dictionary, Data()

    def Initialize(self, project):
        import time as _t

        _t.sleep(self.init_delay)


class Patched(unittest.TestCase):
    def setUp(self):
        self._orig = (epc.reduction_obj, epc.projection_check)
        epc.reduction_obj = lambda ee, r, limits: r  # the fake evaluates the request itself
        epc.projection_check = lambda ee, pts: {str(k): [p[2], p[3]] for k, p in enumerate(pts)}
        self.tmp = Path(tempfile.mkdtemp())
        self.contract = load_contract()

    def tearDown(self):
        epc.reduction_obj, epc.projection_check = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def plan(self, max_cells=400_000):
        import geopandas as gpd
        from shapely.geometry import box

        g = gpd.GeoDataFrame({"admin_code": [11, 12]}, geometry=[box(40.0, 5.0, 40.25, 5.12), box(40.30, 5.0, 40.36, 5.05)],
                             crs="EPSG:4326")
        fps = {"aa1": box(39.9, 4.9, 40.10, 5.3), "bb2": box(40.05, 4.9, 40.5, 5.3)}
        return extraction.plan_c_requests(g, fps, self.contract, max_cells=max_cells)

    def manifest(self, limits=None, select=None, name="production", release="rel-test"):
        plan, geoms = self.plan()
        plan_path = self.tmp / "plan.csv.gz"
        if not plan_path.exists():
            write_csv_gz(plan_path, plan)
        d = self.tmp / "manifest"
        m = extraction.write_production_manifest(plan, geoms, d, self.contract, limits={**LIMITS, **(limits or {})},
                                                 grid_crs_wkt=epc.EE_CRS_WKT, projection_check=PROJ, project="p",
                                                 provenance={"plan_sha256": sha256_file(plan_path)}, select=select, name=name)
        path = d / f"{name}_manifest.json"
        if release:
            path.write_text(json.dumps({**m, "release_id": release}))
        return plan, path

    def run_once(self, path, out, ee, cap=None):
        req = epc.load_released(path)
        if cap is not None:
            req["limits"]["max_high_level_calls"] = cap
        budget = ee_metadata.Budget(int(req["limits"]["max_high_level_calls"]), Path(out) / "request_log.jsonl")
        Path(out).mkdir(parents=True, exist_ok=True)
        return req, epc.run_production(req, Path(out), ee, budget)

    def outputs(self, out):
        return {p.name: p.read_bytes() for p in sorted((Path(out) / "pieces").glob("*.json"))}


class Runner(Patched):
    def test_manifest_is_hash_bound_and_released_copy_loads(self):
        plan, path = self.manifest()
        req = epc.load_released(path)
        self.assertEqual(len(req["requests"]), int(plan["piece_id"].notna().sum()))
        self.assertGreater(len(req["requests"]), 6)
        for r in req["requests"]:
            self.assertLessEqual(r["max_pixels_band_px"], req["limits"]["max_pixels"])
            self.assertGreaterEqual(r["max_pixels_band_px"], 129 * r["bbox_cells"])  # never below the window's lattice box
            self.assertGreaterEqual(r["bbox_cells"], r["expected_cells"] * (1 - 1e-3))
        pieces = path.parent / req["pieces_file"]
        raw = gzip.decompress(pieces.read_bytes()).replace(b'"aa1"', b'"aa0"', 1)
        pieces.write_bytes(gzip.compress(raw))
        with self.assertRaises(SystemExit):
            epc.load_released(path)
        _, unreleased = self.manifest(name="draft", release=None)
        with self.assertRaises(SystemExit):
            epc.load_released(unreleased)

    def test_sliver_piece_cap_covers_whole_cells(self):
        from shapely.geometry import box

        plan, geoms = self.plan()
        k = int(plan.index[plan["piece_id"].notna()][0])
        plan.loc[k, "window_area_m2"] = 10.0  # a 0.1-cell sliver window
        geoms[k] = box(40.0, 5.0, 40.00003, 5.00003)
        m = extraction.write_production_manifest(plan, geoms, self.tmp / "sliver", self.contract, limits=LIMITS,
                                                 grid_crs_wkt=epc.EE_CRS_WKT, projection_check=PROJ, project="p",
                                                 provenance={"plan_sha256": "x"})
        piece = json.loads(gzip.decompress((self.tmp / "sliver" / m["pieces_file"]).read_bytes()).splitlines()[0])
        self.assertGreaterEqual(piece["max_pixels_band_px"], 129)

    def test_interruption_then_resume_completes_without_rewriting_done_pieces(self):
        _, path = self.manifest()
        out = self.tmp / "out"
        req = epc.load_released(path)
        first = req["requests"][3]["request_id"]
        _, s1 = self.run_once(path, out, FakeEE(fail={first}, error="Computation timed out."))
        self.assertFalse(s1["complete"])
        self.assertEqual(s1["remaining"], 1)
        done = self.outputs(out)
        _, s2 = self.run_once(path, out, FakeEE())
        self.assertTrue(s2["complete"])
        self.assertEqual(s2["pending_at_start"], 1)
        after = self.outputs(out)
        for name, data in done.items():
            self.assertEqual(after[name], data)  # never rewritten
        ledger = [json.loads(x) for x in (out / "ledger.jsonl").read_text().splitlines()]
        self.assertEqual([x["status"] for x in ledger if x["piece_id"] == first], ["failed", "ok"])

    def test_corrupt_or_foreign_outputs_stop_and_are_not_overwritten(self):
        _, path = self.manifest()
        out = self.tmp / "out"
        req, _ = self.run_once(path, out, FakeEE())
        target = epc.piece_file(out, req["requests"][0]["request_id"])
        target.write_text("{truncated")
        with self.assertRaises(ee_metadata.Stop):
            self.run_once(path, out, FakeEE())
        self.assertEqual(target.read_text(), "{truncated")
        rec = json.loads(epc.piece_file(out, req["requests"][1]["request_id"]).read_text())
        rec = {**rec, "piece_id": req["requests"][0]["request_id"], "admin_code": req["requests"][0]["admin_code"],
               "candidates": req["requests"][0]["candidates"], "manifest_sha256": "0" * 64}
        target.write_text(json.dumps(rec))  # an output copied in from a different manifest
        with self.assertRaises(ee_metadata.Stop):
            self.run_once(path, out, FakeEE())
        _, other = self.manifest(limits={"batch_size": 2}, name="other")
        with self.assertRaises(SystemExit):
            self.run_once(other, out, FakeEE())  # output dir pinned to a different manifest

    def test_batching_and_concurrency_give_identical_outputs_with_fewer_calls(self):
        _, p1 = self.manifest()
        out1 = self.tmp / "b1"
        req, s1 = self.run_once(p1, out1, FakeEE())
        _, p3 = self.manifest(limits={"batch_size": 3, "concurrency": 4}, name="b3")
        out3 = self.tmp / "b3"
        ee3 = FakeEE()
        req3, s3 = self.run_once(p3, out3, ee3)
        self.assertTrue(s1["complete"] and s3["complete"])
        n = len(req["requests"])
        self.assertEqual(s1["calls_used"], n)
        self.assertEqual(s3["calls_used"], -(-n // 3))
        self.assertEqual(sorted(x for c in ee3.calls for x in c), sorted(r["request_id"] for r in req["requests"]))  # each once
        for r in req["requests"]:
            a = json.loads(epc.piece_file(out1, r["request_id"]).read_text())
            b = json.loads(epc.piece_file(out3, r["request_id"]).read_text())
            self.assertEqual(a["response"], b["response"])
            self.assertEqual(b["batch_size"] in (1, 2, 3), True)
        lines = (out3 / "ledger.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), n)  # ledger lines intact under concurrency
        self.assertTrue(all(json.loads(x)["status"] == "ok" for x in lines))

    def test_stop_rules(self):
        _, path = self.manifest(limits={"max_consecutive_failures": 2, "max_failure_fraction": 1.0})
        req = epc.load_released(path)
        ids = [r["request_id"] for r in req["requests"]]
        _, s = self.run_once(path, self.tmp / "consec", FakeEE(fail=set(ids)))
        self.assertIn("consecutive", s["stopped"])
        self.assertEqual(s["attempted"], 2)
        _, path_f = self.manifest(limits={"max_consecutive_failures": 99, "max_failure_fraction": 0.25,
                                          "min_attempts_for_fraction": 4}, name="frac")
        alt = set(ids[::2])
        _, s = self.run_once(path_f, self.tmp / "frac", FakeEE(fail=alt))
        self.assertIn("fraction", s["stopped"])
        _, s = self.run_once(path, self.tmp / "fatal", FakeEE(fail={ids[0]}, error="429 Too Many Requests"))
        self.assertTrue(s["stopped"].startswith("fatal"))
        self.assertEqual(s["attempted"], 1)
        _, s = self.run_once(path, self.tmp / "px", FakeEE(fail={ids[0]}, error="Too many pixels in the region."))
        self.assertEqual(s["failed"], 1)  # non-retryable: counted once, and the released rule stops the invocation
        self.assertTrue(s["stopped"].startswith("non-retryable request-size rejection"))
        self.assertEqual(s["done"], 0)

    def test_call_budget_is_cumulative_across_resumes(self):
        _, path = self.manifest(limits={"max_high_level_calls": 5})
        out = self.tmp / "cum"
        req = epc.load_released(path)
        self.assertEqual(epc.main_production(req, out, ee=FakeEE()), 0)  # init + projection + 3 pieces
        req = epc.load_released(path)
        self.assertEqual(epc.main_production(req, out, ee=FakeEE()), 0)  # cap already spent: no new call
        runs = [json.loads(x) for x in (out / "run_summaries.jsonl").read_text().splitlines()]
        self.assertEqual(runs[0]["calls_used"], 5)
        self.assertEqual(runs[0]["done"], 3)
        self.assertEqual(runs[1]["calls_used"], 5)
        self.assertIn("cap", runs[1]["stopped"])
        self.assertEqual(len((out / "call_reservations.log").read_text().splitlines()), 5)
        self.assertEqual(len(list((out / "pieces").glob("*.json"))), 3)


class Connection(Patched):
    def prepared_run(self):
        run = self.tmp / "run"
        (run / "prepared").mkdir(parents=True)
        write_json(run / "prepared" / "prepared-manifest.json", {"stage": "prepared", "artifacts_sha256": {}})
        for h in self.contract["calendar"]["horizons_months"]:
            write_csv_gz(run / "prepared" / f"keys_h{h:02d}.csv.gz", pd.DataFrame({"admin_code": [11, 12, 11]}))
        write_json(run / "prepared" / "COMPLETE.json", {"stage": "prepared"})
        return run

    def complete_outputs(self):
        _, path = self.manifest(limits={"batch_size": 2, "concurrency": 2})
        out = self.tmp / "out"
        _, s = self.run_once(path, out, FakeEE())
        self.assertTrue(s["complete"])
        return path, out

    def test_end_to_end_to_descriptor_stage_with_fresh_bindings(self):
        path, out = self.complete_outputs()
        run = self.prepared_run()
        rec = descriptors.ingest_extraction(run, self.contract, self.tmp / "plan.csv.gz", path, out)
        self.assertEqual(rec["admins"], 2)
        self.assertEqual(rec["status_counts"], {"complete": rec["pieces"]})
        done = descriptors.attach_descriptors(run, self.contract)
        self.assertEqual(done["admins"], 2)
        self.assertEqual(done["config_sha256"], config_identity())
        table = pd.read_csv(run / "descriptors" / "descriptor_table.csv.gz")
        self.assertEqual(table["admin_code"].tolist(), [11, 12])
        ledger = pd.read_csv(run / "descriptors" / "coverage_ledger.csv.gz")
        self.assertTrue(np.allclose(ledger["coverage"], 0.9, atol=1e-3))  # whole frozen-area denominator
        self.assertTrue(ledger["usable"].all())
        with self.assertRaises(ContractError):
            descriptors.ingest_extraction(run, self.contract, self.tmp / "plan.csv.gz", path, out)  # immutable
        with self.assertRaises(ContractError):
            descriptors.attach_descriptors(run, self.contract)
        # stale bindings refused before any feature file is read
        rec = json.loads((run / "descriptors" / "COMPLETE.json").read_text())
        (run / "descriptors" / "COMPLETE.json").write_text(json.dumps({**rec, "config_sha256": {"x": "y"}}))
        with self.assertRaisesRegex(ContractError, "different configuration"):
            runinputs.load_run_inputs(run, self.contract)
        (run / "descriptors" / "COMPLETE.json").write_text(json.dumps(rec))
        (run / "prepared" / "prepared-manifest.json").write_text(json.dumps({"stage": "prepared", "changed": 1}))
        with self.assertRaisesRegex(ContractError, "prepared stage"):
            runinputs.load_run_inputs(run, self.contract)

    def test_incomplete_or_sample_or_tampered_extraction_refused(self):
        path, out = self.complete_outputs()
        req = epc.load_released(path)
        run = self.prepared_run()
        victim = epc.piece_file(out, req["requests"][2]["request_id"])
        saved = victim.read_bytes()
        victim.unlink()
        with self.assertRaisesRegex(TechnicalError, "1 missing"):
            descriptors.ingest_extraction(run, self.contract, self.tmp / "plan.csv.gz", path, out)
        rec = json.loads(saved)
        rec["response"]["sums"]["S_A05"] = rec["response"]["sums"]["S_A05"] * 2  # response altered after the run
        victim.write_text(json.dumps(rec))
        with self.assertRaisesRegex(TechnicalError, "binding"):
            descriptors.ingest_extraction(run, self.contract, self.tmp / "plan.csv.gz", path, out)
        victim.write_bytes(saved)
        _, sample = self.manifest(select=[req["requests"][0]["request_id"]], name="sample")
        with self.assertRaisesRegex(ContractError, "sample"):
            descriptors.ingest_extraction(run, self.contract, self.tmp / "plan.csv.gz", sample, out)
        self.assertFalse((run / "extraction" / "COMPLETE.json").exists())
        with self.assertRaisesRegex(ContractError, "no COMPLETE extraction"):
            descriptors.attach_descriptors(run, self.contract)

    def test_failed_or_null_pieces_refused_before_complete_and_corrected_ingest_proceeds(self):
        path, out = self.complete_outputs()
        req = epc.load_released(path)
        run = self.prepared_run()
        target = epc.piece_file(out, req["requests"][0]["request_id"])
        good = target.read_bytes()
        for value in (None, float("nan")):
            rec = json.loads(good)
            rec["response"]["sums"]["Q_A10"] = value
            rec["response_sha256"] = epc.digest(rec["response"])  # an honest null/nonfinite returned by the service
            target.write_text(json.dumps(rec))
            with self.assertRaisesRegex(TechnicalError, "not numerically complete"):
                descriptors.ingest_extraction(run, self.contract, self.tmp / "plan.csv.gz", path, out)
            self.assertFalse((run / "extraction").exists())  # nothing published, nothing immutable
        attempts = sorted((run / "extraction_attempts").glob("refused-*.json"))
        self.assertEqual(len(attempts), 2)
        ctx = json.loads(attempts[0].read_text())
        self.assertEqual([p["piece_id"] for p in ctx["failed_pieces"]], [req["requests"][0]["request_id"]])
        self.assertEqual(ctx["failed_pieces"][0]["status"], "null_result")
        target.write_bytes(good)  # the piece rerun under a release
        descriptors.ingest_extraction(run, self.contract, self.tmp / "plan.csv.gz", path, out)
        self.assertTrue(descriptors.attach_descriptors(run, self.contract)["usable"] == 2)

    def test_valid_all_zero_area_is_genuine_no_coverage(self):
        path, out = self.complete_outputs()
        req = epc.load_released(path)
        run = self.prepared_run()
        for r in req["requests"]:
            if r["admin_code"] != 12:
                continue
            f = epc.piece_file(out, r["request_id"])
            rec = json.loads(f.read_text())
            rec["response"]["sums"] = {k: 0.0 for k in rec["response"]["sums"]}
            rec["response"]["count"] = {"valid_cells": 0}
            rec["response_sha256"] = epc.digest(rec["response"])
            f.write_text(json.dumps(rec))
        descriptors.ingest_extraction(run, self.contract, self.tmp / "plan.csv.gz", path, out)
        descriptors.attach_descriptors(run, self.contract)
        ledger = pd.read_csv(run / "descriptors" / "coverage_ledger.csv.gz").set_index("admin_code")
        self.assertEqual(ledger.loc[12, "route_reason"], "no_coverage")
        self.assertEqual(ledger.loc[11, "route_reason"], "usable")

    def test_plan_bytes_must_equal_the_manifest_pin(self):
        path, out = self.complete_outputs()
        run = self.prepared_run()
        plan = pd.read_csv(self.tmp / "plan.csv.gz", float_precision="round_trip", dtype={"piece_id": str, "asset_id": str})
        forged = self.tmp / "plan_doubled.csv.gz"
        write_csv_gz(forged, plan.assign(polygon_area_m2=plan["polygon_area_m2"] * 2))  # same keys, other denominators
        with self.assertRaisesRegex(ContractError, "plan bytes differ"):
            descriptors.ingest_extraction(run, self.contract, forged, path, out)
        self.assertFalse((run / "extraction").exists())
        self.assertEqual(len(list((run / "extraction_attempts").glob("refused-*.json"))), 1)
        rec = descriptors.ingest_extraction(run, self.contract, self.tmp / "plan.csv.gz", path, out)
        self.assertEqual(rec["plan_sha256"], sha256_file(self.tmp / "plan.csv.gz"))
        self.assertEqual(sha256_file(run / "extraction" / "plan.csv.gz"), rec["plan_sha256"])
        ledger_cov = descriptors.attach_descriptors(run, self.contract)
        self.assertEqual(ledger_cov["usable"], 2)
        cov = pd.read_csv(run / "descriptors" / "coverage_ledger.csv.gz")["coverage"]
        self.assertTrue(np.allclose(cov, 0.9, atol=1e-3))


class SharedStop(Patched):
    def test_published_fatal_blocks_new_retries_in_other_workers(self):
        _, path = self.manifest(limits={"concurrency": 2, "max_consecutive_failures": 99, "max_failure_fraction": 1.0})
        ids = [r["request_id"] for r in epc.load_released(path)["requests"]]
        ee = FakeEE(script={ids[0]: [("429 Too Many Requests", 0.0)], ids[1]: [("Computation timed out.", 0.2), (None, 0.0)]},
                    barrier_ids=ids[:2])  # both first attempts in flight before the fatal is raised
        _, s = self.run_once(path, self.tmp / "fatal2", ee)
        submitted = [c[0] for c in ee.calls]
        self.assertEqual(sorted(submitted[:2]), sorted(ids[:2]))
        self.assertEqual(submitted.count(ids[1]), 1)  # no retry of piece 1 after the published fatal
        self.assertEqual(len(submitted), 2)  # and no new piece started
        self.assertTrue(s["stopped"].startswith("fatal"))
        self.assertFalse(s["complete"])
        led = [json.loads(x) for x in (self.tmp / "fatal2" / "ledger.jsonl").read_text().splitlines()]
        self.assertEqual({x["piece_id"]: x["status"] for x in led}[ids[1]], "halted")

    def test_wall_budget_enforced_at_retry_reservation(self):
        _, path = self.manifest(limits={"wall_budget_s": 0.03})
        ids = [r["request_id"] for r in epc.load_released(path)["requests"]]
        ee = FakeEE(script={ids[0]: [("Computation timed out.", 0.06), (None, 0.0)]})
        _, s = self.run_once(path, self.tmp / "wall", ee)
        self.assertEqual([c[0] for c in ee.calls], [ids[0]])  # the retry is not sent after the budget
        self.assertIn("wall budget", s["stopped"])
        self.assertFalse(s["complete"])
        self.assertEqual(s["done"], 0)

    def test_invocation_clock_starts_before_init_and_reservations_stay_cumulative(self):
        _, path = self.manifest(limits={"wall_budget_s": 0.05, "max_high_level_calls": 50})
        out = self.tmp / "slowinit"
        ee = FakeEE(init_delay=0.08)
        self.assertEqual(epc.main_production(epc.load_released(path), out, ee=ee), 0)
        self.assertEqual(ee.calls, [])  # init + projection used the budget; no reduction started after it
        summary = json.loads((out / "run_summaries.jsonl").read_text().splitlines()[-1])
        self.assertIn("wall budget", summary["stopped"])
        self.assertEqual(summary["calls_used"], 1)  # init only; the projection check was halted
        self.assertEqual(len((out / "call_reservations.log").read_text().splitlines()), 1)

    def test_production_manifest_must_state_outer_timeout_above_wall_budget(self):
        for bad in ({"outer_timeout_s": 600}, {"outer_timeout_s": None}):
            lim = {**LIMITS, **bad}
            if lim["outer_timeout_s"] is None:
                del lim["outer_timeout_s"]
            plan, geoms = self.plan()
            d = self.tmp / f"ot{len(list(self.tmp.glob('ot*')))}"
            m = extraction.write_production_manifest(plan, geoms, d, self.contract, limits=lim, grid_crs_wkt=epc.EE_CRS_WKT,
                                                     projection_check=PROJ, project="p", provenance={"plan_sha256": "x"})
            (d / "production_manifest.json").write_text(json.dumps({**m, "release_id": "r"}))
            with self.assertRaisesRegex(SystemExit, "outer_timeout_s"):
                epc.load_released(d / "production_manifest.json")


class EnvelopeGuards(Patched):
    def test_null_or_nonfinite_response_stays_pending_and_is_retried_on_resume(self):
        _, path = self.manifest(limits={"max_consecutive_failures": 99, "max_failure_fraction": 1.0})
        req = epc.load_released(path)
        victim = req["requests"][1]["request_id"]
        orig = fake_response

        def nulled(r):
            resp = orig(r)
            if r["request_id"] == victim:
                resp["sums"]["Q_A10"] = None
            return resp

        globals()["fake_response"] = nulled
        try:
            _, s1 = self.run_once(path, self.tmp / "nul", FakeEE())
        finally:
            globals()["fake_response"] = orig
        self.assertFalse(epc.piece_file(self.tmp / "nul", victim).exists())  # never a done output
        self.assertEqual(len(list((self.tmp / "nul" / "failed_responses").glob("*.json"))), 1)  # evidence kept
        self.assertEqual(s1["failed"], 1)
        self.assertFalse(s1["complete"])
        _, s2 = self.run_once(path, self.tmp / "nul", FakeEE())
        self.assertEqual(s2["pending_at_start"], 1)
        self.assertTrue(s2["complete"])
        rep = check_c_outputs.check(path, self.tmp / "nul")
        self.assertTrue(rep["complete"])
        self.assertEqual(rep["ledger_statuses"], {"ok": len(req["requests"]), "null_result": 1})

    def test_invocation_cap_counts_started_invocations_and_full_manifest_must_state_it(self):
        _, path = self.manifest(limits={"max_invocations": 2, "max_high_level_calls": 4})
        out = self.tmp / "inv"
        for _ in range(2):
            self.assertEqual(epc.main_production(epc.load_released(path), out, ee=FakeEE()), 0)
        with self.assertRaisesRegex(SystemExit, "invocation cap"):
            epc.main_production(epc.load_released(path), out, ee=FakeEE())
        self.assertEqual(len((out / "invocations.log").read_text().splitlines()), 2)
        plan, geoms = self.plan()
        lim = {k: v for k, v in LIMITS.items() if k != "max_invocations"}
        m = extraction.write_production_manifest(plan, geoms, self.tmp / "nocap", self.contract, limits=lim, grid_crs_wkt=epc.EE_CRS_WKT,
                                                 projection_check=PROJ, project="p", provenance={"plan_sha256": "x"})
        (self.tmp / "nocap" / "production_manifest.json").write_text(json.dumps({**m, "release_id": "r"}))
        with self.assertRaisesRegex(SystemExit, "max_invocations"):
            epc.load_released(self.tmp / "nocap" / "production_manifest.json")  # full (non-sample) manifest without the cap
        _, sample = self.manifest(select=[epc.load_released(path)["requests"][0]["request_id"]], name="smp")
        self.assertEqual(len(epc.load_released(sample)["requests"]), 1)  # samples may omit it

    def test_checker_reports_progress_and_flags_violations(self):
        _, path = self.manifest(limits={"max_invocations": 3})
        out = self.tmp / "chk"
        req = epc.load_released(path)
        first = req["requests"][0]["request_id"]
        _, _ = self.run_once(path, out, FakeEE(fail={first}))
        rep = check_c_outputs.check(path, out)
        self.assertEqual((rep["done"], rep["pending"], rep["n_violations"], rep["complete"]), (len(req["requests"]) - 1, 1, 0, False))
        f = epc.piece_file(out, req["requests"][2]["request_id"])
        rec = json.loads(f.read_text())
        rec["response"]["count"]["valid_cells"] = 10 ** 12
        rec["response_sha256"] = epc.digest(rec["response"])
        f.write_text(json.dumps(rec))
        rep = check_c_outputs.check(path, out)
        self.assertEqual(rep["n_violations"], 1)
        self.assertEqual(check_c_outputs.main(["--request", str(path), "--out", str(out)]), 1)


class ConcurrencyLimitStop(Patched):
    def test_concurrent_aggregation_rejection_is_a_shared_fatal_stop_without_retry(self):
        _, path = self.manifest(limits={"concurrency": 3, "max_consecutive_failures": 99, "max_failure_fraction": 1.0})
        ids = [r["request_id"] for r in epc.load_released(path)["requests"]]
        ee = FakeEE(script={ids[0]: [("Too many concurrent aggregations.", 0.0)],
                            ids[1]: [(None, 0.3)], ids[2]: [(None, 0.3)]},
                    barrier_ids=ids[:3])  # three in flight before the rejection
        out = self.tmp / "conc"
        _, s = self.run_once(path, out, ee)
        submitted = [c[0] for c in ee.calls]
        self.assertEqual(submitted.count(ids[0]), 1)  # zero retry
        self.assertEqual(sorted(submitted), sorted(ids[:3]))  # no request started after the published fatal
        self.assertTrue(s["stopped"].startswith("fatal"))
        self.assertEqual(s["done"], 2)  # already issued calls settled and were saved
        self.assertFalse(s["complete"])
        self.assertEqual(len((out / "request_log.jsonl").read_text().splitlines()), 3)
        led = {json.loads(x)["piece_id"]: json.loads(x)["status"] for x in (out / "ledger.jsonl").read_text().splitlines()}
        self.assertEqual(led, {ids[0]: "failed", ids[1]: "ok", ids[2]: "ok"})

    def test_case_insensitive_and_distinct_from_transient_capacity_and_size_rejections(self):
        budget = ee_metadata.Budget(10, self.tmp / "log.jsonl")
        calls = []

        def boom(msg):
            def f():
                calls.append(msg)
                raise RuntimeError(msg)
            return f

        with self.assertRaisesRegex(ee_metadata.Stop, "^fatal"):
            budget.call("a", boom("TOO MANY CONCURRENT AGGREGATIONS"))
        self.assertEqual(calls.count("TOO MANY CONCURRENT AGGREGATIONS"), 1)
        with self.assertRaisesRegex(ee_metadata.Stop, "failed twice"):
            budget.call("b", boom("Earth Engine capacity exceeded."))  # transient: one bounded retry
        self.assertEqual(calls.count("Earth Engine capacity exceeded."), 2)
        with self.assertRaisesRegex(ee_metadata.Stop, "non-retryable"):
            budget.call("c", boom("Too many pixels in the region."))
        self.assertEqual(calls.count("Too many pixels in the region."), 1)


class StopClassification(Patched):
    """Production-path stop classification (invocation-2 defect 2026-10-11: "cap" substring matched "capacity")."""

    def test_exhausted_transient_capacity_error_counts_one_failure_and_work_continues(self):
        _, path = self.manifest()  # max_consecutive_failures 3, fraction 0.5 after 4
        ids = [r["request_id"] for r in epc.load_released(path)["requests"]]
        ee = FakeEE(script={ids[1]: [("Earth Engine capacity exceeded.", 0.0), ("Earth Engine capacity exceeded.", 0.0)]})
        out = self.tmp / "capx"
        _, s = self.run_once(path, out, ee)
        self.assertIsNone(s["stopped"])
        self.assertEqual((s["failed"], s["done"], s["remaining"]), (1, len(ids) - 1, 1))
        self.assertEqual([c[0] for c in ee.calls].count(ids[1]), 2)  # its one bounded retry
        self.assertIn(ids[-1], [c[0] for c in ee.calls])  # subsequent work ran
        self.assertFalse(epc.piece_file(out, ids[1]).exists())  # failed piece stays pending, no output

    def test_genuine_request_cap_still_stops_without_exceeding_reservations(self):
        _, path = self.manifest(limits={"concurrency": 2})
        out = self.tmp / "cap"
        req = epc.load_released(path)
        out.mkdir()
        budget = ee_metadata.Budget(3, out / "request_log.jsonl")
        s = epc.run_production(req, out, FakeEE(), budget)
        self.assertEqual(budget.used, 3)  # never beyond the cap
        self.assertTrue(s["stopped"].startswith("request cap 3 reached"))
        self.assertEqual(s["done"], 3)
        self.assertFalse(s["complete"])

    def test_deterministic_size_rejection_zero_retry_shared_stop_and_inflight_settles(self):
        _, path = self.manifest(limits={"concurrency": 3, "max_consecutive_failures": 99, "max_failure_fraction": 1.0})
        ids = [r["request_id"] for r in epc.load_released(path)["requests"]]
        for text in ("Too many pixels in the region.", "Found 2e9 pixels; maxPixels allows only 1e9.",
                     "Projection: The CRS of a map projection could not be parsed."):
            ee = FakeEE(script={ids[0]: [(text, 0.0)], ids[1]: [(None, 0.3)], ids[2]: [(None, 0.3)]}, barrier_ids=ids[:3])
            out = self.tmp / f"size{len(list(self.tmp.glob('size*')))}"
            _, s = self.run_once(path, out, ee)
            submitted = [c[0] for c in ee.calls]
            self.assertEqual(submitted.count(ids[0]), 1, text)  # zero retry
            self.assertEqual(sorted(submitted), sorted(ids[:3]), text)  # nothing new after the shared stop
            self.assertTrue(s["stopped"].startswith("non-retryable request-size rejection"), text)
            self.assertEqual(s["done"], 2, text)  # already issued work settled and was saved


if __name__ == "__main__":
    unittest.main()
