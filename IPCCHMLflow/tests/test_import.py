"""Focused importer tests on a tiny P6-format fixture and a scratch SQLite/file store.

Run: /home/swl007007/.venvs/ipcch-mlflow/bin/python -m unittest discover -s IPCCHMLflow/tests -v
"""

import copy
import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import import_runs  # noqa: E402
from extract import SourceConflict  # noqa: E402

H = ("1", "3", "6", "12")


def panel(n, f1=0.5, na=None):
    b = {"counts": {"tp": 1, "fp": 1, "fn": 1, "tn": n - 3}, "na_reasons": {}, "accuracy": 0.6, "precision": 0.5,
         "recall": 0.5, "f1": f1, "f2": 0.5}
    if na:
        b["f1"], b["na_reasons"] = None, {"f1": na}
    return {"n": n, "binary": b, "four_class": {"accuracy": 0.4, "macro_f1": 0.3, "na_reasons": {}},
            "q3_r2_projected": 0.1, "q3_r2_raw": 0.11}


def delta():
    return {"binary.f1": 0.01, "binary.accuracy": -0.002}


def boot(point):
    return {"K": 3, "draws": 20, "seed": 42, "point_delta": point, "defined_draws": 20, "undefined_draws": 0,
            "interval": [point - 0.01, point + 0.01], "na_reason": ""}


def make_fixture(root: Path) -> None:
    rows = []
    for i in range(6):  # 6 keys; persistence available for the first 4 (subset cohort)
        rows.append((100 + i, 24300 + i, "2025-01", "main", 1 if i < 4 else 0))
    rows += [(200, 24400, "2026-01", "supplementary", 1), (201, 24401, "2026-02", "supplementary", 1),
             (202, 24402, "2026-03", "supplementary", 1)]
    horizons = {}
    for h in H:
        d = root / f"stage3/h{int(h):02d}"
        d.mkdir(parents=True)
        with gzip.open(d / "predictions.csv.gz", "wt") as f:
            f.write("admin_code,target_ord,target_month,period,persistence_available\n")
            for r in rows:
                f.write(",".join(map(str, r)) + "\n")
        horizons[h] = {
            "main": {"coverage": {"E_all_keys": 6, "countries": 2}, "routes": {"map": {"terminal_regions": 2}},
                     "E_all": {"n": 6, "status": "scored", "geo": panel(6, na="no positive predictions"),
                               "pool": panel(6), "delta_geo_minus_pool": {"binary.f1": None, "binary.accuracy": 0.0}},
                     "E_persist": {"n": 4, "status": "scored", "geo": panel(4), "persistence": panel(4, 0.4),
                                   "delta_geo_minus_persistence": delta()},
                     "bootstrap": {"geo_vs_pool_E_all": boot(0.0), "geo_vs_persistence_E_persist": boot(0.1)}},
            "supplementary": {"coverage": {"E_all_keys": 3}, "routes": {},
                              "E_all": {"n": 3, "status": "scored", "geo": panel(3), "pool": panel(3),
                                        "delta_geo_minus_pool": delta()},
                              "E_persist": {"n": 3, "status": "scored", "geo": panel(3), "persistence": panel(3),
                                            "delta_geo_minus_persistence": delta()}}}
        (root / f"models/h{int(h):02d}").mkdir(parents=True)
        (root / f"models/h{int(h):02d}/booster.json").write_text(json.dumps({"h": h}))
    (root / "report").mkdir()
    (root / "report/report.json").write_text(json.dumps({"stage": "report", "horizons": horizons}))
    (root / "prepared").mkdir()
    (root / "prepared/X_big_h01.npy").write_bytes(b"\0" * 64)
    (root / "prepared/keys.csv").write_text("a,b\n1,2\n")


def make_cfg(base: Path) -> dict:
    return {"experiment": "IPCCH-test", "repo_root": str(base), "temp_root": str(base), "families": [{
        "family": "fx", "source_run_id": "fx-run", "format": "p6", "root": "{temp}/src",
        "report": "report/report.json", "predictions": "stage3/h{H:02d}/predictions.csv.gz",
        "arms": {"pool": "fresh_trained", "geo": "fresh_trained", "persistence": "persistence_baseline"},
        "model_seed": "42", "include": ["report/**", "stage3/**", "prepared/**"],
        "bundle": {"models.tar": "models/**"},
        "exclude": {"prepared/X_*.npy": "large matrix; referenced by hash"}, "extra": [],
        "tags": {"truth": "synthetic", "periods": "main/supplementary"}}]}


class ImporterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ipcch-mlflow-test-")
        self.base = Path(self.tmp.name)
        make_fixture(self.base / "src")
        self.cfg = make_cfg(self.base)
        self.sources = self.base / "sources.json"
        self.write_cfg()
        self.store = self.base / "store"
        self.uri = f"sqlite:///{self.base}/mlflow.db"
        self.art = (self.base / "artifacts").as_uri()

    def tearDown(self):
        self.tmp.cleanup()

    def write_cfg(self):
        self.sources.write_text(json.dumps(self.cfg))

    def run_cmd(self, command, *extra):
        out = self.base / f"out-{command}.json"
        import_runs.main([command, "--sources", str(self.sources), "--store", str(self.store),
                          "--tracking-uri", self.uri, "--artifact-location", self.art, "--out", str(out), *extra])
        return json.loads(out.read_text())

    def client_runs(self):
        c = import_runs._client(self.uri)
        exp = c.get_experiment_by_name("IPCCH-test")
        return c, c.search_runs([exp.experiment_id], max_results=1000)

    def test_import_verify_and_na_not_logged(self):
        r = self.run_cmd("import")
        self.assertEqual(r["records_planned"], 13)
        self.assertEqual(r["families"][0]["created"], 13)
        v = self.run_cmd("verify")["families"][0]
        self.assertEqual((v["records"], v["child_metrics"]), (13, 552))
        self.assertEqual(v["tar_members"], 4)
        c, runs = self.client_runs()
        geo = [x for x in runs if x.data.tags.get("source_key") == "fx/fx-run/H1/geo/seed42"][0]
        self.assertNotIn("main.E_all.binary.f1", geo.data.metrics)          # undefined -> not logged
        self.assertNotIn("main.E_all.delta.geo_minus_pool.binary.f1", geo.data.metrics)
        self.assertEqual(geo.data.metrics["main.E_all.delta.geo_minus_pool.binary.accuracy"], 0.0)  # defined zero kept
        na = json.loads(Path(c.download_artifacts(geo.info.run_id, "view/na.json", str(self.base))).read_text())
        reasons = {x["metric"]: x["reason"] for x in na}
        self.assertEqual(reasons["main.E_all.binary.f1"], "no positive predictions")
        parent = [x for x in runs if x.data.tags.get("record_kind") == "source_run"][0]
        excl = json.loads(Path(c.download_artifacts(parent.info.run_id, "manifests/excluded.json", str(self.base))).read_text())
        self.assertEqual([e["path"] for e in excl], ["prepared/X_big_h01.npy"])
        arts = import_runs.existing_artifacts(c, parent.info.run_id)
        self.assertNotIn("source/prepared/X_big_h01.npy", arts)

    def test_subset_cohort_digests_and_count_check(self):
        self.run_cmd("import")
        _, runs = self.client_runs()
        geo = [x for x in runs if x.data.tags.get("source_key") == "fx/fx-run/H1/geo/seed42"][0]
        t, m = geo.data.tags, geo.data.metrics
        self.assertNotEqual(t["cohort_keys.main.E_all"], t["cohort_keys.main.E_persist"])
        self.assertEqual((m["main.E_all.n"], m["main.E_persist.n"]), (6.0, 4.0))
        pers = [x for x in runs if x.data.tags.get("source_key") == "fx/fx-run/H1/persistence/seednone"][0]
        self.assertEqual(pers.data.tags["cohort_keys.main.E_persist"], t["cohort_keys.main.E_persist"])
        self.assertNotIn("main.E_all.n", pers.data.metrics)
        rep = self.base / "src/report/report.json"
        obj = json.loads(rep.read_text())
        obj["horizons"]["1"]["main"]["E_persist"]["n"] = 5      # disagrees with the 4 persistence keys
        rep.write_text(json.dumps(obj))
        with self.assertRaises(SourceConflict):
            self.run_cmd("plan")

    def test_conflicting_identity_refused(self):
        self.run_cmd("import")
        _, before = self.client_runs()
        rep = self.base / "src/report/report.json"
        obj = json.loads(rep.read_text())
        obj["horizons"]["3"]["main"]["E_all"]["pool"]["binary"]["f1"] = 0.77
        rep.write_text(json.dumps(obj))
        with self.assertRaises(SourceConflict):
            self.run_cmd("import")
        _, after = self.client_runs()
        self.assertEqual(len(before), len(after))

    def test_interrupted_import_resumes_without_duplicates(self):
        with self.assertRaises(import_runs.Interrupt):
            self.run_cmd("import", "--fail-after", "child-3")
        _, partial = self.client_runs()
        self.assertEqual(len(partial), 4)                       # parent + 3 children
        parent = [x for x in partial if x.data.tags.get("record_kind") == "source_run"][0]
        self.assertEqual(parent.data.tags["import_status"], "in_progress")
        r = self.run_cmd("import")["families"][0]
        self.assertEqual((r["resumed"], r["noop"], r["created"]), (1, 3, 9))
        _, runs = self.client_runs()
        self.assertEqual(len(runs), 13)
        self.assertEqual(self.run_cmd("verify")["families"][0]["records"], 13)

    def test_interrupted_before_bundle_resumes(self):
        with self.assertRaises(import_runs.Interrupt):
            self.run_cmd("import", "--fail-after", "parent-manifests")
        r = self.run_cmd("import")["families"][0]
        self.assertEqual((r["resumed"], r["created"]), (1, 12))
        self.assertEqual(self.run_cmd("verify")["families"][0]["tar_members"], 4)

    def test_idempotent_rerun_is_noop(self):
        self.run_cmd("import")
        _, before = self.client_runs()
        r = self.run_cmd("import")["families"][0]
        self.assertEqual((r["created"], r["resumed"], r["noop"], r["metrics_logged"], r["artifacts_uploaded"]),
                         (0, 0, 13, 0, 0))
        _, after = self.client_runs()
        self.assertEqual(sorted(x.info.run_id for x in before), sorted(x.info.run_id for x in after))

    def test_concurrent_import_refused(self):
        with import_runs.Lock(self.store / "import.lock"):
            with self.assertRaises(SystemExit):
                self.run_cmd("plan")


if __name__ == "__main__":
    unittest.main()
