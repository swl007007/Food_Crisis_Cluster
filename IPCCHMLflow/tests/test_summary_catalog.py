"""Focused tests for summary_catalog on the tiny P6-format fixture in a scratch SQLite/file store.

Run: /home/swl007007/.venvs/ipcch-mlflow/bin/python -m unittest discover -s IPCCHMLflow/tests -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import backup_restore  # noqa: E402
import import_runs  # noqa: E402
import summary_catalog as sc  # noqa: E402
from extract import SourceConflict  # noqa: E402
from test_import import make_cfg, make_fixture  # noqa: E402


class SummaryCatalogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ipcch-summary-test-")
        self.base = Path(self.tmp.name)
        make_fixture(self.base / "src")
        cfg = make_cfg(self.base)
        cfg["experiment"] = "IPCCH-test"
        self.sources = self.base / "sources.json"
        self.sources.write_text(json.dumps(cfg))
        self.store = self.base / "store"
        (self.store / "artifacts").mkdir(parents=True)
        self.uri = f"sqlite:///{self.store}/mlflow.db"
        self.art = (self.store / "artifacts").as_uri()
        import_runs.main(["import", "--sources", str(self.sources), "--store", str(self.store),
                          "--tracking-uri", self.uri, "--artifact-location", self.art])
        self.n = 0

    def tearDown(self):
        self.tmp.cleanup()

    def sc(self, command, *extra):
        out = self.base / f"sc-{command}-{self.n}.json"
        self.n += 1
        sc.main([command, "--sources", str(self.sources), "--store", str(self.store), "--tracking-uri", self.uri,
                 "--skip-expected", "--artifact-location", str(self.store / "artifacts" / "summary"),
                 "--out", str(out), *extra])
        return json.loads(out.read_text())

    def prepare(self):
        plan = self.sc("plan")
        inv = self.base / f"inventory-{self.n}.json"
        sc.main(["inventory", "--sources", str(self.sources), "--store", str(self.store), "--tracking-uri", self.uri,
                 "--out", str(inv)])
        bk = self.base / f"backup-{self.n}"
        backup_restore.backup(self.store, bk)
        return plan, inv, bk

    def apply(self, plan, inv, bk, *extra):
        return self.sc("apply", "--plan-fingerprint", plan["plan_fingerprint"], "--inventory", str(inv),
                       "--backup", str(bk), *extra)

    def client(self):
        return import_runs._client(self.uri)

    def test_plan_apply_verify_counts_links_and_na(self):
        plan, inv, bk = self.prepare()
        c = plan["counts"]
        self.assertEqual((c["rows"], c["model_versions"], c["registered_models"], c["dataset_descriptors"]), (32, 8, 8, 16))
        self.assertLessEqual(len(c["metric_names"]), 9)
        res = self.apply(plan, inv, bk)
        v = res["verify"]
        self.assertEqual((v["rows"], v["logged_models"], v["model_versions"], v["registered_models"],
                          v["dataset_descriptors"], v["original_runs_unchanged"], v["catalog_status"]),
                         (32, 8, 8, 8, 16, 13, "complete"))
        self.assertLessEqual(v["distinct_metric_names"], 9)
        client = self.client()
        exp = client.get_experiment_by_name(sc.SUMMARY_EXPERIMENT)
        runs = client.search_runs([exp.experiment_id], max_results=1000)
        geo = [r for r in runs if r.data.tags["projection_key"] == "fx/fx-run/H1/geo/seed42#main.E_all"][0]
        self.assertNotIn("binary.f1", geo.data.metrics)                    # NA stays absent
        self.assertEqual(geo.data.tags["na_metrics"], "binary.f1")
        mid = geo.data.tags["model_id"]
        lm = client.get_logged_model(mid)
        self.assertEqual(lm.tags["mlflow.model.isExternal"], "true")
        self.assertEqual(lm.source_run_id, geo.data.tags["original_run_id"])   # provenance to the original child
        self.assertTrue(any(m.dataset_name == geo.data.tags["dataset_name"] for m in lm.metrics))
        pers = [r for r in runs if r.data.tags["projection_key"].startswith("fx/fx-run/H1/persistence")][0]
        self.assertNotIn("model_id", pers.data.tags)
        self.assertEqual(pers.inputs.model_inputs or [], [])
        pool = [r for r in runs if r.data.tags["projection_key"] == "fx/fx-run/H1/pool/seed42#main.E_all"][0]
        self.assertEqual((pool.data.tags["dataset_name"], pool.data.tags["dataset_digest"]),
                         (geo.data.tags["dataset_name"], geo.data.tags["dataset_digest"]))   # same keys + truth shared
        self.assertNotEqual(geo.data.tags["dataset_digest"], [r for r in runs if r.data.tags["projection_key"] ==
                            "fx/fx-run/H1/geo/seed42#main.E_persist"][0].data.tags["dataset_digest"])

    def test_repeat_apply_is_noop_without_new_history(self):
        plan, inv, bk = self.prepare()
        self.apply(plan, inv, bk)
        bk2 = self.base / "backup-again"
        backup_restore.backup(self.store, bk2)
        res = self.apply(plan, inv, bk2)["apply"]
        self.assertEqual((res.get("rows_noop"), res.get("rows_created", 0), res.get("logged_models_created", 0),
                          res.get("model_versions_created", 0), res.get("metrics_logged", 0)), (32, 0, 0, 0, 0))

    def test_interrupted_apply_resumes_same_plan(self):
        plan, inv, bk = self.prepare()
        with self.assertRaises(import_runs.Interrupt):
            self.apply(plan, inv, bk, "--fail-after", "row-5")
        client = self.client()
        exp = client.get_experiment_by_name(sc.SUMMARY_EXPERIMENT)
        self.assertEqual(exp.tags["catalog_status"], "incomplete")          # visibly incomplete
        bk2 = self.base / "backup-resume"
        backup_restore.backup(self.store, bk2)
        res = self.apply(plan, inv, bk2)
        self.assertEqual((res["apply"]["rows_noop"], res["apply"]["rows_created"]), (5, 27))
        self.assertEqual((res["verify"]["rows"], res["verify"]["model_versions"]), (32, 8))

    def test_refusals(self):
        plan, inv, bk = self.prepare()
        with self.assertRaisesRegex(SourceConflict, "frozen"):
            self.sc("apply", "--plan-fingerprint", "0" * 64, "--inventory", str(inv), "--backup", str(bk))
        client = self.client()
        client.create_experiment("unrelated", artifact_location=(self.store / "artifacts" / "x").as_uri())
        with self.assertRaisesRegex(SourceConflict, "fresh backup"):
            self.apply(plan, inv, bk)
        self.assertIsNone(client.get_experiment_by_name(sc.SUMMARY_EXPERIMENT))


if __name__ == "__main__":
    unittest.main()
