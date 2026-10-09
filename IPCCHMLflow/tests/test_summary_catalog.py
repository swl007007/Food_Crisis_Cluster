"""Dashboard tests on the tiny GeoXGB-reference fixture in a scratch SQLite/file store.

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
from test_import import assert_provenance_listed_last, make_cfg, make_fixture  # noqa: E402

GEO1 = "geoxgb_reference/lead01/partitioned_gated/seed42"


class DashboardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ipcch-dashboard-test-")
        self.base = Path(self.tmp.name)
        make_fixture(self.base / "src")
        cfg = make_cfg(self.base)
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
                 "--skip-expected", "--artifact-location", str(self.store / "artifacts" / "dashboard"),
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

    def rows(self):
        client = import_runs._client(self.uri)
        exp = client.get_experiment_by_name(sc.DASHBOARD_EXPERIMENT)
        runs = client.search_runs([exp.experiment_id], max_results=1000)
        return client, {r.data.tags["zz_prov.projection_key"]: r for r in runs}

    def test_wide_rows_names_datasets_and_models(self):
        plan, inv, bk = self.prepare()
        c = plan["counts"]
        self.assertEqual((c["rows"], c["model_versions"], c["registered_models"], c["training_datasets"]), (12, 8, 8, 4))
        self.assertEqual(c["datasets"], 16 + 4)       # 4 leads x (primary, holdout) x (all_scored, persistence_available)
        v = self.apply(plan, inv, bk)["verify"]
        self.assertEqual((v["rows"], v["logged_models"], v["model_versions"], v["registered_models"],
                          v["detailed_runs_unchanged"], v["catalog_status"]), (12, 8, 8, 8, 13, "complete"))
        client, rows = self.rows()
        geo = client.get_run(rows[GEO1].info.run_id)
        self.assertEqual(geo.info.run_name, "GeoXGB reference | partitioned_gated | 1-month")
        assert_provenance_listed_last(self, self.store / "mlflow.db", geo.info.run_id)
        t, m = geo.data.tags, geo.data.metrics
        self.assertEqual((t["family"], t["arm"], t["arm_role"], t["lead_months"], t["period.primary"]),
                         ("geoxgb_reference", "partitioned_gated", "candidate", "01", "2023-02..2025-10"))
        self.assertNotIn("primary.all_scored.binary.f1", m)                     # NA stays absent
        self.assertIn("primary.all_scored.binary.f1", t["zz_prov.na_metrics"])
        self.assertEqual(m["primary.persistence_available.binary.f1"], 0.5)
        self.assertEqual(m["primary.persistence_available.binary.f1.minus_persistence"], 0.1)   # bootstrap point
        self.assertAlmostEqual(m["primary.persistence_available.binary.f1.minus_persistence.ci_low"], 0.09)
        self.assertEqual(m["holdout.persistence_available.binary.f1.minus_persistence"], 0.01)  # delta block, no CI
        self.assertNotIn("holdout.persistence_available.binary.f1.minus_persistence.ci_low", m)
        inputs = {d.dataset.name: [x.value for x in d.tags if x.key == "mlflow.data.context"][0]
                  for d in geo.inputs.dataset_inputs}
        self.assertEqual(inputs["IPCCH eval | 1-month | 2023-02..2025-10 | persistence_available"], "evaluation")
        self.assertEqual(inputs["IPCCH training pool | rich561 | 1-month"], "training")
        pers = client.get_run(rows["geoxgb_reference/lead01/persistence/seednone"].info.run_id)
        self.assertEqual(pers.inputs.model_inputs or [], [])
        self.assertNotIn("registered_model", pers.data.tags)
        pdata = {(d.dataset.name, d.dataset.digest) for d in pers.inputs.dataset_inputs}
        self.assertTrue({(d.dataset.name, d.dataset.digest) for d in geo.inputs.dataset_inputs
                         if "persistence_available" in d.dataset.name} <= pdata)      # same rows -> same dataset
        mid = geo.data.tags["zz_prov.model_ids"]
        lm = client.get_logged_model(mid)
        self.assertEqual(lm.name, "GeoXGB reference | partitioned_gated | 1-month | seed 42")
        self.assertEqual(lm.source_run_id, geo.data.tags["zz_prov.original_run_id"])
        rm = client.get_registered_model("IPCCH GeoXGB reference | partitioned_gated | 1-month")
        self.assertIn("Regional model where the historical gate", rm.description)
        exp = client.get_experiment_by_name(sc.DASHBOARD_EXPERIMENT)
        self.assertIn("Start here", exp.tags["mlflow.note.content"])

    def test_seed_means(self):
        r = {"family": "mlp_fixed_map", "old_arm": "pool", "H": "1", "tags": {"arm": "pooled"},
             "family_run_id": "p", "model_keys": ["k"], "training_dataset": "t", "training_files": {},
             "original_source_key": "s"}
        rows = [{**r, "seed": s, "original_run_id": f"r{s}", "values": {"a": v, "b": 1.0, "a.ci_low": 0.0},
                 "value_inputs": {"a": "d1", "b": "d1" if s != "44" else "d2", "a.ci_low": "d1"}, "na": {}}
                for s, v in (("42", 0.1), ("43", 0.2), ("44", 0.6))]
        mean = sc.seed_means(rows)[0]
        self.assertEqual(mean["projection_key"], "mlp_residual_fixed_maps/lead01/pooled/seedmean")
        self.assertAlmostEqual(mean["values"]["a"], 0.3)
        self.assertNotIn("a.ci_low", mean["values"])                          # intervals are never averaged
        self.assertEqual(mean["na"]["b"]["reason"], "evaluation rows differ between seeds")
        with self.assertRaisesRegex(SourceConflict, "incomplete"):
            sc.seed_means(rows[:2])

    def test_repeat_apply_is_noop_without_new_history(self):
        plan, inv, bk = self.prepare()
        self.apply(plan, inv, bk)
        bk2 = self.base / "backup-again"
        backup_restore.backup(self.store, bk2)
        res = self.apply(plan, inv, bk2)["apply"]
        self.assertEqual((res.get("rows_noop"), res.get("rows_created", 0), res.get("logged_models_created", 0),
                          res.get("model_versions_created", 0), res.get("metrics_logged", 0)), (12, 0, 0, 0, 0))

    def test_interrupted_apply_resumes_same_plan(self):
        plan, inv, bk = self.prepare()
        with self.assertRaises(import_runs.Interrupt):
            self.apply(plan, inv, bk, "--fail-after", "row-5")
        client = import_runs._client(self.uri)
        self.assertEqual(client.get_experiment_by_name(sc.DASHBOARD_EXPERIMENT).tags["catalog_status"], "incomplete")
        bk2 = self.base / "backup-resume"
        backup_restore.backup(self.store, bk2)
        res = self.apply(plan, inv, bk2)
        self.assertEqual((res["apply"]["rows_noop"], res["apply"]["rows_created"]), (5, 7))
        self.assertEqual((res["verify"]["rows"], res["verify"]["model_versions"]), (12, 8))

    def test_refusals(self):
        plan, inv, bk = self.prepare()
        with self.assertRaisesRegex(SourceConflict, "frozen"):
            self.sc("apply", "--plan-fingerprint", "0" * 64, "--inventory", str(inv), "--backup", str(bk))
        client = import_runs._client(self.uri)
        client.create_experiment("unrelated", artifact_location=(self.store / "artifacts" / "x").as_uri())
        with self.assertRaisesRegex(SourceConflict, "fresh backup"):
            self.apply(plan, inv, bk)
        self.assertIsNone(client.get_experiment_by_name(sc.DASHBOARD_EXPERIMENT))

    def test_period_span_must_match_scored_months(self):
        with self.assertRaisesRegex(SourceConflict, "differ from the named span"):
            sc.check_span("p6_geoxgb", "1", "main", "E_all",
                          {"num_rows": 3, "target_month_min": "2023-02", "target_month_max": "2025-09"})
        with self.assertRaisesRegex(SourceConflict, "outside named span"):
            sc.check_span("p6_geoxgb", "1", "main", "E_persist",
                          {"num_rows": 3, "target_month_min": "2023-01", "target_month_max": "2025-09"})
        sc.check_span("p6_geoxgb", "1", "main", "E_persist",
                      {"num_rows": 3, "target_month_min": "2023-05", "target_month_max": "2025-09"})


if __name__ == "__main__":
    unittest.main()
