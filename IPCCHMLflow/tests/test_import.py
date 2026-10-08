"""Focused importer tests on a tiny P6-format fixture and a scratch SQLite/file store.

Run: /home/swl007007/.venvs/ipcch-mlflow/bin/python -m unittest discover -s IPCCHMLflow/tests -v
"""

import gzip
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

import pandas as pd

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backup_restore  # noqa: E402
import extract  # noqa: E402
import import_runs  # noqa: E402
import inventory  # noqa: E402
from extract import SourceConflict  # noqa: E402

H = ("1", "3", "6", "12")
MODEL_ID = "ab" + "0" * 62


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
    booster = b"fixture booster bytes"
    (root / f"models/ab/{MODEL_ID}").mkdir(parents=True)
    (root / f"models/ab/{MODEL_ID}/q2.ubj").write_bytes(booster)
    (root / f"models/ab/{MODEL_ID}/record.json").write_text("{}")
    (root / "stage3/model_requests.jsonl").write_text(json.dumps(
        {"identity_sha256": MODEL_ID, "booster_sha256": {"q2": hashlib.sha256(booster).hexdigest()}}) + "\n")
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
        "tags": {"truth": "synthetic", "periods": "main/supplementary"},
        "inventory": {"required_digest_keys": ["booster_sha256"], "name_keys": {"identity_sha256": "dir"},
                      "model_contract": "xgb"}}]}


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
        self.assertEqual(v["tar_members"], 6)
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
        self.assertEqual(self.run_cmd("verify")["families"][0]["tar_members"], 6)

    def test_idempotent_rerun_is_noop(self):
        self.run_cmd("import")
        _, before = self.client_runs()
        r = self.run_cmd("import")["families"][0]
        self.assertEqual((r["created"], r["resumed"], r["noop"], r["metrics_logged"], r["artifacts_uploaded"]),
                         (0, 0, 13, 0, 0))
        _, after = self.client_runs()
        self.assertEqual(sorted(x.info.run_id for x in before), sorted(x.info.run_id for x in after))

    def test_missing_or_changed_retained_model_refused(self):
        inv = self.run_cmd("plan")
        model = self.base / f"src/models/ab/{MODEL_ID}/q2.ubj"
        model.write_bytes(b"changed booster bytes")           # recorded digest no longer matches
        with self.assertRaisesRegex(SourceConflict, "booster_sha256"):
            self.run_cmd("plan")
        model.unlink()                                        # retained model disappeared
        with self.assertRaisesRegex(SourceConflict, "booster_sha256"):
            self.run_cmd("plan")
        self.assertEqual(inv["records_planned"], 13)

    def test_own_booster_required_even_if_another_directory_has_same_digest(self):
        own = self.base / f"src/models/ab/{MODEL_ID}/q2.ubj"
        other = self.base / f"src/models/cd/{'cd' + '1' * 62}"
        other.mkdir(parents=True)
        (other / "q2.ubj").write_bytes(own.read_bytes())      # same SHA in a different identity directory
        self.run_cmd("plan")
        own.unlink()
        with self.assertRaisesRegex(SourceConflict, "own booster q2.ubj missing"):
            self.run_cmd("plan")

    def test_unclassified_digest_key_refused(self):
        rep = self.base / "src/report/report.json"
        obj = json.loads(rep.read_text())
        obj["mystery_sha256"] = "c" * 64
        rep.write_text(json.dumps(obj))
        with self.assertRaisesRegex(SourceConflict, "no inventory decision"):
            self.run_cmd("plan")

    def test_noop_detects_equal_size_artifact_corruption(self):
        self.run_cmd("import")
        stored = next((self.base / "artifacts").rglob("source/report/report.json"))
        b = bytearray(stored.read_bytes())
        b[10] ^= 1                                            # same size, different content
        stored.write_bytes(bytes(b))
        with self.assertRaisesRegex(SourceConflict, "checksum mismatch"):
            self.run_cmd("import")

    def test_split_combined_persist_delta_logged(self):
        def block(n_all, n_persist, f1):
            return {"E_all": {"n": n_all, "status": "scored", "geo": panel(n_all), "pool": panel(n_all),
                              "delta_geo_minus_pool": delta()},
                    "E_persist": {"n": n_persist, "status": "scored", "geo": panel(n_persist),
                                  "persistence": panel(n_persist),
                                  "delta_geo_minus_persistence": {"binary.f1": f1, "binary.accuracy": None}},
                    "local_rows": 0, "unmapped_rows": 1, "old_split_matched": {"geo": panel(n_all), "pool": panel(n_all)},
                    "delta_new_minus_old_geo": delta()}
        comb = {"horizons": {h: {"combined": block(9, 7, 0.0147), "2025": block(6, 4, 0.02), "2026": block(3, 3, 0.01)}
                             for h in H}}
        (self.base / "src/combined.json").write_text(json.dumps(comb))
        fam = self.cfg["families"][0]
        ch, _, _ = extract.extract({**fam, "combined": "combined.json"}, self.base / "src")
        geo = [c for c in ch if c.H == "1" and c.arm == "geo"][0]
        self.assertEqual(geo.metrics["combined.E_persist.delta.geo_minus_persistence.binary.f1"], 0.0147)
        self.assertEqual(geo.provenance["combined.E_persist.delta.geo_minus_persistence.binary.f1"],
                         "combined.horizons.1.combined.E_persist.delta_geo_minus_persistence.binary.f1")
        self.assertNotIn("combined.E_persist.delta.geo_minus_persistence.binary.accuracy", geo.metrics)  # NA
        self.assertEqual(geo.metrics["y2026.E_persist.delta.geo_minus_persistence.binary.f1"], 0.01)

    def test_reconcile_additive_change_keeps_runs_and_superseded_evidence(self):
        self.run_cmd("import")
        _, before = self.client_runs()
        self.cfg["families"][0]["inventory"]["informational_keys"] = {"unused_sha256": "policy change only"}
        self.write_cfg()
        orig = extract.EXTRACTORS["p6"]

        def with_extra_tag(fam, rep, root):
            out = orig(fam, rep, root)
            for c in out:
                if c.arm == "geo":
                    c.tags["cohort_keys.main.extra_subset"] = "d" * 64
            return out
        extract.EXTRACTORS["p6"] = with_extra_tag
        try:
            with self.assertRaises(SourceConflict):          # plain import refuses a changed fingerprint
                self.run_cmd("import")
            r = self.run_cmd("reconcile")["families"][0]
            self.assertEqual((r["parent"], r["children_reconciled"], r["children_unchanged"]), ("reconciled", 4, 8))
            self.assertEqual(self.run_cmd("verify")["families"][0]["records"], 13)
            self.assertEqual(self.run_cmd("import")["families"][0]["noop"], 13)
        finally:
            extract.EXTRACTORS["p6"] = orig
        c, after = self.client_runs()
        self.assertEqual(sorted(x.info.run_id for x in before), sorted(x.info.run_id for x in after))
        parent = [x for x in after if x.data.tags.get("record_kind") == "source_run"][0]
        old = [x for x in before if x.data.tags.get("record_kind") == "source_run"][0]
        self.assertEqual(parent.data.tags["import_fingerprint.previous"], old.data.tags["import_fingerprint"])
        arts = import_runs.existing_artifacts(c, parent.info.run_id)
        self.assertTrue(any(k.startswith("manifests/superseded/plan-summary.") for k in arts))
        geo = [x for x in after if x.data.tags.get("source_key") == "fx/fx-run/H1/geo/seed42"][0]
        self.assertTrue(any(k.startswith("view/superseded/evaluation_view.")
                            for k in import_runs.existing_artifacts(c, geo.info.run_id)))

    def test_reconcile_refuses_value_change(self):
        self.run_cmd("import")
        rep = self.base / "src/report/report.json"
        obj = json.loads(rep.read_text())
        obj["horizons"]["3"]["main"]["E_all"]["pool"]["binary"]["f1"] = 0.77
        rep.write_text(json.dumps(obj))
        with self.assertRaisesRegex(SourceConflict, "would change"):
            self.run_cmd("reconcile")

    def test_reconcile_rebinds_empty_shell_then_import_resumes_it(self):
        c = import_runs._client(self.uri)
        exp = import_runs.get_experiment(c, "IPCCH-test", self.art)
        shell = c.create_run(exp.experiment_id, tags={"source_key": "fx/fx-run", "import_fingerprint": "0" * 64,
                                                       "import_status": "in_progress", "record_kind": "source_run"})
        self.assertEqual(self.run_cmd("reconcile")["families"][0]["parent"], "empty shell rebound")
        r = self.run_cmd("import")["families"][0]
        self.assertEqual((r["resumed"], r["created"]), (1, 12))
        _, runs = self.client_runs()
        parent = [x for x in runs if x.data.tags.get("record_kind") == "source_run"]
        self.assertEqual([x.info.run_id for x in parent], [shell.info.run_id])
        self.assertEqual(parent[0].data.tags["import_fingerprint.previous"], "0" * 64)
        self.assertEqual(self.run_cmd("verify")["families"][0]["records"], 13)

    def test_concurrent_import_refused(self):
        with import_runs.Lock(self.store / "import.lock"):
            with self.assertRaises(SystemExit):
                self.run_cmd("plan")


if __name__ == "__main__":
    unittest.main()


class ExtractionGuardTest(unittest.TestCase):
    def child(self):
        return extract.Child("fx", "run", "1", "pool", "42", "fresh_trained")

    def test_sanitized_name_collision_refused(self):
        c = self.child()
        with self.assertRaisesRegex(SourceConflict, "from both"):
            extract.add_tree(c, "main.routes", {"pool:fallback": 2, "pool_fallback": 2}, "report.routes")

    def test_same_source_readd_is_idempotent_and_cross_check_needs_equal_values(self):
        c = self.child()
        c.add("main.E_all.n", 6, "report.E_all.n")
        c.add("main.E_all.n", 6, "report.E_all.n")
        c.add("main.E_all.n", 6, "report.E_all.pool.n", cross_check=True)
        with self.assertRaisesRegex(SourceConflict, "cross-check"):
            c.add("main.E_all.n", 5, "report.E_all.geo.n", cross_check=True)

    def test_gate_subsets_bound_to_saved_routes(self):
        sub = pd.DataFrame({"admin_code": [1, 2, 3, 4, 5], "target_ord": [10, 10, 10, 10, 10],
                            "local_eligible": [1, 1, 1, 1, 0],
                            "route": ["local", "pool_fallback:gate_support:keys", "pool_fallback:gain_not_above_threshold",
                                      "local", "unmapped_area_pool"]})
        e = {"ungated_local_diagnostic": {"by_gate": {"adopted": {}, "historical_support_rejected": {},
                                                      "gain_rejected": {}}}}
        c = self.child()
        c.add("main.local_eligible.adopted.n", 2, "src.adopted.n")
        c.add("main.local_eligible.gain_rejected.n", 1, "src.gain.n")
        extract.cohort_tag(c, "main.local_eligible", sub[sub["local_eligible"] == 1])
        extract.gate_tags(c, "main", e, sub)
        self.assertEqual(c.metrics["main.local_eligible.historical_support_rejected.n"], 1.0)
        self.assertNotEqual(c.tags["cohort_keys.main.local_eligible.adopted"], c.tags["cohort_keys.main.local_eligible"])
        bad = self.child()
        bad.add("main.local_eligible.adopted.n", 3, "src.adopted.n")      # report n disagrees with saved routes
        with self.assertRaisesRegex(SourceConflict, "adopted"):
            extract.gate_tags(bad, "main", e, sub)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class RestoreScratchServerTest(unittest.TestCase):
    def test_restore_check_downloads_through_independent_scratch_server(self):
        with tempfile.TemporaryDirectory(prefix="ipcch-mlflow-restore-test-") as tmp:
            base = Path(tmp)
            make_fixture(base / "src")
            cfg = make_cfg(base)
            cfg["experiment"] = "IPCCH"
            (base / "sources.json").write_text(json.dumps(cfg))
            store = base / "store"
            (store / "artifacts").mkdir(parents=True)
            port = free_port()
            proc = subprocess.Popen([str(backup_restore.MLFLOW), "server", "--backend-store-uri",
                                     f"sqlite:///{store / 'mlflow.db'}", "--artifacts-destination", str(store / "artifacts"),
                                     "--serve-artifacts", "--host", "127.0.0.1", "--port", str(port), "--workers", "1"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                for _ in range(120):
                    try:
                        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                        break
                    except OSError:
                        time.sleep(1)
                import_runs.main(["import", "--sources", str(base / "sources.json"), "--store", str(store),
                                  "--tracking-uri", f"http://127.0.0.1:{port}"])
            finally:
                proc.terminate()
                proc.wait(timeout=60)
            backup_restore.backup(store, base / "backup")
            with self.assertRaises(SystemExit):
                backup_restore.restore_check(base / "backup", base / "scratch-live", port=5000)
            res = backup_restore.restore_check(base / "backup", base / "scratch", port=free_port())
            self.assertEqual(res["ipcch_runs_by_kind"], {"source_run": 1, "evaluation_view": 12})
            self.assertEqual([d["artifact"] for d in res["downloaded"]], ["manifests/plan-summary.json", "models.tar"])


class MLPModelContractTest(unittest.TestCase):
    """Drives inventory.reconcile with the mlp contract on a tiny store layout."""
    IDENT = "ee" + "2" * 62
    TRANSFORM = "f" * 64

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ipcch-mlflow-mlp-contract-")
        self.root = Path(self.tmp.name)
        d = self.root / f"models/models/ee/{self.IDENT}"
        d.mkdir(parents=True)
        (d / "record.json").write_text(json.dumps({"identity": {"transform_sha256": self.TRANSFORM}}))
        (d / "state.pt").write_bytes(b"state")
        (self.root / "models/transforms").mkdir()
        (self.root / f"models/transforms/{self.TRANSFORM}.npz").write_bytes(b"transform")
        (self.root / "model_requests_stage3_rep42.jsonl").write_text(json.dumps({"identity_sha256": self.IDENT}) + "\n")
        self.fam = {"family": "mlpfx", "inventory": {"name_keys": {"identity_sha256": "dir"}, "model_contract": "mlp"}}

    def tearDown(self):
        self.tmp.cleanup()

    def reconcile(self):
        recs = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file():
                recs.append({"path": p.relative_to(self.root).as_posix(), "bytes": p.stat().st_size,
                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
        files = {"include": [r for r in recs if not r["path"].startswith("models/")],
                 "bundle": {"models.tar": [r for r in recs if r["path"].startswith("models/")]},
                 "excluded": [], "shared": []}
        return inventory.reconcile(self.fam, self.root, files, {})

    def test_complete_store_reconciles(self):
        mc = self.reconcile()["model_contract"]
        self.assertEqual((mc["unique_identities"], mc["states_present"], mc["unique_transforms_referenced"]), (1, 1, 1))

    def test_missing_state_refused(self):
        (self.root / f"models/models/ee/{self.IDENT}/state.pt").unlink()
        with self.assertRaisesRegex(SourceConflict, "state.pt missing"):
            self.reconcile()

    def test_missing_referenced_transform_refused(self):
        (self.root / f"models/transforms/{self.TRANSFORM}.npz").unlink()
        (self.root / "models/transforms/other.npz").write_bytes(b"x")   # keep the directory non-empty
        with self.assertRaisesRegex(SourceConflict, "referenced transform"):
            self.reconcile()
