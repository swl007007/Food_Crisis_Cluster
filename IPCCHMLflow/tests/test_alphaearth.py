"""AlphaEarth family: extractor, strict digest bindings, native model-ledger contract, exact dataset descriptors and
the bounded dashboard extension (state machine, crash windows, recovery, refusals) on tiny fixtures.

Run: /home/swl007007/.venvs/ipcch-mlflow/bin/python -m unittest discover -s IPCCHMLflow/tests -v
"""

import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import backup_restore  # noqa: E402
import import_runs  # noqa: E402
import inventory  # noqa: E402
import naming  # noqa: E402
import summary_catalog as sc  # noqa: E402
from extract import SourceConflict  # noqa: E402
from test_import import make_cfg, make_fixture  # noqa: E402

H = ("1", "3", "6", "12")
FIRST = {"1": "2023-02", "3": "2023-04", "6": "2023-07", "12": "2024-01"}
ARMS = ("B0", "B1", "B2", "G")
sha = lambda b: hashlib.sha256(b).hexdigest()


def _gz_csv(path: Path, header: list, rows: list) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(buf.getvalue().encode())


def flat_panel(n, f1=0.6):
    return {"binary.f1": f1, "binary.accuracy": 0.7, "binary.precision": 0.6, "binary.recall": 0.6, "binary.f2": 0.6,
            "four_class.accuracy": 0.5, "four_class.macro_f1": 0.4, "q3_r2_projected": 0.3, "q3_r2_raw": 0.31, "n": n}


def details(n):
    return {"n": n, "binary_counts": {"tp": 1, "fp": 1, "fn": 1, "tn": n - 3}, "na_reasons": {}}


def delta():
    return {"binary.f1": 0.01, "binary.accuracy": -0.002, "four_class.macro_f1": 0.0, "q3_r2_projected": 0.001}


def identity(arm, h, block, role, fit_origin):
    return {"protocol": "ipcch-alphaearth-backbone-v1", "arm": arm, "horizon": int(h), "block_id": block, "role": role,
            "fit_origin": fit_origin, "n_rows": 10, "X_sha256": "1" * 64, "y_sha256": {q: "2" * 64 for q in ("q2", "q3", "q4", "q5")},
            "weights": {"formula": "w", "sha256": "3" * 64, "sum": 1.0}, "features_sha256": "4" * 64,
            "source": {"prepared_manifest_sha256": None}}


def make_ae_fixture(root: Path, external: Path) -> dict:
    """Small AlphaEarth-format run: 4 leads x (main + supplement) predictions, saved report, 2 fitted identities."""
    horizons = {}
    for h in H:
        months = [FIRST[h], "2024-06", "2025-01", "2025-10"]
        rows = [(300 + i, 24300 + i, months[i], "main", 1 if i < 3 else 0, 2 + i % 2, ["0.40404040404040403", "0.1", "0.25", "0.0"][i])
                for i in range(4)]
        rows += [(400, 24400, "2026-01", "supplement", 1, 3, "0.5"), (401, 24403, "2026-04", "supplement", 0, 1, "0.05")]
        _gz_csv(root / f"predict/predictions_h{int(h):02d}.csv.gz",
                ["admin_code", "target_ord", "target_month", "period", "persistence_available", "phase_truth", "q3"], rows)
        horizons[h] = {
            "main_keys": 4, "panels": {a: flat_panel(4) for a in ARMS}, "panel_details": {a: details(4) for a in ARMS},
            "persistence": {"keys": 3, "panels": {**{a: flat_panel(3) for a in ARMS}, "persistence": flat_panel(3, 0.5)},
                            "panel_details": {**{a: details(3) for a in ARMS}, "persistence": details(3)},
                            "model_minus_persistence": {a: delta() for a in ARMS}},
            "supplement_2026": {"keys": 2, "panels": {a: flat_panel(2) for a in ARMS}, "panel_details": {a: details(2) for a in ARMS}},
            "contrasts": {"G-B2": delta(), "G-B1": delta(), "G-B0": delta(), "B1-B0": delta(), "B2-B1": delta()},
            "bootstrap": {"K": 3, "draws": 20, "seed": 42, "point_delta": 0.0, "defined_draws": 20, "undefined_draws": 0,
                          "lower_bound": 0.0, "descriptive_uncorrected_95": [0.0, 0.0], "na_reason": ""}}
    (root / "report").mkdir(parents=True)
    rep = json.dumps({"stage": "report", "horizons": horizons, "qualifying_horizons": []}).encode()
    (root / "report/report.json").write_bytes(rep)
    (root / "report/COMPLETE.json").write_text(json.dumps({"report_sha256": sha(rep)}))
    # prepared + descriptors (training pool files)
    arts = {}
    for h in H:
        (root / "prepared").mkdir(exist_ok=True)
        (root / f"prepared/X296_h{int(h):02d}.npy").write_bytes(bytes([int(h)]) * 32)
        _gz_csv(root / f"prepared/keys_h{int(h):02d}.csv.gz", ["admin_code", "target_ord"], [[1, int(h)]])
        arts[f"keys_h{int(h):02d}.csv.gz"] = sha((root / f"prepared/keys_h{int(h):02d}.csv.gz").read_bytes())
        arts[f"X296_h{int(h):02d}.npy"] = sha((root / f"prepared/X296_h{int(h):02d}.npy").read_bytes())
    _gz_csv(root / "prepared/feature_order_296.csv.gz", ["name"], [[f"f{i}"] for i in range(3)])
    arts["feature_order_296.csv.gz"] = sha((root / "prepared/feature_order_296.csv.gz").read_bytes())
    sched = [["B1", "1", "h01_main_2023", "main", "24299"], ["B0", "1", "h01_main_2023", "main", "24299"]]
    _gz_csv(root / "prepared/schedule_requests.csv.gz", ["arm", "horizon", "block_id", "role", "fit_origin"], sched)
    arts["schedule_requests.csv.gz"] = sha((root / "prepared/schedule_requests.csv.gz").read_bytes())
    (root / "prepared/prepared-manifest.json").write_text(json.dumps({"artifacts_sha256": arts}))
    _gz_csv(root / "descriptors/descriptor_table.csv.gz", ["admin_code", "ae2017_mean_A00", "ae2017_sd_A00"], [[1, 0.1, 0.2]])
    (root / "descriptors/COMPLETE.json").write_text(json.dumps({"stage": "descriptors"}))
    # two fitted identities, native CSV ledgers, own bundle dirs
    led = []
    for arm, (_, h, block, role, fo) in zip(("B1", "B0"), sched):
        idn = identity(arm, h, block, role, int(fo))
        ident = sha(json.dumps(idn, sort_keys=True, separators=(",", ":")).encode())
        d = root / f"models/{ident[:2]}/{ident}"
        d.mkdir(parents=True)
        tg = {}
        for q in ("q2", "q3", "q4", "q5"):
            b = f"booster {arm} {q}".encode()
            (d / f"{q}.ubj").write_bytes(b)
            tg[q] = {"booster_sha256": sha(b), "resolved_config_sha256": "5" * 64}
        (d / "record.json").write_text(json.dumps({"identity_digest": ident, "identity": idn, "targets": tg}))
        led.append([ident, "fit", arm, h, block, role, fo])
    _gz_csv(root / "develop/request_ledger.csv.gz", ["identity", "action", "arm", "horizon", "block_id", "role", "fit_origin"], led[:1])
    _gz_csv(root / "predict/request_ledger.csv.gz", ["identity", "action", "arm", "horizon", "block_id", "role", "fit_origin"], led[1:])
    external.write_bytes(b"external evidence bytes")
    (root / "extraction").mkdir()
    (root / "extraction/manifest.json").write_text(json.dumps({"pieces_sha256": sha(external.read_bytes())}))
    (root / "upstream.json").write_text(json.dumps({"code_sha256": {"src/x.py": "6" * 64}}))
    rec = r"models/[0-9a-f]{2}/[0-9a-f]{64}/record\.json"
    return {"family": "alphaearth_partition", "source_run_id": "ae-fx-run", "format": "alphaearth", "root": "{temp}/ae",
            "report": "report/report.json", "predictions": "predict/predictions_h{H:02d}.csv.gz",
            "arms": {**{a: "fresh_trained" for a in ARMS}, "persistence": "persistence_baseline"}, "model_seed": "42",
            "include": ["report/**", "predict/**", "prepared/*.csv.gz", "prepared/prepared-manifest.json", "descriptors/**",
                        "develop/**", "extraction/**", "upstream.json"],
            "bundle": {"models.tar": "models/**"}, "exclude": {"prepared/X296_h*.npy": "large matrix; referenced by hash"},
            "extra": [], "tags": {"code_commit": "fixture", "truth": "synthetic"},
            "inventory": {"strict_bindings": True, "scan_bundle_json": True, "expected_leaves": None,
                          "model_contract": "alphaearth_xgb", "model_ledgers": ["develop/request_ledger.csv.gz", "predict/request_ledger.csv.gz"],
                          "model_schedule": "prepared/schedule_requests.csv.gz", "expected_identities": 2, "expected_boosters": 8,
                          "bindings": [
                              {"kind": "own", "key": "artifacts_sha256", "file": r"prepared/prepared-manifest\.json",
                               "path": r"\.artifacts_sha256\.(.+)", "target": "prepared/{0}"},
                              {"kind": "own", "key": "report_sha256", "file": r"report/COMPLETE\.json", "path": r"\.report_sha256",
                               "target": "report/report.json"},
                              {"kind": "own", "key": "booster_sha256", "file": rec, "path": r"\.targets\.(q[2-5])\.booster_sha256",
                               "target": "{dir}/{0}.ubj"},
                              {"kind": "typed", "key": "X_sha256", "file": rec, "path": r"\.identity\.X_sha256", "type": "array_digest", "reason": "t"},
                              {"kind": "typed", "key": "y_sha256", "file": rec, "path": r"\.identity\.y_sha256\.q[2-5]", "type": "array_digest", "reason": "t"},
                              {"kind": "typed", "key": "sha256", "file": rec, "path": r"\.identity\.weights\.sha256", "type": "array_digest", "reason": "t"},
                              {"kind": "typed", "key": "features_sha256", "file": rec, "path": r"\.identity\.features_sha256", "type": "canonical_json", "reason": "t"},
                              {"kind": "typed", "key": "resolved_config_sha256", "file": rec, "path": r"\.targets\.q[2-5]\.resolved_config_sha256",
                               "type": "config_digest", "reason": "t"},
                              {"kind": "external", "key": "pieces_sha256", "file": r"extraction/manifest\.json", "path": r"\.pieces_sha256",
                               "target": str(external), "sha256": sha(external.read_bytes())},
                              {"kind": "historical", "key": "code_sha256", "file": r"upstream\.json", "path": r"\.code_sha256\.src/x\.py",
                               "values": {".code_sha256.src/x.py": "6" * 64}, "reason": "upstream"}]}}


class AEBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ipcch-ae-test-")
        self.base = Path(self.tmp.name)
        self.ext = self.base / "external.bin"
        self.fam = make_ae_fixture(self.base / "ae", self.ext)
        self.cfg = {"experiment": "IPCCH-test", "repo_root": str(self.base), "temp_root": str(self.base), "families": [self.fam]}

    def tearDown(self):
        self.tmp.cleanup()

    def plan(self):
        return import_runs.plan_family(self.cfg, self.fam, import_runs.Hasher(None), {})

    def refused(self, needle):
        with self.assertRaises(SourceConflict) as cm:
            self.plan()
        self.assertIn(needle, str(cm.exception))


class AEExtractTest(AEBase):
    def test_children_cohorts_and_no_supplement_persistence(self):
        body = self.plan()
        self.assertEqual(body["totals"]["children"], 20)
        inv = body["inventory"]
        self.assertEqual((inv["mode"], inv["model_contract"]["unique_identities"], inv["model_contract"]["boosters_checked"]),
                         ("strict_bindings", 2, 8))
        views = {v["key"]: v for v in body["views"]}
        g = views["alphaearth_partition/ae-fx-run/H1/G/seed42"]
        self.assertIn("primary.all_scored.bootstrap.regional_calibration_minus_shared_calibration.binary.f1.lower_bound_one_sided_98_75",
                      g["metrics"])
        self.assertIn("primary.all_scored.delta.regional_calibration_minus_pooled.binary.f1", g["metrics"])
        self.assertFalse(any(k.startswith("holdout.persistence_available") for v in views.values() for k in v["metrics"]))
        self.assertTrue(all(any(x["metric"] == "holdout.persistence_available" for x in v["na"]) for v in views.values()))
        p = views["alphaearth_partition/ae-fx-run/H12/persistence/seednone"]
        self.assertEqual(p["metrics"]["primary.persistence_available.binary.f1"], 0.5)

    def test_exact_text_dataset_descriptor_and_family_name(self):
        f = pd.DataFrame({"admin_code": ["1"], "target_ord": ["24300"], "target_month": ["2023-02"], "phase_truth": ["3"],
                          "q3": ["0.40404040404040403"]})
        g = f.assign(q3=["0.404040404040404"])                     # same float, different saved text
        a = sc.ae_eval_descriptor("1", "main", "E_all", f, "alphaearth_partition")
        b = sc.ae_eval_descriptor("1", "main", "E_all", g, "alphaearth_partition")
        self.assertEqual(a["keys_sha256"], b["keys_sha256"])
        self.assertNotEqual(a["truth_sha256"], b["truth_sha256"])  # no float-normalized alias
        self.assertEqual(naming.eval_dataset_name("alphaearth_partition", "1", "main", "E_all", None),
                         "IPCCH eval | 1-month | 2023-02..2025-10 | all_scored (AlphaEarth supervised partition)")
        self.assertEqual(naming.eval_dataset_name("p6_geoxgb", "1", "main", "E_all", None),
                         "IPCCH eval | 1-month | 2023-02..2025-10 | all_scored")   # legacy names unchanged


class AEModelContractTest(AEBase):
    def ledger(self, which):
        p = self.base / "ae" / f"{which}/request_ledger.csv.gz"
        rows = list(csv.reader(io.StringIO(gzip.decompress(p.read_bytes()).decode())))
        return p, rows

    def write(self, p, rows):
        _gz_csv(p, rows[0], rows[1:])

    def ident_dirs(self):
        return sorted((self.base / "ae/models").glob("*/*"))

    def test_empty_ledgers_refused(self):
        for which in ("develop", "predict"):
            p, rows = self.ledger(which)
            self.write(p, rows[:1])
        self.refused("model ledgers are empty")

    def test_duplicate_row_refused(self):
        p, rows = self.ledger("develop")
        self.write(p, rows + rows[1:])
        self.refused("duplicate identity")

    def test_missing_identity_refused(self):
        p, rows = self.ledger("predict")
        self.write(p, rows[:1])
        self.refused("not one-to-one with the schedule")

    def test_extra_identity_dir_refused(self):
        d = self.base / "ae/models/ff" / ("ff" * 32)
        d.mkdir(parents=True)
        (d / "record.json").write_text("{}")
        self.refused("bundle identity dirs")

    def test_schedule_mismatch_refused(self):
        p, rows = self.ledger("develop")
        rows[1][6] = "24298"
        self.write(p, rows)
        self.refused("not one-to-one with the schedule")

    def test_record_identity_tamper_refused(self):
        d = self.ident_dirs()[0]
        rec = json.loads((d / "record.json").read_text())
        rec["identity"]["n_rows"] = 11                  # any other identity field
        (d / "record.json").write_text(json.dumps(rec))
        self.refused("record identity digest does not recompute")

    def test_missing_own_booster_with_hash_elsewhere_refused(self):
        a, b = self.ident_dirs()
        (b / "q3.ubj").unlink()
        (self.base / "ae/models/copy_q3.ubj").write_bytes((a / "q3.ubj").read_bytes())   # same bytes elsewhere
        with self.assertRaises(SourceConflict):
            self.plan()

    def test_corrupted_booster_refused(self):
        d = self.ident_dirs()[0]
        (d / "q2.ubj").write_bytes(b"corrupt")
        self.refused("is not the SHA of its own file")


class AEDigestPolicyTest(AEBase):
    def test_unclassified_digest_refused(self):
        p = self.base / "ae/report/COMPLETE.json"
        d = json.loads(p.read_text())
        d["mystery_sha256"] = "7" * 64
        p.write_text(json.dumps(d))
        self.refused("matches 0 bindings")

    def test_required_own_file_wrong_refused(self):
        (self.base / "ae/prepared/keys_h03.csv.gz").write_bytes(b"changed")
        self.refused("is not the SHA of its own file prepared/keys_h03.csv.gz")

    def test_external_missing_and_wrong_refused(self):
        self.ext.write_bytes(b"other bytes")
        self.refused("external")
        self.ext.unlink()
        self.refused("external file")

    def test_historical_wrong_path_or_value_refused(self):
        p = self.base / "ae/upstream.json"
        p.write_text(json.dumps({"code_sha256": {"src/x.py": "8" * 64}}))
        self.refused("historical code_sha256 not pinned")

    def test_record_bundle_digest_coverage(self):
        d = sorted((self.base / "ae/models").glob("*/*"))[0]
        rec = json.loads((d / "record.json").read_text())
        rec["extra_sha256"] = "9" * 64                 # outside identity: only digest coverage can catch it
        (d / "record.json").write_text(json.dumps(rec))
        self.refused("extra_sha256 matches 0 bindings")

    def test_unused_rule_and_leaf_count_refused(self):
        self.fam["inventory"]["bindings"].append({"kind": "typed", "key": "never_sha256", "file": "x", "path": "y",
                                                  "type": "t", "reason": "r"})
        self.refused("bindings never used")
        self.fam["inventory"]["bindings"].pop()
        self.fam["inventory"]["expected_leaves"] = 999
        self.refused("digest leaves")


EXT_HOOKS = ("ext-after-journal-intent", "ext-after-intent-tag", "ext-after-status", "logged-model-created", "models",
             "row-created", "row-5", "ext-before-fp", "ext-after-fp", "ext-after-lineage", "ext-after-description",
             "ext-before-complete", "ext-after-complete")


class AEExtensionTest(unittest.TestCase):
    """Legacy (GeoXGB-reference fixture) dashboard, then the AlphaEarth fixture added by the bounded extension."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="ipcch-ae-ext-")
        base = Path(cls.tmp.name)
        make_fixture(base / "src")
        cfg = make_cfg(base)
        ae = make_ae_fixture(base / "ae", base / "external.bin")
        store = base / "store"
        (store / "artifacts").mkdir(parents=True)
        uri = f"sqlite:///{store}/mlflow.db"
        legacy_src = base / "legacy.json"
        legacy_src.write_text(json.dumps(cfg))
        import_runs.main(["import", "--sources", str(legacy_src), "--store", str(store), "--tracking-uri", uri,
                          "--artifact-location", (store / "artifacts").as_uri()])
        common = ["--store", str(store), "--tracking-uri", uri, "--artifact-location", str(store / "artifacts" / "dashboard")]
        out = base / "legacy-plan.json"
        sc.main(["plan", "--sources", str(legacy_src), "--skip-expected", "--out", str(out), *common])
        pfp = json.loads(out.read_text())["plan_fingerprint"]
        inv = base / "legacy-inv.json"
        sc.main(["inventory", "--sources", str(legacy_src), "--store", str(store), "--tracking-uri", uri, "--out", str(inv)])
        bk = base / "legacy-bk"
        backup_restore.backup(store, bk)
        sc.main(["apply", "--sources", str(legacy_src), "--skip-expected", "--plan-fingerprint", pfp, "--inventory", str(inv),
                 "--backup", str(bk), *common])
        cfg["families"].append(ae)
        full_src = base / "full.json"
        full_src.write_text(json.dumps(cfg))
        import_runs.main(["import", "--family", "alphaearth_partition", "--sources", str(full_src), "--store", str(store),
                          "--tracking-uri", uri, "--artifact-location", (store / "artifacts").as_uri()])
        client = import_runs._client(uri)
        plan = sc.build_plan(client, cfg, store)
        add_rows = sorted(r["projection_key"] for r in plan["rows"] if r["family"] == "alphaearth_partition")
        add_models = sorted(m["projection_key"] for m in plan["models"] if m["family"] == "alphaearth_partition")
        add_ds = {d["name"]: d["full_sha256"] for d in plan["datasets"]
                  if "AlphaEarth" in d["name"] or "compact296" in d["name"]}
        pred_file = store / "dashboard" / f"plan-{pfp[:16]}.json"
        legacy_plan = json.loads(pred_file.read_text())
        cfg["dashboard_extensions"] = {"ae": {
            "predecessor_fingerprint": pfp, "predecessor_plan": str(pred_file),
            "predecessor_plan_sha256": sha(pred_file.read_bytes()), "rows": add_rows, "models": add_models, "datasets": add_ds,
            "counts": {"rows": 20, "model_versions": 16, "registered_models": 16, "evaluation_datasets": 12, "training_datasets": 8},
            "legacy_counts": {"rows": len(legacy_plan["rows"]), "model_versions": len(legacy_plan["models"]),
                              "registered_models": len({m["registered_model"] for m in legacy_plan["models"]})}}}
        full_src.write_text(json.dumps(cfg))
        cls.base, cls.store0, cls.cfg, cls.pfp = base, store, cfg, pfp

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.fresh_case()

    def fresh_case(self):
        """A new store path per case: SQLAlchemy caches engines per URI, so a store is never replaced in place."""
        self.d = Path(tempfile.mkdtemp(prefix="case-", dir=self.base))
        self.store = self.d / "store"
        shutil.copytree(self.store0, self.store)
        self.uri = f"sqlite:///{self.store}/mlflow.db"
        self.src = self.d / "sources.json"
        cfg = json.loads(json.dumps(self.cfg).replace(str(self.store0), str(self.store)))
        self.src.write_text(json.dumps(cfg))
        self.n = 0

    def sc(self, command, *extra):
        out = self.d / f"sc-{command}-{self.n}.json"
        self.n += 1
        sc.main([command, "--sources", str(self.src), "--store", str(self.store), "--tracking-uri", self.uri,
                 "--extend", "ae", "--out", str(out), *extra])
        return json.loads(out.read_text())

    def apply(self, *extra):
        plan = self.sc("plan")
        inv = self.d / f"inv-{self.n}.json"
        sc.main(["inventory", "--sources", str(self.src), "--store", str(self.store), "--tracking-uri", self.uri, "--out", str(inv)])
        # research retention: --extend applies without a full-store backup; the inventory and journal still guard it
        return self.sc("apply", "--plan-fingerprint", plan["plan_fingerprint"], "--inventory", str(inv), *extra)

    def snapshot(self):
        """Every dashboard object: IDs, tags, params, metrics incl. history, inputs, row.json bytes."""
        client = import_runs._client(self.uri)
        exp = client.get_experiment_by_name(sc.DASHBOARD_EXPERIMENT)
        runs = {}
        for r in client.search_runs([exp.experiment_id], max_results=1000, run_view_type=3):
            full = client.get_run(r.info.run_id)
            hist = {k: [(m.value, m.step, m.timestamp) for m in client.get_metric_history(r.info.run_id, k)] for k in full.data.metrics}
            runs[r.info.run_id] = {"tags": dict(full.data.tags), "params": dict(full.data.params), "metrics": hist,
                                   "inputs": sorted((d.dataset.name, d.dataset.digest) for d in full.inputs.dataset_inputs)
                                   + sorted(x.model_id for x in (full.inputs.model_inputs or [])),
                                   "row": import_runs._download_bytes(client, r.info.run_id, "dashboard/row.json")}
        models = {m.model_id: (m.name, dict(m.tags)) for m in client.search_logged_models([exp.experiment_id], max_results=500)}
        versions = {(v.name, v.version): (v.source, dict(v.tags)) for v in client.search_model_versions("")}
        return {"exp_tags": dict(client.get_experiment(exp.experiment_id).tags), "runs": runs, "models": models,
                "versions": versions}

    def legacy_part(self, snap):
        keep = lambda tags: tags.get(sc.FP_TAG) == self.pfp
        return {"runs": {k: v for k, v in snap["runs"].items() if keep(v["tags"])},
                "models": {k: v for k, v in snap["models"].items() if keep(v[1])},
                "versions": {k: v for k, v in snap["versions"].items() if keep(v[1])}}

    def assert_complete(self, before_legacy):
        snap = self.snapshot()
        self.assertEqual(self.legacy_part(snap), before_legacy)   # legacy IDs/tags/metrics/history/inputs/bytes
        t = snap["exp_tags"]
        self.assertEqual(t["catalog_status"], "complete")
        self.assertNotIn(sc.EXT_TAG, t)
        self.assertEqual(t[sc.LINEAGE_TAG].split(",")[0], self.pfp)
        add_runs = [v for v in snap["runs"].values() if v["tags"].get(sc.FP_TAG) == t[sc.FP_TAG]]
        self.assertEqual(len(add_runs), 20)
        self.assertEqual(len(snap["models"]), len(before_legacy["models"]) + 16)
        self.assertEqual(len(snap["versions"]), len(before_legacy["versions"]) + 16)
        return snap

    def test_extend_then_repeat_is_mutation_free(self):
        legacy = self.legacy_part(self.snapshot())
        r = self.apply()
        self.assertEqual(r["apply"]["state"], "S3")
        self.assertEqual(r["verify"]["legacy"]["rows"], len(legacy["runs"]))
        snap = self.assert_complete(legacy)
        counts = backup_restore.db_counts(self.store / "mlflow.db")
        self.assertIn("are historical references only: their local model weights (models.tar) were intentionally "
                      "removed", snap["exp_tags"]["mlflow.note.content"])        # reading guide states it in the UI
        for mid, (name, tags) in snap["models"].items():                # legacy tar records keep their wording
            ae = tags.get("family") == "alphaearth_supervised_partition"
            self.assertEqual(tags["zz_prov.model_bundle_uri"] == "", ae, name)
            self.assertIn("not uploaded" if ae else "Weights stay in the family run's models.tar", tags["mlflow.note.content"])
        r2 = self.apply()
        self.assertEqual((r2["apply"]["state"], r2["apply"]["writes"]), ("S3", 0))
        self.assertEqual(self.snapshot(), snap)
        self.assertEqual(backup_restore.db_counts(self.store / "mlflow.db"), counts)

    def test_every_crash_window_resumes_without_duplicates(self):
        legacy = self.legacy_part(self.snapshot())
        for hook in EXT_HOOKS:
            with self.subTest(hook=hook):
                self.fresh_case()
                with self.assertRaises(import_runs.Interrupt):
                    self.apply("--fail-after", hook)
                r = self.apply()
                self.assertEqual(r["apply"]["state"], "S3")
                self.assert_complete(legacy)

    def test_foreign_intent_and_foreign_incomplete_refused(self):
        client = import_runs._client(self.uri)
        eid = client.get_experiment_by_name(sc.DASHBOARD_EXPERIMENT).experiment_id
        client.set_experiment_tag(eid, sc.EXT_TAG, "f" * 64)
        with self.assertRaisesRegex(SourceConflict, "foreign extension intent"):
            self.apply()
        client.delete_experiment_tag(eid, sc.EXT_TAG)
        client.set_experiment_tag(eid, "catalog_status", "incomplete")
        with self.assertRaisesRegex(SourceConflict, "foreign or unverifiable"):
            self.apply()

    def test_ambiguous_same_name_logged_model_refused(self):
        import mlflow
        client = import_runs._client(self.uri)
        eid = client.get_experiment_by_name(sc.DASHBOARD_EXPERIMENT).experiment_id
        mlflow.set_tracking_uri(self.uri)
        mlflow.create_external_model(name="AlphaEarth supervised partition | pooled | 1-month | seed 42", experiment_id=eid,
                                     tags={"note": "no projection key"})
        with self.assertRaisesRegex(SourceConflict, "without this extension's projection key"):
            self.apply()

    def test_legacy_drift_refused_before_any_write(self):
        client = import_runs._client(self.uri)
        exp = client.get_experiment_by_name(sc.DASHBOARD_EXPERIMENT)
        run = client.search_runs([exp.experiment_id], max_results=1)[0]
        client.set_tag(run.info.run_id, "family", "drifted")
        before = backup_restore.db_counts(self.store / "mlflow.db")
        with self.assertRaises(SourceConflict):
            self.apply()
        self.assertNotIn(sc.EXT_TAG, client.get_experiment(exp.experiment_id).tags)
        self.assertEqual(backup_restore.db_counts(self.store / "mlflow.db"), before)

    def test_plan_refuses_unbound_or_drifted_additions(self):
        cfg = json.loads(self.src.read_text())
        cfg["dashboard_extensions"]["ae"]["predecessor_fingerprint"] = "0" * 64
        self.src.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(SourceConflict, "legacy drift"):
            self.sc("plan")
        cfg["dashboard_extensions"]["ae"]["predecessor_fingerprint"] = self.pfp
        cfg["dashboard_extensions"]["ae"]["rows"] = cfg["dashboard_extensions"]["ae"]["rows"][:-1]
        self.src.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(SourceConflict, "legacy drift"):     # an unlisted addition lands in the legacy part
            self.sc("plan")


if __name__ == "__main__":
    unittest.main()
