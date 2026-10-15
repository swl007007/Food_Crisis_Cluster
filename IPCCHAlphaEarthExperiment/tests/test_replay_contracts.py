"""Replay contract list from the supervisor report/replay sample: canonical cohort binding, unique provider identity,
full report recount, bootstrap draw contents, retained count/NA evidence. Each tamper must fail at its semantic check."""

import json
import shutil
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from helpers import small_contract
from ipcch_alphaearth import backbone, pipeline, replay, replay_report as rr, report, synthetic
from ipcch_alphaearth.artifacts import sha256_file, write_csv_gz, write_json
from ipcch_alphaearth.errors import ContractError


class ReplayContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = small_contract(**{"correction.support": {"n_eff": 30, "admins": 10, "target_months": 3},
                                         "evaluation.bootstrap.draws": 200})
        cls.inputs = synthetic.make_inputs(cls.contract, n_admin=150, seed=5)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name) / "base"
        ws, store = pipeline.Workspace(cls.base), backbone.ModelStore(cls.base / "models")
        pipeline.run_develop(cls.inputs, cls.contract, ws, store, {})
        pipeline.run_freeze_map(cls.inputs, cls.contract, ws)
        pipeline.run_predict(cls.inputs, cls.contract, ws, store, {})
        report.run_report(cls.base, cls.contract, sorted(cls.inputs.horizons))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def copy(self, name):
        dst = Path(self.tmp.name) / name
        shutil.copytree(self.base, dst)
        return dst

    def regenerate_report(self, run):
        shutil.rmtree(run / "report")
        report.run_report(run, self.contract, sorted(self.inputs.horizons))

    def replay_failures(self, run):
        with self.assertRaises(ContractError):
            replay.run_replay(self.inputs, self.contract, run, run / "models")
        return json.loads((run / "replay" / "replay.json").read_text())["failed"]

    def rewrite_csv(self, path, fn):
        frame = pd.read_csv(path, float_precision="round_trip")
        frame = fn(frame)
        path.unlink()
        write_csv_gz(path, frame)

    def test_untampered_replay_passes(self):
        run = self.copy("clean")
        self.assertTrue(replay.run_replay(self.inputs, self.contract, run, run / "models")["pass"])

    def test_rewritten_truth_and_keys_fail_canonical_binding(self):
        run = self.copy("truth")

        def tamper(f):
            m = f.index[f["period"] == "main"][:40]
            f.loc[m, "phase_truth"], f.loc[m, "q3"] = 3, 0.999
            f.loc[m[:3], "admin_code"], f.loc[m[:3], "country_key"] = -77, "FALSE"
            f.loc[m[:3], "persistence_source_month"] = "2099-12"
            return f
        self.rewrite_csv(run / "predict" / "predictions_h01.csv.gz", tamper)
        self.regenerate_report(run)
        failed = self.replay_failures(run)
        self.assertTrue(any("canonical_cohort_fields" in f for f in failed), failed)

    def test_supplement_relabelled_main_fails(self):
        run = self.copy("supp")
        self.rewrite_csv(run / "predict" / "predictions_h03.csv.gz", lambda f: f.assign(period="main"))
        self.regenerate_report(run)
        failed = self.replay_failures(run)
        self.assertTrue(any(f.startswith("h03_supplement") and "canonical_cohort_fields" in f for f in failed), failed)

    def test_wrong_provider_identity_fails(self):
        run = self.copy("provider")
        self.rewrite_csv(run / "predict" / "providers_all_h06.csv.gz", lambda f: f.assign(provider_identity="0" * 64))
        self.rewrite_csv(run / "develop" / "providers_h06.csv.gz", lambda f: f.assign(provider_identity="0" * 64))
        failed = self.replay_failures(run)
        self.assertIn("h06:historical_providers_equal_expected", failed)
        self.assertIn("h06:combined_providers_equal_expected", failed)

    def test_fabricated_report_verdict_fails(self):
        run = self.copy("verdict")
        path = run / "report" / "report.json"
        rep = json.loads(path.read_text())
        rep["horizons"]["12"]["panels"]["G"]["binary.f1"] = -999
        rep["horizons"]["12"]["decision"]["status"] = "protected_stable_benefit"
        rep["horizons"]["12"]["decision"]["guards"] = {"pass": True, "items": [], "failed": []}
        rep["qualifying_horizons"] = ["12"]
        path.unlink()
        write_json(path, rep)
        (run / "report" / "COMPLETE.json").unlink()
        write_json(run / "report" / "COMPLETE.json", {"report_sha256": sha256_file(path)})
        failed = self.replay_failures(run)
        self.assertIn("h12:report:main_panels", failed)
        self.assertIn("report:qualifying_horizons", failed)

    def test_altered_bootstrap_draws_fail(self):
        run = self.copy("draws")
        self.rewrite_csv(run / "report" / "bootstrap_draws_h01.csv.gz", lambda f: f.assign(delta=123.0))
        failed = self.replay_failures(run)
        self.assertIn("h01:report:bootstrap_regenerated", failed)

    def tamper_report(self, run, fn):
        path = run / "report" / "report.json"
        rep = json.loads(path.read_text())
        fn(rep)
        path.unlink()
        write_json(path, rep)
        (run / "report" / "COMPLETE.json").unlink()
        write_json(run / "report" / "COMPLETE.json", {"report_sha256": sha256_file(path)})

    def test_residual_emitted_fields_must_agree(self):
        cases = {
            "criteria": (lambda r: r["horizons"]["1"]["decision"].update(
                criterion_annual_signs=not r["horizons"]["1"]["decision"]["criterion_annual_signs"]), "h01:report:decision_criteria"),
            "guard_item": (lambda r: r["horizons"]["1"]["decision"]["guards"]["items"][0].update(
                **{"pass": not r["horizons"]["1"]["decision"]["guards"]["items"][0]["pass"]}), "h01:report:decision_guards"),
            "undefined_draws": (lambda r: r["horizons"]["1"]["bootstrap"].update(undefined_draws=999), "h01:report:bootstrap_regenerated"),
        }
        for name, (fn, check) in cases.items():
            run = self.copy(f"residual_{name}")
            self.tamper_report(run, fn)
            self.assertIn(check, self.replay_failures(run), name)
        run = self.copy("residual_country")
        self.rewrite_csv(run / "report" / "country_diagnostics_h01.csv.gz", lambda f: f.assign(f1_G=123.0))
        self.assertIn("h01:report:country_diagnostics", self.replay_failures(run))

    def test_report_retains_counts_and_na_reasons(self):
        rep = json.loads((self.base / "report" / "report.json").read_text())["horizons"]["1"]
        det = rep["panel_details"]["G"]
        self.assertEqual(set(det["binary_counts"]), {"tp", "fp", "fn", "tn"})
        self.assertEqual(len(det["four_class_confusion_rows_truth_cols_pred"]), 4)
        self.assertIn("country_counts", rep["bootstrap"])
        diag = pd.read_csv(self.base / "report" / "country_diagnostics_h01.csv.gz")
        self.assertTrue({"G_tp", "G_fp", "G_fn", "G_tn", "B2_tp"} <= set(diag.columns))



class UnavailablePersistenceReplay(unittest.TestCase):
    """Main/supplement rows without a persistence source (empty source month, NaN persistence_q3) as written by the real
    prepared keys: the saved predictions must replay against the canonical cohort, while real tampering still fails."""

    @classmethod
    def setUpClass(cls):
        cls.contract = small_contract(**{"correction.support": {"n_eff": 30, "admins": 10, "target_months": 3},
                                         "evaluation.bootstrap.draws": 200})
        cls.inputs = synthetic.make_inputs(cls.contract, n_admin=150, seed=5)
        for hi in cls.inputs.horizons.values():  # latest 40 targets become unavailable, exactly as prepare.persistence writes them
            idx = np.argsort(hi.target, kind="mergesort")[-40:]
            k = hi.keys
            k.loc[idx, "persistence_available"], k.loc[idx, "persistence_phase"] = 0, 0
            k.loc[idx, "persistence_q3"], k.loc[idx, "persistence_source_month"] = np.nan, ""
            k.loc[idx, "persistence_age_months"] = -1
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name) / "base"
        ws, store = pipeline.Workspace(cls.base), backbone.ModelStore(cls.base / "models")
        pipeline.run_develop(cls.inputs, cls.contract, ws, store, {})
        pipeline.run_freeze_map(cls.inputs, cls.contract, ws)
        pipeline.run_predict(cls.inputs, cls.contract, ws, store, {})
        report.run_report(cls.base, cls.contract, sorted(cls.inputs.horizons))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    copy, replay_failures, rewrite_csv, regenerate_report = (ReplayContracts.copy, ReplayContracts.replay_failures,
                                                             ReplayContracts.rewrite_csv, ReplayContracts.regenerate_report)

    def raw_rewrite(self, path, fn):
        """Rewrite a saved CSV as text cells (empty stays empty, no NA inference) so only ``fn``'s edit changes."""
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame = fn(frame)
        path.unlink()
        write_csv_gz(path, frame)

    def test_unavailable_persistence_rows_replay_canonically(self):
        pred = pd.read_csv(self.base / "predict" / "predictions_h01.csv.gz", dtype=str, keep_default_na=False)
        blank = pred["persistence_source_month"] == ""
        self.assertGreater(int(blank.sum()), 0)  # the scenario reaches the scored main/supplement cohort
        self.assertTrue((pred.loc[blank, "persistence_q3"] == "").all())
        run = self.copy("unavailable_clean")
        self.assertTrue(replay.run_replay(self.inputs, self.contract, run, run / "models")["pass"])

    def test_changed_valid_source_month_still_fails(self):
        run = self.copy("unavailable_month")

        def tamper(f):
            i = f.index[(f["period"] == "main") & (f["persistence_source_month"] != "")][0]
            f.loc[i, "persistence_source_month"] = "2000-01"
            return f
        self.raw_rewrite(run / "predict" / "predictions_h01.csv.gz", tamper)
        self.regenerate_report(run)
        self.assertIn("h01_main", " ".join(f for f in self.replay_failures(run) if f.endswith("canonical_cohort_fields")))

    def test_na_looking_text_is_not_an_empty_source_month(self):
        run = self.copy("unavailable_na_text")

        def tamper(f):
            i = f.index[f["persistence_source_month"] == ""][0]
            f.loc[i, "persistence_source_month"] = "NA"
            return f
        self.raw_rewrite(run / "predict" / "predictions_h01.csv.gz", tamper)
        self.regenerate_report(run)
        self.assertTrue(any(f.endswith("canonical_cohort_fields") for f in self.replay_failures(run)))

    def test_changed_provider_target_ordinal_still_fails(self):
        run = self.copy("unavailable_provider_ord")

        def tamper(f):
            f.loc[f.index[-1], "target_ord"] = str(int(f.loc[f.index[-1], "target_ord"]) // 10)  # a dropped last digit
            return f
        self.raw_rewrite(run / "predict" / "providers_all_h03.csv.gz", tamper)
        self.assertIn("h03:combined_providers_equal_expected", self.replay_failures(run))


class UndefinedStaysUndefined(unittest.TestCase):
    def test_undefined_f1_compares_as_undefined(self):
        s = rr.scores([1, 2, 1], [1, 1, 2], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1])
        self.assertIsNone(s["binary.f1"])  # no crisis in truth or prediction
        self.assertIsNone(s["q3_r2_projected"])
        self.assertIn("binary.f1", s["na"])
        self.assertIsNone(replay._indep_f1(np.array([1, 2]), np.array([1, 2])))


if __name__ == "__main__":
    unittest.main()
