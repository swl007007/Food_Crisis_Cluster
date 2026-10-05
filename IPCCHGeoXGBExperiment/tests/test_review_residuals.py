"""Supervisor residual list at 91552d0: field-level replay inventory and small interface fixes."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

import test_p5_e2e as p5
from ipcch_geoxgb import learnmap, predict, report, stage3
from ipcch_geoxgb.errors import TechnicalError
from ipcch_geoxgb.replay import replay_run

H = p5.H
GZ = p5.GZ


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    for module in (learnmap, predict):
        mp.setattr(module, "load_experiment_contract", p5._contract)
    root = tmp_path_factory.mktemp("res") / "run"
    p5._write_prepared(root, west_areas=56)  # adopted-local world: gates, pairs and local providers exist
    learnmap.run_learn_map(root)
    predict.run_predict(root)
    report.run_report(root)
    mp.undo()
    return root


def _tamper(run, tmp_path, edit):
    dest = tmp_path / "t"
    shutil.copytree(run, dest)
    edit(dest)
    return replay_run(dest, p5._contract())


def _failed(result, name):
    return any(name in f for f in result["failures"])


def _regenerate_report(root):
    shutil.rmtree(root / "report")
    report.run_report(root)


def _edit_report(root, fn):
    path = root / "report" / "report.json"
    rep = json.loads(path.read_text())
    fn(rep["horizons"][str(H)]["main"])
    path.write_text(json.dumps(rep))


# ---- 1. authoritative cohort metadata


@pytest.mark.parametrize("change", ["country", "period"])
def test_cohort_metadata_rewrites_are_caught_even_with_regenerated_report(run, tmp_path, change):
    def edit(root):
        if change == "country":
            p5._rewrite_predictions(root, lambda p: p.assign(country_key="fabricated_country"))
        else:
            p5._rewrite_predictions(root, lambda p: p.assign(period="supplementary"))
        _regenerate_report(root)
    result = _tamper(run, tmp_path, edit)
    assert _failed(result, "cohort_metadata")


def test_persistence_q3_rewrite_is_caught(run, tmp_path):
    def edit(root):
        p5._rewrite_predictions(root, lambda p: p.assign(persistence_q3=p["persistence_q3"] + 0.01))
        _regenerate_report(root)
    assert _failed(_tamper(run, tmp_path, edit), "persistence_q3_matches_prepared")


# ---- 2. required report evidence


REPORT_TAMPERS = {
    "delta_removed": (lambda m: m["E_all"].update(delta_geo_minus_pool={}), "report_delta_schema"),
    "confusion_zeroed": (lambda m: m["E_all"]["geo"]["four_class"].update(
        confusion_rows_truth_cols_pred=[[0] * 4] * 4, per_class={}), "report_confusion"),
    "bootstrap_country_counts": (lambda m: m["bootstrap"]["geo_vs_pool_E_all"].update(
        country_counts={"geo": [[1, 1, 1, 1], [1, 1, 1, 1]], "pool": [[1, 1, 1, 1], [1, 1, 1, 1]]}),
        "bootstrap_country_counts"),
    "bootstrap_defined_counts": (lambda m: m["bootstrap"]["geo_vs_pool_E_all"].update(
        defined_draws=7, undefined_draws=1993, na_reason="made up"), "bootstrap_defined_counts"),
    "na_reason_removed": (lambda m: m["E_all"]["geo"]["binary"].update(na_reasons={"f1": "x"}), "report_na_reasons"),
}


@pytest.mark.parametrize("name", sorted(REPORT_TAMPERS))
def test_report_evidence_tampers_are_caught(run, tmp_path, name):
    fn, check = REPORT_TAMPERS[name]
    assert _failed(_tamper(run, tmp_path, lambda r: _edit_report(r, fn)), check)


def test_bootstrap_arm_scores_in_draws_are_caught(run, tmp_path):
    def edit(root):
        path = root / "report" / f"bootstrap_h{H:02d}_geo_vs_pool_E_all.csv.gz"
        draws = pd.read_csv(path)
        draws["f1_geo"], draws["f1_pool"] = 123.0, -456.0  # Delta column left correct
        draws.to_csv(path, index=False, compression=GZ)
    assert _failed(_tamper(run, tmp_path, edit), "bootstrap_draws")


def test_missing_or_wrong_diagnostics_are_caught(run, tmp_path):
    assert _failed(_tamper(run, tmp_path, lambda r: [p.unlink() for p in (r / "report").glob("diag_*.csv")]),
                   "diagnostic_file")

    def wrong(root):
        path = root / "report" / f"diag_h{H:02d}_main_country_key.csv"
        table = pd.read_csv(path)
        table["geo_f1"] = 0.5
        table.to_csv(path, index=False)
    assert _failed(_tamper(run, tmp_path / "w", wrong), "diagnostic_values")


# ---- 3. historical pair and gate detail


def _edit_pairs(root, fn):
    for path in (root / "stage3" / f"h{H:02d}").glob("pairs_*.csv.gz"):
        frame = pd.read_csv(path, float_precision="round_trip")
        fn(frame).to_csv(path, index=False, compression=GZ)


STAR = [f"{side}_{q}_star" for side in ("global", "local_routed") for q in ("q2", "q3", "q4", "q5")]


@pytest.mark.parametrize("name, fn, check", [
    ("stars_removed", lambda f: f.drop(columns=STAR), "pair_columns"),
    ("stars_099", lambda f: f.assign(**{c: 0.99 for c in STAR}), "pair_star"),
    ("truth_1", lambda f: f.assign(phase_truth=1), "pair_truth"),
])
def test_pair_detail_tampers_are_caught(run, tmp_path, name, fn, check):
    assert _failed(_tamper(run, tmp_path, lambda r: _edit_pairs(r, fn)), check)


@pytest.mark.parametrize("field, value", [
    ("keys", 999999), ("areas", 999999), ("target_months", 999999), ("crisis_keys", 999999),
    ("noncrisis_keys", 999999), ("local_fit_dates", 999999),
    ("counts_global", {"tp": 1, "fp": 0, "fn": 0, "tn": 0}), ("counts_local", {"tp": 1, "fp": 0, "fn": 0, "tn": 0}),
])
def test_gate_record_field_tampers_are_caught(run, tmp_path, field, value):
    def edit(root):
        path = root / "stage3" / f"h{H:02d}" / "gate_decisions.jsonl"
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        for d in lines:
            d[field] = value
        path.write_text("\n".join(json.dumps(d) for d in lines) + "\n")
    assert _failed(_tamper(run, tmp_path, edit), "gate_record_fields")


# ---- 4. fit identity integrity


@pytest.mark.parametrize("scope", ["stage1-global", "stage3-global"])
def test_fit_keys_identity_tamper_is_caught(run, tmp_path, scope):
    def edit(root):
        for path in (root / "models").glob("*/*/record.json"):
            record = json.loads(path.read_text())
            if record["identity"]["scope"] == scope:
                record["identity"]["fit_keys"] = "0" * 64
                path.write_text(json.dumps(record))
                return
        pytest.fail(f"no {scope} model")
    assert _failed(_tamper(run, tmp_path, edit), "models.identity_digest")


def test_stage1_child_reference_tamper_is_caught(run, tmp_path):
    def edit(root):
        path = root / "stage1" / "model_requests.jsonl"
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        child = next(e for e in lines if e["purpose"] == "child_local")
        child["prepared_rows"] = child["prepared_rows"][1:]
        path.write_text("\n".join(json.dumps(e) for e in lines) + "\n")
    assert _failed(_tamper(run, tmp_path, edit), "child_rows_are_member_F")


def test_stage1_requests_reconstruct_cleanly(run):
    result = replay_run(run, p5._contract())
    assert result["status"] == "passed", result["failures"][:5]
    assert result["passed_checks"].get("stage1_models.fit_identity_rebuilt", 0) > 0
    assert result["passed_checks"].get("stage1_models.child_rows_are_member_F", 0) > 0


# ---- 6. interface and failure context


def test_route_coverage_has_global_area_and_region_counts(run):
    routes = json.loads((run / "report" / "report.json").read_text())["horizons"][str(H)]["main"]["routes"]
    areas, regions = routes["areas"], routes["regions"]
    for key in ("ever_global_routed", "ever_global_fallback", "ever_global_only_no_accepted_split",
                "ever_unmapped_area_global"):
        assert key in areas
    assert {"denominator_map_regions", "ever_local_routed", "ever_global_fallback"} <= set(regions)


def test_empty_diagnostic_tables_keep_headers(run):
    table = pd.read_csv(run / "report" / f"diag_h{H:02d}_supplementary_country_key.csv")
    assert len(table) == 0 and {"country_key", "geo_f1", "na_reason_E_all"} <= set(table.columns)


def test_local_prediction_failure_carries_region_and_provider(tmp_path, monkeypatch):
    import test_p4_stage3 as p4

    ctx = p4._ctx(tmp_path)
    real = stage3.predict

    def flaky(q, X):
        if q.records["q2"]["kind"] == "local":
            raise TechnicalError("injected local q2 prediction failure")
        return real(q, X)

    monkeypatch.setattr(stage3, "predict", flaky)
    with pytest.raises(TechnicalError) as caught:
        stage3.run_fold(ctx, p4._fold(p4.M0 + 50))
    assert any("local provider" in n and "region" in n for n in caught.value.__notes__)
