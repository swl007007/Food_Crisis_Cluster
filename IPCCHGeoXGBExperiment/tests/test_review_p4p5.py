"""Supervisor P4/P5 review (pinned 13e74b5): frozen-map binding, keyed gate quartets,
all-empty H, report interface and route coverage."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

import test_p5_e2e as p5
from ipcch_geoxgb import learnmap, predict, report
from ipcch_geoxgb.artifacts import sha256_file, write_json
from ipcch_geoxgb.errors import TechnicalError
from ipcch_geoxgb.replay import replay_run

H = p5.H


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    for module in (learnmap, predict):
        mp.setattr(module, "load_experiment_contract", p5._contract)
    root = tmp_path_factory.mktemp("rv") / "run"
    p5._write_prepared(root, west_areas=56)
    learnmap.run_learn_map(root)
    predict.run_predict(root)
    report.run_report(root)
    mp.undo()
    return root


def _manifest_sha(root):
    return sha256_file(root / "prepared" / "prepared-manifest.json")


def _copy(run, tmp_path):
    dest = tmp_path / "copy"
    shutil.copytree(run, dest)
    return dest


def test_frozen_map_accepted_when_bound(run):
    record, region_of = predict.load_frozen(run / "stage1", H, _manifest_sha(run))
    assert record["H"] == H and len(region_of) == p5.N_AREAS


@pytest.mark.parametrize("damage", ["other_h", "other_prepared", "not_summary_record", "duplicate_area"])
def test_frozen_map_rejected_when_mislinked(run, tmp_path, damage):
    root = _copy(run, tmp_path)
    s1 = root / "stage1"
    rec_path = s1 / f"frozen_h{H:02d}.json"
    record = json.loads(rec_path.read_text())
    if damage == "other_h":  # an intact record of another H copied under this H's name
        record["H"] = 12
    elif damage == "other_prepared":
        record["prepared_manifest_sha256"] = "0" * 64
    elif damage == "not_summary_record":
        record["terminal_regions"] += 1
    else:
        map_path = s1 / f"frozen_map_h{H:02d}.csv"
        fmap = pd.read_csv(map_path, dtype={"node_id": str})
        pd.concat([fmap, fmap.iloc[:1]]).to_csv(map_path, index=False)
        record["map_sha256"] = sha256_file(map_path)
        summary = json.loads((s1 / "stage1-summary.json").read_text())
        summary["horizons"][str(H)]["frozen"] = record
        (s1 / "stage1-summary.json").write_text(json.dumps(summary))
    rec_path.write_text(json.dumps(record))
    with pytest.raises(TechnicalError):
        predict.load_frozen(s1, H, _manifest_sha(root))


def test_gate_pairs_carry_keyed_quartets_and_providers(run):
    pairs = sorted((run / "stage3" / f"h{H:02d}").glob("pairs_*.csv.gz"))
    assert pairs
    frame = pd.read_csv(pairs[0])
    for q in ("q2", "q3", "q4", "q5"):
        for side in ("global", "local_routed"):
            assert {f"{side}_{q}_raw", f"{side}_{q}_star"} <= set(frame.columns)
    fallback = frame[~frame["local_fit_ok"].astype(bool)]
    assert (fallback["local_routed_provider"] == fallback["global_identity"]).all()
    fitted = frame[frame["local_fit_ok"].astype(bool)]
    assert (fitted["local_routed_provider"] == fitted["local_identity"]).all()


def test_report_interface_and_route_coverage(run):
    rep = json.loads((run / "report" / "report.json").read_text())
    for period in ("main", "supplementary"):
        entry = rep["horizons"][str(H)][period]
        assert {"coverage", "routes", "E_all", "E_persist"} <= set(entry)
    supp = rep["horizons"][str(H)]["supplementary"]
    assert supp["E_all"]["status"] == "empty_cohort" and supp["E_all"]["na_reason"]
    routes = rep["horizons"][str(H)]["main"]["routes"]
    rows = routes["rows"]
    assert rows["local"] + rows["global_fallback"] + rows["global_only_no_accepted_split"] + \
        rows["unmapped_area_global"] == rows["denominator_cohort_rows"]
    assert routes["areas"]["in_learned_map"] + routes["areas"]["unmapped"] == routes["areas"]["denominator_cohort_areas"]
    assert routes["gate_region_folds"]["adopted_local"] >= 1
    month = pd.read_csv(run / "report" / f"diag_h{H:02d}_main_target_month.csv")
    empty = month[month["keys"] == 0]
    assert len(month) == 9 and len(empty) == 1 and (empty["status"] == "no_valid_target").all()
    assert {"delta_geo_minus_pool_f1", "na_reason_E_all", "delta_geo_minus_persistence_f1",
            "na_reason_E_persist", "persistence_coverage"} <= set(month.columns)


def test_all_empty_horizon_keeps_ledger_and_replays(tmp_path, monkeypatch):
    root = tmp_path / "run"
    p5._write_prepared(root)
    prepared = root / "prepared"
    cal = pd.read_csv(prepared / "fold_calendar.csv")
    cal["target_ord"] += 100  # every scheduled target month has no valid key
    cal["origin_ord"] = cal["target_ord"] - H
    cal["target_month"] = [f"{o // 12:04d}-{o % 12 + 1:02d}" for o in cal["target_ord"]]
    cal["fold_id"] = [f"main_h{H:02d}_{m}" for m in cal["target_month"]]
    cal.to_csv(prepared / "fold_calendar.csv", index=False)
    manifest = json.loads((prepared / "prepared-manifest.json").read_text())
    manifest["artifacts_sha256"]["fold_calendar.csv"] = sha256_file(prepared / "fold_calendar.csv")
    (prepared / "prepared-manifest.json").unlink()
    write_json(prepared / "prepared-manifest.json", manifest)
    for module in (learnmap, predict):
        monkeypatch.setattr(module, "load_experiment_contract", p5._contract)
    learnmap.run_learn_map(root)
    summary = predict.run_predict(root)
    assert summary["horizons"][str(H)]["scored_folds"] == 0 and summary["model_store"]["requests"] == 0
    ledger = pd.read_csv(root / "stage3" / f"h{H:02d}" / "fold_ledger.csv")
    assert len(ledger) == len(cal) and (ledger["status"] == "no_valid_target").all()
    assert len(pd.read_csv(root / "stage3" / f"h{H:02d}" / "predictions.csv.gz")) == 0
    report.run_report(root)
    result = replay_run(root, p5._contract())
    assert result["status"] == "passed", result["failures"][:10]
