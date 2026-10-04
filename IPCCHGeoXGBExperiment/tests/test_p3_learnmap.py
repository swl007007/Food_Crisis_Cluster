"""P3 end-to-end on a synthetic prepared run: real quartets, real adjacency, small contract."""

from __future__ import annotations

import copy
import json
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from ipcch_geoxgb import learnmap, metrics
from ipcch_geoxgb.artifacts import sha256_file, write_json
from ipcch_geoxgb.contract import load_experiment_contract

N_AREAS, N_MONTHS = 80, 12


def _contract():
    c = copy.deepcopy(load_experiment_contract())
    c["calendar"]["horizons_months"] = [1]
    # enough rounds at the frozen eta .05 for the regional regime to be learnable
    c["model"]["global_recipes"] = {"G1": {"max_depth": 3, "rounds": 100}, "G2": {"max_depth": 3, "rounds": 150}}
    c["model"]["local_recipes"] = {"L1": {"max_depth": 1, "appended_rounds": 60}, "L2": {"max_depth": 2, "appended_rounds": 80}}
    c["support"]["local_fit"] = {"keys": 30, "areas": 5, "target_months": 2}
    c["support"]["validation"] = {"keys": 30, "areas": 5, "target_months": 2, "crisis_keys": 5, "noncrisis_keys": 5}
    c["partition"]["scan_iterations"] = 30
    return c


def _prepared(run):
    rng = np.random.default_rng(3)
    prepared = run / "prepared"
    prepared.mkdir(parents=True)
    rows = [(a, 24000 + t) for a in range(N_AREAS) for t in range(N_MONTHS)]
    keys = pd.DataFrame(rows, columns=["admin_code", "target_ord"])
    keys["target_month"] = [f"{o // 12:04d}-{o % 12 + 1:02d}" for o in keys["target_ord"]]
    x0 = rng.normal(size=len(keys))
    west = keys["admin_code"].to_numpy() < N_AREAS // 2
    q3 = np.clip(0.2 + np.where(west, 0.15, -0.15) * x0 + rng.normal(0, 0.03, len(keys)), 0, 0.9)
    keys["q2"], keys["q3"], keys["q4"], keys["q5"] = np.minimum(q3 + 0.2, 1), q3, q3 * 0.4, q3 * 0.05
    phase = 1 + (keys[["q2", "q3", "q4", "q5"]].to_numpy() >= 0.2).sum(axis=1)
    keys["phase_truth"], keys["crisis_truth"] = phase, (phase >= 3).astype(int)
    X = np.full((len(keys), 561), np.nan)
    X[:, 0], X[:, 1] = x0, rng.normal(size=len(keys))
    np.save(prepared / "X_rich561_h01.npy", X)
    keys.to_csv(prepared / "keys_h01.csv.gz", index=False)
    split = keys[["admin_code", "target_ord", "phase_truth", "crisis_truth"]].rename(columns={"target_ord": "month_ord"})
    split["split_role"] = np.where(split["month_ord"] < 24000 + N_MONTHS // 2, "fit", "validation")
    split.to_csv(prepared / "stage1_split.csv.gz", index=False)
    for name in ("target_ledger.csv.gz", "target_ledger_valid.csv.gz", "fold_calendar.csv", "coverage_2026.csv",
                 "feature_order.csv"):  # complete inventory; contents unused by learn-map
        (prepared / name).write_bytes(b"placeholder\n")
    names = ("X_rich561_h01.npy", "keys_h01.csv.gz", "stage1_split.csv.gz", "target_ledger.csv.gz",
             "target_ledger_valid.csv.gz", "fold_calendar.csv", "coverage_2026.csv", "feature_order.csv")
    artifacts = {n: sha256_file(prepared / n) for n in names}
    write_json(prepared / "prepared-manifest.json",
               {"artifacts_sha256": artifacts, "availability_policy": {"id": "observation-month-end-v1"}})
    return keys


@pytest.fixture(scope="module")
def learned(tmp_path_factory, monkeypatch_module):
    run = tmp_path_factory.mktemp("run")
    _prepared(run)
    monkeypatch_module.setattr(learnmap, "load_experiment_contract", _contract)
    return run, learnmap.run_learn_map(run)


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_learn_map_outputs_and_reproducible_selection(learned):
    run, summary = learned
    h = summary["horizons"]["1"]
    selection = json.loads((run / "stage1" / "h01" / "selection.json").read_text())
    assert len(selection["candidates"]) == 4 and selection["selection"]["status"] == "selected"
    # recompute every candidate's exact F1 from its saved keyed S predictions (no training)
    for entry in selection["candidates"]:
        pred = pd.read_csv(run / "stage1" / "h01" / entry["candidate"] / "s_predictions.csv.gz")
        f1 = metrics.exact_f1(metrics.crisis_counts(pred["phase_truth"], pred["phase_pred"]))
        assert str(f1) == entry["f1_exact"]
        assert len(pred) == h["validation_keys"]  # complete common S set, every candidate
    frozen = json.loads((run / "stage1" / "frozen_h01.json").read_text())
    assert frozen["candidate"] == h["winner"] == selection["selection"]["winner"]
    fmap = pd.read_csv(run / "stage1" / "frozen_map_h01.csv")  # default dtype inference on purpose
    assert len(fmap) == N_AREAS and frozen["map_sha256"] == sha256_file(run / "stage1" / "frozen_map_h01.csv")
    assert fmap["node_id"].map(type).eq(str).all() and fmap["node_id"].str.startswith("r").all()
    assert set(frozen["connectivity"]) == set(fmap["node_id"])
    assert sum(v["areas"] for v in frozen["connectivity"].values()) == N_AREAS


def test_roots_fit_once_per_g_and_are_shared_across_l(learned):
    run, summary = learned
    ledger = [json.loads(line) for line in (run / "stage1" / "model_requests.jsonl").read_text().splitlines()]
    roots = [e for e in ledger if e["purpose"] == "root_global"]
    assert sorted(e["G"] for e in roots) == ["G1", "G2"] and all(e["status"] == "fit" for e in roots)
    assert summary["model_store"]["fits"] == len(ledger) - summary["model_store"]["hits"]


def test_split_is_found_on_a_two_regime_world(learned):
    run, summary = learned
    for name in ("G1L1", "G2L2"):
        decisions = json.loads((run / "stage1" / "h01" / name / "decisions.json").read_text())["decisions"]
        root = decisions[0]
        assert root["node_id"] == "r" and root["scan_groups"] == N_AREAS
        assert root["outcome"] in {"accepted", "rejected_gate"}  # a legitimate, recorded decision
    accepted = [c for c, v in summary["horizons"]["1"]["candidates"].items() if v["accepted_splits"] > 0]
    assert accepted, "the west/east regime difference should produce at least one accepted split"


def test_tampered_prepared_artifact_stops(tmp_path, monkeypatch):
    run = tmp_path / "run"
    _prepared(run)
    path = run / "prepared" / "keys_h01.csv.gz"
    path.write_bytes(path.read_bytes() + b"\x00")
    monkeypatch.setattr(learnmap, "load_experiment_contract", _contract)
    from ipcch_geoxgb.errors import TechnicalError

    with pytest.raises(TechnicalError, match="does not match its manifest"):
        learnmap.run_learn_map(run)
