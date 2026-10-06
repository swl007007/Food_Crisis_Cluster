"""Synthetic end to end: develop -> Stage3 -> report -> replay (design section 10).

World (H = 3): region r0 has q3 rising with x0, r1 falling with x0 (the pooled
residual cannot tell them apart; a regional residual can), r2 is too small for
any support, r3 is flat just above the 0.20 threshold (L cannot beat P: a supported
gain fallback), and some areas are outside the map. One scheduled fold has no
valid target.
"""

from __future__ import annotations

import copy
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from ipcch_mlp import contract as C
from ipcch_mlp import develop, planning, replay, report, runtime, sources, stage3
from ipcch_mlp.errors import TechnicalError
from ipcch_mlp.quartets import Engine
from ipcch_mlp.runtime import torch
from ipcch_mlp.store import ModelStore

H, M0, N_MONTHS = 3, 24000, 48
EMPTY = M0 + 44
REGIONS = {"r0": range(0, 20), "r1": range(20, 40), "r2": range(40, 44), "r3": range(44, 60)}
UNMAPPED = range(60, 66)
DEVICE = os.environ.get("IPCCH_MLP_TEST_DEVICE", "cpu")


def _label(o):
    return f"{o // 12:04d}-{o % 12 + 1:02d}"


def world():
    rng = np.random.default_rng(5)
    rows = [(a, M0 + m) for a in range(66) for m in range(N_MONTHS) if M0 + m != EMPTY]
    keys = pd.DataFrame(rows, columns=["admin_code", "target_ord"])
    keys["target_month"] = [_label(o) for o in keys["target_ord"]]
    x0 = rng.normal(size=len(keys))
    a = keys["admin_code"].to_numpy()
    slope = np.select([a < 20, a < 40], [0.15, -0.15], 0.0)
    flat = (a >= 44) & (a < 60)
    level = np.where(flat, 0.24, 0.22)
    noise = rng.normal(0, 1, len(keys)) * np.where(flat, 0.03, 0.02)
    q3 = np.clip(level + slope * x0 + noise, 0, 0.9)
    keys["q2"], keys["q3"], keys["q4"], keys["q5"] = np.minimum(q3 + 0.15, 1), q3, q3 * 0.4, q3 * 0.05
    phase = 1 + (keys[["q2", "q3", "q4", "q5"]].to_numpy() >= 0.2).sum(axis=1)
    keys["phase_truth"], keys["crisis_truth"] = phase, (phase >= 3).astype(int)
    keys["country_key"] = np.where(a % 3 == 0, "A", np.where(a % 3 == 1, "B", "C"))
    keys["horizon_months"], keys["origin_ord"] = H, keys["target_ord"] - H
    prev = keys[["admin_code", "target_ord", "phase_truth", "q3"]].rename(
        columns={"target_ord": "origin_ord", "phase_truth": "persistence_phase", "q3": "persistence_q3"})
    keys = keys.merge(prev, on=["admin_code", "origin_ord"], how="left")
    keys["persistence_available"] = keys["persistence_phase"].notna().astype(int)
    keys["persistence_phase"] = keys["persistence_phase"].fillna(0).astype(int)
    keys["persistence_source_month"] = np.where(keys["persistence_available"] == 1,
                                                [_label(o) for o in keys["origin_ord"]], "")
    keys["persistence_age_months"] = np.where(keys["persistence_available"] == 1, 0, -1)
    X = np.full((len(keys), 561), np.nan)
    X[:, 0] = x0
    X[:, 1:8] = rng.normal(size=(len(keys), 7))
    X[rng.random(len(keys)) < 0.1, 3] = np.nan
    region_of = {int(ar): n for n, r in REGIONS.items() for ar in r}
    hz = sources.Horizon(h=H, keys=keys, X=X, region_of=region_of, x_sha256="syn-x", keys_sha256="syn-k",
                         map_sha256="syn-map")
    folds = [{"period": "main", "horizon_months": H, "target_ord": t, "origin_ord": t - H} for t in range(M0 + 38, M0 + 47)]
    cal = pd.DataFrame(folds)
    cal["fold_id"] = [f"main_h{H:02d}_{_label(t)}" for t in cal["target_ord"]]
    split = keys[keys["target_ord"] < M0 + 30][["admin_code", "target_ord"]].rename(columns={"target_ord": "month_ord"})
    split["split_role"] = np.where(split["month_ord"] < M0 + 15, "fit", "validation")
    return hz, cal, split


def contract_for(hz, split):
    c = copy.deepcopy(C.load_contract())
    c["horizons_months"] = [H]
    c["replicates"] = [42]
    c["training"]["global_epochs"] = 6
    c["training"]["residual_epochs"] = 60
    c["support"]["local_fit"] = {"keys": 60, "areas": 8, "target_months": 3}
    c["support"]["validation"] = {"keys": 40, "areas": 8, "target_months": 3, "crisis_keys": 5, "noncrisis_keys": 5}
    roles = sources.split_roles(hz, split)
    c["development"] = {"fit_keys": int((roles == "fit").sum()), "validation_keys": int((roles == "validation").sum()),
                        "singletons": 0}
    return c


def p6_like(hz, cal):
    rows = np.concatenate([hz.rows_at(t) for t in cal["target_ord"]])
    k = hz.keys.iloc[rows]
    q3 = k["persistence_q3"].fillna(0.25).to_numpy()
    ph = np.where(k["persistence_available"] == 1, k["persistence_phase"], 3)
    return pd.DataFrame({"admin_code": k["admin_code"], "target_ord": k["target_ord"], "horizon_months": H,
                         "geo_phase": ph, "pool_phase": ph, "geo_q3_star": q3, "geo_q3_raw": q3,
                         "pool_q3_star": q3, "pool_q3_raw": q3, "phase_truth": k["phase_truth"]})


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    runtime.configure(DEVICE, 4)
    hz, cal, split = world()
    c = contract_for(hz, split)
    root = tmp_path_factory.mktemp("e2e") / "run"
    root.mkdir()
    mp = pytest.MonkeyPatch()
    mp.setattr(report.sources, "load_p6_predictions", lambda h, inputs=None: p6_like(hz, cal))
    enum = planning.enumerate_horizon(hz, cal, c)
    mp.setattr(report, "PREDECLARED_ELIGIBLE", {H: enum["main_historical_support_eligible_keys"]})
    store = ModelStore(root / "models", root / "model_requests.jsonl")
    engine = Engine(store, c, {"synthetic": True}, DEVICE)
    develop.run_develop(root, engine, c, {H: hz}, split)
    winners = develop.load_winners(root)
    stage3.run_stage3(root, engine, c, {H: hz}, cal, winners)
    report.run_report(root, c)
    yield {"root": root, "hz": hz, "cal": cal, "split": split, "contract": c, "enum": enum, "mp": mp, "store": store}
    mp.undo()


def _pred(run):
    return report.read_predictions(run["root"] / "stage3" / "rep42" / f"h{H:02d}" / "predictions.csv.gz")


def _gates(run):
    path = run["root"] / "stage3" / "rep42" / f"h{H:02d}" / "gate_decisions.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_routes_cover_every_case(run):
    pred = _pred(run)
    routes = set(pred["route"])
    assert "local" in routes, routes
    assert "unmapped_area_pool" in routes
    assert any(r.startswith("pool_fallback:gate_support") for r in routes)
    assert "pool_fallback:gain_not_above_threshold" in routes, routes
    ledger = pd.read_csv(run["root"] / "stage3" / "rep42" / f"h{H:02d}" / "fold_ledger.csv")
    assert ledger["status"].tolist().count("no_valid_target") == 1


def test_gate_rejected_local_exists_and_cannot_alter_geo(run):
    pred = _pred(run)
    rejected = pred[pred["route"] == "pool_fallback:gain_not_above_threshold"]
    assert (rejected["local_eligible"] == 1).any()
    for q in ("q2", "q3", "q4", "q5"):
        assert np.array_equal(rejected[f"geo_{q}_raw"].to_numpy(), rejected[f"pool_{q}_raw"].to_numpy())
    assert np.array_equal(rejected["geo_phase"], rejected["pool_phase"])
    tiny = pred[pred["region"] == "r2"]
    assert (tiny["local_eligible"] == 0).all()


def test_coverage_matches_no_fit_enumeration(run):
    rep = json.loads((run["root"] / "report" / "report.json").read_text())
    cov = rep["replicates"]["42"][str(H)]["main"]["coverage"]
    assert cov["historical_support_eligible_keys"] == run["enum"]["main_historical_support_eligible_keys"]
    assert cov["adopted_keys"] <= cov["historical_support_eligible_keys"]


def test_fit_counts_match_enumeration(run):
    fits = {}
    for line in (run["root"] / "model_requests.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["status"] == "fit":
            fits.setdefault(e["stage"], set()).add(e["identity_sha256"])
    assert len(fits["develop"]) == 4 * (2 + 4)
    assert len(fits["stage3"]) == run["enum"]["scalar_fits_per_seed"]


def test_replay_passes_and_reproduces_everything(run):
    c = run["contract"]
    res = replay.run_replay(run["root"], c, {"synthetic": True}, DEVICE, {H: run["hz"]}, run["split"], run["cal"],
                            expect_inventory=False)
    assert res["status"] == "passed", res["failures"][:5]
    assert res["replay_store_counts"]["fits"] == 0 and res["checks_passed"] > 100


def _copy(run, tmp_path):
    dst = tmp_path / "run"
    shutil.copytree(run["root"], dst, ignore=shutil.ignore_patterns("replay"))
    return dst


def test_replay_catches_tampered_prediction(run, tmp_path):
    dst = _copy(run, tmp_path)
    path = dst / "stage3" / "rep42" / f"h{H:02d}" / "predictions.csv.gz"
    pred = report.read_predictions(path)
    pred.loc[0, "pool_phase"] = 5 if pred.loc[0, "pool_phase"] != 5 else 1
    pred.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    res = replay.run_replay(dst, run["contract"], {"synthetic": True}, DEVICE, {H: run["hz"]}, run["split"], run["cal"],
                            expect_inventory=False)
    assert res["status"] == "failed"


def test_replay_catches_corrupted_model(run, tmp_path):
    dst = _copy(run, tmp_path)
    some = next((dst / "models" / "models").glob("*/*/state.pt"))
    state = torch.load(some, weights_only=True)
    key = sorted(state)[0]
    state[key].view(-1)[0] += 0.5
    torch.save(state, some)
    with pytest.raises(TechnicalError):
        replay.run_replay(dst, run["contract"], {"synthetic": True}, DEVICE, {H: run["hz"]}, run["split"], run["cal"],
                          expect_inventory=False)


def test_changed_map_identity_requires_new_models(run, tmp_path):
    dst = _copy(run, tmp_path)
    hz = run["hz"]
    moved = dict(hz.region_of)
    moved[0] = "r1"
    hz2 = sources.Horizon(h=H, keys=hz.keys, X=hz.X, region_of=moved, x_sha256="syn-x", keys_sha256="syn-k",
                          map_sha256="syn-map-2")
    with pytest.raises(TechnicalError):
        replay.run_replay(dst, run["contract"], {"synthetic": True}, DEVICE, {H: hz2}, run["split"], run["cal"],
                          expect_inventory=False)


def test_temporal_cutoff_of_fitting_pools(run):
    hz = run["hz"]
    for o in (M0 + 35, M0 + 40):
        rows = hz.window_rows(o)
        t = hz.keys["target_ord"].to_numpy()[rows]
        assert t.max() <= o and t.min() >= o - 35


def test_replay_inventory_counts_store_entries(run):
    c = copy.deepcopy(run["contract"])
    fits = 0
    for line in (run["root"] / "model_requests.jsonl").read_text().splitlines():
        fits += json.loads(line)["status"] == "fit"
    c["expected"] = {"development_scalar_fits": 24, "stage3_scalar_fits_per_seed": run["enum"]["scalar_fits_per_seed"],
                     "total_scalar_fits": fits}
    chk = replay.Checker()
    counts = replay.inventory(chk, run["root"], c)
    assert not chk.failures, chk.failures
    assert counts["develop"] == 24
