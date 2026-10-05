"""P4: rolling Stage3 folds, R37 gate replay, routing, reuse and stop conditions (synthetic)."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from ipcch_geoxgb import schedule, stage3
from ipcch_geoxgb.contract import load_experiment_contract
from ipcch_geoxgb.errors import ContractError
from ipcch_geoxgb.modelstore import ModelStore

M0 = 24000  # month ordinal of 2000-01
N_MONTHS = 60
REGION = {a: ("r0" if a < 1030 else "r1") for a in range(1000, 1060)}


def _contract():
    c = copy.deepcopy(load_experiment_contract())
    c["model"]["global_recipes"]["G1"] = {"max_depth": 2, "rounds": 30}
    c["model"]["local_recipes"]["L1"] = {"max_depth": 1, "appended_rounds": 30}
    c["support"]["local_fit"] = {"keys": 40, "areas": 5, "target_months": 2}
    c["support"]["validation"] = {"keys": 20, "areas": 5, "target_months": 2, "crisis_keys": 3, "noncrisis_keys": 3}
    return c


def _world(seed=0, unmapped=(1058, 1059), skip_month=None):
    rng = np.random.default_rng(seed)
    rows = [(a, M0 + m) for a in range(1000, 1060) for m in range(0, N_MONTHS, 2) if M0 + m != skip_month]
    keys = pd.DataFrame(rows, columns=["admin_code", "target_ord"])
    keys["target_month"] = [f"{o // 12:04d}-{o % 12 + 1:02d}" for o in keys["target_ord"]]
    x0 = rng.normal(size=len(keys))
    west = keys["admin_code"].to_numpy() < 1030
    q3 = np.clip(0.2 + np.where(west, 0.2, -0.2) * x0 + rng.normal(0, 0.02, len(keys)), 0, 0.9)
    keys["q2"], keys["q3"], keys["q4"], keys["q5"] = np.minimum(q3 + 0.15, 1), q3, q3 * 0.4, q3 * 0.05
    phase = 1 + (keys[["q2", "q3", "q4", "q5"]].to_numpy() >= 0.2).sum(axis=1)
    keys["phase_truth"], keys["crisis_truth"] = phase, (phase >= 3).astype(int)
    keys["country_key"] = "A"
    keys["persistence_available"] = 1
    keys["persistence_phase"] = 2
    keys["persistence_q3"] = 0.1
    keys["persistence_source_month"] = "1999-12"
    keys["persistence_age_months"] = 1
    X = np.column_stack([x0, rng.normal(size=len(keys)), np.full(len(keys), np.nan)])
    region_of = {a: r for a, r in REGION.items() if a not in unmapped}
    return keys, X, region_of


def _ctx(tmp_path, local_enabled=True, region_of=None, keys=None, X=None, store=None):
    k, x, r = _world()
    keys = k if keys is None else keys
    X = x if X is None else X
    store = store or ModelStore(tmp_path / "models", tmp_path / "ledger.jsonl")
    return stage3.HorizonContext(
        h=3, keys=keys, X=X, artifact_sha={"X": "x" * 64, "keys": "k" * 64},
        region_of=r if region_of is None else region_of, local_enabled=local_enabled,
        gid="G1", lid="L1", contract=_contract(), store=store, base_identity={"env": "test"},
        observed_months=np.unique(keys["target_ord"]),
    )


def _fold(target, period="main"):
    return {"fold_id": f"t_{target}", "period": period, "target_ord": target, "origin_ord": target - 3}


def test_window_is_inclusive_and_cut_at_origin(tmp_path):
    ctx = _ctx(tmp_path)
    rows = ctx.window_rows(M0 + 40)
    months = ctx.keys["target_ord"].to_numpy()[rows]
    assert months.max() == M0 + 40 and months.min() == M0 + 40 - 34  # observed even months in [O-35, O]
    assert not np.isin(M0 + 41, months)


def test_gate_dates_are_latest_six_observed_before_origin():
    observed = np.array([M0 + m for m in range(0, 60, 2)])
    dates = schedule.historical_gate_dates(observed, M0 + 37)
    assert dates.tolist() == [M0 + 36, M0 + 34, M0 + 32, M0 + 30, M0 + 28, M0 + 26]


def test_empty_fold_is_ledger_only_and_requests_no_fit(tmp_path):
    ctx = _ctx(tmp_path)
    out = stage3.run_fold(ctx, _fold(M0 + 41))  # odd month: no valid target
    assert out["ledger"]["status"] == "no_valid_target" and out["predictions"] is None
    assert ctx.store.counts["requests"] == 0


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("s3")
    ctx = _ctx(tmp)
    first = stage3.run_fold(ctx, _fold(M0 + 50))
    counts_after_first = dict(ctx.store.counts)
    second = stage3.run_fold(ctx, _fold(M0 + 52))
    return ctx, first, second, counts_after_first


def test_scored_fold_keeps_every_key_and_routes_unmapped_to_global(scored):
    ctx, first, _, _ = scored
    pred = first["predictions"]
    assert len(pred) == 60 and set(pred["admin_code"]) == set(range(1000, 1060))
    unmapped = pred[pred["admin_code"].isin([1058, 1059])]
    assert (unmapped["route"] == "unmapped_area_global").all()
    assert (unmapped["geo_phase"] == unmapped["pool_phase"]).all()
    not_local = pred["route"] != "local"
    for t in ("q2", "q3", "q4", "q5"):  # non-local rows are exactly the pooled quartet
        assert (pred.loc[not_local, f"geo_{t}_raw"] == pred.loc[not_local, f"pool_{t}_raw"]).all()
    assert pred["persistence_phase"].eq(2).all()


def test_gate_uses_six_lawful_dates_and_pools_regions(scored):
    ctx, first, _, _ = scored
    origin = M0 + 47
    assert first["ledger"]["gate_dates"] == [M0 + 46, M0 + 44, M0 + 42, M0 + 40, M0 + 38, M0 + 36]
    pairs = first["pairs"]
    assert set(pairs["validation_month"]) == set(first["ledger"]["gate_dates"])
    assert (pairs["internal_origin"] == pairs["validation_month"] - 3).all()
    assert (pairs["validation_month"] < origin).all()  # current truth T never enters the gate
    assert set(pairs["region"]) == {"r0", "r1"} and len(first["gate"]) == 2
    for d in first["gate"]:
        assert d["keys"] == int((pairs["region"] == d["region"]).sum())


def test_regional_regime_is_adopted_locally(scored):
    ctx, first, _, _ = scored
    routes = {d["region"]: d["route"] for d in first["gate"]}
    assert "local" in routes.values(), first["gate"]
    pred = first["predictions"]
    local = pred["route"] == "local"
    assert local.any() and (pred.loc[local, "provider"] != pred.loc[local, "global_identity"]).all()


def test_identical_requests_reuse_models_across_folds(scored):
    ctx, first, second, after_first = scored
    # fold T+2 shares five of six gate dates with fold T -> those globals/locals are cache hits
    assert ctx.store.counts["hits"] > after_first["hits"]
    shared = set(first["ledger"]["gate_dates"]) & set(second["ledger"]["gate_dates"])
    assert len(shared) == 5


def test_no_accepted_split_is_global_only_without_local_fits(tmp_path):
    ctx = _ctx(tmp_path, local_enabled=False)
    out = stage3.run_fold(ctx, _fold(M0 + 50))
    assert out["pairs"] is None and out["gate"] == []
    assert set(out["predictions"]["route"]) <= {"global_only_no_accepted_split", "unmapped_area_global"}
    ledger = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert all(e["purpose"] == "global" for e in ledger) and len(ledger) == 1


def test_changed_region_membership_does_not_collide(tmp_path):
    store = ModelStore(tmp_path / "m", tmp_path / "l.jsonl")
    a = _ctx(tmp_path, store=store)
    moved = dict(REGION)
    moved[1029] = "r1"
    b = _ctx(tmp_path, store=store, region_of=moved)
    ga = a.global_quartet(M0 + 40, {})
    gb = b.global_quartet(M0 + 40, {})
    assert ga[0] == gb[0]  # same pooled global identity
    la, _ = a.local_quartet(M0 + 40, "r1", ga[:2], {})
    lb, _ = b.local_quartet(M0 + 40, "r1", gb[:2], {})
    assert la[0] != lb[0]


def test_empty_required_global_pool_stops(tmp_path):
    ctx = _ctx(tmp_path)
    with pytest.raises(ContractError, match="R40"):
        ctx.global_quartet(M0 - 100, {})


def _pairs(n_dates, ok_dates, local_better):
    rows = []
    for d in range(n_dates):
        for a in range(10):
            truth = 3 if a < 5 else 1
            glob = 3 if a < 3 else 1  # misses two crises per date
            loc = truth if local_better else glob
            rows.append({"region": "r0", "admin_code": a, "validation_month": d, "phase_truth": truth,
                         "phase_global": glob, "phase_local_routed": loc if d < ok_dates else glob,
                         "local_fit_ok": d < ok_dates})
    return pd.DataFrame(rows)


def test_gate_counts_successful_dates_and_keeps_fallback_keys():
    c = _contract()
    d = stage3.gate_decision(_pairs(6, 2, True), c)
    assert d["keys"] == 60 and d["local_fit_dates"] == 2 and not d["enabled"] and "local_fit_dates" in d["reason"]
    d = stage3.gate_decision(_pairs(6, 3, True), c)
    assert d["local_fit_dates"] == 3 and d["enabled"]


def test_gate_gain_exactly_one_percent_is_not_enough():
    c = _contract()
    # global: TP=99 FN=2 -> 198/200 = .99 ; local: TP=100 FN=1 -> 200/201 < 1
    base = [{"region": "r", "admin_code": i % 30, "validation_month": i % 3, "phase_truth": 3,
             "phase_global": 3 if i < 99 else 1, "phase_local_routed": 3 if i < 100 else 1, "local_fit_ok": True}
            for i in range(101)]
    neg = [{"region": "r", "admin_code": i % 30, "validation_month": i % 3, "phase_truth": 1, "phase_global": 1,
            "phase_local_routed": 1, "local_fit_ok": True} for i in range(20)]
    pairs = pd.DataFrame(base + neg)
    d = stage3.gate_decision(pairs, c)
    assert not d["enabled"] and d["reason"] == "gain_not_above_threshold"
    exact = pd.DataFrame(base + neg)
    exact["phase_global"] = [3 if i < 99 else 1 for i in range(101)] + [1] * 20
    exact["phase_local_routed"] = [3] * 101 + [1] * 20  # F1 1 vs 198/200: gain exactly .01
    d = stage3.gate_decision(exact, c)
    assert d["f1_local"] == "1" and d["f1_global"] == "99/100" and not d["enabled"]
