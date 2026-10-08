"""Synthetic end to end (design section 9): annual blocks -> gates -> predictions -> report -> zero-fit replay.

World (H = 3, first main target 2023-04, which has no truth): region r0 rises
with x0, r1 falls with x0 (only a regional continuation can tell them apart),
r2 is too small for any support, r3 is flat above 0.20 (supported gain
fallback), and some areas are unmapped. 2024-12 is an empty fold; 2026-01..04
is the supplementary block.
"""

from __future__ import annotations

import copy
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from ipcch_yearly_xgb import contract as C
from ipcch_yearly_xgb import engine, planning, replay, report, run, schedule, sources
from ipcch_yearly_xgb.errors import TechnicalError
from ipcch_yearly_xgb.modelstore import ModelStore

M = schedule.parse_month
H = 3
REGIONS = {"r0": range(0, 20), "r1": range(20, 40), "r2": range(40, 44), "r3": range(44, 60)}
EMPTY = (M("2023-04"), M("2024-12"))
ENV = {"synthetic": True}


def world(truth_shift_from=None):
    rng = np.random.default_rng(3)
    months = [m for m in range(M("2019-01"), M("2026-04") + 1) if m not in EMPTY]
    rows = [(a, m) for a in range(66) for m in months]
    keys = pd.DataFrame(rows, columns=["admin_code", "target_ord"])
    keys["target_month"] = [schedule.month_label(o) for o in keys["target_ord"]]
    x0 = rng.normal(size=len(keys))
    a = keys["admin_code"].to_numpy()
    slope = np.select([a < 20, a < 40], [0.15, -0.15], 0.0)
    flat = (a >= 44) & (a < 60)
    q3 = np.clip(np.where(flat, 0.24, 0.22) + slope * x0 + rng.normal(0, 1, len(keys)) * np.where(flat, .03, .02), 0, .9)
    if truth_shift_from is not None:  # later outcomes change; earlier gates must not
        later = keys["target_ord"].to_numpy() >= truth_shift_from
        q3 = np.where(later, np.clip(q3 + 0.3, 0, 0.9), q3)
    keys["q2"], keys["q3"], keys["q4"], keys["q5"] = np.minimum(q3 + .15, 1), q3, q3 * .4, q3 * .05
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
                                                [schedule.month_label(o) for o in keys["origin_ord"]], "")
    keys["persistence_age_months"] = np.where(keys["persistence_available"] == 1, 0, -1)
    X = np.full((len(keys), 561), np.nan)
    X[:, 0], X[:, 1] = x0, rng.normal(size=len(keys))
    X[:, 2] = x0  # duplicate signal so column subsampling cannot hide it
    region_of = {int(ar): n for n, r in REGIONS.items() for ar in r}
    hz = sources.Horizon(h=H, keys=keys, X=X, region_of=region_of, x_sha256="syn-x", keys_sha256="syn-k",
                         map_sha256="syn-map", lineage={})
    cal = [("main", t) for t in range(M("2023-04"), M("2025-12") + 1)] + \
          [("supplementary", t) for t in range(M("2026-01"), M("2026-04") + 1)]
    calendar = pd.DataFrame({"period": [p for p, _ in cal], "horizon_months": H, "target_ord": [t for _, t in cal]})
    calendar["origin_ord"] = calendar["target_ord"] - H
    calendar["fold_id"] = [f"{p[:4]}_h03_{schedule.month_label(t)}" for p, t in cal]
    calendar["eval_keys"] = [int((keys["target_ord"] == t).sum()) for _, t in cal]
    return hz, calendar


def contract_small():
    c = copy.deepcopy(C.load_contract())
    c["horizons_months"] = [H]
    c["model"]["global_recipes"]["G3"]["rounds"] = 30
    c["model"]["local_recipes"]["L2"]["appended_rounds"] = 30
    c["support"]["local_fit"] = {"keys": 60, "areas": 8, "target_months": 6}
    c["support"]["validation"] = {"keys": 40, "areas": 8, "target_months": 3, "crisis_keys": 5, "noncrisis_keys": 5}
    return c


def p6_like(hz):
    k = hz.keys[hz.keys["target_ord"] >= M("2023-04")]
    q3 = k["persistence_q3"].fillna(0.25).to_numpy()
    ph = np.where(k["persistence_available"] == 1, k["persistence_phase"], 3)
    return pd.DataFrame({"admin_code": k["admin_code"], "target_ord": k["target_ord"], "horizon_months": H,
                         "geo_phase": ph, "pool_phase": ph, "geo_q3_star": q3, "geo_q3_raw": q3,
                         "pool_q3_star": q3, "pool_q3_raw": q3, "phase_truth": k["phase_truth"], "q3_truth": k["q3"]})


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    hz, cal = world()
    c = contract_small()
    root = tmp_path_factory.mktemp("yearly") / "run"
    root.mkdir()
    run.run_predict(root, {H: hz}, cal, c, ENV, {"synthetic": "inventory"})
    report.run_report(root, c, p6_loader=lambda h: p6_like(hz))
    return {"root": root, "hz": hz, "cal": cal, "c": c}


def _pred(r):
    return report.read_predictions(r["root"] / "predict" / "h03" / "predictions.csv.gz")


def _gates(r):
    return [json.loads(x) for x in (r["root"] / "predict" / "h03" / "gate_decisions.jsonl").read_text().splitlines()]


def _blocks(r):
    return json.loads((r["root"] / "predict" / "h03" / "block_ledger.json").read_text())


def test_blocks_anchor_and_fit_origins(run_dir):
    b = _blocks(run_dir)
    assert [(x["block_id"], x["anchor"], x["fit_origin"]) for x in b] == [
        ("main_h03_2023", "2023-04", "2023-01"), ("main_h03_2024", "2024-01", "2023-10"),
        ("main_h03_2025", "2025-01", "2024-10"), ("supp_h03_2026", "2026-01", "2025-10")]
    folds = pd.read_csv(run_dir["root"] / "predict" / "h03" / "fold_ledger.csv")
    assert len(folds) == 37 and folds["status"].tolist().count("no_valid_target") == 2


def test_row_origin_varies_but_fit_origin_fixed(run_dir):
    p = _pred(run_dir)
    for bid, part in p.groupby("block_id"):
        assert part["fit_origin_ord"].nunique() == 1 and part["row_origin_ord"].nunique() > 1
        assert (part["row_origin_ord"] == part["target_ord"] - H).all()


def test_routes_cover_every_case(run_dir):
    p = _pred(run_dir)
    routes = set(p["route"])
    assert "local" in routes and "unmapped_area_pool" in routes
    assert any(r.startswith("pool_fallback:gate_support") for r in routes)
    assert "pool_fallback:gain_not_above_threshold" in routes, routes
    rej = p[p["route"] == "pool_fallback:gain_not_above_threshold"]
    assert (rej["local_eligible"] == 1).any()  # ungated diagnostic under a rejected gate
    assert np.array_equal(rej["geo_q3_raw"], rej["pool_q3_raw"])
    assert (p.loc[p["region"] == "r2", "local_eligible"] == 0).all()


def test_same_year_historical_model_reused(run_dir):
    lines = [json.loads(x) for x in (run_dir["root"] / "predict" / "model_requests.jsonl").read_text().splitlines()]
    o2023 = M("2023-01")
    fits = [e for e in lines if e["status"] == "fit" and e["role"] == "global" and e["fit_origin"] == o2023]
    reuse = [e for e in lines if e["role"] == "global" and e["fit_origin"] == o2023 and e.get("use") == "gate"]
    assert len(fits) == 1 and reuse and all(e["status"] != "fit" for e in reuse)


def test_inventory_matches_no_fit_enumeration(run_dir):
    r = planning.enumerate_horizon(run_dir["hz"], run_dir["cal"], run_dir["c"])
    lines = [json.loads(x) for x in (run_dir["root"] / "predict" / "model_requests.jsonl").read_text().splitlines()]
    fits = [e for e in lines if e["status"] == "fit"]
    assert len([e for e in fits if e["role"] == "global"]) == r["global_quartets"]
    assert len([e for e in fits if e["role"] == "local"]) == r["local_quartets"]
    p = _pred(run_dir)
    assert int(p["local_eligible"].sum()) == sum(b["diagnostic_keys"] for b in r["blocks"])


def test_gates_unchanged_by_later_outcomes(run_dir, tmp_path):
    hz2, cal = world(truth_shift_from=M("2023-02"))  # outcomes after the first block origin change
    store = ModelStore(tmp_path / "m", tmp_path / "l.jsonl")
    eng = engine.Engine(hz2, run_dir["c"], store, ENV)
    first = schedule.blocks(cal, H, M("2023-04"))[0]
    again = eng.run_block(first)["gates"]
    orig = [g for g in _gates(run_dir) if g["block_id"] == first.block_id]
    keys = ("enabled", "reason", "historical_support", "keys", "local_fit_dates", "f1_pool", "f1_local")
    assert [{k: g.get(k) for k in keys} for g in again] == [{k: g.get(k) for k in keys} for g in orig]


def test_replay_passes_with_zero_fits(run_dir):
    res = replay.run_replay(run_dir["root"], {H: run_dir["hz"]}, run_dir["cal"], run_dir["c"], ENV,
                            {"synthetic": "inventory"}, lambda h: p6_like(run_dir["hz"]))
    assert res["status"] == "passed", res["failures"][:5]
    assert res["checks_passed"] > 200


def _copy(r, tmp_path):
    dst = tmp_path / "run"
    shutil.copytree(r["root"], dst, ignore=shutil.ignore_patterns("replay"))
    return dst


def _replay(r, dst):
    return replay.run_replay(dst, {H: r["hz"]}, r["cal"], r["c"], ENV, {"synthetic": "inventory"},
                             lambda h: p6_like(r["hz"]))


def test_replay_catches_within_block_route_change(run_dir, tmp_path):
    dst = _copy(run_dir, tmp_path)
    path = dst / "predict" / "h03" / "predictions.csv.gz"
    p = report.read_predictions(path)
    i = p.index[p["route"] == "local"][0]
    p.loc[i, "route"] = "pool_fallback:gain_not_above_threshold"
    p.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    res = _replay(run_dir, dst)
    assert res["status"] == "failed" and any("route_fixed" in f for f in res["failures"])


def test_replay_catches_omitted_validation_key(run_dir, tmp_path):
    dst = _copy(run_dir, tmp_path)
    path = next((dst / "predict" / "h03").glob("pairs_main_h03_2024.csv.gz"))
    pairs = pd.read_csv(path)
    pairs.iloc[1:].to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    res = _replay(run_dir, dst)
    assert res["status"] == "failed" and any("validation_keys_complete" in f for f in res["failures"])


def test_replay_catches_changed_fit_origin(run_dir, tmp_path):
    dst = _copy(run_dir, tmp_path)
    path = dst / "predict" / "h03" / "block_ledger.json"
    b = json.loads(path.read_text())
    b[1]["fit_origin_ord"] -= 1
    path.write_text(json.dumps(b))
    res = _replay(run_dir, dst)
    assert res["status"] == "failed" and any(":origin" in f for f in res["failures"])


def test_replay_catches_corrupt_model_and_wrong_parent(run_dir, tmp_path):
    dst = _copy(run_dir, tmp_path)
    local = next(p for p in (dst / "models").glob("*/*") if
                 json.loads((p / "record.json").read_text())["identity"]["scope"] == "yearly-local")
    rec = json.loads((local / "record.json").read_text())
    rec["identity"]["global_identity"] = "0" * 64
    (local / "record.json").write_text(json.dumps(rec))
    with pytest.raises(TechnicalError):
        _replay(run_dir, dst)
    dst2 = _copy(run_dir, tmp_path / "b")
    some = next((dst2 / "models").glob("*/*/q2.ubj"))
    some.write_bytes(some.read_bytes()[:-8] + b"\x00" * 8)
    with pytest.raises(TechnicalError):
        _replay(run_dir, dst2)


def test_replay_catches_changed_fit_input_x(run_dir, tmp_path):
    dst = _copy(run_dir, tmp_path)
    hz = run_dir["hz"]
    X2 = np.array(hz.X, copy=True)
    X2[0, 5] = 123.0  # a row inside every fitting pool
    hz2 = sources.Horizon(h=H, keys=hz.keys, X=X2, region_of=hz.region_of, x_sha256="syn-x", keys_sha256="syn-k",
                          map_sha256="syn-map", lineage={})
    with pytest.raises(TechnicalError):  # the reconstructed identity no longer exists: replay refuses to fit
        replay.run_replay(dst, {H: hz2}, run_dir["cal"], run_dir["c"], ENV, {"synthetic": "inventory"},
                          lambda h: p6_like(hz))


def test_replay_catches_tampered_fit_sidecar(run_dir, tmp_path):
    dst = _copy(run_dir, tmp_path)
    side = next((dst / "models").glob("*/*/fit_keys.npy"))
    k = np.load(side)
    k[0, 1] += 1
    np.save(side, k)
    with pytest.raises(TechnicalError):
        _replay(run_dir, dst)


def test_report_local_persistence_cohort(run_dir):
    rep = json.loads((run_dir["root"] / "report" / "report.json").read_text())
    p = _pred(run_dir)
    for period in ("main", "supplementary"):
        e = p[p["period"] == period]
        lp = rep["horizons"]["3"][period]["local_persistence_matched"]
        assert lp["keys"] == int(((e["local_eligible"] == 1) & (e["persistence_available"] == 1)).sum())
        assert set(lp["panels"]) == {"local", "pool", "geo", "persistence"}
        assert "local_minus_persistence" in lp["deltas"]


@pytest.mark.parametrize("path", [
    ("main", "E_persist", "panels", "persistence", "binary", "recall"),
    ("main", "E_all", "panels", "geo", "four_class", "macro_f1"),
    ("supplementary", "local_persistence_matched", "panels", "local", "q3_r2_raw"),
    ("main", "E_persist", "deltas", "geo_minus_persistence", "binary.f2"),
    ("main", "ungated_local_diagnostic", "all", "panels", "local", "binary", "precision"),
])
def test_independent_metric_checker_catches_mutation(run_dir, path):
    rep = json.loads((run_dir["root"] / "report" / "report.json").read_text())
    chk = replay.Checker()
    replay.verify_report(chk, run_dir["root"], rep, [H], lambda h: p6_like(run_dir["hz"]))
    assert not chk.failures, chk.failures[:3]
    node = rep["horizons"]["3"]
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = (node[path[-1]] or 0.0) + 0.01
    chk = replay.Checker()
    replay.verify_report(chk, run_dir["root"], rep, [H], lambda h: p6_like(run_dir["hz"]))
    assert chk.failures
