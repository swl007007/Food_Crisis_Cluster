"""Unit checks: contract, annual calendar, decay weights, weighted quartets/store, projection and gate rules."""

from __future__ import annotations

import copy
import json
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from ipcch_yearly_xgb import contract as C
from ipcch_yearly_xgb import engine, metrics, projection, quartet, replay, schedule
from ipcch_yearly_xgb.errors import ContractError, TechnicalError
from ipcch_yearly_xgb.modelstore import ModelStore, identity_digest

M = schedule.parse_month


def test_config_validates_and_rejects_drift(tmp_path):
    assert C.validate_all()["contract"] == "ipcch-yearly-xgb-contract-v1.0"
    for name in ("yearly-contract.json", "inputs.json", "runtime-lock.json"):
        (tmp_path / name).write_text((C.CONFIG_DIR / name).read_text())
    d = json.loads((tmp_path / "yearly-contract.json").read_text())
    d["protocol"]["half_life_months"] = 12
    (tmp_path / "yearly-contract.json").write_text(json.dumps(d))
    with pytest.raises(ContractError):
        C.load_contract(tmp_path)


def test_fit_origin_partial_first_year_and_january_anchors():
    f1, f6 = M("2023-02"), M("2023-07")
    assert schedule.fit_origin(1, f1, M("2023-02")) == M("2023-01")
    assert schedule.fit_origin(1, f1, M("2023-11")) == M("2023-01")
    assert schedule.fit_origin(1, f1, M("2023-01")) == M("2022-12")  # before F_H in its year: January anchor
    assert schedule.fit_origin(1, f1, M("2024-03")) == M("2023-12")
    assert schedule.fit_origin(6, f6, M("2023-03")) == M("2022-07")  # not the future first-block origin
    assert schedule.fit_origin(6, f6, M("2023-08")) == M("2023-01")
    assert schedule.fit_origin(12, M("2024-01"), M("2024-05")) == M("2023-01")


def test_blocks_anchor_on_scheduled_month_even_if_empty():
    rows = [("main", 3, t, t - 3, 0 if t == M("2023-04") else 5) for t in range(M("2023-04"), M("2025-01") + 1)]
    rows += [("supplementary", 3, t, t - 3, 5) for t in range(M("2026-01"), M("2026-05"))]
    cal = pd.DataFrame(rows, columns=["period", "horizon_months", "target_ord", "origin_ord", "eval_keys"])
    cal["fold_id"] = [f"{p}_{t}" for p, t in zip(cal["period"], cal["target_ord"])]
    blocks = schedule.blocks(cal, 3, M("2023-04"))
    assert [(b.period, b.year, schedule.month_label(b.anchor), schedule.month_label(b.origin)) for b in blocks] == [
        ("main", 2023, "2023-04", "2023-01"), ("main", 2024, "2024-01", "2023-10"),
        ("main", 2025, "2025-01", "2024-10"), ("supplementary", 2026, "2026-01", "2025-10")]


def test_gate_dates_use_observed_months_only():
    observed = np.array([M("2022-01"), M("2022-03"), M("2022-04"), M("2022-05"), M("2022-07"), M("2022-08"),
                         M("2022-09"), M("2023-02")])
    got = schedule.gate_dates(observed, M("2023-01"))
    assert [schedule.month_label(u) for u in got] == ["2022-09", "2022-08", "2022-07", "2022-05", "2022-04",
                                                      "2022-03"]


def test_decay_weights_exact_unnormalized():
    t = np.array([M("2023-01"), M("2022-01"), M("2021-01")])
    w = schedule.decay_weights(t, M("2023-01"), 24)
    assert w.tolist() == [1.0, 0.5 ** 0.5, 0.5]
    with pytest.raises(TechnicalError):
        schedule.decay_weights(np.array([M("2023-02")]), M("2023-01"))


def _contract_small():
    c = copy.deepcopy(C.load_contract())
    c["model"]["global_recipes"]["G1"]["rounds"] = 8
    c["model"]["local_recipes"]["L2"]["appended_rounds"] = 5
    return c


def _xy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 561))
    X[rng.random((n, 561)) < 0.3] = np.nan
    Y = np.clip(np.column_stack([0.5 + 0.1 * np.nan_to_num(X[:, 0])] * 4) * [1, .6, .3, .05], 0, 1)
    t = 24000 + rng.integers(0, 60, n)
    return X, Y, schedule.decay_weights(t, 24060)


def test_weighted_global_and_local_records_and_root_preservation():
    c = _contract_small()
    gp, gr = quartet.global_params(c, "G1")
    lp, lr = quartet.local_params(c, "L2")
    X, Y, w = _xy()
    g = quartet.fit_global_quartet(X, Y, w, gp, gr)
    rec = g.records["q3"]["weights"]
    assert rec["scheme"] == "decay" and rec["sum"] == pytest.approx(float(w.sum()))
    sub = np.arange(150)
    loc = quartet.continue_local_quartet(g, X[sub], Y[sub], w[sub], lp, lr)
    assert loc.records["q3"]["parent_booster_sha256"] == g.booster_shas()["q3"]
    assert loc.records["q3"]["rounds_total"] == gr + lr
    unweighted = quartet.fit_global_quartet(X, Y, np.ones(len(w)), gp, gr)
    assert unweighted.booster_shas() != g.booster_shas()  # weights actually reach XGBoost
    with pytest.raises(TechnicalError):
        quartet.fit_global_quartet(X, Y, -w, gp, gr)


def test_store_binds_weights_and_readonly_refuses(tmp_path):
    c = _contract_small()
    gp, gr = quartet.global_params(c, "G1")
    X, Y, w = _xy()
    import hashlib
    from ipcch_yearly_xgb.modelstore import target_digests
    wid = {"protocol_sha256": hashlib.sha256(w.tobytes()).hexdigest(),
           "effective_float32_sha256": hashlib.sha256(w.astype(np.float32).tobytes()).hexdigest()}
    from ipcch_yearly_xgb.modelstore import array_digest
    rows = np.arange(len(w), dtype=np.int64)
    keys = np.column_stack([rows, rows + 24000]).astype(np.int64)
    side = {"fit_rows": rows, "fit_keys": keys}
    ident = {"scope": "yearly-global", "params": gp, "rounds": gr, "n_rows": len(w), "y_sha256": target_digests(Y),
             "weights": wid, "fit_rows": array_digest(rows), "fit_keys": array_digest(keys)}
    store = ModelStore(tmp_path / "m", tmp_path / "l.jsonl")
    with pytest.raises(TechnicalError):  # a new entry without provenance is refused
        store.get_or_fit(ident, lambda: quartet.fit_global_quartet(X, Y, w, gp, gr), {})
    store.get_or_fit(ident, lambda: quartet.fit_global_quartet(X, Y, w, gp, gr), {}, sidecar=side)
    _, e = store.get_or_fit(ident, lambda: pytest.fail("refit"), {})
    assert e["status"] == "hit"
    bad = {**ident, "weights": {**wid, "protocol_sha256": "0" * 64}}
    with pytest.raises(TechnicalError):  # records' weights disagree with the identity
        store.get_or_fit(bad, lambda: quartet.fit_global_quartet(X, Y, w, gp, gr), {}, sidecar=side)
    ro = ModelStore(tmp_path / "m", tmp_path / "ro.jsonl", readonly=True)
    with pytest.raises(TechnicalError):
        ro.get_or_fit({**ident, "n_rows": len(w)} | {"extra": 1}, lambda: pytest.fail("fit"), {})
    d = identity_digest(ident)
    np.save(tmp_path / "m" / d[:2] / d / "fit_rows.npy", rows[::-1].copy())  # tampered provenance
    with pytest.raises(TechnicalError):
        store.get_or_fit(ident, lambda: pytest.fail("refit"), {})
    np.save(tmp_path / "m" / d[:2] / d / "fit_rows.npy", rows)
    store.get_or_fit(ident, lambda: pytest.fail("refit"), {})
    (tmp_path / "m" / d[:2] / d / "q3.ubj").write_bytes(b"corrupt")
    with pytest.raises(TechnicalError):
        store.get_or_fit(ident, lambda: pytest.fail("refit"), {})


@pytest.mark.parametrize("seed", range(5))
def test_independent_projection_matches(seed):
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.2, 0.3, size=(300, 4))
    raw[:20] = [[0.2, 0.2, 0.1, 0.0]] * 20
    raw[20:40] = [[0.1, 0.3, 0.2, 0.2]] * 20  # pooled exactly to 0.2 mean
    star, phase = projection.project_and_decode(raw)
    for i in range(len(raw)):
        s, p = replay.ind_project(raw[i])
        assert p == phase[i] and np.array_equal(np.asarray(s), star[i])


def _pairs(local_better: bool, n_dates=4):
    rows = []
    for d in range(n_dates):
        for a in range(30):
            truth = 3 if a % 2 else 1
            rows.append({"region": "r0", "admin_code": a, "validation_month": 100 + d, "phase_truth": truth,
                         "phase_pool": 3, "phase_local_routed": truth if local_better else 3, "local_fit_ok": True,
                         "local_provider": "L", "pool_provider": "G"})
    return pd.DataFrame(rows)


def test_gate_strict_gain_and_support():
    c = C.load_contract()
    assert engine.gate_decision(_pairs(True), c)["enabled"]
    same = engine.gate_decision(_pairs(False), c)
    assert same["historical_support"] and not same["enabled"]
    assert same["distinct_local_models"] == 1 and same["local_fit_dates"] == 4
    p = _pairs(True)
    p.loc[p["validation_month"] >= 102, "local_fit_ok"] = False
    assert engine.gate_decision(p, c)["reason"] == "gate_support:local_fit_dates"
    assert metrics.gain_passes({"tp": 1, "fp": 0, "fn": 0, "tn": 0}, {"tp": 99, "fp": 2, "fn": 0, "tn": 0},
                               Fraction(1, 100))[0] is False  # exactly 1/100 fails


# ------------------------------------------------------------ source freeze and staging

def test_source_freeze_rejects_evaluator_change_and_controls_reconciliation(tmp_path):
    from ipcch_yearly_xgb import freeze, runtime
    saved = runtime.source_inventory()
    assert freeze.check_sources(tmp_path, saved, dict(saved))["status"] == "identical"
    for changed in ("ipcch_yearly_xgb/projection.py", "ipcch_yearly_xgb/metrics.py", "ipcch_yearly_xgb/report.py"):
        cur = {**saved, changed: "f" * 64}
        with pytest.raises(ContractError):
            freeze.check_sources(tmp_path, saved, cur)
    cur = {**saved, "ipcch_yearly_xgb/report.py": "f" * 64}
    rec = {"changes": {"ipcch_yearly_xgb/report.py": {"old": saved["ipcch_yearly_xgb/report.py"], "new": "f" * 64,
                                                      "reason": "report-only fix", "authorized_by": "supervisor"}}}
    (tmp_path / "source-reconciliation.json").write_text(json.dumps(rec))
    assert freeze.check_sources(tmp_path, saved, cur)["status"] == "reconciled"
    cur2 = {**saved, "ipcch_yearly_xgb/engine.py": "f" * 64}
    rec["changes"]["ipcch_yearly_xgb/engine.py"] = {"old": saved["ipcch_yearly_xgb/engine.py"], "new": "f" * 64,
                                                    "reason": "x", "authorized_by": "y"}
    (tmp_path / "source-reconciliation.json").write_text(json.dumps(rec))
    with pytest.raises(ContractError):  # fit-defining sources can never be reconciled
        freeze.check_sources(tmp_path, saved, cur2)


def test_cli_context_stops_on_source_mismatch_before_any_work(tmp_path):
    from ipcch_yearly_xgb import cli, runtime
    run = tmp_path / "run"
    (run / "preflight").mkdir(parents=True)
    inv = {**runtime.source_inventory(), "ipcch_yearly_xgb/projection.py": "0" * 64}
    (run / "preflight" / "preflight.json").write_text(json.dumps({"source_inventory": inv, "env": {}}))
    with pytest.raises(ContractError, match="projection.py"):
        cli._context(run)
    assert not (run / "predict").exists() and not (run / "models").exists()


def test_staging_copies_verifies_and_detects_tampering(tmp_path, monkeypatch):
    import platform
    from ipcch_yearly_xgb import sources
    from ipcch_yearly_xgb.artifacts import sha256_file
    repo = tmp_path / "repo"
    (repo / "runs/p6/prepared").mkdir(parents=True)
    (repo / "cfg").mkdir()
    (repo / "runs/p6/prepared/a.csv").write_bytes(b"x,y\n1,2\n")
    (repo / "cfg/c.json").write_bytes(b"{}")
    key = "repo_root_windows" if platform.system() == "Windows" else "repo_root_wsl"
    inputs = {"inputs_version": "t", "source_run": "p6", key: str(repo), "run_relative": "runs/p6",
              "run_files": {"prepared/a.csv": {"bytes": (repo / "runs/p6/prepared/a.csv").stat().st_size, "sha256": sha256_file(repo / "runs/p6/prepared/a.csv")}},
              "source_config_files": {"cfg/c.json": {"bytes": 2, "sha256": sha256_file(repo / "cfg/c.json")}}}
    out = sources.stage(tmp_path / "staged", inputs)
    assert out["files"] == 2 and (tmp_path / "staged/run/prepared/a.csv").is_file()
    assert sources.roots(inputs, tmp_path / "staged")[0] == tmp_path / "staged" / "run"
    (tmp_path / "staged/run/prepared/a.csv").write_bytes(b"x,y\n1,3\n")
    with pytest.raises(ContractError):
        sources.verify(inputs, staged=tmp_path / "staged")
    with pytest.raises(ContractError):  # never re-stage over an existing snapshot
        sources.stage(tmp_path / "staged", inputs)
