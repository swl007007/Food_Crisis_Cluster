"""Unit tests: preprocessing, networks, seeds, store, selection, gate and projection rules."""

from __future__ import annotations

import copy
import json
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from ipcch_mlp import contract as C
from ipcch_mlp import develop, metrics, nets, preprocess, projection, runtime, seeds, sources, stage3
from ipcch_mlp.errors import ContractError, TechnicalError
from ipcch_mlp.runtime import torch
from ipcch_mlp.store import ModelStore

runtime.configure("cpu", 4)
CFG = C.load_contract()["training"]


def _X(n=50, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 561))
    X[rng.random((n, 561)) < 0.3] = np.nan
    return X


# ------------------------------------------------------------ config

def test_configs_validate_and_counts():
    v = C.validate_all()
    assert v["contract"] == "ipcch-mlp-contract-v1.0"
    c = C.load_contract()
    for gid, w in c["architecture"]["global_candidates"].items():
        assert nets.parameter_count(w) == c["architecture"]["parameter_counts"][gid]
    for rid, w in c["architecture"]["residual_candidates"].items():
        assert nets.parameter_count(w) == c["architecture"]["parameter_counts"][rid]
    assert nets.parameter_count([64, 32]) == 73985 and nets.parameter_count([32]) == 35969


def test_contract_rejects_drift(tmp_path):
    for name in ("experiment-mlp.json", "inputs.json", "runtime-lock.json"):
        (tmp_path / name).write_text((C.CONFIG_DIR / name).read_text())
    d = json.loads((tmp_path / "experiment-mlp.json").read_text())
    d["training"]["residual_epochs"] = 41
    (tmp_path / "experiment-mlp.json").write_text(json.dumps(d))
    with pytest.raises(ContractError):
        C.load_contract(tmp_path)


# ------------------------------------------------------------ preprocessing

def test_transform_rules():
    X = _X()
    X[:, 5] = np.nan  # training-all-missing
    X[:, 7] = 3.0  # zero variance
    X[0, 9] = np.nan
    t = preprocess.fit_transform(X)
    assert t.all_missing[5] and t.medians[5] == 0.0 and t.scales[5] == 1.0
    assert t.scales[7] == 1.0 and t.means[7] == 3.0
    obs = X[~np.isnan(X[:, 9]), 9]
    assert t.medians[9] == np.median(obs)
    Z = preprocess.apply(t, X)
    assert Z.shape == (50, 1122) and Z.dtype == np.float32 and Z.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(Z[:, 561:], np.isnan(X).astype(np.float32))
    imputed = np.where(np.isnan(X), t.medians, X)
    np.testing.assert_allclose(Z[:, :561], ((imputed - t.means) / t.scales).astype(np.float32))
    assert abs(Z[:, 9].astype(np.float64).mean()) < 1e-6  # standardized after imputation, ddof 0


def test_transform_unseen_missingness_and_infinity():
    X = _X()
    X[:, 5] = np.nan
    t = preprocess.fit_transform(X)
    Y = _X(seed=1)
    Y[:, 5] = np.nan
    Y[:3, 5] = 0.1
    d = preprocess.unseen_missingness(t, Y)
    assert d["rows_affected"] == 3 and d["per_column"] == {5: 3}
    Z = preprocess.apply(t, Y)  # same frozen zero-fill / unit scale, no refit
    assert np.allclose(Z[:3, 5], 0.1) and Z[0, 561 + 5] == 0.0
    X[0, 0] = np.inf
    with pytest.raises(TechnicalError):
        preprocess.fit_transform(X)
    with pytest.raises(TechnicalError):
        preprocess.apply(t, X)


def test_transform_digest_changes_with_rows():
    X = _X()
    assert preprocess.fit_transform(X).digest() == preprocess.fit_transform(X.copy()).digest()
    assert preprocess.fit_transform(X).digest() != preprocess.fit_transform(X[1:]).digest()


# ------------------------------------------------------------ networks

def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 1122)).astype(np.float32)
    return X, (0.3 + 0.1 * X[:, 0]).astype(np.float32)


@pytest.mark.parametrize("widths", [[16], [32]])
def test_residual_starts_at_zero_in_train_and_eval(widths):
    X, _ = _data()
    net = nets.build(widths, "residual", 3)
    with torch.no_grad():
        net.train()
        assert torch.count_nonzero(net(torch.from_numpy(X))).item() == 0
    assert np.all(nets.predict(net, X, "cpu") == 0.0)
    hidden = net.body[0].weight
    assert torch.count_nonzero(hidden).item() > 0  # hidden layers stay random


def test_global_keeps_default_init():
    net = nets.build([64, 32], "global", 3)
    assert torch.count_nonzero(net.output_layer.weight).item() > 0


def test_zero_target_keeps_zero_output():
    X, _ = _data()
    net = nets.build([16], "residual", 4)
    nets.train(net, X, np.zeros(len(X), np.float32), 5, {"train": 1, "perm": 2}, CFG, "cpu")
    assert np.all(nets.predict(net, X, "cpu") == 0.0)


def test_nonzero_target_learns_output_then_hidden():
    X, y = _data()
    net = nets.build([16], "residual", 5)
    h0 = net.body[0].weight.detach().clone()
    nets.train(net, X, y, 1, {"train": 1, "perm": 2}, CFG, "cpu")
    assert torch.count_nonzero(net.output_layer.weight).item() > 0
    assert not torch.equal(h0, net.body[0].weight.detach())


def test_training_is_repeatable_and_rng_isolated():
    X, y = _data()
    a = nets.build([16], "residual", 7)
    nets.train(a, X, y, 3, {"train": 8, "perm": 9}, CFG, "cpu")
    other = nets.build([64, 32], "global", 1)  # intervening fit consumes the global RNG
    nets.train(other, X, y, 2, {"train": 2, "perm": 3}, CFG, "cpu")
    torch.rand(10)
    b = nets.build([16], "residual", 7)
    nets.train(b, X, y, 3, {"train": 8, "perm": 9}, CFG, "cpu")
    assert nets.state_digest(nets.cpu_state(a)) == nets.state_digest(nets.cpu_state(b))
    c = nets.build([16], "residual", 7)
    nets.train(c, X, y, 3, {"train": 8, "perm": 10}, CFG, "cpu")
    assert nets.state_digest(nets.cpu_state(a)) != nets.state_digest(nets.cpu_state(c))


def test_updates_batches_and_decay_groups():
    X, y = _data(n=600)
    net = nets.build([16], "residual", 1)
    h = nets.train(net, X, y, 2, {"train": 1, "perm": 1}, CFG, "cpu")
    assert h["updates"] == 2 * 3 and h["batch_size"] == 256  # 600 = 256 + 256 + 88 (partial kept)
    small = nets.train(nets.build([16], "residual", 1), X[:100], y[:100], 2, {"train": 1, "perm": 1}, CFG, "cpu")
    assert small["batch_size"] == 100 and small["updates"] == 2
    opt = nets._optimizer(net, CFG)
    assert opt.param_groups[0]["weight_decay"] == 0.01 and all(p.ndim == 2 for p in opt.param_groups[0]["params"])
    assert opt.param_groups[1]["weight_decay"] == 0.0 and all(p.ndim == 1 for p in opt.param_groups[1]["params"])
    assert opt.defaults["foreach"] is False and opt.defaults["fused"] is False and opt.defaults["amsgrad"] is False


def test_nonfinite_training_data_stops():
    X, y = _data()
    y[0] = np.nan
    with pytest.raises(TechnicalError):
        nets.train(nets.build([16], "residual", 1), X, y, 1, {"train": 1, "perm": 1}, CFG, "cpu")


def test_dropout_off_in_prediction():
    X, y = _data()
    net = nets.build([64, 32], "global", 2)
    nets.train(net, X, y, 1, {"train": 1, "perm": 1}, CFG, "cpu")
    torch.manual_seed(0)
    a = nets.predict(net, X, "cpu")
    torch.manual_seed(99)
    assert np.array_equal(a, nets.predict(net, X, "cpu"))


# ------------------------------------------------------------ seeds and store

def test_seed_derivation_is_pinned():
    ident = {"stage": "stage3", "H": 1, "replicate": 42, "target": "q3", "role": "global"}
    assert seeds.derive(ident, "init") == seeds.derive(dict(reversed(list(ident.items()))), "init")
    assert len({seeds.derive(ident, s) for s in seeds.STREAMS}) == 3
    assert seeds.derive(ident, "init") == int(
        __import__("hashlib").sha256((seeds.canonical(ident) + "|init").encode()).hexdigest()[:16], 16) & ((1 << 63) - 1)


def test_store_hit_conflict_corruption_and_readonly(tmp_path):
    store = ModelStore(tmp_path / "m", tmp_path / "ledger.jsonl")
    calls = []

    def fit():
        calls.append(1)
        net = nets.build([16], "residual", 1)
        return nets.cpu_state(net), {"widths": [16]}

    ident = {"a": 1}
    s1, r1, d = store.get_or_fit(ident, fit, {"use": "x"})
    s2, r2, d2 = store.get_or_fit(ident, fit, {"use": "y"})
    assert len(calls) == 1 and d == d2 and nets.state_digest(s1) == nets.state_digest(s2)
    with pytest.raises(TechnicalError):
        store.load(d, {"a": 2})
    path = store._dir(d) / "state.pt"
    state = torch.load(path, weights_only=True)
    state["body.0.bias"][0] += 1.0
    torch.save(state, path)
    with pytest.raises(TechnicalError):
        store.get_or_fit(ident, fit, {})
    ro = ModelStore(tmp_path / "m", tmp_path / "ro.jsonl", readonly=True)
    with pytest.raises(TechnicalError):
        ro.get_or_fit({"a": 3}, fit, {})
    t = preprocess.fit_transform(_X())
    dig = store.put_transform(t)
    assert store.get_transform(dig).digest() == dig


# ------------------------------------------------------------ selection, gate, projection

def _contract():
    return C.load_contract()


def test_selection_mean_ties_and_undefined():
    c = _contract()
    entries = {cid: {42: "1/2", 43: "1/2", 44: "1/2"} for cid in develop.CANDIDATES}
    assert develop.select(entries, c)["winner"] == "G1R1"  # all tied -> fewest parameters
    entries["G2R2"] = {42: "3/4", 43: "1/2", 44: "1/2"}
    assert develop.select(entries, c)["winner"] == "G2R2"  # mean of exact fractions, not the best seed
    entries["G2R2"][44] = None
    out = develop.select(entries, c)
    assert out["winner"] == "G1R1" and out["ineligible"] == {"G2R2": "undefined_seed_f1"}
    allna = {cid: {42: None} for cid in develop.CANDIDATES}
    assert develop.select(allna, c)["status"] == "selection_unavailable"


def _pairs(n_dates=4, local_better=True):
    rows = []
    for d in range(n_dates):
        for a in range(30):
            truth = 3 if a % 2 else 1
            rows.append({"region": "r0", "admin_code": a, "validation_month": 100 + d, "phase_truth": truth,
                         "phase_pool": 3, "phase_local_routed": truth if local_better else 3, "local_fit_ok": True})
    return pd.DataFrame(rows)


def test_gate_support_gain_and_equality():
    c = _contract()
    ok = stage3.gate_decision(_pairs(), c)
    assert ok["historical_support"] and ok["enabled"]
    same = stage3.gate_decision(_pairs(local_better=False), c)
    assert same["historical_support"] and not same["enabled"] and same["reason"] == "gain_not_above_threshold"
    few = _pairs(n_dates=4)
    few.loc[few["validation_month"] >= 102, "local_fit_ok"] = False
    r = stage3.gate_decision(few, c)
    assert not r["historical_support"] and r["reason"] == "gate_support:local_fit_dates"
    # exact threshold: candidate F1 1 vs base 99/100 is a gain of exactly 1/100, which fails
    passed, _ = metrics.gain_passes({"tp": 1, "fp": 0, "fn": 0, "tn": 0}, {"tp": 99, "fp": 2, "fn": 0, "tn": 0},
                                    Fraction(1, 100))
    assert passed is False  # 1 - 99/100 = 1/100 exactly -> not strictly greater


def test_projection_threshold_inclusive():
    star, phase = projection.project_and_decode(np.array([[0.5, 0.2, 0.1, 0.0], [0.5, 0.19999, 0.1, 0.0]]))
    assert phase.tolist() == [3, 2]


def test_calendar_cutoffs():
    assert sources.training_window(100) == (65, 100)
    assert sources.historical_gate_dates(np.array([90, 95, 99, 100, 101, 98, 97, 96]), 100).tolist() == [99, 98, 97, 96, 95, 90]


def _put_many(root: str, n: int) -> None:
    store = ModelStore(__import__("pathlib").Path(root), __import__("pathlib").Path(root) / "l.jsonl")
    t = preprocess.fit_transform(_X())
    for _ in range(n):
        store.put_transform(t)


def test_concurrent_transform_writes(tmp_path):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_put_many, args=(str(tmp_path), 25)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert [p.exitcode for p in procs] == [0, 0, 0, 0]
    files = list((tmp_path / "transforms").iterdir())
    assert len(files) == 1 and not files[0].name.startswith(".tmp")
