"""P2: projection/decoding, quartet boosters, exact-identity store, metrics."""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pytest

from ipcch_geoxgb import metrics, projection, quartet
from ipcch_geoxgb.contract import load_experiment_contract
from ipcch_geoxgb.errors import ContractError, TechnicalError
from ipcch_geoxgb.modelstore import ModelStore, array_digest, target_digests

CONTRACT = load_experiment_contract()


# ------------------------------------------------------------ projection


def _pava_decreasing(y):
    """Independent reference: pool-adjacent-violators for a non-increasing fit."""
    blocks = [[v, 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] / blocks[i][1] < blocks[i + 1][0] / blocks[i + 1][1]:
            blocks[i] = [blocks[i][0] + blocks[i + 1][0], blocks[i][1] + blocks[i + 1][1]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.concatenate([[s / n] * n for s, n in blocks])


@pytest.mark.parametrize(
    "raw, expected",
    [
        ([0.6, 0.4, 0.2, 0.1], [0.6, 0.4, 0.2, 0.1]),  # already feasible
        ([0.3, 0.5, 0.1, 0.0], [0.4, 0.4, 0.1, 0.0]),  # pool first two
        ([0.3, 0.2, 0.1, -0.4], [0.3, 0.2, 0.1, 0.0]),  # lower bound
        ([1.4, 1.2, 0.5, 0.1], [1.0, 1.0, 0.5, 0.1]),  # upper bound
        ([0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]),  # fully reversed -> pooled
        ([0.5, 1.3, 0.1, 0.0], [0.9, 0.9, 0.1, 0.0]),  # bound + pooling interact
    ],
)
def test_known_projection_solutions(raw, expected):
    np.testing.assert_allclose(projection.project(np.array([raw]))[0], expected, atol=1e-12)


def test_clipping_before_isotonic_is_wrong():
    raw = np.array([[0.5, 1.3, 0.1, 0.0]])
    right = projection.project(raw)[0]
    wrong = _pava_decreasing(np.clip(raw[0], 0, 1))  # clip first: (.75,.75,.1,0)
    np.testing.assert_allclose(wrong, [0.75, 0.75, 0.1, 0.0])
    assert ((right - raw[0]) ** 2).sum() < ((wrong - raw[0]) ** 2).sum()


def test_projection_matches_reference_and_is_optimal():
    rng = np.random.default_rng(42)
    raw = rng.normal(0.3, 0.5, size=(2000, 4))
    z = projection.project(raw)
    reference = np.clip(np.array([_pava_decreasing(r) for r in raw]), 0, 1)
    np.testing.assert_allclose(z, reference, atol=1e-12)
    assert np.all(np.diff(z, axis=1) <= 0) and z.min() >= 0 and z.max() <= 1
    # no random feasible point beats it
    feasible = -np.sort(-rng.uniform(0, 1, size=(2000, 4)), axis=1)
    assert np.all(((z - raw) ** 2).sum(1) <= ((feasible - raw) ** 2).sum(1) + 1e-12)


def test_decode_inclusive_and_unrounded():
    q = np.array([[0.2, 0.2, 0.1999999, 0.0], [0.19999999, 0.1, 0.0, 0.0], [0.9, 0.5, 0.2, 0.2]])
    assert projection.decode(q).tolist() == [3, 1, 5]


@pytest.mark.parametrize("bad", [np.array([[np.nan, 0, 0, 0]]), np.array([[np.inf, 0, 0, 0]]), np.zeros((2, 3))])
def test_projection_rejects_nonfinite_or_misshaped(bad):
    with pytest.raises(TechnicalError):
        projection.project(bad)


# ------------------------------------------------------------ quartet


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    X[:, 5] = np.nan  # an all-NaN feature column
    X[rng.random(n) < 0.2, 0] = np.nan
    p = rng.dirichlet(np.ones(5), size=n)
    Y = np.column_stack([p[:, k:].sum(1) for k in range(1, 5)])
    Y[:, 3] = 0.0  # constant q5 target is fitted normally
    return X, Y


@pytest.fixture(scope="module")
def fitted():
    X, Y = _data()
    gp, gr = quartet.global_params(CONTRACT, "G1")
    g = quartet.fit_global_quartet(X, Y, gp, gr)
    lp, lr = quartet.local_params(CONTRACT, "L1")
    payloads_before = dict(g.payloads)
    loc = quartet.continue_local_quartet(g, X[:300], Y[:300], lp, lr)
    return X, Y, g, loc, payloads_before


def test_global_params_are_frozen_contract():
    params, rounds = quartet.global_params(CONTRACT, "G4")
    assert rounds == 400 and params["max_depth"] == 4 and params["base_score"] == 0.5
    assert params["objective"] == "reg:squarederror" and params["seed"] == 42 and params["eta"] == 0.05
    lparams, lrounds = quartet.local_params(CONTRACT, "L2")
    assert lrounds == 40 and lparams["max_depth"] == 2 and "base_score" not in lparams


def test_constant_target_and_all_nan_column(fitted):
    X, Y, g, _, _ = fitted
    assert g.records["q5"]["constant"] and g.records["q5"]["constant_value"] == 0.0
    assert not g.records["q3"]["constant"]
    raw = g.predict_raw(X)
    assert raw.shape == (len(X), 4) and np.isfinite(raw).all()


def test_local_appends_exact_rounds_and_keeps_global_prefix(fitted):
    X, _, g, loc, before = fitted
    assert g.payloads == before  # global immutable
    for q in quartet.TARGETS:
        rec = loc.records[q]
        assert rec["rounds_total"] == 220 and rec["rounds_added"] == 20 and rec["parent_rounds"] == 200
        assert rec["parent_booster_sha256"] == g.booster_shas()[q]  # own target's global only
        assert rec["child_prefix_structure_sha256"] == g.records[q]["structure_sha256"]
        assert rec["base_score"] == g.records[q]["base_score"]
    assert not np.array_equal(loc.predict_raw(X), g.predict_raw(X))


def test_reload_reproduces_predictions(fitted):
    X, _, g, loc, _ = fitted
    again = quartet.Quartet(dict(loc.payloads), dict(loc.records))
    np.testing.assert_array_equal(again.predict_raw(X), loc.predict_raw(X))


def test_quartet_is_atomic():
    X, Y = _data(n=80)
    gp, _ = quartet.global_params(CONTRACT, "G1")
    g = quartet.fit_global_quartet(X, Y, gp, 5)
    partial = {q: g.payloads[q] for q in ("q2", "q3", "q4")}
    with pytest.raises(TechnicalError):
        quartet.Quartet(partial, {q: g.records[q] for q in partial})


class _BadBooster:
    def __init__(self, out):
        self.out = out

    def predict(self, _dm):
        return self.out


@pytest.mark.parametrize("out", [np.array([np.nan, 0.1]), np.array([np.inf, 0.1]), np.array([0.1, 0.2, 0.3])])
def test_nonfinite_or_misshaped_predictions_stop(out):
    with pytest.raises(TechnicalError):
        quartet.predict_scalar(_BadBooster(out), np.zeros((2, 3)))


def test_infinite_input_and_misaligned_keys_stop():
    with pytest.raises(TechnicalError):
        quartet.check_X(np.array([[np.inf, 1.0]]))
    with pytest.raises(TechnicalError):
        quartet.check_aligned(np.array([[1, 2], [3, 4]]), np.array([[3, 4], [1, 2]]))


def test_empty_global_pool_is_a_stop():
    gp, gr = quartet.global_params(CONTRACT, "G1")
    with pytest.raises(ContractError):
        quartet.fit_global(np.zeros((0, 3)), np.zeros(0), gp, gr)


def test_local_cannot_override_base_score(fitted):
    X, Y, g, _, _ = fitted
    lp, lr = quartet.local_params(CONTRACT, "L1")
    with pytest.raises(ContractError):
        quartet.continue_local(g.payloads["q3"], X, Y[:, 1], {**lp, "base_score": 0.3}, lr)


# ------------------------------------------------------------ store


def _identity(X, Y, keys, scope="stage3-global", rounds=5):
    params, _ = quartet.global_params(CONTRACT, "G1")
    return {"scope": scope, "h": 3, "keys": array_digest(keys), "X": array_digest(X), "Y": array_digest(Y),
            "recipe": "G1", "params": params, "rounds": rounds, "n_rows": int(len(Y)),
            "y_sha256": target_digests(Y)}


def test_store_fits_once_then_hits_and_distinct_identities_do_not_collide(tmp_path):
    X, Y = _data(n=120)
    keys = np.arange(120)
    gp, _ = quartet.global_params(CONTRACT, "G1")
    store = ModelStore(tmp_path / "models", tmp_path / "ledger.jsonl")
    calls = []

    def fit():
        calls.append(1)
        return quartet.fit_global_quartet(X, Y, gp, 5)

    a, e1 = store.get_or_fit(_identity(X, Y, keys), fit, {"use": "U1"})
    b, e2 = store.get_or_fit(_identity(X, Y, keys), fit, {"use": "U2"})
    assert (e1["status"], e2["status"], len(calls)) == ("fit", "hit", 1)
    np.testing.assert_array_equal(a.predict_raw(X), b.predict_raw(X))
    keys2 = keys.copy()
    keys2[[0, 1]] = keys2[[1, 0]]  # same set, different order -> different identity
    _, e3 = store.get_or_fit(_identity(X, Y, keys2), fit, {"use": "U3"})
    assert e3["status"] == "fit" and e3["identity_sha256"] != e1["identity_sha256"]
    assert store.counts == {"requests": 3, "hits": 1, "fits": 2, "failed": 0}
    assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize("damage", ["bytes", "identity", "partial"])
def test_corrupt_or_conflicting_entry_stops(tmp_path, damage):
    X, Y = _data(n=80)
    keys = np.arange(80)
    gp, _ = quartet.global_params(CONTRACT, "G1")
    store = ModelStore(tmp_path / "m", tmp_path / "l.jsonl")
    _, entry = store.get_or_fit(_identity(X, Y, keys), lambda: quartet.fit_global_quartet(X, Y, gp, 5), {})
    d = entry["identity_sha256"]
    directory = tmp_path / "m" / d[:2] / d
    if damage == "bytes":
        (directory / "q3.ubj").write_bytes((directory / "q3.ubj").read_bytes() + b"x")
    elif damage == "identity":
        record = json.loads((directory / "record.json").read_text())
        record["identity"]["h"] = 6
        (directory / "record.json").write_text(json.dumps(record))
    else:
        (directory / "q5.ubj").unlink()
    with pytest.raises(TechnicalError):
        store.get_or_fit(_identity(X, Y, keys), lambda: pytest.fail("must not refit"), {})


# ------------------------------------------------------------ metrics


def test_binary_na_rules_keep_legal_zeros():
    all_tn = metrics.binary_metrics({"tp": 0, "fp": 0, "fn": 0, "tn": 5})
    assert all_tn["accuracy"] == 1.0 and all_tn["precision"] is None and all_tn["recall"] is None
    assert all_tn["f1"] is None and all_tn["f2"] is None and set(all_tn["na_reasons"]) == {"precision", "recall", "f1", "f2"}
    fp_only = metrics.binary_metrics({"tp": 0, "fp": 3, "fn": 0, "tn": 1})
    assert fp_only["precision"] == 0.0 and fp_only["f1"] == 0.0 and fp_only["recall"] is None
    m = metrics.binary_metrics({"tp": 2, "fp": 1, "fn": 1, "tn": 0})
    assert m["f2"] == pytest.approx(10 / 15) and m["f1"] == pytest.approx(4 / 6)


def test_four_class_fixed_axis_and_absent_class():
    truth = np.array([1, 2, 3, 4])
    assert metrics.four_class_metrics(metrics.four_class_confusion(truth, truth))["macro_f1"] == 1.0
    out = metrics.four_class_metrics(metrics.four_class_confusion(np.array([1, 2, 3]), np.array([1, 2, 2])))
    assert out["macro_f1"] is None and "4/5" in out["na_reasons"]["macro_f1"]
    assert out["per_class"]["3"]["f1"] == 0.0  # present in truth, legal zero
    phase5 = metrics.four_class_confusion(np.array([5, 4]), np.array([4, 5]))
    assert phase5[3, 3] == 2  # 4 and 5 merge


def test_r_squared_rules():
    assert metrics.r_squared([0.1, 0.2, 0.3], [0.3, 0.1, 0.5])[0] < 0  # negative kept
    assert metrics.r_squared([0.2, 0.2], [0.1, 0.3]) == (None, "constant truth (SST = 0)")
    assert metrics.r_squared([0.2], [0.2]) == (None, "n < 2")


def test_gate_uses_exact_strict_gain_and_na_fails():
    base = {"tp": 1, "fp": 1, "fn": 0, "tn": 0}  # F1 = 2/3
    assert metrics.gain_passes(base, base, Fraction(0)) == (False, "gain_not_above_threshold")
    better = {"tp": 3, "fp": 1, "fn": 0, "tn": 0}  # 6/7
    assert metrics.gain_passes(better, base, Fraction(0))[0]
    exact = {"tp": 1, "fp": 0, "fn": 0, "tn": 0}  # 1
    almost = {"tp": 99, "fp": 0, "fn": 2, "tn": 0}  # 198/200 = .99 -> gain exactly .01
    assert metrics.gain_passes(exact, almost, Fraction(1, 100)) == (False, "gain_not_above_threshold")
    empty = {"tp": 0, "fp": 0, "fn": 0, "tn": 4}
    assert metrics.gain_passes(better, empty, Fraction(0)) == (False, "f1_undefined")


def test_crisis_scan_masses():
    groups, Y, A, counts = metrics.crisis_scan_masses([3, 3, 1, 1], [3, 1, 3, 1], [20, 20, 7, 7])
    assert groups.tolist() == [7, 20]
    # group 7: FP=1 -> D=1; group 20: TP=1, FN=1 -> D=3; total 4
    np.testing.assert_allclose(Y, [0.25, 0.75])
    np.testing.assert_allclose(A, [0.0, 0.5])
    _, Y0, A0, _ = metrics.crisis_scan_masses([1, 1], [1, 1], [1, 2])
    assert Y0.tolist() == [0.0, 0.0] and A0.tolist() == [0.0, 0.0]


# ------------------------------------------------------------ supervisor P2 review regressions


def test_constant_q3_truth_is_na_before_any_mean():
    for pred in (np.full(3, 0.1), np.full(3, 0.2), np.array([0.1, 0.2, 0.3])):
        assert metrics.r_squared(np.full(3, 0.2), pred) == (None, "constant truth (SST = 0)")
    value, reason = metrics.r_squared([0.1, 0.2, 0.3], [0.3, 0.1, 0.5])
    assert value < 0 and reason == ""  # valid negative R² kept


@pytest.mark.parametrize(
    "call",
    [
        lambda: metrics.crisis_scan_masses([1, 3], [3], [1, 2]),
        lambda: metrics.crisis_scan_masses([1, 3], [3, 3], [1]),
        lambda: metrics.crisis_scan_masses([[1, 3]], [[3, 3]], [[1, 2]]),
        lambda: metrics.metric_panel([1, 2, 3, 4], [1, 2, 3, 4], [.1, .3], [.1, .3], [.1, .3]),
        lambda: metrics.metric_panel([1, 2], [1, 2], [.1, np.nan], [.1, .3]),
        lambda: metrics.r_squared([0.1, np.inf], [0.1, 0.2]),
        lambda: metrics.r_squared([0.1, 0.2, 0.3], [0.1, 0.2]),
        lambda: metrics.crisis_counts([0, 3], [1, 3]),  # phase-0 sentinel is never scored
        lambda: metrics.crisis_counts([1.5, 3.0], [1.0, 3.0]),
        lambda: metrics.four_class_confusion([1, 2], [1]),
    ],
)
def test_metric_entrypoints_reject_misaligned_or_nonfinite_input(call):
    with pytest.raises(TechnicalError):
        call()


def test_projection_is_stable_for_extreme_finite_inputs():
    raw = np.array([[-1e8, 1e8, 0.9, 0.1], [1e15, -1e15, 0.5, 0.6], [0.5, 1.3, 0.1, 0.0]])
    z = projection.project(raw)
    np.testing.assert_allclose(z[0], [0.3, 0.3, 0.3, 0.1], atol=1e-12)
    assert projection.decode(z[:1]).tolist() == [4]
    np.testing.assert_allclose(z[1], [1.0, 0.0, 0.0, 0.0])  # 1e15, -1e15, then .55,.55 -> clip
    np.testing.assert_allclose(z[2], [0.9, 0.9, 0.1, 0.0])  # clip-first counterexample unchanged


def test_local_continuation_refuses_an_already_local_quartet(fitted):
    X, Y, g, loc, _ = fitted
    lp, lr = quartet.local_params(CONTRACT, "L1")
    with pytest.raises(TechnicalError, match="requires a global quartet"):
        quartet.continue_local_quartet(loc, X[:100], Y[:100], lp, lr)  # would be 220 -> 240


def test_fit_records_carry_fit_time_resolved_config(fitted):
    _, _, g, loc, _ = fitted
    for q in quartet.TARGETS:
        for record in (g.records[q], loc.records[q]):
            learner = record["resolved_config"]["learner"]
            assert learner["objective"]["name"] == "reg:squarederror"
            assert learner["learner_model_param"]["base_score"] == record["base_score"]


@pytest.mark.parametrize("damage", ["empty_q3_record", "objective", "rounds", "structure", "kind"])
def test_incomplete_or_inconsistent_fit_record_stops_on_load(tmp_path, damage):
    X, Y = _data(n=80)
    gp, _ = quartet.global_params(CONTRACT, "G1")
    store = ModelStore(tmp_path / "m", tmp_path / "l.jsonl")
    identity = _identity(X, Y, np.arange(80))
    _, entry = store.get_or_fit(identity, lambda: quartet.fit_global_quartet(X, Y, gp, 5), {})
    d = entry["identity_sha256"]
    path = tmp_path / "m" / d[:2] / d / "record.json"
    record = json.loads(path.read_text())
    rec = record["fit_records"]["q3"]
    if damage == "empty_q3_record":
        record["fit_records"]["q3"] = {}
    elif damage == "objective":
        rec["resolved_config"]["learner"]["objective"]["name"] = "reg:absoluteerror"
    elif damage == "rounds":
        rec["rounds_total"] = 6
    elif damage == "structure":
        rec["structure_sha256"] = "0" * 64
    else:
        rec["kind"] = "local"
    path.write_text(json.dumps(record))
    with pytest.raises(TechnicalError):
        store.get_or_fit(identity, lambda: pytest.fail("must not refit"), {})


def test_object_arrays_cannot_be_identity_digests():
    with pytest.raises(TechnicalError):
        array_digest(np.array(["a", 1], dtype=object))
    assert array_digest(np.array([1, 2])) != array_digest(np.array([2, 1]))
