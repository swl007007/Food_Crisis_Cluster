"""P3: R46 prefix, R32 scan, R38 smoothing, R28/R29 support, route gate, R45 recursion, R44 selection."""

from __future__ import annotations

import copy
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from ipcch_geoxgb import stage1
from ipcch_geoxgb.contract import load_experiment_contract

CONTRACT = load_experiment_contract()


# ------------------------------------------------------------ R46


@pytest.mark.parametrize("n, bounds", [(100, (45, 55)), (101, (46, 55)), (2, (1, 1)), (20, (9, 11)), (3, None), (1, None), (0, None)])
def test_size_bounds_integer_arithmetic(n, bounds):
    assert stage1.size_bounds(n) == bounds


def test_prefix_ties_use_area_id_then_half_then_smaller_m():
    # equal scores: order by ascending area id
    s0, s1, m = stage1.choose_prefix(np.array([1.0, 1.0, 1.0, 1.0]), np.array([40, 10, 30, 20]))
    assert m == 2 and sorted(np.array([40, 10, 30, 20])[s0]) == [10, 20]
    # N=20 allows m=9..11; scores make prefix sums equal for m=9,10,11 -> prefer |2m-N| = 0 (m=10)
    scores = np.array([1.0] * 9 + [0.0, 0.0] + [-1.0] * 9)
    assert stage1.choose_prefix(scores, np.arange(20))[2] == 10
    # N=21 allows m=10..11; equal sums at 10 and 11 tie on |2m-N|=1 -> smaller m
    scores = np.array([1.0] * 10 + [0.0] + [-1.0] * 10)
    assert stage1.choose_prefix(scores, np.arange(21))[2] == 10
    # largest prefix sum wins inside the window
    scores = np.array([5.0] * 11 + [-0.1] * 9)
    assert stage1.choose_prefix(scores, np.arange(20))[2] == 11
    assert stage1.choose_prefix(np.array([1.0, 2.0, 3.0]), np.arange(3)) is None


# ------------------------------------------------------------ R32 scan


def test_scan_concentrates_error_and_is_deterministic():
    rng = np.random.default_rng(1)
    n = 40
    D = rng.integers(1, 10, size=n).astype(float)
    tp = np.where(np.arange(n) < 20, D * 0.45, D * 0.1)  # first half has fewer errors
    Y, A = D / D.sum(), 2 * tp / D.sum()
    ids = np.arange(100, 100 + n)
    first = stage1.scan(Y, A, ids, iterations=1000)
    again = stage1.scan(Y, A, ids, iterations=1000)
    np.testing.assert_array_equal(first.s0, again.s0)
    assert 18 <= first.m_final <= 22 and set(first.s0) == set(range(20, 40))  # high-error half
    assert first.rho > 1


def test_scan_zero_mass_groups_still_count_toward_n():
    Y = np.array([0.5, 0.5, 0.0, 0.0])
    A = np.array([0.0, 0.5, 0.0, 0.0])
    found = stage1.scan(Y, A, np.array([1, 2, 3, 4]), iterations=10)
    assert found is not None and found.m_final == 2
    assert 0 in found.s0  # the group carrying all error mass is selected


# ------------------------------------------------------------ R38 smoothing


def test_smoothing_is_synchronous_and_uses_strict_four_ninths():
    # path 1-2-3: 2 is surrounded by the other label -> switches; 1 and 3 keep (2 of 2 votes)
    neighbours = {1: [2], 2: [1, 3], 3: [2]}
    assert stage1.smooth({1: 0, 2: 1, 3: 0}, neighbours, rounds=1) == {1: 0, 2: 0, 3: 0}
    # synchronous: both ends of an alternating pair read the OLD labels
    assert stage1.smooth({1: 0, 2: 1}, {1: [2], 2: [1]}, rounds=1) == {1: 0, 2: 1}  # 1/2 >= 4/9 keeps
    # exactly 4/9 keeps (not strictly below): self + 8 neighbours, 3 neighbours share the label
    star = {0: list(range(1, 9))}
    labels = {0: 0, **{i: (0 if i <= 3 else 1) for i in range(1, 9)}}
    assert stage1.smooth(labels, star, rounds=1)[0] == 0
    labels[3] = 1  # now 3 of 9 share -> strictly below 4/9 -> switch
    assert stage1.smooth(labels, star, rounds=1)[0] == 1


def test_smoothing_ignores_outside_and_isolated_areas():
    neighbours = {1: [2, 99], 2: [1], 5: []}
    out = stage1.smooth({1: 0, 2: 0, 5: 1}, neighbours, rounds=3)
    assert out == {1: 0, 2: 0, 5: 1}  # 99 is outside the node and does not vote; 5 isolated keeps


def test_component_diagnostics():
    neighbours = {1: [2], 2: [1, 3], 3: [2], 4: [], 5: [9]}
    d = stage1.component_diagnostics(np.array([1, 2, 3, 4, 5]), neighbours)
    assert d == {"areas": 5, "components": 3, "largest_component": 3, "isolated_areas": 2,
                 "component_sizes": [3, 1, 1]}


# ------------------------------------------------------------ support and route gate


def test_support_floor_equality_edges():
    floor = CONTRACT["support"]["validation"]
    exact = {"keys": 100, "areas": 20, "target_months": 3, "crisis_keys": 20, "noncrisis_keys": 20}
    assert stage1.meets(exact, floor)
    for key in exact:
        assert not stage1.meets({**exact, key: exact[key] - 1}, floor)


def _phase(*flags):
    return np.array([3 if f else 1 for f in flags])


def test_route_gate_order_ties_and_ineligible_children():
    truth = (_phase(1, 1, 0), _phase(1, 0, 0))
    parent = (_phase(1, 0, 0), _phase(0, 0, 0))  # TP1 FN2 -> F1 = 2/4
    good0 = _phase(1, 1, 0)  # fixes side 0
    good1 = _phase(1, 0, 0)  # fixes side 1
    out = stage1.select_route(truth, parent, (good0, good1), (True, True))
    assert out["accepted"] and out["choice"] == (True, True)
    # side 1 ineligible: only child/parent available
    out = stage1.select_route(truth, parent, (good0, None), (True, False))
    assert out["accepted"] and out["choice"] == (True, False) and "parent_child" not in out["scores"]
    # equal improvement from child/parent and child/child: the first in fixed order wins
    out = stage1.select_route(truth, parent, (good0, parent[1]), (True, True))
    assert out["choice"] == (True, False)
    # no strict gain: parent kept
    out = stage1.select_route(truth, parent, parent, (True, True))
    assert not out["accepted"] and out["reason"] == "no_strict_gain"


def test_route_gate_undefined_parent_never_splits():
    truth = (_phase(0, 0), _phase(0))
    parent = (_phase(0, 0), _phase(0))
    out = stage1.select_route(truth, parent, (_phase(1, 0), _phase(1)), (True, True))
    assert not out["accepted"] and out["reason"] == "parent_f1_undefined"


# ------------------------------------------------------------ recursion with deterministic fake quartets


class FakeQuartet:
    """predict_raw = f(X): q3 from a per-row column so routing is exactly controllable."""

    def __init__(self, column):
        self.column = column

    def predict_raw(self, X):
        q3 = X[:, self.column]
        return np.column_stack([np.minimum(q3 + 0.3, 1.0), q3, q3 * 0.5, q3 * 0.1])


def _world(n_areas=120, months=10, seed=0):
    """Areas 1000.. ; west half (area < 1060) truth crisis iff x0 > 0, east half iff x0 < 0.

    Column 1 = root guess (crisis iff x0 > 0: right in the west, wrong in the east);
    column 2 = oracle (q3 above/below .20 exactly as the truth).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(1000, 1000 + n_areas):
        for t in range(months):
            rows.append((a, 24000 + t))
    keys = pd.DataFrame(rows, columns=["admin_code", "target_ord"])
    x0 = rng.normal(size=len(keys))
    west = keys["admin_code"].to_numpy() < 1000 + n_areas // 2
    crisis = np.where(west, x0 > 0, x0 < 0)
    keys["phase_truth"] = np.where(crisis, 3, 1)
    keys["crisis_truth"] = crisis.astype(int)
    X = np.column_stack([x0, np.where(x0 > 0, 0.3, 0.1), np.where(crisis, 0.3, 0.1)])
    fit = keys["target_ord"].to_numpy() < 24000 + months // 2
    neighbours = {a: [b for b in (a - 1, a + 1) if 1000 <= b < 1000 + n_areas] for a in range(1000, 1000 + n_areas)}
    return keys[fit].reset_index(drop=True), keys[~fit].reset_index(drop=True), X[fit], X[~fit], neighbours


def _small_contract():
    c = copy.deepcopy(CONTRACT)
    c["support"]["local_fit"] = {"keys": 20, "areas": 5, "target_months": 2}
    c["support"]["validation"] = {"keys": 20, "areas": 5, "target_months": 2, "crisis_keys": 5, "noncrisis_keys": 5}
    c["partition"]["scan_iterations"] = 50
    return c


def _run(contract, oracle=True, max_depth=None):
    fit_keys, val_keys, X_fit, X_val, neighbours = _world()
    if max_depth is not None:
        contract["partition"]["max_member_depth"] = max_depth
    calls = []

    def fit_local(areas, mask, child_id, parent_id):
        assert child_id.startswith(parent_id) and len(child_id) == len(parent_id) + 1
        calls.append((tuple(areas), int(mask.sum())))
        return f"local{len(calls)}", FakeQuartet(2 if oracle else 1)

    result, providers = stage1.search_candidate(
        fit_keys=fit_keys, val_keys=val_keys, X_fit=X_fit, X_val=X_val, root=("root", FakeQuartet(1)),
        fit_local=fit_local, neighbours=neighbours, contract=contract,
    )
    return result, providers, calls, val_keys, X_val


def test_recursion_accepts_beneficial_split_and_respects_depth_budget():
    result, providers, calls, val_keys, X_val = _run(_small_contract())
    first = result.decisions[0]
    assert first["node_id"] == "r" and first["outcome"] == "accepted"
    assert first["sizes_before_smoothing"][0] + first["sizes_before_smoothing"][1] == 120
    assert all(stage1.node_depth(n.node_id) <= 4 for n in result.terminal) and len(result.terminal) <= 16
    assert all(d["depth"] <= 3 for d in result.decisions)  # parents searched only at depth 0..3
    assert len(calls) <= 30 and sum(1 for d in result.decisions if "m_final" in d) <= 15
    # local children are oracles: once accepted, the deeper nodes have zero error -> stop
    assert any(d["outcome"] == "no_candidate_zero_error_mass" for d in result.decisions)


def test_root_only_equivalence_when_no_split_is_accepted():
    # children that are no better than the root are never accepted
    result, providers, calls, val_keys, X_val = _run(_small_contract(), oracle=False)
    assert [n.node_id for n in result.terminal] == ["r"] and result.terminal[0].provider == "root"
    nodes, prov = stage1.terminal_routing(result.terminal, val_keys["admin_code"].to_numpy())
    assert set(prov) == {"root"}


def test_unsupported_side_inherits_parent_provider():
    contract = _small_contract()
    contract["support"]["validation"]["crisis_keys"] = 10_000  # nobody eligible
    result, _, calls, _, _ = _run(contract)
    assert calls == [] and result.decisions[0]["outcome"] == "rejected_no_eligible_child"


def test_depth_zero_budget_means_root_only():
    result, _, calls, _, _ = _run(_small_contract(), max_depth=0)
    assert result.decisions == [] and [n.node_id for n in result.terminal] == ["r"] and calls == []


@pytest.mark.parametrize("bad", ["", "0", "01", "r2", "rx", "root"])
def test_node_ids_are_non_numeric_and_validated(bad):
    with pytest.raises(Exception):
        stage1.node_depth(bad)
    assert stage1.node_depth("r") == 0 and stage1.node_depth("r010") == 3


# ------------------------------------------------------------ R44


def _entry(name, f1, regions=1, g=("G1", 200, 3), l=("L1", 20, 1)):
    return {"candidate": name, "f1_exact": f1, "terminal_regions": regions, "g_id": g[0], "global_rounds": g[1],
            "global_depth": g[2], "l_id": l[0], "local_rounds": l[1], "local_depth": l[2]}


def test_selection_absolute_f1_and_tie_order():
    entries = [
        _entry("G2L1", Fraction(1, 2), g=("G2", 400, 3)),
        _entry("G1L2", Fraction(1, 2), l=("L2", 40, 2)),
        _entry("G1L1", Fraction(1, 2), regions=3),
        _entry("G3L1", None, g=("G3", 200, 4)),
        _entry("G4L1", Fraction(2, 5), g=("G4", 400, 4)),
    ]
    out = stage1.select_winner(entries)
    # fewer terminal regions first, then fewer G+L rounds (G1L2=240 < G2L1=420)
    assert out["winner"] == "G1L2" and out["ranking"] == ["G1L2", "G2L1", "G1L1", "G4L1"]
    assert out["undefined"] == ["G3L1"]


def test_selection_all_na_is_unavailable():
    assert stage1.select_winner([_entry("G1L1", None)])["status"] == "selection_unavailable"
