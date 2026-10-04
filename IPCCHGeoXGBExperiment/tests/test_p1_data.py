"""P1: target QC/phase truth, rich561 at own origin, persistence, F/S split, calendars."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from ipcch_geoxgb import features as feat
from ipcch_geoxgb import prepare, schedule, targets
from ipcch_geoxgb.contract import load_experiment_contract, load_feature_schema
from ipcch_geoxgb.errors import ContractError

D = Decimal


def _verdict(p1, p2, p3, p4, p5, pop="100"):
    parse = lambda v: None if v is None else D(v)  # noqa: E731
    return targets.classify_row([parse(v) for v in (p1, p2, p3, p4, p5)], parse(pop))


# ---------------------------------------------------------------- targets


@pytest.mark.parametrize(
    "shares, phase",
    [
        (("0.8", "0.0", "0.2", "0", "0"), 3),  # q3 exactly .20 -> phase 3 (old strict rule said 0)
        (("0.80001", "0.0", "0.19999", "0", "0"), 1),  # q2=q3=.19999 below .20
        (("0.5", "0.3", "0.0", "0.2", "0"), 4),  # q4=.2 exactly
        (("0.6", "0.1", "0.05", "0.05", "0.2"), 5),
        (("0.85", "0.1", "0.05", "0", "0"), 1),
        (("0.7", "0.25", "0.05", "0", "0"), 2),  # q2 = .30, q3 = .05
    ],
)
def test_phase_uses_inclusive_threshold(shares, phase):
    verdict = _verdict(*shares)
    assert verdict.valid == 1 and verdict.phase == phase


def test_threshold_is_relative_to_unnormalized_sum():
    # S = 1.05; P3 = .21 -> q3 = .21/1.05 = .20 exactly -> phase 3
    v = _verdict("0.6", "0.24", "0.21", "0", "0")
    assert v.total == D("1.05") and v.phase == 3
    assert v.cumulative[1] == D("0.2")
    # one unit lower in the last place: q3 < .20
    assert _verdict("0.6", "0.24", "0.20999", "0", "0").phase == 2


@pytest.mark.parametrize(
    "shares, valid, reason",
    [
        (("0.5", "0.2", "0.1", "0.1", "0.0"), 1, ""),  # S = .90 inclusive
        (("0.5", "0.2", "0.1", "0.09", "0.0"), 0, "sum_out_of_bounds"),  # .89
        (("0.6", "0.2", "0.1", "0.1", "0.1"), 1, ""),  # 1.10 inclusive
        (("0.6", "0.2", "0.1", "0.1", "0.11"), 0, "sum_out_of_bounds"),
        (("0.5", None, "0.2", "0.1", "0.2"), 0, "missing_phase_1_to_4"),
        (("1.2", "0", "0", "0", "0"), 0, "phase_share_out_of_bounds"),
    ],
)
def test_sum_bounds_and_missing_rules(shares, valid, reason):
    v = _verdict(*shares)
    assert (v.valid, v.reason) == (valid, reason)


def test_missing_p5_is_filled_and_flagged_but_p1_to_p4_are_not():
    v = _verdict("0.7", "0.1", "0.1", "0.1", None)
    assert v.valid == 1 and v.p5_filled and v.cumulative[3] == 0
    assert _verdict("0.7", "0.1", None, "0.1", "0.1").reason == "missing_phase_1_to_4"
    assert _verdict("0.7", "0.1", "0.1", "0.1", "0", pop="0").reason == "population_not_positive"
    assert _verdict("0.7", "0.1", "0.1", "0.1", "0", pop=None).reason == "population_missing"


def test_class_mappings():
    assert targets.four_class(np.array([1, 2, 3, 4, 5])).tolist() == [0, 1, 2, 3, 3]
    assert targets.binary_crisis(np.array([1, 2, 3, 4, 5])).tolist() == [0, 0, 1, 1, 1]
    with pytest.raises(ContractError):
        targets.four_class(np.array([0]))


def _write_raw(tmp_path, rows):
    cols = ["admin_code", "year", "month", *targets.PHASE_COLUMNS, "estimated_population", "overall_phase"]
    frame = pd.DataFrame(rows, columns=cols)
    path = tmp_path / "raw.csv"
    frame.to_csv(path, index=False)
    return path


def test_ledger_keeps_invalid_rows_and_reported_phase(tmp_path):
    path = _write_raw(
        tmp_path,
        [
            (7, 2020, 2, "0.8", "0", "0.2", "0", "", "10", "2"),
            (7, 2020, 1, "", "", "", "", "", "10", "3"),
        ],
    )
    ledger = targets.build_target_ledger(path)
    assert ledger["month"].tolist() == [1, 2]  # sorted by (area, month)
    assert ledger["target_invalid_reason"].tolist() == ["missing_phase_1_to_4", ""]
    assert ledger["overall_phase_raw"].tolist() == ["3", "2"]  # preserved, never used as truth
    valid = targets.valid_targets(ledger)
    assert valid["phase_truth"].tolist() == [3] and valid["p5_missing_filled"].tolist() == [1]
    assert valid["q3"].tolist() == [0.2]


@pytest.mark.parametrize(
    "rows",
    [
        [(7, 2020, 1, "0.8", "0", "0.2", "0", "0", "1", ""), (7, 2020, 1, "0.8", "0", "0.2", "0", "0", "1", "")],
        [(7, 2020, 13, "0.8", "0", "0.2", "0", "0", "1", "")],
    ],
    ids=["duplicate-key", "bad-month"],
)
def test_ledger_rejects_malformed_keys(tmp_path, rows):
    with pytest.raises(ContractError):
        targets.build_target_ledger(_write_raw(tmp_path, rows))


# ---------------------------------------------------------------- features


def _synthetic(seed=0):
    """3 areas; monthly panel 2019-01..2022-12; sparse valid outcomes."""
    rng = np.random.default_rng(seed)
    months = np.arange(feat.month_ordinal(2019, 1), feat.month_ordinal(2022, 12) + 1)
    rows = []
    for area in (7, 10, 35):
        for m in months:
            if area == 35 and m == feat.month_ordinal(2021, 6):
                continue  # a missing panel month
            rows.append((area, m))
    panel = pd.DataFrame(rows, columns=["admin_code", "month_ord"])
    for name in feat.RAW_FEATURE_COLUMNS:
        panel[name] = rng.normal(size=len(panel))
    observed = {
        7: [(2020, 2), (2020, 6), (2020, 10), (2021, 2), (2021, 6), (2021, 10), (2022, 2), (2022, 6)],
        10: [(2021, 2), (2022, 6)],
        35: [(2021, 7), (2021, 8), (2021, 9)],
    }
    recs = []
    for area, dates in observed.items():
        for y, mo in dates:
            p = rng.dirichlet(np.ones(5))
            q = [p[k:].sum() for k in range(1, 5)]
            phase = max([1] + [k + 2 for k, value in enumerate(q) if value >= 0.2])
            recs.append(
                {
                    "admin_code": area,
                    "month_ord": int(feat.month_ordinal(y, mo)),
                    **{f"p{i + 1}": p[i] for i in range(5)},
                    **{f"q{k + 2}": q[k] for k in range(4)},
                    "phase_truth": phase,
                    "crisis_truth": int(phase >= 3),
                }
            )
    valid = pd.DataFrame(recs).sort_values(["admin_code", "month_ord"]).reset_index(drop=True)
    return panel, valid


def _rich(panel, valid, h):
    schema = load_feature_schema()
    grid = feat.build_panel_grid(panel)
    index = feat.build_history_index(valid)
    country = {7: "A", 10: "A", 35: "B"}
    return prepare.build_horizon(valid, panel, grid, index, schema, h, country)


def test_rich561_order_and_no_history_rows():
    panel, valid = _synthetic()
    X, keys, audit = _rich(panel, valid, 3)
    names = load_feature_schema()["ordered_names"]
    assert X.shape == (len(valid), 561)
    first = keys.index[(keys.admin_code == 10)][0]  # 2021-02, H3: no earlier observation
    for name in ("last_observed_label", "hist_q3_obs1", "hist_q3_all_mean"):
        assert np.isnan(X[first, names.index(name)])
    assert X[first, names.index("no_observed_label_history")] == 1.0
    assert keys.loc[first, "persistence_available"] == 0 and np.isnan(keys.loc[first, "persistence_q3"])
    assert set(audit["aliases_verified"]) == set(load_feature_schema()["aliases"])


def test_future_perturbation_leaves_own_origin_features_unchanged():
    panel, valid = _synthetic()
    X, keys, _ = _rich(panel, valid, 3)
    row = keys.index[(keys.admin_code == 7) & (keys.target_month == "2021-06")][0]
    origin = keys.loc[row, "origin_ord"]  # 2021-03
    p2, v2 = panel.copy(), valid.copy()
    later_panel = p2["month_ord"] > origin
    p2.loc[later_panel, list(feat.RAW_FEATURE_COLUMNS)] += 1000.0
    later = v2["month_ord"] > origin
    v2.loc[later, ["q2", "q3", "q4", "q5"]] = 0.999
    v2.loc[later, "phase_truth"] = 5
    v2.loc[later, "crisis_truth"] = 1
    X2, keys2, _ = _rich(p2, v2, 3)
    np.testing.assert_array_equal(X[row], X2[row])
    assert keys.loc[row, ["persistence_phase", "persistence_q3", "persistence_source_month"]].tolist() == \
        keys2.loc[row, ["persistence_phase", "persistence_q3", "persistence_source_month"]].tolist()


def test_persistence_is_latest_observation_at_or_before_origin():
    panel, valid = _synthetic()
    _, keys, _ = _rich(panel, valid, 4)
    row = keys.index[(keys.admin_code == 7) & (keys.target_month == "2021-06")][0]  # O = 2021-02
    assert keys.loc[row, "persistence_source_month"] == "2021-02"  # observation AT O counts
    assert keys.loc[row, "persistence_age_months"] == 0
    src = valid[(valid.admin_code == 7) & (valid.month_ord == feat.month_ordinal(2021, 2))].iloc[0]
    assert keys.loc[row, "persistence_phase"] == src.phase_truth and keys.loc[row, "persistence_q3"] == src.q3


def test_calendar_lag_is_not_a_row_shift():
    panel, valid = _synthetic()
    X, keys, _ = _rich(panel, valid, 1)
    names = load_feature_schema()["ordered_names"]
    row = keys.index[(keys.admin_code == 35) & (keys.target_month == "2021-08")][0]  # O = 2021-07
    lag1 = X[row, names.index("EVI_mean_lag1_asof")]  # 2021-06 is absent from the panel
    assert np.isnan(lag1)
    assert np.isnan(X[row, names.index("WFP_Price_sum4_asof")])  # window includes the gap


def test_history_window_edges():
    panel, valid = _synthetic()
    X, keys, _ = _rich(panel, valid, 1)
    names = load_feature_schema()["ordered_names"]
    row = keys.index[(keys.admin_code == 7) & (keys.target_month == "2022-06")][0]  # O = 2022-05
    # m06 covers 2021-12..2022-05 -> only 2022-02; m12 covers 2021-06..2022-05 -> 3 obs
    assert X[row, names.index("hist_support_common_m06_count")] == 1
    assert X[row, names.index("hist_support_common_m12_count")] == 3
    assert np.isnan(X[row, names.index("hist_q3_m06_std")])  # std needs 2 observations
    assert np.isfinite(X[row, names.index("hist_q3_m12_slope")])  # slope needs 3


def test_history_infinity_stops():
    panel, valid = _synthetic()
    index = feat.build_history_index(valid)
    index.series["q3"][0] = np.inf
    names = [n for b in load_feature_schema()["additional_blocks"].values() for n in b]
    # origin 2020-02 makes the poisoned observation slot 1, so an emitted column is infinite
    with pytest.raises(ContractError, match="infinity"):
        feat.build_history_block(index, np.array([7]), np.array([feat.month_ordinal(2020, 2)]), names)
    # and an infinite share never reaches the index in the first place
    valid.loc[0, "q3"] = np.inf
    with pytest.raises(ContractError, match="non-finite"):
        feat.build_history_index(valid)


def test_raw_infinity_becomes_nan_with_audit():
    panel, valid = _synthetic()
    panel.loc[panel.index[0], "CPI"] = np.inf
    _, audit = feat.assemble_original93(valid, panel, 1)
    assert audit["panel_infinities_converted"] == {"CPI": 1}


def test_crisis_state_features_use_inclusive_threshold():
    panel, valid = _synthetic()
    valid.loc[0, ["q3", "phase_truth", "crisis_truth"]] = [0.2, 3, 1]
    X, keys, _ = _rich(panel, valid, 1)
    names = load_feature_schema()["ordered_names"]
    row = keys.index[(keys.admin_code == 7) & (keys.target_month == "2020-06")][0]
    assert X[row, names.index("last_observed_label")] == 1.0


# ---------------------------------------------------------------- schedule


def test_main_calendar_and_supplement():
    contract = load_experiment_contract()
    main = schedule.main_fold_calendar(contract)
    assert len(main) == 122
    assert main.groupby("horizon_months").size().to_dict() == {1: 35, 3: 33, 6: 30, 12: 24}
    assert main.groupby("horizon_months")["target_month"].min().to_dict() == {
        1: "2023-02", 3: "2023-04", 6: "2023-07", 12: "2024-01"}
    assert (main["origin_month"] >= "2023-01").all()
    valid = pd.DataFrame({"month_ord": [feat.month_ordinal(2026, 2)] * 3 + [feat.month_ordinal(2025, 5)]})
    supp, coverage = schedule.supplementary_calendar(contract, valid)
    assert len(coverage) == 12 and coverage["valid_outcomes"].sum() == 3
    assert supp["target_month"].unique().tolist() == ["2026-02"] and len(supp) == 4
    support = schedule.attach_fold_support(main, valid)
    assert (support.loc[support.target_month == "2025-05", "eval_keys"] == 1).all()
    assert (support.loc[support.target_month == "2025-06", "status"] == "no_valid_target").all()


def test_stage1_split_within_area_halves():
    rows = [(1, m) for m in range(5)] + [(2, 0), (2, 1)] + [(3, 7)]
    valid = pd.DataFrame(rows, columns=["admin_code", "month_ord"])
    valid["month_ord"] += feat.month_ordinal(2015, 1)
    valid["phase_truth"] = 1
    valid["crisis_truth"] = 0
    split, audit = schedule.stage1_split(valid, "2014-01", "2022-12", [1, 2, 3, 4])
    roles = split.groupby("admin_code")["split_role"].apply(list).to_dict()
    assert roles[1] == ["fit", "fit", "validation", "validation", "validation"]  # floor(5/2) F
    assert roles[2] == ["fit", "validation"]
    assert roles[3] == ["singleton"]
    assert audit["zero_outcome_areas"] == 1


def test_window_and_gate_dates():
    assert schedule.training_window(100) == (65, 100)
    dates = schedule.historical_gate_dates(np.array([90, 91, 91, 95, 97, 98, 99, 100, 101]), 100)
    assert dates.tolist() == [99, 98, 97, 95, 91, 90]  # U < O, distinct, newest first, max 6
