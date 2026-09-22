"""Runnable contract checks for the population-history experiment.

Hand-computable fixtures only. Every expected number below was worked out
independently of the implementation -- the point is to catch a formula that is
wrong, not to echo back the constant the code already uses.

    python3.12.exe -B IPCCHPopulationHistoryExperiment/test_contracts.py

The gate on the real 42,695 valid rows lives in ``prepare_data.prepare`` and is
exercised by the runner, not here.
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IPCCHGeoRFExperiment import prepare_data as ipcch  # noqa: E402
from IPCCHPopulationHistoryExperiment import prepare_data as prep  # noqa: E402
from IPCCHPopulationHistoryExperiment import report_results as rep  # noqa: E402
from IPCCHPopulationHistoryExperiment import run_pipeline as pipe  # noqa: E402


class SkipTest(Exception):
    """Raised when an optional pinned input is unavailable."""


SPEC = prep.load_frozen_spec()
NAMES = {name: i for i, name in enumerate(SPEC.additional_features)}


def ordinal(year: int, month: int) -> int:
    return int(ipcch.month_ordinal(year, month))


# --------------------------------------------------------------------------
# The fixture
#
# Area 100 has four observations chosen to hit every awkward rule at once:
# an exactly-.20 share, a q3 == 0 month that leaves severe_fraction undefined,
# a uniform month whose entropy is exactly 1, and an off-by-one window edge.
# Area 200 has a single observation, which is where "one value" rules bite.
# --------------------------------------------------------------------------

FIXTURE = [
    # (admin, year, month, label, p1, p2, p3, p4, p5)
    (100, 2019, 1, 0, "0.5", "0.3", "0.2", "0.0", "0.0"),
    (100, 2019, 7, 1, "0.2", "0.2", "0.2", "0.2", "0.2"),
    (100, 2020, 1, 0, "1.0", "0.0", "0.0", "0.0", "0.0"),
    (100, 2020, 3, 1, "0.0", "0.0", "0.5", "0.5", "0.0"),
    (200, 2019, 6, 1, "0.0", "0.0", "0.0", "0.0", "1.0"),
]

A100 = [ordinal(2019, 1), ordinal(2019, 7), ordinal(2020, 1), ordinal(2020, 3)]
A200 = [ordinal(2019, 6)]


def make_valid() -> pd.DataFrame:
    rows = []
    for admin, year, month, label, *shares in FIXTURE:
        row = {
            "admin_code": admin,
            "year": year,
            "month": month,
            "ipcch_food_crisis": label,
        }
        for column, value in zip(ipcch.NORMALIZED_PHASE_COLUMNS, shares):
            row[column] = value
        rows.append(row)
    return pd.DataFrame(rows)


def block(origins: list[tuple[int, int]]) -> np.ndarray:
    index = prep.build_history_index(make_valid())
    admin = np.array([a for a, _ in origins], dtype=np.int64)
    origin = np.array([o for _, o in origins], dtype=np.int64)
    return prep.build_history_block(index, admin, origin, SPEC.additional_features)


def value(matrix: np.ndarray, row: int, name: str) -> float:
    return float(matrix[row, NAMES[name]])


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if math.isnan(expected):
        assert math.isnan(actual), f"expected NaN, got {actual}"
        return
    assert abs(actual - expected) <= tolerance, f"expected {expected}, got {actual}"


# --------------------------------------------------------------------------
# Series definitions
# --------------------------------------------------------------------------


def test_series_match_hand_computation():
    series = prep.compute_series(make_valid())
    # Row 1: the uniform distribution. Entropy is exactly 1 there, because
    # -sum(.2 ln .2) = ln 5, and the definition divides by ln 5.
    close(series["q3"].iloc[1], 0.6)
    close(series["q4"].iloc[1], 0.4)
    close(series["severity_index"].iloc[1], 3.0)
    close(series["concentration"].iloc[1], 0.2)
    close(series["entropy"].iloc[1], 1.0)
    close(series["severe_fraction"].iloc[1], 0.4 / 0.6)
    # Row 0: exactly at the .20 threshold, and q4 == 0 makes the ratio 0, not
    # missing -- undefined is reserved for q3 == 0.
    close(series["q3"].iloc[0], 0.2)
    close(series["severe_fraction"].iloc[0], 0.0)
    close(series["severity_index"].iloc[0], 1.7)
    close(series["concentration"].iloc[0], 0.38)
    # Row 2: a degenerate all-P1 month. 0 ln 0 = 0, so entropy is exactly 0.
    close(series["entropy"].iloc[2], 0.0)
    close(series["concentration"].iloc[2], 1.0)
    assert math.isnan(series["severe_fraction"].iloc[2]), "q3 == 0 must leave the ratio missing"


def test_exact_threshold_history_uses_the_ledger_label():
    """A q3 of exactly .20 is non-crisis, and the state comes from the ledger."""
    matrix = block([(100, ordinal(2019, 1))])
    close(value(matrix, 0, "hist_q3_obs1"), 0.2)
    close(value(matrix, 0, "hist_q3_margin_obs1"), 0.0)
    close(value(matrix, 0, "hist_crisis_all_fraction"), 0.0)

    # Flipping only the stored label must move the classification history while
    # leaving q3 untouched: the code must not re-derive the state from a share.
    frame = make_valid()
    frame.loc[0, "ipcch_food_crisis"] = 1
    index = prep.build_history_index(frame)
    flipped = prep.build_history_block(
        index, np.array([100]), np.array([ordinal(2019, 1)]), SPEC.additional_features
    )
    close(float(flipped[0, NAMES["hist_q3_obs1"]]), 0.2)
    close(float(flipped[0, NAMES["hist_crisis_all_fraction"]]), 1.0)


# --------------------------------------------------------------------------
# Own-origin masking and window edges
# --------------------------------------------------------------------------


def test_history_never_reaches_past_its_own_origin():
    matrix = block([(100, ordinal(2019, 7)), (100, ordinal(2019, 6))])
    # At 2019-07 the observation of that same month IS visible (<= o).
    close(value(matrix, 0, "hist_q3_obs1"), 0.6)
    close(value(matrix, 0, "hist_support_common_all_count"), 2.0)
    # One month earlier it is not, and only the 2019-01 record remains.
    close(value(matrix, 1, "hist_q3_obs1"), 0.2)
    close(value(matrix, 1, "hist_support_common_all_count"), 1.0)


def test_no_history_at_all_is_missing_not_zero():
    matrix = block([(100, ordinal(2018, 12))])
    close(value(matrix, 0, "hist_support_common_all_count"), 0.0)
    close(value(matrix, 0, "hist_support_common_m12_span"), float("nan"))
    close(value(matrix, 0, "hist_q3_obs1"), float("nan"))
    close(value(matrix, 0, "hist_q3_all_mean"), float("nan"))
    close(value(matrix, 0, "hist_crisis_all_fraction"), float("nan"))
    close(value(matrix, 0, "hist_crisis_all_pairs"), 0.0)
    close(value(matrix, 0, "hist_no_noncrisis"), 1.0)
    close(value(matrix, 0, "hist_no_entry"), 1.0)
    close(value(matrix, 0, "hist_current_run_count"), float("nan"))


def test_window_edges_are_inclusive_on_both_ends():
    """m12 at o covers [o-11, o]; one month of drift changes the membership."""
    # At 2020-01 the 12-month window starts 2019-02, so 2019-01 is outside.
    at_2020_01 = block([(100, ordinal(2020, 1))])
    close(value(at_2020_01, 0, "hist_support_common_m12_count"), 2.0)
    # At 2019-12 it starts 2019-01, so that record is the oldest member.
    at_2019_12 = block([(100, ordinal(2019, 12))])
    close(value(at_2019_12, 0, "hist_support_common_m12_count"), 2.0)
    close(value(at_2019_12, 0, "hist_support_common_m12_span"), float(A100[1] - A100[0]))
    # The 6-month window at 2020-01 holds only that month's own record.
    close(value(at_2020_01, 0, "hist_support_common_m06_count"), 1.0)
    close(value(at_2020_01, 0, "hist_support_common_m06_span"), 0.0)
    # A single-observation window has a span of 0 but no standard deviation.
    close(value(at_2020_01, 0, "hist_q3_m06_std"), float("nan"))
    close(value(at_2020_01, 0, "hist_q3_m06_mean"), 0.0)


def test_rates_use_the_actual_month_gap():
    matrix = block([(100, ordinal(2020, 3))])
    # obs1 = 2020-03 (q3 1.0), obs2 = 2020-01 (q3 0.0); the gap is two months.
    close(value(matrix, 0, "hist_gap_obs1_obs2"), 2.0)
    close(value(matrix, 0, "hist_q3_change_obs1_obs2"), 1.0)
    close(value(matrix, 0, "hist_q3_rate_obs1_obs2"), 0.5)
    # obs3 = 2019-07 is six months before obs2.
    close(value(matrix, 0, "hist_gap_obs2_obs3"), 6.0)
    close(value(matrix, 0, "hist_q3_change_obs2_obs3"), -0.6)
    close(value(matrix, 0, "hist_q3_rate_obs2_obs3"), -0.1)
    # A slot that does not exist gives a missing difference AND a missing rate.
    close(value(matrix, 0, "hist_q3_change_obs4_obs5"), float("nan"))
    close(value(matrix, 0, "hist_q3_rate_obs4_obs5"), float("nan"))


def test_ages_are_measured_from_the_row_origin():
    matrix = block([(100, ordinal(2020, 6))])
    close(value(matrix, 0, "hist_age_obs2"), float(ordinal(2020, 6) - A100[2]))
    close(value(matrix, 0, "hist_age_obs3"), float(ordinal(2020, 6) - A100[1]))
    close(value(matrix, 0, "hist_support_common_m36_age"), float(ordinal(2020, 6) - A100[3]))


# --------------------------------------------------------------------------
# Support rules
# --------------------------------------------------------------------------


def test_ratio_support_excludes_undefined_observations():
    """severe_fraction skips the q3 == 0 month in its count, span and age."""
    matrix = block([(100, ordinal(2020, 3))])
    close(value(matrix, 0, "hist_support_common_all_count"), 4.0)
    close(value(matrix, 0, "hist_support_severe_fraction_all_count"), 3.0)
    # Oldest and newest DEFINED ratio observations are 2019-01 and 2020-03.
    close(value(matrix, 0, "hist_support_severe_fraction_all_span"), float(A100[3] - A100[0]))
    close(value(matrix, 0, "hist_support_severe_fraction_all_age"), 0.0)
    # The undefined month is still carried as a missing level, not dropped.
    close(value(matrix, 0, "hist_severe_fraction_obs2"), float("nan"))
    close(value(matrix, 0, "hist_severe_fraction_obs3"), 0.4 / 0.6)


def test_statistic_support_thresholds():
    """mean needs one value, std two, slope three; short of that it is missing."""
    one = block([(200, ordinal(2019, 6))])
    close(value(one, 0, "hist_q3_all_mean"), 1.0)
    close(value(one, 0, "hist_q3_all_std"), float("nan"))
    close(value(one, 0, "hist_q3_all_slope"), float("nan"))
    close(value(one, 0, "hist_q3_all_min"), 1.0)
    close(value(one, 0, "hist_q3_all_latest_minus_mean"), 0.0)
    close(value(one, 0, "hist_support_common_all_span"), 0.0)

    two = block([(100, ordinal(2019, 7))])
    close(value(two, 0, "hist_q3_all_mean"), 0.4)
    close(value(two, 0, "hist_q3_all_std"), 0.2)  # population sd of {.2, .6}
    close(value(two, 0, "hist_q3_all_slope"), float("nan"))

    three = block([(100, ordinal(2020, 1))])
    close(value(three, 0, "hist_q3_all_mean"), (0.2 + 0.6 + 0.0) / 3)


def test_constant_window_has_exactly_zero_standard_deviation():
    """The one-pass E[x^2]-mean^2 identity fails here; a two-pass one does not."""
    rows = []
    for month in (1, 4, 7):
        rows.append((300, 2019, month, 1, "0.0", "0.5", "0.5", "0.0", "0.0"))
    frame = pd.DataFrame(
        [
            {
                "admin_code": a,
                "year": y,
                "month": m,
                "ipcch_food_crisis": label,
                **dict(zip(ipcch.NORMALIZED_PHASE_COLUMNS, shares)),
            }
            for a, y, m, label, *shares in rows
        ]
    )
    index = prep.build_history_index(frame)
    matrix = prep.build_history_block(
        index, np.array([300]), np.array([ordinal(2019, 7)]), SPEC.additional_features
    )
    assert float(matrix[0, NAMES["hist_q3_all_std"]]) == 0.0, "constant window must give exactly 0"
    assert float(matrix[0, NAMES["hist_q3_all_slope"]]) == 0.0
    close(float(matrix[0, NAMES["hist_q3_all_latest_minus_mean"]]), 0.0)


def test_slope_matches_an_independent_least_squares_fit():
    matrix = block([(100, ordinal(2020, 3))])
    months = np.array(A100, dtype=float)
    values = np.array([0.2, 0.6, 0.0, 1.0])
    expected = np.polyfit(months, values, 1)[0]
    close(value(matrix, 0, "hist_q3_all_slope"), float(expected), tolerance=1e-10)
    # last3 uses the three newest slots and their real ordinals.
    expected3 = np.polyfit(months[1:], values[1:], 1)[0]
    close(value(matrix, 0, "hist_q3_slope_last3"), float(expected3), tolerance=1e-10)


# --------------------------------------------------------------------------
# Events, runs and pair counting
# --------------------------------------------------------------------------


def test_events_are_dated_at_the_later_observed_endpoint():
    origin = ordinal(2020, 6)
    matrix = block([(100, origin)])
    # States oldest to newest: 0, 1, 0, 1.
    close(value(matrix, 0, "hist_noncrisis_age"), float(origin - A100[2]))
    close(value(matrix, 0, "hist_no_noncrisis"), 0.0)
    close(value(matrix, 0, "hist_entry_age"), float(origin - A100[3]))
    close(value(matrix, 0, "hist_exit_age"), float(origin - A100[2]))
    close(value(matrix, 0, "hist_crisis_all_entries"), 2.0)
    close(value(matrix, 0, "hist_crisis_all_exits"), 1.0)
    close(value(matrix, 0, "hist_crisis_all_pairs"), 3.0)
    # The current run is the newest block of equal states: just 2020-03.
    close(value(matrix, 0, "hist_current_run_count"), 1.0)
    close(value(matrix, 0, "hist_current_run_span"), 0.0)


def test_absent_event_is_a_flag_not_an_age():
    matrix = block([(200, ordinal(2019, 6))])
    close(value(matrix, 0, "hist_noncrisis_age"), float("nan"))
    close(value(matrix, 0, "hist_no_noncrisis"), 1.0)
    close(value(matrix, 0, "hist_entry_age"), float("nan"))
    close(value(matrix, 0, "hist_no_entry"), 1.0)
    close(value(matrix, 0, "hist_no_exit"), 1.0)
    close(value(matrix, 0, "hist_current_run_count"), 1.0)
    close(value(matrix, 0, "hist_current_run_span"), 0.0)
    # One record means no pair exists, so the counts are missing while the
    # eligible-pair count is a definite zero.
    close(value(matrix, 0, "hist_crisis_all_entries"), float("nan"))
    close(value(matrix, 0, "hist_crisis_all_pairs"), 0.0)


def test_pairs_need_both_endpoints_inside_the_window():
    origin = ordinal(2020, 3)
    matrix = block([(100, origin)])
    # The 6-month window holds 2020-01 and 2020-03 only: one pair, one entry.
    close(value(matrix, 0, "hist_support_common_m06_count"), 2.0)
    close(value(matrix, 0, "hist_crisis_m06_pairs"), 1.0)
    close(value(matrix, 0, "hist_crisis_m06_entries"), 1.0)
    close(value(matrix, 0, "hist_crisis_m06_exits"), 0.0)
    close(value(matrix, 0, "hist_crisis_m06_fraction"), 0.5)


def test_history_does_not_leak_across_areas():
    matrix = block([(200, ordinal(2020, 3))])
    close(value(matrix, 0, "hist_support_common_all_count"), 1.0)
    close(value(matrix, 0, "hist_q3_obs1"), 1.0)
    close(value(matrix, 0, "hist_q3_obs2"), float("nan"))


def test_alias_recomputation_detects_a_mismatch():
    valid = make_valid()
    index = prep.build_history_index(valid)
    admin = np.array([100, 100, 200], dtype=np.int64)
    origin = np.array([ordinal(2020, 3), ordinal(2018, 12), ordinal(2019, 6)], dtype=np.int64)

    original = np.zeros((3, len(SPEC.original_features)), dtype=np.float64)
    position = {name: i for i, name in enumerate(SPEC.original_features)}
    original[:, position["last_observed_label_age_months"]] = [0.0, np.nan, 0.0]
    original[:, position["last_observed_label"]] = [1.0, np.nan, 1.0]
    original[:, position["months_since_last_observed_crisis"]] = [0.0, np.nan, 0.0]
    original[:, position["no_prior_observed_crisis"]] = [0.0, 1.0, 0.0]
    prep.check_aliases(original, SPEC, index, admin, origin)

    original[0, position["last_observed_label"]] = 0.0
    try:
        prep.check_aliases(original, SPEC, index, admin, origin)
    except prep.PreparationError:
        return
    raise AssertionError("a perturbed alias target was not detected")


# --------------------------------------------------------------------------
# Scores, thresholds and constants
# --------------------------------------------------------------------------


def test_correction_scores_are_oriented_toward_crisis():
    arm = pipe.ARMS_BY_NAME["correction_xgb"]
    raw = np.array([0.1, 0.9, 0.1, 0.9])
    b = np.array([0.0, 0.0, 1.0, 1.0])
    crisis = pipe.crisis_oriented(arm, raw, b)
    # b = 0: a high error probability means "crisis", so the score passes through.
    close(float(crisis[0]), 0.1)
    close(float(crisis[1]), 0.9)
    # b = 1: a high error probability means "not crisis", so it is mirrored.
    close(float(crisis[2]), 0.9)
    close(float(crisis[3]), 0.1)


def test_share_scores_are_clipped_but_the_raw_output_survives():
    arm = pipe.ARMS_BY_NAME["share_xgb"]
    raw = np.array([-0.3, 0.4, 1.7])
    crisis = pipe.crisis_oriented(arm, raw, np.zeros(3))
    assert list(crisis) == [0.0, 0.4, 1.0]
    assert list(raw) == [-0.3, 0.4, 1.7], "clipping must not mutate the raw score"


def test_decision_at_the_cutoff_is_negative():
    crisis = np.array([0.5, 0.5000001, 0.4999999])
    b = np.zeros(3)
    decision = pipe.apply_thresholds(crisis, b, 0.5, 0.5)
    assert list(decision) == [0, 1, 0], "score == t must be negative"


def test_no_flip_thresholds_reproduce_persistence():
    crisis = np.array([0.99, 0.01, 0.99, 0.01])
    b = np.array([0.0, 0.0, 1.0, 1.0])
    decision = pipe.apply_thresholds(crisis, b, np.inf, -np.inf)
    assert list(decision) == [0, 0, 1, 1], "the no-flip pair must return b exactly"


def test_threshold_candidate_grid_is_bounded_and_contains_the_fixed_cutoff():
    scores = np.random.default_rng(0).random(500)
    grid = pipe.threshold_candidates(scores, 0.5)
    assert np.isneginf(grid[0]) and np.isposinf(grid[-1])
    assert 0.5 in set(grid.tolist())
    assert len(grid) <= 104, len(grid)
    assert np.all(np.diff(grid) > 0), "the grid must be sorted and deduplicated"
    # An all-constant score collapses to a handful of distinct candidates.
    assert len(pipe.threshold_candidates(np.full(50, 0.3), 0.5)) == 4


def test_subgroup_counts_match_a_brute_force_sweep():
    rng = np.random.default_rng(3)
    scores = rng.random(200)
    truth = (rng.random(200) < 0.4).astype(float)
    grid = pipe.threshold_candidates(scores, 0.5)
    counts = pipe._subgroup_counts(scores, truth, grid)
    for i, t in enumerate(grid):
        pred = scores > t
        assert counts["tp"][i] == float(((truth == 1) & pred).sum())
        assert counts["fp"][i] == float(((truth == 0) & pred).sum())
        assert counts["fn"][i] == float(((truth == 1) & ~pred).sum())
        assert counts["predicted_positive"][i] == float(pred.sum())


def test_threshold_tie_breaks_prefer_fewer_changes_then_lower_thresholds():
    """Two pairs reach the same F1; the one that changes b less often wins."""
    frame = pd.DataFrame(
        {
            # b = 0 rows: scores .1 and .9, truth 0 and 1. Any threshold in
            # [.1, .9) recovers both, so several pairs tie at the top F1.
            "persistence_b": [0.0, 0.0, 1.0, 1.0],
            "crisis_score": [0.1, 0.9, 0.9, 0.1],
            "ipcch_food_crisis": [0, 1, 1, 0],
        }
    )
    outcome = pipe.select_thresholds(frame, 0.5)
    close(outcome["f1"], 1.0)
    assert outcome["changes_from_b"] == 2, outcome
    t0 = pipe._decode(outcome["thresholds"]["t0"])
    t1 = pipe._decode(outcome["thresholds"]["t1"])
    # Among the tied optima the smallest admissible pair is chosen.
    assert 0.1 <= t0 < 0.9, t0
    assert 0.1 <= t1 < 0.9, t1
    decisions = pipe.apply_thresholds(
        frame["crisis_score"].to_numpy(), frame["persistence_b"].to_numpy(), t0, t1
    )
    assert list(decisions) == [0, 1, 1, 0]


def test_unsupported_subgroup_falls_back_to_its_no_flip_threshold():
    frame = pd.DataFrame(
        {
            "persistence_b": [0.0, 0.0, 0.0],
            "crisis_score": [0.2, 0.8, 0.9],
            "ipcch_food_crisis": [0, 1, 1],
        }
    )
    outcome = pipe.select_thresholds(frame, 0.5)
    assert outcome["subgroups"]["1"]["supported"] is False
    assert pipe._decode(outcome["thresholds"]["t1"]) == -np.inf


def test_infinities_round_trip_through_json_tokens():
    for token in (np.inf, -np.inf, 0.25):
        assert pipe._decode(pipe._encode(token)) == token
    assert isinstance(pipe._encode(np.inf), str)


def test_constant_targets_take_the_declared_route():
    arm = pipe.ARMS_BY_NAME["rich_direct_xgb"]
    resolved = pipe.resolve_config(SPEC.configs, "xgb", {"id": "X0"}, regressor=False)
    X = np.zeros((10, 4))
    scores, route, model = pipe.fit_and_score(arm, resolved, X, np.ones(10), np.zeros((3, 4)))
    assert model is None, "a constant route has no estimator to retain"
    assert route["route"] == "constant_single_class"
    assert list(scores) == [1.0, 1.0, 1.0]

    share = pipe.ARMS_BY_NAME["share_xgb"]
    resolved = pipe.resolve_config(SPEC.configs, "xgb", {"id": "X0"}, regressor=True)
    scores, route, model = pipe.fit_and_score(share, resolved, X, np.full(10, 0.37), np.zeros((2, 4)))
    assert model is None
    assert route["route"] == "constant_target"
    assert list(scores) == [0.37, 0.37]


def test_constant_correction_target_stays_conditional_on_b():
    """An all-error fitting pool must still flip in both directions."""
    arm = pipe.ARMS_BY_NAME["correction_xgb"]
    resolved = pipe.resolve_config(SPEC.configs, "xgb", {"id": "X0"}, regressor=False)
    raw, route, _model = pipe.fit_and_score(arm, resolved, np.zeros((8, 3)), np.ones(8), np.zeros((2, 3)))
    assert route["route"] == "constant_single_class"
    crisis = pipe.crisis_oriented(arm, raw, np.array([0.0, 1.0]))
    assert list(crisis) == [1.0, 0.0], "a certain error means crisis under b=0, calm under b=1"


def test_regressor_and_classifier_config_blocks_do_not_mix():
    classifier = pipe.resolve_config(SPEC.configs, "xgb", {"id": "X0"}, regressor=False)
    regressor = pipe.resolve_config(SPEC.configs, "xgb", {"id": "X0"}, regressor=True)
    assert classifier["objective"] == "binary:logistic"
    assert regressor["objective"] == "reg:squarederror"
    assert "scale_pos_weight" not in regressor
    assert classifier["n_jobs"] == 1 and regressor["n_jobs"] == 1
    overlay = pipe.resolve_config(SPEC.configs, "xgb", {"id": "X5", "max_depth": 3}, False)
    assert overlay["max_depth"] == 3, "a candidate value must override the common block"


# --------------------------------------------------------------------------
# Calendar causality
# --------------------------------------------------------------------------


def test_development_and_main_calendars_do_not_overlap():
    calendar = prep.build_fold_calendar()
    development = calendar[calendar["stage"] == "development"]
    main = calendar[calendar["stage"] == "main"]
    assert set(development["target_month"].str.slice(0, 4)) == {"2020", "2021", "2022"}
    assert development["target_ord"].max() < main["origin_ord"].min(), (
        "every main fitting origin must be later than the last development target"
    )
    assert int(main["origin_ord"].min()) == ordinal(2023, 1)
    assert len(development) == 144 and len(main) == 122


def test_fold_masks_keep_fitting_and_evaluation_disjoint():
    keys = pd.DataFrame(
        {
            "horizon_months": [1] * 5,
            "target_ord": [ordinal(2023, 1) - d for d in (0, 1, 2, 40, -1)],
            "has_history": [1, 1, 0, 1, 1],
        }
    )
    record = pd.Series(
        {
            "fold_id": "t",
            "stage": "main",
            "horizon_months": 1,
            "target_ord": ordinal(2023, 2),
            "origin_ord": ordinal(2023, 1),
        }
    )
    support = pipe.fold_support(keys, record)
    assert not (support.test & support.full).any()
    # The 40-month-old row is outside the 36-month window; the history-less one
    # is in the full pool but not the matched pool.
    assert support.full.sum() == 3
    assert support.matched.sum() == 2
    assert support.test.sum() == 1


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _scored_fixture(identical: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    countries = ["A", "B", "C", "D"]
    for h in rep.HORIZONS:
        for i in range(60):
            truth = int(rng.random() < 0.4)
            base = int(rng.random() < 0.4)
            for method in rep.ALL_METHODS:
                pred = base if identical else int(rng.random() < 0.4)
                rows.append(
                    {
                        "arm": method,
                        "horizon_months": h,
                        "admin_code": i,
                        "target_month": f"202{3 + i % 3}-01",
                        "target_year": 2023 + i % 3,
                        "country_key": countries[i % 4],
                        "support": "E_history",
                        "ipcch_food_crisis": truth,
                        "decision": pred,
                        "persistence_b": base,
                    }
                )
    return pd.DataFrame(rows)


def test_bootstrap_shares_one_multiplicity_vector_across_methods():
    """Identical predictions must give an exactly zero delta in every draw."""
    scored = _scored_fixture(identical=True)
    pairs = [("rich_direct_xgb", "binary_history_xgb"), ("share_xgb", "persistence")]
    outcome = rep.joint_bootstrap(scored, pairs)
    assert outcome["valid_draws"] == rep.BOOTSTRAP_DRAWS
    for pair in pairs:
        draws = np.asarray(outcome["raw_draws"][f"{pair[0]}_vs_{pair[1]}"])
        assert draws.size == rep.BOOTSTRAP_DRAWS
        assert np.all(draws == 0.0), "a shared draw cannot separate identical predictions"
    interval = outcome["intervals"]["rich_direct_xgb_vs_binary_history_xgb"]
    assert interval["lower_2.5"] == 0.0 and interval["upper_97.5"] == 0.0


def test_bootstrap_is_reproducible_from_its_seed():
    scored = _scored_fixture(identical=False)
    pairs = [("rich_direct_xgb", "persistence")]
    first = rep.joint_bootstrap(scored, pairs)["raw_draws"]
    second = rep.joint_bootstrap(scored, pairs)["raw_draws"]
    assert first == second


def test_undefined_f1_rejects_the_whole_draw():
    """A cohort with no positive truth and no positive prediction is rejected."""
    scored = _scored_fixture(identical=True)
    scored.loc[:, "ipcch_food_crisis"] = 0
    scored.loc[:, "decision"] = 0
    outcome = rep.joint_bootstrap(scored, [("rich_direct_xgb", "persistence")])
    assert outcome["valid_draws"] == 0
    assert outcome["attempts"] == rep.BOOTSTRAP_MAX_ATTEMPTS
    assert not outcome["intervals"]["rich_direct_xgb_vs_persistence"]["complete"]


def test_incomplete_interval_makes_the_verdict_incomplete_not_a_pass():
    deltas = pd.DataFrame(
        {
            "method": ["a"] * 4,
            "baseline": ["b"] * 4,
            "horizon_months": list(rep.HORIZONS),
            "delta_f1": [0.05] * 4,
        }
    )
    omitted = {str(y): {"mean_delta": 0.05} for y in rep.LEAVE_OUT_YEARS}
    verdict = rep.stability_verdict(("a", "b"), deltas, {"complete": False}, omitted)
    assert verdict["verdict"] == "incomplete"
    full = rep.stability_verdict(
        ("a", "b"), deltas, {"complete": True, "lower_2.5": 0.01}, omitted
    )
    assert full["verdict"] == "stable_gain"


def test_one_negative_horizon_blocks_a_stable_gain():
    deltas = pd.DataFrame(
        {
            "method": ["a"] * 4,
            "baseline": ["b"] * 4,
            "horizon_months": list(rep.HORIZONS),
            "delta_f1": [0.2, 0.2, 0.2, -0.01],
        }
    )
    omitted = {str(y): {"mean_delta": 0.1} for y in rep.LEAVE_OUT_YEARS}
    verdict = rep.stability_verdict(
        ("a", "b"), deltas, {"complete": True, "lower_2.5": 0.05}, omitted
    )
    assert verdict["verdict"] == "no_stable_gain"
    assert verdict["conditions"]["no_negative_horizon"] is False


def test_one_bad_omitted_year_blocks_a_stable_gain():
    deltas = pd.DataFrame(
        {
            "method": ["a"] * 4,
            "baseline": ["b"] * 4,
            "horizon_months": list(rep.HORIZONS),
            "delta_f1": [0.2] * 4,
        }
    )
    omitted = {"2023": {"mean_delta": 0.1}, "2024": {"mean_delta": -0.01}, "2025": {"mean_delta": 0.1}}
    verdict = rep.stability_verdict(
        ("a", "b"), deltas, {"complete": True, "lower_2.5": 0.05}, omitted
    )
    assert verdict["verdict"] == "no_stable_gain"


def test_required_pairs_follow_the_selected_family():
    pairs, claims, prerequisites = rep.required_pairs("rich_direct_xgb")
    assert "formulation_advantage" not in claims
    assert prerequisites == {}
    assert ("rich_direct_xgb", "binary_history_xgb") in pairs
    pairs, claims, prerequisites = rep.required_pairs("correction_xgb")
    assert ("correction_xgb", "rich_direct_xgb") in pairs
    assert ("correction_xgb", "fullpool_xgb") in pairs
    assert ("correction_xgb", "persistence") in pairs
    assert prerequisites == {"formulation_advantage": "prediction_gain"}


def test_a_formulation_advantage_needs_the_prediction_gain_under_it():
    """§6 says "additionally", so beating the direct arms alone proves nothing.

    The counterexample: the primary beats both direct classifiers but loses to
    `rich_rf`. Reporting that as a supported formulation advantage while the
    prediction gain is unsupported would claim a better formulation of a
    problem the method has not been shown to predict better at all.
    """
    _pairs, claims, prerequisites = rep.required_pairs("share_xgb")
    verdicts = {
        "share_xgb_vs_rich_rf": {"verdict": "no_stable_gain"},
        "share_xgb_vs_persistence": {"verdict": "stable_gain"},
        "share_xgb_vs_rich_direct_xgb": {"verdict": "stable_gain"},
        "share_xgb_vs_fullpool_xgb": {"verdict": "stable_gain"},
        "rich_direct_xgb_vs_binary_history_xgb": {"verdict": "no_stable_gain"},
    }
    resolved = rep.evaluate_claims(claims, prerequisites, verdicts)
    assert resolved["prediction_gain"]["result"] == "not_supported"
    formulation = resolved["formulation_advantage"]
    assert formulation["own_comparisons_result"] == "supported"
    assert formulation["prerequisite_result"] == "not_supported"
    assert formulation["result"] == "not_supported", formulation

    # With the prerequisite met, its own comparisons carry it.
    verdicts["share_xgb_vs_rich_rf"] = {"verdict": "stable_gain"}
    resolved = rep.evaluate_claims(claims, prerequisites, verdicts)
    assert resolved["prediction_gain"]["result"] == "supported"
    assert resolved["formulation_advantage"]["result"] == "supported"

    # Missing evidence upstream makes it incomplete, never a pass.
    verdicts["share_xgb_vs_persistence"] = {"verdict": "incomplete"}
    resolved = rep.evaluate_claims(claims, prerequisites, verdicts)
    assert resolved["formulation_advantage"]["result"] == "incomplete"

    # Its own failure settles it regardless of what the prerequisite did.
    verdicts["share_xgb_vs_persistence"] = {"verdict": "stable_gain"}
    verdicts["share_xgb_vs_fullpool_xgb"] = {"verdict": "no_stable_gain"}
    resolved = rep.evaluate_claims(claims, prerequisites, verdicts)
    assert resolved["formulation_advantage"]["result"] == "not_supported"

    # A claim with no prerequisite is unaffected by any of this.
    assert "prerequisite" not in resolved["information_gain"]


def test_class1_metrics_report_an_undefined_denominator():
    result = rep.metrics(0, 0, 0, 10)
    assert math.isnan(result["f1"]) and "f1" in result["undefined"]
    assert math.isnan(result["precision"]) and math.isnan(result["recall"])
    exact = rep.metrics(3, 1, 2, 4)
    close(exact["f1"], 6 / 9)
    close(exact["precision"], 0.75)
    close(exact["recall"], 0.6)


# --------------------------------------------------------------------------
# Fitted-state provenance and stage immutability
# --------------------------------------------------------------------------


def test_fitted_identity_reads_the_estimator_back():
    """The record must describe the fitted model, not the dictionary passed in."""
    arm = pipe.ARMS_BY_NAME["rich_direct_xgb"]
    resolved = pipe.resolve_config(SPEC.configs, "xgb", {"id": "X1", "max_depth": 3}, False)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((120, 5))
    y = (rng.random(120) < 0.4).astype(float)
    _scores, route, model = pipe.fit_and_score(arm, resolved, X, y, X[:4])

    assert model is not None, "a real fit must hand back its estimator"
    assert route["route"] == "model"
    # get_params() reports the library's full resolved parameter set, which is
    # strictly larger than the candidate block we asked for.
    assert len(route["fitted_params"]) > len(resolved)
    assert route["fitted_params"]["max_depth"] == 3
    assert route["boosted_rounds"] == resolved["n_estimators"]
    assert route["booster_features"] == 5
    assert isinstance(route["booster_config"], dict) and route["booster_config"]
    assert "requested_params" not in route, "run_fold attaches that, not fit_and_score"


def test_saved_model_round_trips_and_is_digest_bound():
    import tempfile

    arm = pipe.ARMS_BY_NAME["rich_direct_xgb"]
    resolved = pipe.resolve_config(SPEC.configs, "xgb", {"id": "X1", "max_depth": 3}, False)
    rng = np.random.default_rng(2)
    X = rng.standard_normal((150, 6))
    y = (rng.random(150) < 0.4).astype(float)
    scores, _route, model = pipe.fit_and_score(arm, resolved, X, y, X[:20])

    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / "run" / "main" / "models" / "fold__arm__X1"
        record = pipe.save_model(model, arm, stem)
        assert record["path"] == "main/models/fold__arm__X1.ubj"
        path = Path(tmp) / "run" / record["path"]
        assert prep.sha256_file(path) == record["sha256"]
        reloaded = pipe.load_model(arm, path)
        assert np.array_equal(pipe.class1_probability(reloaded, X[:20]), scores)


def test_fold_reuse_refuses_a_record_from_another_identity():
    """A fold recorded under different inputs is not reused, it is refused."""
    import tempfile

    identity = {"spec": {"rich_count": 561}, "matrix_sha256": "aaa", "code_sha256": {"x": "1"}}
    with tempfile.TemporaryDirectory() as tmp:
        folds = Path(tmp) / "development" / "folds"
        folds.mkdir(parents=True)
        for name, recorded in (
            ("same", identity),
            ("moved", {**identity, "matrix_sha256": "bbb"}),
        ):
            (folds / f"{name}.json").write_text(
                __import__("json").dumps(
                    {"fold_id": name, "status": "complete", "identity": recorded}
                )
            )
        # A record from before identities were kept is also not reusable.
        (folds / "legacy.json").write_text(
            __import__("json").dumps({"fold_id": "legacy", "status": "complete"})
        )

        done, stale = pipe.completed_folds(tmp, "development", identity)
        assert done == {"same"}, done
        assert len(stale) == 2, stale
        assert any("matrix_sha256" in reason for reason in stale)
        assert any("no identity recorded" in reason for reason in stale)

        # Without an identity to compare against, every complete record counts.
        done, stale = pipe.completed_folds(tmp, "development", None)
        assert done == {"same", "moved", "legacy"} and stale == []


def test_a_freshly_frozen_run_reaches_main_scheduling():
    """The gate must pass on an unchanged run, or the pipeline cannot be run.

    This is the regression that a partial identity record caused: the freeze
    carried two fields, the run computed five, and the two extra ones read as
    drift, so `--stage main` refused on a run nothing had touched.
    """
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "run"
        _synthetic_run(root)
        identity = pipe.run_identity(root)
        freeze = {
            "spec": identity["spec"],
            "matrix_sha256": identity["matrix_sha256"],
            "identity": identity,
            "code_sha256": pipe.code_identity(),
        }
        outcome = pipe.check_main_preconditions(root, freeze)
        assert outcome["legacy_freeze"] is False
        assert outcome["identity_fields_unprovable"] == []
        assert outcome["code_drift_from_freeze"] is False
        assert "keys_sha256" in outcome["identity_fields_compared"]
        assert "calendar_sha256" in outcome["identity_fields_compared"]

        # Real drift in those same fields must still be refused.
        drifted = json.loads(json.dumps(freeze))
        drifted["identity"]["calendar_sha256"] = "something else"
        try:
            pipe.check_main_preconditions(root, drifted)
        except pipe.PipelineError as error:
            assert "calendar_sha256" in str(error)
        else:
            raise AssertionError("calendar drift was not refused")


def test_a_legacy_freeze_is_refused_by_name_not_waved_through():
    """An old freeze cannot prove the new fields; say so instead of guessing."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "run"
        _synthetic_run(root)
        identity = pipe.run_identity(root)
        legacy = {
            "spec": identity["spec"],
            "matrix_sha256": identity["matrix_sha256"],
            "code_sha256": pipe.code_identity(),
        }
        try:
            pipe.check_main_preconditions(root, legacy)
        except pipe.PipelineError as error:
            assert "--allow-legacy-freeze" in str(error)
            assert "keys_sha256" in str(error)
        else:
            raise AssertionError("a legacy freeze was accepted silently")

        outcome = pipe.check_main_preconditions(root, legacy, allow_legacy_freeze=True)
        assert outcome["legacy_freeze"] is True
        assert set(outcome["identity_fields_unprovable"]) == {
            "keys_sha256",
            "calendar_sha256",
        }


def test_a_stale_completed_fold_is_never_queued_for_refitting():
    """Mismatched history is rejected outright; it is never overwritten."""
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "run"
        _synthetic_run(root)
        folds = root / "main" / "folds"
        # Give one completed fold an identity from a different experiment.
        (folds / "mai_h01_2023-02.json").write_text(
            json.dumps(
                {
                    "fold_id": "mai_h01_2023-02",
                    "status": "complete",
                    "identity": {"spec": {"rich_count": 999}, "matrix_sha256": "other"},
                }
            )
        )
        try:
            pipe.stage_folds(root, "main", workers=1, selections=None)
        except pipe.PipelineError as error:
            message = str(error)
            assert "immutable" in message and "fresh run directory" in message
            assert "mai_h01_2023-02" in message
        else:
            raise AssertionError("a stale completed fold did not stop the run")


def test_a_partial_identity_record_is_not_proof():
    """Omitted fields cannot be compared, so the record cannot be reused."""
    import json
    import tempfile

    identity = {
        "spec": {"rich_count": 561},
        "matrix_sha256": "m",
        "keys_sha256": "k",
        "calendar_sha256": "c",
        "code_sha256": {"run_pipeline.py": "1"},
    }
    with tempfile.TemporaryDirectory() as tmp:
        folds = Path(tmp) / "development" / "folds"
        folds.mkdir(parents=True)
        cases = {
            "full": identity,
            "partial": {"spec": identity["spec"], "matrix_sha256": "m"},
            "empty": {},
            "no_code": {k: v for k, v in identity.items() if k != "code_sha256"},
        }
        for name, recorded in cases.items():
            (folds / f"{name}.json").write_text(
                json.dumps({"fold_id": name, "status": "complete", "identity": recorded})
            )
        done, stale = pipe.completed_folds(tmp, "development", identity)
        assert done == {"full"}, done
        assert len(stale) == 3, stale
        assert any("omits" in reason and "keys_sha256" in reason for reason in stale)
        assert any("omits" in reason and "code_sha256" in reason for reason in stale)


def _development_cohort_fixture(tmp: Path):
    """A tiny but complete development cohort: 2 folds, 2 arms, 2 candidates."""
    import json

    months = ["2020-01", "2020-02"]
    areas = [1, 2, 3]
    keys = pd.DataFrame(
        [
            {
                "admin_code": a,
                "target_month": m,
                "horizon_months": h,
                "has_history": 1,
            }
            for h in rep.HORIZONS
            for m in months
            for a in areas
        ]
    )
    calendar = pd.DataFrame(
        [
            {
                "fold_id": f"dev_h{h:02d}_{m}",
                "stage": "development",
                "horizon_months": h,
                "target_month": m,
                "test_rows": len(areas),
                "test_rows_with_history": len(areas),
            }
            for h in rep.HORIZONS
            for m in months
        ]
    )
    folds = tmp / "development" / "folds"
    folds.mkdir(parents=True, exist_ok=True)
    for fold_id in calendar["fold_id"]:
        (folds / f"{fold_id}.json").write_text(
            json.dumps({"fold_id": fold_id, "status": "complete"})
        )
    history = pd.DataFrame(
        [
            {
                "fold_id": f"dev_h{h:02d}_{m}",
                "arm": arm,
                "config_id": cfg,
                "horizon_months": h,
                "admin_code": a,
                "target_month": m,
            }
            for h in rep.HORIZONS
            for m in months
            for arm in ("rich_direct_xgb", "share_xgb")
            for cfg in ("X0", "X1")
            for a in areas
        ]
    )
    return keys, calendar, history


def test_selection_refuses_an_incomplete_development_cohort():
    """A missing fold must stop the freeze, not quietly shrink the search."""
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        keys, calendar, history = _development_cohort_fixture(tmp)
        intact = pipe.reconcile_development_cohort(tmp, keys, calendar, history)
        assert intact["supported_folds"] == 8
        assert intact["folds_with_predictions"] == 8

        # One fold's predictions vanish: persistence would still be scored on
        # the full calendar, so the comparison would be mismatched.
        lost = history[history["fold_id"] != "dev_h01_2020-02"]
        try:
            pipe.reconcile_development_cohort(tmp, keys, calendar, lost)
        except pipe.PipelineError as error:
            assert "contributed no E_history predictions" in str(error)
        else:
            raise AssertionError("a missing development fold was not detected")

        # One candidate silently covering fewer keys is equally fatal.
        short = history.drop(
            history[
                (history["arm"] == "share_xgb")
                & (history["config_id"] == "X1")
                & (history["admin_code"] == 3)
            ].index
        )
        try:
            pipe.reconcile_development_cohort(tmp, keys, calendar, short)
        except pipe.PipelineError as error:
            assert "expected" in str(error)
        else:
            raise AssertionError("an unequal candidate support was not detected")

        # A duplicated key would double-count into the pooled confusion table.
        doubled = pd.concat([history, history.iloc[[0]]], ignore_index=True)
        try:
            pipe.reconcile_development_cohort(tmp, keys, calendar, doubled)
        except pipe.PipelineError as error:
            assert "duplicated" in str(error)
        else:
            raise AssertionError("a duplicated evaluation key was not detected")


def _pilot_run(root: Path) -> str:
    """Minimal run directory with one supported 2020 h=1 development fold."""
    import json

    root.mkdir(parents=True, exist_ok=True)
    (root / "inputs").mkdir(exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    (root / "folds").mkdir(exist_ok=True)
    for name in ("feature-schema.json", "candidate-configs.json"):
        (root / "inputs" / name).write_bytes((prep.CONFIG_DIR / name).read_bytes())
    keys = pd.DataFrame(
        {
            "admin_code": [1, 2],
            "target_month": ["2020-01", "2020-01"],
            "horizon_months": [1, 1],
            "target_ord": [ordinal(2020, 1)] * 2,
            "has_history": [1, 1],
        }
    )
    keys.to_csv(root / "data" / "keys.csv.gz", index=False)
    pd.DataFrame(
        [
            {
                "fold_id": "dev_h01_2020-01",
                "stage": "development",
                "horizon_months": 1,
                "target_month": "2020-01",
                "target_ord": ordinal(2020, 1),
                "origin_ord": ordinal(2020, 1) - 1,
                "test_rows": 2,
                "test_rows_with_history": 2,
            }
        ]
    ).to_csv(root / "folds" / "calendar.csv", index=False)
    (root / "manifest.json").write_text(
        json.dumps({"matrix": {"matrix_sha256": "deadbeef"}})
    )
    return "dev_h01_2020-01"


def test_the_pilot_reuses_a_matching_fold_and_refuses_a_mismatched_one():
    """Repeating the pilot command must never refit over a completed fold."""
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw) / "run"
        fold_id = _pilot_run(root)
        folds = root / "development" / "folds"
        folds.mkdir(parents=True)

        # An exactly matching completed pilot is reused, not refitted.
        (folds / f"{fold_id}.json").write_text(
            json.dumps(
                {
                    "fold_id": fold_id,
                    "status": "complete",
                    "identity": pipe.run_identity(root, None),
                }
            )
        )
        summary = pipe.stage_pilot(root, workers=1)
        assert summary["folds_reused"] == 1 and summary["fits"] == 0

        # One whose identity moved stops the run instead of overwriting it.
        (folds / f"{fold_id}.json").write_text(
            json.dumps(
                {
                    "fold_id": fold_id,
                    "status": "complete",
                    "identity": {**pipe.run_identity(root, None), "matrix_sha256": "other"},
                }
            )
        )
        try:
            pipe.stage_pilot(root, workers=1)
        except pipe.PipelineError as error:
            assert "immutable" in str(error) and fold_id in str(error)
        else:
            raise AssertionError("the pilot refitted over a mismatched completed fold")


def test_a_new_freeze_refuses_folds_it_cannot_bind_to_this_run():
    """A complete cohort is not enough; it has to be *this* run's cohort.

    Otherwise `--stage select` attaches current inputs, candidates and code to
    predictions generated under different ones, and the main precondition then
    trusts that freshly minted freeze.
    """
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        keys, calendar, history = _development_cohort_fixture(tmp)
        identity = {
            "spec": {"rich_count": 561},
            "matrix_sha256": "m",
            "keys_sha256": "k",
            "calendar_sha256": "c",
            "code_sha256": {"run_pipeline.py": "1"},
        }
        folds = tmp / "development" / "folds"
        for fold_id in calendar["fold_id"]:
            (folds / f"{fold_id}.json").write_text(
                json.dumps(
                    {"fold_id": fold_id, "status": "complete", "identity": identity}
                )
            )
        # Matching identity: the cohort binds.
        bound = pipe.reconcile_development_cohort(
            tmp, keys, calendar, history, identity=identity
        )
        assert bound["identity_enforced"] is True
        assert bound["supported_folds"] == 8

        # One fold produced under different inputs must stop a new freeze,
        # even though the cohort is otherwise complete.
        (folds / "dev_h03_2020-01.json").write_text(
            json.dumps(
                {
                    "fold_id": "dev_h03_2020-01",
                    "status": "complete",
                    "identity": {**identity, "matrix_sha256": "different"},
                }
            )
        )
        try:
            pipe.reconcile_development_cohort(tmp, keys, calendar, history, identity=identity)
        except pipe.PipelineError as error:
            assert "not produced under this run's identity" in str(error)
        else:
            raise AssertionError("a foreign development fold was folded into a new freeze")

        # A read-only replay of historical evidence is still allowed, and says so.
        replayed = pipe.reconcile_development_cohort(tmp, keys, calendar, history, identity=None)
        assert replayed["identity_enforced"] is False


def test_a_replay_freeze_can_never_drive_a_main_schedule():
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw) / "run"
        _synthetic_run(root)
        identity = pipe.run_identity(root)
        replay = {
            "spec": identity["spec"],
            "matrix_sha256": identity["matrix_sha256"],
            "identity": identity,
            "code_sha256": pipe.code_identity(),
            "authoritative": False,
            "kind": "read_only_replay",
        }
        try:
            pipe.check_main_preconditions(root, replay)
        except pipe.PipelineError as error:
            assert "read_only_replay" in str(error)
        else:
            raise AssertionError("a replay freeze was accepted as a commitment")


def test_identity_comparison_separates_code_drift_from_science():
    recorded = {"spec": {"a": 1}, "matrix_sha256": "m", "code_sha256": {"f": "old"}}
    current = {"spec": {"a": 1}, "matrix_sha256": "m", "code_sha256": {"f": "new"}}
    assert pipe.compare_identity(recorded, current) == [
        "code_sha256: recorded {'f': 'old'} != current {'f': 'new'}"
    ]
    assert pipe.compare_identity(recorded, current, scientific_only=True) == []
    moved = {**current, "matrix_sha256": "other"}
    assert pipe.compare_identity(recorded, moved, scientific_only=True)


# --------------------------------------------------------------------------
# End-to-end reporting on a synthetic run
# --------------------------------------------------------------------------


def _synthetic_run(root: Path) -> None:
    """A miniature but structurally complete run directory.

    Small enough to reason about, complete enough that ``generate`` exercises
    the real joins, the cohort audit, the threshold application and the
    bootstrap rather than a stubbed version of them.
    """
    rng = np.random.default_rng(5)
    areas = list(range(1, 21))
    months = {1: ["2023-02", "2024-02", "2025-02"], 3: ["2023-04", "2024-04", "2025-04"],
              6: ["2023-07", "2024-07", "2025-07"], 12: ["2024-01", "2025-01", "2025-02"]}

    rows = []
    for h, month_list in months.items():
        for month in month_list:
            for area in areas:
                # Every fourth area has no persistence at any horizon. Area 1
                # is the harder, real case: its first observation sits between
                # the h=12 and h=1 origins, so the SAME area-month is
                # E_history at short horizons and E_no_history at h=12.
                has_history = 0 if area % 4 == 0 else 1
                if area == 1 and h == 12:
                    has_history = 0
                rows.append(
                    {
                        "admin_code": area,
                        "target_month": month,
                        "horizon_months": h,
                        "target_ord": int(month[:4]) * 12 + int(month[5:7]) - 1,
                        "origin_ord": int(month[:4]) * 12 + int(month[5:7]) - 1 - h,
                        "origin_month": month,
                        "ipcch_food_crisis": int(rng.random() < 0.4),
                        "persistence_b": float(rng.integers(0, 2)) if has_history else np.nan,
                        "has_history": has_history,
                        "country_key": f"C{area % 5}",
                        "cohort": "CH" if area > 15 else "IPC",
                        "q3_target": float(rng.random()),
                    }
                )
    keys = pd.DataFrame(rows)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "folds").mkdir(parents=True, exist_ok=True)
    (root / "main" / "folds").mkdir(parents=True, exist_ok=True)
    keys.to_csv(root / "data" / "keys.csv.gz", index=False)
    np.save(root / "data" / "rich561_X.npy", np.zeros((len(keys), 561)))
    for name in ("feature-schema.json", "candidate-configs.json"):
        (root / "inputs" / name).write_bytes((prep.CONFIG_DIR / name).read_bytes())

    provenance = keys[["admin_code", "target_month", "horizon_months"]].copy()
    provenance["observations_available"] = 3
    provenance["obs1_month"] = "2022-12"
    provenance.to_csv(root / "data" / "history_source_keys.csv.gz", index=False)

    calendar_rows = []
    for h, month_list in months.items():
        for month in month_list:
            block = keys[(keys["horizon_months"] == h) & (keys["target_month"] == month)]
            calendar_rows.append(
                {
                    "fold_id": f"mai_h{h:02d}_{month}",
                    "stage": "main",
                    "horizon_months": h,
                    "target_month": month,
                    "origin_month": month,
                    "target_ord": int(block["target_ord"].iloc[0]),
                    "origin_ord": int(block["origin_ord"].iloc[0]),
                    "test_rows": len(block),
                    "test_rows_with_history": int(block["has_history"].sum()),
                    "test_rows_without_history": int((1 - block["has_history"]).sum()),
                    "full_pool_rows": 100,
                    "matched_pool_rows": 80,
                }
            )
    pd.DataFrame(calendar_rows).to_csv(root / "folds" / "calendar.csv", index=False)

    selections = {
        arm.name: {
            str(h): {"config_id": "R0" if arm.name == "rich_rf" else "X0", "t0": 0.5, "t1": 0.5}
            for h in pipe.HORIZONS
        }
        for arm in pipe.ARMS
    }
    (root / "freeze.json").write_text(
        __import__("json").dumps(
            {
                "primary_family": "correction_xgb",
                "primary_mean_delta": {"correction_xgb": 0.01},
                "selections": selections,
            }
        )
    )

    for record in calendar_rows:
        h, month = record["horizon_months"], record["target_month"]
        block = keys[(keys["horizon_months"] == h) & (keys["target_month"] == month)]
        frames = []
        for arm in pipe.ARMS:
            scored = block if arm.name == "fullpool_xgb" else block[block["has_history"] == 1]
            frames.append(
                pd.DataFrame(
                    {
                        "fold_id": record["fold_id"],
                        "arm": arm.name,
                        "config_id": "R0" if arm.name == "rich_rf" else "X0",
                        "row_index": scored.index.to_numpy(),
                        "support": np.where(
                            scored["has_history"].to_numpy() == 1, "E_history", "E_no_history"
                        ),
                        "raw_score": rng.random(len(scored)),
                        "crisis_score": rng.random(len(scored)),
                        "route": "model",
                    }
                )
            )
        pd.concat(frames, ignore_index=True).to_csv(
            root / "main" / "folds" / f"{record['fold_id']}.csv.gz", index=False
        )


def test_report_runs_end_to_end_on_a_synthetic_run():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "run"
        _synthetic_run(root)
        summary = rep.generate(root, root / "reports")

        assert summary["primary_family"] == "correction_xgb"
        audit = summary["cohort_audit"]
        # 20 areas, 5 history-less at every horizon, plus area 1 which is
        # history-less only at h=12: 3 extra no-history rows there.
        assert audit["e_history_rows"] == 15 * 12 - 3
        assert audit["e_no_history_rows"] == 5 * 12 + 3
        assert audit["e_all_rows"] == audit["e_history_rows"] + audit["e_no_history_rows"]
        for h in rep.HORIZONS:
            assert audit["per_horizon"][str(h)]["common_keys"] == (42 if h == 12 else 45)

        # The primary is a reformulation, so the formulation claim is required.
        assert set(summary["claims"]) == {
            "prediction_gain",
            "formulation_advantage",
            "information_gain",
        }
        assert all(
            entry["result"] in ("supported", "not_supported", "incomplete")
            for entry in summary["claims"].values()
        )
        for name in (
            "metrics_e_history.csv",
            "deltas_e_history.csv",
            "metrics_e_all_combined.csv",
            "correction_flips.csv",
            "share_diagnostics.csv",
            "bootstrap_draws.csv.gz",
            "summary.json",
        ):
            assert (root / "reports" / name).is_file(), name

        replay = rep.replay_check(root, root / "reports", root / "reports_replay")
        assert replay["all_identical"], replay["mismatched"]


def test_report_rejects_a_missing_evaluation_key():
    """Dropping one scheduled row must fail the cohort audit, not be averaged over."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "run"
        _synthetic_run(root)
        path = root / "main" / "folds" / "mai_h01_2023-02.csv.gz"
        frame = pd.read_csv(path)
        drop = frame[(frame["arm"] == "fullpool_xgb")].index[:1]
        frame.drop(index=drop).to_csv(path, index=False)
        try:
            rep.generate(root, root / "reports")
        except rep.ReportError as error:
            assert "cohort audit failed" in str(error), error
            return
    raise AssertionError("a missing evaluation key was not detected")


# --------------------------------------------------------------------------
# Imputation (needs the pinned release)
# --------------------------------------------------------------------------


def test_imputer_fills_from_training_columns_only():
    zip_path = pipe.DEFAULT_RELEASE_ZIP
    if not zip_path.is_file():
        raise SkipTest(f"pinned release not present at {zip_path}")
    import tempfile

    from IPCCHGeoRFExperiment import baseline_runtime as brt

    with tempfile.TemporaryDirectory() as tmp:
        runtime = brt.extract_baseline(zip_path, Path(tmp) / "baseline")
        with brt.baseline_imports(runtime) as (_config, _georf):
            from src.customize.customize import OutOfRangeImputer

            imputer = OutOfRangeImputer(strategy="max_plus", multiplier=100.0)
            train = np.array(
                [
                    [1.0, 0.0, np.nan, -5.0],
                    [2.0, 0.0, np.nan, -7.0],
                    [np.nan, 0.0, np.nan, -9.0],
                ]
            )
            imputer.fit(train)
            out = np.asarray(imputer.transform(train), dtype=np.float64)
            assert np.isfinite(out).all()
            assert out.shape == train.shape, "every column is retained"
            close(out[2, 0], 200.0)  # max 2 * 100
            close(out[0, 1], 0.0)  # a zero max fills with 100, but nothing is missing here
            close(out[0, 2], 0.0)  # an all-missing column fills with 0
            # A negative column's fill is max*100 = -500, which is INSIDE the
            # column's range. The contract keeps this for baseline continuity
            # and requires it to be disclosed rather than silently changed.
            assert out[:, 3].min() <= -5.0

            later = np.array([[np.nan, np.nan, np.nan, np.nan]])
            after = np.asarray(imputer.transform(later), dtype=np.float64)
            close(after[0, 0], 200.0), "transform must reuse the stored training fill"


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def main() -> int:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    skipped = 0
    for test in tests:
        try:
            test()
        except SkipTest as reason:
            skipped += 1
            print(f"skip {test.__name__}: {reason}")
        except Exception:  # noqa: BLE001 - report every failure, keep going
            failures += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures - skipped}/{len(tests)} passed, {skipped} skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
