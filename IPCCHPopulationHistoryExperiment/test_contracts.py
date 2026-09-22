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
    scores, route = pipe.fit_and_score(arm, resolved, X, np.ones(10), np.zeros((3, 4)))
    assert route["route"] == "constant_single_class"
    assert list(scores) == [1.0, 1.0, 1.0]

    share = pipe.ARMS_BY_NAME["share_xgb"]
    resolved = pipe.resolve_config(SPEC.configs, "xgb", {"id": "X0"}, regressor=True)
    scores, route = pipe.fit_and_score(share, resolved, X, np.full(10, 0.37), np.zeros((2, 4)))
    assert route["route"] == "constant_target"
    assert list(scores) == [0.37, 0.37]


def test_constant_correction_target_stays_conditional_on_b():
    """An all-error fitting pool must still flip in both directions."""
    arm = pipe.ARMS_BY_NAME["correction_xgb"]
    resolved = pipe.resolve_config(SPEC.configs, "xgb", {"id": "X0"}, regressor=False)
    raw, route = pipe.fit_and_score(arm, resolved, np.zeros((8, 3)), np.ones(8), np.zeros((2, 3)))
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
    pairs, claims = rep.required_pairs("rich_direct_xgb")
    assert "formulation_advantage" not in claims
    assert ("rich_direct_xgb", "binary_history_xgb") in pairs
    pairs, claims = rep.required_pairs("correction_xgb")
    assert ("correction_xgb", "rich_direct_xgb") in pairs
    assert ("correction_xgb", "fullpool_xgb") in pairs
    assert ("correction_xgb", "persistence") in pairs


def test_class1_metrics_report_an_undefined_denominator():
    result = rep.metrics(0, 0, 0, 10)
    assert math.isnan(result["f1"]) and "f1" in result["undefined"]
    assert math.isnan(result["precision"]) and math.isnan(result["recall"])
    exact = rep.metrics(3, 1, 2, 4)
    close(exact["f1"], 6 / 9)
    close(exact["precision"], 0.75)
    close(exact["recall"], 0.6)


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
