"""Compact runnable contract checks for the IPCCH data boundary.

Small hand-computable inputs only: these assert the R3 temporal contract
(Q6a-Q6f), not the real source. Run with the preferred interpreter::

    python3.12.exe -B IPCCHGeoRFExperiment/test_contracts.py

The gate on the real 42,695/15,206 counts lives in
``prepare_data.check_target_gate`` and is exercised by the runner, not here.
"""

from __future__ import annotations

import math
import sys
import tempfile
import traceback
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_data as pdata  # noqa: E402

FUTURE_SENTINEL = 999999.0


# --------------------------------------------------------------------------
# Tiny fixtures
# --------------------------------------------------------------------------


def make_panel(rows: list[dict]) -> pd.DataFrame:
    """Build a covariate panel from ``{admin_code, year, month, <raw>...}``.

    Unspecified raw columns are NaN, which is what the real source looks like
    for a sparse field, and keeps each check focused on one column.
    """
    frame = pd.DataFrame(rows)
    for name in pdata.RAW_FEATURE_COLUMNS:
        if name not in frame.columns:
            frame[name] = np.nan
        frame[name] = frame[name].astype(np.float64)
    frame["admin_code"] = frame["admin_code"].astype(np.int64)
    frame["month_ord"] = pdata.month_ordinal(frame["year"], frame["month"])
    return frame.sort_values(["admin_code", "month_ord"]).reset_index(drop=True)


def make_labels(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["admin_code"] = frame["admin_code"].astype(np.int64)
    frame["ipcch_food_crisis"] = frame["ipcch_food_crisis"].astype(np.int64)
    for name in ("country_en", "ISO3"):
        if name not in frame.columns:
            frame[name] = "X"
    return frame


def months(area: int, start: tuple[int, int], count: int, **values) -> list[dict]:
    """``count`` consecutive months from ``start``; each value is a list."""
    year, month = start
    rows = []
    for offset in range(count):
        total = pdata.month_ordinal(year, month) + offset
        row = {"admin_code": area, "year": int(total // 12), "month": int(total % 12 + 1)}
        for name, series in values.items():
            row[name] = series[offset]
        rows.append(row)
    return rows


def one_row(matrix: pdata.FeatureMatrix, horizon: int) -> pd.Series:
    frame = matrix.frame
    selected = frame[frame["horizon_months"] == float(horizon)]
    assert len(selected) == 1, f"expected exactly one h{horizon} row, got {len(selected)}"
    return selected.iloc[0]


# --------------------------------------------------------------------------
# R1 — target, provenance and exact boundaries
#
# The gate on the real counts lives in the runner; these use a hand-written
# stand-in CSV carrying only the columns R1 reads, with every value written as
# a string so the source decimals survive the round trip.
# --------------------------------------------------------------------------


def target_row(area: int, month: int, phases: list[str], population: str = "100") -> dict:
    row = {
        "admin_code": str(area),
        "year": "2020",
        "month": str(month),
        "estimated_population": population,
        "overall_phase": "3",
        "country_en": "Alpha",
        "ISO3": "AAA",
    }
    for name, value in zip(pdata.PHASE_COLUMNS, phases):
        row[name] = value
    return row


def build_target_fixture(tmp: Path, rows: list[dict]) -> pd.DataFrame:
    path = tmp / "target_fixture.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    ledger = pdata.build_target_ledger(path, verify_hash=False)
    return ledger.frame


def test_r1_preserves_the_five_normalized_components():
    rows = [
        # S = 1.00 exactly: the normalization is exact, so each component must
        # come back byte-for-byte equal to its raw share.
        target_row(1, 1, ["0.5", "0.2", "0.2", "0.05", "0.05"]),
        # S = 1.10, the inclusive upper bound: every quotient repeats forever
        # and is therefore kept at the declared provenance precision.
        target_row(1, 2, ["0.50", "0.30", "0.20", "0.05", "0.05"]),
        # S = 0.50: rejected at step 3, so step 4 is never reached.
        target_row(1, 3, ["0.1", "0.1", "0.1", "0.1", "0.1"]),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        frame = build_target_fixture(Path(tmp), rows)

    assert len(pdata.NORMALIZED_PHASE_COLUMNS) == 5
    for column in pdata.NORMALIZED_PHASE_COLUMNS:
        assert column in frame.columns, "R1 requires the normalized components"

    exact, repeating, invalid = (frame.iloc[i] for i in range(3))

    # Exact case: S = 1, so normalized == raw, to the digit.
    assert exact["target_valid"] == 1 and exact["phase_sum_S_str"] == "1.00"
    for raw_column, norm_column in zip(pdata.PHASE_COLUMNS, pdata.NORMALIZED_PHASE_COLUMNS):
        assert Decimal(exact[norm_column]) == Decimal(exact[f"raw_{raw_column}"])

    # Repeating case: each component matches the true ratio to the declared
    # precision, checked against exact rational arithmetic, not against the
    # same Decimal helper that produced it.
    assert repeating["target_valid"] == 1
    total = Fraction(Decimal(repeating["phase_sum_S_str"]))
    tolerance = Fraction(1, 10 ** (pdata.NORMALIZATION_PRECISION - 2))
    components = []
    for raw_column, norm_column in zip(pdata.PHASE_COLUMNS, pdata.NORMALIZED_PHASE_COLUMNS):
        value = Decimal(repeating[norm_column])
        components.append(value)
        expected = Fraction(Decimal(repeating[f"raw_{raw_column}"])) / total
        assert abs(Fraction(value) - expected) < tolerance

    # The five normalized components are S/S = 1 within that same precision.
    # Summed as exact rationals, so the check is not itself rounded.
    assert abs(sum((Fraction(value) for value in components), Fraction(0)) - 1) < tolerance

    # Invalid row: missing, not zero-filled and not silently normalized.
    assert invalid["target_valid"] == 0
    assert invalid["target_invalid_reason"] == "sum_out_of_bounds"
    assert invalid["normalized_p3plus_str"] == ""
    for column in pdata.NORMALIZED_PHASE_COLUMNS:
        assert invalid[column] == ""


def test_r1_label_is_exact_beyond_the_default_decimal_context():
    # 29 significant digits: one more than the default decimal context carries.
    sharp = "0.20000000000000000000000000001"
    rows = [
        target_row(1, 1, ["0.8", "0", sharp, "0", "0"]),
        # The plain boundary: P3+ / S is exactly .20, which stays negative.
        target_row(1, 2, ["0.6", "0.2", "0.2", "0", "0"]),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        frame = build_target_fixture(Path(tmp), rows)

    above, at_threshold = frame.iloc[0], frame.iloc[1]
    assert above["target_valid"] == 1 and at_threshold["target_valid"] == 1
    assert above["ipcch_food_crisis"] == 1, "exact arithmetic puts this above .20"
    assert at_threshold["ipcch_food_crisis"] == 0, "exact equality with .20 is negative"

    # Document the defect this guards: under the default 28-digit context the
    # same row rounds to exactly .20 and would have been labelled 0.
    with localcontext() as ctx:
        ctx.prec = 28
        phases = [Decimal(t) for t in ("0.8", "0", sharp, "0", "0")]
        rounded = (phases[2] + phases[3] + phases[4]) / sum(phases, Decimal(0))
    assert rounded == pdata.CRISIS_THRESHOLD

    # And the exact route does not depend on that quotient at all: the label
    # comes from 5*(P3+P4+P5) > S, which is decided by integer-scaled digits.
    verdict = pdata._classify_row(
        [Decimal(t) for t in ("0.8", "0", sharp, "0", "0")], Decimal(100)
    )
    assert verdict.crisis == 1
    assert Fraction(Decimal(sharp)) > Fraction(1, 5) * Fraction(
        Decimal(verdict.total)
    ), "exact rational arithmetic agrees"


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


def test_schema_is_93_columns_in_a_stable_order():
    assert len(pdata.FEATURE_COLUMNS) == 93
    assert len(set(pdata.FEATURE_COLUMNS)) == 93
    blocks = (
        pdata.RAW_FEATURE_COLUMNS
        + pdata.DERIVED_FEATURE_COLUMNS
        + pdata.CALENDAR_FEATURE_COLUMNS
        + pdata.HISTORY_FEATURE_COLUMNS
        + pdata.RECENCY_FEATURE_COLUMNS
        + pdata.HORIZON_FEATURE_COLUMNS
    )
    assert tuple(pdata.FEATURE_COLUMNS) == blocks
    assert (len(pdata.RAW_FEATURE_COLUMNS), len(pdata.DERIVED_FEATURE_COLUMNS)) == (70, 15)
    # The first and last names pin the order against an accidental reshuffle.
    assert pdata.FEATURE_COLUMNS[0] == "distance_to_nearest_acled"
    assert pdata.FEATURE_COLUMNS[69] == "lon"
    assert pdata.FEATURE_COLUMNS[70] == "WFP_Price_sum4_asof"
    assert pdata.FEATURE_COLUMNS[-1] == "horizon_months"

    panel = make_panel(months(1, (2020, 1), 24, WFP_Price=[1.0] * 24))
    labels = make_labels([{"admin_code": 1, "year": 2021, "month": 12, "ipcch_food_crisis": 1}])
    matrix = pdata.assemble_feature_matrix(labels, panel)
    assert list(matrix.frame.columns) == list(pdata.METADATA_COLUMNS) + list(
        pdata.FEATURE_COLUMNS
    )
    assert list(matrix.X().columns) == list(pdata.FEATURE_COLUMNS)
    assert len(matrix.frame) == 4  # one outcome x four horizons


def test_row_count_is_outcomes_times_horizons():
    panel = make_panel(months(1, (2020, 1), 36) + months(2, (2020, 1), 36))
    labels = make_labels(
        [
            {"admin_code": 1, "year": 2022, "month": 1, "ipcch_food_crisis": 0},
            {"admin_code": 1, "year": 2022, "month": 6, "ipcch_food_crisis": 1},
            {"admin_code": 2, "year": 2022, "month": 6, "ipcch_food_crisis": 0},
        ]
    )
    matrix = pdata.assemble_feature_matrix(labels, panel)
    assert len(matrix.frame) == 3 * 4
    assert matrix.audit["valid_outcomes"] == 3
    assert matrix.audit["feature_rows"] == 12


# --------------------------------------------------------------------------
# Q6d — window sums
# --------------------------------------------------------------------------


def test_window_sum_is_nan_on_any_missing_month_and_exact_when_complete():
    # 2021-01..2021-12 present. h1 target 2021-12 => O = 2021-11.
    # sum4 window is 2021-08..2021-11 = 8+9+10+11 = 38.
    # sum12 window is 2020-12..2021-11; 2020-12 is absent => NaN.
    panel = make_panel(
        months(1, (2021, 1), 12, WFP_Price=[float(m) for m in range(1, 13)])
    )
    labels = make_labels([{"admin_code": 1, "year": 2021, "month": 12, "ipcch_food_crisis": 0}])
    matrix = pdata.assemble_feature_matrix(labels, panel, horizons=(1,))
    row = one_row(matrix, 1)
    assert row["origin_month"] == "2021-11"
    assert row["WFP_Price_sum4_asof"] == 38.0
    assert math.isnan(row["WFP_Price_sum12_asof"]), "incomplete window must stay NaN"

    # Same months, but 2021-09 has a missing value rather than a missing row:
    # the sum must still be NaN, never treated as zero.
    values = [float(m) for m in range(1, 13)]
    values[8] = np.nan
    holed = make_panel(months(1, (2021, 1), 12, WFP_Price=values))
    holed_row = one_row(pdata.assemble_feature_matrix(labels, holed, horizons=(1,)), 1)
    assert math.isnan(holed_row["WFP_Price_sum4_asof"])

    # A full 12-month window computed exactly: O = 2021-12 via h1 on 2022-01.
    full = make_panel(
        months(1, (2021, 1), 12, nightlight_mean=[2.0] * 12)
    )
    full_labels = make_labels(
        [{"admin_code": 1, "year": 2022, "month": 1, "ipcch_food_crisis": 0}]
    )
    full_row = one_row(pdata.assemble_feature_matrix(full_labels, full, horizons=(1,)), 1)
    assert full_row["origin_month"] == "2021-12"
    assert full_row["nightlight_mean_sum12_asof"] == 24.0


def test_window_sums_never_cross_areas():
    # Area 1 lacks 2021-01; area 2 has it. A cross-area read would produce a
    # number instead of NaN.
    rows = months(1, (2021, 2), 11, WFP_Price=[1.0] * 11)
    rows += months(2, (2021, 1), 12, WFP_Price=[100.0] * 12)
    panel = make_panel(rows)
    labels = make_labels([{"admin_code": 1, "year": 2022, "month": 1, "ipcch_food_crisis": 0}])
    matrix = pdata.assemble_feature_matrix(labels, panel, horizons=(1,))
    row = one_row(matrix, 1)
    assert math.isnan(row["WFP_Price_sum12_asof"])
    assert row["WFP_Price_sum4_asof"] == 4.0
    # Scaffold completeness is measured, not assumed: area 1 has a real gap.
    assert matrix.audit["panel_grid_gaps"] == 1
    assert matrix.audit["panel_areas_with_month_gaps"] == 1


# --------------------------------------------------------------------------
# Q6d — EVI lags
# --------------------------------------------------------------------------


def test_evi_lag_reads_exactly_o_minus_k_and_a_gap_hits_only_that_lag():
    # O = 2021-12 (h1 on 2022-01); EVI value encodes its own month ordinal so
    # an off-by-one is visible.
    rows = []
    for offset in range(24):
        total = pdata.month_ordinal(2020, 1) + offset
        if total == pdata.month_ordinal(2021, 9):
            continue  # gap: the 2021-09 row is absent entirely
        rows.append(
            {
                "admin_code": 1,
                "year": int(total // 12),
                "month": int(total % 12 + 1),
                "EVI_mean": float(total),
            }
        )
    panel = make_panel(rows)
    labels = make_labels([{"admin_code": 1, "year": 2022, "month": 1, "ipcch_food_crisis": 0}])
    row = one_row(pdata.assemble_feature_matrix(labels, panel, horizons=(1,)), 1)

    origin = pdata.month_ordinal(2021, 12)
    gap_lag = origin - pdata.month_ordinal(2021, 9)
    assert gap_lag == 3
    for k in range(1, 13):
        value = row[f"EVI_mean_lag{k}_asof"]
        if k == gap_lag:
            assert math.isnan(value), "a missing source month must NaN only its own lag"
        else:
            assert value == float(origin - k), f"lag{k} read the wrong calendar month"
    assert row["EVI_mean"] == float(origin), "raw EVI_mean must be the value at O"


# --------------------------------------------------------------------------
# No future information
# --------------------------------------------------------------------------


def test_no_value_after_the_origin_is_ever_read():
    # Every month carries 1.0 except O+1 .. T, which carry a sentinel.
    origin = pdata.month_ordinal(2021, 6)
    rows = []
    for offset in range(-24, 13):
        total = origin + offset
        value = 1.0 if offset <= 0 else FUTURE_SENTINEL
        rows.append(
            {
                "admin_code": 1,
                "year": int(total // 12),
                "month": int(total % 12 + 1),
                "EVI_mean": value,
                "WFP_Price": value,
                "nightlight_mean": value,
                "lat": value,
            }
        )
    panel = make_panel(rows)
    labels = make_labels([{"admin_code": 1, "year": 2021, "month": 7, "ipcch_food_crisis": 0}])
    matrix = pdata.assemble_feature_matrix(labels, panel, horizons=(1,))
    row = one_row(matrix, 1)
    assert row["origin_month"] == "2021-06"
    values = row[list(pdata.FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    assert not np.any(values == FUTURE_SENTINEL), "a post-origin value leaked"
    # Positive control: the window sums are real numbers, so the absence of the
    # sentinel is not merely an all-NaN row.
    assert row["WFP_Price_sum12_asof"] == 12.0
    assert row["lat"] == 1.0


def test_label_after_the_origin_is_not_used_as_history():
    # A valid label at 2021-07 sits strictly after O = 2021-06 and must be
    # invisible to the h1 view of the 2021-08 target.
    panel = make_panel(months(1, (2020, 1), 36))
    labels = make_labels(
        [
            {"admin_code": 1, "year": 2021, "month": 3, "ipcch_food_crisis": 0},
            {"admin_code": 1, "year": 2021, "month": 7, "ipcch_food_crisis": 1},
            {"admin_code": 1, "year": 2021, "month": 8, "ipcch_food_crisis": 0},
        ]
    )
    matrix = pdata.assemble_feature_matrix(labels, panel, horizons=(2,))
    row = matrix.frame[matrix.frame["target_month"] == "2021-08"].iloc[0]
    assert row["origin_month"] == "2021-06"
    assert row["last_observed_label_month"] == "2021-03"
    assert row["last_observed_label"] == 0.0
    assert row["no_prior_observed_crisis"] == 1.0, "the 2021-07 positive is future"


# --------------------------------------------------------------------------
# Q6a / Q6f — history and recency
# --------------------------------------------------------------------------


def test_history_at_the_origin_is_inclusive():
    panel = make_panel(months(1, (2020, 1), 36))
    labels = make_labels(
        [
            {"admin_code": 1, "year": 2021, "month": 6, "ipcch_food_crisis": 1},
            {"admin_code": 1, "year": 2021, "month": 9, "ipcch_food_crisis": 0},
        ]
    )
    # h3 on 2021-09 gives O = 2021-06, exactly the other label's month.
    matrix = pdata.assemble_feature_matrix(labels, panel, horizons=(3,))
    row = matrix.frame[matrix.frame["target_month"] == "2021-09"].iloc[0]
    assert row["origin_month"] == "2021-06"
    assert row["last_observed_label"] == 1.0, "a label at exactly O must count"
    assert row["last_observed_label_age_months"] == 0.0
    assert row["no_observed_label_history"] == 0.0
    assert row["months_since_last_observed_crisis"] == 0.0
    assert row["last_observed_crisis_month"] == "2021-06"


def test_recency_counts_from_the_latest_positive_and_a_negative_does_not_reset():
    panel = make_panel(months(1, (2020, 1), 48))
    labels = make_labels(
        [
            {"admin_code": 1, "year": 2021, "month": 1, "ipcch_food_crisis": 1},
            {"admin_code": 1, "year": 2021, "month": 5, "ipcch_food_crisis": 0},
            {"admin_code": 1, "year": 2021, "month": 11, "ipcch_food_crisis": 0},
        ]
    )
    # O = 2021-08 (h3 on 2021-11): latest label is the 2021-05 negative, latest
    # positive is still 2021-01, seven months earlier.
    matrix = pdata.assemble_feature_matrix(labels, panel, horizons=(3,))
    row = matrix.frame[matrix.frame["target_month"] == "2021-11"].iloc[0]
    assert row["origin_month"] == "2021-08"
    assert row["last_observed_label"] == 0.0
    assert row["last_observed_label_age_months"] == 3.0
    assert row["months_since_last_observed_crisis"] == 7.0
    assert row["no_prior_observed_crisis"] == 0.0


def test_recency_search_is_not_truncated_at_36_months():
    panel = make_panel(months(1, (2015, 1), 120))
    labels = make_labels(
        [
            {"admin_code": 1, "year": 2016, "month": 1, "ipcch_food_crisis": 1},
            {"admin_code": 1, "year": 2022, "month": 1, "ipcch_food_crisis": 0},
        ]
    )
    matrix = pdata.assemble_feature_matrix(labels, panel, horizons=(1,))
    row = matrix.frame[matrix.frame["target_month"] == "2022-01"].iloc[0]
    assert row["origin_month"] == "2021-12"
    assert row["months_since_last_observed_crisis"] == 71.0
    assert row["no_prior_observed_crisis"] == 0.0


def test_no_label_history_and_no_prior_crisis_are_distinguished():
    panel = make_panel(months(1, (2020, 1), 36) + months(2, (2020, 1), 36))
    labels = make_labels(
        [
            # Area 1: negative history only.
            {"admin_code": 1, "year": 2021, "month": 1, "ipcch_food_crisis": 0},
            {"admin_code": 1, "year": 2021, "month": 6, "ipcch_food_crisis": 0},
            # Area 2: its first ever label is the target itself.
            {"admin_code": 2, "year": 2021, "month": 6, "ipcch_food_crisis": 1},
        ]
    )
    matrix = pdata.assemble_feature_matrix(labels, panel, horizons=(1,))
    frame = matrix.frame

    negative_only = frame[
        (frame["admin_code"] == 1) & (frame["target_month"] == "2021-06")
    ].iloc[0]
    assert negative_only["no_observed_label_history"] == 0.0
    assert negative_only["no_prior_observed_crisis"] == 1.0
    assert negative_only["last_observed_label"] == 0.0
    assert math.isnan(negative_only["months_since_last_observed_crisis"])
    assert negative_only["last_observed_crisis_month"] == ""

    no_history = frame[
        (frame["admin_code"] == 2) & (frame["target_month"] == "2021-06")
    ].iloc[0]
    assert no_history["no_observed_label_history"] == 1.0
    assert no_history["no_prior_observed_crisis"] == 1.0
    assert math.isnan(no_history["last_observed_label"])
    assert math.isnan(no_history["last_observed_label_age_months"])
    assert no_history["last_observed_label_month"] == ""


def test_history_never_crosses_areas():
    panel = make_panel(months(1, (2020, 1), 36) + months(2, (2020, 1), 36))
    labels = make_labels(
        [
            {"admin_code": 1, "year": 2021, "month": 1, "ipcch_food_crisis": 1},
            {"admin_code": 2, "year": 2021, "month": 6, "ipcch_food_crisis": 0},
        ]
    )
    matrix = pdata.assemble_feature_matrix(labels, panel, horizons=(1,))
    row = matrix.frame[matrix.frame["admin_code"] == 2].iloc[0]
    assert row["no_observed_label_history"] == 1.0
    assert row["no_prior_observed_crisis"] == 1.0


# --------------------------------------------------------------------------
# Horizon views
# --------------------------------------------------------------------------


def test_four_horizon_views_differ_only_by_horizon_and_origin_dependent_values():
    # Constant covariates and no label history make every origin-dependent
    # value identical, isolating horizon_months as the only difference.
    panel = make_panel(
        months(
            1,
            (2018, 1),
            72,
            WFP_Price=[3.0] * 72,
            nightlight_mean=[5.0] * 72,
            EVI_mean=[7.0] * 72,
            lat=[11.0] * 72,
        )
    )
    labels = make_labels([{"admin_code": 1, "year": 2022, "month": 5, "ipcch_food_crisis": 1}])
    matrix = pdata.assemble_feature_matrix(labels, panel)
    frame = matrix.frame
    assert len(frame) == 4
    assert list(frame["horizon_months"]) == [1.0, 3.0, 6.0, 12.0]
    assert list(frame["origin_month"]) == ["2022-04", "2022-02", "2021-11", "2021-05"]
    assert frame["target_month"].nunique() == 1
    assert frame["ipcch_food_crisis"].nunique() == 1

    shared = [name for name in pdata.FEATURE_COLUMNS if name != "horizon_months"]
    block = frame[shared].to_numpy(dtype=np.float64)
    same = np.all((block == block[0]) | (np.isnan(block) & np.isnan(block[0])), axis=1)
    assert same.all(), "constant inputs must make the four views identical"
    assert frame["WFP_Price_sum4_asof"].unique().tolist() == [12.0]
    assert frame["target_month_sin"].nunique() == 1

    # With a horizon-straddling label the views must then actually differ.
    labels2 = make_labels(
        [
            {"admin_code": 1, "year": 2022, "month": 3, "ipcch_food_crisis": 1},
            {"admin_code": 1, "year": 2022, "month": 5, "ipcch_food_crisis": 1},
        ]
    )
    second = pdata.assemble_feature_matrix(labels2, panel)
    views = second.frame[second.frame["target_month"] == "2022-05"]
    assert views.loc[views["horizon_months"] == 1.0, "no_prior_observed_crisis"].iloc[0] == 0.0
    assert views.loc[views["horizon_months"] == 12.0, "no_prior_observed_crisis"].iloc[0] == 1.0


def test_calendar_encoding_matches_the_target_month():
    panel = make_panel(months(1, (2020, 1), 48))
    labels = make_labels(
        [
            {"admin_code": 1, "year": 2022, "month": 1, "ipcch_food_crisis": 0},
            {"admin_code": 1, "year": 2022, "month": 4, "ipcch_food_crisis": 0},
        ]
    )
    frame = pdata.assemble_feature_matrix(labels, panel, horizons=(1,)).frame
    january = frame[frame["target_month"] == "2022-01"].iloc[0]
    april = frame[frame["target_month"] == "2022-04"].iloc[0]
    assert abs(january["target_month_sin"] - 0.0) < 1e-12
    assert abs(january["target_month_cos"] - 1.0) < 1e-12
    assert abs(april["target_month_sin"] - math.sin(2 * math.pi * 3 / 12)) < 1e-12
    # The encoding follows T, not O: both rows share h1 but differ here.
    assert january["target_month_cos"] != april["target_month_cos"]


# --------------------------------------------------------------------------
# Missing panel coverage and infinities
# --------------------------------------------------------------------------


def test_missing_origin_row_yields_nan_without_dropping_the_outcome():
    panel = make_panel(months(1, (2022, 1), 6, WFP_Price=[1.0] * 6))
    labels = make_labels([{"admin_code": 1, "year": 2022, "month": 2, "ipcch_food_crisis": 1}])
    matrix = pdata.assemble_feature_matrix(labels, panel)
    row = one_row(matrix, 12)  # O = 2021-02, before the panel starts
    assert row["origin_month"] == "2021-02"
    assert math.isnan(row["WFP_Price"])
    assert math.isnan(row["WFP_Price_sum4_asof"])
    assert row["ipcch_food_crisis"] == 1
    # Panel is 2022-01..2022-06, so h3/h6/h12 all land before it starts.
    assert matrix.audit["rows_with_missing_origin_row"] == 3
    assert not one_row(matrix, 1)[list(pdata.RAW_FEATURE_COLUMNS)].isna().all()


def test_infinities_become_nan_with_an_audit():
    panel = make_panel(months(1, (2021, 1), 24, WFP_Price=[1.0] * 24))
    panel.loc[panel["month_ord"] == pdata.month_ordinal(2021, 12), "WFP_Price"] = np.inf
    labels = make_labels([{"admin_code": 1, "year": 2022, "month": 1, "ipcch_food_crisis": 0}])
    matrix = pdata.assemble_feature_matrix(labels, panel, horizons=(1,))
    row = one_row(matrix, 1)
    assert math.isinf(panel["WFP_Price"].max()), "fixture sanity"
    assert not np.isinf(matrix.X().to_numpy(dtype=np.float64)).any()
    assert math.isnan(row["WFP_Price"]), "an infinity at O must become NaN"
    # The infinite month is removed before the cumulative sum, so its window is
    # incomplete rather than 1.8e308 (np.nan_to_num's default substitution).
    assert math.isnan(row["WFP_Price_sum4_asof"]), "a poisoned window must be NaN"
    assert matrix.infinity_audit["panel_input"].get("WFP_Price") == 1
    assert matrix.infinity_audit["assembled"] == {}, "sums must not re-create one"


def test_duplicate_or_non_binary_inputs_are_refused():
    panel = make_panel(months(1, (2021, 1), 12))
    duplicate = make_labels(
        [
            {"admin_code": 1, "year": 2021, "month": 6, "ipcch_food_crisis": 1},
            {"admin_code": 1, "year": 2021, "month": 6, "ipcch_food_crisis": 0},
        ]
    )
    try:
        pdata.assemble_feature_matrix(duplicate, panel)
    except pdata.DataContractError:
        pass
    else:
        raise AssertionError("duplicate (admin_code, month) must stop the run")

    bad = make_labels([{"admin_code": 1, "year": 2021, "month": 6, "ipcch_food_crisis": 1}])
    bad["ipcch_food_crisis"] = 2
    try:
        pdata.assemble_feature_matrix(bad, panel)
    except pdata.DataContractError:
        pass
    else:
        raise AssertionError("a non-binary target must stop the run")


def test_month_ordinal_round_trips():
    for year, month in ((2010, 1), (2021, 12), (2026, 7)):
        ordinal = int(pdata.month_ordinal([year], [month])[0])
        assert pdata.month_label([ordinal])[0] == f"{year:04d}-{month:02d}"
    assert pdata.month_label([-1])[0] == ""
    assert (
        int(pdata.month_ordinal([2022], [1])[0]) - int(pdata.month_ordinal([2021], [1])[0])
    ) == 12


# ==========================================================================
# Geography (R2 / Q8 / Q8r / Q8g)
#
# The geometric libraries live only in the Windows runtime, so these checks
# skip cleanly under a bare WSL interpreter rather than breaking the 18
# data-boundary checks above.
# ==========================================================================


class SkipTest(Exception):
    """Raised when an optional geometric dependency is unavailable."""


def require_shapely():
    try:
        import shapely  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SkipTest(f"shapely unavailable: {error}") from error
    return shapely


def require_geopandas():
    require_shapely()
    try:
        import geopandas  # noqa: PLC0415
        import pyproj  # noqa: PLC0415,F401
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SkipTest(f"geopandas/pyproj unavailable: {error}") from error
    return geopandas


def square(x0: float, y0: float, size: float = 1.0):
    from shapely.geometry import Polygon  # noqa: PLC0415

    return Polygon(
        [(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)]
    )


def bowtie():
    """A self-intersecting quadrilateral: the classic invalid polygon."""
    from shapely.geometry import Polygon  # noqa: PLC0415

    return Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])


class _StubPart:
    """Minimal stand-in used to exercise the measured zero-area assertion.

    ``authorized_areal_component`` reads only ``geom_type`` and ``area`` from a
    part, so a stub can present a linear type with a non-zero area -- a state
    real Shapely types cannot reach, and exactly the state the guard must
    refuse rather than trust the type name.
    """

    def __init__(self, geom_type: str, area: float):
        self.geom_type = geom_type
        self.area = area


class _StubCollection:
    def __init__(self, parts):
        self.geom_type = "GeometryCollection"
        self.geoms = parts


def write_geography_fixture(root: Path) -> dict:
    """A four-area source tree: two adjacent, one corner-only, one invalid.

    Layout in degrees::

        area 2 (1,1)-(2,2)      corner-only contact with area 0
        area 0 (0,0)-(1,1)      shares the x=1 edge with area 1
        area 1 (1,0)-(2,1)
        area 3                  a bow-tie at (5,5): invalid, isolated

    Reference coordinates are deliberately NOT the centroids, so a test can
    tell the donor-distance authority from the refinement centroids.
    """
    geopandas = require_geopandas()
    spatial = root / "spatial"
    spatial.mkdir(parents=True, exist_ok=True)

    geometries = [square(0, 0), square(1, 0), square(1, 1), bowtie_at(5, 5)]
    frame = geopandas.GeoDataFrame(
        {
            # Stored as strings, like the real DBF, so ID normalization is real.
            "admin_code": ["0", "1", "2", "3"],
            "lat": [0.5, 0.5, 1.5, 5.5],
            "lon": [0.5, 1.5, 1.5, 5.5],
            "geometry": geometries,
        },
        crs="EPSG:4326",
    )
    geometry_path = spatial / "ipcch_admin_geometry.shp"
    frame.to_file(geometry_path)

    pd.DataFrame(
        {
            "area_id": [0, 1, 2, 3],
            "lat": [0.11, 0.12, 1.13, 5.14],  # distinct from every centroid
            "lon": [0.21, 1.22, 1.23, 5.24],
        }
    ).to_csv(spatial / "unique_area_id_lat_lon.csv", index=False)

    pd.DataFrame(
        {
            "area_id": [0, 1, 2, 3],
            "iso3": ["AAA", "", "BBB", "BBB"],  # area 1 has no ISO3 and is kept
            "country": ["Alpha", "Alpha", "Beta", "Beta"],
            "country_code": ["AA", "AA", "", "BB"],
            "country_en": ["Alpha", "Alpha", "Beta", "Beta"],
        }
    ).to_csv(root / "country_area_id_lookup.csv", index=False)

    return {"geometry": geometry_path, "root": root}


def bowtie_at(x: float, y: float):
    from shapely.affinity import translate  # noqa: PLC0415

    return translate(bowtie(), xoff=x, yoff=y)


def prepared_fixture(tmp: Path) -> pdata.GeographyArtifacts:
    write_geography_fixture(tmp / "source")
    return pdata.prepare_geography(
        tmp / "source",
        tmp / "geography",
        expected_universe=4,
    )


# --------------------------------------------------------------------------
# Keys and identity
# --------------------------------------------------------------------------


def test_geography_area_ids_normalize_to_canonical_integers():
    ids = pdata.normalize_area_ids(["0", " 100341 ", 1425, "101324.0", np.int64(7)])
    assert ids.dtype == np.int64
    assert ids.tolist() == [0, 100341, 1425, 101324, 7]

    for bad in ([""], ["abc"], ["12.5"], [None], [np.nan]):
        try:
            pdata.normalize_area_ids(bad)
        except pdata.DataContractError:
            continue
        raise AssertionError(f"{bad!r} must be refused, not truncated")


def test_geography_missing_or_duplicate_mappings_are_refused():
    full = [0, 1, 2, 3]
    assert pdata.reconcile_area_universe(full, full, full, expected=4)["universe"] == 4

    # A missing mapping in any one source stops the run.
    for missing_in in range(3):
        sources = [list(full), list(full), list(full)]
        sources[missing_in] = full[:-1]
        try:
            pdata.reconcile_area_universe(*sources, expected=4)
        except pdata.DataContractError as error:
            assert "missing" in str(error)
        else:
            raise AssertionError("a missing area mapping must stop the run")

    # A universe of the wrong size stops the run even when all three agree.
    try:
        pdata.reconcile_area_universe(full, full, full, expected=5)
    except pdata.DataContractError:
        pass
    else:
        raise AssertionError("an unexpected universe size must stop the run")


def test_geography_duplicate_area_id_in_the_shapefile_is_refused():
    geopandas = require_geopandas()
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dup.shp"
        geopandas.GeoDataFrame(
            {"admin_code": ["7", "7"], "geometry": [square(0, 0), square(3, 3)]},
            crs="EPSG:4326",
        ).to_file(path)
        try:
            pdata.load_area_geometry(path)
        except pdata.DataContractError as error:
            assert "more than one feature" in str(error)
        else:
            raise AssertionError("duplicate area ids must stop the run")


def test_geography_missing_iso3_areas_are_preserved():
    require_geopandas()
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "source"
        write_geography_fixture(root)
        lookup, audit = pdata.load_country_lookup(root / "country_area_id_lookup.csv")
        assert len(lookup) == 4, "no area may be dropped for a blank ISO3 (R2)"
        assert audit["areas_missing_iso3"] == 1
        assert audit["countries_missing_iso3"] == ["Alpha"]
        assert audit["areas_missing_country_code"] == 1
        # The reporting key still resolves for every area.
        assert (lookup["country_key"] != "").all()


# --------------------------------------------------------------------------
# Q8g repair, including the 2026-09-20 authorized extension
# --------------------------------------------------------------------------


def test_geography_valid_geometry_passes_through_unchanged():
    require_shapely()
    geometry = square(0, 0)
    repaired, record = pdata.repair_geometry(geometry)
    assert repaired is geometry, "a valid geometry must not be rebuilt"
    assert record["outcome"] == "unchanged_valid"
    assert record["geometry_changed"] == 0
    assert record["original_wkb_sha256"] == record["repaired_wkb_sha256"]
    assert record["footprint_comparison"] == "exact_unchanged"
    assert record["area_relative_change"] == 0.0
    # The declared measure is geodesic square metres, not square degrees.
    assert record["repaired_area_m2"] > 1.2e10


def test_geography_valid_non_polygon_input_is_refused():
    require_shapely()
    from shapely.geometry import LineString, Point  # noqa: PLC0415

    # Validity is not the requirement: R2 asks for one Polygon/MultiPolygon per
    # area. A *valid* point or line must not slip past the repair path, which
    # is what happened while the type check ran only on repaired geometries.
    for geometry in (Point(0, 0), LineString([(0, 0), (1, 1)])):
        assert geometry.is_valid and not geometry.is_empty, "fixture sanity"
        try:
            pdata.repair_geometry(geometry)
        except pdata.GeometryRepairError as error:
            assert "non-polygonal" in str(error)
            assert geometry.geom_type in str(error)
        else:
            raise AssertionError(
                f"a valid {geometry.geom_type} must stop the run, not pass through"
            )

    # The polygonal path is untouched: same object, same outcome as before.
    polygon = square(0, 0)
    repaired, record = pdata.repair_geometry(polygon)
    assert repaired is polygon
    assert record["outcome"] == "unchanged_valid"


def test_geography_invalid_geometry_is_repaired():
    require_shapely()
    geometry = bowtie()
    assert not geometry.is_valid, "fixture sanity"
    repaired, record = pdata.repair_geometry(geometry)
    assert repaired.is_valid and not repaired.is_empty
    assert repaired.geom_type in pdata.AREAL_GEOMETRY_TYPES
    assert record["original_valid"] == 0
    assert record["original_validity_reason"].startswith("Self-intersection")
    assert record["geometry_changed"] == 1
    assert record["original_wkb_sha256"] != record["repaired_wkb_sha256"]
    # An invalid original's footprint is diagnostic only, and is labelled so.
    assert record["footprint_comparison"] in {
        "diagnostic_invalid_original",
        "unavailable",
    }


def test_geography_collection_of_one_polygon_plus_zero_area_line_is_extracted():
    require_shapely()
    from shapely.geometry import GeometryCollection, LineString, MultiLineString  # noqa: PLC0415

    polygon = square(0, 0)
    line = LineString([(2, 2), (3, 3)])
    assert line.area == 0.0, "fixture sanity"

    for linear in (line, MultiLineString([[(2, 2), (3, 3)], [(4, 4), (5, 5)]])):
        collection = GeometryCollection([linear, polygon])
        areal, discarded = pdata.authorized_areal_component(collection)
        assert areal.equals(polygon)
        assert discarded["discarded_parts"] == 1
        assert discarded["discarded_area_total"] == 0.0
        assert discarded["discarded_types"] == linear.geom_type

    # A bare polygonal result discards nothing at all.
    areal, discarded = pdata.authorized_areal_component(polygon)
    assert areal is polygon
    assert discarded["discarded_parts"] == 0


def test_geography_collection_with_two_areal_parts_stops():
    require_shapely()
    from shapely.geometry import GeometryCollection, LineString, MultiPolygon  # noqa: PLC0415

    two_polygons = GeometryCollection([square(0, 0), square(3, 3)])
    try:
        pdata.authorized_areal_component(two_polygons)
    except pdata.GeometryRepairError as error:
        assert "2 areal components" in str(error)
    else:
        raise AssertionError(
            "two areal parts must stop; this is not a keep-the-biggest rule"
        )

    # Polygon + MultiPolygon is still two areal parts, even with a zero-area part.
    mixed = GeometryCollection(
        [LineString([(9, 9), (9, 10)]), square(0, 0), MultiPolygon([square(3, 3)])]
    )
    try:
        pdata.authorized_areal_component(mixed)
    except pdata.GeometryRepairError:
        pass
    else:
        raise AssertionError("polygon + multipolygon must stop the run")


def test_geography_collection_with_a_non_zero_area_part_stops():
    require_shapely()
    from shapely.geometry import GeometryCollection, LineString  # noqa: PLC0415

    # A part that is neither an authorized areal type nor a zero-area
    # linear/point part: a nested collection carrying real surface.
    nested = GeometryCollection(
        [GeometryCollection([square(3, 3)]), square(0, 0)]
    )
    try:
        pdata.authorized_areal_component(nested)
    except pdata.GeometryRepairError as error:
        assert "GeometryCollection" in str(error)
    else:
        raise AssertionError("a nested collection part must stop the run")

    # The zero area is MEASURED, not inferred from the type name.
    stub = _StubCollection([_StubPart("LineString", 4.0), square(0, 0)])
    try:
        pdata.authorized_areal_component(stub)
    except pdata.GeometryRepairError as error:
        assert "area 4.0" in str(error)
    else:
        raise AssertionError("a non-zero-area linear part must stop the run")

    # The same stub with zero area is accepted, so the refusal above is the
    # area check rather than the stub itself.
    ok_stub = _StubCollection([_StubPart("LineString", 0.0), square(0, 0)])
    areal, discarded = pdata.authorized_areal_component(ok_stub)
    assert areal.geom_type == "Polygon"
    assert discarded["discarded_parts"] == 1

    # A non-polygonal make_valid result that is not a collection also stops.
    try:
        pdata.authorized_areal_component(LineString([(0, 0), (1, 1)]))
    except pdata.GeometryRepairError as error:
        assert "non-polygonal" in str(error)
    else:
        raise AssertionError("a bare LineString result must stop the run")


# --------------------------------------------------------------------------
# Adjacency and the donor-distance boundary
# --------------------------------------------------------------------------


def test_geography_point_contact_is_excluded_from_adjacency():
    require_geopandas()
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        artifacts = prepared_fixture(Path(tmp))
        adjacency = artifacts.adjacency
        assert adjacency is not None

        def neighbours(area_id: int) -> set:
            index = adjacency.polygon_id_mapping[area_id]
            return {
                adjacency.polygon_group_mapping[int(other)]
                for other in adjacency.adjacency_dict[index]
            }

        # Area 0 shares the x=1 edge with area 1 but only the (1,1) corner
        # with area 2; the inherited definition keeps the edge and drops the
        # corner.
        assert neighbours(0) == {1}, "point-only contact must not be an edge"
        assert neighbours(1) == {0, 2}
        assert neighbours(2) == {1}
        assert neighbours(3) == set()
        assert adjacency.audit["symmetric"] is True
        assert adjacency.audit["edges_undirected"] == 2
        assert adjacency.audit["isolated_polygons"] == 1

        # index -> group is the direction GeoRF needs, and is the inverse of
        # the helper's id -> index mapping.
        for area_id, index in adjacency.polygon_id_mapping.items():
            assert adjacency.polygon_group_mapping[index] == area_id
        assert len(adjacency.polygon_group_mapping) == 4, "one record per group"

        # The repair audit reached the run-local copy, and the raw source is
        # untouched: the repaired layer lives in a different directory.
        audit = artifacts.audit
        assert audit["repair"]["unchanged_valid"] == 3
        assert audit["repair"]["features"] == 4
        assert audit["area_universe"]["universe"] == 4
        assert Path(audit["local_copy"]["path"]).parent.name == "geography"


def test_geography_reference_coordinates_are_not_polygon_centroids():
    require_geopandas()
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        artifacts = prepared_fixture(Path(tmp))
        centroids = artifacts.adjacency.polygon_centroids
        reference = artifacts.reference_coordinate_array(artifacts.adjacency.area_ids)

        assert reference is not centroids, "distinct objects"
        assert not np.shares_memory(reference, centroids)
        assert reference.shape == centroids.shape
        # Donor distance uses the reference coordinates (Q8r); they genuinely
        # differ from the refinement centroids, so a silent swap is visible.
        assert not np.allclose(reference, centroids)
        assert abs(centroids[0, 0] - 0.5) < 1e-9, "centroid of the unit square"
        assert abs(reference[0, 0] - 0.11) < 1e-9, "keyed reference coordinate"
        assert artifacts.audit["reference_coordinates"]["role"].startswith(
            "donor-distance authority"
        )


def test_geography_adjacency_cache_is_bound_to_content_not_path():
    geopandas = require_geopandas()
    import pickle  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        artifacts = prepared_fixture(Path(tmp))
        cache_path = Path(tmp) / "geography" / "adjacency_cache.pkl"
        repaired_path = Path(artifacts.audit["local_copy"]["path"])
        with open(cache_path, "rb") as handle:
            payload = pickle.load(handle)

        assert pdata.adjacency_cache_is_current(payload, repaired_path, "admin_code")
        assert pdata.adjacency_cache_is_current(payload, repaired_path, "OTHER") is False

        # Same path, different content: the cache must no longer be current.
        moved = geopandas.read_file(repaired_path)
        moved["geometry"] = moved.geometry.translate(xoff=10.0)
        moved.to_file(repaired_path)
        assert not pdata.adjacency_cache_is_current(payload, repaired_path, "admin_code")


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
