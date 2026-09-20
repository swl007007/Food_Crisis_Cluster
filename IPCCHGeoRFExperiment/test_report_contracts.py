"""Compact runnable contract checks for the IPCCH reporter.

Hand-computable synthetic rows only. No real Stage3 predictions exist yet, so
every fixture below is written to match the saved-row input contract in
``report_results.REQUIRED_PREDICTION_COLUMNS`` exactly -- same column names,
same string representation, same missing tokens.

Run with the preferred interpreter::

    python3.12.exe -B IPCCHGeoRFExperiment/test_report_contracts.py

The existing data-boundary checks live in ``test_contracts.py`` and are not
touched by this file.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report_results as rr  # noqa: E402


# --------------------------------------------------------------------------
# Tiny fixtures that match the saved-row contract exactly
# --------------------------------------------------------------------------


def shift_month(label: str, months: int) -> str:
    year, month = (int(part) for part in label.split("-"))
    ordinal = year * 12 + (month - 1) + months
    return f"{ordinal // 12:04d}-{ordinal % 12 + 1:02d}"


def row(
    admin_code: int,
    country: str,
    target_month: str,
    horizon: int,
    truth: int,
    partitioned: int,
    pooled: int | None = None,
    xgb: int | None = None,
    persistence: int | None = None,
    assignment: str = "learned",
    route: str = "local_partition",
    partition_code: int = 3,
    branch: str = "01",
    **overrides,
) -> dict:
    """One saved prediction row, every field as written text.

    Probabilities are generated on the correct side of .5 for the requested
    hard label, so the fixture satisfies Q7a's consistency check by default and
    a violation has to be introduced deliberately.
    """
    pooled = partitioned if pooled is None else pooled
    xgb = partitioned if xgb is None else xgb
    origin = shift_month(target_month, -horizon)

    def prob(label: int) -> str:
        return "0.90" if label == 1 else "0.10"

    record = {
        "admin_code": str(admin_code),
        "country_id": country,
        "country_en": country,
        "ISO3": country[:3].upper(),
        "target_month": target_month,
        "origin_month": origin,
        "horizon_months": str(horizon),
        "ipcch_food_crisis": str(truth),
        "prob_partitioned_rf": prob(partitioned),
        "pred_partitioned_rf": str(partitioned),
        "prob_pooled_rf": prob(pooled),
        "pred_pooled_rf": str(pooled),
        "prob_xgb": prob(xgb),
        "pred_xgb": str(xgb),
        "persistence_pred": "" if persistence is None else str(persistence),
        "persistence_source_month": "" if persistence is None else origin,
        "persistence_age_months": "" if persistence is None else "0",
        "branch_id": branch,
        "partition_code": str(partition_code),
        "assignment_source": assignment,
        "donor_admin_code": "",
        "donor_distance_km": "",
        "model_route": route,
        "model_fallback_reason": "",
        "fold_id": f"h{horizon}_{origin}",
    }
    record.update({k: str(v) for k, v in overrides.items()})
    return record


def frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, dtype=str)


def ledger_rows(rows: list[dict], extra: list[tuple[int, str, int]] = ()) -> pd.DataFrame:
    """The R1 valid-label ledger implied by a fixture, as the runner saves it.

    Every scored area-month is a valid label, and every row that carries a
    persistence value needs the label it claims to carry to exist at its stated
    source month. ``extra`` adds history that predates the test window.
    """
    entries: dict[tuple[int, str], int] = {}

    def put(area: int, month: str, label: int) -> None:
        key = (int(area), month)
        if key in entries and entries[key] != int(label):
            raise AssertionError(f"fixture contradicts itself at {key}")
        entries[key] = int(label)

    for record in rows:
        put(record["admin_code"], record["target_month"], record["ipcch_food_crisis"])
        if record["persistence_pred"] != "":
            put(
                record["admin_code"],
                record["persistence_source_month"],
                record["persistence_pred"],
            )
    for area, month, label in extra:
        put(area, month, label)

    return pd.DataFrame(
        [
            {
                "admin_code": area,
                "year": int(month.split("-")[0]),
                "month": int(month.split("-")[1]),
                "target_valid": 1,
                "ipcch_food_crisis": label,
            }
            for (area, month), label in sorted(entries.items())
        ]
    )


def ledger_of(rows: list[dict], extra: list[tuple[int, str, int]] = ()) -> rr.LabelHistory:
    return rr.normalize_label_history(
        ledger_rows(rows, extra),
        mode=rr.HISTORY_MODE_COMPLETE,
        source="test fixture ledger",
    )


def cohort_of(rows: list[dict], name: str = rr.COHORT_E_ALL, horizon: int = 1) -> rr.Cohort:
    prepared, _ = rr.validate_predictions(frame(rows))
    cohorts = rr.build_cohorts(prepared)
    for cohort in cohorts:
        if cohort.name == name and cohort.horizon == horizon:
            return cohort
    raise AssertionError(f"no {name} cohort at h{horizon}")


def expect_contract_error(call, message: str):
    try:
        call()
    except rr.ReportContractError:
        return
    raise AssertionError(message)


# --------------------------------------------------------------------------
# The worked example, reused by several checks
# --------------------------------------------------------------------------
#
# Country A: (truth, pred) = (1,1) (1,0) (0,1) (0,0)  -> TP1 FP1 FN1 TN1
# Country B: (1,1) (1,1) (0,0)                        -> TP2 FP0 FN0 TN1
#
# Whole cohort  TP3 FP1 FN1 TN2
#   F1        = 2*3 / (2*3 + 1 + 1) = 6/8   = 0.75
#   precision = 3 / (3 + 1)                 = 0.75
#   recall    = 3 / (3 + 1)                 = 0.75
#
# With K = 2 a draw is one of three country multisets:
#   {A,A}: TP2 FP2 FN2 -> 4/(4+2+2) = 0.5
#   {A,B}: the whole cohort          = 0.75
#   {B,B}: TP4 FP0 FN0 -> 8/8        = 1.0

WORKED_EXAMPLE_ROWS = [
    row(11, "Aland", "2023-02", 1, truth=1, partitioned=1, pooled=1, xgb=0, persistence=1),
    row(12, "Aland", "2023-02", 1, truth=1, partitioned=0, pooled=0, xgb=0, persistence=0),
    row(13, "Aland", "2023-02", 1, truth=0, partitioned=1, pooled=0, xgb=0, persistence=0),
    row(14, "Aland", "2023-02", 1, truth=0, partitioned=0, pooled=0, xgb=0, persistence=0),
    row(21, "Bland", "2023-02", 1, truth=1, partitioned=1, pooled=0, xgb=1, persistence=1),
    row(22, "Bland", "2023-02", 1, truth=1, partitioned=1, pooled=0, xgb=1, persistence=1),
    row(23, "Bland", "2023-02", 1, truth=0, partitioned=0, pooled=0, xgb=0, persistence=0),
]


# --------------------------------------------------------------------------
# Q9a metric definitions
# --------------------------------------------------------------------------


def test_zero_numerator_with_positive_denominator_is_zero():
    # TP=0, FP=1, FN=1 -> F1 denominator 2 is positive, so F1 is 0, not NaN
    # and certainly not class 0's score (evaluation.md's legacy nan_option warning).
    result = rr.class1_metrics(tp=0, fp=1, fn=1, tn=5)
    assert result.f1 == 0.0
    assert result.f1_reason == ""
    assert result.precision == 0.0 and result.precision_reason == ""
    assert result.recall == 0.0 and result.recall_reason == ""


def test_zero_denominator_is_nan_with_a_reason():
    # All-negative truth AND all-negative predictions: many TN, no class-1 mass.
    all_negative = rr.class1_metrics(tp=0, fp=0, fn=0, tn=9)
    assert math.isnan(all_negative.f1)
    assert all_negative.f1_reason == "no_positive_truth_and_no_positive_prediction"
    assert math.isnan(all_negative.precision)
    assert all_negative.precision_reason == "no_predicted_positives"
    assert math.isnan(all_negative.recall)
    assert all_negative.recall_reason == "no_observed_positives"

    empty = rr.class1_metrics(tp=0, fp=0, fn=0, tn=0)
    assert math.isnan(empty.f1) and empty.f1_reason == "empty_cohort"
    assert empty.precision_reason == "empty_cohort"
    assert empty.recall_reason == "empty_cohort"

    # Only one side undefined: no predicted positives, but positives exist.
    no_predictions = rr.class1_metrics(tp=0, fp=0, fn=4, tn=6)
    assert no_predictions.f1 == 0.0 and no_predictions.f1_reason == ""
    assert math.isnan(no_predictions.precision)
    assert no_predictions.precision_reason == "no_predicted_positives"
    assert no_predictions.recall == 0.0


def test_worked_example_f1_matches_the_hand_computation():
    cohort = cohort_of(WORKED_EXAMPLE_ROWS)
    result = rr.evaluate_arm(cohort.frame, rr.ARM_PARTITIONED)
    assert (result.tp, result.fp, result.fn, result.tn) == (3, 1, 1, 2)
    assert result.f1 == 0.75          # 6/8
    assert result.precision == 0.75   # 3/4
    assert result.recall == 0.75      # 3/4

    # Pooled RF here predicts one positive that is true (11) and misses three:
    # TP1 FP0 FN3 -> F1 = 2/(2+0+3) = 0.4
    pooled = rr.evaluate_arm(cohort.frame, rr.ARM_POOLED)
    assert (pooled.tp, pooled.fp, pooled.fn) == (1, 0, 3)
    assert pooled.f1 == 2 / 5

    deltas = rr.deltas_table([cohort])
    pooled_delta = deltas[deltas["baseline_arm"] == "pooled_rf"].iloc[0]
    assert pooled_delta["delta_f1"] == 0.75 - 0.4


def test_counts_are_aggregated_before_the_ratio_not_averaged():
    # Per-country F1s are 0.5 and 1.0; their mean is 0.75 only by coincidence,
    # so use a cohort where the two differ: pooled RF gives 0.0 and 1.0 per
    # country (mean 0.5) but the pooled-count F1 is 2*2/(4+0+2) = 0.666...
    cohort = cohort_of(WORKED_EXAMPLE_ROWS)
    xgb = rr.evaluate_arm(cohort.frame, rr.ARM_XGB)
    assert (xgb.tp, xgb.fp, xgb.fn) == (2, 0, 2)
    assert xgb.f1 == 4 / 6
    per_country = [
        rr.evaluate_arm(group, rr.ARM_XGB).f1
        for _, group in cohort.frame.groupby("country_id")
    ]
    assert per_country == [0.0, 1.0]
    assert xgb.f1 != float(np.mean(per_country))


# --------------------------------------------------------------------------
# Q4b cohorts and paired keys
# --------------------------------------------------------------------------


def test_e_persist_is_the_history_available_subset_on_identical_keys():
    rows = [
        row(1, "Aland", "2023-02", 1, truth=1, partitioned=1, persistence=1),
        row(2, "Aland", "2023-02", 1, truth=0, partitioned=0, persistence=0),
        row(3, "Bland", "2023-02", 1, truth=1, partitioned=0, persistence=None),
        row(4, "Bland", "2023-02", 1, truth=1, partitioned=1, persistence=None),
    ]
    prepared, report = rr.validate_predictions(frame(rows))
    cohorts = {c.name: c for c in rr.build_cohorts(prepared) if c.horizon == 1}

    e_all = cohorts[rr.COHORT_E_ALL]
    e_persist = cohorts[rr.COHORT_E_PERSIST]
    key = ["admin_code", "target_month", "horizon_months"]

    all_keys = set(map(tuple, e_all.frame[key].to_numpy()))
    persist_keys = set(map(tuple, e_persist.frame[key].to_numpy()))
    assert len(all_keys) == 4
    assert persist_keys == {(1, "2023-02", 1), (2, "2023-02", 1)}
    assert persist_keys < all_keys
    # Rows without history stay inside E_all and are never imputed away.
    assert report["persistence_available"] == 2

    # Every eligible arm scores exactly the same keys inside one cohort.
    assert [a.name for a in e_all.arms] == ["partitioned_rf", "pooled_rf", "xgb"]
    assert [a.name for a in e_persist.arms] == [
        "partitioned_rf",
        "pooled_rf",
        "xgb",
        "persistence",
    ]
    for arm in e_persist.arms:
        scored = e_persist.frame.loc[
            e_persist.frame[arm.pred_column] >= 0, key
        ]
        assert set(map(tuple, scored.to_numpy())) == persist_keys

    # A subset persistence score is never paired against a full-sample score:
    # deltas only ever compare arms inside one cohort object.
    deltas = rr.deltas_table([e_all, e_persist])
    persistence_rows = deltas[deltas["baseline_arm"] == "persistence"]
    assert set(persistence_rows["cohort"]) == {rr.COHORT_E_PERSIST}
    assert (persistence_rows["n_observations"] == 2).all()


def test_e_persist_membership_comes_from_the_history_not_the_runner():
    """Finding 1, exactly as the review reproduced it.

    Area 1 has February truth = 1 and March truth = 0 at h1. March's origin is
    February, so March *does* have available history: leaving both persistence
    columns blank must not quietly produce an empty E_persist.
    """
    february = row(1, "Aland", "2023-02", 1, truth=1, partitioned=1, persistence=None)
    march = row(1, "Aland", "2023-03", 1, truth=0, partitioned=0, persistence=None)

    expect_contract_error(
        lambda: rr.validate_predictions(frame([february, march])),
        "a row whose origin has available history must not be silently dropped "
        "out of E_persist",
    )

    # The same rows with the correct persistence pass, and March joins E_persist.
    march_ok = row(
        1,
        "Aland",
        "2023-03",
        1,
        truth=0,
        partitioned=0,
        persistence=1,
        persistence_source_month="2023-02",
        persistence_age_months="0",
    )
    prepared, report = rr.validate_predictions(frame([february, march_ok]))
    assert list(prepared["persistence_available"]) == [0, 1]
    assert report["persistence_available"] == 1
    assert report["history_verification"]["mode"] == rr.HISTORY_MODE_PARTIAL

    # And a persistence value that contradicts the saved February truth fails,
    # even though it names a legitimate source month.
    march_wrong = dict(march_ok)
    march_wrong["persistence_pred"] = "0"
    expect_contract_error(
        lambda: rr.validate_predictions(frame([february, march_wrong])),
        "a persistence value contradicting the truth at its own source month "
        "must fail",
    )


def test_persistence_is_verified_exactly_against_the_label_ledger():
    """Finding 1 in complete mode: the ledger decides, the runner does not."""
    rows = [
        row(
            1,
            "Aland",
            "2023-06",
            1,
            truth=1,
            partitioned=1,
            persistence=1,
            persistence_source_month="2023-01",
            persistence_age_months="4",
        ),
        row(2, "Bland", "2023-06", 1, truth=0, partitioned=0, persistence=None),
    ]
    # Area 1's latest valid label at or before O = 2023-05 is 2023-01 = 1;
    # area 2 has nothing before its own target month.
    history = ledger_of(rows, extra=[(1, "2023-01", 1)])
    prepared, report = rr.validate_predictions(frame(rows), history)
    assert list(prepared["persistence_available"]) == [1, 0]
    assert report["history_verification"]["mode"] == rr.HISTORY_MODE_COMPLETE
    assert report["history_verification"]["rows_with_history"] == 1

    # A label the ledger does not hold at that month.
    wrong_source = [dict(rows[0]), dict(rows[1])]
    wrong_source[0]["persistence_source_month"] = "2023-02"
    wrong_source[0]["persistence_age_months"] = "3"
    expect_contract_error(
        lambda: rr.validate_predictions(frame(wrong_source), history),
        "a source month that is not the latest valid label <= O must fail",
    )

    # The right month, the wrong value.
    wrong_value = [dict(rows[0]), dict(rows[1])]
    wrong_value[0]["persistence_pred"] = "0"
    expect_contract_error(
        lambda: rr.validate_predictions(frame(wrong_value), history),
        "a persistence value that contradicts the ledger must fail",
    )

    # History the ledger says does not exist.
    invented = [dict(rows[0]), dict(rows[1])]
    invented[1]["persistence_pred"] = "0"
    invented[1]["persistence_source_month"] = "2023-05"
    invented[1]["persistence_age_months"] = "0"
    expect_contract_error(
        lambda: rr.validate_predictions(frame(invented), history),
        "persistence with no valid label at or before O must fail",
    )

    # And the row that the ledger says does have history cannot be left blank.
    withheld = [dict(rows[0]), dict(rows[1])]
    withheld[0]["persistence_pred"] = ""
    withheld[0]["persistence_source_month"] = ""
    withheld[0]["persistence_age_months"] = ""
    expect_contract_error(
        lambda: rr.validate_predictions(frame(withheld), history),
        "a history-available row must not be missing from E_persist",
    )


def test_malformed_numbers_fail_instead_of_becoming_missing_or_truncated():
    """Finding 2: coercion must not change cohort membership or identity."""
    broken = [row(1, "Aland", "2023-02", 1, truth=1, partitioned=1, persistence=1)]
    broken[0]["persistence_pred"] = "BROKEN"
    expect_contract_error(
        lambda: rr.validate_predictions(frame(broken)),
        "a malformed persistence_pred must fail, not drop the row out of E_persist",
    )

    fractional = [row(1, "Aland", "2023-02", 1, truth=1, partitioned=1)]
    fractional[0]["admin_code"] = "2.7"
    expect_contract_error(
        lambda: rr.validate_predictions(frame(fractional)),
        "a fractional admin_code must fail, not silently become area 2",
    )

    # A declared missing token still means missing, and "2" is still area 2.
    blank = [row(1, "Aland", "2023-02", 1, truth=1, partitioned=1, persistence=None)]
    blank[0]["persistence_pred"] = "NA"
    prepared, _ = rr.validate_predictions(frame(blank))
    assert prepared["persistence_available"].iloc[0] == 0
    assert prepared["admin_code"].iloc[0] == 1

    for field, value in (
        ("prob_xgb", "high"),
        ("horizon_months", "1.5"),
        ("partition_code", "3.5"),
        ("ipcch_food_crisis", "yes"),
    ):
        bad = [row(1, "Aland", "2023-02", 1, truth=1, partitioned=1)]
        bad[0][field] = value
        expect_contract_error(
            lambda bad=bad: rr.validate_predictions(frame(bad)),
            f"a malformed {field} must fail",
        )


def test_every_approved_horizon_keeps_a_cohort():
    """Finding 3: an entire empty horizon must not vanish from the report."""
    prepared, _ = rr.validate_predictions(
        frame([row(1, "Aland", "2023-02", 1, truth=1, partitioned=1, persistence=None)])
    )
    cohorts = rr.build_cohorts(prepared)
    assert sorted({c.horizon for c in cohorts}) == [1, 3, 6, 12]
    assert len(cohorts) == 8  # four horizons x (E_all, E_persist)

    metrics = rr.metrics_table(cohorts)
    for horizon in (3, 6, 12):
        empty = metrics[metrics["horizon_months"] == horizon]
        assert len(empty) == 3 + 4  # three learned arms on E_all, four on E_persist
        assert (empty["n_observations"] == 0).all()
        assert empty["f1"].isna().all()
        assert set(empty["f1_reason"]) == {"empty_cohort"}

    support = rr.cohort_support_table(cohorts, prepared).set_index(
        ["horizon_months", "cohort"]
    )
    # R4's schedule: 35/33/30/24 main target months.
    for horizon, scheduled in ((1, 35), (3, 33), (6, 30), (12, 24)):
        assert support.loc[(horizon, "E_all"), "n_scheduled_target_months"] == scheduled
    assert support.loc[(1, "E_all"), "n_scheduled_target_months_present"] == 1
    assert support.loc[(3, "E_all"), "n_scheduled_target_months_present"] == 0
    assert support.loc[(3, "E_all"), "n_scheduled_target_months_absent"] == 33


def test_main_and_partial_2026_stay_separate():
    rows = [
        row(1, "Aland", "2025-12", 1, truth=1, partitioned=1, persistence=1),
        row(1, "Aland", "2026-01", 1, truth=1, partitioned=1, persistence=1),
        row(2, "Aland", "2024-01", 12, truth=0, partitioned=0, persistence=0),
    ]
    prepared, _ = rr.validate_predictions(frame(rows))
    assert list(prepared["period"]) == ["main", "partial_2026", "main"]

    # h12's first main target is 2024-01, so 2023-12 is outside the schedule.
    expect_contract_error(
        lambda: rr.validate_predictions(
            frame([row(2, "Aland", "2023-12", 12, truth=0, partitioned=0)])
        ),
        "a target month outside the R4 schedule must fail",
    )


# --------------------------------------------------------------------------
# Input contract failures
# --------------------------------------------------------------------------


def test_missing_learned_prediction_fails():
    bad = [row(1, "Aland", "2023-02", 1, truth=1, partitioned=1)]
    bad[0]["pred_partitioned_rf"] = ""
    expect_contract_error(
        lambda: rr.validate_predictions(frame(bad)),
        "a missing learned prediction must fail, never shrink the cohort",
    )

    bad = [row(1, "Aland", "2023-02", 1, truth=1, partitioned=1)]
    bad[0]["prob_xgb"] = ""
    expect_contract_error(
        lambda: rr.validate_predictions(frame(bad)),
        "a missing probability must fail",
    )


def test_missing_country_fails():
    bad = [row(1, "Aland", "2023-02", 1, truth=1, partitioned=1)]
    bad[0]["country_id"] = ""
    bad[0]["country_en"] = ""
    expect_contract_error(
        lambda: rr.validate_predictions(frame(bad)),
        "a missing country mapping must fail before the bootstrap",
    )

    # One area cannot belong to two countries: the cluster key must be stable.
    ambiguous = [
        row(1, "Aland", "2023-02", 1, truth=1, partitioned=1),
        row(1, "Bland", "2023-05", 1, truth=1, partitioned=1),
    ]
    expect_contract_error(
        lambda: rr.validate_predictions(frame(ambiguous)),
        "an area mapping to two countries must fail",
    )


def test_key_and_consistency_failures():
    duplicate = [
        row(1, "Aland", "2023-02", 1, truth=1, partitioned=1),
        row(1, "Aland", "2023-02", 1, truth=0, partitioned=0),
    ]
    expect_contract_error(
        lambda: rr.validate_predictions(frame(duplicate)),
        "duplicate cohort keys must fail",
    )

    tie = [row(1, "Aland", "2023-02", 1, truth=1, partitioned=1)]
    tie[0]["prob_partitioned_rf"] = "0.5"  # Q7a: exactly .5 must give 0
    expect_contract_error(
        lambda: rr.validate_predictions(frame(tie)),
        "a hard label of 1 at p = .5 must fail (Q7a ties give 0)",
    )
    tie[0]["pred_partitioned_rf"] = "0"
    prepared, _ = rr.validate_predictions(frame(tie))
    assert prepared["pred_partitioned_rf"].iloc[0] == 0

    bad_origin = [row(1, "Aland", "2023-02", 1, truth=1, partitioned=1)]
    bad_origin[0]["origin_month"] = "2022-12"  # T - H is 2023-01
    expect_contract_error(
        lambda: rr.validate_predictions(frame(bad_origin)),
        "origin != target - horizon must fail",
    )

    future_history = [row(1, "Aland", "2023-06", 1, truth=1, partitioned=1, persistence=1)]
    future_history[0]["persistence_source_month"] = "2023-06"  # after O = 2023-05
    future_history[0]["persistence_age_months"] = "-1"
    expect_contract_error(
        lambda: rr.validate_predictions(frame(future_history)),
        "a persistence label dated after its own origin must fail",
    )

    orphan = [row(1, "Aland", "2023-02", 1, truth=1, partitioned=1, persistence=1)]
    orphan[0]["persistence_source_month"] = ""
    expect_contract_error(
        lambda: rr.validate_predictions(frame(orphan)),
        "a persistence label without a source month must fail",
    )

    bad_assignment = [
        row(1, "Aland", "2023-02", 1, truth=1, partitioned=1, assignment="guessed")
    ]
    expect_contract_error(
        lambda: rr.validate_predictions(frame(bad_assignment)),
        "an unknown assignment_source must fail",
    )

    mismatched_code = [
        row(
            1,
            "Aland",
            "2023-02",
            1,
            truth=1,
            partitioned=1,
            assignment="unresolved",
            partition_code=7,
        )
    ]
    expect_contract_error(
        lambda: rr.validate_predictions(frame(mismatched_code)),
        "'unresolved' with a partition code other than -1 must fail",
    )


def test_branch_string_keeps_its_leading_zeros():
    prepared, _ = rr.validate_predictions(
        frame([row(1, "Aland", "2023-02", 1, truth=1, partitioned=1, branch="001")])
    )
    assert prepared["branch_id"].iloc[0] == "001"


# --------------------------------------------------------------------------
# Q9b bootstrap
# --------------------------------------------------------------------------


def brute_force_replicates(cohort: rr.Cohort, arm: rr.Arm, n_draws: int, seed: int):
    """Reference bootstrap that really materialises the duplicated rows.

    This is the definition Q9b states ("keeps all rows per copy, preserving
    multiplicity"); the production path sums per-country counts instead, which
    is only valid because Q9a's counts are additive over rows. Comparing the
    two is the check.
    """
    countries = sorted(cohort.frame["country_id"].unique())
    k = len(countries)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_draws, k))
    values = []
    sizes = []
    for draw in idx:
        parts = [cohort.frame[cohort.frame["country_id"] == countries[i]] for i in draw]
        sample = pd.concat(parts, ignore_index=True)
        sizes.append(len(sample))
        counts = rr.confusion_counts(
            sample["ipcch_food_crisis"].to_numpy(), sample[arm.pred_column].to_numpy()
        )
        values.append(rr.class1_metrics(*counts).f1)
    return np.asarray(values, dtype=np.float64), idx, sizes


def test_country_multiplicity_is_retained_inside_a_draw():
    cohort = cohort_of(WORKED_EXAMPLE_ROWS)
    n_draws = 40
    result = rr.bootstrap_cohort(cohort, n_draws=n_draws, seed=rr.BOOTSTRAP_SEED)
    reference, idx, sizes = brute_force_replicates(
        cohort, rr.ARM_PARTITIONED, n_draws, rr.BOOTSTRAP_SEED
    )

    produced = result.replicates[result.replicates["statistic"] == "f1__partitioned_rf"]
    produced = produced.sort_values("draw_index")["value"].to_numpy()
    assert np.array_equal(produced, reference, equal_nan=True)

    # The exported draw table must reproduce the multiplicities, including a
    # country drawn twice (4 + 4 rows) versus once each (4 + 3 rows).
    rows_per_country = cohort.frame["country_id"].value_counts().to_dict()
    for draw_index in range(n_draws):
        drawn = result.draws[result.draws["draw_index"] == draw_index]
        assert int(drawn["multiplicity"].sum()) == 2
        rebuilt = sum(
            int(m) * rows_per_country[c]
            for c, m in zip(drawn["country_id"], drawn["multiplicity"])
        )
        assert rebuilt == sizes[draw_index]
    assert {8, 7, 6} >= set(sizes)
    assert 8 in sizes and 6 in sizes  # {A,A} and {B,B} both occur


def test_deltas_use_the_same_draw_for_both_arms():
    cohort = cohort_of(WORKED_EXAMPLE_ROWS, name=rr.COHORT_E_PERSIST)
    result = rr.bootstrap_cohort(cohort, n_draws=200, seed=rr.BOOTSTRAP_SEED)
    wide = result.replicates.pivot(index="draw_index", columns="statistic", values="value")

    for baseline in ("pooled_rf", "xgb", "persistence"):
        delta = wide[f"delta_f1__partitioned_rf_minus_{baseline}"].to_numpy()
        expected = wide["f1__partitioned_rf"].to_numpy() - wide[f"f1__{baseline}"].to_numpy()
        assert np.array_equal(delta, expected, equal_nan=True)

    # Independent draws per arm would break this: a shared draw means the
    # delta is exactly reconstructible from the two per-arm replicate columns.
    assert len(wide) == 200
    assert set(result.summary["statistic"]) == {
        "f1__partitioned_rf",
        "f1__pooled_rf",
        "f1__xgb",
        "f1__persistence",
        "delta_f1__partitioned_rf_minus_pooled_rf",
        "delta_f1__partitioned_rf_minus_xgb",
        "delta_f1__partitioned_rf_minus_persistence",
    }


def test_worked_example_interval_matches_the_hand_computation():
    cohort = cohort_of(WORKED_EXAMPLE_ROWS)
    result = rr.bootstrap_cohort(cohort, n_draws=rr.BOOTSTRAP_DRAWS, seed=rr.BOOTSTRAP_SEED)
    replicates = result.replicates[
        result.replicates["statistic"] == "f1__partitioned_rf"
    ].sort_values("draw_index")["value"].to_numpy()

    # With K = 2 only three multisets exist, so only three F1 values can occur.
    assert sorted(set(replicates)) == [0.5, 0.75, 1.0]
    n_aa = int((replicates == 0.5).sum())
    n_bb = int((replicates == 1.0).sum())

    # The 2.5 percentile of 1000 sorted values sits at index (1000-1)*0.025 =
    # 24.975, and the 97.5 percentile at 974.025. Both neighbours of 24.975 are
    # 0.5 whenever n_aa >= 26, and both neighbours of 974.025 are 1.0 whenever
    # n_bb >= 26, so the interval is exactly [0.5, 1.0] by inspection.
    assert n_aa >= 26 and n_bb >= 26
    summary = result.summary.set_index("statistic")
    assert summary.loc["f1__partitioned_rf", "point"] == 0.75
    assert summary.loc["f1__partitioned_rf", "ci_lower"] == 0.5
    assert summary.loc["f1__partitioned_rf", "ci_upper"] == 1.0
    assert summary.loc["f1__partitioned_rf", "ci_available"] == 1
    assert summary.loc["f1__partitioned_rf", "defined_replicates"] == 1000
    assert summary.loc["f1__partitioned_rf", "uses_defined_replicates_only"] == 0

    # And the same numbers come straight out of numpy on the saved replicates.
    lower, upper = np.percentile(replicates, [2.5, 97.5], method="linear")
    assert (lower, upper) == (0.5, 1.0)
    print(
        f"      worked example: F1 = 6/8 = 0.75, draws {{A,A}}={n_aa} "
        f"{{A,B}}={1000 - n_aa - n_bb} {{B,B}}={n_bb}, 95% CI [0.5, 1.0]"
    )


def test_ci_is_suppressed_when_the_point_is_undefined():
    # All-negative truth and all-negative predictions: F1 denominator is 0.
    rows = [
        row(1, "Aland", "2023-02", 1, truth=0, partitioned=0),
        row(2, "Bland", "2023-02", 1, truth=0, partitioned=0),
    ]
    cohort = cohort_of(rows)
    result = rr.bootstrap_cohort(cohort, n_draws=50, seed=rr.BOOTSTRAP_SEED)
    line = result.summary.set_index("statistic").loc["f1__partitioned_rf"]
    assert math.isnan(line["point"])
    assert line["ci_available"] == 0
    assert line["suppression_reason"] == "point_estimate_undefined"
    assert math.isnan(line["ci_lower"]) and math.isnan(line["ci_upper"])


def test_only_the_interval_is_suppressed_below_two_countries():
    # Q9b suppresses the *interval* when K < 2; it does not cancel the
    # sampling. The 50 replicates must still exist and be reported -- with
    # K = 1 every draw is that one country, so each replicate equals the point.
    rows = [
        row(1, "Aland", "2023-02", 1, truth=1, partitioned=1),
        row(2, "Aland", "2023-05", 1, truth=0, partitioned=0),
    ]
    cohort = cohort_of(rows)
    result = rr.bootstrap_cohort(cohort, n_draws=50, seed=rr.BOOTSTRAP_SEED)
    line = result.summary.set_index("statistic").loc["f1__partitioned_rf"]
    assert line["point"] == 1.0
    assert line["n_countries"] == 1
    assert line["ci_available"] == 0
    assert line["suppression_reason"] == "fewer_than_2_countries"
    assert math.isnan(line["ci_lower"]) and math.isnan(line["ci_upper"])

    # No silent "requested 1000, produced 0": the replicates are real.
    assert line["requested_draws"] == 50
    assert line["defined_replicates"] == 50
    assert line["undefined_replicates"] == 0
    replicates = result.replicates[
        result.replicates["statistic"] == "f1__partitioned_rf"
    ]
    assert len(replicates) == 50
    assert (replicates["value"] == 1.0).all()
    assert sorted(result.draws["draw_index"].unique()) == list(range(50))
    assert set(result.draws["country_id"]) == {"Aland"}
    assert (result.draws["multiplicity"] == 1).all()


def test_an_empty_cohort_has_no_countries_to_resample():
    # The only case where the sampler legitimately produces nothing.
    prepared, _ = rr.validate_predictions(
        frame([row(1, "Aland", "2023-02", 1, truth=1, partitioned=1)])
    )
    empty = [
        c
        for c in rr.build_cohorts(prepared)
        if c.horizon == 12 and c.name == rr.COHORT_E_ALL
    ][0]
    assert len(empty.frame) == 0
    result = rr.bootstrap_cohort(empty, n_draws=50, seed=rr.BOOTSTRAP_SEED)
    line = result.summary.set_index("statistic").loc["f1__partitioned_rf"]
    assert math.isnan(line["point"])
    assert line["n_countries"] == 0
    assert line["suppression_reason"] == "empty_cohort_no_countries_to_resample"
    assert result.replicates.empty and result.draws.empty


def test_undefined_replicates_are_counted_not_redrawn_or_zeroed():
    # Country B contributes no class-1 mass at all, so the {B,B} draw has a
    # zero F1 denominator: NaN, kept as NaN, counted, never resampled away.
    rows = [
        row(1, "Aland", "2023-02", 1, truth=1, partitioned=1),
        row(2, "Aland", "2023-02", 1, truth=0, partitioned=0),
        row(3, "Bland", "2023-02", 1, truth=0, partitioned=0),
        row(4, "Bland", "2023-05", 1, truth=0, partitioned=0),
    ]
    cohort = cohort_of(rows)
    result = rr.bootstrap_cohort(cohort, n_draws=rr.BOOTSTRAP_DRAWS, seed=rr.BOOTSTRAP_SEED)
    line = result.summary.set_index("statistic").loc["f1__partitioned_rf"]

    replicates = result.replicates[
        result.replicates["statistic"] == "f1__partitioned_rf"
    ]["value"].to_numpy()
    n_undefined = int(np.isnan(replicates).sum())

    assert line["point"] == 1.0
    assert n_undefined > 0
    assert line["undefined_replicates"] == n_undefined
    assert line["defined_replicates"] == rr.BOOTSTRAP_DRAWS - n_undefined
    assert line["requested_draws"] == rr.BOOTSTRAP_DRAWS  # no redraw-until-valid
    assert line["uses_defined_replicates_only"] == 1
    assert line["undefined_reason"] == "replicate_f1_denominator_zero"
    assert line["ci_available"] == 1
    # A NaN-to-zero policy would have dragged the lower bound to 0.
    assert line["ci_lower"] == 1.0 and line["ci_upper"] == 1.0
    defined = replicates[~np.isnan(replicates)]
    lower, upper = np.percentile(defined, [2.5, 97.5], method="linear")
    assert (line["ci_lower"], line["ci_upper"]) == (lower, upper)


def test_ci_is_suppressed_below_two_defined_replicates():
    # Two countries, both class-1 empty except one row in A, and only 1 draw
    # requested -- a single defined replicate cannot bound anything.
    rows = [
        row(1, "Aland", "2023-02", 1, truth=1, partitioned=1),
        row(2, "Bland", "2023-02", 1, truth=0, partitioned=0),
    ]
    cohort = cohort_of(rows)
    result = rr.bootstrap_cohort(cohort, n_draws=1, seed=rr.BOOTSTRAP_SEED)
    line = result.summary.set_index("statistic").loc["f1__partitioned_rf"]
    assert line["defined_replicates"] <= 1
    assert line["ci_available"] == 0
    assert line["suppression_reason"] == "fewer_than_2_defined_replicates"


def test_bootstrap_point_agrees_with_the_metrics_table():
    cohort = cohort_of(WORKED_EXAMPLE_ROWS, name=rr.COHORT_E_PERSIST)
    metrics = rr.metrics_table([cohort]).set_index("arm")
    result = rr.bootstrap_cohort(cohort, n_draws=10, seed=rr.BOOTSTRAP_SEED)
    summary = result.summary.set_index("statistic")
    for arm in cohort.arms:
        assert summary.loc[f"f1__{arm.name}", "point"] == metrics.loc[arm.name, "f1"]


# --------------------------------------------------------------------------
# End-to-end report
# --------------------------------------------------------------------------


def _report_rows() -> list[dict]:
    rows = list(WORKED_EXAMPLE_ROWS)
    rows += [
        # Area 15's first valid label is its own target month, which is after
        # its origin, so this row genuinely has no available history. (Area 11
        # could not play that part: its 2023-02 label precedes the 2023-04
        # origin, so leaving persistence blank there would be a Q4b violation,
        # not a missing-history row.)
        row(15, "Aland", "2023-05", 1, truth=0, partitioned=0, persistence=None),
        row(
            31,
            "Cland",
            "2024-01",
            12,
            truth=1,
            partitioned=0,
            pooled=1,
            xgb=1,
            persistence=1,
            assignment="nearest_donor",
            route="pooled_fallback",
            partition_code=2,
            donor_admin_code="11",
            donor_distance_km="42.5",
        ),
        row(
            32,
            "Cland",
            "2024-01",
            12,
            truth=0,
            partitioned=0,
            pooled=0,
            xgb=0,
            persistence=0,
            assignment="unresolved",
            route="pooled_fallback",
            partition_code=-1,
            branch="",
        ),
        row(11, "Aland", "2026-02", 1, truth=1, partitioned=1, persistence=1),
    ]
    return rows


def test_generate_report_writes_the_expected_tree():
    tmp = Path(tempfile.mkdtemp())
    try:
        source = tmp / "predictions.csv"
        frame(_report_rows()).to_csv(source, index=False)
        out = tmp / "report"
        manifest = rr.generate_report(out, predictions_path=source)

        assert manifest["status"] == "complete"
        assert manifest["fitting_performed"] is False
        assert manifest["map_modified"] is False
        assert manifest["periods"] == ["main", "partial_2026"]

        for relative in (
            "reporting_config.json",
            "validation.json",
            "limitations.txt",
            "report_manifest.json",
            "cohort_keys.csv.gz",
            "main/metrics.csv",
            "main/deltas.csv",
            "main/cohort_support.csv",
            "main/breakdown_target_month.csv",
            "main/breakdown_target_year.csv",
            "main/breakdown_country.csv",
            "main/breakdown_assignment_provenance.csv",
            "main/bootstrap_summary.csv",
            "main/bootstrap_replicates.csv.gz",
            "main/bootstrap_draws.csv.gz",
            "main/bootstrap_config.json",
            "partial_2026/metrics.csv",
            "partial_2026/bootstrap_suppressed.json",
        ):
            assert (out / relative).is_file(), relative

        # Q9b: no partial-2026 or subgroup intervals in this baseline.
        assert not (out / "partial_2026/bootstrap_summary.csv").exists()

        text = (out / "limitations.txt").read_text(encoding="utf-8")
        assert rr.UNCERTAINTY_SCOPE_SENTENCE in text

        metrics = pd.read_csv(out / "main/metrics.csv")
        h1_all = metrics[
            (metrics["horizon_months"] == 1) & (metrics["cohort"] == "E_all")
        ].set_index("arm")
        # The worked example plus one extra Aland negative both arms call 0.
        assert h1_all.loc["partitioned_rf", "tp"] == 3
        assert h1_all.loc["partitioned_rf", "f1"] == 0.75
        assert h1_all.loc["partitioned_rf", "n_observations"] == 8
        assert h1_all.loc["partitioned_rf", "n_countries"] == 2

        support = pd.read_csv(out / "main/cohort_support.csv")
        h1 = support[(support["horizon_months"] == 1)].set_index("cohort")
        assert h1.loc["E_all", "n_persistence_available"] == 7
        assert h1.loc["E_all", "persistence_coverage"] == 7 / 8
        assert h1.loc["E_persist", "n_observations"] == 7
        assert h1.loc["E_persist", "persistence_coverage"] == 7 / 8

        h12 = support[support["horizon_months"] == 12].set_index("cohort")
        assert h12.loc["E_all", "assignment_nearest_donor"] == 1
        assert h12.loc["E_all", "assignment_unresolved"] == 1
        assert h12.loc["E_all", "route_pooled_fallback"] == 2

        # h12 has a single country, so Q9b suppresses its intervals.
        boot = pd.read_csv(out / "main/bootstrap_summary.csv")
        h12_boot = boot[boot["horizon_months"] == 12]
        assert set(h12_boot["suppression_reason"]) == {"fewer_than_2_countries"}

        keys = pd.read_csv(out / "cohort_keys.csv.gz")
        assert int(keys["in_e_all"].sum()) == len(_report_rows())
        # 11 rows, one of which (area 15 at 2023-05) has no available history.
        assert int(keys["in_e_persist"].sum()) == 10

        # Q9a: h3 and h6 have no saved rows at all and must still be reported
        # as empty cohorts, reconciled against R4's schedule.
        for horizon, scheduled in ((3, 33), (6, 30)):
            empty = support[support["horizon_months"] == horizon].set_index("cohort")
            assert set(empty.index) == {"E_all", "E_persist"}
            assert empty.loc["E_all", "n_observations"] == 0
            assert empty.loc["E_all", "n_scheduled_target_months"] == scheduled
            assert empty.loc["E_all", "n_scheduled_target_months_absent"] == scheduled
        empty_metrics = metrics[
            (metrics["horizon_months"] == 3) & (metrics["arm"] == "partitioned_rf")
        ]
        assert len(empty_metrics) == 2  # E_all and E_persist
        assert empty_metrics["f1"].isna().all()
        assert set(empty_metrics["f1_reason"]) == {"empty_cohort"}

        # h1 is fully scheduled but only two of its 35 months carry rows.
        assert h1.loc["E_all", "n_scheduled_target_months"] == 35
        assert h1.loc["E_all", "n_target_months_present"] == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_report_is_byte_for_byte_reproducible():
    tmp = Path(tempfile.mkdtemp())
    try:
        source = tmp / "predictions.csv"
        frame(_report_rows()).to_csv(source, index=False)
        first = rr.generate_report(tmp / "a", predictions_path=source)
        second = rr.generate_report(tmp / "b", predictions_path=source)

        files = sorted(first["files"])
        assert files == sorted(second["files"])
        for relative in files:
            left = (tmp / "a" / relative).read_bytes()
            right = (tmp / "b" / relative).read_bytes()
            assert left == right, f"{relative} is not reproducible"

        # report_manifest.json carries the only non-deterministic field.
        for key in first:
            if key in {"generated_at", "out_dir"}:
                continue
            assert first[key] == second[key], key
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_report_refuses_a_non_empty_out_dir_and_records_validation_failure():
    tmp = Path(tempfile.mkdtemp())
    try:
        source = tmp / "predictions.csv"
        frame(_report_rows()).to_csv(source, index=False)
        out = tmp / "report"
        rr.generate_report(out, predictions_path=source)
        expect_contract_error(
            lambda: rr.generate_report(out, predictions_path=source),
            "a non-empty output directory must be refused",
        )

        broken = frame(_report_rows())
        broken.loc[0, "pred_pooled_rf"] = ""
        broken_path = tmp / "broken.csv"
        broken.to_csv(broken_path, index=False)
        failed_out = tmp / "failed"
        expect_contract_error(
            lambda: rr.generate_report(failed_out, predictions_path=broken_path),
            "a missing prediction must fail the whole report",
        )
        # Diagnostic evidence is preserved and the run is never called complete.
        assert (failed_out / "validation.json").is_file()
        assert not (failed_out / "report_manifest.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_history_falls_back_to_the_pinned_source_through_prepare_data():
    """R1 is never reimplemented: the fallback reads build_target_ledger."""
    tmp = Path(tempfile.mkdtemp())
    try:
        # A miniature source in the pinned CSV's shape; area 1 has a valid
        # January label (P3+ = .30 > .20) and area 2's February row is invalid
        # (the five shares sum to .50, outside R1's [.90, 1.10]).
        source_csv = tmp / "IPCCH_mini.csv"
        pd.DataFrame(
            [
                {
                    "admin_code": 1,
                    "year": 2023,
                    "month": 1,
                    "phase1_percent": "0.40",
                    "phase2_percent": "0.30",
                    "phase3_percent": "0.20",
                    "phase4_percent": "0.10",
                    "phase5_percent": "0.00",
                    "estimated_population": "1000",
                    "overall_phase": "3",
                    "country_en": "Aland",
                    "ISO3": "ALA",
                },
                {
                    "admin_code": 2,
                    "year": 2023,
                    "month": 2,
                    "phase1_percent": "0.20",
                    "phase2_percent": "0.20",
                    "phase3_percent": "0.10",
                    "phase4_percent": "0.00",
                    "phase5_percent": "0.00",
                    "estimated_population": "1000",
                    "overall_phase": "2",
                    "country_en": "Aland",
                    "ISO3": "ALA",
                },
            ]
        ).to_csv(source_csv, index=False)

        run_dir = tmp / "run"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(
            json.dumps({"source": {"csv": str(source_csv)}}), encoding="utf-8"
        )

        history = rr.find_label_history(run_dir)
        assert history.mode == rr.HISTORY_MODE_COMPLETE
        assert "build_target_ledger" in history.source
        # Only the valid January label survives; the invalid row is not history.
        assert list(history.area) == [1]
        assert list(history.label) == [1]
        assert history.month_ord[0] == 2023 * 12 + 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cli_round_trip():
    tmp = Path(tempfile.mkdtemp())
    try:
        source = tmp / "predictions.csv"
        frame(_report_rows()).to_csv(source, index=False)
        run_dir = tmp / "run"
        (run_dir / "stage3").mkdir(parents=True)
        (run_dir / "data").mkdir(parents=True)
        shutil.copy(source, run_dir / "stage3" / "predictions.csv")

        # A real run must be able to prove persistence availability, so the
        # reporter refuses to guess it from the runner's own column.
        assert rr.main(["--run-dir", str(run_dir), "--out-dir", str(tmp / "cli0")]) == 2

        ledger_rows(_report_rows()).to_csv(
            run_dir / "data" / "target_ledger_valid.csv.gz", index=False
        )
        code = rr.main(["--run-dir", str(run_dir), "--out-dir", str(tmp / "cli")])
        assert code == 0
        assert (tmp / "cli" / "report_manifest.json").is_file()
        verification = json.loads(
            (tmp / "cli" / "validation.json").read_text(encoding="utf-8")
        )["history_verification"]
        assert verification["mode"] == rr.HISTORY_MODE_COMPLETE
        assert verification["verified"] == "availability_source_month_and_value"

        broken = frame(_report_rows())
        broken.loc[0, "country_en"] = ""
        broken.loc[0, "country_id"] = ""
        broken.to_csv(run_dir / "stage3" / "predictions.csv", index=False)
        assert rr.main(["--run-dir", str(run_dir), "--out-dir", str(tmp / "cli2")]) == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
    for test in tests:
        try:
            test()
        except Exception:  # noqa: BLE001 - report every failure, keep going
            failures += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
