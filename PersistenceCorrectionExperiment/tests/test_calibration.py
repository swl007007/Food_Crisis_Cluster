"""Contract tests for Phase 3 calibration (task AC3, plus the purity guarantees).

Run from the repository root::

    PYTHONPATH="$PWD/PersistenceCorrectionExperiment" python3 -m pytest \
        PersistenceCorrectionExperiment/tests -q

Every test here is synthetic; none reads the real probability artifacts, so the
suite stays fast and cannot itself touch the 2021-2024 labels.
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parent
for _path in (EXPERIMENT_DIR, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from persistencecorrection import calibration as C  # noqa: E402
from persistencecorrection.calibration import CalibrationContractError  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _frame(
    years=(2018, 2019),
    months=(2, 6, 10),
    partitions=(0, 1),
    rows_per_group=120,
    seed=0,
):
    """A Phase 2-shaped frame whose crisis rate genuinely depends on the score."""
    rng = np.random.default_rng(seed)
    records = []
    for year in years:
        for month in months:
            for partition in partitions:
                probability = np.round(rng.random(rows_per_group), 2)
                # Partition-specific distortion, so a per-group calibrator and a
                # month-pooled calibrator cannot agree by accident.
                true_rate = np.clip(probability ** (1 + partition), 0, 1)
                truth = (rng.random(rows_per_group) < true_rate).astype(int)
                for index in range(rows_per_group):
                    records.append(
                        {
                            "FEWSNET_admin_code": partition * 100000
                            + index
                            + 1000 * month,
                            "month_start": pd.Timestamp(year=year, month=month, day=1),
                            "partition_id": partition,
                            "y_true": int(truth[index]),
                            "y_prob_partitioned": float(probability[index]),
                        }
                    )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# AC3: no selection-window or test-window row may enter a calibrator fit
# ---------------------------------------------------------------------------


def test_fit_refuses_any_row_outside_the_declared_fit_window():
    intruded = pd.concat(
        [_frame(), _frame(years=(2020,), seed=1)], ignore_index=True
    )
    with pytest.raises(CalibrationContractError) as excinfo:
        C.fit_calibrators(intruded, scope=1)
    assert "2020" in str(excinfo.value)


def test_fit_refuses_test_window_rows_even_when_they_are_the_majority():
    for year in (2021, 2022, 2023, 2024):
        intruded = pd.concat(
            [_frame(), _frame(years=(year,), rows_per_group=400, seed=2)],
            ignore_index=True,
        )
        with pytest.raises(CalibrationContractError):
            C.fit_calibrators(intruded, scope=1)


def test_declared_windows_are_disjoint_and_cover_the_declared_roles():
    assert set(C.FIT_YEARS).isdisjoint(C.SELECTION_YEARS)
    assert set(C.FIT_YEARS).isdisjoint(C.TEST_YEARS)
    assert set(C.SELECTION_YEARS).isdisjoint(C.TEST_YEARS)
    assert C.FIT_YEARS == (2018, 2019)
    assert C.SELECTION_YEARS == (2020,)
    assert C.TEST_YEARS == (2021, 2022, 2023, 2024)


def test_calibrators_fitted_on_2018_2019_ignore_later_rows_entirely():
    """The fitted object must be identical whether or not later rows exist."""
    fit_frame = _frame()
    baseline = C.fit_calibrators(fit_frame, scope=1).digest()
    # A caller that filters correctly gets the same answer; a caller that does
    # not gets an exception, never a silently different calibrator.
    combined = pd.concat([fit_frame, _frame(years=(2020,), seed=3)], ignore_index=True)
    filtered = combined.loc[combined["month_start"].dt.year.isin(C.FIT_YEARS)]
    assert C.fit_calibrators(filtered, scope=1).digest() == baseline


# ---------------------------------------------------------------------------
# AC3: the under-50-row fallback fires and is counted
# ---------------------------------------------------------------------------


def test_under_min_rows_group_falls_back_to_the_month_pool_and_is_counted():
    """Induced: the real data's smallest group is 128 rows, so this is synthetic."""
    frame = _frame()
    tiny = _frame(years=(2018,), months=(2,), partitions=(9,), rows_per_group=12, seed=4)
    frame = pd.concat([frame, tiny], ignore_index=True)

    calibrators = C.fit_calibrators(frame, scope=1)
    reports = {
        (row["calendar_month"], row["partition_id"]): row
        for row in calibrators.group_reports
    }
    tiny_report = reports[(2, 9)]
    assert tiny_report["n_fit_rows"] == 12
    assert tiny_report["is_fallback"] is True
    assert tiny_report["reason"] == "min_rows"
    assert (2, 9) not in calibrators.groups

    summary = C.fallback_summary(calibrators, {})
    assert summary["fit_fallback_reason_counts"]["min_rows"] == 1
    assert summary["groups_routed_to_month_pool"] >= 1

    applied = C.apply_calibrators(tiny, calibrators)
    assert (applied[C.ROUTE_COLUMN] == C.ROUTE_MONTH_POOLED).all()
    assert (applied[C.ROUTE_REASON_COLUMN] == "min_rows").all()
    # It must equal the month pool's own output, not some other calibrator's.
    expected = calibrators.month_pooled[2].transform(
        tiny[C.PROB_COLUMN].to_numpy(dtype=float)
    )
    assert np.allclose(applied[C.CALIBRATED_COLUMN].to_numpy(), expected)


def test_min_rows_threshold_is_the_step3_abstention_floor():
    assert C.MIN_GROUP_ROWS == 50


def test_single_class_group_routes_to_the_month_pool_with_the_platt_failure_recorded():
    frame = _frame()
    flat = _frame(years=(2018,), months=(6,), partitions=(8,), rows_per_group=200, seed=5)
    flat["y_true"] = 0
    calibrators = C.fit_calibrators(pd.concat([frame, flat], ignore_index=True), scope=1)
    report = next(
        row
        for row in calibrators.group_reports
        if (row["calendar_month"], row["partition_id"]) == (6, 8)
    )
    assert report["reason"] == "single_class"
    assert report["is_fallback"] is True
    assert "at least 2 classes" in str(report["platt_attempt_error"])
    assert (6, 8) not in calibrators.groups


def test_too_few_distinct_scores_falls_back_to_platt():
    frame = _frame()
    flat = _frame(years=(2018,), months=(10,), partitions=(7,), rows_per_group=200, seed=6)
    flat[C.PROB_COLUMN] = 0.4
    flat.loc[flat.index[:100], "y_true"] = 1
    flat.loc[flat.index[100:], "y_true"] = 0
    calibrators = C.fit_calibrators(pd.concat([frame, flat], ignore_index=True), scope=1)
    report = next(
        row
        for row in calibrators.group_reports
        if (row["calendar_month"], row["partition_id"]) == (10, 7)
    )
    assert report["reason"] == "platt_too_few_distinct_probabilities"
    assert report["calibrator_kind"] == C.PLATT
    assert calibrators.groups[(10, 7)].kind == C.PLATT
    # A constant score with a 50% base rate must calibrate to ~0.5.
    out = calibrators.groups[(10, 7)].transform(np.array([0.4]))
    assert out[0] == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------------------
# AC3: grouping is (month, partition_id), never partition_id alone
# ---------------------------------------------------------------------------


def test_grouping_uses_the_month_key_not_partition_alone():
    """Same partition id, opposite month-specific score->rate maps.

    If the month key were dropped, the two months would be pooled into one
    calibrator and both would be pulled towards the average, so the calibrated
    output for a 0.8 score could not stay on both sides of it.
    """
    rng = np.random.default_rng(7)
    records = []
    for year in (2018, 2019):
        for month, exponent in ((2, 3.0), (6, 0.25)):
            probability = np.round(rng.random(600), 2)
            truth = (rng.random(600) < np.clip(probability**exponent, 0, 1)).astype(int)
            for index in range(600):
                records.append(
                    {
                        "FEWSNET_admin_code": index + 1000 * month + 100000 * year,
                        "month_start": pd.Timestamp(year=year, month=month, day=1),
                        "partition_id": 3,  # one partition id, two months
                        "y_true": int(truth[index]),
                        "y_prob_partitioned": float(probability[index]),
                    }
                )
    frame = pd.DataFrame(records)
    calibrators = C.fit_calibrators(frame, scope=1)

    assert set(calibrators.groups) == {(2, 3), (6, 3)}
    february = calibrators.groups[(2, 3)].transform(np.array([0.8]))[0]
    june = calibrators.groups[(6, 3)].transform(np.array([0.8]))[0]
    # February's rate at 0.8 is 0.8**3 = 0.51; June's is 0.8**0.25 = 0.95.
    assert february < 0.65 < june
    assert june - february > 0.25

    # And the routing honours the month: identical score + partition, different
    # month, must produce different calibrated values.
    probe = pd.DataFrame(
        {
            "month_start": [pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-01")],
            "partition_id": [3, 3],
            C.PROB_COLUMN: [0.8, 0.8],
        }
    )
    applied = C.apply_calibrators(probe, calibrators)
    assert applied[C.CALIBRATED_COLUMN].iloc[0] != applied[C.CALIBRATED_COLUMN].iloc[1]
    assert applied[C.CALIBRATED_COLUMN].iloc[0] == pytest.approx(february)
    assert applied[C.CALIBRATED_COLUMN].iloc[1] == pytest.approx(june)


def test_group_key_is_declared_as_month_and_partition_in_the_frozen_artifact():
    calibrators = C.fit_calibrators(_frame(), scope=1)
    payload = calibrators.to_dict()
    assert payload["group_key"] == ["calendar_month", "partition_id"]
    assert all("|" in key for key in payload["groups"])


# ---------------------------------------------------------------------------
# AC3: the frozen calibrator is deterministic
# ---------------------------------------------------------------------------


def test_refitting_the_same_input_reproduces_an_identical_hash():
    frame = _frame()
    first = C.fit_calibrators(frame, scope=1)
    second = C.fit_calibrators(frame.copy(), scope=1)
    assert first.digest() == second.digest()
    assert first.canonical_json() == second.canonical_json()


def test_row_order_does_not_change_the_frozen_hash():
    frame = _frame()
    shuffled = frame.sample(frac=1.0, random_state=11).reset_index(drop=True)
    assert C.fit_calibrators(shuffled, scope=1).digest() == C.fit_calibrators(
        frame, scope=1
    ).digest()


def test_a_different_fit_input_changes_the_hash():
    frame = _frame()
    altered = frame.copy()
    altered.loc[altered.index[:500], "y_true"] = 1 - altered.loc[
        altered.index[:500], "y_true"
    ]
    assert C.fit_calibrators(frame, scope=1).digest() != C.fit_calibrators(
        altered, scope=1
    ).digest()


def test_frozen_artifact_round_trips_through_json_unchanged(tmp_path):
    calibrators = C.fit_calibrators(_frame(), scope=2)
    path = tmp_path / "calibrators.json"
    path.write_text(calibrators.canonical_json(), encoding="utf-8")
    reloaded = C.CalibratorSet.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert reloaded.digest() == calibrators.digest()
    probe = np.round(np.linspace(0, 1, 101), 2)
    months = np.full(probe.shape, 2)
    partitions = np.zeros(probe.shape, dtype=int)
    original, _, _ = calibrators.transform(probe, months, partitions)
    restored, _, _ = reloaded.transform(probe, months, partitions)
    assert np.array_equal(original, restored)


def test_isotonic_knots_reproduce_sklearn_predict_exactly():
    """The frozen artifact stores knots, not an estimator; they must agree."""
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(13)
    probability = np.round(rng.random(800), 2)
    truth = (rng.random(800) < probability).astype(int)
    ours = C.fit_isotonic(probability, truth)
    theirs = IsotonicRegression(
        y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
    ).fit(probability, truth)
    probe = np.round(rng.random(300), 2)
    assert np.allclose(ours.transform(probe), theirs.predict(probe), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Applying the calibrator is a pure transform: it cannot see y_true
# ---------------------------------------------------------------------------


def test_transform_signature_admits_no_label_argument():
    parameters = list(
        inspect.signature(C.CalibratorSet.transform).parameters
    )
    assert parameters == ["self", "prob", "calendar_month", "partition_id"]


def test_apply_and_transform_source_never_reference_a_label():
    """Static check: no truth/label identifier appears in the apply path."""
    source = Path(C.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned = {"y_true", "truth", "label", "TRUTH_COLUMN", "labels"}
    for node in ast.walk(tree):
        is_apply = isinstance(node, ast.FunctionDef) and node.name == "apply_calibrators"
        is_transform = (
            isinstance(node, ast.FunctionDef)
            and node.name == "transform"
            and "prob" in [arg.arg for arg in node.args.args]
        )
        if not (is_apply or is_transform):
            continue
        names = {
            child.id for child in ast.walk(node) if isinstance(child, ast.Name)
        } | {
            child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)
        } | {
            child.value
            for child in ast.walk(node)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        }
        leaked = banned & names
        assert not leaked, f"{node.name} references label(s) {sorted(leaked)}"


def test_calibrated_values_are_identical_with_and_without_a_truth_column():
    calibrators = C.fit_calibrators(_frame(), scope=1)
    target = _frame(years=(2020,), seed=17)
    with_truth = C.apply_calibrators(target, calibrators)
    without_truth = C.apply_calibrators(target.drop(columns=["y_true"]), calibrators)
    assert np.array_equal(
        with_truth[C.CALIBRATED_COLUMN].to_numpy(),
        without_truth[C.CALIBRATED_COLUMN].to_numpy(),
    )


def test_corrupting_the_labels_does_not_change_the_calibrated_output():
    calibrators = C.fit_calibrators(_frame(), scope=1)
    target = _frame(years=(2020,), seed=19)
    flipped = target.copy()
    flipped["y_true"] = 1 - flipped["y_true"]
    assert np.array_equal(
        C.apply_calibrators(target, calibrators)[C.CALIBRATED_COLUMN].to_numpy(),
        C.apply_calibrators(flipped, calibrators)[C.CALIBRATED_COLUMN].to_numpy(),
    )


# ---------------------------------------------------------------------------
# Apply-time routing and output-domain guarantees
# ---------------------------------------------------------------------------


def test_group_absent_from_the_fit_window_routes_to_the_month_pool():
    """The real ``partition_id == -1`` unmapped bucket takes this path."""
    calibrators = C.fit_calibrators(_frame(), scope=1)
    unseen = pd.DataFrame(
        {
            "month_start": [pd.Timestamp("2020-02-01")] * 3,
            "partition_id": [C.UNMAPPED_PARTITION_ID] * 3,
            C.PROB_COLUMN: [0.1, 0.5, 0.9],
        }
    )
    applied = C.apply_calibrators(unseen, calibrators)
    assert (applied[C.ROUTE_COLUMN] == C.ROUTE_MONTH_POOLED).all()
    assert (applied[C.ROUTE_REASON_COLUMN] == "group_absent_from_fit_window").all()
    expected = calibrators.month_pooled[2].transform(np.array([0.1, 0.5, 0.9]))
    assert np.allclose(applied[C.CALIBRATED_COLUMN].to_numpy(), expected)


def test_an_unseen_calendar_month_halts_rather_than_guessing():
    calibrators = C.fit_calibrators(_frame(months=(2, 6)), scope=1)
    probe = pd.DataFrame(
        {
            "month_start": [pd.Timestamp("2020-10-01")],
            "partition_id": [0],
            C.PROB_COLUMN: [0.5],
        }
    )
    with pytest.raises(CalibrationContractError):
        C.apply_calibrators(probe, calibrators)


def test_calibrated_output_stays_in_the_unit_interval_and_is_monotone():
    calibrators = C.fit_calibrators(_frame(), scope=1)
    grid = np.round(np.linspace(0, 1, 101), 2)
    probe = pd.DataFrame(
        {
            "month_start": [pd.Timestamp("2020-06-01")] * len(grid),
            "partition_id": [0] * len(grid),
            C.PROB_COLUMN: grid,
        }
    )
    values = C.apply_calibrators(probe, calibrators)[C.CALIBRATED_COLUMN].to_numpy()
    assert values.min() >= 0.0 and values.max() <= 1.0
    assert np.all(np.diff(values) >= -1e-12), "isotonic output must be non-decreasing"


def test_apply_rejects_a_score_outside_the_unit_interval():
    calibrators = C.fit_calibrators(_frame(), scope=1)
    probe = pd.DataFrame(
        {
            "month_start": [pd.Timestamp("2020-06-01")],
            "partition_id": [0],
            C.PROB_COLUMN: [1.5],
        }
    )
    with pytest.raises(CalibrationContractError):
        C.apply_calibrators(probe, calibrators)


def test_in_sample_overall_mean_matches_the_base_rate():
    """Sanity: isotonic preserves the mean, so in-sample overall gap ~ 0."""
    frame = _frame()
    calibrators = C.fit_calibrators(frame, scope=1)
    applied = C.apply_calibrators(frame, calibrators)
    assert applied[C.CALIBRATED_COLUMN].mean() == pytest.approx(
        frame["y_true"].mean(), abs=0.005
    )


# ---------------------------------------------------------------------------
# Reliability reporting
# ---------------------------------------------------------------------------


def test_persistence_group_reliability_reports_each_group_and_the_tolerance():
    frame = _frame()
    frame["persistence"] = (frame.index % 2).astype(int)
    rows = C.persistence_group_reliability(
        frame,
        C.PROB_COLUMN,
        scope=1,
        window="fit_2018_2019",
        stage="pre",
        sample_status="in_sample_for_calibrator",
    )
    labels = [row["persistence_group"] for row in rows]
    assert labels == ["0", "1", "all"]
    for row in rows:
        assert row["tolerance"] == 0.05
        assert row["abs_gap"] == pytest.approx(
            abs(row["mean_predicted_probability"] - row["observed_crisis_rate"])
        )
        assert row["within_tolerance"] == (row["abs_gap"] <= 0.05)


def test_reliability_bins_come_from_the_repository_implementation():
    from scripts.paper_artifacts.analyze_georf_probability_uncertainty import (
        reliability_bins,
    )

    frame = _frame()
    ours = C.reliability_bin_table(frame, C.PROB_COLUMN)
    theirs = reliability_bins(frame, C.PROB_COLUMN, n_bins=10)
    pd.testing.assert_frame_equal(ours, theirs)
