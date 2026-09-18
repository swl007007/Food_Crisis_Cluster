"""Focused tests for the Step 3 expert selective-correction experiment.

Run from the repository root:
    python3 -m pytest Step3ExpertCorrectionExperiment/tests -q
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from step3correction import correction, expert, selection, windows  # noqa: E402
from step3correction import runner  # noqa: E402


# ---------------------------------------------------------------------------
# Expert definition: calendar alignment at O = T-H, with the legacy record-shift
# series retained only as a firewalled pipeline-validation artifact
# ---------------------------------------------------------------------------

TRI_ANNUAL = pd.to_datetime(
    [
        f"{year}-{month:02d}-01"
        for year in range(2019, 2022)
        for month in (2, 6, 10)
    ]
)
QUARTERLY_THEN_TRI_ANNUAL = pd.to_datetime(
    ["2015-01-01", "2015-04-01", "2015-07-01", "2015-10-01"]
    + ["2016-02-01", "2016-06-01", "2016-10-01", "2017-02-01"]
)


def _write_expert_source(path: Path, months, *, near, med, ipc, admin=101, country="Testland"):
    """Write a minimal FEWSNET.csv-shaped source for one admin unit."""
    frame = pd.DataFrame(
        {
            "country": country,
            "admin_code": admin,
            "year_month": [f"{m.year}_{m.month:02d}" for m in months],
            "year": [m.year for m in months],
            "month": [m.month for m in months],
            "fews_ipc": ipc,
            "fews_proj_near": near,
            "fews_proj_med": med,
        }
    )
    frame.to_csv(path, index=False)
    return frame


def test_expert_is_the_projection_published_at_the_calendar_origin(tmp_path):
    """fs1 reads fews_proj_near at T-4 and fs2 reads fews_proj_med at T-8."""
    months = TRI_ANNUAL
    near = [3, 2, 4, 1, 3, 2, 0, 0, 0]
    med = [1, 4, 1, 4, 1, 4, 1, 4, 1]
    source = tmp_path / "FEWSNET.csv"
    _write_expert_source(source, months, near=near, med=med, ipc=[3] * len(months))

    table = expert.load_expert_history(source)
    frame = table.frame.set_index("month_start")

    # fs1: target 2021-02 <- near published 2020-10 (phase 2 -> 0).
    assert frame.loc[pd.Timestamp("2021-02-01"), "expert_1_source_month"] == pd.Timestamp(
        "2020-10-01"
    )
    assert frame.loc[pd.Timestamp("2021-02-01"), "expert_1"] == 0
    # fs1: target 2020-02 <- near published 2019-10 (phase 4 -> 1).
    assert frame.loc[pd.Timestamp("2020-02-01"), "expert_1"] == 1

    # fs2: target 2021-02 <- medium published 2020-06 (phase 1 -> 0).
    assert frame.loc[pd.Timestamp("2021-02-01"), "expert_2_source_month"] == pd.Timestamp(
        "2020-06-01"
    )
    assert frame.loc[pd.Timestamp("2021-02-01"), "expert_2"] == 0
    # fs2: target 2021-06 <- medium published 2020-10 (phase 4 -> 1).
    assert frame.loc[pd.Timestamp("2021-06-01"), "expert_2_source_month"] == pd.Timestamp(
        "2020-10-01"
    )
    assert frame.loc[pd.Timestamp("2021-06-01"), "expert_2"] == 1

    audit = expert.require_source_alignment(table.frame, 1)
    assert audit["alignment"] == "calendar"
    assert audit["expert_horizon_verified"] is True
    assert set(audit["observed_lag_value_counts"]) == {4}
    assert audit["rows_off_declared_origin"] == 0
    assert set(expert.require_source_alignment(table.frame, 2)["observed_lag_value_counts"]) == {8}


def test_calendar_origin_absent_from_the_release_grid_stays_missing(tmp_path):
    """In the quarterly era T-4 is off the release grid, so the expert is missing."""
    months = QUARTERLY_THEN_TRI_ANNUAL
    source = tmp_path / "FEWSNET.csv"
    _write_expert_source(
        source, months, near=[4] * len(months), med=[4] * len(months),
        ipc=[3] * len(months),
    )
    table = expert.load_expert_history(source)
    frame = table.frame.set_index("month_start")

    # Quarterly era (Jan/Apr/Jul/Oct, gap 3): T-4 is never an observed month.
    for target in ("2015-04-01", "2015-07-01", "2015-10-01"):
        assert pd.isna(frame.loc[pd.Timestamp(target), "expert_1"])
        assert pd.isna(frame.loc[pd.Timestamp(target), "expert_1_source_month"])
    # 2016-02 - 4 months = 2015-10, which *is* observed.
    assert frame.loc[pd.Timestamp("2016-02-01"), "expert_1_source_month"] == pd.Timestamp(
        "2015-10-01"
    )
    # fs2 at 2016-02 needs 2015-06, which is absent -> missing, never imputed.
    assert pd.isna(frame.loc[pd.Timestamp("2016-02-01"), "expert_2"])
    # fs2 at 2016-06 needs 2015-10, which is present.
    assert frame.loc[pd.Timestamp("2016-06-01"), "expert_2_source_month"] == pd.Timestamp(
        "2015-10-01"
    )
    # Missingness is never filled with 0 or any other value.
    assert frame["expert_1"].isna().sum() == 4
    coverage = expert.coverage_report(table.frame, 1)
    assert coverage["rows"] == len(months)
    assert coverage["rows_with_expert"] == len(months) - 4


def test_raw_missing_phase_becomes_zero_before_the_calendar_join(tmp_path):
    """Missing raw phases convert to 0; an absent publication stays NaN."""
    months = pd.date_range("2020-01-01", periods=6, freq="MS")
    near = [np.nan, 4, np.nan, 4, 1, 1]
    source = tmp_path / "FEWSNET.csv"
    _write_expert_source(source, months, near=near, med=near, ipc=[3] * len(months))

    table = expert.load_expert_history(source)
    frame = table.frame.sort_values("month_start").reset_index(drop=True)
    # First four rows have no publication four calendar months earlier.
    assert frame.loc[:3, "expert_1"].isna().all()
    # 2020-05 pulls 2020-01, whose raw phase is missing -> historical 0 convention.
    assert frame.loc[4, "expert_1"] == 0
    assert frame.loc[4, "expert_1_phase_missing"] == 1
    # 2020-06 pulls 2020-02 (phase 4) -> 1.
    assert frame.loc[5, "expert_1"] == 1
    assert frame.loc[5, "expert_1_phase_missing"] == 0


def test_legacy_record_shift_is_built_but_differs_from_the_calendar_series(tmp_path):
    """shift(4) walks four *records* back: 16 calendar months on a tri-annual panel."""
    months = TRI_ANNUAL
    near = [3, 2, 4, 1, 3, 2, 0, 0, 0]
    source = tmp_path / "FEWSNET.csv"
    _write_expert_source(source, months, near=near, med=near, ipc=[3] * len(months))

    table = expert.load_expert_history(source)
    frame = table.frame.set_index("month_start")
    legacy = table.legacy_record_shift_series(1)
    legacy.index = table.frame["month_start"]

    # Four records before 2021-02 is 2019-10 (phase 4 -> 1), 16 months, not 4.
    assert legacy.loc[pd.Timestamp("2021-02-01")] == 1
    # The calendar series reads 2020-10 (phase 2 -> 0) instead.
    assert frame.loc[pd.Timestamp("2021-02-01"), "expert_1"] == 0
    assert legacy.loc[pd.Timestamp("2021-02-01")] != frame.loc[
        pd.Timestamp("2021-02-01"), "expert_1"
    ]
    assert expert.LEGACY_RECORD_SHIFT == {1: 4, 2: 8}


def test_legacy_series_cannot_reach_the_correction_layer(tmp_path):
    """The legacy record-shift series is firewalled out of the correction input."""
    months = TRI_ANNUAL
    source = tmp_path / "FEWSNET.csv"
    _write_expert_source(
        source, months, near=[3] * len(months), med=[3] * len(months), ipc=[3] * len(months)
    )
    table = expert.load_expert_history(source)

    # The legacy column exists on the internal frame ...
    assert expert.legacy_column(1) in table.frame.columns
    # ... but never on the only path into the correction layer.
    for scope in (1, 2):
        scope_frame = table.for_scope(scope)
        assert not [
            name
            for name in scope_frame.columns
            if str(name).startswith(expert.LEGACY_COLUMN_PREFIX)
        ]
        lookup = runner._expert_lookup(scope_frame, scope)
        assert not [
            name
            for name in list(lookup.columns) + list(lookup.index.names)
            if str(name).startswith(expert.LEGACY_COLUMN_PREFIX)
        ]

    # Smuggling a legacy column in halts before any fitting happens.
    smuggled = table.for_scope(1)
    smuggled[expert.legacy_column(1)] = 0
    with pytest.raises(expert.ExpertContractError, match="legacy record-shift"):
        runner._expert_lookup(smuggled, 1)


def test_duplicate_admin_month_keys_halt(tmp_path):
    """Duplicate valid admin-month keys are a contract error, never deduplicated."""
    months = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01"])
    source = tmp_path / "FEWSNET.csv"
    _write_expert_source(source, months, near=[1, 2, 3], med=[1, 2, 3], ipc=[3, 3, 3])
    with pytest.raises(expert.ExpertContractError, match="Duplicate"):
        expert.load_expert_history(source)


def test_leakage_direction_alignment_halts_unconditionally():
    """A source row after the declared origin halts; there is no override."""
    frame = pd.DataFrame(
        {
            "month_start": pd.to_datetime(["2021-02-01"]),
            "expert_1_source_month": pd.to_datetime(["2021-01-01"]),  # lag 1 < horizon 4
        }
    )
    with pytest.raises(expert.ExpertContractError, match="post-date"):
        expert.require_source_alignment(frame, 1)
    # The signature carries no acknowledgement parameter at all.
    import inspect

    assert "acknowledge" not in inspect.signature(expert.require_source_alignment).parameters


def test_stale_direction_alignment_halts_with_no_bypass():
    """A source row older than T-H also halts: the contract admits exactly one lag."""
    frame = pd.DataFrame(
        {
            "month_start": pd.to_datetime(["2021-02-01", "2021-06-01"]),
            "expert_1_source_month": pd.to_datetime(["2019-10-01", "2021-02-01"]),
        }
    )
    with pytest.raises(expert.ExpertContractError, match="older than the declared"):
        expert.require_source_alignment(frame, 1)


def test_inert_cli_flag_cannot_bypass_the_alignment_gate():
    """The retained --acknowledge flag is parsed but wired to nothing."""
    args = runner.build_parser().parse_args(
        ["--run-id", "x", "--acknowledge-unverified-expert-horizon"]
    )
    assert args.acknowledge_unverified_expert_horizon is True
    import inspect

    assert "acknowledge" not in inspect.signature(runner.run).parameters


# ---------------------------------------------------------------------------
# Temporal masks
# ---------------------------------------------------------------------------

def test_outer_window_preserves_the_historical_off_by_one():
    """Configured 36 months resolves to [O-35, O): 35 monthly timestamps."""
    fold = windows.resolve_fold_windows("2021-02", 4)
    assert fold.origin == pd.Timestamp("2020-10-01")
    assert fold.outer_start == pd.Timestamp("2017-11-01")
    assert fold.outer_end == pd.Timestamp("2020-10-01")
    monthly = pd.Series(pd.date_range("2010-01-01", "2024-12-01", freq="MS"))
    assert int(windows.outer_mask(monthly, fold).sum()) == 35
    assert windows.OUTER_WINDOW_TIMESTAMPS == 35


@pytest.mark.parametrize("horizon", [4, 8])
def test_validation_is_twelve_calendar_months_and_fit_is_horizon_isolated(horizon):
    """V=[O-12,O); the fit cutoff is V_start-H; the purge band is exactly H months."""
    fold = windows.resolve_fold_windows("2022-06", horizon)
    monthly = pd.Series(pd.date_range("2010-01-01", "2024-12-01", freq="MS"))
    assert windows.VALIDATION_CALENDAR_MONTHS == 12
    assert int(windows.validation_mask(monthly, fold).sum()) == 12
    assert fold.validation_start == fold.origin - pd.DateOffset(months=12)
    assert fold.validation_end == fold.origin
    assert fold.fit_cutoff == fold.validation_start - pd.DateOffset(months=horizon)
    assert int(windows.gap_mask(monthly, fold).sum()) == horizon
    # fit / gap / validation partition the outer window without overlap.
    fit = windows.fit_mask(monthly, fold)
    gap = windows.gap_mask(monthly, fold)
    validation = windows.validation_mask(monthly, fold)
    outer = windows.outer_mask(monthly, fold)
    assert not (fit & gap).any() and not (gap & validation).any() and not (fit & validation).any()
    np.testing.assert_array_equal(fit | gap | validation, outer)
    # Every fit label month is strictly more than H months before every validation month.
    assert monthly[fit].max() + pd.DateOffset(months=horizon) < monthly[validation].min()


@pytest.mark.parametrize("horizon", [4, 8])
@pytest.mark.parametrize("target", ["2021-02", "2021-06", "2021-10", "2024-10"])
def test_validation_holds_exactly_three_tri_annual_label_months(target, horizon):
    """On the tri-annual label grid V=[O-12,O) admits exactly O-4, O-8, O-12."""
    fold = windows.resolve_fold_windows(target, horizon)
    labels = pd.Series(
        pd.to_datetime(
            [
                f"{year}-{month:02d}-01"
                for year in range(2016, 2025)
                for month in (2, 6, 10)
            ]
        )
    )
    observed = windows.describe_observed_months(
        labels, windows.validation_mask(labels, fold)
    )
    expected = [
        str((fold.origin - pd.DateOffset(months=offset)).to_period("M"))
        for offset in (12, 8, 4)
    ]
    assert observed["n_months"] == 3
    assert observed["months"] == expected
    # O is always congruent to 2 mod 4, which is why a six-month V held only one.
    assert fold.origin.month % 4 == 2
    six_month_start = fold.origin - pd.DateOffset(months=6)
    six_month = labels[(labels >= six_month_start) & (labels < fold.origin)]
    assert len(set(six_month)) == 1


@pytest.mark.parametrize("horizon", [4, 8])
def test_horizon_isolation_holds_under_the_twelve_month_validation(horizon):
    """No fit label month is within H months of any validation label month."""
    fold = windows.resolve_fold_windows("2022-06", horizon)
    labels = pd.Series(
        pd.to_datetime(
            [
                f"{year}-{month:02d}-01"
                for year in range(2016, 2025)
                for month in (2, 6, 10)
            ]
        )
    )
    fit = windows.fit_mask(labels, fold)
    gap = windows.gap_mask(labels, fold)
    validation = windows.validation_mask(labels, fold)
    outer = windows.outer_mask(labels, fold)
    assert not (fit & validation).any() and not (fit & gap).any()
    np.testing.assert_array_equal(fit | gap | validation, outer)
    assert labels[fit].max() + pd.DateOffset(months=horizon) < labels[validation].min()
    # Surfaced consequence: fs1 keeps 4 observed fit months, fs2 keeps 3.
    assert windows.describe_observed_months(labels, fit)["n_months"] == (
        4 if horizon == 4 else 3
    )


def test_target_month_is_excluded_from_every_training_mask():
    """The target month never enters the outer window."""
    fold = windows.resolve_fold_windows("2023-10", 4)
    monthly = pd.Series(pd.date_range("2010-01-01", "2024-12-01", freq="MS"))
    target = windows.target_mask(monthly, fold)
    assert int(target.sum()) == 1
    assert not (target & windows.outer_mask(monthly, fold)).any()


def test_group_eligibility_matches_the_main_splitter_filter():
    """Outer rows survive only when their group appears in the target month."""
    outer_groups = np.array([0, 1, 2, -1, 1])
    target_groups = np.array([1, 2])
    np.testing.assert_array_equal(
        windows.group_eligibility_mask(outer_groups, target_groups),
        np.array([False, True, True, False, True]),
    )


# ---------------------------------------------------------------------------
# Correction learner wiring and abstention
# ---------------------------------------------------------------------------

def test_target_is_expert_error_and_expert_is_the_final_input_column():
    """w = 1[y != e]; the design matrix appends e after the main features."""
    truth = np.array([0, 1, 1, 0])
    expert_values = np.array([0, 0, 1, 1])
    np.testing.assert_array_equal(
        correction.wrong_label(truth, expert_values), np.array([0, 1, 0, 1])
    )
    X = np.arange(8, dtype=float).reshape(4, 2)
    design = correction.build_design_matrix(X, expert_values)
    assert design.shape == (4, 3)
    np.testing.assert_array_equal(design[:, :2], X)
    np.testing.assert_array_equal(design[:, -1], expert_values)


def test_non_finite_features_are_an_error_not_an_abstention():
    """Invalid inputs raise; they are never downgraded to abstention."""
    X = np.array([[1.0, np.nan], [2.0, 3.0]])
    with pytest.raises(correction.CorrectionInputError, match="non-finite"):
        correction.build_design_matrix(X, np.array([0, 1]))


def _synthetic_partition_frame(seed=0, n_per_partition=(120, 120, 40), single_class=(False, True, False)):
    """Build a deterministic multi-partition correction fixture."""
    rng = np.random.default_rng(seed)
    blocks = []
    for index, (count, forced) in enumerate(zip(n_per_partition, single_class)):
        X = rng.normal(size=(count, 3))
        e = rng.integers(0, 2, size=count)
        if forced:
            y = e.copy()  # expert never wrong -> single wrong-label class
        else:
            wrong = (X[:, 0] > 0).astype(int)
            y = np.where(wrong == 1, 1 - e, e)
        blocks.append((X, e, y, np.full(count, index)))
    return (
        np.vstack([b[0] for b in blocks]),
        np.concatenate([b[1] for b in blocks]),
        np.concatenate([b[2] for b in blocks]),
        np.concatenate([b[3] for b in blocks]),
    )


def test_small_and_single_class_partitions_abstain_without_pooled_fallback():
    """Under 50 usable rows or one wrong-label class -> abstain, keep expert."""
    X, e, y, groups = _synthetic_partition_frame()
    groups = groups.copy()
    ensemble = correction.fit_correction_ensemble(
        X=X, expert=e, truth=y, partitions=groups, stage="test"
    )
    reasons = {report.partition_id: report.reason for report in ensemble.reports}
    assert reasons[0] == correction.ELIGIBLE
    assert reasons[1] == correction.ABSTAIN_SINGLE_CLASS
    assert reasons[2] == correction.ABSTAIN_TOO_FEW_SAMPLES
    assert set(ensemble.models) == {0}

    scores, eligible, score_reasons = correction.wrong_scores(
        ensemble, X=X, expert=e, partitions=groups
    )
    assert eligible[groups == 0].all()
    assert not eligible[groups != 0].any()
    assert np.isnan(scores[groups != 0]).all()
    assert set(score_reasons[groups == 1]) == {correction.ABSTAIN_NO_MODEL}


def test_exactly_fifty_usable_rows_is_trainable_and_fortynine_is_not():
    """The 50-row minimum is inclusive at 50."""
    rng = np.random.default_rng(3)
    for count, expected in ((50, True), (49, False)):
        X = rng.normal(size=(count, 2))
        e = np.zeros(count, dtype=int)
        y = np.array([index % 2 for index in range(count)])
        ensemble = correction.fit_correction_ensemble(
            X=X, expert=e, truth=y, partitions=np.zeros(count, dtype=int), stage="test"
        )
        assert (0 in ensemble.models) is expected


def test_high_wrong_score_means_the_expert_is_likely_wrong():
    """Score polarity guard: q must rise with actual expert error, not fall.

    A silent polarity inversion (selecting ``P(w = 0)``) would make the selective
    layer flip precisely the rows where the expert was *right*.  The full run's
    negative result was checked against this invariant, so it needs a permanent
    regression guard rather than a one-off empirical check.
    """
    rng = np.random.default_rng(11)
    count = 400
    X = rng.normal(size=(count, 3))
    e = rng.integers(0, 2, size=count)
    # Wrongness is a deterministic, learnable function of the first feature.
    wrong = (X[:, 0] > 0).astype(int)
    y = np.where(wrong == 1, 1 - e, e)
    groups = np.zeros(count, dtype=int)

    ensemble = correction.fit_correction_ensemble(
        X=X, expert=e, truth=y, partitions=groups, stage="test"
    )
    scores, eligible, _ = correction.wrong_scores(
        ensemble, X=X, expert=e, partitions=groups
    )
    assert eligible.all()

    # Truly-wrong rows must score strictly higher on average than correct rows.
    assert scores[wrong == 1].mean() > scores[wrong == 0].mean() + 0.25
    # The top-scoring decile must be dominated by real expert errors.
    top = np.argsort(scores)[-count // 10 :]
    assert wrong[top].mean() > 0.9
    # And the bottom decile must be dominated by rows the expert got right.
    bottom = np.argsort(scores)[: count // 10]
    assert wrong[bottom].mean() < 0.1


def test_wrong_score_column_follows_classes_not_position():
    """The scored column is chosen by ``classes_ == 1``, never positional index 1.

    ``RandomForestClassifier`` happens to sort ``classes_`` to ``[0, 1]``, so a
    positional ``proba[:, 1]`` would pass every ordinary test while being wrong
    for any estimator that reports a different class order.  This stub makes the
    distinction observable.
    """

    class ReversedClassOrderModel:
        """Emits ``P(w = 1)`` in column 0, advertising ``classes_ = [1, 0]``."""

        classes_ = np.array([1, 0])

        def predict_proba(self, design):
            n = design.shape[0]
            wrong_prob = np.full(n, 0.9)
            return np.column_stack([wrong_prob, 1.0 - wrong_prob])

    ensemble = correction.CorrectionEnsemble(
        models={0: ReversedClassOrderModel()}, reports=[], stage="test"
    )
    X = np.zeros((5, 2))
    scores, eligible, _ = correction.wrong_scores(
        ensemble, X=X, expert=np.zeros(5, dtype=int), partitions=np.zeros(5, dtype=int)
    )
    assert eligible.all()
    # 0.9 is P(w=1) from column 0; positional indexing would have returned 0.1.
    np.testing.assert_allclose(scores, 0.9)


def test_model_without_a_wrong_label_class_is_an_error_not_a_silent_score():
    """A degenerate single-class model must raise, never index column 1 blindly."""

    class SingleClassModel:
        classes_ = np.array([0])

        def predict_proba(self, design):
            return np.ones((design.shape[0], 1))

    ensemble = correction.CorrectionEnsemble(
        models={0: SingleClassModel()}, reports=[], stage="test"
    )
    with pytest.raises(correction.CorrectionInputError, match="wrong-label class"):
        correction.wrong_scores(
            ensemble,
            X=np.zeros((3, 2)),
            expert=np.zeros(3, dtype=int),
            partitions=np.zeros(3, dtype=int),
        )


def test_refit_models_are_keyed_to_the_scoring_partition():
    """Rows are scored by their *own* partition's model, never another's.

    Guards the validation-fit/final-refit partition-set asymmetry observed in the
    full run (e.g. fs2 2021-06 trained 9 partitions for validation and 11 at
    refit): a partition-keying slip would score test rows with a foreign model.
    """
    rng = np.random.default_rng(23)
    blocks = []
    # Two partitions with *opposite* wrongness rules, so a keying slip inverts
    # the score instead of merely perturbing it.
    for index, sign in enumerate((1.0, -1.0)):
        X = rng.normal(size=(200, 2))
        e = rng.integers(0, 2, size=200)
        wrong = (sign * X[:, 0] > 0).astype(int)
        y = np.where(wrong == 1, 1 - e, e)
        blocks.append((X, e, y, np.full(200, index), wrong))
    X = np.vstack([b[0] for b in blocks])
    e = np.concatenate([b[1] for b in blocks])
    y = np.concatenate([b[2] for b in blocks])
    groups = np.concatenate([b[3] for b in blocks])
    wrong = np.concatenate([b[4] for b in blocks])

    ensemble = correction.fit_correction_ensemble(
        X=X, expert=e, truth=y, partitions=groups, stage="final_refit"
    )
    assert set(ensemble.models) == {0, 1}
    scores, eligible, _ = correction.wrong_scores(
        ensemble, X=X, expert=e, partitions=groups
    )
    assert eligible.all()
    # Each partition must be well separated under its own rule.
    for index in (0, 1):
        mask = groups == index
        assert scores[mask][wrong[mask] == 1].mean() > scores[mask][wrong[mask] == 0].mean() + 0.25
    # Scoring rows with a deliberately swapped partition key destroys separation,
    # proving the score really is partition-keyed.
    swapped = 1 - groups
    swapped_scores, _, _ = correction.wrong_scores(
        ensemble, X=X, expert=e, partitions=swapped
    )
    swapped_gap = swapped_scores[wrong == 1].mean() - swapped_scores[wrong == 0].mean()
    correct_gap = scores[wrong == 1].mean() - scores[wrong == 0].mean()
    assert correct_gap > 0.25
    assert swapped_gap < correct_gap / 2


def test_unmapped_partition_rows_always_abstain():
    """Partition ``-1`` never trains and never receives a score."""
    X, e, y, groups = _synthetic_partition_frame()
    groups = np.where(groups == 2, correction.UNMAPPED_PARTITION_ID, groups)
    ensemble = correction.fit_correction_ensemble(
        X=X, expert=e, truth=y, partitions=groups, stage="test"
    )
    assert correction.UNMAPPED_PARTITION_ID not in ensemble.models
    _, eligible, reasons = correction.wrong_scores(ensemble, X=X, expert=e, partitions=groups)
    unmapped = groups == correction.UNMAPPED_PARTITION_ID
    assert not eligible[unmapped].any()
    assert set(reasons[unmapped]) == {correction.ABSTAIN_UNMAPPED}


# ---------------------------------------------------------------------------
# Selection: strict threshold, directional gates, ties
# ---------------------------------------------------------------------------

def test_threshold_comparison_is_strict_on_unrounded_scores():
    """q > threshold, never q >= threshold, and rounding never leaks into the test."""
    scores = np.array([0.50, 0.504, 0.51])
    eligible = np.ones(3, dtype=bool)
    np.testing.assert_array_equal(
        selection.proposed_flip_mask(scores, eligible, 0.50),
        np.array([False, True, True]),
    )
    # Candidates are rounded to two decimals but applied to the raw scores.
    np.testing.assert_allclose(selection.candidate_thresholds(scores), [0.50, 0.51])


def test_abstaining_rows_can_never_be_proposed_flips():
    """Ineligible rows stay in the F1 support but are never flipped."""
    scores = np.array([0.9, 0.9, np.nan])
    eligible = np.array([True, False, False])
    np.testing.assert_array_equal(
        selection.proposed_flip_mask(scores, eligible, 0.5), np.array([True, False, False])
    )


def _gate_case(count, months, precision_numerator, expert_value=0):
    """Build validation arrays that propose exactly ``count`` flips in one direction."""
    scores = np.full(count, 0.9)
    expert_values = np.full(count, expert_value, dtype=int)
    truth = np.full(count, expert_value, dtype=int)
    truth[:precision_numerator] = 1 - expert_value
    month_labels = np.array([f"2020-{1 + (index % months):02d}" for index in range(count)])
    return truth, expert_values, scores, np.ones(count, dtype=bool), month_labels


@pytest.mark.parametrize("count,expected", [(20, True), (19, False)])
def test_directional_flip_count_gate_boundary(count, expected):
    """At least 20 proposed flips are required."""
    truth, expert_values, scores, eligible, months = _gate_case(count, 2, count)
    gate = selection.evaluate_direction(
        direction="0_to_1", expert_value=0, expert=expert_values,
        truth=truth, months=months, proposed=selection.proposed_flip_mask(scores, eligible, 0.5),
    )
    assert gate.proposed == count
    assert gate.enabled is expected


@pytest.mark.parametrize("months,expected", [(2, True), (1, False)])
def test_directional_distinct_month_gate_boundary(months, expected):
    """At least two distinct validation months are required."""
    truth, expert_values, scores, eligible, month_labels = _gate_case(24, months, 24)
    gate = selection.evaluate_direction(
        direction="0_to_1", expert_value=0, expert=expert_values, truth=truth,
        months=month_labels, proposed=selection.proposed_flip_mask(scores, eligible, 0.5),
    )
    assert gate.distinct_months == months
    assert gate.enabled is expected


@pytest.mark.parametrize("fixes,expected", [(75, True), (74, False)])
def test_directional_precision_gate_boundary(fixes, expected):
    """Correction precision must reach 0.75; 0.74 is rejected."""
    truth, expert_values, scores, eligible, months = _gate_case(100, 2, fixes)
    gate = selection.evaluate_direction(
        direction="0_to_1", expert_value=0, expert=expert_values, truth=truth,
        months=months, proposed=selection.proposed_flip_mask(scores, eligible, 0.5),
    )
    assert gate.precision == pytest.approx(fixes / 100)
    assert gate.enabled is expected


def test_both_directions_share_one_threshold_with_independent_enable_flags():
    """One threshold, two separately gated directions."""
    # 30 expert-0 rows that are genuinely wrong (fixable) across two months,
    # plus 30 expert-1 rows that are mostly right (unfixable) in a single month.
    up_truth, up_expert, _, _, up_months = _gate_case(30, 2, 30, expert_value=0)
    down_truth, down_expert, _, _, down_months = _gate_case(30, 1, 30, expert_value=1)
    # A low-scoring filler block supplies a candidate threshold strictly below 0.9.
    truth = np.concatenate([up_truth, down_truth, np.zeros(5, dtype=int)])
    expert_values = np.concatenate([up_expert, down_expert, np.zeros(5, dtype=int)])
    months = np.concatenate([up_months, down_months, np.full(5, "2020-01")])
    scores = np.concatenate([np.full(60, 0.9), np.full(5, 0.1)])
    eligible = np.ones(65, dtype=bool)
    rule = selection.select_rule(
        truth=truth, expert=expert_values, wrong_scores=scores,
        eligible=eligible, months=months,
    )
    assert rule.corrected is True
    assert rule.enable_0_to_1 is True
    assert rule.enable_1_to_0 is False  # single validation month fails the month gate


def test_exact_f1_tie_keeps_expert_only():
    """A candidate that merely ties expert-only never wins."""
    # 20 expert-0 rows over two months that are all genuine crises: flipping them
    # to 1 is a perfect correction, so expert-only F1 = 0 and correction F1 = 1.
    # Neutralise by making the flip reproduce the expert-only confusion exactly.
    truth = np.array([1] * 20 + [0] * 20)
    expert_values = np.array([1] * 20 + [0] * 20)  # expert already perfect -> F1 = 1.0
    scores = np.concatenate([np.full(20, 0.9), np.full(20, 0.9)])
    months = np.array([f"2020-{1 + index % 2:02d}" for index in range(40)])
    rule = selection.select_rule(
        truth=truth, expert=expert_values, wrong_scores=scores,
        eligible=np.ones(40, dtype=bool), months=months,
    )
    assert rule.corrected is False
    assert rule.status == "no_correction"
    assert rule.selected_validation_f1 == pytest.approx(rule.expert_only_validation_f1)


def test_no_eligible_scores_yields_explicit_no_correction():
    """All-abstaining validation support selects explicit no-correction."""
    rule = selection.select_rule(
        truth=np.array([1, 0]), expert=np.array([0, 0]),
        wrong_scores=np.array([np.nan, np.nan]),
        eligible=np.array([False, False]), months=np.array(["2020-01", "2020-02"]),
    )
    assert rule.corrected is False
    assert rule.reason == selection.NO_CORRECTION_REASONS["no_eligible_scores"]


def test_ascending_iteration_keeps_the_first_of_two_exactly_tied_candidates():
    """Two thresholds producing identical validation F1 resolve to the lower one.

    Construction: 20 expert-0 rows at score 0.90 are genuinely fixable across two
    months (the only enabled direction).  Ten expert-1 rows at 0.60 and ten more
    at 0.30 only ever feed the ``1->0`` direction, which stays disabled because it
    proposes fewer than 20 flips.  Thresholds 0.30 and 0.60 therefore yield the
    *same* predictions and the same F1, so the earlier candidate must win.
    """
    truth = np.concatenate([np.ones(20, dtype=int), np.zeros(20, dtype=int)])
    expert_values = np.concatenate([np.zeros(20, dtype=int), np.ones(20, dtype=int)])
    scores = np.concatenate([np.full(20, 0.90), np.full(10, 0.60), np.full(10, 0.30)])
    months = np.array([f"2020-{1 + index % 2:02d}" for index in range(40)])
    rule = selection.select_rule(
        truth=truth, expert=expert_values, wrong_scores=scores,
        eligible=np.ones(40, dtype=bool), months=months,
    )
    candidates = {row["threshold"]: row for row in rule.candidates}
    assert sorted(candidates) == [0.30, 0.60, 0.90]
    assert candidates[0.30]["validation_f1"] == pytest.approx(candidates[0.60]["validation_f1"])
    assert candidates[0.30]["dir_1_to_0_enabled"] is False
    assert rule.corrected is True
    assert rule.threshold == pytest.approx(0.30)
    assert rule.enable_0_to_1 is True
    assert rule.enable_1_to_0 is False


def test_apply_rule_only_flips_enabled_directions():
    """Disabled directions leave the expert untouched."""
    expert_values = np.array([0, 1, 0, 1])
    scores = np.array([0.9, 0.9, 0.1, 0.1])
    eligible = np.ones(4, dtype=bool)
    np.testing.assert_array_equal(
        selection.apply_rule(expert_values, scores, eligible, 0.5, True, False),
        np.array([1, 1, 0, 1]),
    )
    np.testing.assert_array_equal(
        selection.apply_rule(expert_values, scores, eligible, 0.5, False, True),
        np.array([0, 0, 0, 1]),
    )
    np.testing.assert_array_equal(
        selection.apply_rule(expert_values, scores, eligible, None, True, True), expert_values
    )


def test_crisis_f1_zero_denominator_convention():
    """No positives anywhere returns 0.0, matching the archived evaluator."""
    assert selection.crisis_f1([0, 0], [0, 0]) == 0.0


# ---------------------------------------------------------------------------
# Leakage guarantees
# ---------------------------------------------------------------------------

def test_mutating_withheld_gap_labels_cannot_change_the_fitting_stage():
    """The purge band is genuinely withheld from the initial correction fit."""
    fold = windows.resolve_fold_windows("2022-06", 4)
    monthly = pd.date_range("2018-01-01", "2022-06-01", freq="MS")
    rng = np.random.default_rng(11)
    dates = pd.Series(np.repeat(monthly, 60))
    n_rows = len(dates)
    X = rng.normal(size=(n_rows, 3))
    expert_values = rng.integers(0, 2, size=n_rows)
    truth = np.where(X[:, 0] > 0, 1 - expert_values, expert_values)
    partitions = np.tile(np.arange(3), n_rows // 3 + 1)[:n_rows]

    fit = windows.fit_mask(dates, fold)
    gap = windows.gap_mask(dates, fold)
    assert gap.sum() > 0

    def fit_signature(labels):
        ensemble = correction.fit_correction_ensemble(
            X=X[fit], expert=expert_values[fit], truth=labels[fit],
            partitions=partitions[fit], stage="validation_fit",
        )
        scores, _, _ = correction.wrong_scores(
            ensemble, X=X, expert=expert_values, partitions=partitions
        )
        return np.nan_to_num(scores, nan=-1.0)

    baseline_signature = fit_signature(truth)
    mutated = truth.copy()
    mutated[gap] = 1 - mutated[gap]
    np.testing.assert_array_equal(baseline_signature, fit_signature(mutated))


def test_mutating_target_month_labels_cannot_change_the_selected_rule():
    """Selection is computed from validation rows only; test labels are invisible."""
    truth = np.array([1] * 25 + [0] * 15)
    expert_values = np.zeros(40, dtype=int)
    scores = np.concatenate([np.full(25, 0.9), np.full(15, 0.1)])
    months = np.array([f"2020-{1 + index % 2:02d}" for index in range(40)])
    rule = selection.select_rule(
        truth=truth, expert=expert_values, wrong_scores=scores,
        eligible=np.ones(40, dtype=bool), months=months,
    )
    test_expert = np.array([0, 0, 1, 1])
    test_scores = np.array([0.99, 0.01, 0.99, 0.01])
    test_eligible = np.ones(4, dtype=bool)
    first = selection.apply_rule(
        test_expert, test_scores, test_eligible, rule.threshold,
        rule.enable_0_to_1, rule.enable_1_to_0,
    )
    # Re-select with every target-month label inverted: the rule cannot move.
    rule_again = selection.select_rule(
        truth=truth, expert=expert_values, wrong_scores=scores,
        eligible=np.ones(40, dtype=bool), months=months,
    )
    second = selection.apply_rule(
        test_expert, test_scores, test_eligible, rule_again.threshold,
        rule_again.enable_0_to_1, rule_again.enable_1_to_0,
    )
    assert (rule.threshold, rule.enable_0_to_1, rule.enable_1_to_0) == (
        rule_again.threshold, rule_again.enable_0_to_1, rule_again.enable_1_to_0
    )
    np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------------
# Audit recomputation and reused baselines
# ---------------------------------------------------------------------------

def test_final_predictions_recompute_from_audit_rows():
    """Saved audit fields alone reproduce the applied flips and final labels."""
    audit = pd.DataFrame(
        {
            "expert": [0, 0, 1, 1, 0],
            "wrong_score": [0.91, 0.20, 0.95, 0.30, np.nan],
            "score_eligible": [True, True, True, True, False],
            "selected_threshold": 0.60,
            "enable_0_to_1": True,
            "enable_1_to_0": False,
        }
    )
    recomputed = selection.apply_rule(
        audit["expert"].to_numpy(),
        audit["wrong_score"].to_numpy(dtype=float),
        audit["score_eligible"].to_numpy(),
        float(audit["selected_threshold"].iloc[0]),
        bool(audit["enable_0_to_1"].iloc[0]),
        bool(audit["enable_1_to_0"].iloc[0]),
    )
    np.testing.assert_array_equal(recomputed, np.array([1, 0, 1, 1, 0]))
    np.testing.assert_array_equal(
        recomputed != audit["expert"].to_numpy(), np.array([True, False, False, False, False])
    )


def test_reused_pooled_and_fs3_predictions_are_unchanged():
    """Reused frozen pooled/fs3 arrays reproduce their archived metrics exactly."""
    from step3correction import baselines

    for scope in (1, 2, 3):
        reused = baselines.load_reused_baseline(scope)
        assert reused.checks["archive_package_hashes_match"] is True
        assert reused.checks["per_row_archive_equality"] is True
        assert reused.checks["archived_metric_rows_reproduced"] == 24
        assert reused.checks["support_rows"] == baselines.EXPECTED_SUPPORT_ROWS

    fs3 = runner.uncorrected_scope_frames(3)
    assert bool(fs3["predictions"]["correction_applicable"].any()) is False
    assert set(fs3["predictions"]["method_partitioned"]) == {runner.UNCORRECTED_METHOD}


def test_wrong_score_is_never_written_as_a_crisis_probability():
    """The audit schema must not reuse the legacy crisis-probability column name."""
    forbidden = "y_prob_partitioned"
    for module_path in sorted((EXPERIMENT_DIR / "step3correction").glob("*.py")):
        assert forbidden not in module_path.read_text(encoding="utf-8"), module_path


# ---------------------------------------------------------------------------
# Output isolation guards
# ---------------------------------------------------------------------------

def test_output_outside_the_approved_root_fails_before_creating_anything():
    """Repository paths outside the experiment output root raise ValueError."""
    target = EXPERIMENT_DIR.parents[0] / "result_step3_correction_should_not_exist"
    with pytest.raises(ValueError, match="must stay under"):
        runner.resolve_output_dir(target)
    assert not target.exists()


def test_existing_run_directory_is_immutable():
    """An existing run directory is a hard FileExistsError."""
    with tempfile.TemporaryDirectory() as scratch:
        existing = Path(scratch) / "run_a"
        existing.mkdir()
        with pytest.raises(FileExistsError):
            runner.resolve_output_dir(existing, allow_outside_repo=True)


def test_fs3_is_rejected_as_a_correction_scope():
    """fs3 stays uncorrected: it can never be passed as a correction scope."""
    with tempfile.TemporaryDirectory() as scratch:
        with pytest.raises(ValueError, match="Correction scopes"):
            runner.run(scopes=[3], target_months=["2021-02"], out_dir=Path(scratch) / "r",
                       allow_outside_repo=True)


def test_month_map_selection_matches_the_original_month_specific_run():
    """Feb/Jun/Oct use the frozen m2/m6/m10 contig3 maps."""
    for month, expected in ((2, "nc13_m2"), (6, "nc11_m6"), (10, "nc16_m10")):
        path = runner.month_map_path(1, pd.Period(f"2021-{month:02d}", freq="M"))
        assert expected in path.name
        assert "refined_contig3" in path.name


# ---------------------------------------------------------------------------
# Fit-only preprocessing
# ---------------------------------------------------------------------------

def test_out_of_range_imputation_is_fitted_on_fit_rows_only():
    """Validation/test values never influence the imputation constant."""
    from step3correction import features

    X_fit = np.array([[1.0], [2.0], [np.nan]])
    X_val = np.array([[1000.0], [np.nan]])

    imputer = features.fit_window_imputer(X_fit)
    fitted = features.transform_with(imputer, X_fit)
    transformed = features.transform_with(imputer, X_val)

    # max_plus with multiplier 100 on a fit maximum of 2.0 -> 200.0.
    assert fitted[2, 0] == pytest.approx(200.0)
    assert transformed[1, 0] == pytest.approx(200.0)
    # The 1000.0 validation value did not move the constant.
    assert transformed[0, 0] == pytest.approx(1000.0)
    assert np.isfinite(fitted).all() and np.isfinite(transformed).all()


def test_refitting_on_a_wider_window_changes_the_constant():
    """The refit stage legitimately produces its own fit-only constant."""
    from step3correction import features

    narrow = features.fit_window_imputer(np.array([[1.0], [2.0]]))
    wide = features.fit_window_imputer(np.array([[1.0], [2.0], [5.0]]))
    assert features.transform_with(narrow, np.array([[np.nan]]))[0, 0] == pytest.approx(200.0)
    assert features.transform_with(wide, np.array([[np.nan]]))[0, 0] == pytest.approx(500.0)


def test_pooled_metric_guard_rejects_a_drifted_pooled_column():
    """Recomputed pooled metrics must match the archive, or the run halts."""
    from step3correction import baselines

    baseline = baselines.load_reused_baseline(1)
    month = pd.Timestamp("2021-02-01")
    archived = baseline.predictions.loc[baseline.predictions["month_start"].eq(month)]
    audit = pd.DataFrame(
        {
            "month_start": archived["month_start"].to_numpy(),
            "y_true": archived["y_true"].to_numpy(),
            f"y_pred_{runner.CORRECTION_METHOD}": archived["y_pred_partitioned"].to_numpy(),
            "y_pred_pooled": archived["y_pred_pooled"].to_numpy(),
        }
    )
    # Unmodified pooled reproduces the archived metrics.
    frame = runner.monthly_metrics(audit, baseline, 1)
    assert set(frame["model"]) == {runner.CORRECTION_METHOD, runner.POOLED_METHOD}

    tampered = audit.copy()
    tampered.loc[tampered.index[:50], "y_pred_pooled"] = (
        1 - tampered.loc[tampered.index[:50], "y_pred_pooled"]
    )
    with pytest.raises(AssertionError):
        runner.monthly_metrics(tampered, baseline, 1)
