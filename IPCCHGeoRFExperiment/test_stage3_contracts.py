"""Compact contract checks for the Stage 3 rolling four-arm forecasts.

Hand-computable fixtures and tiny synthetic fits only: no pinned source, no
geopandas, no 122-fold run. What is checked here is the logic the runner adds
around the released helpers — the 36 calendar-month window, own-origin
historical features, the frozen map, imputer discipline, model routing,
persistence, the strict p1 > .5 rule and the single-partition degeneracy.

The released helpers themselves are additionally exercised *inside* every real
fold: :func:`run_pipeline.fit_fold` re-predicts each routed subset and refuses
to continue unless the probabilities it was handed match the route it recorded.
One test below also loads those real helpers when an extracted baseline happens
to be available, and skips otherwise.

Run with the preferred interpreter::

    python3.12.exe -B IPCCHGeoRFExperiment/test_stage3_contracts.py
"""

from __future__ import annotations

import contextlib
import io
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_data as pdata  # noqa: E402
import run_pipeline as runner  # noqa: E402


class SkipTest(Exception):
    pass


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def ordinal(label: str) -> int:
    return int(pdata.month_ordinal(int(label[:4]), int(label[5:7])))


class RecordingImputer:
    """Records the rows it was fitted on; transform reuses the stored fills."""

    instances: list = []

    def __init__(self):
        self.fit_calls = 0
        self.fit_rows = None
        self.fills = None
        self.impute_values_ = {}
        self.column_stats_ = {}
        RecordingImputer.instances.append(self)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.fit_calls += 1
        self.fit_rows = X.copy()
        with np.errstate(all="ignore"):
            maxima = np.nanmax(np.where(np.isnan(X), -np.inf, X), axis=0)
        maxima = np.where(np.isfinite(maxima), maxima, 0.0)
        self.fills = np.where(maxima == 0, 100.0, maxima * 100.0)
        for index in range(X.shape[1]):
            self.impute_values_[index] = float(self.fills[index])
            self.column_stats_[index] = {
                "min": float(np.nanmin(X[:, index])) if np.isfinite(X[:, index]).any() else np.nan,
                "max": float(maxima[index]),
                "has_missing": bool(np.isnan(X[:, index]).any()),
            }
        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float64, copy=True)
        missing = np.isnan(X)
        if missing.any():
            X[missing] = np.broadcast_to(self.fills, X.shape)[missing]
        return X


class RecordingXGB:
    """Stands in for XGBClassifier; records exactly what matrix it was given."""

    instances: list = []

    def __init__(self, probability=0.75):
        self.fit_X = None
        self.fit_y = None
        self.probability = probability
        self.classes_ = np.array([0, 1])
        RecordingXGB.instances.append(self)

    def fit(self, X, y, sample_weight=None):
        self.fit_X = np.array(X, dtype=np.float64, copy=True)
        self.fit_y = np.array(y, copy=True)
        self.sample_weight = sample_weight
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.probability, dtype=np.float64)
        return np.column_stack([1.0 - p, p])


def _predict_class1(model, X):
    """Mirror of the released ``predict_class1_probability``."""
    proba = model.predict_proba(X)
    classes = np.asarray(getattr(model, "classes_", []))
    if proba.shape[1] == 1:
        only = int(classes[0]) if classes.size else 0
        return np.ones(len(X)) if only == 1 else np.zeros(len(X))
    index = np.where(classes == 1)[0]
    return proba[:, index[0]].astype(float) if index.size else np.zeros(len(X))


def _predict_partitioned(models, pooled, X, group):
    """Mirror of the released ``predict_partitioned_probability``."""
    out = np.zeros(len(X), dtype=float)
    handled = np.zeros(len(X), dtype=bool)
    for pid, model in models.items():
        mask = group == pid
        if mask.sum() == 0:
            continue
        source = model if model is not None else pooled
        out[mask] = _predict_class1(source, X[mask])
        handled[mask] = True
    if (~handled).any():
        out[~handled] = _predict_class1(pooled, X[~handled])
    return out


def simple_helpers(min_rows: int = 2) -> runner.Stage3Helpers:
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

    def train_pooled(X, y, lower_model="rf"):
        model = RandomForestClassifier(**runner.STAGE3_RF_PARAMS)
        model.fit(X, y)
        return model

    def train_partitioned(X, y, group, lower_model="rf", min_samples=min_rows):
        models = {}
        for pid in np.unique(group):
            pid = int(pid)
            if pid < 0:
                continue
            mask = group == pid
            if mask.sum() < min_samples or np.unique(y[mask]).size < 2:
                models[pid] = None
                continue
            model = RandomForestClassifier(**runner.STAGE3_RF_PARAMS)
            model.fit(X[mask], y[mask])
            models[pid] = model
        return models

    return runner.Stage3Helpers(
        train_pooled=train_pooled,
        train_partitioned=train_partitioned,
        predict_partitioned_probability=_predict_partitioned,
        predict_class1_probability=_predict_class1,
        min_partition_rows=min_rows,
        source="test fixture mirroring the released helpers",
    )


def panel_fixture(rows, assignments, columns=3):
    """``rows`` are ``(admin, 'YYYY-MM' target, horizon, label, feature)``.

    ``assignments`` maps admin -> (partition_code, branch_id, assignment_source).
    Persistence is derived from the rows' own labels, so the fixture is a
    self-consistent little world.
    """
    admin = np.array([r[0] for r in rows], dtype=np.int64)
    target_ord = np.array([ordinal(r[1]) for r in rows], dtype=np.int64)
    horizon = np.array([r[2] for r in rows], dtype=np.int64)
    origin_ord = target_ord - horizon
    y = np.array([r[3] for r in rows], dtype=np.int64)
    base = np.array([r[4] for r in rows], dtype=np.float64)

    X = np.column_stack([base, base * 2.0 + 1.0, np.arange(len(rows), dtype=np.float64)])
    X = X[:, :columns]

    partition_code = np.array([assignments[a][0] for a in admin], dtype=np.int64)
    branch_id = np.array([assignments[a][1] for a in admin], dtype=object)
    source = np.array([assignments[a][2] for a in admin], dtype=object)

    labels = (
        pd.DataFrame({"admin_code": admin, "month_ord": target_ord, "label": y})
        .drop_duplicates(["admin_code", "month_ord"])
        .sort_values(["admin_code", "month_ord"])
    )
    valid = pd.DataFrame(
        {
            "admin_code": labels["admin_code"].to_numpy(),
            "year": labels["month_ord"].to_numpy() // 12,
            "month": labels["month_ord"].to_numpy() % 12 + 1,
            "ipcch_food_crisis": labels["label"].to_numpy(),
        }
    )
    persistence = runner.persistence_lookup(valid, admin, origin_ord)

    metadata = pd.DataFrame(
        {
            "admin_code": admin,
            "country_en": ["Testland"] * len(rows),
            "ISO3": ["TST"] * len(rows),
            "target_month": pdata.month_label(target_ord),
            "origin_month": pdata.month_label(origin_ord),
            "ipcch_food_crisis": y,
            "last_observed_label_month": pdata.month_label(persistence["source_ord"]),
            "last_observed_crisis_month": "",
            "assignment_source": source,
            "donor_admin_code": -1,
            "donor_distance_km": np.nan,
            "branch_id": branch_id,
            "partition_code": partition_code,
        }
    )
    branch_by_code = {}
    for code, branch, _ in assignments.values():
        if code >= 0:
            branch_by_code[int(code)] = branch
    return runner.Stage3Panel(
        metadata=metadata,
        X=X,
        y=y,
        admin=admin,
        target_ord=target_ord,
        origin_ord=origin_ord,
        horizon=horizon,
        partition_code=partition_code,
        branch_id=branch_id,
        persistence=persistence,
        branch_by_code=branch_by_code,
        feature_columns=tuple(f"f{i}" for i in range(columns)),
        audit={},
    )


def monthly_rows(admin, start: str, count: int, horizon: int, label_cycle=(0, 1),
                 feature_base: float = 1.0):
    """``count`` consecutive monthly outcomes for one area at one horizon."""
    first = ordinal(start)
    return [
        (
            admin,
            pdata.month_label([first + k])[0],
            horizon,
            label_cycle[k % len(label_cycle)],
            feature_base + k,
        )
        for k in range(count)
    ]


def run_fold(fold, panel, helpers=None, min_rows=2, probability=0.75):
    RecordingImputer.instances.clear()
    RecordingXGB.instances.clear()
    helpers = helpers or simple_helpers(min_rows)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        return runner.fit_fold(
            fold,
            panel,
            helpers,
            RecordingImputer,
            lambda: RecordingXGB(probability),
        )


# --------------------------------------------------------------------------
# R4 fold schedule
# --------------------------------------------------------------------------


def test_main_schedule_is_exactly_122_folds_with_the_approved_endpoints():
    folds = runner.build_fold_schedule()
    main = [f for f in folds if f.period == runner.PERIOD_MAIN]
    assert len(main) == len(folds) == runner.EXPECTED_MAIN_FOLDS == 122

    counts = {h: sum(1 for f in main if f.horizon == h) for h in (1, 3, 6, 12)}
    assert counts == {1: 35, 3: 33, 6: 30, 12: 24}

    first = {f.horizon: f.target_month for f in main if f.target_month == min(
        g.target_month for g in main if g.horizon == f.horizon)}
    assert first == {1: "2023-02", 3: "2023-04", 6: "2023-07", 12: "2024-01"}
    assert {f.target_month for f in main if f.horizon == 12} >= {"2025-12"}
    assert max(f.target_month for f in main) == "2025-12"


def test_every_fold_satisfies_o_equals_t_minus_h_and_clears_the_2022_cutoff():
    cutoff = ordinal(runner.PARTITION_INFORMATION_CUTOFF)
    for fold in runner.build_fold_schedule():
        assert fold.target_ord - fold.origin_ord == fold.horizon
        assert fold.origin_ord > cutoff
        assert fold.origin_month >= "2023-01"


def test_the_training_window_is_36_calendar_months_ending_at_the_origin():
    fold = next(
        f for f in runner.build_fold_schedule()
        if f.horizon == 6 and f.target_month == "2024-01"
    )
    # R3's worked example: h6 target 2024-01 has O = 2023-07.
    assert fold.origin_month == "2023-07"
    assert pdata.month_label([fold.window_start_ord])[0] == "2020-08"
    assert fold.origin_ord - fold.window_start_ord + 1 == 36


def test_partial_2026_months_form_a_separate_opportunistic_period():
    available = {
        1: {ordinal("2026-01"), ordinal("2026-03"), ordinal("2025-06")},
        3: set(),
        6: set(),
        12: {ordinal("2026-02")},
    }
    folds = runner.build_fold_schedule(available_target_ords=available)
    partial = [f for f in folds if f.period == runner.PERIOD_PARTIAL]
    assert sorted((f.horizon, f.target_month) for f in partial) == [
        (1, "2026-01"), (1, "2026-03"), (12, "2026-02")
    ]
    # 2025-06 is inside the main window, so it is NOT re-added as partial.
    assert sum(1 for f in folds if f.period == runner.PERIOD_MAIN) == 122
    assert len({f.fold_id for f in folds}) == len(folds)


def test_a_schedule_reaching_the_partition_cutoff_is_refused():
    try:
        runner.build_fold_schedule(schedule={1: ("2023-01", "2023-01")})
    except runner.PipelineError as error:
        assert "partition cutoff" in str(error)
    else:
        raise AssertionError("an origin of 2022-12 must be refused")


# --------------------------------------------------------------------------
# Window selection, own-origin features, leakage
# --------------------------------------------------------------------------


def test_the_fold_trains_on_exactly_the_36_calendar_month_window():
    # O = 2023-07 for h6/2024-01, so the window is 2020-08..2023-07 inclusive.
    rows = []
    for label_month in ("2020-07", "2020-08", "2023-07", "2023-08", "2023-12"):
        rows.append((10, label_month, 6, 1, 1.0))
        rows.append((11, label_month, 6, 0, 2.0))
    rows += [(10, "2024-01", 6, 1, 3.0), (11, "2024-01", 6, 0, 4.0)]
    assignments = {10: (0, "", "learned"), 11: (0, "", "learned")}
    panel = panel_fixture(rows, assignments)
    fold = runner.Fold("h6_2024-01", runner.PERIOD_MAIN, 6,
                       ordinal("2024-01"), ordinal("2023-07"))

    outcome = run_fold(fold, panel)
    keys = outcome["training_keys"]
    months = sorted(set(keys["target_month"]))
    # 2020-07 is one month too early; 2023-08 and 2023-12 are after the origin;
    # 2024-01 is the target itself.
    assert months == ["2020-08", "2023-07"]
    assert outcome["record"]["train_rows"] == 4
    assert outcome["record"]["test_rows"] == 2


def test_each_training_row_keeps_its_own_origin_not_the_fold_origin():
    rows = monthly_rows(10, "2022-01", 30, horizon=6) + monthly_rows(
        11, "2022-01", 30, horizon=6, feature_base=50.0
    )
    rows += [(10, "2024-08", 6, 1, 9.0), (11, "2024-08", 6, 0, 9.0)]
    assignments = {10: (0, "", "learned"), 11: (0, "", "learned")}
    panel = panel_fixture(rows, assignments)
    fold = runner.Fold("h6_2024-08", runner.PERIOD_MAIN, 6,
                       ordinal("2024-08"), ordinal("2024-02"))
    outcome = run_fold(fold, panel)
    keys = outcome["training_keys"]
    origins = pd.Series([ordinal(m) for m in keys["origin_month"]])
    targets = pd.Series([ordinal(m) for m in keys["target_month"]])
    # Each row's own T-H, all strictly before the fit origin.
    assert ((targets - origins) == 6).all()
    assert origins.max() < fold.origin_ord
    assert targets.max() <= fold.origin_ord


def test_no_target_month_or_post_origin_label_enters_training():
    rows = monthly_rows(10, "2022-06", 24, horizon=1) + monthly_rows(
        11, "2022-06", 24, horizon=1, feature_base=7.0
    )
    rows += [(10, "2024-09", 1, 1, 2.0), (11, "2024-09", 1, 0, 2.0)]
    assignments = {10: (0, "", "learned"), 11: (0, "", "learned")}
    panel = panel_fixture(rows, assignments)
    fold = runner.Fold("h1_2024-09", runner.PERIOD_MAIN, 1,
                       ordinal("2024-09"), ordinal("2024-08"))
    outcome = run_fold(fold, panel)
    train = set(zip(outcome["training_keys"]["admin_code"],
                    outcome["training_keys"]["target_month"]))
    assert ("2024-09" not in {m for _, m in train})
    assert max(ordinal(m) for _, m in train) <= fold.origin_ord


def test_an_empty_test_month_is_zero_support_and_not_a_failure():
    rows = monthly_rows(10, "2022-01", 24, horizon=1) + monthly_rows(
        11, "2022-01", 24, horizon=1, feature_base=5.0
    )
    assignments = {10: (0, "", "learned"), 11: (0, "", "learned")}
    panel = panel_fixture(rows, assignments)
    fold = runner.Fold("h1_2024-03", runner.PERIOD_MAIN, 1,
                       ordinal("2024-03"), ordinal("2024-02"))
    outcome = run_fold(fold, panel)
    record = outcome["record"]
    assert record["test_rows"] == 0
    assert record["fitted"] is False
    assert record["skip_reason"]
    assert outcome["rows"] is None
    assert record["train_rows"] > 0


def test_an_empty_global_training_pool_is_a_reported_stop():
    rows = [(10, "2025-06", 1, 1, 1.0), (11, "2025-06", 1, 0, 2.0)]
    assignments = {10: (0, "", "learned"), 11: (0, "", "learned")}
    panel = panel_fixture(rows, assignments)
    fold = runner.Fold("h1_2025-06", runner.PERIOD_MAIN, 1,
                       ordinal("2025-06"), ordinal("2025-05"))
    try:
        run_fold(fold, panel)
    except runner.PipelineError as error:
        assert "empty training pool" in str(error)
    else:
        raise AssertionError("an empty training pool must stop the run")


# --------------------------------------------------------------------------
# Q6c imputation discipline
# --------------------------------------------------------------------------


def test_one_imputer_per_fold_is_fitted_on_training_rows_only():
    rows = monthly_rows(10, "2022-01", 24, horizon=1) + monthly_rows(
        11, "2022-01", 24, horizon=1, feature_base=3.0
    )
    rows += [(10, "2024-02", 1, 1, 10_000.0), (11, "2024-02", 1, 0, 20_000.0)]
    assignments = {10: (0, "", "learned"), 11: (0, "", "learned")}
    panel = panel_fixture(rows, assignments)
    fold = runner.Fold("h1_2024-02", runner.PERIOD_MAIN, 1,
                       ordinal("2024-02"), ordinal("2024-01"))
    outcome = run_fold(fold, panel)

    assert len(RecordingImputer.instances) == 1
    imputer = RecordingImputer.instances[0]
    assert imputer.fit_calls == 1
    assert len(imputer.fit_rows) == outcome["record"]["train_rows"]
    # The held-out extrema (10_000 / 20_000) must not have reached the fit.
    assert imputer.fit_rows[:, 0].max() < 10_000.0
    fills = outcome["imputer_fills"]
    assert set(fills["fold_id"]) == {"h1_2024-02"}
    assert len(fills) == len(panel.feature_columns)


def test_xgb_sees_the_pre_imputation_matrix_and_rf_sees_the_imputed_one():
    rows = monthly_rows(10, "2022-01", 24, horizon=1) + monthly_rows(
        11, "2022-01", 24, horizon=1, feature_base=3.0
    )
    rows += [(10, "2024-02", 1, 1, 4.0), (11, "2024-02", 1, 0, 5.0)]
    assignments = {10: (0, "", "learned"), 11: (0, "", "learned")}
    panel = panel_fixture(rows, assignments)
    panel.X[0, 0] = np.nan  # a training row with a genuine gap
    fold = runner.Fold("h1_2024-02", runner.PERIOD_MAIN, 1,
                       ordinal("2024-02"), ordinal("2024-01"))
    run_fold(fold, panel)

    fitted_xgb = RecordingXGB.instances[0]
    assert np.isnan(fitted_xgb.fit_X).any(), "XGB must receive native NaN (Q6c)"
    assert fitted_xgb.sample_weight is None, "Q7b: unit row weights only"
    # The RF side is imputed: the recording imputer leaves no NaN behind.
    imputed = RecordingImputer.instances[0].transform(panel.X[:2])
    assert np.isfinite(imputed).all()


# --------------------------------------------------------------------------
# Routing and the partitioned/pooled identity
# --------------------------------------------------------------------------


def test_routes_name_the_local_model_or_the_reason_for_the_pooled_fallback():
    models = {0: object(), 1: None}
    group = np.array([0, 1, 2, -1])
    route, reason = runner.local_model_routes(models, group, min_rows=50)
    assert list(route) == ["partition:0", runner.ROUTE_POOLED,
                           runner.ROUTE_POOLED, runner.ROUTE_POOLED]
    assert reason[0] == ""
    assert "fewer than 50 training rows" in reason[1]
    assert "unseen" in reason[2]
    assert "unassigned" in reason[3]


def test_an_unassigned_test_row_is_identical_in_both_rf_arms():
    rows = monthly_rows(10, "2022-01", 24, horizon=1) + monthly_rows(
        11, "2022-01", 24, horizon=1, feature_base=3.0
    ) + monthly_rows(12, "2022-01", 24, horizon=1, feature_base=9.0)
    rows += [
        (10, "2024-02", 1, 1, 4.0),
        (11, "2024-02", 1, 0, 5.0),
        (12, "2024-02", 1, 1, 6.0),
    ]
    assignments = {
        10: (0, "", "learned"),
        11: (0, "", "learned"),
        12: (-1, "", "unresolved"),
    }
    panel = panel_fixture(rows, assignments)
    fold = runner.Fold("h1_2024-02", runner.PERIOD_MAIN, 1,
                       ordinal("2024-02"), ordinal("2024-01"))
    outcome = run_fold(fold, panel)
    frame = outcome["rows"]
    unassigned = frame[frame["partition_code"] == -1]
    assert len(unassigned) == 1
    assert (unassigned["prob_partitioned_rf"] == unassigned["prob_pooled_rf"]).all()
    assert (unassigned["model_route"] == runner.ROUTE_POOLED).all()
    assert "unassigned" in unassigned["model_fallback_reason"].iloc[0]
    # The unassigned rows were excluded from the local model's training pool, so
    # the local forest saw a SMALLER pool than the pooled forest. The arms are
    # therefore allowed to disagree on locally-served rows, and the run must not
    # assert any identity for them: it completes, and only records the measurement.
    record = outcome["record"]
    assert record["unassigned_train_rows_excluded_from_local"] == 24
    assert record["local_model_covers_entire_pool"] is False
    assert record["train_areas_in_local_pools"] == 2
    assert record["train_areas"] == 3
    assert isinstance(record["partitioned_identical_to_pooled"], bool)


def test_every_written_row_carries_the_frozen_maps_provenance():
    rows = monthly_rows(10, "2022-01", 24, horizon=1) + monthly_rows(
        11, "2022-01", 24, horizon=1, feature_base=3.0
    ) + monthly_rows(12, "2022-01", 24, horizon=1, feature_base=9.0)
    rows += [
        (10, "2024-02", 1, 1, 4.0),
        (11, "2024-02", 1, 0, 5.0),
        (12, "2024-02", 1, 1, 6.0),
    ]
    assignments = {
        10: (0, "", "learned"),
        11: (0, "", "learned"),
        12: (-1, "", "unresolved"),
    }
    panel = panel_fixture(rows, assignments)
    fold = runner.Fold("h1_2024-02", runner.PERIOD_MAIN, 1,
                       ordinal("2024-02"), ordinal("2024-01"))
    frame = run_fold(fold, panel)["rows"]

    assert list(frame.columns) == list(runner.PREDICTION_COLUMNS)
    for _, row in frame.iterrows():
        code, branch, source = assignments[row["admin_code"]]
        assert row["partition_code"] == code
        assert row["branch_id"] == branch
        assert row["assignment_source"] == source
        # The frozen assignment and the ACTUAL route are separate facts.
        expected = runner.ROUTE_POOLED if code < 0 else f"partition:{code}"
        assert row["model_route"] == expected
    assert set(frame["fold_id"]) == {"h1_2024-02"}
    assert set(frame["period"]) == {runner.PERIOD_MAIN}
    assert set(frame["origin_month"]) == {"2024-01"}


def test_a_local_model_trained_on_the_whole_pool_is_the_pooled_forest():
    """Same rows and the same seed give the same forest. This is the ONLY case
    where the two RF arms are required to agree; it never holds once an
    unassigned area contributes a training row (see the test above)."""
    rows = monthly_rows(10, "2022-01", 24, horizon=1) + monthly_rows(
        11, "2022-01", 24, horizon=1, feature_base=3.0
    )
    rows += [(10, "2024-02", 1, 1, 4.0), (11, "2024-02", 1, 0, 5.0)]
    assignments = {10: (0, "", "learned"), 11: (0, "", "learned")}
    panel = panel_fixture(rows, assignments)
    fold = runner.Fold("h1_2024-02", runner.PERIOD_MAIN, 1,
                       ordinal("2024-02"), ordinal("2024-01"))
    outcome = run_fold(fold, panel)
    record = outcome["record"]
    assert record["unassigned_train_rows_excluded_from_local"] == 0
    assert record["local_models_fitted"] == 1
    assert record["partitioned_identical_to_pooled"] is True
    frame = outcome["rows"]
    assert (frame["prob_partitioned_rf"] == frame["prob_pooled_rf"]).all()
    assert (frame["pred_partitioned_rf"] == frame["pred_pooled_rf"]).all()


def test_a_helper_whose_probabilities_contradict_the_route_is_refused():
    rows = monthly_rows(10, "2022-01", 24, horizon=1) + monthly_rows(
        11, "2022-01", 24, horizon=1, feature_base=3.0
    )
    rows += [(10, "2024-02", 1, 1, 4.0), (11, "2024-02", 1, 0, 5.0)]
    assignments = {10: (0, "", "learned"), 11: (0, "", "learned")}
    panel = panel_fixture(rows, assignments)
    fold = runner.Fold("h1_2024-02", runner.PERIOD_MAIN, 1,
                       ordinal("2024-02"), ordinal("2024-01"))

    broken = simple_helpers(2)
    broken.predict_partitioned_probability = (
        lambda models, pooled, X, group: np.full(len(X), 0.123)
    )
    try:
        run_fold(fold, panel, helpers=broken)
    except runner.PipelineError as error:
        assert "local model's probability" in str(error) or "pooled" in str(error)
    else:
        raise AssertionError("a route that contradicts the probabilities must stop")


def test_a_small_or_single_class_partition_falls_back_to_the_same_pooled_model():
    rows = monthly_rows(10, "2022-01", 24, horizon=1)
    rows += monthly_rows(11, "2022-01", 24, horizon=1, feature_base=3.0)
    # Area 12 has only single-class history, so its partition cannot be fitted.
    rows += monthly_rows(12, "2022-01", 24, horizon=1, label_cycle=(1,), feature_base=8.0)
    rows += [
        (10, "2024-02", 1, 1, 4.0),
        (11, "2024-02", 1, 0, 5.0),
        (12, "2024-02", 1, 1, 6.0),
    ]
    assignments = {
        10: (0, "", "learned"),
        11: (0, "", "learned"),
        12: (1, "1", "learned"),
    }
    panel = panel_fixture(rows, assignments)
    fold = runner.Fold("h1_2024-02", runner.PERIOD_MAIN, 1,
                       ordinal("2024-02"), ordinal("2024-01"))
    outcome = run_fold(fold, panel)
    record = outcome["record"]
    assert record["local_fallback_codes"] == [1]
    frame = outcome["rows"]
    fallback = frame[frame["partition_code"] == 1]
    assert (fallback["model_route"] == runner.ROUTE_POOLED).all()
    assert (fallback["prob_partitioned_rf"] == fallback["prob_pooled_rf"]).all()
    assert "single class" in fallback["model_fallback_reason"].iloc[0]


# --------------------------------------------------------------------------
# Q7a thresholds and degenerate probability cases
# --------------------------------------------------------------------------


def test_a_probability_of_exactly_one_half_is_class_zero_for_every_arm():
    rows = monthly_rows(10, "2022-01", 24, horizon=1) + monthly_rows(
        11, "2022-01", 24, horizon=1, feature_base=3.0
    )
    rows += [(10, "2024-02", 1, 1, 4.0), (11, "2024-02", 1, 0, 5.0)]
    assignments = {10: (0, "", "learned"), 11: (0, "", "learned")}
    panel = panel_fixture(rows, assignments)
    fold = runner.Fold("h1_2024-02", runner.PERIOD_MAIN, 1,
                       ordinal("2024-02"), ordinal("2024-01"))

    half = simple_helpers(2)
    half.predict_class1_probability = lambda model, X: np.full(len(X), 0.5)
    half.predict_partitioned_probability = (
        lambda models, pooled, X, group: np.full(len(X), 0.5)
    )
    outcome = run_fold(fold, panel, helpers=half, probability=0.5)
    frame = outcome["rows"]
    for arm in ("partitioned_rf", "pooled_rf", "xgb"):
        assert (frame[f"prob_{arm}"] == 0.5).all()
        assert (frame[f"pred_{arm}"] == 0).all(), f"{arm} must call an exact .5 class 0"

    just_over = np.array([0.5 + np.finfo(float).eps])
    assert (just_over > runner.DECISION_THRESHOLD).all()


def test_an_all_negative_and_an_all_positive_pooled_rf_are_handled():
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

    X = np.arange(20, dtype=np.float64).reshape(10, 2)
    for only_class, expected in ((0, 0.0), (1, 1.0)):
        model = RandomForestClassifier(**runner.STAGE3_RF_PARAMS)
        model.fit(X, np.full(10, only_class))
        probability = _predict_class1(model, X)
        assert (probability == expected).all()
        assert ((probability > runner.DECISION_THRESHOLD).astype(int)
                == only_class).all()


def test_a_fold_whose_training_pool_is_single_class_still_produces_predictions():
    rows = monthly_rows(10, "2022-01", 24, horizon=1, label_cycle=(0,))
    rows += monthly_rows(11, "2022-01", 24, horizon=1, label_cycle=(0,), feature_base=3.0)
    rows += [(10, "2024-02", 1, 1, 4.0), (11, "2024-02", 1, 0, 5.0)]
    assignments = {10: (0, "", "learned"), 11: (0, "", "learned")}
    panel = panel_fixture(rows, assignments)
    fold = runner.Fold("h1_2024-02", runner.PERIOD_MAIN, 1,
                       ordinal("2024-02"), ordinal("2024-01"))
    outcome = run_fold(fold, panel, probability=0.25)
    frame = outcome["rows"]
    assert outcome["record"]["train_classes"] == 1
    assert (frame["prob_pooled_rf"] == 0.0).all()
    assert (frame["pred_pooled_rf"] == 0).all()
    # The single local partition is single-class too, so it falls back.
    assert outcome["record"]["local_fallback_codes"] == [0]


# --------------------------------------------------------------------------
# Q4 persistence
# --------------------------------------------------------------------------


def test_persistence_is_the_latest_valid_label_at_or_before_the_origin():
    valid = pd.DataFrame(
        {
            "admin_code": [7, 7, 7, 8],
            "year": [2019, 2023, 2024, 2024],
            "month": [1, 5, 8, 8],
            "ipcch_food_crisis": [1, 0, 1, 1],
        }
    )
    admin = np.array([7, 7, 7, 8, 9], dtype=np.int64)
    origin = np.array(
        [ordinal("2023-05"), ordinal("2024-07"), ordinal("2018-12"),
         ordinal("2030-01"), ordinal("2024-08")],
        dtype=np.int64,
    )
    out = runner.persistence_lookup(valid, admin, origin)

    # Inclusive at O.
    assert out["available"][0] and out["value"][0] == 0 and out["age"][0] == 0
    # Latest strictly before O, with no maximum age.
    assert out["available"][1] and out["value"][1] == 0
    assert out["age"][1] == ordinal("2024-07") - ordinal("2023-05")
    # No history before the first label: missing, never zero-filled.
    assert not out["available"][2] and np.isnan(out["value"][2])
    assert out["source_ord"][2] == -1 and np.isnan(out["age"][2])
    # A very old label is still used: Q4 has no 36-month truncation.
    assert out["available"][3] and out["value"][3] == 1
    assert out["age"][3] == ordinal("2030-01") - ordinal("2024-08")
    # An area with no label at all.
    assert not out["available"][4]


def test_persistence_is_computed_from_the_ledger_not_from_the_training_pool():
    # The label that persistence must find is OUTSIDE this fold's 36-month
    # training window, so it can only come from the independent lookup.
    rows = monthly_rows(10, "2022-01", 24, horizon=1) + monthly_rows(
        11, "2022-01", 24, horizon=1, feature_base=3.0
    )
    rows += [(10, "2024-02", 1, 1, 4.0), (11, "2024-02", 1, 0, 5.0)]
    assignments = {10: (0, "", "learned"), 11: (0, "", "learned")}
    panel = panel_fixture(rows, assignments)

    fold = runner.Fold("h1_2024-02", runner.PERIOD_MAIN, 1,
                       ordinal("2024-02"), ordinal("2024-01"))
    outcome = run_fold(fold, panel)
    frame = outcome["rows"]
    assert frame["persistence_pred"].notna().all()
    for _, row in frame.iterrows():
        age = ordinal(row["origin_month"]) - ordinal(row["persistence_source_month"])
        assert row["persistence_age_months"] == age
        assert age >= 0
    assert (frame["persistence_source_month"] <= frame["origin_month"]).all()


# --------------------------------------------------------------------------
# Frozen map, panel and output contract
# --------------------------------------------------------------------------


def test_the_prediction_schema_matches_the_reporters_required_columns():
    schema = runner.verify_prediction_schema()
    assert set(schema["required"]) <= set(runner.PREDICTION_COLUMNS)
    assert schema["extra_columns_written"] == []
    assert list(runner.PREDICTION_COLUMNS)[:1] == ["admin_code"]


def test_the_panel_refuses_a_scored_area_outside_the_frozen_map():
    matrix = _tiny_matrix([(10, "2024-02"), (99, "2024-02")])
    assignments = pd.DataFrame(
        {
            "admin_code": [10],
            "branch_id": [""],
            "partition_code": [0],
            "assignment_source": ["learned"],
            "donor_admin_code": [-1],
            "donor_distance_km": [np.nan],
        }
    )
    valid = pd.DataFrame(
        {"admin_code": [10, 99], "year": [2024, 2024], "month": [2, 2],
         "ipcch_food_crisis": [1, 0]}
    )
    try:
        runner.build_stage3_panel(matrix, assignments, valid)
    except runner.PipelineError as error:
        assert "outside the frozen map" in str(error)
    else:
        raise AssertionError("an unmapped scored area must stop the run")


def test_the_panel_cross_checks_q6a_history_against_q4_persistence():
    matrix = _tiny_matrix([(10, "2024-02")])
    # Break the Q6a feature: claim there is no history where the ledger has one.
    matrix.frame.loc[:, "last_observed_label"] = np.nan
    assignments = pd.DataFrame(
        {
            "admin_code": [10],
            "branch_id": [""],
            "partition_code": [0],
            "assignment_source": ["learned"],
            "donor_admin_code": [-1],
            "donor_distance_km": [np.nan],
        }
    )
    valid = pd.DataFrame(
        {"admin_code": [10, 10], "year": [2023, 2024], "month": [1, 2],
         "ipcch_food_crisis": [1, 1]}
    )
    try:
        runner.build_stage3_panel(matrix, assignments, valid)
    except runner.PipelineError as error:
        assert "persistence lookup" in str(error)
    else:
        raise AssertionError("Q6a history and Q4 persistence must agree")


def _tiny_matrix(outcomes):
    """A minimal FeatureMatrix: four horizon views per (area, target month)."""
    records = []
    for area, target in outcomes:
        target_ord = ordinal(target)
        for horizon in sorted(pdata.ACTIVE_HORIZONS):
            records.append(
                {
                    "admin_code": area,
                    "country_en": "Testland",
                    "ISO3": "TST",
                    "target_month": target,
                    "origin_month": pdata.month_label([target_ord - horizon])[0],
                    "ipcch_food_crisis": 1,
                    "last_observed_label_month": "",
                    "last_observed_crisis_month": "",
                    "horizon_months": float(horizon),
                }
            )
    frame = pd.DataFrame(records)
    for name in pdata.FEATURE_COLUMNS:
        if name not in frame.columns:
            frame[name] = 1.0
    frame["last_observed_label"] = 1.0
    frame = frame[list(pdata.METADATA_COLUMNS) + list(pdata.FEATURE_COLUMNS)]
    return pdata.FeatureMatrix(
        frame=frame,
        feature_columns=pdata.FEATURE_COLUMNS,
        metadata_columns=pdata.METADATA_COLUMNS,
        infinity_audit={},
        audit={},
    )


def test_coverage_refuses_a_missing_learned_prediction_and_reports_e_persist():
    predictions = pd.DataFrame(
        {
            "admin_code": [1, 2, 3],
            "target_month": ["2024-02"] * 3,
            "horizon_months": [1, 1, 1],
            "period": ["main"] * 3,
            "ipcch_food_crisis": [1, 0, 1],
            "country_en": ["A", "A", "B"],
            "prob_partitioned_rf": [0.6, 0.4, 0.9],
            "pred_partitioned_rf": [1, 0, 1],
            "prob_pooled_rf": [0.6, 0.4, 0.9],
            "pred_pooled_rf": [1, 0, 1],
            "prob_xgb": [0.7, 0.3, 0.8],
            "pred_xgb": [1, 0, 1],
            "persistence_pred": [1.0, np.nan, 0.0],
            "model_route": ["partition:0"] * 3,
        }
    )
    records = [{"test_rows": 3, "fitted": True}]
    folds = [runner.Fold("h1_2024-02", "main", 1, ordinal("2024-02"), ordinal("2024-01"))]

    coverage = runner.stage3_coverage(predictions, folds, records)
    assert coverage["arms"]["partitioned_rf"]["rows"] == 3
    assert coverage["cohorts"]["main"]["E_all"] == 3
    assert coverage["cohorts"]["main"]["E_persist"] == 2
    assert coverage["arms"]["persistence"]["missing"] == 1

    broken = predictions.copy()
    broken.loc[1, "pred_xgb"] = np.nan
    try:
        runner.stage3_coverage(broken, folds, records)
    except runner.PipelineError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("a missing learned prediction must fail the run")


def test_coverage_refuses_a_tie_classified_as_class_one():
    predictions = pd.DataFrame(
        {
            "admin_code": [1],
            "target_month": ["2024-02"],
            "horizon_months": [1],
            "period": ["main"],
            "ipcch_food_crisis": [1],
            "country_en": ["A"],
            "prob_partitioned_rf": [0.5],
            "pred_partitioned_rf": [1],
            "prob_pooled_rf": [0.4],
            "pred_pooled_rf": [0],
            "prob_xgb": [0.4],
            "pred_xgb": [0],
            "persistence_pred": [1.0],
            "model_route": ["partition:0"],
        }
    )
    records = [{"test_rows": 1, "fitted": True}]
    folds = [runner.Fold("h1_2024-02", "main", 1, ordinal("2024-02"), ordinal("2024-01"))]
    try:
        runner.stage3_coverage(predictions, folds, records)
    except runner.PipelineError as error:
        assert "exact .5" in str(error)
    else:
        raise AssertionError("an exact .5 classified as 1 must fail (Q7a)")


# --------------------------------------------------------------------------
# Q7b booster verification
# --------------------------------------------------------------------------


def test_the_fitted_booster_is_checked_against_q7b_and_catches_a_wrong_depth():
    try:
        import xgboost  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SkipTest(f"xgboost unavailable: {error}") from error

    rng = np.random.default_rng(0)
    X = rng.random((120, 4))
    X[0, 0] = np.nan
    y = (rng.random(120) > 0.5).astype(int)

    params = dict(runner.STAGE3_XGB_PARAMS)
    params["n_estimators"] = 12
    good = xgboost.XGBClassifier(missing=np.nan, **params)
    good.fit(X, y)
    report = runner.verify_xgb_configuration(
        good, expected={**runner.STAGE3_XGB_PARAMS, "n_estimators": 12}
    )
    effective = report["booster_effective"]
    assert effective["objective"] == "binary:logistic"
    assert effective["num_boosted_rounds"] == 12
    assert effective["max_depth"] == 6
    assert effective["eval_metric"] == ["logloss"]
    assert effective["scale_pos_weight"] == 1.0
    assert effective["num_parallel_tree"] == 1
    assert effective["missing"] == "nan"

    params["max_depth"] = 3
    wrong = xgboost.XGBClassifier(missing=np.nan, **params)
    wrong.fit(X, y)
    try:
        runner.verify_xgb_configuration(
            wrong, expected={**runner.STAGE3_XGB_PARAMS, "n_estimators": 12}
        )
    except runner.PipelineError as error:
        assert "max_depth" in str(error)
    else:
        raise AssertionError("a booster that contradicts Q7b must be refused")


def test_the_released_helpers_agree_with_the_local_mirror_when_available():
    """Exercise the REAL baseline helpers if an extracted copy is on disk."""
    import baseline_runtime as brt  # noqa: PLC0415

    candidates = sorted(
        (runner.DEFAULT_RUNS_DIR).glob("*/baseline/GeoRFBaseline")
    ) if runner.DEFAULT_RUNS_DIR.is_dir() else []
    if not candidates:
        raise SkipTest("no extracted baseline copy under runs/")
    root = candidates[-1].resolve()

    runtime = brt.BaselineRuntime(
        root=root,
        release_sha256="",
        manifest_version="",
        manifest_source_commit="",
        payload_files_verified=0,
        patch_applied=True,
        patch_diff="",
        pristine_target_sha256="",
        patched_target_sha256="",
    )
    rng = np.random.default_rng(3)
    X_train = rng.random((200, 4))
    y_train = (rng.random(200) > 0.5).astype(int)
    group_train = np.where(np.arange(200) < 120, 0, np.where(np.arange(200) < 190, 1, -1))
    X_test = rng.random((20, 4))
    group_test = np.array([0] * 8 + [1] * 6 + [2] * 3 + [-1] * 3)

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        with brt.baseline_imports(runtime):
            helpers, _module = runner._load_stage3_helpers(runtime)
            pooled = helpers.train_pooled(X_train, y_train)
            models = helpers.train_partitioned(
                X_train, y_train, group_train, min_samples=helpers.min_partition_rows
            )
            released = helpers.predict_partitioned_probability(
                models, pooled, X_test, group_test
            )
            pooled_probability = helpers.predict_class1_probability(pooled, X_test)
            sys.modules.pop("ipcch_baseline_stage3", None)

    assert helpers.min_partition_rows == runner.MIN_PARTITION_TRAIN_ROWS == 50
    mirror = _predict_partitioned(models, pooled, X_test, group_test)
    assert np.array_equal(released, mirror)

    route, _reason = runner.local_model_routes(models, group_test, 50)
    pooled_routed = route == runner.ROUTE_POOLED
    assert np.array_equal(released[pooled_routed], pooled_probability[pooled_routed])
    # Partition 1 has 70 rows: above the 50-row gate, so it is fitted locally.
    assert models[1] is not None and models[0] is not None
    assert set(route[group_test == 2]) == {runner.ROUTE_POOLED}
    assert set(route[group_test == -1]) == {runner.ROUTE_POOLED}


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
