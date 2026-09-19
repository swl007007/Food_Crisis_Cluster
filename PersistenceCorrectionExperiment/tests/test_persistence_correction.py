"""Contract tests for the persistence-correction experiment, Phase 0 and Phase 1.

Run from the repository root:
    python3 -m pytest PersistenceCorrectionExperiment/tests -q
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from persistencecorrection import persistence, protected  # noqa: E402
from persistencecorrection.persistence import PersistenceContractError  # noqa: E402

TRI_ANNUAL = pd.to_datetime(
    [f"{year}-{month:02d}-01" for year in range(2018, 2023) for month in (2, 6, 10)]
)


def _write_source(path: Path, months, ipc, *, admin=101, near=None, med=None):
    """Write a minimal FEWSNET.csv-shaped observed-phase source."""
    frame = pd.DataFrame(
        {
            "country": "Testland",
            "admin_code": admin,
            "year_month": [f"{m.year}_{m.month:02d}" for m in months],
            "year": [m.year for m in months],
            "month": [m.month for m in months],
            "fews_ipc": ipc,
            "fews_proj_near": near if near is not None else [None] * len(months),
            "fews_proj_med": med if med is not None else [None] * len(months),
        }
    )
    frame.to_csv(path, index=False)
    return frame


def _support(months, admin=101, y_true=None):
    months = pd.to_datetime(list(months))
    return pd.DataFrame(
        {
            persistence.SUPPORT_ADMIN_COLUMN: admin,
            persistence.SUPPORT_MONTH_COLUMN: months,
            "y_true": [0] * len(months) if y_true is None else y_true,
        }
    )


# ---------------------------------------------------------------------------
# Phase 0: protected-hash gate extension (PRD R26)
# ---------------------------------------------------------------------------


def test_protected_paths_extend_step3_without_duplicates():
    from step3correction.protected import protected_paths as step3_paths

    step3 = [str(p) for p in step3_paths()]
    ours = [str(p) for p in protected.protected_paths()]
    assert len(ours) == len(set(ours)), "protected path list must be de-duplicated"
    assert ours[: len(step3)] == step3, "Step 3 entries must be preserved in order"
    assert set(step3).issubset(set(ours))


def test_protected_paths_cover_fs1_and_fs2_stage3_inputs():
    ours = {str(p) for p in protected.protected_paths()}
    for scope in (1, 2):
        assert str(protected.package_stage3_path(scope, "predictions_monthly.csv")) in ours
        for key in ("general", "m2", "m6", "m10"):
            assert str(protected.refined_map_path(scope, key)) in ours


def test_resolve_output_path_refuses_paths_outside_the_experiment_tree(tmp_path):
    inside = protected.resolve_output_path("run_x/file.json")
    assert protected.OUTPUT_ROOT.resolve() in inside.parents
    for escape in ("../escape.json", str(tmp_path / "escape.json"), "/etc/passwd"):
        with pytest.raises(protected.ProtectedPathError):
            protected.resolve_output_path(escape)
    assert not (protected.OUTPUT_ROOT / "run_x").exists(), "must not create directories"


# ---------------------------------------------------------------------------
# Phase 1: the persistence contract (PRD R8)
# ---------------------------------------------------------------------------


def test_source_month_is_exactly_target_minus_horizon(tmp_path):
    source = tmp_path / "FEWSNET.csv"
    _write_source(source, TRI_ANNUAL, ipc=[1] * len(TRI_ANNUAL))
    table = persistence.load_observed_phase_history(source)
    for scope, horizon in persistence.PERSISTENCE_HORIZON_MONTHS.items():
        targets = TRI_ANNUAL[TRI_ANNUAL >= TRI_ANNUAL[0] + pd.DateOffset(months=horizon)]
        series = persistence.attach_persistence(_support(targets), table, scope)
        lag = (
            (series[persistence.SUPPORT_MONTH_COLUMN].dt.year
             - series["persistence_source_month"].dt.year) * 12
            + series[persistence.SUPPORT_MONTH_COLUMN].dt.month
            - series["persistence_source_month"].dt.month
        )
        assert (lag == horizon).all()
        assert (series["persistence_horizon_months"] == horizon).all()


def test_both_horizons_land_on_observed_tri_annual_months(tmp_path):
    """H=4 and H=8 both hit Feb/Jun/Oct origins for Feb/Jun/Oct targets."""
    source = tmp_path / "FEWSNET.csv"
    _write_source(source, TRI_ANNUAL, ipc=[1] * len(TRI_ANNUAL))
    table = persistence.load_observed_phase_history(source)
    observed = set(TRI_ANNUAL)
    for scope, horizon in persistence.PERSISTENCE_HORIZON_MONTHS.items():
        targets = TRI_ANNUAL[TRI_ANNUAL >= TRI_ANNUAL[0] + pd.DateOffset(months=horizon)]
        series = persistence.attach_persistence(_support(targets), table, scope)
        assert set(series["persistence_source_month"]) <= observed
        assert series["persistence_source_month"].dt.month.isin([2, 6, 10]).all()
        coverage = persistence.persistence_coverage(series, scope)
        assert coverage["coverage"] == 1.0


def test_missing_origin_observation_halts_instead_of_imputing(tmp_path):
    """An induced gap at T-H must abort; coverage below 1.0 is never tolerated."""
    source = tmp_path / "FEWSNET.csv"
    kept = TRI_ANNUAL.delete(3)  # drop one release month
    _write_source(source, kept, ipc=[1] * len(kept))
    table = persistence.load_observed_phase_history(source)
    targets = TRI_ANNUAL[TRI_ANNUAL >= TRI_ANNUAL[0] + pd.DateOffset(months=4)]
    with pytest.raises(PersistenceContractError) as excinfo:
        persistence.attach_persistence(_support(targets), table, 1)
    message = str(excinfo.value)
    assert "coverage" in message and "never imputed" in message


def test_binarisation_happens_before_the_join(tmp_path):
    """Fractional and null phases cannot leak through the join."""
    source = tmp_path / "FEWSNET.csv"
    ipc = [2.9, None, 3.0, 2.5, 4.7, None, 1.0, 3.5, 0.0, 3.2, 2.0, 5.0, 1.0, 1.0, 1.0]
    assert len(ipc) == len(TRI_ANNUAL)
    _write_source(source, TRI_ANNUAL, ipc=ipc)
    table = persistence.load_observed_phase_history(source)
    targets = TRI_ANNUAL[TRI_ANNUAL >= TRI_ANNUAL[0] + pd.DateOffset(months=4)]
    series = persistence.attach_persistence(_support(targets), table, 1)

    assert series["persistence"].isin([0, 1]).all()
    assert str(series["persistence"].dtype).startswith("int")
    expected = [int(value is not None and value >= 3) for value in ipc[: len(targets)]]
    assert series["persistence"].tolist() == expected
    # A raw-missing phase is recorded as 0 and flagged, never propagated as null.
    missing_rows = series.loc[series["persistence_phase_missing"] == 1]
    assert len(missing_rows) == 2
    assert (missing_rows["persistence"] == 0).all()
    assert series["persistence_source_month"].notna().all()


def test_no_imputation_or_record_shift_code_path():
    """Static check: the module calls no fill/shift/reindex API anywhere."""
    module_path = Path(persistence.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    called, keywords = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                called.add(func.attr)
            elif isinstance(func, ast.Name):
                called.add(func.id)
            keywords.update(kw.arg for kw in node.keywords if kw.arg)
    forbidden = {
        "fillna", "ffill", "bfill", "pad", "backfill", "interpolate",
        "reindex", "asfreq", "resample", "shift", "SimpleImputer", "combine_first",
    }
    assert not (called & forbidden), f"forbidden call(s): {sorted(called & forbidden)}"
    assert not ({"fill_value", "limit_direction"} & keywords)
    assert "method" not in keywords


def test_undefined_scope_is_rejected():
    for scope in (0, 3, 4):
        with pytest.raises(PersistenceContractError):
            persistence.require_horizon(scope)


def test_duplicate_support_keys_are_rejected(tmp_path):
    source = tmp_path / "FEWSNET.csv"
    _write_source(source, TRI_ANNUAL, ipc=[1] * len(TRI_ANNUAL))
    table = persistence.load_observed_phase_history(source)
    targets = list(TRI_ANNUAL[4:6]) + [TRI_ANNUAL[5]]
    with pytest.raises(PersistenceContractError):
        persistence.attach_persistence(_support(targets), table, 1)


def test_persistence_metrics_match_a_hand_computed_confusion(tmp_path):
    source = tmp_path / "FEWSNET.csv"
    ipc = [3, 3, 1, 1, 3, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1]
    _write_source(source, TRI_ANNUAL, ipc=ipc)
    table = persistence.load_observed_phase_history(source)
    targets = TRI_ANNUAL[1:5]
    truth = [1, 0, 1, 1]
    series = persistence.attach_persistence(
        _support(targets, y_true=truth), table, 1
    )
    # fs1 origins for targets[0..3] are TRI_ANNUAL[0..3] -> persistence 1,1,0,0
    assert series["persistence"].tolist() == [1, 1, 0, 0]
    metrics = persistence.persistence_metrics(series)
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (1, 1, 2, 0)
    assert metrics["f1_class1"] == pytest.approx(2 * 1 / (2 * 1 + 1 + 2))


def test_manifest_carries_the_r6_assumption_verbatim(tmp_path):
    source = tmp_path / "FEWSNET.csv"
    _write_source(source, TRI_ANNUAL, ipc=[1] * len(TRI_ANNUAL))
    table = persistence.load_observed_phase_history(source)
    manifest = persistence.build_persistence_manifest(table, {})
    assert (
        manifest["availability_assumption_r6_verbatim"]
        == persistence.R6_AVAILABILITY_ASSUMPTION
    )
    assert "`fews_ipc(D)` is treated as available to a forecaster at month D." in (
        manifest["availability_assumption_r6_verbatim"]
    )
    contradiction = manifest["availability_assumption_recorded_contradiction"]
    assert contradiction["assumes_available_at_origin"].startswith("src/preprocess")
    assert any(
        ".trellis/spec/backend/local-forecasting-experiments.md" in entry
        for entry in contradiction["assumes_not_available_at_origin"]
    )
    assert manifest["persistence"]["record_shift_used"] is False
    assert manifest["persistence"]["imputation"].startswith("none")


def test_no_legacy_record_shift_column_can_reach_the_persistence_series(tmp_path):
    from step3correction.expert import ExpertContractError

    source = tmp_path / "FEWSNET.csv"
    _write_source(source, TRI_ANNUAL, ipc=[1] * len(TRI_ANNUAL))
    table = persistence.load_observed_phase_history(source)
    assert not any(
        str(column).startswith("legacy_record_shift_") for column in table.frame.columns
    )
    targets = TRI_ANNUAL[TRI_ANNUAL >= TRI_ANNUAL[0] + pd.DateOffset(months=4)]
    smuggled = _support(targets)
    smuggled["legacy_record_shift_expert_1"] = 1
    with pytest.raises(ExpertContractError):
        persistence.attach_persistence(smuggled, table, 1)
