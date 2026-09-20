"""Compact contract checks for the Stage 1 runner.

Hand-computable fixtures only: no pinned source, no geopandas, no model fit.
The real-count gates (42,695 / 19,591 / 8,561 ...) live in ``prepare_data`` and
are exercised by the runner itself. What is checked here is the logic the runner
adds on top: the Stage 1 design matrix, imputer discipline, learned-map
extraction, reconciliation, donor eligibility and the 100 km completion rule.

Run with the preferred interpreter::

    python3.12.exe -B IPCCHGeoRFExperiment/test_stage1_contracts.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_data as pdata  # noqa: E402
import run_pipeline as runner  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def s_branch_frame(columns: dict[str, list[int]], height: int | None = None) -> pd.DataFrame:
    """Build an ``s_branch``-shaped frame: branch label -> members, -1 padded."""
    height = height or max(len(v) for v in columns.values())
    data = {}
    for label, members in columns.items():
        padded = list(members) + [-1] * (height - len(members))
        data[label] = np.asarray(padded, dtype=np.int32)
    return pd.DataFrame(data)


def split_fixture(rows: list[tuple[int, str, str, int]]) -> pdata.Stage1Split:
    """``rows`` are ``(admin_code, 'YYYY-MM', split_role, label)``."""
    frame = pd.DataFrame(
        rows, columns=["admin_code", "target_month", "split_role", "ipcch_food_crisis"]
    )
    frame["target_month"] = pd.to_datetime(frame["target_month"] + "-01")
    frame["country_en"] = "X"
    frame["ISO3"] = "XXX"
    counts = frame["split_role"].value_counts().to_dict()
    audit = {
        "fit": int(counts.get("fit", 0)),
        "validation": int(counts.get("validation", 0)),
        "singleton_areas": int(counts.get("singleton", 0)),
    }
    return pdata.Stage1Split(
        outcomes=frame, zero_label_areas=np.asarray([], dtype=np.int64), audit=audit
    )


def coordinate_frame(rows: list[tuple[int, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[pdata.REFERENCE_ID_COLUMN, "ref_lat", "ref_lon"])


def feature_matrix_fixture(outcomes: list[tuple[int, str, int]]) -> pdata.FeatureMatrix:
    """Four horizon views per outcome, with a distinguishable single feature."""
    horizons = sorted(pdata.ACTIVE_HORIZONS)
    records = []
    for area, target, label in outcomes:
        year, month = (int(part) for part in target.split("-"))
        target_ord = pdata.month_ordinal(year, month)
        for horizon in horizons:
            origin = int(target_ord) - horizon
            records.append(
                {
                    "admin_code": area,
                    "country_en": "X",
                    "ISO3": "XXX",
                    "target_month": target,
                    "origin_month": pdata.month_label([origin])[0],
                    "ipcch_food_crisis": label,
                    "last_observed_label_month": "",
                    "last_observed_crisis_month": "",
                    "horizon_months": float(horizon),
                }
            )
    frame = pd.DataFrame(records)
    for name in pdata.FEATURE_COLUMNS:
        if name not in frame.columns:
            frame[name] = 1.0
    frame = frame[list(pdata.METADATA_COLUMNS) + list(pdata.FEATURE_COLUMNS)]
    return pdata.FeatureMatrix(
        frame=frame,
        feature_columns=pdata.FEATURE_COLUMNS,
        metadata_columns=pdata.METADATA_COLUMNS,
        infinity_audit={},
        audit={},
    )


class RecordingImputer:
    """Stand-in that records the rows it was fitted on and stores its fills.

    Deliberately not the baseline class: the contract under test is *which rows
    reach fit* and *that transform reuses stored values*, which must hold for any
    conforming imputer.
    """

    def __init__(self):
        self.fit_calls = 0
        self.fit_rows = None
        self.fills = None

    def fit(self, X):
        self.fit_calls += 1
        self.fit_rows = np.array(X, copy=True)
        with np.errstate(all="ignore"):
            maxima = np.nanmax(np.where(np.isnan(X), -np.inf, X), axis=0)
        maxima = np.where(np.isfinite(maxima), maxima, 0.0)
        self.fills = np.where(maxima == 0, 100.0, maxima * 100.0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float64, copy=True)
        missing = np.isnan(X)
        if missing.any():
            X[missing] = np.broadcast_to(self.fills, X.shape)[missing]
        return X


class SkipTest(Exception):
    pass


# --------------------------------------------------------------------------
# Run scaffolding
# --------------------------------------------------------------------------


def test_run_context_refuses_an_existing_run_id():
    with tempfile.TemporaryDirectory() as tmp:
        runner.RunContext(Path(tmp), "alpha")
        try:
            runner.RunContext(Path(tmp), "alpha")
        except runner.PipelineError as error:
            assert "already exists" in str(error)
        else:
            raise AssertionError("a second run with the same id must be refused")


def test_manifest_is_written_atomically_and_failure_keeps_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        context = runner.RunContext(Path(tmp), "beta")
        assert context.manifest_path.is_file()
        for name in runner.RunContext.SUBDIRECTORIES:
            assert (context.root / name).is_dir(), name

        context.log("something happened")
        context.fail("gate failed", "detail")

        payload = json.loads(context.manifest_path.read_text(encoding="utf-8"))
        assert payload["status"] == "failed"
        assert payload["failure"]["reason"] == "gate failed"
        assert context.log_path.read_text(encoding="utf-8").strip().endswith(
            "something happened"
        )
        # No temp file is left behind by the replace-based write.
        assert not list(context.root.glob("*.tmp"))
        # A failed run is never silently promoted.
        assert payload["status"] != "complete"


# --------------------------------------------------------------------------
# Stage 1 design matrix
# --------------------------------------------------------------------------


def test_design_keeps_four_views_on_one_side_and_uses_real_admin_codes():
    matrix = feature_matrix_fixture(
        [(100341, "2019-06", 1), (100341, "2020-06", 0), (7, "2018-03", 1), (7, "2021-09", 0)]
    )
    split = split_fixture(
        [
            (100341, "2019-06", "fit", 1),
            (100341, "2020-06", "validation", 0),
            (7, "2018-03", "fit", 1),
            (7, "2021-09", "validation", 0),
        ]
    )
    design = runner.build_stage1_design(matrix, split)

    assert design.audit["learning_rows"] == 16
    assert design.audit["fit_rows"] == 8 and design.audit["validation_rows"] == 8
    assert set(np.unique(design.groups)) == {7, 100341}, "actual non-contiguous ids"
    assert set(np.unique(design.x_set)) == {0, 1}

    # All four views of one outcome share a side.
    for (area, target), block in design.frame.groupby(["admin_code", "target_month"]):
        assert block["split_role"].nunique() == 1, (area, target)
        assert sorted(block["horizon_months"]) == [1.0, 3.0, 6.0, 12.0]

    # Own-origin contract survives the join.
    for _, row in design.frame.iterrows():
        target = pdata.month_ordinal(*(int(p) for p in row["target_month"].split("-")))
        origin = pdata.month_ordinal(*(int(p) for p in row["origin_month"].split("-")))
        assert int(target) - int(origin) == int(row["horizon_months"])


def test_design_rejects_a_target_past_the_partition_cutoff():
    matrix = feature_matrix_fixture([(7, "2023-01", 1), (7, "2018-03", 0)])
    split = split_fixture(
        [(7, "2018-03", "fit", 0), (7, "2023-01", "validation", 1)]
    )
    try:
        runner.build_stage1_design(matrix, split)
    except runner.PipelineError as error:
        assert "cutoff" in str(error)
    else:
        raise AssertionError("2023-01 must not enter the partition-learning pool")


def test_design_drops_stage3_rows_without_a_role():
    matrix = feature_matrix_fixture(
        [(7, "2018-03", 1), (7, "2019-03", 0), (7, "2024-05", 1)]
    )
    split = split_fixture([(7, "2018-03", "fit", 1), (7, "2019-03", "validation", 0)])
    design = runner.build_stage1_design(matrix, split)
    assert design.audit["learning_rows"] == 8
    assert "2024-05" not in set(design.frame["target_month"])


def test_design_rejects_a_singleton_count_mismatch():
    matrix = feature_matrix_fixture([(7, "2018-03", 1), (9, "2019-03", 0)])
    split = split_fixture([(7, "2018-03", "singleton", 1)])  # 9 has no role at all
    split.audit["singleton_areas"] = 2  # claim one more than the frame holds
    try:
        runner.build_stage1_design(matrix, split)
    except runner.PipelineError as error:
        assert "singleton" in str(error)
    else:
        raise AssertionError("a support-count mismatch must be investigated, not absorbed")


# --------------------------------------------------------------------------
# Imputer discipline (Q6c)
# --------------------------------------------------------------------------


def test_imputer_fits_on_fitting_rows_only_and_held_out_extrema_do_not_move_fills():
    X = np.array(
        [
            [1.0, np.nan],
            [2.0, 5.0],
            [900.0, np.nan],   # validation extremum
            [np.nan, 7.0],
        ]
    )
    x_set = np.array([0, 0, 1, 1])
    singleton = np.array([[np.nan, np.nan]])

    imputer = RecordingImputer()
    imputer, (transformed, singles) = runner.apply_stage1_imputation(
        imputer, X, x_set, singleton
    )

    assert imputer.fit_calls == 1, "exactly one shared imputer"
    assert imputer.fit_rows.shape == (2, 2), "validation rows must not reach fit"
    assert np.allclose(imputer.fit_rows, X[:2], equal_nan=True)
    # Column 0's fit maximum is 2.0, so 900 (validation) cannot set the fill.
    assert np.isclose(imputer.fills[0], 200.0)
    assert np.isclose(transformed[3, 0], 200.0)
    # Held-out rows are transformed with the stored values, never refitted.
    assert np.isclose(singles[0, 0], 200.0)
    assert np.isclose(singles[0, 1], 500.0)
    assert imputer.fit_calls == 1
    # Observed values are untouched.
    assert np.isclose(transformed[2, 0], 900.0)


def test_imputer_refuses_an_empty_fitting_side():
    try:
        runner.apply_stage1_imputation(RecordingImputer(), np.ones((2, 2)), np.array([1, 1]))
    except runner.PipelineError as error:
        assert "no fitting rows" in str(error)
    else:
        raise AssertionError("an empty fitting side must stop the run")


# --------------------------------------------------------------------------
# Learned map extraction
# --------------------------------------------------------------------------


def test_default_root_placeholder_is_not_learned_membership():
    # Root split into '0'/'1'; area 30 stayed behind in the root column because
    # it had no validation row in that branch. The inherited helper would call it
    # root; R4 says that is a default, not membership.
    s_branch = s_branch_frame({"": [10, 20, 30], "0": [10], "1": [20]})
    learned = runner.extract_learned_map(s_branch)

    assert learned.branch_by_area == {10: "0", 20: "1"}
    assert learned.placeholder_areas == (30,)
    assert learned.terminal_branches == ("0", "1")


def test_unsplit_root_keeps_every_area_as_genuine_root_membership():
    s_branch = s_branch_frame({"": [10, 20, 30]})
    learned = runner.extract_learned_map(s_branch)
    assert learned.branch_by_area == {10: "", 20: "", 30: ""}
    assert learned.placeholder_areas == ()
    assert learned.terminal_branches == ("",)


def test_leading_zeros_and_ancestor_lineage_are_preserved():
    s_branch = s_branch_frame(
        {
            "": [1, 2, 3, 4],
            "0": [1, 2],
            "1": [3, 4],
            "00": [1],
            "01": [2],
            "000": [1],
            "001": [],
        }
    )
    learned = runner.extract_learned_map(s_branch)
    # '000' must not collapse to '0'; area 1's deepest explicit placement wins.
    assert learned.branch_by_area[1] == "000"
    assert learned.branch_by_area[2] == "01"
    assert learned.branch_by_area[3] == "1" and learned.branch_by_area[4] == "1"
    assert learned.audit["max_branch_depth"] == 3
    assert "00" not in learned.terminal_branches, "a split branch is not terminal"


def test_extract_rejects_a_malformed_branch_label():
    frame = s_branch_frame({"": [1], "0": [1]})
    frame = frame.rename(columns={"0": "0x"})
    try:
        runner.extract_learned_map(frame)
    except runner.PipelineError as error:
        assert "unexpected branch label" in str(error)
    else:
        raise AssertionError("branch labels must be root or 0/1 strings")


# --------------------------------------------------------------------------
# Reconciliation
# --------------------------------------------------------------------------


def _checkpoint_dir(tmp: Path, branches) -> Path:
    directory = Path(tmp) / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    for branch in branches:
        (directory / f"rf_{branch}").write_bytes(b"checkpoint")
    return directory


def test_reconciliation_accepts_agreeing_map_rows_and_checkpoints():
    with tempfile.TemporaryDirectory() as tmp:
        s_branch = s_branch_frame({"": [10, 20, 30], "0": [10], "1": [20]})
        learned = runner.extract_learned_map(s_branch)
        groups = np.array([10, 10, 20, 30])
        saved = np.array(["0", "0", "1", ""])
        checkpoints = _checkpoint_dir(Path(tmp), ["", "0", "1"])

        report = runner.reconcile_learned_map(learned, groups, saved, checkpoints)
        assert report["learned_areas"] == 2
        assert report["placeholder_areas"] == 1
        assert report["placeholder_rows_not_default_root"] == {}
        assert set(report["row_branches_used"]) == {"", "0", "1"}


def test_rejected_candidate_checkpoints_are_recorded_as_not_routed():
    # A rejected split still leaves rf_0/rf_1 on disk. They are trained REJECTED
    # candidates; nothing may route to them, and they must be visibly separated
    # from the checkpoint the root actually uses.
    with tempfile.TemporaryDirectory() as tmp:
        s_branch = s_branch_frame({"": [10, 20]})
        learned = runner.extract_learned_map(s_branch)
        report = runner.reconcile_learned_map(
            learned,
            np.array([10, 20]),
            np.array(["", ""]),
            _checkpoint_dir(Path(tmp), ["", "0", "1"]),
        )
        assert report["row_branches_used"] == [""]
        assert sorted(report["checkpoints_not_routed"]) == ["rf_0", "rf_1"]


def test_reconciliation_rejects_a_fragmented_admin_unit():
    with tempfile.TemporaryDirectory() as tmp:
        s_branch = s_branch_frame({"": [10, 20], "0": [10], "1": [20]})
        learned = runner.extract_learned_map(s_branch)
        groups = np.array([10, 10, 20])
        saved = np.array(["0", "1", "1"])  # one admin unit in two partitions
        try:
            runner.reconcile_learned_map(
                learned, groups, saved, _checkpoint_dir(Path(tmp), ["", "0", "1"])
            )
        except runner.PipelineError as error:
            assert "more than one branch" in str(error)
        else:
            raise AssertionError("an admin unit split across branches is an error")


def test_reconciliation_rejects_a_missing_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        s_branch = s_branch_frame({"": [10, 20], "0": [10], "1": [20]})
        learned = runner.extract_learned_map(s_branch)
        groups = np.array([10, 20])
        saved = np.array(["0", "1"])
        try:
            runner.reconcile_learned_map(
                learned, groups, saved, _checkpoint_dir(Path(tmp), ["", "0"])
            )
        except runner.PipelineError as error:
            assert "no checkpoint" in str(error)
        else:
            raise AssertionError(
                "inherited code catches rendering errors, so artefacts must be asserted"
            )


def test_reconciliation_rejects_a_fitted_group_absent_from_the_root_branch():
    with tempfile.TemporaryDirectory() as tmp:
        s_branch = s_branch_frame({"": [10]})
        learned = runner.extract_learned_map(s_branch)
        try:
            runner.reconcile_learned_map(
                learned, np.array([10, 99]), np.array(["", ""]),
                _checkpoint_dir(Path(tmp), [""]),
            )
        except runner.PipelineError as error:
            assert "absent from the root branch" in str(error)
        else:
            raise AssertionError("every fitted group must appear in s_branch")


def test_checkpoint_lookup_accepts_the_pkl_variant():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "rf_01.pkl").write_bytes(b"x")
        assert runner.checkpoint_exists(directory, "01")
        assert not runner.checkpoint_exists(directory, "0")


# --------------------------------------------------------------------------
# Donor eligibility and completion
# --------------------------------------------------------------------------


def test_haversine_matches_a_known_degree_of_latitude():
    one_degree = runner.haversine_km(0.0, 0.0, 1.0, 0.0)
    assert abs(float(one_degree) - runner.EARTH_RADIUS_KM * np.pi / 180.0) < 1e-9
    assert abs(float(runner.haversine_km(10.0, 20.0, 10.0, 20.0))) < 1e-12


def test_donor_eligibility_needs_fit_and_validation_and_learned_membership():
    s_branch = s_branch_frame({"": [1, 2, 3, 4], "0": [1, 2, 3], "1": [4]})
    learned = runner.extract_learned_map(s_branch)
    split = split_fixture(
        [
            (1, "2018-01", "fit", 0),
            (1, "2019-01", "validation", 1),   # eligible
            (2, "2018-01", "fit", 0),
            (2, "2019-01", "fit", 1),          # no validation outcome
            (3, "2018-01", "validation", 0),   # no fitting outcome
            (4, "2018-01", "fit", 0),
            (4, "2019-01", "validation", 1),   # eligible
            (5, "2018-01", "fit", 0),
            (5, "2019-01", "validation", 1),   # never learned
        ]
    )
    coords = coordinate_frame([(i, 0.0, float(i)) for i in range(1, 6)])
    donors = runner.eligible_donor_table(learned, split, coords)
    assert list(donors["admin_code"]) == [1, 4]
    assert list(donors["branch_id"]) == ["0", "1"]


def test_donor_eligibility_excludes_a_placeholder_root_area():
    s_branch = s_branch_frame({"": [1, 2], "0": [1], "1": []})
    learned = runner.extract_learned_map(s_branch)
    assert 2 in learned.placeholder_areas
    split = split_fixture(
        [
            (1, "2018-01", "fit", 0),
            (1, "2019-01", "validation", 1),
            (2, "2018-01", "fit", 0),
            (2, "2019-01", "validation", 1),
        ]
    )
    coords = coordinate_frame([(1, 0.0, 0.0), (2, 0.0, 1.0)])
    donors = runner.eligible_donor_table(learned, split, coords)
    assert list(donors["admin_code"]) == [1], "a default-root area is not a donor"


def test_completion_respects_the_inclusive_100km_cap():
    # One degree of latitude is ~111.19 km, so place recipients by fraction of it.
    degree_km = float(runner.haversine_km(0.0, 0.0, 1.0, 0.0))
    exactly_100 = 100.0 / degree_km
    just_over = 100.4 / degree_km

    s_branch = s_branch_frame({"": [1], "0": [1], "1": []})
    learned = runner.extract_learned_map(s_branch)
    split = split_fixture([(1, "2018-01", "fit", 0), (1, "2019-01", "validation", 1)])
    coords = coordinate_frame(
        [(1, 0.0, 0.0), (2, exactly_100, 0.0), (3, just_over, 0.0)]
    )
    donors = runner.eligible_donor_table(learned, split, coords)
    table = runner.complete_assignments(
        np.array([1, 2, 3]), coords, learned, donors
    ).set_index("admin_code")

    assert table.at[1, "assignment_source"] == "learned"
    assert table.at[2, "assignment_source"] == "nearest_donor"
    assert abs(table.at[2, "donor_distance_km"] - 100.0) < 1e-6
    assert table.at[2, "branch_id"] == "0"
    assert table.at[3, "assignment_source"] == "unresolved"
    assert table.at[3, "partition_code"] == -1
    assert table.at[3, "model_route"] == "pooled_root"
    assert table.at[3, "branch_id"] == ""
    # The distance is still recorded for an unresolved area, as evidence.
    assert table.at[3, "nearest_eligible_admin_code"] == 1
    assert table.at[3, "nearest_eligible_distance_km"] > 100.0


def test_completion_never_chains_through_a_recipient():
    # 1 is the only donor. 2 sits 60 km from it, 3 sits 60 km beyond 2, i.e. 120 km
    # from the only eligible donor. Chaining would rescue 3; Q8r forbids it.
    degree_km = float(runner.haversine_km(0.0, 0.0, 1.0, 0.0))
    step = 60.0 / degree_km

    s_branch = s_branch_frame({"": [1], "0": [1], "1": []})
    learned = runner.extract_learned_map(s_branch)
    split = split_fixture([(1, "2018-01", "fit", 0), (1, "2019-01", "validation", 1)])
    coords = coordinate_frame([(1, 0.0, 0.0), (2, step, 0.0), (3, 2 * step, 0.0)])
    donors = runner.eligible_donor_table(learned, split, coords)
    table = runner.complete_assignments(
        np.array([1, 2, 3]), coords, learned, donors
    ).set_index("admin_code")

    assert table.at[2, "assignment_source"] == "nearest_donor"
    assert table.at[3, "assignment_source"] == "unresolved"


def test_completion_breaks_distance_ties_on_the_lower_area_id():
    s_branch = s_branch_frame({"": [5, 9], "0": [5], "1": [9]})
    learned = runner.extract_learned_map(s_branch)
    split = split_fixture(
        [
            (5, "2018-01", "fit", 0),
            (5, "2019-01", "validation", 1),
            (9, "2018-01", "fit", 0),
            (9, "2019-01", "validation", 1),
        ]
    )
    # Area 7 is equidistant from 5 and 9.
    coords = coordinate_frame([(5, 0.0, -0.1), (9, 0.0, 0.1), (7, 0.0, 0.0)])
    donors = runner.eligible_donor_table(learned, split, coords)
    table = runner.complete_assignments(
        np.array([5, 7, 9]), coords, learned, donors
    ).set_index("admin_code")
    assert table.at[7, "donor_admin_code"] == 5
    assert table.at[7, "branch_id"] == "0"


def test_completion_covers_the_whole_universe_including_zero_label_areas():
    s_branch = s_branch_frame({"": [1], "0": [1], "1": []})
    learned = runner.extract_learned_map(s_branch)
    split = split_fixture([(1, "2018-01", "fit", 0), (1, "2019-01", "validation", 1)])
    universe = np.array([1, 2, 3, 4])
    coords = coordinate_frame([(i, 0.0, 0.001 * i) for i in universe])
    donors = runner.eligible_donor_table(learned, split, coords)
    table = runner.complete_assignments(universe, coords, learned, donors)

    assert len(table) == len(universe)
    assert set(table["admin_code"]) == set(int(v) for v in universe)
    for column in ("assignment_source", "donor_admin_code", "donor_distance_km",
                   "model_route", "partition_code", "branch_id"):
        assert column in table.columns, column
    summary = runner.assignment_summary(table)
    assert summary["universe"] == 4 and summary["learned"] == 1
    assert summary["nearest_donor"] + summary["unresolved"] == 3


def test_completion_refuses_a_universe_without_coordinates():
    s_branch = s_branch_frame({"": [1], "0": [1], "1": []})
    learned = runner.extract_learned_map(s_branch)
    split = split_fixture([(1, "2018-01", "fit", 0), (1, "2019-01", "validation", 1)])
    coords = coordinate_frame([(1, 0.0, 0.0)])
    donors = runner.eligible_donor_table(learned, split, coords)
    try:
        runner.complete_assignments(np.array([1, 2]), coords, learned, donors)
    except runner.PipelineError as error:
        assert "reference coordinates" in str(error)
    else:
        raise AssertionError("an area without coordinates cannot be assigned")


def test_partition_codes_are_reversible_and_unresolved_is_minus_one():
    s_branch = s_branch_frame({"": [1, 2], "0": [1], "1": [2]})
    learned = runner.extract_learned_map(s_branch)
    split = split_fixture(
        [
            (1, "2018-01", "fit", 0),
            (1, "2019-01", "validation", 1),
            (2, "2018-01", "fit", 0),
            (2, "2019-01", "validation", 1),
        ]
    )
    coords = coordinate_frame([(1, 0.0, 0.0), (2, 0.0, 0.001), (3, 40.0, 40.0)])
    donors = runner.eligible_donor_table(learned, split, coords)
    table = runner.complete_assignments(np.array([1, 2, 3]), coords, learned, donors)

    codes = {branch: index for index, branch in enumerate(learned.terminal_branches)}
    for _, row in table.iterrows():
        if row["assignment_source"] == "unresolved":
            assert row["partition_code"] == -1
        else:
            assert codes[row["branch_id"]] == row["partition_code"]


# --------------------------------------------------------------------------
# Singleton routing and scoring arithmetic
# --------------------------------------------------------------------------


def test_singleton_rows_route_through_their_own_frozen_assignment():
    frame = pd.DataFrame({"admin_code": [7, 7, 8, 9]})
    assignments = pd.DataFrame(
        {
            "admin_code": [7, 8, 9],
            "branch_id": ["01", "", "1"],
        }
    )
    branches = runner.singleton_branch_ids(frame, assignments)
    assert list(branches) == ["01", "01", "", "1"]


def test_singleton_routing_refuses_an_area_outside_the_frozen_map():
    frame = pd.DataFrame({"admin_code": [7, 11]})
    assignments = pd.DataFrame({"admin_code": [7], "branch_id": ["0"]})
    try:
        runner.singleton_branch_ids(frame, assignments)
    except runner.PipelineError as error:
        assert "outside the frozen map" in str(error)
    else:
        raise AssertionError("every scored area must be in the completed map")


def test_confusion_counts_follow_the_q9a_undefined_rule():
    # A positive denominator with a zero numerator is 0, not undefined.
    zero = runner.confusion_counts([1, 0], [0, 1])
    assert zero["class1_f1"] == 0.0

    # An all-negative cohort with no positive prediction is undefined, with a reason.
    undefined = runner.confusion_counts([0, 0], [0, 0])
    assert undefined["class1_f1"] is None
    assert undefined["undefined_reason"]
    assert undefined["tn"] == 2

    perfect = runner.confusion_counts([1, 1, 0], [1, 1, 0])
    assert perfect["class1_f1"] == 1.0 and perfect["class1_precision"] == 1.0


def test_decision_threshold_is_strict_so_an_exact_half_is_class_zero():
    probabilities = np.array([0.49, 0.5, 0.51])
    predicted = (probabilities > runner.DECISION_THRESHOLD).astype(int)
    assert list(predicted) == [0, 0, 1]


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
