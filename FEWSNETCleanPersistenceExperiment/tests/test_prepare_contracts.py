"""Contract tests for FEWSNETCleanPersistenceExperiment/prepare_data.py.

These exercise scientific boundaries that a defect could silently cross:
calendar/origin alignment, complete-window and zero-variance rules, observed-phase
and persistence availability, exact source joins and WB distance ties, AEZ parsing,
fitting-row-only imputation, and the frozen ordered schemas. Small synthetic
fixtures cover the boundaries; the pinned real-source preflight is exercised by
`test_real_source_preflight` when FEWS_RUN_REAL_PREFLIGHT=1.
"""

from __future__ import annotations

import copy
import csv
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prepare_data as pdat  # noqa: E402


# --------------------------------------------------------------------------------------
# Synthetic master / ledger fixtures
# --------------------------------------------------------------------------------------

SYNTHETIC_AREAS: Tuple[int, ...] = (7, 11, 23)
LABEL_MONTHS: Tuple[int, ...] = (2, 6, 10)
LABEL_YEARS: Tuple[int, ...] = tuple(range(2012, 2025))


def _month_pairs() -> List[Tuple[int, int]]:
    return [(year, month) for year in range(2010, 2025) for month in range(1, 13)]


def _default_phase(area: int, year: int, month: int) -> Optional[float]:
    if year not in LABEL_YEARS or month not in LABEL_MONTHS:
        return None
    # Deterministic mix of crisis and non-crisis observations.
    value = 1 + ((area + year + month) % 5)
    return float(value)


def synthetic_master_rows(
    overrides: Optional[Dict[Tuple[int, str], Dict[str, object]]] = None,
    extra_rows: Optional[List[Dict[str, object]]] = None,
) -> List[Dict[str, object]]:
    overrides = overrides or {}
    rows: List[Dict[str, object]] = []
    for area in SYNTHETIC_AREAS:
        for year, month in _month_pairs():
            date = f"{year:04d}-{month:02d}"
            m_index = pdat.month_index(year, month)
            row: Dict[str, object] = {name: "" for name in pdat.MASTER_HEADER}
            row.update({
                "unit_name": f"unit {area}",
                "ADMIN0": "Country", "ADMIN1": "R1", "ADMIN2": "R2", "ADMIN3": "",
                "FEWSNET_admin_code": str(area), "ISO": "XX", "ISO3": "XXX",
                "lat": 10.0 + area * 0.1, "lon": 30.0 + area * 0.1,
                "date": date, "month": month,
            })
            for field in pdat.AEZ_FIELDS:
                row[field] = "false"
            row[pdat.AEZ_FIELDS[SYNTHETIC_AREAS.index(area) % len(pdat.AEZ_FIELDS)]] = "true"
            for field in ("crop", "range", "distance_to_river", "elevation", "market_access",
                          "ruggedness", "slope", "sg_cec_5-15cm", "sg_cfvo_5-15cm",
                          "sg_nitrogen_5-15cm", "sg_phh2o_5-15cm", "sg_soc_5-15cm"):
                row[field] = float(area)
            for field in pdat.MASTER_MONTHLY_FIELDS:
                row[field] = float(m_index)
            row["EVI"] = float(m_index) / 10.0
            row["market_distance"] = 2.0
            row["Rainf_f_tavg_mean"] = float(m_index % 7)
            for field in ("event_count_battles", "event_count_explosions", "event_count_violence"):
                row[field] = 1.0
            row["FAO_price"] = float(m_index)
            row["WFP_Price"] = float(m_index)
            row["WFP_Price_std"] = 0.5
            for field in ("CPI", "GDP", "CC", "gini"):
                row[field] = float(year)
            # Population varies by month so D48's last-valid-month rule is observable.
            row["pop"] = 1000.0 + year + month / 100.0
            phase = _default_phase(area, year, month)
            if phase is not None:
                row["fews_ipc"] = phase
                row["fews_ipc_crisis"] = 1 if phase >= 3 else 0
            override = overrides.get((area, date))
            if override:
                row.update(override)
            rows.append(row)
    if extra_rows:
        rows.extend(extra_rows)
    return rows


def write_master_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pdat.MASTER_HEADER))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ledger_rows_from_master(
    rows: Sequence[Dict[str, object]],
    overrides: Optional[Dict[Tuple[int, str], Dict[str, object]]] = None,
) -> List[Dict[str, object]]:
    overrides = overrides or {}
    out: List[Dict[str, object]] = []
    for row in rows:
        area = int(row["FEWSNET_admin_code"])
        date = str(row["date"])
        year, month = int(date[:4]), int(date[5:7])
        ledger_row = {name: "" for name in pdat.LEDGER_HEADER}
        ledger_row.update({
            "country": "Country",
            "admin_code": str(area),
            "year_month": f"{year:04d}_{month:02d}",
            "year": year,
            "month": month,
            "fews_ipc": row["fews_ipc"],
            "pop": row["pop"],
            "pop_source": "GPW",
            "admin_name": f"unit {area}",
        })
        override = overrides.get((area, date))
        if override:
            ledger_row.update(override)
        out.append(ledger_row)
    return out


def write_ledger_csv(path: Path, rows: Sequence[Dict[str, object]],
                     trailing_raw_lines: Sequence[str] = ()) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pdat.LEDGER_HEADER))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        for raw in trailing_raw_lines:
            handle.write(raw + "\r\n")


class SyntheticPanel:
    """Synthetic master/ledger pair plus prepared sources for boundary tests."""

    def __init__(
        self,
        directory: Path,
        master_overrides: Optional[Dict[Tuple[int, str], Dict[str, object]]] = None,
        ledger_overrides: Optional[Dict[Tuple[int, str], Dict[str, object]]] = None,
        master_extra_rows: Optional[List[Dict[str, object]]] = None,
        ledger_trailing: Sequence[str] = (),
    ) -> None:
        self.directory = directory
        self.master_path = directory / "master.csv"
        self.ledger_path = directory / "ledger.csv"
        rows = synthetic_master_rows(master_overrides, master_extra_rows)
        write_master_csv(self.master_path, rows)
        write_ledger_csv(
            self.ledger_path,
            ledger_rows_from_master(rows, ledger_overrides),
            trailing_raw_lines=ledger_trailing,
        )

    def load(self):
        master = pdat.load_master_grid(self.master_path, verbose=False)
        ledger = pdat.load_ledger(self.ledger_path, master.areas, verbose=False)
        return master, ledger


def prepared_from_master(master: "pdat.MasterGrid", ledger: "pdat.LedgerData",
                         enso: Optional[np.ndarray] = None,
                         bloomberg: Optional[Dict[str, np.ndarray]] = None,
                         wb: Optional[Dict[str, np.ndarray]] = None,
                         coastline: Optional[np.ndarray] = None) -> "pdat.PreparedSources":
    n_areas = master.n_areas
    if enso is None:
        enso = np.arange(pdat.N_GRID_MONTHS, dtype=np.float64) / 100.0
    if bloomberg is None:
        bloomberg = {
            field: np.arange(pdat.N_GRID_MONTHS, dtype=np.float64)
            for field in pdat.BBG_FIELDS
        }
    if wb is None:
        wb = {
            field: np.tile(
                np.arange(pdat.N_GRID_MONTHS, dtype=np.float64), (n_areas, 1)
            )
            for field in pdat.WB_FIELDS
        }
    if coastline is None:
        coastline = np.full(n_areas, -5.0)
    return pdat.assemble_prepared_sources(
        master=master, ledger=ledger, enso=enso, bloomberg=bloomberg,
        wb_values=wb, coastline=coastline, extra_report={},
    )


class SyntheticPanelTestCase(unittest.TestCase):
    """Base class providing one prepared synthetic panel per test class."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        panel = SyntheticPanel(Path(cls._tmp.name))
        cls.master, cls.ledger = panel.load()
        cls.sources = prepared_from_master(cls.master, cls.ledger)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def row_for(self, area: int, target: str) -> int:
        area_position = int(np.searchsorted(self.sources.areas, area))
        t_index = pdat.month_index(int(target[:4]), int(target[5:7]))
        matches = np.flatnonzero(
            (self.sources.target_area_idx == area_position)
            & (self.sources.target_month_idx == t_index)
        )
        self.assertEqual(matches.size, 1, f"expected one labeled row for {area}/{target}")
        return int(matches[0])


# --------------------------------------------------------------------------------------
# D63: target/label validation, artifact handling and reconciliation
# --------------------------------------------------------------------------------------


class TestLabelContract(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_valid_panel_reconciles_and_counts_labels(self) -> None:
        master, ledger = SyntheticPanel(self.directory).load()
        cross = pdat.reconcile_master_and_ledger(master, ledger)
        expected_labels = len(SYNTHETIC_AREAS) * len(LABEL_YEARS) * len(LABEL_MONTHS)
        self.assertEqual(master.report["master_valid_labels"], expected_labels)
        self.assertEqual(cross["cross_same_key_agree"], expected_labels)
        self.assertEqual(cross["cross_same_key_conflict"], 0)
        self.assertEqual(cross["cross_same_key_one_missing"], 0)
        self.assertEqual(cross["cross_ledger_absent_master_observed"], 0)
        self.assertEqual(master.report["master_duplicate_keys"], 0)
        self.assertTrue(master.report["master_complete_monthly_grid"])
        # 1[phase >= 3] holds and the crisis counts follow the phase counts.
        phases = master.report["master_phase_counts"]
        self.assertEqual(
            master.report["master_crisis_one"], phases[3] + phases[4] + phases[5]
        )
        self.assertEqual(master.report["master_crisis_zero"], phases[1] + phases[2])

    def test_non_integral_phase_stops_preflight(self) -> None:
        panel = SyntheticPanel(
            self.directory,
            master_overrides={(7, "2014-02"): {"fews_ipc": 2.5, "fews_ipc_crisis": 0}},
        )
        with self.assertRaisesRegex(pdat.PreflightError, "non-integral"):
            panel.load()

    def test_out_of_range_phase_stops_preflight(self) -> None:
        panel = SyntheticPanel(
            self.directory,
            master_overrides={(7, "2014-02"): {"fews_ipc": 6, "fews_ipc_crisis": 1}},
        )
        with self.assertRaisesRegex(pdat.PreflightError, "outside the valid 1..5 range"):
            panel.load()

    def test_binary_label_inconsistent_with_phase_stops_preflight(self) -> None:
        panel = SyntheticPanel(
            self.directory,
            master_overrides={(7, "2014-02"): {"fews_ipc": 4, "fews_ipc_crisis": 0}},
        )
        with self.assertRaisesRegex(pdat.PreflightError, r"fews_ipc_crisis == 1\[fews_ipc >= 3\]"):
            panel.load()

    def test_missingness_disagreement_stops_preflight(self) -> None:
        panel = SyntheticPanel(
            self.directory,
            master_overrides={(7, "2014-02"): {"fews_ipc_crisis": ""}},
        )
        with self.assertRaisesRegex(pdat.PreflightError, "missingness disagree"):
            panel.load()

    def test_duplicate_master_key_stops_preflight(self) -> None:
        rows = synthetic_master_rows()
        duplicate = copy.deepcopy(rows[0])
        panel_dir = self.directory
        write_master_csv(panel_dir / "master.csv", list(rows) + [duplicate])
        with self.assertRaisesRegex(pdat.PreflightError, "duplicate canonical area/month"):
            pdat.load_master_grid(panel_dir / "master.csv", verbose=False)

    def test_master_ledger_phase_conflict_stops_preflight(self) -> None:
        panel = SyntheticPanel(
            self.directory,
            ledger_overrides={(7, "2014-02"): {"fews_ipc": 1.0}},
        )
        master, ledger = pdat.load_master_grid(panel.master_path, verbose=False), None
        ledger = pdat.load_ledger(panel.ledger_path, master.areas, verbose=False)
        with self.assertRaisesRegex(pdat.PreflightError, "phase conflicts"):
            pdat.reconcile_master_and_ledger(master, ledger)

    def test_ledger_missing_key_for_observed_master_label_stops_preflight(self) -> None:
        rows = synthetic_master_rows()
        ledger_rows = [
            row for row in ledger_rows_from_master(rows)
            if not (row["admin_code"] == "7" and row["year_month"] == "2014_02")
        ]
        write_master_csv(self.directory / "master.csv", rows)
        write_ledger_csv(self.directory / "ledger.csv", ledger_rows)
        master = pdat.load_master_grid(self.directory / "master.csv", verbose=False)
        ledger = pdat.load_ledger(self.directory / "ledger.csv", master.areas, verbose=False)
        with self.assertRaisesRegex(pdat.PreflightError, "no ledger key"):
            pdat.reconcile_master_and_ledger(master, ledger)

    def test_ledger_redundant_date_fields_must_agree(self) -> None:
        panel = SyntheticPanel(
            self.directory, ledger_overrides={(7, "2014-02"): {"month": 3}},
        )
        master = pdat.load_master_grid(panel.master_path, verbose=False)
        with self.assertRaisesRegex(pdat.PreflightError, "redundant date fields"):
            pdat.load_ledger(panel.ledger_path, master.areas, verbose=False)

    def test_ledger_duplicate_key_stops_preflight(self) -> None:
        rows = synthetic_master_rows()
        ledger_rows = ledger_rows_from_master(rows)
        ledger_rows.append(copy.deepcopy(ledger_rows[0]))
        write_master_csv(self.directory / "master.csv", rows)
        write_ledger_csv(self.directory / "ledger.csv", ledger_rows)
        master = pdat.load_master_grid(self.directory / "master.csv", verbose=False)
        with self.assertRaisesRegex(pdat.PreflightError, "Duplicate ledger canonical key"):
            pdat.load_ledger(self.directory / "ledger.csv", master.areas, verbose=False)

    def test_terminal_artifact_excluded_only_at_the_verified_line(self) -> None:
        rows = synthetic_master_rows()
        ledger_rows = ledger_rows_from_master(rows)
        write_master_csv(self.directory / "master.csv", rows)
        write_ledger_csv(
            self.directory / "ledger.csv", ledger_rows,
            trailing_raw_lines=[pdat.LEDGER_TERMINAL_ARTIFACT],
        )
        master = pdat.load_master_grid(self.directory / "master.csv", verbose=False)
        expected_line = len(ledger_rows) + 2  # header + data rows + artifact line
        original = pdat.EXPECTED_PREFLIGHT["ledger_artifact_physical_line"]
        try:
            pdat.EXPECTED_PREFLIGHT["ledger_artifact_physical_line"] = expected_line
            ledger = pdat.load_ledger(self.directory / "ledger.csv", master.areas, verbose=False)
            self.assertEqual(ledger.report["ledger_malformed_records"], 1)
            record = ledger.report["ledger_excluded_records"][0]
            self.assertEqual(record["physical_line"], expected_line)
            self.assertEqual(record["raw_value"], pdat.LEDGER_TERMINAL_ARTIFACT)
            self.assertEqual(record["field_count"], 1)
            self.assertEqual(
                ledger.report["ledger_valid_key_records"], len(ledger_rows)
            )
            # The same artifact at any other physical line is not excluded.
            pdat.EXPECTED_PREFLIGHT["ledger_artifact_physical_line"] = expected_line + 5
            with self.assertRaisesRegex(pdat.PreflightError, "malformed ledger record"):
                pdat.load_ledger(self.directory / "ledger.csv", master.areas, verbose=False)
        finally:
            pdat.EXPECTED_PREFLIGHT["ledger_artifact_physical_line"] = original

    def test_other_malformed_ledger_record_stops_preflight(self) -> None:
        rows = synthetic_master_rows()
        ledger_rows = ledger_rows_from_master(rows)
        write_master_csv(self.directory / "master.csv", rows)
        write_ledger_csv(
            self.directory / "ledger.csv", ledger_rows,
            trailing_raw_lines=["Country,7,2024_10,2024,10"],
        )
        master = pdat.load_master_grid(self.directory / "master.csv", verbose=False)
        with self.assertRaisesRegex(pdat.PreflightError, "malformed ledger record"):
            pdat.load_ledger(self.directory / "ledger.csv", master.areas, verbose=False)

    def test_invalid_ledger_phase_stops_preflight(self) -> None:
        panel = SyntheticPanel(
            self.directory, ledger_overrides={(7, "2011-02"): {"fews_ipc": 0}},
        )
        master = pdat.load_master_grid(panel.master_path, verbose=False)
        with self.assertRaisesRegex(pdat.PreflightError, "invalid observed"):
            pdat.load_ledger(panel.ledger_path, master.areas, verbose=False)

    def test_aez_unknown_token_stops_preflight(self) -> None:
        panel = SyntheticPanel(
            self.directory, master_overrides={(7, "2014-02"): {"AEZ_10000": "TRUE"}},
        )
        with self.assertRaisesRegex(pdat.PreflightError, "unexpected boolean token"):
            panel.load()

    def test_static_field_conflict_stops_preflight(self) -> None:
        panel = SyntheticPanel(
            self.directory, master_overrides={(7, "2014-02"): {"elevation": 999.0}},
        )
        with self.assertRaisesRegex(pdat.PreflightError, "not constant within area"):
            panel.load()

    def test_conflicting_annual_value_stops_preflight(self) -> None:
        panel = SyntheticPanel(
            self.directory, master_overrides={(7, "2014-02"): {"CPI": 1.0}},
        )
        with self.assertRaisesRegex(pdat.PreflightError, "conflicting annual values"):
            panel.load()


class TestAezAndPopulation(SyntheticPanelTestCase):
    def test_aez_parsed_to_one_and_zero_and_zero_is_not_missing(self) -> None:
        first_area = int(np.searchsorted(self.sources.areas, SYNTHETIC_AREAS[0]))
        member = pdat.AEZ_FIELDS[0]
        nonmember = pdat.AEZ_FIELDS[1]
        self.assertEqual(self.sources.static[member][first_area], 1.0)
        self.assertEqual(self.sources.static[nonmember][first_area], 0.0)
        row = self.row_for(SYNTHETIC_AREAS[0], "2016-06")
        area_idx, _, o_idx = pdat.origin_alignment(self.sources, 4, np.array([row]))
        block_e = pdat.build_block_e(self.sources, area_idx, o_idx)
        columns = list(pdat.BLOCK_E_COLUMNS)
        self.assertEqual(block_e[0, columns.index(f"{nonmember}_missing")], 0.0)
        self.assertNotIn(f"{nonmember}_age_months", columns)

    def test_population_uses_last_valid_month_of_the_preceding_year(self) -> None:
        area_position = int(np.searchsorted(self.sources.areas, SYNTHETIC_AREAS[0]))
        year = 2015
        expected = 1000.0 + year + 12 / 100.0
        value = self.sources.annual["pop"][area_position, pdat.year_index(year)]
        self.assertAlmostEqual(float(value), expected, places=9)
        self.assertEqual(
            int(self.master.pop_selected_month[area_position, pdat.year_index(year)]), 12
        )
        # Origins in 2016 use the 2015 selection, including January and December.
        for target, horizon in (("2016-06", 4), ("2016-10", 4)):
            row = self.row_for(SYNTHETIC_AREAS[0], target)
            area_idx, _, o_idx = pdat.origin_alignment(self.sources, horizon, np.array([row]))
            base = pdat.build_base_block(self.sources, area_idx, o_idx, pdat.UPDATED_BASE_FIELDS)
            column = list(pdat.UPDATED_BASE_FIELDS).index("pop")
            self.assertAlmostEqual(float(base[0, column]), expected, places=9)


# --------------------------------------------------------------------------------------
# Calendar alignment, windows and blocks
# --------------------------------------------------------------------------------------


class TestOriginAlignment(SyntheticPanelTestCase):
    def test_origin_is_target_minus_horizon(self) -> None:
        row = np.array([self.row_for(SYNTHETIC_AREAS[0], "2020-02")])
        for horizon, expected in ((4, "2019-10"), (8, "2019-06"), (12, "2019-02")):
            _, t_idx, o_idx = pdat.origin_alignment(self.sources, horizon, row)
            self.assertEqual(pdat.month_label(int(t_idx[0])), "2020-02")
            self.assertEqual(pdat.month_label(int(o_idx[0])), expected)

    def test_unapproved_horizon_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pdat.origin_alignment(self.sources, 6, np.array([0]))

    def test_monthly_base_uses_exact_origin_without_backfill(self) -> None:
        area = SYNTHETIC_AREAS[0]
        area_position = int(np.searchsorted(self.sources.areas, area))
        target = "2020-06"
        o_index = pdat.month_index(2020, 2)
        self.sources.area_monthly["EVI"][area_position, o_index] = np.nan
        try:
            row = np.array([self.row_for(area, target)])
            area_idx, _, o_idx = pdat.origin_alignment(self.sources, 4, row)
            base = pdat.build_base_block(self.sources, area_idx, o_idx, pdat.UPDATED_BASE_FIELDS)
            column = list(pdat.UPDATED_BASE_FIELDS).index("EVI")
            self.assertTrue(math.isnan(float(base[0, column])),
                            "a missing origin-month value must not be backfilled")
            # E still reports the older valid value's age without filling BASE.
            block_e = pdat.build_block_e(self.sources, area_idx, o_idx)
            columns = list(pdat.BLOCK_E_COLUMNS)
            self.assertEqual(block_e[0, columns.index("EVI_missing")], 1.0)
            self.assertEqual(block_e[0, columns.index("EVI_age_months")], 1.0)
        finally:
            self.sources.area_monthly["EVI"][area_position, o_index] = float(o_index) / 10.0

    def test_annual_alignment_uses_previous_year_including_january(self) -> None:
        area = SYNTHETIC_AREAS[1]
        column = list(pdat.UPDATED_BASE_FIELDS).index("CPI")
        # Target 2016-02 with H=12 has origin 2015-02 -> reference year 2014.
        row = np.array([self.row_for(area, "2016-02")])
        area_idx, _, o_idx = pdat.origin_alignment(self.sources, 12, row)
        base = pdat.build_base_block(self.sources, area_idx, o_idx, pdat.UPDATED_BASE_FIELDS)
        self.assertEqual(float(base[0, column]), 2014.0)
        # Target 2016-02 with H=4 has a January 2016 origin -> still reference year 2015.
        # The synthetic label calendar has no January targets, so construct the origin directly.
        january_origin = np.array([pdat.month_index(2016, 1)])
        base_january = pdat.build_base_block(
            self.sources, area_idx, january_origin, pdat.UPDATED_BASE_FIELDS
        )
        self.assertEqual(float(base_january[0, column]), 2015.0)

    def test_annual_e_age_is_anchored_on_december_of_the_reference_year(self) -> None:
        area_idx = np.array([int(np.searchsorted(self.sources.areas, SYNTHETIC_AREAS[0]))])
        columns = list(pdat.BLOCK_E_COLUMNS)
        january = pdat.build_block_e(self.sources, area_idx, np.array([pdat.month_index(2016, 1)]))
        december = pdat.build_block_e(self.sources, area_idx, np.array([pdat.month_index(2016, 12)]))
        self.assertEqual(january[0, columns.index("CPI_age_months")], 1.0)
        self.assertEqual(december[0, columns.index("CPI_age_months")], 12.0)
        # 2010 origins have no eligible 2009 annual value: flag 1, age missing.
        early = pdat.build_block_e(self.sources, area_idx, np.array([pdat.month_index(2010, 5)]))
        self.assertEqual(early[0, columns.index("CPI_missing")], 1.0)
        self.assertTrue(math.isnan(float(early[0, columns.index("CPI_age_months")])))

    def test_static_fields_have_a_flag_but_no_age(self) -> None:
        columns = list(pdat.BLOCK_E_COLUMNS)
        for field in ("lat", "elevation", pdat.COASTLINE_FIELD):
            self.assertIn(f"{field}_missing", columns)
            self.assertNotIn(f"{field}_age_months", columns)
        self.assertEqual(
            sum(1 for column in columns if column.endswith("_age_months")), 54
        )
        self.assertEqual(sum(1 for column in columns if column.endswith("_missing")), 86)


class TestBlockBAndC(SyntheticPanelTestCase):
    def setUp(self) -> None:
        self.area = SYNTHETIC_AREAS[2]
        self.area_position = int(np.searchsorted(self.sources.areas, self.area))
        self.row = np.array([self.row_for(self.area, "2020-06")])
        self.area_idx, _, self.o_idx = pdat.origin_alignment(self.sources, 4, self.row)
        self.b_columns = list(pdat.BLOCK_B_COLUMNS)
        self.c_columns = list(pdat.BLOCK_C_COLUMNS)

    def test_window_mean_and_population_sd_end_at_the_origin(self) -> None:
        block = pdat.build_block_b(self.sources, self.area_idx, self.o_idx)
        o_index = int(self.o_idx[0])
        expected_window = np.array([
            self.sources.area_monthly["EVI"][self.area_position, o_index - offset]
            for offset in (2, 1, 0)
        ])
        self.assertAlmostEqual(
            float(block[0, self.b_columns.index("EVI_mean_3m")]),
            float(expected_window.mean()), places=12,
        )
        self.assertAlmostEqual(
            float(block[0, self.b_columns.index("EVI_sd_3m")]),
            float(expected_window.std()), places=12,
        )

    def test_a_single_missing_month_invalidates_the_whole_window(self) -> None:
        o_index = int(self.o_idx[0])
        original = self.sources.area_monthly["EVI"][self.area_position, o_index - 5]
        self.sources.area_monthly["EVI"][self.area_position, o_index - 5] = np.nan
        try:
            block = pdat.build_block_b(self.sources, self.area_idx, self.o_idx)
            self.assertFalse(math.isnan(float(block[0, self.b_columns.index("EVI_mean_3m")])))
            self.assertTrue(math.isnan(float(block[0, self.b_columns.index("EVI_mean_6m")])))
            self.assertTrue(math.isnan(float(block[0, self.b_columns.index("EVI_mean_12m")])))
        finally:
            self.sources.area_monthly["EVI"][self.area_position, o_index - 5] = original

    def test_missing_conflict_month_is_not_treated_as_zero(self) -> None:
        o_index = int(self.o_idx[0])
        field = "event_count_battles"
        original = self.sources.area_monthly[field][self.area_position, o_index - 1]
        self.sources.area_monthly[field][self.area_position, o_index - 1] = np.nan
        try:
            block = pdat.build_block_b(self.sources, self.area_idx, self.o_idx)
            self.assertTrue(math.isnan(float(block[0, self.b_columns.index(f"{field}_sum_3m")])))
        finally:
            self.sources.area_monthly[field][self.area_position, o_index - 1] = original

    def test_count_sum_uses_all_three_months_when_complete(self) -> None:
        block = pdat.build_block_b(self.sources, self.area_idx, self.o_idx)
        self.assertEqual(
            float(block[0, self.b_columns.index("event_count_battles_sum_3m")]), 3.0
        )

    def test_differences_need_only_their_endpoints(self) -> None:
        o_index = int(self.o_idx[0])
        field = "gpp_mean"
        original = self.sources.area_monthly[field][self.area_position, o_index - 2].copy()
        self.sources.area_monthly[field][self.area_position, o_index - 2] = np.nan
        try:
            block = pdat.build_block_c(self.sources, self.area_idx, self.o_idx)
            self.assertEqual(float(block[0, self.c_columns.index(f"{field}_chg_3m")]), 3.0)
            self.assertEqual(float(block[0, self.c_columns.index(f"{field}_chg_12m")]), 12.0)
            # The standardized deviation needs the complete preceding window.
            self.assertTrue(math.isnan(float(block[0, self.c_columns.index(f"{field}_zdev_12m")])))
        finally:
            self.sources.area_monthly[field][self.area_position, o_index - 2] = original

    def test_missing_difference_endpoint_yields_missing(self) -> None:
        o_index = int(self.o_idx[0])
        field = "gpp_mean"
        original = self.sources.area_monthly[field][self.area_position, o_index - 3].copy()
        self.sources.area_monthly[field][self.area_position, o_index - 3] = np.nan
        try:
            block = pdat.build_block_c(self.sources, self.area_idx, self.o_idx)
            self.assertTrue(math.isnan(float(block[0, self.c_columns.index(f"{field}_chg_3m")])))
            self.assertEqual(float(block[0, self.c_columns.index(f"{field}_chg_12m")]), 12.0)
        finally:
            self.sources.area_monthly[field][self.area_position, o_index - 3] = original

    def test_zero_reference_variance_yields_missing_without_epsilon(self) -> None:
        o_index = int(self.o_idx[0])
        field = "Food_CPI"
        original = self.sources.area_monthly[field][
            self.area_position, o_index - 12:o_index + 1
        ].copy()
        self.sources.area_monthly[field][self.area_position, o_index - 12:o_index] = 5.0
        self.sources.area_monthly[field][self.area_position, o_index] = 9.0
        try:
            block = pdat.build_block_c(self.sources, self.area_idx, self.o_idx)
            self.assertTrue(math.isnan(float(block[0, self.c_columns.index(f"{field}_zdev_12m")])))
        finally:
            self.sources.area_monthly[field][
                self.area_position, o_index - 12:o_index + 1
            ] = original

    def test_standardized_deviation_excludes_the_current_month(self) -> None:
        o_index = int(self.o_idx[0])
        field = "Tair_f_tavg_mean"
        window = np.array([
            self.sources.area_monthly[field][self.area_position, o_index - offset]
            for offset in range(12, 0, -1)
        ])
        current = self.sources.area_monthly[field][self.area_position, o_index]
        expected = (current - window.mean()) / window.std()
        block = pdat.build_block_c(self.sources, self.area_idx, self.o_idx)
        self.assertAlmostEqual(
            float(block[0, self.c_columns.index(f"{field}_zdev_12m")]), float(expected), places=12
        )


class TestBlockDAndA(SyntheticPanelTestCase):
    def test_block_d_products_and_missing_operands(self) -> None:
        area = SYNTHETIC_AREAS[0]
        area_position = int(np.searchsorted(self.sources.areas, area))
        row = np.array([self.row_for(area, "2020-10")])
        area_idx, _, o_idx = pdat.origin_alignment(self.sources, 8, row)
        o_index = int(o_idx[0])
        block = pdat.build_block_d(self.sources, area_idx, o_idx)
        z_rain = pdat._trailing_standardized_deviation(
            self.sources, "Rainf_f_tavg_mean", area_idx, o_idx
        )
        z_evi = pdat._trailing_standardized_deviation(self.sources, "EVI", area_idx, o_idx)
        self.assertAlmostEqual(float(block[0, 0]), float(z_rain[0] * z_evi[0]), places=12)
        # conflict_events_3m sums three categories over three months: 3 x 3 x 1.0 = 9.
        self.assertAlmostEqual(float(block[0, 1]), float(z_rain[0] * 9.0), places=12)
        inflation = self.sources.area_monthly["food_inflation_wb"][area_position, o_index]
        self.assertAlmostEqual(float(block[0, 2]), float(inflation * 9.0), places=12)
        self.assertAlmostEqual(float(block[0, 3]), float(inflation * 2.0), places=12)

        # One missing category-month invalidates the conflict operand and both products.
        original = self.sources.area_monthly["event_count_violence"][area_position, o_index - 2]
        self.sources.area_monthly["event_count_violence"][area_position, o_index - 2] = np.nan
        try:
            block = pdat.build_block_d(self.sources, area_idx, o_idx)
            self.assertTrue(math.isnan(float(block[0, 1])))
            self.assertTrue(math.isnan(float(block[0, 2])))
            self.assertFalse(math.isnan(float(block[0, 3])))
        finally:
            self.sources.area_monthly["event_count_violence"][area_position, o_index - 2] = original

    def test_block_d_market_distance_operand_is_the_master_field(self) -> None:
        area = SYNTHETIC_AREAS[0]
        area_position = int(np.searchsorted(self.sources.areas, area))
        row = np.array([self.row_for(area, "2020-10")])
        area_idx, _, o_idx = pdat.origin_alignment(self.sources, 4, row)
        o_index = int(o_idx[0])
        self.sources.area_monthly["market_distance"][area_position, o_index] = 11.0
        try:
            block = pdat.build_block_d(self.sources, area_idx, o_idx)
            inflation = self.sources.area_monthly["food_inflation_wb"][area_position, o_index]
            self.assertAlmostEqual(float(block[0, 3]), float(inflation * 11.0), places=12)
        finally:
            self.sources.area_monthly["market_distance"][area_position, o_index] = 2.0

    def test_block_a_history_recency_and_flags(self) -> None:
        area = SYNTHETIC_AREAS[1]
        area_position = int(np.searchsorted(self.sources.areas, area))
        row = np.array([self.row_for(area, "2016-02")])
        area_idx, t_idx, o_idx = pdat.origin_alignment(self.sources, 4, row)
        block = pdat.build_block_a(self.sources, area_idx, t_idx, o_idx)
        columns = list(pdat.BLOCK_A_COLUMNS)
        target_month = 2
        self.assertAlmostEqual(
            float(block[0, columns.index("target_month_sin")]),
            math.sin(2 * math.pi * (target_month - 1) / 12), places=12,
        )
        # Origin 2015-10 is an observed label month, so history has age zero.
        o_index = int(o_idx[0])
        self.assertEqual(pdat.month_label(o_index), "2015-10")
        self.assertEqual(float(block[0, columns.index("last_observed_ipc_age_months")]), 0.0)
        self.assertEqual(
            float(block[0, columns.index("last_observed_ipc_phase")]),
            float(self.sources.observed_phase[area_position, o_index]),
        )
        self.assertEqual(float(block[0, columns.index("no_observed_ipc_history")]), 0.0)

    def test_block_a_history_exists_even_when_persistence_is_unavailable(self) -> None:
        area = SYNTHETIC_AREAS[1]
        area_position = int(np.searchsorted(self.sources.areas, area))
        row_index = self.row_for(area, "2016-06")
        rows = np.array([row_index])
        area_idx, t_idx, o_idx = pdat.origin_alignment(self.sources, 4, rows)
        o_index = int(o_idx[0])
        self.assertEqual(pdat.month_label(o_index), "2016-02")
        original = self.sources.observed_phase[area_position, o_index]
        self.sources.observed_phase[area_position, o_index] = np.nan
        self.sources.observed_crisis[area_position, o_index] = np.nan
        try:
            metadata = pdat.build_row_metadata(self.sources, 4, rows)
            self.assertFalse(bool(metadata.loc[0, "persistence_available"]))
            self.assertTrue(pd.isna(metadata.loc[0, "persistence"]))
            block = pdat.build_block_a(self.sources, area_idx, t_idx, o_idx)
            columns = list(pdat.BLOCK_A_COLUMNS)
            self.assertEqual(float(block[0, columns.index("no_observed_ipc_history")]), 0.0)
            self.assertGreater(float(block[0, columns.index("last_observed_ipc_age_months")]), 0.0)
        finally:
            self.sources.observed_phase[area_position, o_index] = original
            self.sources.observed_crisis[area_position, o_index] = (
                np.nan if math.isnan(float(original)) else float(original >= 3)
            )

    def test_persistence_uses_the_exact_origin_observation(self) -> None:
        area = SYNTHETIC_AREAS[0]
        area_position = int(np.searchsorted(self.sources.areas, area))
        rows = np.array([self.row_for(area, "2020-06")])
        metadata = pdat.build_row_metadata(self.sources, 4, rows)
        self.assertEqual(metadata.loc[0, "origin_month"], "2020-02")
        expected_phase = self.sources.observed_phase[area_position, pdat.month_index(2020, 2)]
        self.assertEqual(
            float(metadata.loc[0, "persistence"]), float(expected_phase >= 3)
        )
        # An origin month with no observed label has no persistence at all.
        rows = np.array([self.row_for(area, "2020-02")])
        metadata = pdat.build_row_metadata(self.sources, 4, rows)
        self.assertEqual(metadata.loc[0, "origin_month"], "2019-10")
        self.assertTrue(bool(metadata.loc[0, "persistence_available"]))
        metadata = pdat.build_row_metadata(self.sources, 8, rows)
        self.assertEqual(metadata.loc[0, "origin_month"], "2019-06")
        self.assertTrue(bool(metadata.loc[0, "persistence_available"]))


class TestReferenceArm(SyntheticPanelTestCase):
    def test_history_offsets_use_the_origin_not_the_target(self) -> None:
        area = SYNTHETIC_AREAS[0]
        area_position = int(np.searchsorted(self.sources.areas, area))
        rows = np.array([self.row_for(area, "2022-02")])
        matrix, columns = pdat.build_reference_matrix(self.sources, 12, rows)
        columns = list(columns)
        o_index = pdat.month_index(2021, 2)
        for lag in pdat.REFERENCE_IPC_LAGS:
            expected = self.sources.observed_phase[area_position, o_index - lag]
            actual = matrix[0, columns.index(f"fews_ipc_lag_{lag}")]
            if math.isnan(float(expected)):
                self.assertTrue(math.isnan(float(actual)))
            else:
                self.assertEqual(float(actual), float(expected))
        # T-4 for H=12 would be 2021-10, which is after the origin and must not appear.
        future = self.sources.observed_phase[area_position, pdat.month_index(2021, 10)]
        self.assertFalse(
            np.any(np.isclose(matrix[0, [columns.index(f"fews_ipc_lag_{lag}")
                                         for lag in pdat.REFERENCE_IPC_LAGS]], future))
            and not math.isnan(float(future))
            and float(future) not in {
                float(self.sources.observed_phase[area_position, o_index - lag])
                for lag in pdat.REFERENCE_IPC_LAGS
            }
        )

    def test_calendar_indicators_use_the_target_month(self) -> None:
        rows = np.array([self.row_for(SYNTHETIC_AREAS[0], "2022-06")])
        matrix, columns = pdat.build_reference_matrix(self.sources, 12, rows)
        columns = list(columns)
        self.assertEqual(matrix[0, columns.index("year_2022")], 1.0)
        self.assertEqual(matrix[0, columns.index("year_2021")], 0.0)
        self.assertEqual(matrix[0, columns.index("month_6")], 1.0)
        self.assertEqual(matrix[0, columns.index("month_2")], 0.0)

    def test_inherited_sums_exclude_the_origin_month(self) -> None:
        area = SYNTHETIC_AREAS[0]
        area_position = int(np.searchsorted(self.sources.areas, area))
        rows = np.array([self.row_for(area, "2020-06")])
        matrix, columns = pdat.build_reference_matrix(self.sources, 4, rows)
        columns = list(columns)
        o_index = pdat.month_index(2020, 2)
        expected_m4 = sum(
            float(self.sources.area_monthly["WFP_Price"][area_position, o_index - offset])
            for offset in (4, 3, 2, 1)
        )
        self.assertAlmostEqual(
            float(matrix[0, columns.index("WFP_Price_m4")]), expected_m4, places=9
        )
        expected_nightlight = sum(
            float(self.sources.area_monthly["nightlight"][area_position, o_index - offset])
            for offset in range(1, 13)
        )
        self.assertAlmostEqual(
            float(matrix[0, columns.index("nightlight_m12")]), expected_nightlight, places=9
        )
        self.assertAlmostEqual(
            float(matrix[0, columns.index("EVI_l1")]),
            float(self.sources.area_monthly["EVI"][area_position, o_index - 1]), places=12,
        )

    def test_incomplete_sum_window_stays_missing(self) -> None:
        area = SYNTHETIC_AREAS[0]
        area_position = int(np.searchsorted(self.sources.areas, area))
        rows = np.array([self.row_for(area, "2020-06")])
        o_index = pdat.month_index(2020, 2)
        original = self.sources.area_monthly["WFP_Price"][area_position, o_index - 6].copy()
        self.sources.area_monthly["WFP_Price"][area_position, o_index - 6] = np.nan
        try:
            matrix, columns = pdat.build_reference_matrix(self.sources, 4, rows)
            columns = list(columns)
            self.assertFalse(math.isnan(float(matrix[0, columns.index("WFP_Price_m4")])))
            self.assertTrue(math.isnan(float(matrix[0, columns.index("WFP_Price_m12")])))
        finally:
            self.sources.area_monthly["WFP_Price"][area_position, o_index - 6] = original

    def test_reference_keeps_legacy_prices_and_excludes_new_sources(self) -> None:
        columns = set(pdat.REFERENCE_COLUMNS)
        for field in pdat.REFERENCE_ONLY_PRICES:
            self.assertIn(field, columns)
        for field in pdat.ADDITIONAL_SOURCE_FIELDS:
            self.assertNotIn(field, columns)
        for block_column in pdat.BLOCK_A_COLUMNS + pdat.BLOCK_E_COLUMNS:
            self.assertNotIn(block_column, columns)


# --------------------------------------------------------------------------------------
# Additional sources
# --------------------------------------------------------------------------------------


class TestEnso(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, rows: Iterable[Tuple[str, str]]) -> Path:
        path = self.directory / "enso.csv"
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("Date,   Nino Anom 3.4 Index missing value -99.99\n")
            for date, value in rows:
                handle.write(f"{date},{value}\n")
        return path

    def test_both_documented_sentinels_are_missing(self) -> None:
        path = self._write([
            ("2011-01-01", "-9999.000"),
            ("2011-02-01", "-99.99"),
            ("2011-03-01", "-0.75"),
            ("2011-04-01", "0.00"),
        ])
        series, report = pdat.load_enso_series(path)
        self.assertTrue(math.isnan(float(series[pdat.month_index(2011, 1)])))
        self.assertTrue(math.isnan(float(series[pdat.month_index(2011, 2)])))
        self.assertEqual(float(series[pdat.month_index(2011, 3)]), -0.75)
        self.assertEqual(float(series[pdat.month_index(2011, 4)]), 0.0)
        self.assertEqual(report["sentinel_rows_total"], 2)
        self.assertEqual(report["valid_months_on_grid"], 2)

    def test_duplicate_month_stops_preflight(self) -> None:
        path = self._write([("2011-01-01", "0.1"), ("2011-01-01", "0.2")])
        with self.assertRaisesRegex(pdat.PreflightError, "duplicate year-month"):
            pdat.load_enso_series(path)


class TestBloomberg(unittest.TestCase):
    def test_duplicate_month_stops_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            paths = {}
            for key, fields in pdat.BLOOMBERG_FILE_FIELDS.items():
                path = directory / f"{key}.csv"
                frame = pd.DataFrame({"year": [2011, 2011], "month": [1, 1]})
                for field in fields:
                    frame[field] = [1.0, 2.0]
                frame.to_csv(path, index=False)
                paths[key] = path
            with self.assertRaisesRegex(pdat.PreflightError, "duplicate year/month"):
                pdat.load_bloomberg_series(paths)

    def test_entirely_missing_series_is_retained_in_the_schema(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            paths = {}
            for key, fields in pdat.BLOOMBERG_FILE_FIELDS.items():
                path = directory / f"{key}.csv"
                frame = pd.DataFrame({"year": [2011], "month": [1]})
                for field in fields:
                    frame[field] = [np.nan] if field == "bbg_TZTX2_Comdty" else [3.5]
                frame.to_csv(path, index=False)
                paths[key] = path
            series, report = pdat.load_bloomberg_series(paths)
            self.assertIn("bbg_TZTX2_Comdty", series)
            self.assertIn("bbg_TZTX2_Comdty", report["fields_entirely_missing"])
            self.assertEqual(len(series), 18)


class TestWorldBankJoin(unittest.TestCase):
    def _markets(self) -> pd.DataFrame:
        return pd.DataFrame({
            "geo_id": ["gid_B", "gid_A", "gid_far"],
            "year": [2015, 2015, 2015],
            "month": [6, 6, 6],
            "lat": [0.0, 0.0, 5.0],
            "lon": [0.0, 0.0, 0.0],
            "food_price_index_WB": [1.5, np.nan, 9.9],
            "food_inflation_wb": [0.25, 0.75, 9.9],
        })

    def test_exact_distance_tie_chooses_the_lexically_smallest_geo_id(self) -> None:
        result, report = pdat.join_wb_to_areas(
            self._markets(), np.array([0.0]), np.array([0.0]), verbose=False
        )
        m_index = pdat.month_index(2015, 6)
        # gid_A wins the co-located tie; its missing price is not replaced by gid_B's value.
        self.assertTrue(math.isnan(float(result["values"]["food_price_index_WB"][0, m_index])))
        self.assertEqual(float(result["values"]["food_inflation_wb"][0, m_index]), 0.75)
        chosen = int(result["provenance"]["match_market_index"][0, m_index])
        self.assertEqual(result["provenance"]["market_ids"][chosen], "gid_A")
        self.assertGreaterEqual(report["exact_distance_tie_queries"], 1)

    def test_tie_result_is_independent_of_input_row_order(self) -> None:
        markets = self._markets().iloc[::-1].reset_index(drop=True)
        result, _ = pdat.join_wb_to_areas(
            markets, np.array([0.0]), np.array([0.0]), verbose=False
        )
        m_index = pdat.month_index(2015, 6)
        chosen = int(result["provenance"]["match_market_index"][0, m_index])
        self.assertEqual(result["provenance"]["market_ids"][chosen], "gid_A")

    def test_markets_beyond_100km_stay_unmatched(self) -> None:
        markets = self._markets().iloc[[2]].reset_index(drop=True)
        result, report = pdat.join_wb_to_areas(
            markets, np.array([0.0]), np.array([0.0]), verbose=False
        )
        m_index = pdat.month_index(2015, 6)
        self.assertTrue(math.isnan(float(result["values"]["food_inflation_wb"][0, m_index])))
        self.assertTrue(math.isnan(float(result["provenance"]["match_distance_km"][0, m_index])))
        self.assertGreaterEqual(report["unmatched_area_months_over_100km"], 1)

    def test_one_hundred_kilometres_is_inclusive(self) -> None:
        offset_degrees = 100.0 / (pdat.WB_EARTH_RADIUS_KM * math.pi / 180.0)
        markets = pd.DataFrame({
            "geo_id": ["gid_edge"], "year": [2015], "month": [6],
            "lat": [offset_degrees * 0.999999], "lon": [0.0],
            "food_price_index_WB": [2.0], "food_inflation_wb": [0.5],
        })
        result, report = pdat.join_wb_to_areas(
            markets, np.array([0.0]), np.array([0.0]), verbose=False
        )
        m_index = pdat.month_index(2015, 6)
        distance = float(result["provenance"]["match_distance_km"][0, m_index])
        self.assertLessEqual(distance, pdat.WB_MAX_MATCH_DISTANCE_KM)
        self.assertAlmostEqual(distance, 100.0, places=3)
        self.assertEqual(float(result["values"]["food_price_index_WB"][0, m_index]), 2.0)

    def test_only_the_same_source_month_is_used(self) -> None:
        result, _ = pdat.join_wb_to_areas(
            self._markets(), np.array([0.0]), np.array([0.0]), verbose=False
        )
        other = pdat.month_index(2015, 7)
        self.assertTrue(math.isnan(float(result["values"]["food_inflation_wb"][0, other])))

    def test_invalid_area_coordinates_are_never_matched(self) -> None:
        result, report = pdat.join_wb_to_areas(
            self._markets(), np.array([np.nan]), np.array([0.0]), verbose=False
        )
        self.assertEqual(report["areas_without_valid_coordinates"], 1)
        self.assertTrue(np.all(np.isnan(result["values"]["food_inflation_wb"])))


class TestWorldBankLineage(unittest.TestCase):
    def _write_pair(self, directory: Path, mutate_raw=None, mutate_derived=None) -> Tuple[Path, Path]:
        derived = pd.DataFrame({
            "inflation_food_price_index": [0.5, np.nan],
            "year": [2015, 2015],
            "month": [6, 7],
            "lat": [0.0, 1.0],
            "lon": [0.0, 1.0],
            "food_price_index_WB": [1.5, 2.5],
            "food_inflation_wb": [0.5, np.nan],
        })
        raw = pd.DataFrame({
            "geo_id": ["gid_A", "gid_B"],
            "year": [2015, 2015],
            "month": [6, 7],
            "lat": [0.0, 1.0],
            "lon": [0.0, 1.0],
            "o_food_price_index": [1.0, 2.0],
            "c_food_price_index": [2.0, 3.0],
            "inflation_food_price_index": [0.5, np.nan],
        })
        if mutate_raw is not None:
            raw = mutate_raw(raw)
        if mutate_derived is not None:
            derived = mutate_derived(derived)
        derived_path = directory / "derived.csv"
        raw_path = directory / "raw.csv"
        derived.to_csv(derived_path, index=False)
        raw.to_csv(raw_path, index=False)
        return derived_path, raw_path

    def test_verified_lineage_restores_market_ids(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            derived_path, raw_path = self._write_pair(Path(name))
            markets, report = pdat.load_wb_markets(derived_path, raw_path, verbose=False)
            self.assertEqual(list(markets["geo_id"]), ["gid_A", "gid_B"])
            self.assertTrue(report["lineage_verified"])
            self.assertEqual(report["unique_geo_ids"], 2)

    def test_price_recomputation_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            derived_path, raw_path = self._write_pair(
                Path(name),
                mutate_derived=lambda frame: frame.assign(food_price_index_WB=[1.4, 2.5]),
            )
            with self.assertRaisesRegex(pdat.PreflightError, "does not reconcile to mean"):
                pdat.load_wb_markets(derived_path, raw_path, verbose=False)

    def test_row_count_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            derived_path, raw_path = self._write_pair(
                Path(name), mutate_raw=lambda frame: frame.iloc[[0]],
            )
            with self.assertRaisesRegex(pdat.PreflightError, "row counts differ"):
                pdat.load_wb_markets(derived_path, raw_path, verbose=False)

    def test_duplicate_market_month_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            derived_path, raw_path = self._write_pair(
                Path(name),
                mutate_raw=lambda frame: frame.assign(
                    geo_id=["gid_A", "gid_A"], month=[6, 6],
                ),
                mutate_derived=lambda frame: frame.assign(month=[6, 6]),
            )
            with self.assertRaisesRegex(pdat.PreflightError, "duplicate WB market/month keys"):
                pdat.load_wb_markets(derived_path, raw_path, verbose=False)


class TestCoastline(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self._tmp.name)
        import rasterio
        from rasterio.transform import from_origin

        self.values = np.array([
            [-5, 0, 3, 7],
            [-11, -1, 2, 40],
            [8, 9, -2, 0],
            [1, 2, 3, 4],
        ], dtype=np.int16)
        self.path = self.directory / "coast.tif"
        with rasterio.open(
            self.path, "w", driver="GTiff", height=4, width=4, count=1,
            dtype="int16", crs="EPSG:4326", transform=from_origin(-1.0, 1.0, 0.5, 0.5),
        ) as dataset:
            dataset.write(self.values, 1)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_native_signed_zero_and_positive_values_are_preserved(self) -> None:
        # Pixel centres: lon -0.75/-0.25/0.25/0.75, lat 0.75/0.25/-0.25/-0.75.
        lat = np.array([0.75, 0.75, 0.75, 0.25, -0.25])
        lon = np.array([-0.75, -0.25, 0.75, -0.75, 0.25])
        values, result = pdat.extract_coastline(self.path, lat, lon)
        np.testing.assert_array_equal(values, np.array([-5.0, 0.0, 7.0, -11.0, -2.0]))
        report = result["report"]
        self.assertEqual(report["value_summary"]["negative"], 3)
        self.assertEqual(report["value_summary"]["zero"], 1)
        self.assertEqual(report["value_summary"]["positive"], 1)
        self.assertEqual(report["transform_applied"],
                         "none (no absolute value, rescaling or interpolation)")
        rows = result["provenance"]["pixel_row"]
        cols = result["provenance"]["pixel_col"]
        self.assertEqual((int(rows[0]), int(cols[0])), (0, 0))
        self.assertEqual((int(rows[4]), int(cols[4])), (2, 2))

    def test_out_of_bounds_and_invalid_coordinates_yield_reasons_not_rows_dropped(self) -> None:
        lat = np.array([0.75, 50.0, np.nan, 200.0])
        lon = np.array([-0.75, 0.0, 0.0, 0.0])
        values, result = pdat.extract_coastline(self.path, lat, lon)
        self.assertEqual(values.shape[0], 4)
        self.assertEqual(float(values[0]), -5.0)
        self.assertTrue(np.all(np.isnan(values[1:])))
        reasons = list(result["provenance"]["reason"])
        self.assertEqual(reasons[1], "pixel_outside_raster_bounds")
        self.assertEqual(reasons[2], "invalid_or_missing_coordinate")
        self.assertEqual(reasons[3], "invalid_or_missing_coordinate")


# --------------------------------------------------------------------------------------
# Imputation, schemas and schedule
# --------------------------------------------------------------------------------------


class TestMaxPlusImputer(unittest.TestCase):
    def test_statistics_come_only_from_real_fitting_rows(self) -> None:
        matrix = np.array([
            [1.0, np.nan],
            [2.0, 5.0],
            [900.0, 6.0],     # validation row: must not raise the fitted maximum
            [np.nan, 7.0],    # prediction row
        ])
        fitting = np.array([True, True, False, False])
        imputer = pdat.MaxPlusImputer().fit(matrix, fitting, columns=["a", "b"])
        self.assertEqual(imputer.fill_values_[0], 200.0)
        self.assertEqual(imputer.fill_values_[1], 500.0)
        transformed = imputer.transform(matrix)
        self.assertEqual(transformed[0, 1], 500.0)
        self.assertEqual(transformed[3, 0], 200.0)
        self.assertEqual(transformed[2, 0], 900.0)
        self.assertEqual(imputer.n_fitting_rows_, 2)

    def test_zero_maximum_uses_one_hundred(self) -> None:
        matrix = np.array([[0.0], [np.nan], [-3.0]])
        imputer = pdat.MaxPlusImputer().fit(matrix, np.array([True, True, True]))
        self.assertEqual(imputer.fill_values_[0], 100.0)

    def test_negative_maximum_keeps_the_multiplied_formula(self) -> None:
        matrix = np.array([[-2.0], [-5.0], [np.nan]])
        imputer = pdat.MaxPlusImputer().fit(matrix, np.array([True, True, True]))
        self.assertEqual(imputer.fill_values_[0], -200.0)

    def test_all_missing_fitting_column_uses_the_released_zero_fallback(self) -> None:
        matrix = np.array([[np.nan, 1.0], [np.nan, 2.0], [4.0, 3.0]])
        fitting = np.array([True, True, False])
        imputer = pdat.MaxPlusImputer().fit(matrix, fitting, columns=["ttf", "other"])
        self.assertEqual(imputer.fill_values_[0], 0.0)
        manifest = imputer.manifest()
        self.assertEqual(manifest["columns_all_missing_in_fitting_rows"], ["ttf"])
        self.assertTrue(manifest["ordered_column_statistics"][0]["all_missing_in_fitting_rows"])
        # The sentinel must not be described as an observed zero.
        self.assertIn("sentinel", manifest["ordered_column_statistics"][0]["fill_rule"])

    def test_empty_fitting_set_is_an_error_not_a_silent_fill(self) -> None:
        matrix = np.array([[1.0], [2.0]])
        with self.assertRaises(ValueError):
            pdat.MaxPlusImputer().fit(matrix, np.array([False, False]))


class TestFrozenSchemas(unittest.TestCase):
    def test_realized_widths_match_the_approved_manifest_arithmetic(self) -> None:
        manifest = pdat.build_schema_manifest()
        counts = manifest["counts"]
        self.assertEqual(counts["common_source_fields"], 64)
        self.assertEqual(counts["updated_base_fields"], 86)
        self.assertEqual(counts["reference_source_fields"], 67)
        self.assertEqual(counts["reference_columns"], 109)
        self.assertEqual(
            counts["reference_columns"],
            counts["reference_source_fields"] + 21 + 6 + 12 + 2 + 1,
        )
        for name, expected in pdat.DECLARED_RECIPE_WIDTHS.items():
            realized = len(pdat.recipe_columns(name))
            self.assertEqual(realized, expected, f"recipe {name}")
        self.assertEqual(
            counts["block_A"] + counts["block_B"] + counts["block_C"]
            + counts["block_D"] + counts["block_E"] + counts["updated_base_fields"],
            counts["updated_superset"],
        )
        self.assertEqual(counts["block_E"], 86 + 54)
        self.assertEqual(manifest["updated_base_time_role_counts"],
                         {"static": 32, "monthly": 49, "annual": 5})

    def test_every_master_header_field_is_accounted_for_exactly_once(self) -> None:
        groups = (
            set(pdat.COMMON_SOURCE_FIELDS),
            set(pdat.REFERENCE_ONLY_PRICES),
            set(pdat.DIRECT_INPUT_EXCLUSIONS),
        )
        union: set = set()
        for group in groups:
            self.assertFalse(union & group, "master header groups overlap")
            union |= group
        self.assertEqual(union, set(pdat.MASTER_HEADER))
        self.assertEqual(len(pdat.MASTER_HEADER), 88)

    def test_excluded_predictors_never_enter_any_schema(self) -> None:
        all_columns = set(pdat.updated_superset_columns()) | set(pdat.REFERENCE_COLUMNS)
        for field in ("fews_ha", "Tair_zscore", "Rainf_zscore", "FEWSNET_admin_code",
                      "ISO", "ISO3", "date", "fews_ipc_adjusted", "fews_proj_near",
                      "inflation_food_price_index", "bbg_soybean_oil_futures_bid"):
            self.assertNotIn(field, all_columns)
            self.assertFalse(
                any(column.startswith(f"{field}_") for column in all_columns
                    if not column.startswith(("fews_ipc_lag_", "fews_ipc_crisis_lag_"))),
                f"a derivative of {field} leaked into a schema",
            )
        for field in pdat.REFERENCE_ONLY_PRICES:
            self.assertNotIn(field, set(pdat.updated_superset_columns()))
        self.assertIn("market_distance", set(pdat.updated_superset_columns()))
        self.assertIn("Food_CPI", set(pdat.updated_superset_columns()))
        self.assertIn("Food_food_inflation", set(pdat.updated_superset_columns()))
        for field in ("lat", "lon") + pdat.AEZ_FIELDS:
            self.assertIn(field, set(pdat.updated_superset_columns()))
            self.assertIn(field, set(pdat.REFERENCE_COLUMNS))

    def test_recipes_are_ordered_subsets_of_the_superset(self) -> None:
        superset = pdat.updated_superset_columns()
        for name, _ in pdat.RECIPE_MANIFEST:
            columns = pdat.recipe_columns(name)
            self.assertEqual(len(set(columns)), len(columns))
            indices = [superset.index(column) for column in columns]
            self.assertEqual(indices, sorted(indices), f"recipe {name} reorders the superset")
            self.assertEqual(columns[:86], superset[:86])

    def test_additional_sources_reconcile_to_the_approved_counts(self) -> None:
        self.assertEqual(len(pdat.ADDITIONAL_SOURCE_FIELDS), 22)
        self.assertEqual(len(pdat.BBG_FIELDS), 18)
        self.assertEqual(pdat.ADDITIONAL_SOURCE_FIELDS[0], "nino34_anom")
        self.assertEqual(
            pdat.ADDITIONAL_SOURCE_FIELDS[1:4],
            ("food_price_index_WB", "food_inflation_wb", "coastline_dist"),
        )
        self.assertEqual(len(pdat.B_CONTINUOUS_FIELDS), 28)
        self.assertEqual(len(pdat.B_COUNT_FIELDS), 6)
        self.assertEqual(
            len(pdat.BLOCK_B_COLUMNS), len(pdat.B_CONTINUOUS_FIELDS) * 6 + len(pdat.B_COUNT_FIELDS) * 3
        )
        self.assertEqual(
            len(pdat.BLOCK_C_COLUMNS),
            (len(pdat.B_CONTINUOUS_FIELDS) + len(pdat.B_COUNT_FIELDS)) * 3,
        )
        # w5/w10 conflict variants are BASE inputs but never B/C transforms.
        self.assertIn("event_count_battles_w5", set(pdat.UPDATED_BASE_FIELDS))
        self.assertFalse(any("_w5" in column for column in pdat.BLOCK_B_COLUMNS))
        self.assertFalse(any("_w10" in column for column in pdat.BLOCK_C_COLUMNS))


class TestSchedule(unittest.TestCase):
    def test_candidate_windows_and_fold_counts(self) -> None:
        schedule = pdat.build_schedule()
        self.assertEqual(schedule["n_arms"], 13)
        stage1 = schedule["stage1"]
        self.assertEqual(stage1["candidate_jobs_calibration_window"], 33)
        self.assertEqual(stage1["candidate_jobs_selection_window"], 27)
        self.assertEqual(stage1["exact_overlap_jobs_2016"], 9)
        self.assertEqual(stage1["development_jobs_upper_bound"], 663)
        self.assertEqual(stage1["total_jobs_upper_bound"], 699)
        self.assertEqual(schedule["stage2"]["total_map_builds"], 28)
        self.assertEqual(schedule["stage3"]["development_folds"], 234)
        self.assertEqual(schedule["stage3"]["final_folds"], 60)
        self.assertEqual(
            schedule["stage3"]["final_target_date_counts"], {"4": 11, "8": 10, "12": 9}
        )

    def test_2014_and_2015_candidates_keep_their_january_april_july_months(self) -> None:
        self.assertEqual(pdat.candidate_months_for_year(2014), (1, 4, 7, 10))
        self.assertEqual(pdat.candidate_months_for_year(2015), (1, 4, 7, 10))
        self.assertEqual(pdat.candidate_months_for_year(2016), (2, 6, 10))
        jobs = pdat.role_candidate_jobs("calibration")
        self.assertIn((2014, 1, 3), jobs)
        self.assertIn((2015, 7, 1), jobs)
        self.assertNotIn((2016, 1, 1), jobs)

    def test_final_windows_start_per_horizon_and_end_in_october_2024(self) -> None:
        self.assertEqual(pdat.final_target_dates(4)[0], (2021, 6))
        self.assertEqual(pdat.final_target_dates(8)[0], (2021, 10))
        self.assertEqual(pdat.final_target_dates(12)[0], (2022, 2))
        for horizon in pdat.HORIZONS:
            self.assertEqual(pdat.final_target_dates(horizon)[-1], (2024, 10))

    def test_all_map_roles_are_general_maps_without_month_routing(self) -> None:
        schedule = pdat.build_schedule()
        for role, detail in schedule["map_roles"].items():
            self.assertFalse(detail["month_specific_maps"], role)


# --------------------------------------------------------------------------------------
# Pinned real-source preflight (opt-in; reads the 716 MB master)
# --------------------------------------------------------------------------------------


class TestRealSourcePreflight(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("FEWS_RUN_REAL_PREFLIGHT") == "1",
        "set FEWS_RUN_REAL_PREFLIGHT=1 to run the pinned real-source preflight",
    )
    def test_pinned_sources_reproduce_the_recorded_contract(self) -> None:
        data_root = pdat.resolve_data_root(os.environ.get("FEWS_DATA_ROOT"))
        master = pdat.load_master_grid(
            data_root / str(pdat.PINNED_SOURCES["master"]["relpath"]), verbose=False
        )
        ledger = pdat.load_ledger(
            data_root / str(pdat.PINNED_SOURCES["ledger"]["relpath"]), master.areas, verbose=False
        )
        report: Dict[str, object] = {}
        report.update(master.report)
        report.update(ledger.report)
        report.update(pdat.reconcile_master_and_ledger(master, ledger))
        comparison = pdat.compare_expectations(report)
        self.assertTrue(
            comparison["all_expectations_match"], comparison["failed_expectations"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
