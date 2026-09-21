#!/usr/bin/env python3
"""FEWS NET clean persistence experiment: pinned-source preflight and feature preparation.

Authority: .trellis/tasks/09-20-fewsnet-clean-persistence-baseline/prd.md (R1-R68/A1-A64),
design.md, research/approved-feature-sources.md (D23-D36, D46-D54),
research/target-label-contract.md (D63), research/monthly-source-validation.md (D64),
research/runtime-and-preflight.md, research/fs3-and-time-support.md (D17-D22, D40).

This module implements ONLY data preparation:

  pinned sources -> D63-validated monthly grid + observed IPC ledger
    -> frozen ordered source schemas (64 / 86 / 67 / 109)
    -> origin-aligned BASE + blocks A/B/C/D/E and the corrected reference
    -> fitting-row-only max_plus imputation statistics
    -> source / schema / coverage / schedule manifests

Stage 1/2/3 orchestration (run_pipeline.py) and reporting (report_results.py) are
out of scope here and are owned by later phases.

Nothing in this module fits a model, selects a feature recipe or scores a forecast.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# 1. Pinned sources
# --------------------------------------------------------------------------------------

DATA_ROOT_CANDIDATES = (
    Path(r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data"),
    Path("/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data"),
)

# Identities verified in research/target-label-contract.md and
# research/monthly-source-validation.md and re-verified in the main session.
PINNED_SOURCES: Dict[str, Dict[str, object]] = {
    "master": {
        "relpath": "FEWSNET_forecast_unadjusted_bm.csv",
        "bytes": 716303754,
        "sha256": "611f9e776380e28da3fc845888d66a117626f91b37868e236ce849d44bc8f651",
        "role": "master panel: target cohort, canonical keys, legacy source covariates",
    },
    "ledger": {
        "relpath": "Outcome/FEWSNET_IPC/FEWSNET.csv",
        "bytes": 51422500,
        "sha256": "8fdd4cca6f6ba26b84efc209c8eb36492e1257d51e24edd2c2ad4962df7b38d0",
        "role": "observed IPC history ledger (reconciliation + eligible history)",
    },
    "enso": {
        "relpath": "NOAA_ENSO/nina34.anom.csv",
        "bytes": None,
        "sha256": "ce67f5c52a2a4695f82ee9acdb4a5dc69eb42f34af6cf1e5801fa7ac20118ca5",
        "role": "selected ENSO source (nino34_anom)",
    },
    "wb_derived": {
        "relpath": "WB_RTP_price/wb_food_price_index.csv",
        "bytes": None,
        "sha256": "bc5f310c008d69cf67539633a4af5ef5da2ba724897269b602aed66771ad523a",
        "role": "selected WB derived monthly market prices (2 retained fields)",
    },
    "wb_raw": {
        "relpath": "WB_RTP_price/WLD_RTFP_mkt_2026-04-20.csv",
        "bytes": None,
        "sha256": "cd81efe9ec6c3c1ab7aa14d1a1dbd6092606c015608cec354beccfd960192730",
        "role": "WB raw market file: geo_id lineage restoration for D64 tie rule",
    },
    "coastline": {
        "relpath": "Coastline_distance_NOAA/GMT_intermediate_coast_distance_01d.tif",
        "bytes": None,
        "sha256": None,  # recorded at run time; no pinned identity in the research files
        "role": "EPSG:4326 int16 raster for D50 containing-pixel coastline_dist",
    },
    "bbg_fertiliser": {
        "relpath": "Bloomberg_food_and_derivative/bbg_fertiliser_monthly_051326.csv",
        "bytes": None,
        "sha256": None,
        "role": "Bloomberg fertiliser/gas monthly export (8 fertiliser + 2 gas fields)",
    },
    "bbg_oil_and_gas": {
        "relpath": "Bloomberg_food_and_derivative/bbg_oil_and_gas_monthly_050826.csv",
        "bytes": None,
        "sha256": None,
        "role": "Bloomberg oil/gas monthly export (2 fields)",
    },
    "bbg_soybean_oil": {
        "relpath": "Bloomberg_food_and_derivative/bbg_soybean_oil_futures_monthly_050826.csv",
        "bytes": None,
        "sha256": None,
        "role": "Bloomberg soybean-oil futures monthly export (last price retained)",
    },
    "bbg_staple_food": {
        "relpath": "Bloomberg_food_and_derivative/bbg_staple_food_x1_monthly_050826.csv",
        "bytes": None,
        "sha256": None,
        "role": "Bloomberg staple crop monthly export (5 fields)",
    },
    "fews_shapefile": {
        "relpath": "Outcome/FEWSNET_IPC/FEWS NET Admin Boundaries/FEWS_Admin_LZ_v3.shp",
        "bytes": None,
        "sha256": None,
        "role": "Stage 1 polygon geometry / adjacency source (identity bound here only)",
    },
}

SHAPEFILE_SIDECAR_SUFFIXES = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".shp.xml")


# --------------------------------------------------------------------------------------
# 2. Frozen schemas (D46-D54)
# --------------------------------------------------------------------------------------

# Exact 88-field master header, in file order (checked against CSV line 1).
MASTER_HEADER: Tuple[str, ...] = (
    "unit_name", "ADMIN0", "ADMIN1", "ADMIN2", "ADMIN3", "FEWSNET_admin_code", "ISO",
    "lat", "lon", "date", "month", "distance_to_nearest_acled",
    "event_count_battles", "event_count_explosions", "event_count_violence",
    "sum_fatalities_battles", "sum_fatalities_explosions", "sum_fatalities_violence",
    "event_count_battles_w5", "event_count_explosions_w5", "event_count_violence_w5",
    "sum_fatalities_battles_w5", "sum_fatalities_explosions_w5", "sum_fatalities_violence_w5",
    "event_count_battles_w10", "event_count_explosions_w10", "event_count_violence_w10",
    "sum_fatalities_battles_w10", "sum_fatalities_explosions_w10", "sum_fatalities_violence_w10",
    "AEZ_10000", "AEZ_12000", "AEZ_15000", "AEZ_17000", "AEZ_19000", "AEZ_25000",
    "AEZ_31000", "AEZ_32000", "AEZ_33000", "AEZ_34000", "AEZ_36000", "AEZ_38000",
    "AEZ_4000", "AEZ_40000", "AEZ_43000", "AEZ_7000", "AEZ_9000",
    "crop", "range", "distance_to_river", "nightlight", "nightlight_sd", "elevation",
    "EVI", "market_distance", "FAO_price", "Rainf_f_tavg_mean", "Tair_f_tavg_mean",
    "gpp_mean", "sg_cec_5-15cm", "sg_cfvo_5-15cm", "sg_nitrogen_5-15cm",
    "sg_phh2o_5-15cm", "sg_soc_5-15cm", "market_access", "ruggedness", "slope",
    "CPI", "GDP", "CC", "gini", "WFP_Price", "WFP_Price_std",
    "fews_ipc", "fews_ha", "fews_proj_near", "fews_proj_near_ha", "fews_proj_med",
    "fews_proj_med_ha", "pop", "fews_ipc_adjusted", "fews_proj_med_adjusted", "ISO3",
    "Food_CPI", "Food_food_inflation", "Tair_zscore", "Rainf_zscore", "fews_ipc_crisis",
)

AEZ_FIELDS: Tuple[str, ...] = tuple(f for f in MASTER_HEADER if f.startswith("AEZ_"))

# D52: 21 direct-input exclusions. Keys/labels are kept as metadata only (D51/D47/D46).
DIRECT_INPUT_EXCLUSIONS: Tuple[str, ...] = (
    "unit_name", "ADMIN0", "ADMIN1", "ADMIN2", "ADMIN3", "FEWSNET_admin_code",
    "ISO", "ISO3", "date", "month",
    "fews_ipc", "fews_ipc_crisis", "fews_ha",
    "fews_proj_near", "fews_proj_near_ha", "fews_proj_med", "fews_proj_med_ha",
    "fews_ipc_adjusted", "fews_proj_med_adjusted",
    "Tair_zscore", "Rainf_zscore",
)

# D52 reference-only legacy prices (kept in the original-feature arm only).
REFERENCE_ONLY_PRICES: Tuple[str, ...] = ("FAO_price", "WFP_Price", "WFP_Price_std")

# D52 common retained source fields (64), in master header order.
COMMON_SOURCE_FIELDS: Tuple[str, ...] = tuple(
    f for f in MASTER_HEADER
    if f not in DIRECT_INPUT_EXCLUSIONS and f not in REFERENCE_ONLY_PRICES
)

# D31: exactly 22 additional BASE fields, in the approved table order.
ENSO_FIELD = "nino34_anom"
WB_FIELDS: Tuple[str, ...] = ("food_price_index_WB", "food_inflation_wb")
COASTLINE_FIELD = "coastline_dist"
BBG_FERTILISER_FIELDS: Tuple[str, ...] = (
    "bbg_GCFPURGB", "bbg_GCFPDANO", "bbg_GCFPPOBA", "bbg_GCFPURBS",
    "bbg_GCFPAMME", "bbg_GCFPAMBS", "bbg_GCFPURMG", "bbg_GCFPDAIN",
)
BBG_ENERGY_FIELDS: Tuple[str, ...] = (
    "bbg_NGUSHHUB", "bbg_TZTX2_Comdty",
    "bbg_oilgas_CL1_COMB_Comdty_price", "bbg_oilgas_NG1_Comdty_price",
)
BBG_STAPLE_FIELDS: Tuple[str, ...] = (
    "bbg_staple_food_corn_x1_price", "bbg_staple_food_hard_wheat_x1_price",
    "bbg_staple_food_rough_rice_x1_price", "bbg_staple_food_soft_wheat_x1_price",
    "bbg_staple_food_soybeans_x1_price",
)
BBG_SOYBEAN_OIL_FIELDS: Tuple[str, ...] = ("bbg_soybean_oil_futures_last_price",)
BBG_FIELDS: Tuple[str, ...] = (
    BBG_FERTILISER_FIELDS + BBG_ENERGY_FIELDS + BBG_STAPLE_FIELDS + BBG_SOYBEAN_OIL_FIELDS
)
ADDITIONAL_SOURCE_FIELDS: Tuple[str, ...] = (
    (ENSO_FIELD,) + WB_FIELDS + (COASTLINE_FIELD,) + BBG_FIELDS
)

# Explicitly excluded duplicate/alternate quotes (D31).
EXCLUDED_ADDITIONAL_FIELDS: Tuple[str, ...] = (
    "inflation_food_price_index", "bbg_soybean_oil_futures_bid",
)

UPDATED_BASE_FIELDS: Tuple[str, ...] = COMMON_SOURCE_FIELDS + ADDITIONAL_SOURCE_FIELDS
REFERENCE_SOURCE_FIELDS: Tuple[str, ...] = tuple(
    f for f in MASTER_HEADER
    if f in COMMON_SOURCE_FIELDS or f in REFERENCE_ONLY_PRICES
)

# Time roles (D34 monthly / D35 annual / D48 population / D49-D50 static).
ANNUAL_FIELDS: Tuple[str, ...] = ("CPI", "GDP", "CC", "gini", "pop")
MASTER_MONTHLY_FIELDS: Tuple[str, ...] = (
    "distance_to_nearest_acled",
    "event_count_battles", "event_count_explosions", "event_count_violence",
    "sum_fatalities_battles", "sum_fatalities_explosions", "sum_fatalities_violence",
    "event_count_battles_w5", "event_count_explosions_w5", "event_count_violence_w5",
    "sum_fatalities_battles_w5", "sum_fatalities_explosions_w5", "sum_fatalities_violence_w5",
    "event_count_battles_w10", "event_count_explosions_w10", "event_count_violence_w10",
    "sum_fatalities_battles_w10", "sum_fatalities_explosions_w10", "sum_fatalities_violence_w10",
    "nightlight", "nightlight_sd", "EVI", "market_distance",
    "Rainf_f_tavg_mean", "Tair_f_tavg_mean", "gpp_mean", "Food_CPI", "Food_food_inflation",
)
# Per-area monthly sources: master monthly + the two WB market fields.
AREA_MONTHLY_FIELDS: Tuple[str, ...] = MASTER_MONTHLY_FIELDS + WB_FIELDS
# Global (area-invariant) monthly sources.
GLOBAL_MONTHLY_FIELDS: Tuple[str, ...] = (ENSO_FIELD,) + BBG_FIELDS

MASTER_STATIC_FIELDS: Tuple[str, ...] = (
    ("lat", "lon") + AEZ_FIELDS
    + ("crop", "range", "distance_to_river", "elevation",
       "sg_cec_5-15cm", "sg_cfvo_5-15cm", "sg_nitrogen_5-15cm", "sg_phh2o_5-15cm",
       "sg_soc_5-15cm", "market_access", "ruggedness", "slope")
)
STATIC_FIELDS: Tuple[str, ...] = MASTER_STATIC_FIELDS + (COASTLINE_FIELD,)


def field_time_role(field: str) -> str:
    """Return 'static', 'monthly' or 'annual' for a declared source field."""
    if field in STATIC_FIELDS:
        return "static"
    if field in ANNUAL_FIELDS:
        return "annual"
    if field in AREA_MONTHLY_FIELDS or field in GLOBAL_MONTHLY_FIELDS or field in REFERENCE_ONLY_PRICES:
        return "monthly"
    raise KeyError(f"No declared time role for source field {field!r}")


# D32: B/C whitelist (28 continuous, 6 conflict counts).
B_CONTINUOUS_FIELDS: Tuple[str, ...] = (
    "Rainf_f_tavg_mean", "Tair_f_tavg_mean", "EVI", "gpp_mean",
    "nightlight",
    "Food_CPI", "Food_food_inflation",
    ENSO_FIELD, "food_price_index_WB", "food_inflation_wb",
) + BBG_FIELDS
B_COUNT_FIELDS: Tuple[str, ...] = (
    "event_count_battles", "event_count_explosions", "event_count_violence",
    "sum_fatalities_battles", "sum_fatalities_explosions", "sum_fatalities_violence",
)
B_WINDOWS: Tuple[int, ...] = (3, 6, 12)

# D25 block A.
BLOCK_A_COLUMNS: Tuple[str, ...] = (
    "target_month_sin", "target_month_cos",
    "last_observed_ipc_phase", "last_observed_crisis",
    "last_observed_ipc_age_months", "months_since_last_observed_crisis",
    "no_observed_ipc_history", "no_prior_observed_crisis",
)

# D28 block D.
BLOCK_D_COLUMNS: Tuple[str, ...] = (
    "rain_evi_interaction", "rain_conflict_interaction",
    "inflation_conflict_interaction", "inflation_market_distance_interaction",
)


def block_b_columns() -> Tuple[str, ...]:
    """186 columns: mean/population-SD per continuous series and sums per count series."""
    cols: List[str] = []
    for field in B_CONTINUOUS_FIELDS:
        for window in B_WINDOWS:
            cols.append(f"{field}_mean_{window}m")
            cols.append(f"{field}_sd_{window}m")
    for field in B_COUNT_FIELDS:
        for window in B_WINDOWS:
            cols.append(f"{field}_sum_{window}m")
    return tuple(cols)


def block_c_columns() -> Tuple[str, ...]:
    """102 columns: 3/12-month changes and the trailing standardized deviation."""
    cols: List[str] = []
    for field in B_CONTINUOUS_FIELDS + B_COUNT_FIELDS:
        cols.append(f"{field}_chg_3m")
        cols.append(f"{field}_chg_12m")
        cols.append(f"{field}_zdev_12m")
    return tuple(cols)


def block_e_columns() -> Tuple[str, ...]:
    """140 columns: one missing flag per updated BASE field plus 54 dynamic ages."""
    cols: List[str] = []
    for field in UPDATED_BASE_FIELDS:
        cols.append(f"{field}_missing")
        if field_time_role(field) != "static":
            cols.append(f"{field}_age_months")
    return tuple(cols)


BLOCK_B_COLUMNS: Tuple[str, ...] = block_b_columns()
BLOCK_C_COLUMNS: Tuple[str, ...] = block_c_columns()
BLOCK_E_COLUMNS: Tuple[str, ...] = block_e_columns()

BLOCK_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "A": BLOCK_A_COLUMNS,
    "B": BLOCK_B_COLUMNS,
    "C": BLOCK_C_COLUMNS,
    "D": BLOCK_D_COLUMNS,
    "E": BLOCK_E_COLUMNS,
}

# D24/D53: the frozen 12-recipe manifest order.
RECIPE_MANIFEST: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("BASE", ()),
    ("A", ("A",)),
    ("B", ("B",)),
    ("C", ("C",)),
    ("D", ("D",)),
    ("E", ("E",)),
    ("ABCDE", ("A", "B", "C", "D", "E")),
    ("BCDE", ("B", "C", "D", "E")),
    ("ACDE", ("A", "C", "D", "E")),
    ("ABDE", ("A", "B", "D", "E")),
    ("ABCE", ("A", "B", "C", "E")),
    ("ABCD", ("A", "B", "C", "D")),
)
DECLARED_RECIPE_WIDTHS: Dict[str, int] = {
    "BASE": 86, "A": 94, "B": 272, "C": 188, "D": 90, "E": 226,
    "ABCDE": 526, "BCDE": 518, "ACDE": 340, "ABDE": 424, "ABCE": 522, "ABCD": 386,
}
REFERENCE_ARM = "reference"

# D54 inherited reference transforms.
REFERENCE_YEARS: Tuple[int, ...] = tuple(range(2010, 2025))
REFERENCE_MONTHS: Tuple[int, ...] = (1, 2, 4, 6, 7, 10)
REFERENCE_IPC_LAGS: Tuple[int, ...] = (4, 8, 12)


def recipe_columns(recipe: str) -> Tuple[str, ...]:
    """Ordered model-input columns for an updated-feature recipe: BASE then enabled blocks."""
    blocks = dict(RECIPE_MANIFEST).get(recipe)
    if blocks is None:
        raise KeyError(f"Unknown recipe {recipe!r}")
    cols: List[str] = list(UPDATED_BASE_FIELDS)
    for block in ("A", "B", "C", "D", "E"):
        if block in blocks:
            cols.extend(BLOCK_COLUMNS[block])
    return tuple(cols)


def updated_superset_columns() -> Tuple[str, ...]:
    """BASE + every block, i.e. the ABCDE recipe; all recipes are ordered subsets."""
    return recipe_columns("ABCDE")


def reference_columns() -> Tuple[str, ...]:
    """The 109 ordered corrected-reference inputs (D54)."""
    cols: List[str] = list(REFERENCE_SOURCE_FIELDS)
    cols.extend(f"year_{year}" for year in REFERENCE_YEARS)
    cols.extend(f"month_{month}" for month in REFERENCE_MONTHS)
    cols.extend(f"fews_ipc_crisis_lag_{lag}" for lag in REFERENCE_IPC_LAGS)
    cols.extend(f"fews_ipc_lag_{lag}" for lag in REFERENCE_IPC_LAGS)
    cols.extend(f"EVI_l{lag}" for lag in range(1, 13))
    cols.extend(("WFP_Price_m4", "WFP_Price_m12", "nightlight_m12"))
    return tuple(cols)


REFERENCE_COLUMNS: Tuple[str, ...] = reference_columns()


# --------------------------------------------------------------------------------------
# 3. Calendar grid and schedule constants
# --------------------------------------------------------------------------------------

GRID_START_YEAR = 2008      # 24 months before the master's first month, covering O-12 lookback
GRID_END_YEAR = 2024
N_GRID_MONTHS = (GRID_END_YEAR - GRID_START_YEAR + 1) * 12       # 204
N_GRID_YEARS = GRID_END_YEAR - GRID_START_YEAR + 1               # 17
MASTER_FIRST_MONTH = (2010, 1)
MASTER_LAST_MONTH = (2024, 12)

HORIZONS: Tuple[int, ...] = (4, 8, 12)
FORECASTING_SCOPES: Dict[int, int] = {1: 4, 2: 8, 3: 12}

TRAIN_WINDOW_CALENDAR_MONTHS = 35   # effective span of the released 36-month config (D20)

MAP_ROLES: Dict[str, Dict[str, object]] = {
    "calibration": {"candidate_years": (2014, 2015, 2016), "cutoff": (2016, 12),
                    "prediction_targets": ((2018, 2), (2018, 6), (2018, 10))},
    "selection": {"candidate_years": (2016, 2017, 2018), "cutoff": (2018, 12),
                  "prediction_targets": ((2020, 2), (2020, 6), (2020, 10))},
    "final": {"candidate_years": (2018, 2019, 2020), "cutoff": (2020, 12),
              "prediction_targets": None},
}
CANDIDATE_MONTHS_EARLY: Tuple[int, ...] = (1, 4, 7, 10)   # 2014-2015 observed label months
CANDIDATE_MONTHS_LATE: Tuple[int, ...] = (2, 6, 10)       # 2016 onward observed label months
FINAL_TARGET_START: Dict[int, Tuple[int, int]] = {4: (2021, 6), 8: (2021, 10), 12: (2022, 2)}
FINAL_TARGET_END: Tuple[int, int] = (2024, 10)
SUPPLEMENTARY_COMMON_START: Tuple[int, int] = (2022, 2)

# D64 / D9 geometry constants.
WB_EARTH_RADIUS_KM = 6371.0088
WB_MAX_MATCH_DISTANCE_KM = 100.0

# D23 imputation.
IMPUTER_STRATEGY = "max_plus"
IMPUTER_MULTIPLIER = 100.0
IMPUTER_ALL_MISSING_FILL = 0.0

# D63 pinned preflight expectations (research/target-label-contract.md).
EXPECTED_PREFLIGHT: Dict[str, object] = {
    "master_records": 1029240,
    "master_fields": 88,
    "master_unique_keys": 1029240,
    "master_duplicate_keys": 0,
    "master_areas": 5718,
    "master_months": 180,
    "master_first_month": "2010-01",
    "master_last_month": "2024-12",
    "master_phase_counts": {1: 133211, 2: 82907, 3: 39062, 4: 4186, 5: 74},
    "master_phase_missing": 769800,
    "master_crisis_zero": 216118,
    "master_crisis_one": 43322,
    "master_crisis_missing": 769800,
    "master_valid_labels": 259440,
    "ledger_records": 302949,
    "ledger_valid_key_records": 302948,
    "ledger_unique_keys": 302948,
    "ledger_duplicate_keys": 0,
    "ledger_malformed_records": 1,
    "ledger_areas": 5716,
    "ledger_months": 53,
    "ledger_first_month": "2009-07",
    "ledger_last_month": "2024-10",
    "ledger_phase_counts": {1: 137113, 2: 85268, 3: 41169, 4: 4601, 5: 74},
    "ledger_phase_missing": 34723,
    "ledger_artifact_physical_line": 303003,
    "ledger_artifact_raw": "System.IO.MemoryStream",
    "cross_same_key_agree": 259440,
    "cross_same_key_both_missing": 32076,
    "cross_same_key_one_missing": 0,
    "cross_same_key_conflict": 0,
    "cross_ledger_absent_master_missing": 737724,
    "cross_ledger_absent_master_observed": 0,
    "cross_overlap_keys": 291516,
    "cross_ledger_only_keys": 11432,
}

# research/runtime-and-preflight.md secondary expectations (annual support).
EXPECTED_ANNUAL_WHOLLY_MISSING: Dict[str, int] = {
    "GDP": 8854, "CPI": 15178, "CC": 5718, "gini": 44280,
}
EXPECTED_AREA_YEARS = 85770
EXPECTED_POP_WITHIN_YEAR_DIFF_GROUPS = 36


class PreflightError(RuntimeError):
    """Raised when pinned-source validation fails; never downgraded to missingness."""


# --------------------------------------------------------------------------------------
# 4. Small utilities
# --------------------------------------------------------------------------------------


def sha256_file(path: Path, chunk_bytes: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def month_index(year: int, month: int) -> int:
    """Index of (year, month) on the prepared grid; 2008-01 is 0."""
    return (int(year) - GRID_START_YEAR) * 12 + (int(month) - 1)


def index_to_month(index: int) -> Tuple[int, int]:
    return GRID_START_YEAR + index // 12, index % 12 + 1


def month_label(index: int) -> str:
    year, month = index_to_month(index)
    return f"{year:04d}-{month:02d}"


def year_index(year: int) -> int:
    return int(year) - GRID_START_YEAR


def resolve_data_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        root = Path(explicit)
        if not root.is_dir():
            raise PreflightError(f"--data-root does not exist: {root}")
        return root
    for candidate in DATA_ROOT_CANDIDATES:
        if candidate.is_dir():
            return candidate
    raise PreflightError(
        "Could not locate the pinned source root; pass --data-root explicitly."
    )


def runtime_identity() -> Dict[str, object]:
    import scipy
    import sklearn
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Unserializable value of type {type(value)!r}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, default=_json_default)
        handle.write("\n")


# --------------------------------------------------------------------------------------
# 5. Master panel: D63 validation and grid assembly
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class MasterGrid:
    """Validated master panel expressed on the complete (area, month) grid."""

    areas: np.ndarray                             # int64, ascending canonical codes
    monthly: Dict[str, np.ndarray]                 # field -> (n_areas, N_GRID_MONTHS)
    static: Dict[str, np.ndarray]                  # field -> (n_areas,)
    annual: Dict[str, np.ndarray]                  # field -> (n_areas, N_GRID_YEARS)
    pop_selected_month: np.ndarray                 # (n_areas, N_GRID_YEARS) int16 month or -1
    phase: np.ndarray                              # (n_areas, N_GRID_MONTHS) observed phase
    crisis: np.ndarray                             # (n_areas, N_GRID_MONTHS) observed binary
    present: np.ndarray                            # (n_areas, N_GRID_MONTHS) master row present
    target_area_idx: np.ndarray                    # labeled target rows (int32)
    target_month_idx: np.ndarray                   # labeled target rows (int32)
    target_label: np.ndarray                       # labeled target rows (int8)
    report: Dict[str, object]

    @property
    def n_areas(self) -> int:
        return int(self.areas.shape[0])


def _read_master_header(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def _validate_canonical_codes(values: pd.Series, source: str) -> np.ndarray:
    stripped = values.astype(str)
    canonical = stripped.str.fullmatch(r"\d+")
    if not bool(canonical.all()):
        bad = stripped[~canonical].unique()[:10].tolist()
        raise PreflightError(f"{source}: non-canonical area codes, examples {bad}")
    codes = stripped.astype(np.int64)
    if not bool((stripped == codes.astype(str)).all()):
        bad = stripped[stripped != codes.astype(str)].unique()[:10].tolist()
        raise PreflightError(f"{source}: area codes normalize ambiguously, examples {bad}")
    return codes.to_numpy()


def load_master_grid(path: Path, verbose: bool = True) -> MasterGrid:
    """Read the pinned master once and validate it under D63/D49/D35/D48."""
    report: Dict[str, object] = {}
    header = _read_master_header(path)
    if tuple(header) != MASTER_HEADER:
        raise PreflightError(
            "Master header does not match the frozen 88-field schema; "
            f"got {len(header)} fields."
        )
    report["master_fields"] = len(header)

    usecols = ["FEWSNET_admin_code", "date", "fews_ipc", "fews_ipc_crisis"]
    usecols += [f for f in REFERENCE_SOURCE_FIELDS if f not in usecols]
    dtype: Dict[str, object] = {"FEWSNET_admin_code": "string", "date": "string"}
    for field in AEZ_FIELDS:
        dtype[field] = "category"
    for field in usecols:
        if field not in dtype:
            dtype[field] = "float64"

    started = time.time()
    frame = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype,
        na_values=[""],
        keep_default_na=True,
        on_bad_lines="error",
        engine="c",
        encoding="utf-8-sig",
    )
    read_seconds = time.time() - started
    report["master_read_seconds"] = round(read_seconds, 2)
    report["master_records"] = int(len(frame))
    if verbose:
        print(f"[master] read {len(frame):,} records in {read_seconds:.1f}s", flush=True)

    if frame["FEWSNET_admin_code"].isna().any() or frame["date"].isna().any():
        raise PreflightError("Master has rows with an empty canonical key field.")
    codes = _validate_canonical_codes(frame["FEWSNET_admin_code"], "master")
    dates = frame["date"].astype(str)
    if not bool(dates.str.fullmatch(r"\d{4}-\d{2}").all()):
        bad = dates[~dates.str.fullmatch(r"\d{4}-\d{2}")].unique()[:10].tolist()
        raise PreflightError(f"Master date values are not YYYY-MM, examples {bad}")
    years = dates.str.slice(0, 4).astype(np.int32).to_numpy()
    months = dates.str.slice(5, 7).astype(np.int32).to_numpy()
    if months.min() < 1 or months.max() > 12:
        raise PreflightError("Master date month is outside 1-12.")
    if years.min() != MASTER_FIRST_MONTH[0] or years.max() != MASTER_LAST_MONTH[0]:
        raise PreflightError(
            f"Master year extent {years.min()}-{years.max()} is not the pinned 2010-2024."
        )

    areas = np.unique(codes)
    area_idx = np.searchsorted(areas, codes).astype(np.int32)
    t_idx = ((years - GRID_START_YEAR) * 12 + (months - 1)).astype(np.int32)
    if t_idx.min() < 0 or t_idx.max() >= N_GRID_MONTHS:
        raise PreflightError("Master month falls outside the prepared calendar grid.")

    n_areas = int(areas.shape[0])
    present = np.zeros((n_areas, N_GRID_MONTHS), dtype=bool)
    flat = area_idx.astype(np.int64) * N_GRID_MONTHS + t_idx
    present.reshape(-1)[flat] = True
    unique_keys = int(present.sum())
    duplicate_rows = int(len(frame) - unique_keys)
    if duplicate_rows != 0:
        raise PreflightError(
            f"Master has {duplicate_rows} duplicate canonical area/month rows; "
            "silent deduplication is prohibited."
        )
    report["master_unique_keys"] = unique_keys
    report["master_duplicate_keys"] = 0
    report["master_areas"] = n_areas
    observed_months = np.unique(t_idx)
    report["master_months"] = int(observed_months.shape[0])
    report["master_first_month"] = month_label(int(observed_months.min()))
    report["master_last_month"] = month_label(int(observed_months.max()))

    master_month_slice = slice(month_index(*MASTER_FIRST_MONTH), month_index(*MASTER_LAST_MONTH) + 1)
    complete_grid = bool(present[:, master_month_slice].all())
    outside = bool(present[:, : master_month_slice.start].any() or present[:, master_month_slice.stop:].any())
    if not complete_grid or outside:
        raise PreflightError(
            "Master is not a complete monthly area grid over 2010-01..2024-12."
        )
    report["master_complete_monthly_grid"] = True

    # ---- labels (D63) -------------------------------------------------------------
    phase_values = frame["fews_ipc"].to_numpy(dtype=np.float64)
    crisis_values = frame["fews_ipc_crisis"].to_numpy(dtype=np.float64)
    phase_valid = ~np.isnan(phase_values)
    crisis_valid = ~np.isnan(crisis_values)
    if int(np.sum(phase_valid != crisis_valid)) != 0:
        raise PreflightError(
            "Master phase and binary-label missingness disagree; "
            f"{int(np.sum(phase_valid != crisis_valid))} rows affected."
        )
    observed_phase = phase_values[phase_valid]
    if not np.all(np.isfinite(observed_phase)):
        raise PreflightError("Master has nonfinite observed phase values.")
    if not np.all(observed_phase == np.round(observed_phase)):
        raise PreflightError("Master has non-integral observed phase values.")
    if observed_phase.size and (observed_phase.min() < 1 or observed_phase.max() > 5):
        raise PreflightError("Master observed phase outside the valid 1..5 range.")
    observed_crisis = crisis_values[crisis_valid]
    if not np.all(np.isin(observed_crisis, (0.0, 1.0))):
        raise PreflightError("Master observed binary label outside {0, 1}.")
    derived = (observed_phase >= 3).astype(np.float64)
    mismatch = int(np.sum(derived != observed_crisis))
    if mismatch:
        raise PreflightError(
            f"{mismatch} master rows violate fews_ipc_crisis == 1[fews_ipc >= 3]."
        )
    report["master_phase_counts"] = {
        int(value): int(np.sum(observed_phase == value)) for value in (1, 2, 3, 4, 5)
    }
    report["master_phase_missing"] = int(np.sum(~phase_valid))
    report["master_crisis_zero"] = int(np.sum(observed_crisis == 0))
    report["master_crisis_one"] = int(np.sum(observed_crisis == 1))
    report["master_crisis_missing"] = int(np.sum(~crisis_valid))
    report["master_valid_labels"] = int(observed_phase.size)

    def scatter(values: np.ndarray) -> np.ndarray:
        grid = np.full(n_areas * N_GRID_MONTHS, np.nan, dtype=np.float64)
        grid[flat] = values
        return grid.reshape(n_areas, N_GRID_MONTHS)

    phase_grid = scatter(phase_values)
    crisis_grid = scatter(crisis_values)

    # ---- static fields (D49) ------------------------------------------------------
    static: Dict[str, np.ndarray] = {}
    static_conflicts: Dict[str, Dict[str, int]] = {}
    aez_true_counts = np.zeros(n_areas, dtype=np.int32)
    for field in MASTER_STATIC_FIELDS:
        if field in AEZ_FIELDS:
            column = frame[field].astype("string")
            allowed = column.isna() | column.isin(["true", "false"])
            if not bool(allowed.all()):
                bad = column[~allowed].unique()[:10].tolist()
                raise PreflightError(
                    f"{field}: unexpected boolean token(s) {bad}; refusing to coerce "
                    "unknown tokens into scientific missingness."
                )
            if bool(column.isna().any()):
                raise PreflightError(f"{field}: empty AEZ indicator values are not supported.")
            values = (column == "true").to_numpy().astype(np.float64)
        else:
            values = frame[field].to_numpy(dtype=np.float64)
        grid = scatter(values)[:, master_month_slice]
        valid = ~np.isnan(grid)
        n_valid = valid.sum(axis=1)
        mixed = int(np.sum((n_valid > 0) & (n_valid < grid.shape[1])))
        with np.errstate(invalid="ignore"):
            spread_conflict = int(np.sum(
                (n_valid > 0) & (np.nanmax(np.where(valid, grid, np.nan), axis=1)
                                 != np.nanmin(np.where(valid, grid, np.nan), axis=1))
            ))
        if mixed or spread_conflict:
            static_conflicts[field] = {
                "areas_with_mixed_missingness": mixed,
                "areas_with_conflicting_values": spread_conflict,
            }
            raise PreflightError(
                f"{field}: static source field is not constant within area "
                f"(mixed-missing areas={mixed}, conflicting areas={spread_conflict}); "
                "averaging or picking a row is prohibited."
            )
        first = np.where(n_valid > 0, np.nanmax(np.where(valid, grid, -np.inf), axis=1), np.nan)
        static[field] = np.where(n_valid > 0, first, np.nan)
        if field in AEZ_FIELDS:
            aez_true_counts += (static[field] == 1.0).astype(np.int32)
    report["static_conflicts"] = static_conflicts
    report["aez_areas_with_exactly_one_true"] = int(np.sum(aez_true_counts == 1))
    report["aez_areas_without_exactly_one_true"] = int(np.sum(aez_true_counts != 1))

    # ---- annual fields (D35) and population (D48) ---------------------------------
    annual: Dict[str, np.ndarray] = {}
    annual_report: Dict[str, Dict[str, int]] = {}
    year_slice_start = year_index(MASTER_FIRST_MONTH[0])
    n_master_years = MASTER_LAST_MONTH[0] - MASTER_FIRST_MONTH[0] + 1
    for field in ("CPI", "GDP", "CC", "gini"):
        grid = scatter(frame[field].to_numpy(dtype=np.float64))[:, master_month_slice]
        cube = grid.reshape(n_areas, n_master_years, 12)
        valid = ~np.isnan(cube)
        any_valid = valid.any(axis=2)
        with np.errstate(invalid="ignore"):
            high = np.where(valid, cube, -np.inf).max(axis=2)
            low = np.where(valid, cube, np.inf).min(axis=2)
        conflicting = int(np.sum(any_valid & (high != low)))
        if conflicting:
            raise PreflightError(
                f"{field}: {conflicting} area/year groups have conflicting annual values; "
                "averaging is prohibited."
            )
        values = np.full((n_areas, N_GRID_YEARS), np.nan, dtype=np.float64)
        values[:, year_slice_start:year_slice_start + n_master_years] = np.where(any_valid, high, np.nan)
        annual[field] = values
        annual_report[field] = {
            "area_years": int(any_valid.size),
            "wholly_missing_area_years": int(np.sum(~any_valid)),
            "conflicting_area_years": conflicting,
        }

    pop_grid = scatter(frame["pop"].to_numpy(dtype=np.float64))[:, master_month_slice]
    pop_cube = pop_grid.reshape(n_areas, n_master_years, 12)
    pop_valid = ~np.isnan(pop_cube)
    pop_any = pop_valid.any(axis=2)
    with np.errstate(invalid="ignore"):
        pop_high = np.where(pop_valid, pop_cube, -np.inf).max(axis=2)
        pop_low = np.where(pop_valid, pop_cube, np.inf).min(axis=2)
    pop_within_year_diff = int(np.sum(pop_any & (pop_high != pop_low)))
    # D48: last valid source month within the year, never an average.
    reversed_valid = pop_valid[:, :, ::-1]
    last_offset = np.argmax(reversed_valid, axis=2)
    last_month_zero_based = 11 - last_offset
    selected = np.take_along_axis(pop_cube, last_month_zero_based[:, :, None], axis=2)[:, :, 0]
    pop_values = np.full((n_areas, N_GRID_YEARS), np.nan, dtype=np.float64)
    pop_values[:, year_slice_start:year_slice_start + n_master_years] = np.where(pop_any, selected, np.nan)
    pop_month = np.full((n_areas, N_GRID_YEARS), -1, dtype=np.int16)
    pop_month[:, year_slice_start:year_slice_start + n_master_years] = np.where(
        pop_any, (last_month_zero_based + 1).astype(np.int16), -1
    )
    annual["pop"] = pop_values
    annual_report["pop"] = {
        "area_years": int(pop_any.size),
        "wholly_missing_area_years": int(np.sum(~pop_any)),
        "area_years_with_within_year_differences": pop_within_year_diff,
    }
    report["annual_fields"] = annual_report
    report["pop_within_year_difference_groups"] = pop_within_year_diff

    # ---- monthly fields ----------------------------------------------------------
    monthly: Dict[str, np.ndarray] = {}
    for field in MASTER_MONTHLY_FIELDS + REFERENCE_ONLY_PRICES:
        monthly[field] = scatter(frame[field].to_numpy(dtype=np.float64))

    target_mask = ~np.isnan(phase_values)
    order = np.lexsort((t_idx[target_mask], area_idx[target_mask]))
    target_area_idx = area_idx[target_mask][order]
    target_month_idx = t_idx[target_mask][order]
    target_label = crisis_values[target_mask][order].astype(np.int8)

    del frame

    return MasterGrid(
        areas=areas,
        monthly=monthly,
        static=static,
        annual=annual,
        pop_selected_month=pop_month,
        phase=phase_grid,
        crisis=crisis_grid,
        present=present,
        target_area_idx=target_area_idx.astype(np.int32),
        target_month_idx=target_month_idx.astype(np.int32),
        target_label=target_label,
        report=report,
    )


# --------------------------------------------------------------------------------------
# 6. Observed IPC ledger: D63 parse, artifact exclusion and reconciliation
# --------------------------------------------------------------------------------------

LEDGER_HEADER: Tuple[str, ...] = (
    "country", "admin_code", "year_month", "year", "month", "fews_ipc", "fews_ha",
    "fews_proj_near", "fews_proj_near_ha", "fews_proj_med", "fews_proj_med_ha",
    "pop", "pop_source", "fews_ipc_adjusted", "fews_proj_med_adjusted", "admin_name",
)
LEDGER_TERMINAL_ARTIFACT = "System.IO.MemoryStream"


@dataclasses.dataclass
class LedgerData:
    phase: np.ndarray        # (n_areas, N_GRID_MONTHS) observed phase, NaN elsewhere
    present: np.ndarray      # (n_areas, N_GRID_MONTHS) ledger key present
    report: Dict[str, object]


def load_ledger(path: Path, areas: np.ndarray, verbose: bool = True) -> LedgerData:
    """Parse the pinned ledger, excluding only the verified terminal artifact (D63)."""
    report: Dict[str, object] = {}
    n_areas = int(areas.shape[0])
    phase = np.full((n_areas, N_GRID_MONTHS), np.nan, dtype=np.float64)
    present = np.zeros((n_areas, N_GRID_MONTHS), dtype=bool)

    started = time.time()
    n_records = 0
    n_valid_key = 0
    phase_counts = {value: 0 for value in (1, 2, 3, 4, 5)}
    phase_missing = 0
    duplicate_keys = 0
    ledger_only_areas: set = set()
    excluded: List[Dict[str, object]] = []
    observed_months: set = set()
    areas_seen: set = set()

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if tuple(header) != LEDGER_HEADER:
            raise PreflightError("Ledger header does not match the frozen 16-field schema.")
        previous_line = reader.line_num
        for row in reader:
            start_line = previous_line + 1
            end_line = reader.line_num
            previous_line = end_line
            n_records += 1
            if len(row) != len(LEDGER_HEADER):
                raw = row[0] if len(row) == 1 else ",".join(row)
                is_terminal_artifact = (
                    len(row) == 1
                    and raw == LEDGER_TERMINAL_ARTIFACT
                    and end_line == EXPECTED_PREFLIGHT["ledger_artifact_physical_line"]
                )
                if not is_terminal_artifact:
                    raise PreflightError(
                        f"Unexpected malformed ledger record at physical line {end_line} "
                        f"(width {len(row)}): {raw[:120]!r}"
                    )
                excluded.append({
                    "physical_line": end_line,
                    "field_count": len(row),
                    "raw_value": raw,
                    "reason": "verified terminal non-observation artifact (D63)",
                })
                continue

            record = dict(zip(LEDGER_HEADER, row))
            code_text = record["admin_code"].strip()
            if not code_text.isdigit():
                raise PreflightError(
                    f"Ledger record at physical line {start_line} has a non-canonical "
                    f"admin_code {code_text!r}"
                )
            code = int(code_text)
            ym = record["year_month"].strip()
            if len(ym) != 7 or ym[4] != "_" or not (ym[:4].isdigit() and ym[5:].isdigit()):
                raise PreflightError(
                    f"Ledger record at physical line {start_line} has an invalid "
                    f"year_month {ym!r}"
                )
            year = int(ym[:4])
            month = int(ym[5:])
            if not 1 <= month <= 12:
                raise PreflightError(
                    f"Ledger record at physical line {start_line} has month {month}"
                )
            if int(record["year"]) != year or int(record["month"]) != month:
                raise PreflightError(
                    f"Ledger record at physical line {start_line} has redundant date fields "
                    f"disagreeing with year_month {ym!r}"
                )
            n_valid_key += 1
            areas_seen.add(code)
            position = int(np.searchsorted(areas, code))
            if position >= n_areas or int(areas[position]) != code:
                ledger_only_areas.add(code)
                continue
            m_index = month_index(year, month)
            if not 0 <= m_index < N_GRID_MONTHS:
                raise PreflightError(
                    f"Ledger month {ym} at physical line {start_line} is outside the "
                    "prepared calendar grid."
                )
            observed_months.add(m_index)
            if present[position, m_index]:
                duplicate_keys += 1
                raise PreflightError(
                    f"Duplicate ledger canonical key {code}/{year}-{month:02d} at physical "
                    f"line {start_line}; silent deduplication is prohibited."
                )
            present[position, m_index] = True
            phase_text = record["fews_ipc"].strip()
            if phase_text == "":
                phase_missing += 1
                continue
            value = float(phase_text)
            if not math.isfinite(value) or value != round(value) or not 1 <= value <= 5:
                raise PreflightError(
                    f"Ledger record at physical line {start_line} has invalid observed "
                    f"phase {phase_text!r}"
                )
            phase_counts[int(value)] += 1
            phase[position, m_index] = value

    if ledger_only_areas:
        raise PreflightError(
            f"{len(ledger_only_areas)} ledger areas are absent from the master cohort: "
            f"{sorted(ledger_only_areas)[:10]}"
        )
    report["ledger_parse_seconds"] = round(time.time() - started, 2)
    report["ledger_records"] = n_records
    report["ledger_valid_key_records"] = n_valid_key
    report["ledger_unique_keys"] = int(present.sum())
    report["ledger_duplicate_keys"] = duplicate_keys
    report["ledger_malformed_records"] = len(excluded)
    report["ledger_excluded_records"] = excluded
    report["ledger_areas"] = len(areas_seen)
    report["ledger_months"] = len(observed_months)
    report["ledger_first_month"] = month_label(min(observed_months))
    report["ledger_last_month"] = month_label(max(observed_months))
    report["ledger_phase_counts"] = {int(k): int(v) for k, v in phase_counts.items()}
    report["ledger_phase_missing"] = phase_missing
    if verbose:
        print(
            f"[ledger] parsed {n_records:,} records "
            f"({len(excluded)} excluded artifact) in {report['ledger_parse_seconds']}s",
            flush=True,
        )
    return LedgerData(phase=phase, present=present, report=report)


def reconcile_master_and_ledger(master: MasterGrid, ledger: LedgerData) -> Dict[str, object]:
    """Exact same-key reconciliation over every master key (D63)."""
    master_slice = slice(month_index(*MASTER_FIRST_MONTH), month_index(*MASTER_LAST_MONTH) + 1)
    m_phase = master.phase[:, master_slice]
    l_phase = ledger.phase[:, master_slice]
    l_present = ledger.present[:, master_slice]

    m_valid = ~np.isnan(m_phase)
    l_valid = ~np.isnan(l_phase)
    same_key = l_present
    agree = int(np.sum(same_key & m_valid & l_valid & (m_phase == l_phase)))
    conflict = int(np.sum(same_key & m_valid & l_valid & (m_phase != l_phase)))
    both_missing = int(np.sum(same_key & ~m_valid & ~l_valid))
    one_missing = int(np.sum(same_key & (m_valid != l_valid)))
    absent = ~same_key
    absent_master_missing = int(np.sum(absent & ~m_valid))
    absent_master_observed = int(np.sum(absent & m_valid))

    if conflict:
        raise PreflightError(f"{conflict} master/ledger same-key phase conflicts.")
    if one_missing:
        raise PreflightError(
            f"{one_missing} master/ledger same-key rows disagree about phase missingness."
        )
    if absent_master_observed:
        raise PreflightError(
            f"{absent_master_observed} observed master labels have no ledger key."
        )

    ledger_only = int(np.sum(ledger.present) - np.sum(same_key))
    ledger_only_by_month: Dict[str, int] = {}
    outside = ledger.present.copy()
    outside[:, master_slice] = False
    for m in np.flatnonzero(outside.any(axis=0)):
        ledger_only_by_month[month_label(int(m))] = int(outside[:, m].sum())

    return {
        "cross_same_key_agree": agree,
        "cross_same_key_both_missing": both_missing,
        "cross_same_key_one_missing": one_missing,
        "cross_same_key_conflict": conflict,
        "cross_ledger_absent_master_missing": absent_master_missing,
        "cross_ledger_absent_master_observed": absent_master_observed,
        "cross_overlap_keys": int(np.sum(same_key)),
        "cross_ledger_only_keys": ledger_only,
        "cross_ledger_only_keys_by_month": ledger_only_by_month,
    }


def build_observed_history(master: MasterGrid, ledger: LedgerData) -> Dict[str, np.ndarray]:
    """Combined observed-phase history: master observations plus ledger-only observations.

    Earlier ledger observations support eligible history (block A, reference IPC lags,
    exact-origin persistence) without extending the master target cohort.
    """
    phase = master.phase.copy()
    source = np.where(~np.isnan(master.phase), 1, 0).astype(np.int8)  # 1 = master
    ledger_only = np.isnan(phase) & ~np.isnan(ledger.phase)
    phase[ledger_only] = ledger.phase[ledger_only]
    source[ledger_only] = 2  # 2 = ledger-only observation
    crisis = np.where(np.isnan(phase), np.nan, (phase >= 3).astype(np.float64))
    return {"phase": phase, "crisis": crisis, "source": source}


# --------------------------------------------------------------------------------------
# 7. Additional approved sources (D31 / D50 / D64)
# --------------------------------------------------------------------------------------

ENSO_SENTINELS: Tuple[float, ...] = (-99.99, -9999.0)


def load_enso_series(path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    """nino34_anom on the prepared grid; both documented sentinel encodings are missing."""
    frame = pd.read_csv(path)
    if "Date" not in frame.columns:
        raise PreflightError("ENSO source is missing its Date column.")
    value_columns = [c for c in frame.columns if c != "Date"]
    if len(value_columns) != 1:
        raise PreflightError(f"ENSO source must have one value column, found {value_columns}.")
    parsed = pd.to_datetime(frame["Date"], errors="coerce")
    if parsed.isna().any():
        raise PreflightError("ENSO source has unparseable Date values.")
    values = pd.to_numeric(frame[value_columns[0]], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(values).any():
        raise PreflightError("ENSO source has non-numeric value entries.")
    sentinel_mask = np.zeros(values.shape, dtype=bool)
    sentinel_counts: Dict[str, int] = {}
    for sentinel in ENSO_SENTINELS:
        hit = np.isclose(values, sentinel, rtol=0.0, atol=1e-9)
        sentinel_counts[str(sentinel)] = int(hit.sum())
        sentinel_mask |= hit
    years = parsed.dt.year.to_numpy()
    months = parsed.dt.month.to_numpy()
    keys = list(zip(years.tolist(), months.tolist()))
    if len(set(keys)) != len(keys):
        raise PreflightError("ENSO source has duplicate year-month rows.")

    series = np.full(N_GRID_MONTHS, np.nan, dtype=np.float64)
    in_grid = 0
    for (year, month), value, is_sentinel in zip(keys, values, sentinel_mask):
        if is_sentinel:
            continue
        index = month_index(year, month)
        if 0 <= index < N_GRID_MONTHS:
            series[index] = value
            in_grid += 1
    valid_indices = np.flatnonzero(~np.isnan(series))
    report = {
        "source_rows": int(len(frame)),
        "value_column": value_columns[0],
        "sentinel_counts": sentinel_counts,
        "sentinel_rows_total": int(sentinel_mask.sum()),
        "valid_months_on_grid": in_grid,
        "grid_first_valid": month_label(int(valid_indices.min())) if valid_indices.size else None,
        "grid_last_valid": month_label(int(valid_indices.max())) if valid_indices.size else None,
    }
    return series, report


BLOOMBERG_FILE_FIELDS: Dict[str, Tuple[str, ...]] = {
    "bbg_fertiliser": BBG_FERTILISER_FIELDS + ("bbg_NGUSHHUB", "bbg_TZTX2_Comdty"),
    "bbg_oil_and_gas": ("bbg_oilgas_CL1_COMB_Comdty_price", "bbg_oilgas_NG1_Comdty_price"),
    "bbg_soybean_oil": BBG_SOYBEAN_OIL_FIELDS,
    "bbg_staple_food": BBG_STAPLE_FIELDS,
}


def load_bloomberg_series(paths: Dict[str, Path]) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """The approved 18 Bloomberg monthly series on the prepared grid; missing stays missing."""
    series: Dict[str, np.ndarray] = {
        field: np.full(N_GRID_MONTHS, np.nan, dtype=np.float64) for field in BBG_FIELDS
    }
    report: Dict[str, object] = {"files": {}, "excluded_fields": list(EXCLUDED_ADDITIONAL_FIELDS)}
    for key, fields in BLOOMBERG_FILE_FIELDS.items():
        path = paths[key]
        frame = pd.read_csv(path)
        missing = [field for field in fields if field not in frame.columns]
        if missing:
            raise PreflightError(f"{path.name} is missing approved fields {missing}.")
        if frame.duplicated(["year", "month"]).any():
            raise PreflightError(f"{path.name} has duplicate year/month rows.")
        years = pd.to_numeric(frame["year"], errors="raise").astype(int).to_numpy()
        months = pd.to_numeric(frame["month"], errors="raise").astype(int).to_numpy()
        if months.min() < 1 or months.max() > 12:
            raise PreflightError(f"{path.name} has a month outside 1-12.")
        indices = (years - GRID_START_YEAR) * 12 + (months - 1)
        on_grid = (indices >= 0) & (indices < N_GRID_MONTHS)
        field_report: Dict[str, object] = {
            "rows": int(len(frame)),
            "months_on_grid": int(on_grid.sum()),
            "fields": {},
        }
        for field in fields:
            values = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=np.float64)
            raw_nonempty = frame[field].notna().to_numpy()
            coerced_loss = int(np.sum(raw_nonempty & np.isnan(values)))
            if coerced_loss:
                raise PreflightError(
                    f"{path.name}:{field} has {coerced_loss} non-numeric nonempty values; "
                    "refusing to coerce unknown tokens into missingness."
                )
            series[field][indices[on_grid]] = values[on_grid]
            valid = np.flatnonzero(~np.isnan(series[field]))
            field_report["fields"][field] = {
                "valid_months_on_grid": int(valid.size),
                "first_valid": month_label(int(valid.min())) if valid.size else None,
                "last_valid": month_label(int(valid.max())) if valid.size else None,
                "entirely_missing_on_grid": bool(valid.size == 0),
            }
        report["files"][key] = field_report
    report["fields_entirely_missing"] = sorted(
        field for field in BBG_FIELDS if not np.any(~np.isnan(series[field]))
    )
    return series, report


WB_DERIVED_COLUMNS: Tuple[str, ...] = (
    "inflation_food_price_index", "year", "month", "lat", "lon",
    "food_price_index_WB", "food_inflation_wb",
)
WB_RAW_LINEAGE_COLUMNS: Tuple[str, ...] = (
    "geo_id", "year", "month", "lat", "lon",
    "o_food_price_index", "c_food_price_index", "inflation_food_price_index",
)
_TIE_PROBE_K = 8


def _valid_coordinates(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(lat) & np.isfinite(lon)
        & (lat >= -90.0) & (lat <= 90.0) & (lon >= -180.0) & (lon <= 180.0)
    )


def load_wb_markets(derived_path: Path, raw_path: Path, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Restore raw geo_id onto the selected derived WB values after D64 lineage checks."""
    started = time.time()
    derived = pd.read_csv(derived_path)
    if tuple(derived.columns) != WB_DERIVED_COLUMNS:
        raise PreflightError(
            f"WB derived header changed; expected {WB_DERIVED_COLUMNS}, got {tuple(derived.columns)}"
        )
    raw = pd.read_csv(raw_path, usecols=list(WB_RAW_LINEAGE_COLUMNS))
    raw = raw[list(WB_RAW_LINEAGE_COLUMNS)]
    read_seconds = time.time() - started
    if verbose:
        print(
            f"[wb] read derived ({len(derived):,}) and raw ({len(raw):,}) rows "
            f"in {read_seconds:.1f}s",
            flush=True,
        )

    if len(raw) != len(derived):
        raise PreflightError(
            f"WB raw/derived row counts differ ({len(raw)} vs {len(derived)}); "
            "lineage restoration rejected."
        )
    for column in ("year", "month"):
        if not np.array_equal(
            pd.to_numeric(raw[column], errors="raise").to_numpy(),
            pd.to_numeric(derived[column], errors="raise").to_numpy(),
        ):
            raise PreflightError(f"WB raw/derived {column} ordering differs; lineage rejected.")
    for column in ("lat", "lon"):
        raw_values = pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=np.float64)
        derived_values = pd.to_numeric(derived[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.isclose(raw_values, derived_values, equal_nan=True, rtol=0.0, atol=0.0)):
            raise PreflightError(f"WB raw/derived {column} values differ; lineage rejected.")

    recomputed_index = (
        pd.to_numeric(raw["o_food_price_index"], errors="coerce").to_numpy(dtype=np.float64)
        + pd.to_numeric(raw["c_food_price_index"], errors="coerce").to_numpy(dtype=np.float64)
    ) / 2.0
    retained_index = pd.to_numeric(derived["food_price_index_WB"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isclose(recomputed_index, retained_index, equal_nan=True)):
        mismatches = int(np.sum(~np.isclose(recomputed_index, retained_index, equal_nan=True)))
        raise PreflightError(
            f"WB food_price_index_WB does not reconcile to mean(o, c) on {mismatches} rows."
        )
    raw_inflation = pd.to_numeric(raw["inflation_food_price_index"], errors="coerce").to_numpy(dtype=np.float64)
    retained_inflation = pd.to_numeric(derived["food_inflation_wb"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isclose(raw_inflation, retained_inflation, equal_nan=True)):
        mismatches = int(np.sum(~np.isclose(raw_inflation, retained_inflation, equal_nan=True)))
        raise PreflightError(
            f"WB food_inflation_wb does not reconcile to the raw inflation field on "
            f"{mismatches} rows."
        )

    markets = pd.DataFrame({
        "geo_id": raw["geo_id"].astype(str).to_numpy(),
        "year": pd.to_numeric(derived["year"], errors="raise").astype(int).to_numpy(),
        "month": pd.to_numeric(derived["month"], errors="raise").astype(int).to_numpy(),
        "lat": pd.to_numeric(derived["lat"], errors="coerce").to_numpy(dtype=np.float64),
        "lon": pd.to_numeric(derived["lon"], errors="coerce").to_numpy(dtype=np.float64),
        "food_price_index_WB": retained_index,
        "food_inflation_wb": retained_inflation,
    })
    if markets.duplicated(["geo_id", "year", "month"]).any():
        n_dup = int(markets.duplicated(["geo_id", "year", "month"]).sum())
        raise PreflightError(
            f"{n_dup} conflicting duplicate WB market/month keys; assembly stopped."
        )
    if not markets["month"].between(1, 12).all():
        raise PreflightError("WB source has a month outside 1-12.")

    valid_coord = _valid_coordinates(markets["lat"].to_numpy(), markets["lon"].to_numpy())
    coordinate_groups = markets.loc[valid_coord].groupby(["lat", "lon", "year", "month"]).size()
    report = {
        "derived_rows": int(len(derived)),
        "raw_rows": int(len(raw)),
        "read_seconds": round(read_seconds, 2),
        "unique_geo_ids": int(markets["geo_id"].nunique()),
        "months": int(markets.groupby(["year", "month"]).ngroups),
        "rows_missing_coordinates": int(np.sum(~valid_coord)),
        "rows_missing_price": int(np.sum(np.isnan(retained_index))),
        "rows_missing_inflation": int(np.sum(np.isnan(retained_inflation))),
        "colocated_market_month_groups": int(np.sum(coordinate_groups.to_numpy() > 1)),
        "lineage_verified": True,
        "excluded_alias": "inflation_food_price_index",
    }
    return markets, report


def join_wb_to_areas(
    markets: pd.DataFrame,
    area_lat: np.ndarray,
    area_lon: np.ndarray,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """D9/D64 same-month nearest-market join with deterministic lexical geo_id ties."""
    from sklearn.neighbors import BallTree

    n_areas = int(area_lat.shape[0])
    values = {
        field: np.full((n_areas, N_GRID_MONTHS), np.nan, dtype=np.float64) for field in WB_FIELDS
    }
    match_distance = np.full((n_areas, N_GRID_MONTHS), np.nan, dtype=np.float64)
    match_market = np.full((n_areas, N_GRID_MONTHS), -1, dtype=np.int32)

    area_valid = _valid_coordinates(area_lat, area_lon)
    query_rows = np.flatnonzero(area_valid)
    query_coords = np.radians(np.column_stack([area_lat[query_rows], area_lon[query_rows]]))
    if query_rows.size == 0:
        return (
            {"values": values,
             "provenance": {"match_distance_km": match_distance,
                            "match_market_index": match_market,
                            "market_ids": np.array(sorted(markets["geo_id"].unique()), dtype=object)}},
            {
                "seconds": 0.0,
                "areas_with_valid_coordinates": 0,
                "areas_without_valid_coordinates": int(n_areas),
                "grid_months": N_GRID_MONTHS,
                "matched_area_months": 0,
                "unmatched_area_months_over_100km": 0,
                "exact_distance_tie_queries": 0,
                "note": "no area has valid coordinates; every WB input stays missing",
                "earth_radius_km": WB_EARTH_RADIUS_KM,
                "max_distance_km": WB_MAX_MATCH_DISTANCE_KM,
                "tie_rule": "ascending lexical raw geo_id",
            },
        )

    geo_ids_sorted_unique = np.array(sorted(markets["geo_id"].unique()), dtype=object)
    geo_id_position = {geo: index for index, geo in enumerate(geo_ids_sorted_unique.tolist())}

    candidate_valid = _valid_coordinates(markets["lat"].to_numpy(), markets["lon"].to_numpy())
    candidates = markets.loc[candidate_valid]
    grouped = {key: group for key, group in candidates.groupby(["year", "month"], sort=False)}

    started = time.time()
    months_matched = 0
    tie_rows = 0
    over_threshold = 0
    no_candidate_months = 0
    distances: List[np.ndarray] = []
    for m_index in range(N_GRID_MONTHS):
        year, month = index_to_month(m_index)
        group = grouped.get((year, month))
        if group is None or group.empty:
            no_candidate_months += 1
            continue
        group = group.sort_values("geo_id", kind="stable")
        coords = np.radians(group[["lat", "lon"]].to_numpy(dtype=np.float64))
        tree = BallTree(coords, metric="haversine")
        k = min(_TIE_PROBE_K, len(group))
        while True:
            distance, index = tree.query(query_coords, k=k)
            minimum = distance[:, :1]
            tie_mask = distance == minimum
            if k == len(group) or not bool(np.all(tie_mask, axis=1).any()):
                break
            k = min(len(group), k * 4)
        chosen_position = np.where(tie_mask, index, np.iinfo(np.int64).max).min(axis=1)
        tie_rows += int(np.sum(tie_mask.sum(axis=1) > 1))
        distance_km = minimum[:, 0] * WB_EARTH_RADIUS_KM
        within = distance_km <= WB_MAX_MATCH_DISTANCE_KM
        over_threshold += int(np.sum(~within))
        if not bool(within.any()):
            continue
        rows = query_rows[within]
        selected = group.iloc[chosen_position[within]]
        for field in WB_FIELDS:
            values[field][rows, m_index] = selected[field].to_numpy(dtype=np.float64)
        match_distance[rows, m_index] = distance_km[within]
        match_market[rows, m_index] = [
            geo_id_position[geo] for geo in selected["geo_id"].tolist()
        ]
        distances.append(distance_km[within])
        months_matched += 1

    all_distances = np.concatenate(distances) if distances else np.zeros(0)
    report = {
        "seconds": round(time.time() - started, 2),
        "areas_with_valid_coordinates": int(area_valid.sum()),
        "areas_without_valid_coordinates": int((~area_valid).sum()),
        "grid_months": N_GRID_MONTHS,
        "months_with_candidates": N_GRID_MONTHS - no_candidate_months,
        "months_without_candidates": no_candidate_months,
        "matched_area_months": int(np.sum(~np.isnan(match_distance))),
        "unmatched_area_months_over_100km": over_threshold,
        "exact_distance_tie_queries": tie_rows,
        "match_distance_km": {
            "min": float(all_distances.min()) if all_distances.size else None,
            "p25": float(np.percentile(all_distances, 25)) if all_distances.size else None,
            "median": float(np.median(all_distances)) if all_distances.size else None,
            "p75": float(np.percentile(all_distances, 75)) if all_distances.size else None,
            "p95": float(np.percentile(all_distances, 95)) if all_distances.size else None,
            "max": float(all_distances.max()) if all_distances.size else None,
        },
        "matched_missing_price_area_months": int(
            np.sum(~np.isnan(match_distance) & np.isnan(values["food_price_index_WB"]))
        ),
        "matched_missing_inflation_area_months": int(
            np.sum(~np.isnan(match_distance) & np.isnan(values["food_inflation_wb"]))
        ),
        "earth_radius_km": WB_EARTH_RADIUS_KM,
        "max_distance_km": WB_MAX_MATCH_DISTANCE_KM,
        "tie_rule": "ascending lexical raw geo_id",
    }
    if verbose:
        print(
            f"[wb] matched {report['matched_area_months']:,} area-months "
            f"({report['exact_distance_tie_queries']:,} exact-distance tie queries) "
            f"in {report['seconds']}s",
            flush=True,
        )
    provenance = {
        "match_distance_km": match_distance,
        "match_market_index": match_market,
        "market_ids": geo_ids_sorted_unique,
    }
    return {"values": values, "provenance": provenance}, report


def extract_coastline(
    raster_path: Path, area_lat: np.ndarray, area_lon: np.ndarray
) -> Tuple[np.ndarray, Dict[str, object]]:
    """D50 containing-pixel lookup at (lon, lat); native signed/zero values preserved."""
    import rasterio

    n_areas = int(area_lat.shape[0])
    values = np.full(n_areas, np.nan, dtype=np.float64)
    rows = np.full(n_areas, -1, dtype=np.int64)
    cols = np.full(n_areas, -1, dtype=np.int64)
    reasons = np.empty(n_areas, dtype=object)
    reasons[:] = "valid_pixel"

    valid_coord = _valid_coordinates(area_lat, area_lon)
    reasons[~valid_coord] = "invalid_or_missing_coordinate"

    with rasterio.open(raster_path) as dataset:
        raster_identity = {
            "crs": str(dataset.crs),
            "transform": [float(v) for v in dataset.transform.to_gdal()],
            "width": int(dataset.width),
            "height": int(dataset.height),
            "dtype": dataset.dtypes[0],
            "nodata": None if dataset.nodata is None else float(dataset.nodata),
            "bounds": [float(v) for v in dataset.bounds],
            "tags": {
                key: dataset.tags().get(key)
                for key in ("AREA_OR_POINT", "TIFFTAG_IMAGEDESCRIPTION")
                if key in dataset.tags()
            },
        }
        if str(dataset.crs).upper() not in ("EPSG:4326",):
            raise PreflightError(f"Coastline raster CRS is {dataset.crs}, expected EPSG:4326.")
        indices = np.flatnonzero(valid_coord)
        for position in indices:
            row, col = dataset.index(float(area_lon[position]), float(area_lat[position]))
            rows[position] = row
            cols[position] = col
            if not (0 <= row < dataset.height and 0 <= col < dataset.width):
                reasons[position] = "pixel_outside_raster_bounds"
                continue
            window = rasterio.windows.Window(col, row, 1, 1)
            block = dataset.read(1, window=window, masked=True)
            if bool(np.ma.getmaskarray(block)[0, 0]):
                reasons[position] = "raster_invalid_mask"
                continue
            values[position] = float(block[0, 0])

    finite = ~np.isnan(values)
    report = {
        "raster": raster_identity,
        "areas": n_areas,
        "areas_with_value": int(finite.sum()),
        "areas_missing": int((~finite).sum()),
        "reason_counts": {
            reason: int(np.sum(reasons == reason)) for reason in sorted(set(reasons.tolist()))
        },
        "value_summary": {
            "min": float(values[finite].min()) if finite.any() else None,
            "max": float(values[finite].max()) if finite.any() else None,
            "negative": int(np.sum(values[finite] < 0)),
            "zero": int(np.sum(values[finite] == 0)),
            "positive": int(np.sum(values[finite] > 0)),
        },
        "units": "native raster units (kilometre/sign interpretation unverified, D50)",
        "transform_applied": "none (no absolute value, rescaling or interpolation)",
    }
    provenance = {"pixel_row": rows, "pixel_col": cols, "reason": reasons}
    return values, {"report": report, "provenance": provenance}


# --------------------------------------------------------------------------------------
# 8. Prepared source grid
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class PreparedSources:
    """Validated, origin-independent source panel on the complete calendar grid."""

    areas: np.ndarray
    area_monthly: Dict[str, np.ndarray]
    global_monthly: Dict[str, np.ndarray]
    static: Dict[str, np.ndarray]
    annual: Dict[str, np.ndarray]
    pop_selected_month: np.ndarray
    observed_phase: np.ndarray
    observed_crisis: np.ndarray
    observed_source: np.ndarray
    target_area_idx: np.ndarray
    target_month_idx: np.ndarray
    target_label: np.ndarray
    report: Dict[str, object]

    @property
    def n_areas(self) -> int:
        return int(self.areas.shape[0])

    @property
    def n_targets(self) -> int:
        return int(self.target_month_idx.shape[0])

    # -- monthly access ------------------------------------------------------------
    def monthly_at(self, field: str, area_idx: np.ndarray, m_idx: np.ndarray) -> np.ndarray:
        out = np.full(m_idx.shape, np.nan, dtype=np.float64)
        usable = (m_idx >= 0) & (m_idx < N_GRID_MONTHS)
        if not usable.any():
            return out
        if field in self.global_monthly:
            out[usable] = self.global_monthly[field][m_idx[usable]]
        elif field in self.area_monthly:
            out[usable] = self.area_monthly[field][area_idx[usable], m_idx[usable]]
        else:
            raise KeyError(f"{field!r} is not a prepared monthly source")
        return out

    def monthly_window(
        self, field: str, area_idx: np.ndarray, m_end: np.ndarray, width: int,
        include_end: bool = True,
    ) -> np.ndarray:
        """Stack of `width` consecutive calendar months ending at m_end (or m_end-1)."""
        shift = 0 if include_end else 1
        stack = np.empty((m_end.shape[0], width), dtype=np.float64)
        for position in range(width):
            offset = (width - 1 - position) + shift
            stack[:, position] = self.monthly_at(field, area_idx, m_end - offset)
        return stack

    def monthly_last_valid(self, field: str) -> np.ndarray:
        """Per-month index of the latest month at or before it holding a valid value."""
        if field in self.global_monthly:
            valid = ~np.isnan(self.global_monthly[field])
            return np.maximum.accumulate(np.where(valid, np.arange(N_GRID_MONTHS), -1))
        grid = self.area_monthly[field]
        valid = ~np.isnan(grid)
        return np.maximum.accumulate(
            np.where(valid, np.arange(N_GRID_MONTHS)[None, :], -1), axis=1
        )

    # -- annual access -------------------------------------------------------------
    def annual_at(self, field: str, area_idx: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
        out = np.full(y_idx.shape, np.nan, dtype=np.float64)
        usable = (y_idx >= 0) & (y_idx < N_GRID_YEARS)
        if usable.any():
            out[usable] = self.annual[field][area_idx[usable], y_idx[usable]]
        return out

    def annual_last_valid(self, field: str) -> np.ndarray:
        valid = ~np.isnan(self.annual[field])
        return np.maximum.accumulate(
            np.where(valid, np.arange(N_GRID_YEARS)[None, :], -1), axis=1
        )

    def static_at(self, field: str, area_idx: np.ndarray) -> np.ndarray:
        return self.static[field][area_idx]


def assemble_prepared_sources(
    master: MasterGrid,
    ledger: LedgerData,
    enso: np.ndarray,
    bloomberg: Dict[str, np.ndarray],
    wb_values: Dict[str, np.ndarray],
    coastline: np.ndarray,
    extra_report: Dict[str, object],
) -> PreparedSources:
    area_monthly: Dict[str, np.ndarray] = dict(master.monthly)
    for field in WB_FIELDS:
        area_monthly[field] = wb_values[field]
    global_monthly: Dict[str, np.ndarray] = {ENSO_FIELD: enso}
    global_monthly.update(bloomberg)
    static = dict(master.static)
    static[COASTLINE_FIELD] = coastline

    missing_monthly = [
        field for field in AREA_MONTHLY_FIELDS + GLOBAL_MONTHLY_FIELDS
        if field not in area_monthly and field not in global_monthly
    ]
    if missing_monthly:
        raise PreflightError(f"Prepared grid is missing monthly sources {missing_monthly}.")
    missing_static = [field for field in STATIC_FIELDS if field not in static]
    if missing_static:
        raise PreflightError(f"Prepared grid is missing static sources {missing_static}.")

    history = build_observed_history(master, ledger)
    report = dict(master.report)
    report.update(extra_report)
    return PreparedSources(
        areas=master.areas,
        area_monthly=area_monthly,
        global_monthly=global_monthly,
        static=static,
        annual=master.annual,
        pop_selected_month=master.pop_selected_month,
        observed_phase=history["phase"],
        observed_crisis=history["crisis"],
        observed_source=history["source"],
        target_area_idx=master.target_area_idx,
        target_month_idx=master.target_month_idx,
        target_label=master.target_label,
        report=report,
    )


# --------------------------------------------------------------------------------------
# 9. Origin-aligned feature construction (D23-D36, D46-D54)
# --------------------------------------------------------------------------------------


def _complete_window_mean_sd(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and population SD (ddof=0) over complete windows only."""
    complete = ~np.isnan(stack).any(axis=1)
    mean = np.full(stack.shape[0], np.nan, dtype=np.float64)
    sd = np.full(stack.shape[0], np.nan, dtype=np.float64)
    if complete.any():
        block = stack[complete]
        mean[complete] = block.mean(axis=1)
        sd[complete] = block.std(axis=1)  # ddof=0
    return mean, sd


def _complete_window_sum(stack: np.ndarray) -> np.ndarray:
    complete = ~np.isnan(stack).any(axis=1)
    out = np.full(stack.shape[0], np.nan, dtype=np.float64)
    if complete.any():
        out[complete] = stack[complete].sum(axis=1)
    return out


def _trailing_standardized_deviation(
    sources: PreparedSources, field: str, area_idx: np.ndarray, o_idx: np.ndarray
) -> np.ndarray:
    """[x(O) - mean(x(O-12..O-1))] / population SD of that reference window (D27)."""
    current = sources.monthly_at(field, area_idx, o_idx)
    reference = sources.monthly_window(field, area_idx, o_idx, 12, include_end=False)
    mean, sd = _complete_window_mean_sd(reference)
    out = np.full(current.shape, np.nan, dtype=np.float64)
    usable = ~np.isnan(current) & ~np.isnan(mean) & ~np.isnan(sd) & (sd != 0.0)
    out[usable] = (current[usable] - mean[usable]) / sd[usable]
    return out


def _conflict_events_3m(
    sources: PreparedSources, area_idx: np.ndarray, o_idx: np.ndarray
) -> np.ndarray:
    """Sum of the three event categories over three months ending at O; all nine required."""
    total = np.zeros(o_idx.shape[0], dtype=np.float64)
    valid = np.ones(o_idx.shape[0], dtype=bool)
    for field in ("event_count_battles", "event_count_explosions", "event_count_violence"):
        stack = sources.monthly_window(field, area_idx, o_idx, 3, include_end=True)
        valid &= ~np.isnan(stack).any(axis=1)
        total += np.nan_to_num(stack, nan=0.0).sum(axis=1)
    return np.where(valid, total, np.nan)


def build_base_block(
    sources: PreparedSources, area_idx: np.ndarray, o_idx: np.ndarray, fields: Sequence[str]
) -> np.ndarray:
    """Ordered source-level BASE values at each row's own origin (D34/D35/D48/D49/D50)."""
    origin_year = GRID_START_YEAR + o_idx // 12
    reference_year_index = (origin_year - 1) - GRID_START_YEAR
    out = np.empty((o_idx.shape[0], len(fields)), dtype=np.float64)
    for position, field in enumerate(fields):
        role = field_time_role(field)
        if role == "monthly":
            out[:, position] = sources.monthly_at(field, area_idx, o_idx)
        elif role == "annual":
            out[:, position] = sources.annual_at(field, area_idx, reference_year_index)
        else:
            out[:, position] = sources.static_at(field, area_idx)
    return out


def build_block_a(
    sources: PreparedSources, area_idx: np.ndarray, t_idx: np.ndarray, o_idx: np.ndarray
) -> np.ndarray:
    """Eight season/observed-IPC-history features (D25)."""
    target_month = (t_idx % 12) + 1
    angle = 2.0 * np.pi * (target_month - 1) / 12.0
    phase_valid = ~np.isnan(sources.observed_phase)
    crisis_flag = np.where(np.isnan(sources.observed_crisis), 0.0, sources.observed_crisis) == 1.0
    last_obs = np.maximum.accumulate(
        np.where(phase_valid, np.arange(N_GRID_MONTHS)[None, :], -1), axis=1
    )[area_idx, o_idx]
    last_crisis = np.maximum.accumulate(
        np.where(crisis_flag, np.arange(N_GRID_MONTHS)[None, :], -1), axis=1
    )[area_idx, o_idx]

    has_history = last_obs >= 0
    has_crisis = last_crisis >= 0
    safe_obs = np.where(has_history, last_obs, 0)
    phase = np.where(has_history, sources.observed_phase[area_idx, safe_obs], np.nan)
    crisis = np.where(has_history, sources.observed_crisis[area_idx, safe_obs], np.nan)
    age = np.where(has_history, (o_idx - last_obs).astype(np.float64), np.nan)
    crisis_age = np.where(has_crisis, (o_idx - last_crisis).astype(np.float64), np.nan)

    columns = {
        "target_month_sin": np.sin(angle),
        "target_month_cos": np.cos(angle),
        "last_observed_ipc_phase": phase,
        "last_observed_crisis": crisis,
        "last_observed_ipc_age_months": age,
        "months_since_last_observed_crisis": crisis_age,
        "no_observed_ipc_history": (~has_history).astype(np.float64),
        "no_prior_observed_crisis": (~has_crisis).astype(np.float64),
    }
    return np.column_stack([columns[name] for name in BLOCK_A_COLUMNS])


def build_block_b(
    sources: PreparedSources, area_idx: np.ndarray, o_idx: np.ndarray
) -> np.ndarray:
    """186 complete-window means/population SDs and count sums (D26/D32)."""
    blocks: List[np.ndarray] = []
    for field in B_CONTINUOUS_FIELDS:
        for window in B_WINDOWS:
            stack = sources.monthly_window(field, area_idx, o_idx, window, include_end=True)
            mean, sd = _complete_window_mean_sd(stack)
            blocks.append(mean)
            blocks.append(sd)
    for field in B_COUNT_FIELDS:
        for window in B_WINDOWS:
            stack = sources.monthly_window(field, area_idx, o_idx, window, include_end=True)
            blocks.append(_complete_window_sum(stack))
    matrix = np.column_stack(blocks)
    if matrix.shape[1] != len(BLOCK_B_COLUMNS):
        raise PreflightError("Block B width does not match its frozen schema.")
    return matrix


def build_block_c(
    sources: PreparedSources, area_idx: np.ndarray, o_idx: np.ndarray
) -> np.ndarray:
    """102 changes and trailing standardized deviations (D27/D32)."""
    blocks: List[np.ndarray] = []
    for field in B_CONTINUOUS_FIELDS + B_COUNT_FIELDS:
        current = sources.monthly_at(field, area_idx, o_idx)
        blocks.append(current - sources.monthly_at(field, area_idx, o_idx - 3))
        blocks.append(current - sources.monthly_at(field, area_idx, o_idx - 12))
        blocks.append(_trailing_standardized_deviation(sources, field, area_idx, o_idx))
    matrix = np.column_stack(blocks)
    if matrix.shape[1] != len(BLOCK_C_COLUMNS):
        raise PreflightError("Block C width does not match its frozen schema.")
    return matrix


def build_block_d(
    sources: PreparedSources, area_idx: np.ndarray, o_idx: np.ndarray
) -> np.ndarray:
    """Exactly four approved products; missing operands leave the product missing (D28)."""
    z_rain = _trailing_standardized_deviation(sources, "Rainf_f_tavg_mean", area_idx, o_idx)
    z_evi = _trailing_standardized_deviation(sources, "EVI", area_idx, o_idx)
    conflict_3m = _conflict_events_3m(sources, area_idx, o_idx)
    inflation = sources.monthly_at("food_inflation_wb", area_idx, o_idx)
    market_distance = sources.monthly_at("market_distance", area_idx, o_idx)
    return np.column_stack([
        z_rain * z_evi,
        z_rain * conflict_3m,
        inflation * conflict_3m,
        inflation * market_distance,
    ])


def build_block_e(
    sources: PreparedSources, area_idx: np.ndarray, o_idx: np.ndarray
) -> np.ndarray:
    """140 availability features: 86 pre-imputation missing flags plus 54 ages (D29/D53)."""
    origin_year = GRID_START_YEAR + o_idx // 12
    origin_month = (o_idx % 12) + 1
    reference_year_index = (origin_year - 1) - GRID_START_YEAR
    blocks: List[np.ndarray] = []
    for field in UPDATED_BASE_FIELDS:
        role = field_time_role(field)
        if role == "monthly":
            value = sources.monthly_at(field, area_idx, o_idx)
            blocks.append(np.isnan(value).astype(np.float64))
            last_valid = sources.monthly_last_valid(field)
            latest = last_valid[o_idx] if field in sources.global_monthly else last_valid[area_idx, o_idx]
            age = np.where(latest >= 0, (o_idx - latest).astype(np.float64), np.nan)
            blocks.append(age)
        elif role == "annual":
            value = sources.annual_at(field, area_idx, reference_year_index)
            blocks.append(np.isnan(value).astype(np.float64))
            last_valid = sources.annual_last_valid(field)
            safe_index = np.clip(reference_year_index, 0, N_GRID_YEARS - 1)
            latest = np.where(
                (reference_year_index >= 0) & (reference_year_index < N_GRID_YEARS),
                last_valid[area_idx, safe_index],
                -1,
            )
            # Annual endpoint is December of the latest eligible valid reference year.
            age = np.where(
                latest >= 0,
                (origin_year - (GRID_START_YEAR + latest)) * 12.0 + origin_month - 12.0,
                np.nan,
            )
            blocks.append(age)
        else:
            blocks.append(np.isnan(sources.static_at(field, area_idx)).astype(np.float64))
    matrix = np.column_stack(blocks)
    if matrix.shape[1] != len(BLOCK_E_COLUMNS):
        raise PreflightError("Block E width does not match its frozen schema.")
    return matrix


def build_updated_superset(
    sources: PreparedSources, horizon: int, rows: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """BASE + A + B + C + D + E on labeled target rows; every recipe is an ordered subset."""
    area_idx, t_idx, o_idx = origin_alignment(sources, horizon, rows)
    parts = [
        build_base_block(sources, area_idx, o_idx, UPDATED_BASE_FIELDS),
        build_block_a(sources, area_idx, t_idx, o_idx),
        build_block_b(sources, area_idx, o_idx),
        build_block_c(sources, area_idx, o_idx),
        build_block_d(sources, area_idx, o_idx),
        build_block_e(sources, area_idx, o_idx),
    ]
    matrix = np.concatenate(parts, axis=1)
    columns = updated_superset_columns()
    if matrix.shape[1] != len(columns):
        raise PreflightError(
            f"Updated superset width {matrix.shape[1]} does not match the frozen "
            f"{len(columns)}-column schema."
        )
    return matrix, columns


def build_reference_matrix(
    sources: PreparedSources, horizon: int, rows: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """The 109-column corrected original-feature reference (D54)."""
    area_idx, t_idx, o_idx = origin_alignment(sources, horizon, rows)
    parts: List[np.ndarray] = [
        build_base_block(sources, area_idx, o_idx, REFERENCE_SOURCE_FIELDS)
    ]

    target_year = GRID_START_YEAR + t_idx // 12
    target_month = (t_idx % 12) + 1
    parts.append(np.column_stack([
        (target_year == year).astype(np.float64) for year in REFERENCE_YEARS
    ]))
    parts.append(np.column_stack([
        (target_month == month).astype(np.float64) for month in REFERENCE_MONTHS
    ]))

    phase_lags: List[np.ndarray] = []
    crisis_lags: List[np.ndarray] = []
    for lag in REFERENCE_IPC_LAGS:
        m_idx = o_idx - lag
        usable = m_idx >= 0
        phase = np.full(m_idx.shape, np.nan, dtype=np.float64)
        crisis = np.full(m_idx.shape, np.nan, dtype=np.float64)
        phase[usable] = sources.observed_phase[area_idx[usable], m_idx[usable]]
        crisis[usable] = sources.observed_crisis[area_idx[usable], m_idx[usable]]
        phase_lags.append(phase)
        crisis_lags.append(crisis)
    parts.append(np.column_stack(crisis_lags))
    parts.append(np.column_stack(phase_lags))

    parts.append(np.column_stack([
        sources.monthly_at("EVI", area_idx, o_idx - lag) for lag in range(1, 13)
    ]))
    parts.append(np.column_stack([
        _complete_window_sum(
            sources.monthly_window("WFP_Price", area_idx, o_idx, 4, include_end=False)
        ),
        _complete_window_sum(
            sources.monthly_window("WFP_Price", area_idx, o_idx, 12, include_end=False)
        ),
        _complete_window_sum(
            sources.monthly_window("nightlight", area_idx, o_idx, 12, include_end=False)
        ),
    ]))

    matrix = np.concatenate(parts, axis=1)
    if matrix.shape[1] != len(REFERENCE_COLUMNS):
        raise PreflightError(
            f"Reference width {matrix.shape[1]} does not match the frozen "
            f"{len(REFERENCE_COLUMNS)}-column schema."
        )
    return matrix, REFERENCE_COLUMNS


def origin_alignment(
    sources: PreparedSources, horizon: int, rows: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (area index, target month index, origin month index) with O = T - H."""
    if horizon not in HORIZONS:
        raise ValueError(f"Horizon {horizon} is outside the approved lag schedule {HORIZONS}.")
    area_idx = sources.target_area_idx
    t_idx = sources.target_month_idx
    if rows is not None:
        area_idx = area_idx[rows]
        t_idx = t_idx[rows]
    return area_idx, t_idx, t_idx - horizon


def build_row_metadata(
    sources: PreparedSources, horizon: int, rows: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Keys, labels and exact-origin persistence for each labeled target row (D36/D63)."""
    area_idx, t_idx, o_idx = origin_alignment(sources, horizon, rows)
    label = sources.target_label if rows is None else sources.target_label[rows]
    usable = o_idx >= 0
    phase = np.full(o_idx.shape, np.nan, dtype=np.float64)
    source = np.zeros(o_idx.shape, dtype=np.int8)
    phase[usable] = sources.observed_phase[area_idx[usable], o_idx[usable]]
    source[usable] = sources.observed_source[area_idx[usable], o_idx[usable]]
    available = ~np.isnan(phase)
    persistence = np.where(available, (phase >= 3).astype(np.float64), np.nan)
    reason = np.where(
        available, "valid_observed_phase_at_origin", "no_valid_observed_phase_at_origin"
    )
    return pd.DataFrame({
        "FEWSNET_admin_code": sources.areas[area_idx],
        "target_month": [month_label(int(index)) for index in t_idx],
        "origin_month": [month_label(int(index)) for index in o_idx],
        "horizon_months": horizon,
        "target_label": label.astype(np.int8),
        "persistence_available": available,
        "persistence": persistence,
        "persistence_source": np.where(source == 1, "master", np.where(source == 2, "ledger_only", "")),
        "persistence_reason": reason,
    })


# --------------------------------------------------------------------------------------
# 10. max_plus imputation fitted on real fitting rows only (D23)
# --------------------------------------------------------------------------------------


class MaxPlusImputer:
    """Released max_plus behavior with the corrected fit scope.

    Fit statistics come only from the supplied real fitting rows; validation/prediction
    rows are transformed with the same frozen statistics. A wholly missing fitting
    column keeps the released zero fallback, which is a model-input sentinel and not
    an observed zero.
    """

    def __init__(self, multiplier: float = IMPUTER_MULTIPLIER) -> None:
        self.multiplier = float(multiplier)
        self.fill_values_: Optional[np.ndarray] = None
        self.column_stats_: List[Dict[str, object]] = []
        self.n_fitting_rows_: int = 0

    def fit(self, X: np.ndarray, fitting_mask: Optional[np.ndarray] = None,
            columns: Optional[Sequence[str]] = None) -> "MaxPlusImputer":
        matrix = np.asarray(X, dtype=np.float64)
        if fitting_mask is None:
            fitting = matrix
        else:
            fitting_mask = np.asarray(fitting_mask, dtype=bool)
            if fitting_mask.shape[0] != matrix.shape[0]:
                raise ValueError("fitting_mask length does not match X.")
            fitting = matrix[fitting_mask]
        self.n_fitting_rows_ = int(fitting.shape[0])
        if self.n_fitting_rows_ == 0:
            raise ValueError("max_plus imputation requires at least one real fitting row.")
        fill = np.empty(matrix.shape[1], dtype=np.float64)
        self.column_stats_ = []
        for position in range(matrix.shape[1]):
            column = fitting[:, position]
            observed = column[~np.isnan(column)]
            if observed.size == 0:
                fill[position] = IMPUTER_ALL_MISSING_FILL
                self.column_stats_.append({
                    "column": None if columns is None else columns[position],
                    "index": position,
                    "all_missing_in_fitting_rows": True,
                    "min": None, "max": None,
                    "fill_value": IMPUTER_ALL_MISSING_FILL,
                    "fill_rule": "all-missing fitting column -> released zero fallback sentinel",
                })
                continue
            maximum = float(observed.max())
            value = self.multiplier if maximum == 0.0 else maximum * self.multiplier
            fill[position] = value
            self.column_stats_.append({
                "column": None if columns is None else columns[position],
                "index": position,
                "all_missing_in_fitting_rows": False,
                "min": float(observed.min()),
                "max": maximum,
                "fill_value": value,
                "fill_rule": "max*100" if maximum != 0.0 else "max==0 -> 100",
            })
        self.fill_values_ = fill
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.fill_values_ is None:
            raise RuntimeError("MaxPlusImputer.transform called before fit.")
        matrix = np.array(X, dtype=np.float64, copy=True)
        if matrix.shape[1] != self.fill_values_.shape[0]:
            raise ValueError("Column count differs from the fitted schema.")
        missing = np.isnan(matrix)
        if missing.any():
            fill = np.broadcast_to(self.fill_values_, matrix.shape)
            matrix[missing] = fill[missing]
        return matrix

    def fit_transform(self, X: np.ndarray, fitting_mask: Optional[np.ndarray] = None,
                      columns: Optional[Sequence[str]] = None) -> np.ndarray:
        return self.fit(X, fitting_mask=fitting_mask, columns=columns).transform(X)

    def manifest(self) -> Dict[str, object]:
        return {
            "strategy": IMPUTER_STRATEGY,
            "multiplier": self.multiplier,
            "all_missing_fill": IMPUTER_ALL_MISSING_FILL,
            "fitting_rows": self.n_fitting_rows_,
            "columns_all_missing_in_fitting_rows": [
                stat["column"] if stat["column"] is not None else stat["index"]
                for stat in self.column_stats_ if stat["all_missing_in_fitting_rows"]
            ],
            "ordered_column_statistics": self.column_stats_,
        }


# --------------------------------------------------------------------------------------
# 11. Finite candidate/fold schedule (D17-D22, D40, D55)
# --------------------------------------------------------------------------------------


def candidate_months_for_year(year: int) -> Tuple[int, ...]:
    """Observed candidate target months: 2014/2015 Jan/Apr/Jul/Oct, 2016 onward Feb/Jun/Oct."""
    return CANDIDATE_MONTHS_EARLY if year <= 2015 else CANDIDATE_MONTHS_LATE


def role_candidate_jobs(role: str) -> List[Tuple[int, int, int]]:
    """(year, month, scope) Stage 1 candidate jobs for a map role, before support checks."""
    years = MAP_ROLES[role]["candidate_years"]
    jobs: List[Tuple[int, int, int]] = []
    for year in years:  # type: ignore[union-attr]
        for month in candidate_months_for_year(int(year)):
            for scope in sorted(FORECASTING_SCOPES):
                jobs.append((int(year), month, scope))
    return jobs


def final_target_dates(horizon: int) -> List[Tuple[int, int]]:
    start_year, start_month = FINAL_TARGET_START[horizon]
    dates: List[Tuple[int, int]] = []
    for year in range(start_year, FINAL_TARGET_END[0] + 1):
        for month in CANDIDATE_MONTHS_LATE:
            if (year, month) < (start_year, start_month):
                continue
            if (year, month) > FINAL_TARGET_END:
                continue
            dates.append((year, month))
    return sorted(dates)


def build_schedule() -> Dict[str, object]:
    """Deterministic job/fold accounting; job counts are not runtime estimates."""
    arms = [name for name, _ in RECIPE_MANIFEST] + [REFERENCE_ARM]
    calibration = role_candidate_jobs("calibration")
    selection = role_candidate_jobs("selection")
    final = role_candidate_jobs("final")
    development_unique = sorted(set(calibration) | set(selection))
    overlap = sorted(set(calibration) & set(selection))
    final_new = sorted(set(final) - set(selection))

    development_jobs = len(arms) * len(development_unique)
    final_jobs = 2 * len(final_new)
    development_folds = len(arms) * 2 * len(FORECASTING_SCOPES) * 3
    final_pairs = sum(len(final_target_dates(h)) for h in HORIZONS)
    final_folds = 2 * final_pairs

    schedule = {
        "arms": arms,
        "n_arms": len(arms),
        "map_roles": {
            role: {
                "candidate_years": list(MAP_ROLES[role]["candidate_years"]),  # type: ignore[arg-type]
                "information_cutoff": "%04d-%02d" % MAP_ROLES[role]["cutoff"],  # type: ignore[arg-type]
                "candidate_jobs_per_arm": len(role_candidate_jobs(role)),
                "prediction_targets": (
                    None if MAP_ROLES[role]["prediction_targets"] is None
                    else ["%04d-%02d" % date for date in MAP_ROLES[role]["prediction_targets"]]  # type: ignore[union-attr]
                ),
                "month_specific_maps": False,
            }
            for role in ("calibration", "selection", "final")
        },
        "stage1": {
            "candidate_jobs_calibration_window": len(calibration),
            "candidate_jobs_selection_window": len(selection),
            "candidate_jobs_final_window": len(final),
            "exact_overlap_jobs_2016": len(overlap),
            "unique_development_jobs_per_arm": len(development_unique),
            "development_jobs_upper_bound": development_jobs,
            "new_final_jobs_per_final_arm": len(final_new),
            "final_jobs_upper_bound": final_jobs,
            "total_jobs_upper_bound": development_jobs + final_jobs,
            "training_window_calendar_months": TRAIN_WINDOW_CALENDAR_MONTHS,
            "training_mask": "[O-35 calendar months, O) with origin-month labels excluded",
        },
        "stage2": {
            "development_map_builds": len(arms) * 2,
            "final_map_builds": 2,
            "total_map_builds": len(arms) * 2 + 2,
        },
        "stage3": {
            "development_folds": development_folds,
            "final_scope_date_pairs": final_pairs,
            "final_folds": final_folds,
            "final_target_dates": {
                str(horizon): ["%04d-%02d" % date for date in final_target_dates(horizon)]
                for horizon in HORIZONS
            },
            "final_target_date_counts": {
                str(horizon): len(final_target_dates(horizon)) for horizon in HORIZONS
            },
            "supplementary_common_window": [
                "%04d-%02d" % SUPPLEMENTARY_COMMON_START, "%04d-%02d" % FINAL_TARGET_END,
            ],
        },
        "note": (
            "Planning arithmetic only: these are job counts before D20 support exclusions, "
            "not measured runtime, individual forest fits or guarantees of support."
        ),
    }

    expected = {
        "candidate_jobs_calibration_window": 33,
        "candidate_jobs_selection_window": 27,
        "candidate_jobs_final_window": 27,
        "exact_overlap_jobs_2016": 9,
        "unique_development_jobs_per_arm": 51,
        "development_jobs_upper_bound": 663,
        "new_final_jobs_per_final_arm": 18,
        "final_jobs_upper_bound": 36,
        "total_jobs_upper_bound": 699,
    }
    mismatch = {
        key: (schedule["stage1"][key], value)  # type: ignore[index]
        for key, value in expected.items() if schedule["stage1"][key] != value  # type: ignore[index]
    }
    if mismatch:
        raise PreflightError(f"Schedule arithmetic does not match the approved plan: {mismatch}")
    if schedule["stage2"]["total_map_builds"] != 28:  # type: ignore[index]
        raise PreflightError("Stage 2 map-build accounting does not match the approved 26 + 2.")
    if development_folds != 234 or final_folds != 60:
        raise PreflightError(
            f"Stage 3 fold accounting ({development_folds}, {final_folds}) does not match "
            "the approved 234 development and 60 final folds."
        )
    if [len(final_target_dates(h)) for h in HORIZONS] != [11, 10, 9]:
        raise PreflightError("Final target-date counts do not match the approved 11/10/9.")
    return schedule


# --------------------------------------------------------------------------------------
# 12. Manifests
# --------------------------------------------------------------------------------------


def build_source_manifest(data_root: Path, verify_hashes: bool = True) -> Dict[str, object]:
    entries: Dict[str, object] = {}
    for key, spec in PINNED_SOURCES.items():
        path = data_root / str(spec["relpath"])
        if not path.exists():
            raise PreflightError(f"Pinned source missing: {path}")
        stat = path.stat()
        entry: Dict[str, object] = {
            "path": str(path),
            "role": spec["role"],
            "bytes": int(stat.st_size),
            "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
            "expected_bytes": spec["bytes"],
            "expected_sha256": spec["sha256"],
        }
        if spec["bytes"] is not None and int(stat.st_size) != int(spec["bytes"]):  # type: ignore[arg-type]
            raise PreflightError(
                f"{key}: byte size {stat.st_size} differs from the pinned {spec['bytes']}."
            )
        if verify_hashes:
            digest = sha256_file(path)
            entry["sha256"] = digest
            if spec["sha256"] is not None and digest != spec["sha256"]:
                raise PreflightError(
                    f"{key}: sha256 {digest} differs from the pinned {spec['sha256']}."
                )
            entry["sha256_matches_pinned"] = (
                None if spec["sha256"] is None else digest == spec["sha256"]
            )
        entries[key] = entry

    shapefile = data_root / str(PINNED_SOURCES["fews_shapefile"]["relpath"])
    sidecars: Dict[str, object] = {}
    for suffix in SHAPEFILE_SIDECAR_SUFFIXES:
        candidate = shapefile.parent / (shapefile.stem + suffix)
        if candidate.exists():
            sidecars[suffix] = {
                "bytes": int(candidate.stat().st_size),
                "sha256": sha256_file(candidate) if verify_hashes else None,
            }
    entries["fews_shapefile_sidecars"] = sidecars
    return entries


def build_schema_manifest() -> Dict[str, object]:
    superset = updated_superset_columns()
    recipes: Dict[str, object] = {}
    for position, (name, blocks) in enumerate(RECIPE_MANIFEST):
        columns = recipe_columns(name)
        if len(columns) != DECLARED_RECIPE_WIDTHS[name]:
            raise PreflightError(
                f"Recipe {name} realized width {len(columns)} differs from the approved "
                f"{DECLARED_RECIPE_WIDTHS[name]}."
            )
        if len(set(columns)) != len(columns):
            raise PreflightError(f"Recipe {name} has duplicate input columns.")
        missing = [column for column in columns if column not in superset]
        if missing:
            raise PreflightError(f"Recipe {name} has columns outside the superset: {missing[:5]}")
        recipes[name] = {
            "manifest_order": position,
            "blocks": list(blocks),
            "n_columns": len(columns),
            "superset_column_indices": [superset.index(column) for column in columns],
        }

    # D46/D47/D51: excluded source fields must not appear as inputs, and no derivative can
    # exist because every block generates columns only from the approved whitelists below.
    generating_whitelists: Dict[str, Sequence[str]] = {
        "updated_base": UPDATED_BASE_FIELDS,
        "reference_source": REFERENCE_SOURCE_FIELDS,
        "b_c_continuous": B_CONTINUOUS_FIELDS,
        "b_c_counts": B_COUNT_FIELDS,
        "block_e_expansion": UPDATED_BASE_FIELDS,
        "block_d_operands": (
            "Rainf_f_tavg_mean", "EVI", "event_count_battles", "event_count_explosions",
            "event_count_violence", "food_inflation_wb", "market_distance",
        ),
    }
    forbidden_inputs = tuple(DIRECT_INPUT_EXCLUSIONS) + (
        "ISO_encoded", "AEZ_group", "AEZ_country_group", "partition_id",
    ) + EXCLUDED_ADDITIONAL_FIELDS
    for name, whitelist in generating_whitelists.items():
        leaked = [field for field in whitelist if field in forbidden_inputs]
        if leaked:
            raise PreflightError(f"Excluded fields {leaked} appear in whitelist {name!r}.")
    for field in REFERENCE_ONLY_PRICES:
        if field in UPDATED_BASE_FIELDS or field in B_CONTINUOUS_FIELDS:
            raise PreflightError(f"Legacy price {field!r} leaked into the updated arm.")
    for recipe_name in [n for n, _ in RECIPE_MANIFEST]:
        for column in recipe_columns(recipe_name):
            if column in forbidden_inputs or column in REFERENCE_ONLY_PRICES:
                raise PreflightError(
                    f"Excluded field {column!r} is a model input in recipe {recipe_name}."
                )
    allowed_reference_history = {
        f"fews_ipc_crisis_lag_{lag}" for lag in REFERENCE_IPC_LAGS
    } | {f"fews_ipc_lag_{lag}" for lag in REFERENCE_IPC_LAGS}
    allowed_reference_calendar = (
        {f"year_{year}" for year in REFERENCE_YEARS}
        | {f"month_{month}" for month in REFERENCE_MONTHS}
    )
    for column in REFERENCE_COLUMNS:
        if column in forbidden_inputs and column not in allowed_reference_calendar:
            raise PreflightError(f"Excluded field {column!r} is a reference model input.")
    if not allowed_reference_history <= set(REFERENCE_COLUMNS):
        raise PreflightError("Reference IPC history columns do not match the approved family.")
    if any(column.startswith("fews_ha") for column in REFERENCE_COLUMNS + superset):
        raise PreflightError("An assistance-derived column leaked into a model schema.")
    if any(column.startswith(("Tair_zscore", "Rainf_zscore")) for column in REFERENCE_COLUMNS + superset):
        raise PreflightError("An inherited climate z-score leaked into a model schema.")

    accounted = set(COMMON_SOURCE_FIELDS) | set(REFERENCE_ONLY_PRICES) | set(DIRECT_INPUT_EXCLUSIONS)
    if accounted != set(MASTER_HEADER) or len(MASTER_HEADER) != 88:
        raise PreflightError("Master header fields are not accounted for exactly once.")
    if len(COMMON_SOURCE_FIELDS) != 64 or len(UPDATED_BASE_FIELDS) != 86:
        raise PreflightError("Common/updated BASE schema widths are not 64/86.")
    if len(REFERENCE_SOURCE_FIELDS) != 67 or len(REFERENCE_COLUMNS) != 109:
        raise PreflightError("Reference schema widths are not 67/109.")
    if len(ADDITIONAL_SOURCE_FIELDS) != 22 or len(BBG_FIELDS) != 18:
        raise PreflightError("Additional-source counts are not 22 fields / 18 Bloomberg series.")
    if len(BLOCK_E_COLUMNS) != 140:
        raise PreflightError("Block E schema width is not 140.")

    roles = {field: field_time_role(field) for field in UPDATED_BASE_FIELDS}
    role_counts: Dict[str, int] = {}
    for role in roles.values():
        role_counts[role] = role_counts.get(role, 0) + 1

    return {
        "master_header": list(MASTER_HEADER),
        "direct_input_exclusions": list(DIRECT_INPUT_EXCLUSIONS),
        "reference_only_prices": list(REFERENCE_ONLY_PRICES),
        "common_source_fields": list(COMMON_SOURCE_FIELDS),
        "additional_source_fields": list(ADDITIONAL_SOURCE_FIELDS),
        "excluded_additional_fields": list(EXCLUDED_ADDITIONAL_FIELDS),
        "updated_base_fields": list(UPDATED_BASE_FIELDS),
        "reference_source_fields": list(REFERENCE_SOURCE_FIELDS),
        "reference_columns": list(REFERENCE_COLUMNS),
        "updated_superset_columns": list(superset),
        "counts": {
            "master_header": len(MASTER_HEADER),
            "common_source_fields": len(COMMON_SOURCE_FIELDS),
            "additional_source_fields": len(ADDITIONAL_SOURCE_FIELDS),
            "updated_base_fields": len(UPDATED_BASE_FIELDS),
            "reference_source_fields": len(REFERENCE_SOURCE_FIELDS),
            "reference_columns": len(REFERENCE_COLUMNS),
            "block_A": len(BLOCK_A_COLUMNS),
            "block_B": len(BLOCK_B_COLUMNS),
            "block_C": len(BLOCK_C_COLUMNS),
            "block_D": len(BLOCK_D_COLUMNS),
            "block_E": len(BLOCK_E_COLUMNS),
            "updated_superset": len(superset),
        },
        "updated_base_time_roles": roles,
        "updated_base_time_role_counts": role_counts,
        "block_columns": {name: list(columns) for name, columns in BLOCK_COLUMNS.items()},
        "recipes": recipes,
        "reference_arm": REFERENCE_ARM,
        "b_c_whitelist": {
            "continuous": list(B_CONTINUOUS_FIELDS),
            "counts": list(B_COUNT_FIELDS),
            "windows": list(B_WINDOWS),
        },
        "metadata_only_identifiers": [
            "FEWSNET_admin_code", "ISO", "ISO3", "ADMIN0", "ADMIN1", "ADMIN2", "ADMIN3",
            "unit_name", "date", "month", "learned partition IDs",
        ],
    }


def compare_expectations(report: Dict[str, object]) -> Dict[str, object]:
    """Reconcile the computed preflight against research/target-label-contract.md."""
    comparison: Dict[str, object] = {}
    failures: List[str] = []
    for key, expected in EXPECTED_PREFLIGHT.items():
        if key in ("ledger_artifact_physical_line", "ledger_artifact_raw"):
            continue
        actual = report.get(key)
        if isinstance(expected, dict):
            actual_normalized = {int(k): int(v) for k, v in dict(actual or {}).items()}
            expected_normalized = {int(k): int(v) for k, v in expected.items()}
            matches = actual_normalized == expected_normalized
        else:
            matches = actual == expected
        comparison[key] = {"expected": expected, "actual": actual, "matches": matches}
        if not matches:
            failures.append(key)
    artifacts = report.get("ledger_excluded_records") or []
    artifact_ok = (
        len(artifacts) == 1
        and artifacts[0]["physical_line"] == EXPECTED_PREFLIGHT["ledger_artifact_physical_line"]
        and artifacts[0]["raw_value"] == EXPECTED_PREFLIGHT["ledger_artifact_raw"]
    )
    comparison["ledger_terminal_artifact"] = {
        "expected": {
            "physical_line": EXPECTED_PREFLIGHT["ledger_artifact_physical_line"],
            "raw_value": EXPECTED_PREFLIGHT["ledger_artifact_raw"],
        },
        "actual": artifacts,
        "matches": artifact_ok,
    }
    if not artifact_ok:
        failures.append("ledger_terminal_artifact")

    secondary: Dict[str, object] = {}
    annual = report.get("annual_fields") or {}
    for field, expected_missing in EXPECTED_ANNUAL_WHOLLY_MISSING.items():
        actual = (annual.get(field) or {}).get("wholly_missing_area_years")
        matches = actual == expected_missing
        secondary[f"{field}_wholly_missing_area_years"] = {
            "expected": expected_missing, "actual": actual, "matches": matches,
        }
        if not matches:
            failures.append(f"{field}_wholly_missing_area_years")
    pop_diff = report.get("pop_within_year_difference_groups")
    secondary["pop_within_year_difference_groups"] = {
        "expected": EXPECTED_POP_WITHIN_YEAR_DIFF_GROUPS,
        "actual": pop_diff,
        "matches": pop_diff == EXPECTED_POP_WITHIN_YEAR_DIFF_GROUPS,
    }
    if pop_diff != EXPECTED_POP_WITHIN_YEAR_DIFF_GROUPS:
        failures.append("pop_within_year_difference_groups")
    area_years = ((annual.get("CPI") or {}).get("area_years"))
    secondary["area_years"] = {
        "expected": EXPECTED_AREA_YEARS, "actual": area_years,
        "matches": area_years == EXPECTED_AREA_YEARS,
    }
    if area_years != EXPECTED_AREA_YEARS:
        failures.append("area_years")

    return {
        "target_label_contract": comparison,
        "runtime_and_preflight": secondary,
        "all_expectations_match": not failures,
        "failed_expectations": failures,
    }


# --------------------------------------------------------------------------------------
# 13. Preparation driver and run-local artifacts
# --------------------------------------------------------------------------------------

CACHE_FILENAME = "prepared_sources.npz"


def prepare_sources(
    data_root: Path, verify_hashes: bool = True, verbose: bool = True
) -> Tuple[PreparedSources, Dict[str, object]]:
    """Run the full pinned-source preflight and assemble the prepared grid."""
    manifests: Dict[str, object] = {}
    manifests["runtime"] = runtime_identity()
    manifests["sources"] = build_source_manifest(data_root, verify_hashes=verify_hashes)
    manifests["schemas"] = build_schema_manifest()
    manifests["schedule"] = build_schedule()

    master = load_master_grid(data_root / str(PINNED_SOURCES["master"]["relpath"]), verbose=verbose)
    ledger = load_ledger(
        data_root / str(PINNED_SOURCES["ledger"]["relpath"]), master.areas, verbose=verbose
    )
    cross = reconcile_master_and_ledger(master, ledger)

    preflight: Dict[str, object] = {}
    preflight.update(master.report)
    preflight.update(ledger.report)
    preflight.update(cross)
    preflight["expectations"] = compare_expectations(preflight)
    if not preflight["expectations"]["all_expectations_match"]:  # type: ignore[index]
        raise PreflightError(
            "Pinned-source preflight does not reconcile to the recorded contract: "
            f"{preflight['expectations']['failed_expectations']}"  # type: ignore[index]
        )
    manifests["preflight"] = preflight

    coverage: Dict[str, object] = {}
    enso, enso_report = load_enso_series(data_root / str(PINNED_SOURCES["enso"]["relpath"]))
    coverage["enso"] = enso_report
    bloomberg, bloomberg_report = load_bloomberg_series({
        key: data_root / str(PINNED_SOURCES[key]["relpath"])
        for key in BLOOMBERG_FILE_FIELDS
    })
    coverage["bloomberg"] = bloomberg_report

    area_lat = master.static["lat"]
    area_lon = master.static["lon"]
    markets, wb_source_report = load_wb_markets(
        data_root / str(PINNED_SOURCES["wb_derived"]["relpath"]),
        data_root / str(PINNED_SOURCES["wb_raw"]["relpath"]),
        verbose=verbose,
    )
    wb_join, wb_join_report = join_wb_to_areas(markets, area_lat, area_lon, verbose=verbose)
    coverage["wb"] = {"source": wb_source_report, "join": wb_join_report}
    del markets

    coastline, coastline_result = extract_coastline(
        data_root / str(PINNED_SOURCES["coastline"]["relpath"]), area_lat, area_lon
    )
    coverage["coastline"] = coastline_result["report"]
    manifests["coverage"] = coverage

    sources = assemble_prepared_sources(
        master=master,
        ledger=ledger,
        enso=enso,
        bloomberg=bloomberg,
        wb_values=wb_join["values"],
        coastline=coastline,
        extra_report={"coverage": coverage},
    )
    manifests["provenance_arrays"] = {
        "wb_match_distance_km": "cache/prepared_sources.npz::wb_match_distance_km",
        "wb_match_market_index": "cache/prepared_sources.npz::wb_match_market_index",
        "wb_market_ids": "manifests/wb_market_ids.json",
        "coastline_pixels": "manifests/coastline_pixels.csv",
    }
    manifests["_wb_provenance"] = wb_join["provenance"]
    manifests["_coastline_provenance"] = coastline_result["provenance"]
    return sources, manifests


def save_prepared_sources(sources: PreparedSources, path: Path, provenance: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {
        "areas": sources.areas,
        "observed_phase": sources.observed_phase,
        "observed_crisis": sources.observed_crisis,
        "observed_source": sources.observed_source,
        "pop_selected_month": sources.pop_selected_month,
        "target_area_idx": sources.target_area_idx,
        "target_month_idx": sources.target_month_idx,
        "target_label": sources.target_label,
    }
    for field, grid in sources.area_monthly.items():
        payload[f"area_monthly::{field}"] = grid
    for field, series in sources.global_monthly.items():
        payload[f"global_monthly::{field}"] = series
    for field, values in sources.static.items():
        payload[f"static::{field}"] = values
    for field, values in sources.annual.items():
        payload[f"annual::{field}"] = values
    payload["wb_match_distance_km"] = provenance["match_distance_km"]
    payload["wb_match_market_index"] = provenance["match_market_index"]
    np.savez(path, **payload)


def load_prepared_sources(path: Path) -> PreparedSources:
    with np.load(path, allow_pickle=False) as handle:
        area_monthly = {
            key.split("::", 1)[1]: handle[key] for key in handle.files
            if key.startswith("area_monthly::")
        }
        global_monthly = {
            key.split("::", 1)[1]: handle[key] for key in handle.files
            if key.startswith("global_monthly::")
        }
        static = {
            key.split("::", 1)[1]: handle[key] for key in handle.files if key.startswith("static::")
        }
        annual = {
            key.split("::", 1)[1]: handle[key] for key in handle.files if key.startswith("annual::")
        }
        return PreparedSources(
            areas=handle["areas"],
            area_monthly=area_monthly,
            global_monthly=global_monthly,
            static=static,
            annual=annual,
            pop_selected_month=handle["pop_selected_month"],
            observed_phase=handle["observed_phase"],
            observed_crisis=handle["observed_crisis"],
            observed_source=handle["observed_source"],
            target_area_idx=handle["target_area_idx"],
            target_month_idx=handle["target_month_idx"],
            target_label=handle["target_label"],
            report={"loaded_from_cache": str(path)},
        )


def missingness_profile(matrix: np.ndarray, columns: Sequence[str]) -> List[Dict[str, object]]:
    """Descriptive pre-imputation missingness per column; not an imputation fit."""
    missing = np.isnan(matrix).sum(axis=0)
    total = matrix.shape[0]
    return [
        {
            "column": columns[position],
            "missing_rows": int(missing[position]),
            "missing_fraction": float(missing[position] / total) if total else None,
        }
        for position in range(matrix.shape[1])
    ]


def _persist_arm(
    arm: str,
    horizon: int,
    matrix: np.ndarray,
    columns: Sequence[str],
    run_dir: Path,
    seconds: float,
    write_matrix: bool,
    verbose: bool,
) -> Dict[str, object]:
    target = run_dir / "features" / f"h{horizon:02d}"
    target.mkdir(parents=True, exist_ok=True)
    if write_matrix:
        np.save(target / f"{arm}.npy", matrix)
    write_json(target / f"{arm}_columns.json", {
        "arm": arm,
        "horizon_months": horizon,
        "n_rows": int(matrix.shape[0]),
        "n_columns": int(matrix.shape[1]),
        "columns": list(columns),
    })
    write_json(target / f"{arm}_missingness.json", {
        "arm": arm,
        "horizon_months": horizon,
        "n_rows": int(matrix.shape[0]),
        "note": "descriptive pre-imputation missingness; imputation is fitted per fold",
        "columns": missingness_profile(matrix, columns),
    })
    summary = {
        "arm": arm,
        "horizon_months": horizon,
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "build_seconds": round(seconds, 2),
        "matrix_megabytes": round(matrix.nbytes / (1 << 20), 1),
        "written": bool(write_matrix),
        "all_missing_columns": int(np.sum(np.isnan(matrix).all(axis=0))),
        "mean_missing_fraction": float(np.isnan(matrix).mean()),
    }
    if verbose:
        print(
            f"[features] {arm} h{horizon}: {summary['rows']:,} rows x {summary['columns']} cols "
            f"in {summary['build_seconds']}s ({summary['matrix_megabytes']} MB, "
            f"missing {summary['mean_missing_fraction']:.3f})",
            flush=True,
        )
    return summary


def materialize_horizon(
    sources: PreparedSources,
    arms: Sequence[str],
    horizon: int,
    run_dir: Path,
    write_matrix: bool = True,
    verbose: bool = True,
) -> List[Dict[str, object]]:
    """Materialize the requested arms for one horizon.

    The updated superset is built once per horizon; every recipe is an ordered column
    subset of it, so recipes never diverge from each other or from Stage 1/Stage 3.
    """
    summaries: List[Dict[str, object]] = []
    if REFERENCE_ARM in arms:
        started = time.time()
        matrix, columns = build_reference_matrix(sources, horizon)
        summaries.append(_persist_arm(
            REFERENCE_ARM, horizon, matrix, columns, run_dir,
            time.time() - started, write_matrix, verbose,
        ))
        del matrix
    recipes = [arm for arm in arms if arm != REFERENCE_ARM]
    if recipes:
        started = time.time()
        superset, superset_columns = build_updated_superset(sources, horizon)
        superset_seconds = time.time() - started
        if verbose:
            print(
                f"[features] updated superset h{horizon}: {superset.shape[0]:,} rows x "
                f"{superset.shape[1]} cols in {superset_seconds:.1f}s",
                flush=True,
            )
        index_of = {column: position for position, column in enumerate(superset_columns)}
        for arm in recipes:
            started = time.time()
            columns = recipe_columns(arm)
            matrix = superset[:, [index_of[column] for column in columns]]
            summary = _persist_arm(
                arm, horizon, matrix, columns, run_dir,
                time.time() - started, write_matrix, verbose,
            )
            summary["superset_build_seconds"] = round(superset_seconds, 2)
            summaries.append(summary)
            del matrix
        del superset
    return summaries


def write_row_metadata(sources: PreparedSources, horizon: int, run_dir: Path) -> Dict[str, object]:
    frame = build_row_metadata(sources, horizon)
    target = run_dir / "features" / f"h{horizon:02d}"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / "rows.csv", index=False)
    available = frame["persistence_available"].to_numpy()
    by_year: Dict[str, Dict[str, int]] = {}
    years = frame["target_month"].str.slice(0, 4)
    for year in sorted(years.unique()):
        mask = (years == year).to_numpy()
        by_year[year] = {
            "labeled_rows": int(mask.sum()),
            "with_valid_origin_persistence": int(np.sum(mask & available)),
        }
    return {
        "horizon_months": horizon,
        "labeled_rows": int(len(frame)),
        "with_valid_origin_persistence": int(available.sum()),
        "without_valid_origin_persistence": int((~available).sum()),
        "persistence_source_counts": {
            key: int(value) for key, value in
            frame.loc[available, "persistence_source"].value_counts().to_dict().items()
        },
        "by_target_year": by_year,
    }


# --------------------------------------------------------------------------------------
# 14. CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FEWS NET clean persistence experiment data preparation (preflight + features)."
    )
    parser.add_argument("--run-dir", required=True, help="Fresh experiment-local run root.")
    parser.add_argument("--data-root", default=None, help="Override the pinned source root.")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Validate sources, write manifests and cache the prepared grid.")
    parser.add_argument("--arms", default="",
                        help="Comma-separated arms to materialize (recipe names and/or 'reference').")
    parser.add_argument("--horizons", default="4,8,12",
                        help="Comma-separated horizons in months (subset of 4,8,12).")
    parser.add_argument("--no-write-matrices", action="store_true",
                        help="Build and validate matrices without persisting the .npy payloads.")
    parser.add_argument("--reuse-cache", action="store_true",
                        help="Reuse this run root's prepared-source cache instead of re-reading sources.")
    parser.add_argument("--skip-hash-verification", action="store_true",
                        help="Skip source hashing (diagnostics only; not valid for formal runs).")
    parser.add_argument("--force", action="store_true",
                        help="Allow re-running a stage that already completed in this run root.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _load_run_manifest(run_dir: Path) -> Optional[Dict[str, object]]:
    path = run_dir / "manifests" / "run_manifest.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    verbose = not args.quiet
    run_dir = Path(args.run_dir)
    data_root = resolve_data_root(args.data_root)
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    horizons = [int(item) for item in args.horizons.split(",") if item.strip()]
    for horizon in horizons:
        if horizon not in HORIZONS:
            raise SystemExit(f"Horizon {horizon} is outside the approved schedule {HORIZONS}.")
    known_arms = {name for name, _ in RECIPE_MANIFEST} | {REFERENCE_ARM}
    unknown = [arm for arm in arms if arm not in known_arms]
    if unknown:
        raise SystemExit(f"Unknown arms {unknown}; known arms are {sorted(known_arms)}.")
    stage = "preflight" if args.preflight_only or not arms else "materialize"

    existing = _load_run_manifest(run_dir)
    if existing is not None:
        completed = existing.get("completed_stages", [])
        if stage in completed and not args.force:
            raise SystemExit(
                f"Run root {run_dir} already completed stage {stage!r}; use a fresh run root "
                "(or --force only for a recorded repair)."
            )

    cache_path = run_dir / "cache" / CACHE_FILENAME
    manifests: Dict[str, object] = {}
    if args.reuse_cache and cache_path.exists():
        if verbose:
            print(f"[cache] reusing {cache_path}", flush=True)
        sources = load_prepared_sources(cache_path)
        manifests = {
            "runtime": runtime_identity(),
            "schemas": build_schema_manifest(),
            "schedule": build_schedule(),
        }
    else:
        sources, manifests = prepare_sources(
            data_root, verify_hashes=not args.skip_hash_verification, verbose=verbose
        )
        wb_provenance = manifests.pop("_wb_provenance")
        coastline_provenance = manifests.pop("_coastline_provenance")
        save_prepared_sources(sources, cache_path, wb_provenance)
        write_json(run_dir / "manifests" / "wb_market_ids.json", {
            "note": "index positions of wb_match_market_index in cache/prepared_sources.npz",
            "market_ids": list(wb_provenance["market_ids"]),
        })
        pixels = pd.DataFrame({
            "FEWSNET_admin_code": sources.areas,
            "lat": sources.static["lat"],
            "lon": sources.static["lon"],
            "pixel_row": coastline_provenance["pixel_row"],
            "pixel_col": coastline_provenance["pixel_col"],
            "coastline_dist": sources.static[COASTLINE_FIELD],
            "validity_reason": coastline_provenance["reason"],
        })
        (run_dir / "manifests").mkdir(parents=True, exist_ok=True)
        pixels.to_csv(run_dir / "manifests" / "coastline_pixels.csv", index=False)
        for name in ("runtime", "sources", "schemas", "schedule", "preflight", "coverage"):
            if name in manifests:
                write_json(run_dir / "manifests" / f"{name}.json", manifests[name])

    feature_summary: List[Dict[str, object]] = []
    row_summary: List[Dict[str, object]] = []
    if arms:
        for horizon in horizons:
            row_summary.append(write_row_metadata(sources, horizon, run_dir))
            feature_summary.extend(
                materialize_horizon(
                    sources, arms, horizon, run_dir,
                    write_matrix=not args.no_write_matrices, verbose=verbose,
                )
            )
        write_json(run_dir / "manifests" / "features.json", {
            "arms": arms,
            "horizons": horizons,
            "matrices_written": not args.no_write_matrices,
            "row_support": row_summary,
            "matrices": feature_summary,
        })

    completed = list(dict.fromkeys((existing or {}).get("completed_stages", []) + [stage]))
    if "sources" in manifests:
        source_hashes = {
            key: value.get("sha256") if isinstance(value, dict) else None
            for key, value in manifests["sources"].items()
        }
    else:
        # Cache-reuse stages inherit the hashes recorded when the cache was built.
        source_hashes = (existing or {}).get("source_hashes", {})
        if not source_hashes:
            raise SystemExit(
                "Cache reuse requires an existing run manifest with recorded source hashes."
            )
    write_json(run_dir / "manifests" / "run_manifest.json", {
        "experiment": "FEWSNETCleanPersistenceExperiment",
        "component": "prepare_data",
        "run_dir": str(run_dir),
        "data_root": str(data_root),
        "completed_stages": completed,
        "last_stage": stage,
        "last_stage_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arms": arms,
        "horizons": horizons,
        "hash_verification": not args.skip_hash_verification,
        "reused_prepared_source_cache": bool(args.reuse_cache and cache_path.exists()),
        "runtime": manifests.get("runtime", runtime_identity()),
        "source_hashes": source_hashes,
    })
    if verbose:
        print(f"[done] stage={stage} run_dir={run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

