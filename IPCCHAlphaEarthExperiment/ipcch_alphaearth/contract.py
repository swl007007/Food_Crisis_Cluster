"""Load and validate the frozen configuration (experiment contract, inputs, runtime lock).

``validate_frozen`` restates the accepted D1-D29 constants; a production run refuses any contract that
differs. Tests construct scaled-down copies explicitly and never pass them through ``load_contract``.
"""

from __future__ import annotations

import copy
from fractions import Fraction
from pathlib import Path

from ipcch_alphaearth import CONFIG_DIR
from ipcch_alphaearth.artifacts import read_json, sha256_file
from ipcch_alphaearth.errors import ContractError

CONTRACT_FILE = "experiment-contract.json"
INPUTS_FILE = "inputs.json"
RUNTIME_FILE = "runtime-lock.json"

FROZEN = {
    ("calendar", "horizons_months"): [1, 3, 6, 12],
    ("calendar", "first_origin"): "2018-01",
    ("calendar", "development_target_years"): [2021, 2022],
    ("calendar", "map_freeze_cutoff"): "2022-12",
    ("calendar", "main_first_origin"): "2023-01",
    ("calendar", "main_last_target"): "2025-12",
    ("calendar", "main_first_target"): {"1": "2023-02", "3": "2023-04", "6": "2023-07", "12": "2024-01"},
    ("calendar", "main_fold_total"): 122,
    ("calendar", "main_expected_keys"): {"1": 17322, "3": 16919, "6": 16413, "12": 14087},
    ("calendar", "supplement_target_months"): ["2026-01", "2026-02", "2026-03", "2026-04"],
    ("calendar", "supplement_expected_keys_per_h"): 4092,
    ("calendar", "half_life_months"): 24,
    ("calendar", "bridge_slots_total"): 22,
    ("calendar", "fit_ceiling"): {"B1_quartets": 38, "B0_quartets": 23, "quartets": 61, "scalar_fits": 244},
    ("targets", "phase_threshold"): "0.20",
    ("target_qc", "expected_valid_rows"): 42695,
    ("target_qc", "expected_valid_admins"): 6224,
    ("features", "compact296_list_sha256"): "e97388f848e7095bf0535f0cfee1a2d798e0c1863dee843462e60cf6a36f9909",
    ("features", "alphaearth_bands"): 64,
    ("correction", "core_inputs"): ["prcp_anom_month_ensmean__ma3_at_origin", "tmean_anom_month_ensmean__ma3_at_origin",
                                    "event_count_violence__ma3_at_origin"],
    ("correction", "shared_lambda"): 0.01,
    ("correction", "regional_lambdas"): [0.01, 0.1, 1.0],
    ("correction", "leaf_budgets"): [2, 4, 8],
    ("correction", "max_depth"): 3,
    ("correction", "cuts_per_descriptor"): 9,
    ("correction", "gain_tolerance_relative"): 1e-12,
    ("correction", "support"): {"n_eff": 200, "admins": 50, "target_months": 6},
    ("alphaearth", "collection"): "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
    ("alphaearth", "year"): 2017,
    ("alphaearth", "coverage_threshold"): "0.80",
    ("alphaearth", "best_effort"): False,
    ("alphaearth", "analysis_grid"): {
        "amendment": "D30", "crs": "EPSG:6933", "affine": [10, 0, 0, 0, -10, 0], "resampling": "nearest",
        "priority": "ascending frozen asset id; first candidate whose 64 bands are all unmasked and finite supplies the whole 64-band vector",
        "weights": "values, common validity, pixelArea and EE fractional admin-boundary weights on the same analysis grid",
        "windows": "grid-aligned only (cuts on lattice lines); each cell contributes to at most one window",
        "estimand": "area moments of native vectors nearest-sampled onto the fixed equal-area 10 m grid, not source-native UTM integration"},
    ("evaluation", "main_f1_gain_minimum"): "0.005",
    ("evaluation", "classification_guard"): "0.01",
    ("evaluation", "projected_r2_guard"): "0.02",
}
FROZEN_BOOTSTRAP = {"unit": "country", "draws": 20000, "seed": 42, "generator": "numpy.random.default_rng",
                    "lower_percentile": 1.25, "percentile_method": "linear", "family_size": 4, "min_countries": 2}
FROZEN_FIXED_XGB = {"objective": "reg:squarederror", "tree_method": "hist", "device": "cpu", "booster": "gbtree",
                    "seed": 42, "n_jobs": 16, "scale_pos_weight": 1.0}
FROZEN_RECIPE_COMMON = {"max_depth": 11, "n_estimators": 200, "learning_rate": 0.1, "subsample": 1.0, "colsample_bytree": 0.5,
                        "gamma": 0.0, "min_child_weight": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0}
FROZEN_RECIPE_Q3 = {"max_depth": 9, "n_estimators": 200, "learning_rate": 0.1, "subsample": 0.5, "colsample_bytree": 0.7,
                    "gamma": 0.1, "min_child_weight": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0}


def config_path(name: str, config_dir: Path | None = None) -> Path:
    return Path(config_dir or CONFIG_DIR) / name


def load_contract(config_dir: Path | None = None) -> dict:
    contract = read_json(config_path(CONTRACT_FILE, config_dir))
    validate_frozen(contract)
    return contract


def load_inputs(config_dir: Path | None = None) -> dict:
    return read_json(config_path(INPUTS_FILE, config_dir))


def load_runtime_lock(config_dir: Path | None = None) -> dict:
    return read_json(config_path(RUNTIME_FILE, config_dir))


def config_identity(config_dir: Path | None = None) -> dict:
    return {name: sha256_file(config_path(name, config_dir)) for name in (CONTRACT_FILE, INPUTS_FILE, RUNTIME_FILE)}


def validate_frozen(contract: dict) -> None:
    bad = []
    for (section, key), want in FROZEN.items():
        got = contract.get(section, {}).get(key)
        if got != want:
            bad.append(f"{section}.{key}: {got!r} != frozen {want!r}")
    if contract.get("evaluation", {}).get("bootstrap") != FROZEN_BOOTSTRAP:
        bad.append("evaluation.bootstrap differs")
    backbone = contract.get("backbone", {})
    if backbone.get("fixed") != FROZEN_FIXED_XGB:
        bad.append("backbone.fixed differs")
    recipes = backbone.get("recipes", {})
    for q in ("q2", "q4", "q5"):
        if recipes.get(q) != FROZEN_RECIPE_COMMON:
            bad.append(f"backbone.recipes.{q} differs")
    if recipes.get("q3") != FROZEN_RECIPE_Q3:
        bad.append("backbone.recipes.q3 differs")
    if backbone.get("early_stopping") is not False:
        bad.append("backbone.early_stopping must be false")
    if bad:
        raise ContractError("experiment contract differs from the frozen D1-D29 constants: " + "; ".join(bad))


def scaled_copy(contract: dict, **overrides) -> dict:
    """Test helper: deep copy with dotted-path overrides (never validated as frozen)."""
    out = copy.deepcopy(contract)
    for dotted, value in overrides.items():
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    out["_scaled_test_contract"] = True
    return out


def fraction(text: str) -> Fraction:
    return Fraction(text)
