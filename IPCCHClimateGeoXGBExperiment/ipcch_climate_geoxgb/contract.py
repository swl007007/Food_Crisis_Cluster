"""Load and validate the package's frozen configuration files.

The JSON files under ``config/`` restate the accepted contract
(prd.md R1-R51, design.md v1.0). Validation here checks internal consistency
and the accepted values; it does not open any freedom to change them.
"""

from __future__ import annotations

import json
import os
import re
from fractions import Fraction
from pathlib import Path

from ipcch_climate_geoxgb import CONFIG_DIR
from ipcch_climate_geoxgb.artifacts import sha256_bytes
from ipcch_climate_geoxgb.errors import ContractError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

HORIZONS = (1, 3, 6, 12)
TARGETS = ("q2", "q3", "q4", "q5")
GLOBAL_RECIPES = {
    "G1": {"max_depth": 3, "rounds": 200},
    "G2": {"max_depth": 3, "rounds": 400},
    "G3": {"max_depth": 4, "rounds": 200},
    "G4": {"max_depth": 4, "rounds": 400},
}
LOCAL_RECIPES = {
    "L1": {"max_depth": 1, "appended_rounds": 20},
    "L2": {"max_depth": 2, "appended_rounds": 40},
}

REQUIRED_INPUTS = (
    "raw_panel",
    "country_lookup_source",
    "reference_coordinates_source",
    "geometry_shp",
    "geometry_shx",
    "geometry_dbf",
    "geometry_prj",
    "geometry_cpg",
    "adjacency_cache",
    "saved_reference_coordinates",
    "saved_country_lookup",
    "geometry_repair_audit",
    "geography_audit",
    "climate_monthly",
    "climate_growing_season",
)
GEOMETRY_COMPONENTS = ("geometry_shp", "geometry_shx", "geometry_dbf", "geometry_prj", "geometry_cpg")


def _load(name: str, config_dir: Path | str = CONFIG_DIR) -> dict:
    path = Path(config_dir) / name
    if not path.is_file():
        raise ContractError(f"configuration file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def load_experiment_contract(config_dir: Path | str = CONFIG_DIR) -> dict:
    """Experiment contract with the accepted R17/R19/R43/R45/R47 values enforced."""
    c = _load("experiment-contract.json", config_dir)
    cal, model, part = c["calendar"], c["model"], c["partition"]
    _require(tuple(c["targets"]["regressors"]) == TARGETS, "regressors must be q2..q5")
    _require(
        c["targets"]["phase_threshold"] == "0.20" and c["targets"]["threshold_rule"] == ">=",
        "decoder must be >=0.20",
    )
    _require(tuple(cal["horizons_months"]) == HORIZONS, "horizons must be 1/3/6/12")
    counts = {int(k): v for k, v in cal["main_fold_counts"].items()}
    _require(counts == {1: 35, 3: 33, 6: 30, 12: 24}, f"main fold counts drifted: {counts}")
    _require(sum(counts.values()) == cal["main_fold_total"] == 122, "main folds must total 122")
    for h, first in cal["main_first_target_month"].items():
        year, month = map(int, first.split("-"))
        span = (2025 - year) * 12 + (12 - month) + 1
        _require(span == counts[int(h)], f"H{h}: first month {first} gives {span} folds")
    _require(cal["rolling_window_calendar_months"] == 36, "rolling window must be 36 months")
    _require(cal["historical_gate_max_dates"] == 6, "historical gate uses up to 6 dates")
    _require(model["objective"] == "reg:squarederror", "objective must be reg:squarederror")
    _require(model["global_recipes"] == GLOBAL_RECIPES, "global recipe table drifted")
    _require(model["local_recipes"] == LOCAL_RECIPES, "local recipe table drifted")
    _require(model["global_base_score"] == 0.5 and model["early_stopping"] is False, "fixed-round base_score .5")
    _require(model["fixed"]["seed"] == 42 and model["fixed"]["eta"] == 0.05, "seed42/eta.05")
    _require(part["max_member_depth"] == 4 and part["max_terminal_regions"] == 16, "R45 depth budget")
    _require(part["scan_iterations"] == 1000 and part["smoothing_rounds"] == 3, "R45/R38 iterations")
    _require(
        [Fraction(v) for v in part["candidate_size_fraction"]] == [Fraction(9, 20), Fraction(11, 20)],
        "R46 size fractions",
    )
    _require(part["stage2_consensus"] is False and part["donor_assignment"] is False, "no Stage2/donor")
    sup = c["support"]
    _require(sup["local_fit"] == {"keys": 500, "areas": 50, "target_months": 6}, "R28 fit floors")
    _require(
        sup["validation"]
        == {"keys": 100, "areas": 20, "target_months": 3, "crisis_keys": 20, "noncrisis_keys": 20},
        "R28/R29 validation floors",
    )
    boot = c["reporting"]["bootstrap"]
    _require(boot["draws"] == 2000 and boot["seed"] == 42, "R49 bootstrap 2000/seed42")
    return c


def load_feature_schema(config_dir: Path | str = CONFIG_DIR) -> dict:
    """Ordered rich601 schema; names/order are checked, values are P1 work."""
    s = _load("feature-schema.json", config_dir)
    original = list(s["original_features"])
    additional = [name for block in s["additional_blocks"].values() for name in block]
    ordered = list(s["ordered_names"])
    _require(len(original) == s["original_count"] == 133, "base133 (original103 + growing-season30) count")
    _require(len(additional) == s["additional_count"] == 468, "history468 count")
    _require(ordered == original + additional, "ordered_names != base133 + blocks in order")
    _require(len(ordered) == s["rich_count"] == 601 and len(set(ordered)) == 601, "601 unique names")
    _require(
        sha256_bytes("\n".join(ordered).encode("utf-8")) == s["ordered_names_sha256"],
        "ordered_names_sha256 does not match the names",
    )
    _require(s["crisis_threshold"] == "0.20" and s["threshold_rule"] == ">=", "schema semantics must be >=0.20")
    _require(not str(s["schema_version"]).startswith("final_review"), "old approval status text carried over")
    _require(set(s["aliases"].values()) <= set(original), "alias targets must be original93 columns")
    return s


def load_inputs(config_dir: Path | str = CONFIG_DIR) -> dict:
    d = _load("inputs.json", config_dir)
    inputs = d["inputs"]
    _require(tuple(inputs) == REQUIRED_INPUTS, f"input names drifted: {tuple(inputs)}")
    for name, entry in inputs.items():
        _require(bool(_SHA256.match(entry["sha256"])), f"{name}: bad sha256")
        _require(isinstance(entry["bytes"], int) and entry["bytes"] > 0, f"{name}: bad byte length")
        _require(bool(entry["path_windows"]) and bool(entry["path_wsl"]), f"{name}: both paths required")
    return d


def load_runtime_lock(config_dir: Path | str = CONFIG_DIR) -> dict:
    lock = _load("runtime-lock.json", config_dir)
    _require(lock["packages"].get("xgboost") == "3.0.0", "XGBoost must be pinned to 3.0.0")
    _require(lock["geometry_io_engine"] == "pyogrio" and "pyogrio" in lock["packages"], "pin pyogrio")
    return lock


def input_path(entry: dict) -> Path:
    """Platform path of one pinned input (Windows runtime uses ``path_windows``)."""
    return Path(entry["path_windows"] if os.name == "nt" else entry["path_wsl"])


def validate_all(config_dir: Path | str = CONFIG_DIR) -> dict:
    return {
        "experiment_contract": load_experiment_contract(config_dir)["contract_version"],
        "feature_schema": load_feature_schema(config_dir)["schema_version"],
        "inputs": load_inputs(config_dir)["inputs_version"],
        "runtime_lock": load_runtime_lock(config_dir)["lock_version"],
    }
