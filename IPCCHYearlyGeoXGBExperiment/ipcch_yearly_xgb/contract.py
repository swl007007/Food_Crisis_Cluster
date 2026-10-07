"""Load and validate the frozen yearly configuration (config/*.json; design v1.0)."""

from __future__ import annotations

import json
import platform
from fractions import Fraction
from pathlib import Path

from ipcch_yearly_xgb import CONFIG_DIR
from ipcch_yearly_xgb.errors import ContractError

TARGETS = ("q2", "q3", "q4", "q5")
HORIZONS = (1, 3, 6, 12)
RECIPES = {"1": "G1L2", "3": "G3L2", "6": "G4L2", "12": "G2L2"}
GLOBAL_RECIPES = {"G1": {"max_depth": 3, "rounds": 200}, "G2": {"max_depth": 3, "rounds": 400},
                  "G3": {"max_depth": 4, "rounds": 200}, "G4": {"max_depth": 4, "rounds": 400}}
LOCAL_RECIPES = {"L1": {"max_depth": 1, "appended_rounds": 20}, "L2": {"max_depth": 2, "appended_rounds": 40}}


def _load(name: str, config_dir: Path | str = CONFIG_DIR) -> dict:
    path = Path(config_dir) / name
    if not path.is_file():
        raise ContractError(f"configuration file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def load_contract(config_dir: Path | str = CONFIG_DIR) -> dict:
    c = _load("yearly-contract.json", config_dir)
    m, p, s = c["model"], c["protocol"], c["support"]
    _require(tuple(c["targets"]["regressors"]) == TARGETS, "targets must be q2..q5")
    _require(c["targets"]["phase_threshold"] == "0.20" and c["targets"]["threshold_rule"] == ">=", "decoder >=0.20")
    _require(tuple(c["horizons_months"]) == HORIZONS, "horizons 1/3/6/12")
    _require(c["recipes"] == RECIPES, "frozen P6 recipes per H drifted")
    _require(m["global_recipes"] == GLOBAL_RECIPES and m["local_recipes"] == LOCAL_RECIPES, "recipe table drifted")
    _require(m["objective"] == "reg:squarederror" and m["early_stopping"] is False and m["global_base_score"] == 0.5,
             "objective/early stopping/base score drifted")
    _require(m["fixed"]["seed"] == 42 and m["fixed"]["nthread"] == 4 and m["fixed"]["eta"] == 0.05
             and m["fixed"]["tree_method"] == "hist" and m["fixed"]["device"] == "cpu", "fixed XGB params drifted")
    _require(m["global_common"] == {"min_child_weight": 10, "reg_lambda": 10, "reg_alpha": 0, "subsample": 0.8,
                                    "colsample_bytree": 0.8}, "global common params drifted")
    _require(m["local_common"] == {"min_child_weight": 20, "reg_lambda": 20, "reg_alpha": 1, "subsample": 1.0,
                                   "colsample_bytree": 1.0}, "local common params drifted")
    _require(p["half_life_months"] == 24 and p["historical_gate_max_dates"] == 6, "half-life/gate dates drifted")
    _require(p["first_main_target"] == {"1": "2023-02", "3": "2023-04", "6": "2023-07", "12": "2024-01"},
             "first main targets drifted")
    _require(s["local_fit"] == {"keys": 500, "areas": 50, "target_months": 6}, "local fit floors")
    _require(s["validation"] == {"keys": 100, "areas": 20, "target_months": 3, "crisis_keys": 20,
                                 "noncrisis_keys": 20}, "validation floors")
    _require(s["min_successful_local_dates"] == 3 and Fraction(s["gain_strictly_greater_than"]) == Fraction(1, 100),
             "gate rule drifted")
    _require(c["bootstrap"]["draws"] == 2000 and c["bootstrap"]["seed"] == 42, "bootstrap 2000/seed42")
    _require(c["expected"]["total_scalar_fits"] == 724, "fit inventory drifted")
    return c


def load_inputs(config_dir: Path | str = CONFIG_DIR) -> dict:
    d = _load("inputs.json", config_dir)
    _require(d["source_run"] == "p6-formal-20261004b", "authority is p6-formal-20261004b")
    _require(len(d["run_files"]) == 33 and len(d["source_config_files"]) == 3, "input inventory size drifted")
    return d


def repo_root(inputs: dict) -> Path:
    return Path(inputs["repo_root_windows"] if platform.system() == "Windows" else inputs["repo_root_wsl"])


def run_root(inputs: dict) -> Path:
    return repo_root(inputs) / inputs["run_relative"]


def load_runtime_lock(config_dir: Path | str = CONFIG_DIR) -> dict:
    lock = _load("runtime-lock.json", config_dir)
    _require(lock["packages"]["xgboost"] == "3.0.0" and lock["python"] == "3.12.10", "runtime lock drifted")
    return lock


def validate_all(config_dir: Path | str = CONFIG_DIR) -> dict:
    return {"contract": load_contract(config_dir)["contract_version"],
            "inputs": load_inputs(config_dir)["inputs_version"],
            "runtime_lock": load_runtime_lock(config_dir)["lock_version"]}
