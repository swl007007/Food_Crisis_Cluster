"""Load and validate the frozen configuration (``config/*.json``).

Validation checks the approved values (design v0.2); it opens no freedom to change them.
"""

from __future__ import annotations

import json
import platform
from fractions import Fraction
from pathlib import Path

from ipcch_mlp import CONFIG_DIR
from ipcch_mlp.errors import ContractError

TARGETS = ("q2", "q3", "q4", "q5")
HORIZONS = (1, 3, 6, 12)
GLOBAL_CANDIDATES = {"G1": [64, 32], "G2": [128, 64]}
RESIDUAL_CANDIDATES = {"R1": [16], "R2": [32]}


def _load(name: str, config_dir: Path | str = CONFIG_DIR) -> dict:
    path = Path(config_dir) / name
    if not path.is_file():
        raise ContractError(f"configuration file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def load_contract(config_dir: Path | str = CONFIG_DIR) -> dict:
    c = _load("experiment-mlp.json", config_dir)
    a, t, sup = c["architecture"], c["training"], c["support"]
    _require(tuple(c["targets"]) == TARGETS, "targets must be q2..q5")
    _require(tuple(c["horizons_months"]) == HORIZONS, "horizons must be 1/3/6/12")
    _require(c["phase_threshold"] == "0.20", "decoder threshold must be 0.20")
    _require(c["calendar"]["rolling_window_calendar_months"] == 36, "36-month window")
    _require(c["calendar"]["historical_gate_max_dates"] == 6, "six historical gate dates")
    _require(c["replicates"] == [42, 43, 44], "replicates 42/43/44")
    _require(a["global_candidates"] == GLOBAL_CANDIDATES and a["residual_candidates"] == RESIDUAL_CANDIDATES,
             "architecture grid drifted")
    _require(a["n_features"] == 561 and a["n_inputs"] == 1122 and a["dropout"] == 0.10, "inputs/dropout drifted")
    _require(a["residual_output_init"] == "zero" and a["global_init"] == "pytorch-default", "initialization drifted")
    _require((t["global_epochs"], t["residual_epochs"], t["batch_size"], t["inference_batch_size"]) == (100, 40, 256, 256),
             "epochs/batch drifted")
    _require((t["lr"], t["betas"], t["eps"], t["weight_decay"]) == (0.001, [0.9, 0.999], 1e-8, 0.01), "AdamW drifted")
    _require((t["amsgrad"], t["foreach"], t["fused"]) == (False, False, False), "optimizer flags drifted")
    _require(sup["local_fit"] == {"keys": 500, "areas": 50, "target_months": 6}, "local fit floors")
    _require(sup["validation"] == {"keys": 100, "areas": 20, "target_months": 3, "crisis_keys": 20, "noncrisis_keys": 20},
             "validation floors")
    _require(sup["stage3_min_successful_local_dates"] == 3, "three successful local dates")
    _require(Fraction(sup["stage3_gain_strictly_greater_than"]) == Fraction(1, 100), "gain threshold 1/100")
    _require(c["bootstrap"]["draws"] == 2000 and c["bootstrap"]["seed"] == 42, "bootstrap 2000/seed42")
    _require(c["expected"]["total_scalar_fits"] == 13260, "fit inventory drifted")
    return c


def load_inputs(config_dir: Path | str = CONFIG_DIR) -> dict:
    d = _load("inputs.json", config_dir)
    _require(d["source_run"] == "p6-formal-20261004b", "authority is p6-formal-20261004b")
    _require(len(d["files"]) == 28, "source inventory must list 28 files")
    return d


def source_root(inputs: dict) -> Path:
    return Path(inputs["source_root_windows"] if platform.system() == "Windows" else inputs["source_root_wsl"])


def load_runtime_lock(config_dir: Path | str = CONFIG_DIR) -> dict:
    lock = _load("runtime-lock.json", config_dir)
    _require(lock["numerics"]["dtype"] == "float32" and lock["numerics"]["deterministic_algorithms"] is True,
             "numerics drifted")
    _require(lock["device"] in (None, "cpu", "cuda"), "device must be null/cpu/cuda")
    return lock


def validate_all(config_dir: Path | str = CONFIG_DIR) -> dict:
    return {"contract": load_contract(config_dir)["contract_version"],
            "inputs": load_inputs(config_dir)["inputs_version"],
            "runtime_lock": load_runtime_lock(config_dir)["lock_version"]}
