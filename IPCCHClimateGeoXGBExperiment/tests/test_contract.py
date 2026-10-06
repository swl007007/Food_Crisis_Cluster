"""Frozen configuration: accepted values and tamper detection."""

from __future__ import annotations

import json
import shutil

import pytest

from ipcch_climate_geoxgb import CONFIG_DIR
from ipcch_climate_geoxgb.contract import (
    load_experiment_contract,
    load_feature_schema,
    load_inputs,
    load_runtime_lock,
    validate_all,
)
from ipcch_climate_geoxgb.errors import ContractError


def test_all_configs_validate():
    versions = validate_all()
    assert versions["feature_schema"] == "ipcch-climate-geoxgb-rich601-ge020-v1"


def test_accepted_r43_recipes_and_fixed_parameters():
    model = load_experiment_contract()["model"]
    assert model["global_recipes"] == {
        "G1": {"max_depth": 3, "rounds": 200},
        "G2": {"max_depth": 3, "rounds": 400},
        "G3": {"max_depth": 4, "rounds": 200},
        "G4": {"max_depth": 4, "rounds": 400},
    }
    assert model["local_recipes"] == {
        "L1": {"max_depth": 1, "appended_rounds": 20},
        "L2": {"max_depth": 2, "appended_rounds": 40},
    }
    assert model["global_common"] == {
        "min_child_weight": 10, "reg_lambda": 10, "reg_alpha": 0, "subsample": 0.8, "colsample_bytree": 0.8,
    }
    assert model["local_common"] == {
        "min_child_weight": 20, "reg_lambda": 20, "reg_alpha": 1, "subsample": 1.0, "colsample_bytree": 1.0,
    }
    assert model["fixed"] == {
        "booster": "gbtree", "tree_method": "hist", "device": "cpu", "num_parallel_tree": 1, "seed": 42,
        "nthread": 4, "gamma": 0, "max_delta_step": 0, "max_bin": 256, "grow_policy": "depthwise", "eta": 0.05,
    }
    assert model["global_base_score"] == 0.5 and model["early_stopping"] is False


def test_calendar_support_and_gates():
    c = load_experiment_contract()
    assert c["calendar"]["main_first_target_month"] == {"1": "2023-02", "3": "2023-04", "6": "2023-07", "12": "2024-01"}
    assert c["partition"]["stage1_gain_strictly_greater_than"] == "0"
    assert c["partition"]["stage3_gain_strictly_greater_than"] == "0.01"
    assert c["support"]["stage3_min_successful_local_dates"] == 3
    assert c["reporting"]["fit_ceiling_upper_bound"] == {"stage1": 3904, "stage3": 80920, "total": 84824}


def test_schema_order_and_semantics():
    s = load_feature_schema()
    names = s["ordered_names"]
    assert len(names) == 601 and names[:133] == s["original_features"]
    assert "EVI_mean" not in names and "GPP_mean" not in names and "evi_anom_month_ensmean_lag12_asof" in names
    assert len(s["perturbation"]["removed"]) == 16 and len(s["perturbation"]["added"]) == 56
    assert names[-1] == s["additional_blocks"]["event_and_run"][-1]
    assert s["crisis_threshold"] == "0.20" and s["threshold_rule"] == ">="
    assert "status" not in s


def test_inputs_and_lock():
    inputs = load_inputs()
    assert len(inputs["inputs"]) == 15
    assert inputs["inputs"]["climate_monthly"]["sha256"] == "8082b72ea5fa5c30a7b975b89c1fbbb4530f27a9ce6c5ba4d0e1a4f923320c89"
    assert inputs["inputs"]["climate_growing_season"]["sha256"] == "f024a66c8979fb4a8c66fba1f04e7e69355fa75b499793d29001a146d8c2958e"
    assert inputs["inputs"]["raw_panel"]["sha256"] == "ae696087c3bbb280537ae269a05924133acdb51060d31290523404fa8a717673"
    lock = load_runtime_lock()
    assert lock["packages"]["xgboost"] == "3.0.0" and lock["geometry_io_engine"] == "pyogrio"


def _tampered(tmp_path, name, mutate):
    target = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, target)
    data = json.loads((target / name).read_text(encoding="utf-8"))
    mutate(data)
    (target / name).write_text(json.dumps(data), encoding="utf-8")
    return target


@pytest.mark.parametrize(
    "name, mutate, loader",
    [
        ("experiment-contract.json", lambda d: d["model"]["global_recipes"]["G1"].update(rounds=300), load_experiment_contract),
        ("experiment-contract.json", lambda d: d["targets"].update(threshold_rule=">"), load_experiment_contract),
        ("experiment-contract.json", lambda d: d["calendar"]["main_fold_counts"].update({"12": 25}), load_experiment_contract),
        ("experiment-contract.json", lambda d: d["support"]["validation"].update(crisis_keys=10), load_experiment_contract),
        ("feature-schema.json", lambda d: d["ordered_names"].reverse(), load_feature_schema),
        ("feature-schema.json", lambda d: d.update(threshold_rule=">"), load_feature_schema),
        ("inputs.json", lambda d: d["inputs"]["raw_panel"].update(sha256="0" * 63), load_inputs),
        ("runtime-lock.json", lambda d: d["packages"].update(xgboost="2.1.4"), load_runtime_lock),
    ],
)
def test_tampered_config_is_rejected(tmp_path, name, mutate, loader):
    with pytest.raises(ContractError):
        loader(_tampered(tmp_path, name, mutate))
