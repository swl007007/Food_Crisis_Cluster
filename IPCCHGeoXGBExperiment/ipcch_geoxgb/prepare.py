"""P1 preparation: QC ledger, per-H rich561 matrices, F/S split and calendars.

Writes into ``runs/<run-id>/prepared/``; every artifact's SHA256 goes into
``prepared-manifest.json`` so later stages bind to exact data identities. No
model is fitted here.

Each supervised row is one QC-valid original (area, target month) for one H.
Its features use only information observed at or before its own origin
O = T - H (observation-month-end convention, R22). Persistence for the row is
the latest lawful valid observation at or before O (R10): its share-derived
phase and q3, with source month and age.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_geoxgb import features as feat
from ipcch_geoxgb import schedule, targets
from ipcch_geoxgb.artifacts import sha256_file, write_json
from ipcch_geoxgb.contract import input_path, load_experiment_contract, load_feature_schema, load_inputs
from ipcch_geoxgb.errors import ContractError
from ipcch_geoxgb.geography import load_country_lookup
from ipcch_geoxgb.preflight import verify_identities

AVAILABILITY_POLICY = {
    "id": "observation-month-end-v1",
    "rule": "covariates, population history and training labels must have observation month <= O = T - H",
    "persistence": "latest QC-valid observation of the same area with month <= O; phase and q3 from that observation",
    "limitation": "observation-month convention only; historical release timing/vintages not verified",
}


def persistence_lookup(valid: pd.DataFrame, admin: np.ndarray, origin_ord: np.ndarray) -> pd.DataFrame:
    """Latest lawful observation at or before each origin; NaN where none (R10)."""
    slot = feat.as_of_index(valid["admin_code"], valid["month_ord"], admin, origin_ord)
    has = slot >= 0
    safe = np.clip(slot, 0, None)
    source = np.where(has, valid["month_ord"].to_numpy()[safe], -1)
    return pd.DataFrame(
        {
            "persistence_available": has.astype(np.int64),
            "persistence_phase": np.where(has, valid["phase_truth"].to_numpy()[safe], 0).astype(np.int64),
            "persistence_q3": np.where(has, valid["q3"].to_numpy()[safe], np.nan),
            "persistence_source_month": feat.month_label(source),
            "persistence_age_months": np.where(has, origin_ord - source, -1).astype(np.int64),
        }
    )


def build_horizon(
    valid: pd.DataFrame,
    panel: pd.DataFrame,
    grid: feat.PanelGrid,
    index: feat.HistoryIndex,
    schema: dict,
    horizon: int,
    country: dict,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """rich561 X and keyed metadata for every valid outcome at one horizon."""
    original, audit = feat.assemble_original93(valid, panel, horizon, grid)
    admin = valid["admin_code"].to_numpy(dtype=np.int64)
    target_ord = valid["month_ord"].to_numpy(dtype=np.int64)
    origin_ord = target_ord - horizon
    history_names = [n for block in schema["additional_blocks"].values() for n in block]
    history = feat.build_history_block(index, admin, origin_ord, history_names)
    aliases = feat.check_aliases(original, feat.ORIGINAL_FEATURES, schema["aliases"], index, admin, origin_ord)
    X = np.ascontiguousarray(np.concatenate([original, history], axis=1))
    if X.shape[1] != 561:
        raise ContractError(f"H{horizon}: rich matrix has {X.shape[1]} columns")
    if np.isinf(X).any():
        raise ContractError(f"H{horizon}: final matrix contains an infinity")
    newest_age = original[:, feat.ORIGINAL_FEATURES.index("last_observed_label_age_months")]
    if np.nanmin(newest_age, initial=0.0) < 0:
        raise ContractError(f"H{horizon}: an observation after the origin was used")

    keys = pd.DataFrame(
        {
            "admin_code": admin,
            "target_month": feat.month_label(target_ord),
            "target_ord": target_ord,
            "horizon_months": horizon,
            "origin_ord": origin_ord,
            "origin_month": feat.month_label(origin_ord),
            "country_key": [country[int(a)] for a in admin],
            "phase_truth": valid["phase_truth"].to_numpy(dtype=np.int64),
            "crisis_truth": valid["crisis_truth"].to_numpy(dtype=np.int64),
            **{q: valid[q].to_numpy(dtype=np.float64) for q in targets.TARGETS},
            "last_observed_month_at_origin": audit.pop("last_observed_month"),
        }
    )
    keys = pd.concat([keys, persistence_lookup(valid, admin, origin_ord)], axis=1)
    audit.update(
        {
            "aliases_verified": aliases,
            "nan_fraction_original93": float(np.isnan(original).mean()),
            "nan_fraction_history468": float(np.isnan(history).mean()),
            "all_nan_columns": [schema["ordered_names"][i] for i in np.flatnonzero(np.isnan(X).all(axis=0))],
            "persistence_available": int(keys["persistence_available"].sum()),
        }
    )
    return X, keys, audit


def run_prepare(run_dir: Path, inputs_config: dict | None = None) -> dict:
    started = time.time()
    contract = load_experiment_contract()
    schema = load_feature_schema()
    if tuple(schema["original_features"]) != feat.ORIGINAL_FEATURES:
        raise ContractError("original93 implementation order differs from config/feature-schema.json")
    config = inputs_config or load_inputs()
    inputs = config["inputs"]
    identities = verify_identities({k: inputs[k] for k in ("raw_panel", "country_lookup_source")})
    raw_path = input_path(inputs["raw_panel"])

    out = run_dir / "prepared"
    out.mkdir()
    ledger = targets.build_target_ledger(raw_path)
    valid = targets.valid_targets(ledger)
    lookup, _ = load_country_lookup(input_path(inputs["country_lookup_source"]))
    country = dict(zip(lookup["area_id"].tolist(), lookup["country_key"].tolist()))
    missing = sorted(set(valid["admin_code"].tolist()) - set(country))
    if missing:
        raise ContractError(f"{len(missing)} areas with valid outcomes lack a country key")

    panel, panel_audit = feat.load_covariate_panel(raw_path)
    grid = feat.build_panel_grid(panel)
    index = feat.build_history_index(valid)

    artifacts: dict[str, str] = {}

    def save_csv(frame: pd.DataFrame, name: str) -> None:
        path = out / name
        compression = {"method": "gzip", "mtime": 0} if name.endswith(".gz") else None
        frame.to_csv(path, index=False, compression=compression, lineterminator="\n")
        artifacts[name] = sha256_file(path)

    save_csv(ledger, "target_ledger.csv.gz")
    save_csv(valid, "target_ledger_valid.csv.gz")

    horizon_audit = {}
    for h in contract["calendar"]["horizons_months"]:
        X, keys, audit = build_horizon(valid, panel, grid, index, schema, int(h), country)
        x_name = f"X_rich561_h{h:02d}.npy"
        np.save(out / x_name, X)
        artifacts[x_name] = sha256_file(out / x_name)
        save_csv(keys, f"keys_h{h:02d}.csv.gz")
        horizon_audit[str(h)] = {"rows": int(len(keys)), **audit}
        del X

    universe = lookup["area_id"].to_numpy()
    split, split_audit = schedule.stage1_split(
        valid, contract["calendar"]["development_first_target_month"], contract["calendar"]["development_last_month"], universe
    )
    save_csv(split, "stage1_split.csv.gz")

    main = schedule.attach_fold_support(schedule.main_fold_calendar(contract), valid)
    supp, coverage = schedule.supplementary_calendar(contract, valid)
    supp = schedule.attach_fold_support(supp, valid)
    save_csv(pd.concat([main, supp], ignore_index=True), "fold_calendar.csv")
    save_csv(coverage, "coverage_2026.csv")

    schema_frame = pd.DataFrame(
        {
            "position": np.arange(561),
            "name": schema["ordered_names"],
            "block": ["original93"] * 93
            + [b for b, names in schema["additional_blocks"].items() for _ in names],
        }
    )
    save_csv(schema_frame, "feature_order.csv")

    manifest = {
        "stage": "P1-prepare",
        "schema_version": schema["schema_version"],
        "ordered_names_sha256": schema["ordered_names_sha256"],
        "contract_version": contract["contract_version"],
        "availability_policy": AVAILABILITY_POLICY,
        "input_identities": identities,
        "ledger": targets.ledger_summary(ledger),
        "panel": panel_audit,
        "history_index": {"observations": index.n, "max_observations_per_area": index.max_area_span},
        "horizons": horizon_audit,
        "stage1_split": split_audit,
        "folds": {
            "main_scheduled": int(len(main)),
            "main_nonempty": int((main["eval_keys"] > 0).sum()),
            "main_by_horizon": {str(h): int((main["horizon_months"] == h).sum()) for h in contract["calendar"]["horizons_months"]},
            "supplementary_folds": int(len(supp)),
            "supplementary_months": coverage.loc[coverage["valid_outcomes"] > 0, "target_month"].tolist(),
        },
        "artifacts_sha256": artifacts,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    digest = write_json(out / "prepared-manifest.json", manifest)
    return {**manifest, "manifest_sha256": digest}
