"""Assemble pipeline Inputs for a real run from ``prepared/`` (P0) and ``descriptors/`` (P1). Fails clearly when the
descriptor stage is absent: project-data fitting never starts without the frozen AlphaEarth table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_alphaearth import descriptors
from ipcch_alphaearth.artifacts import read_json, require_stage, sha256_file
from ipcch_alphaearth.errors import ContractError
from ipcch_alphaearth.pipeline import HorizonInputs, Inputs


def load_run_inputs(run_dir: Path, contract: dict) -> Inputs:
    run_dir = Path(run_dir)
    require_stage(run_dir, "prepared")
    if not (run_dir / "descriptors" / "COMPLETE.json").exists():
        raise ContractError("descriptors stage is not COMPLETE: AlphaEarth descriptors exist only after the P1 extraction release")
    from ipcch_alphaearth.contract import config_identity  # noqa: PLC0415

    done = read_json(run_dir / "descriptors" / "COMPLETE.json")
    if done.get("config_sha256") != config_identity() or done.get("prepared_manifest_sha256") != sha256_file(run_dir / "prepared" / "prepared-manifest.json"):
        raise ContractError("descriptors stage was built under a different configuration or prepared stage (no stale reuse)")
    if sha256_file(run_dir / "descriptors" / "descriptor_table.csv.gz") != done["files"]["descriptor_table.csv.gz"]:
        raise ContractError("descriptor table differs from its COMPLETE record")
    prepared = run_dir / "prepared"
    manifest = read_json(prepared / "prepared-manifest.json")
    for name, sha in manifest["artifacts_sha256"].items():
        if name.startswith(("X296_", "keys_")) and sha256_file(prepared / name) != sha:
            raise ContractError(f"prepared artifact {name} differs from the prepared manifest")
    order = pd.read_csv(prepared / "feature_order_296.csv.gz")["name"].tolist()
    core_idx = [order.index(c) for c in contract["correction"]["core_inputs"]]
    table_path = run_dir / "descriptors" / "descriptor_table.csv.gz"
    table = pd.read_csv(table_path, float_precision="round_trip")
    names = descriptors.ae_names(contract)
    horizons = {}
    for h in contract["calendar"]["horizons_months"]:
        keys = pd.read_csv(prepared / f"keys_h{h:02d}.csv.gz", float_precision="round_trip", keep_default_na=False,
                           na_values={c: [""] for c in ("q2", "q3", "q4", "q5", "persistence_q3")})
        X = np.load(prepared / f"X296_h{h:02d}.npy")
        D, routable = descriptors.row_descriptors(keys["admin_code"].to_numpy(np.int64), table, contract)
        horizons[h] = HorizonInputs(h, keys, X, D, routable, core_idx)
    D_admin = table[names].to_numpy(np.float64)
    return Inputs(horizons, order, order + names, table["admin_code"].to_numpy(np.int64), D_admin,
                  {"prepared_manifest_sha256": sha256_file(prepared / "prepared-manifest.json"),
                   "descriptor_table_sha256": sha256_file(table_path)}, names)
