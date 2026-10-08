"""Read-only access to the verified p6-formal-20261004b inputs (design sections 1, 6).

``stage`` copies the 33 pinned run files and the 3 original config files once
into a hash-verified snapshot outside Dropbox (``<run>/inputs/run/<rel>`` and
``<run>/inputs/source/<rel>``, source-relative paths preserved, plus
``staging-manifest.json``); every loader takes that snapshot (``staged``)
and ``verify`` rehashes it. Without ``staged`` the loaders read the original
Dropbox locations (used only by staging itself and planning checks).
``load_horizon`` re-validates frozen-map lineage against the original P6
contract, adapted from ipcch_geoxgb/predict.py ``load_frozen`` at 6798df2:
the record must be for this H, bound to the pinned prepared manifest, the
original contract version and feature schema, equal its Stage1 summary entry
and selection ledger, and the map bytes must equal its record. Keys are parsed
with pandas defaults exactly as ipcch_geoxgb/learnmap.py ``load_horizon``.
``support``/``meets`` are copies of ipcch_geoxgb/stage1.py (unchanged logic).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_yearly_xgb.artifacts import sha256_file
from ipcch_yearly_xgb.contract import load_inputs, repo_root, run_root
from ipcch_yearly_xgb.errors import ContractError, TechnicalError

TARGETS = ("q2", "q3", "q4", "q5")


def roots(inputs: dict, staged: Path | None = None) -> tuple[Path, Path]:
    """(run-file base, source-config base): the staged snapshot if given, else the original locations."""
    if staged is not None:
        return Path(staged) / "run", Path(staged) / "source"
    return run_root(inputs), repo_root(inputs)


def stage(dest: Path, inputs: dict | None = None) -> dict:
    """Copy the 36 pinned inputs once into ``dest`` (outside Dropbox) and verify the copies."""
    inputs = inputs or load_inputs()
    verify(inputs)  # originals first
    dest = Path(dest)
    if dest.exists():
        raise ContractError(f"staging directory already exists: {dest}")
    manifest = {"inputs_version": inputs["inputs_version"], "source_run": inputs["source_run"], "files": {}}
    for kind, base, group in (("run", run_root(inputs), inputs["run_files"]),
                              ("source", repo_root(inputs), inputs["source_config_files"])):
        for rel, entry in group.items():
            target = dest / kind / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(base / rel, target)
            manifest["files"][f"{kind}/{rel}"] = {"origin": str(base / rel), "sha256": entry["sha256"],
                                                  "bytes": entry["bytes"]}
    observed = verify(inputs, staged=dest)
    (dest / "staging-manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return {"files": len(observed), "manifest": manifest}


def verify(inputs: dict | None = None, staged: Path | None = None) -> dict:
    inputs = inputs or load_inputs()
    observed = {}
    run_base, src_base = roots(inputs, staged)
    for base, group in ((run_base, inputs["run_files"]), (src_base, inputs["source_config_files"])):
        for rel, entry in group.items():
            path = base / rel
            if not path.is_file():
                raise ContractError(f"pinned input missing: {path}")
            if path.stat().st_size != entry["bytes"]:
                raise ContractError(f"{rel}: size {path.stat().st_size} != pinned {entry['bytes']}")
            digest = sha256_file(path)
            if digest != entry["sha256"]:
                raise ContractError(f"{rel}: sha256 {digest} != pinned {entry['sha256']}")
            observed[rel] = digest
    return observed


def support(frame: pd.DataFrame) -> dict:
    crisis = int(frame["crisis_truth"].sum())
    return {"keys": int(len(frame)), "areas": int(frame["admin_code"].nunique()),
            "target_months": int(frame["target_ord"].nunique()), "crisis_keys": crisis,
            "noncrisis_keys": int(len(frame) - crisis)}


def meets(record: dict, floor: dict) -> bool:
    return all(record[k] >= v for k, v in floor.items())


def frozen_lineage(h: int, inputs: dict, recipe: str, staged: Path | None = None) -> tuple[dict, dict]:
    root, repo = roots(inputs, staged)
    orig_contract = json.loads((repo / "IPCCHGeoXGBExperiment/config/experiment-contract.json").read_text("utf-8"))
    schema = json.loads((repo / "IPCCHGeoXGBExperiment/config/feature-schema.json").read_text("utf-8"))
    schema_identity = [schema["schema_version"], schema["ordered_names_sha256"]]
    manifest_sha = inputs["run_files"]["prepared/prepared-manifest.json"]["sha256"]
    record = json.loads((root / f"stage1/frozen_h{h:02d}.json").read_text("utf-8"))
    summary = json.loads((root / "stage1/stage1-summary.json").read_text("utf-8"))
    selection_path = root / f"stage1/h{h:02d}/selection.json"
    selection = json.loads(selection_path.read_text("utf-8"))
    problems = []
    if record.get("H") != h:
        problems.append("record H")
    if record.get("prepared_manifest_sha256") != manifest_sha:
        problems.append("prepared manifest")
    if record.get("contract_version") != orig_contract["contract_version"]:
        problems.append("contract version")
    if record.get("schema") != schema_identity:
        problems.append("feature schema")
    if summary.get("horizons", {}).get(str(h), {}).get("frozen") != record:
        problems.append("stage1 summary")
    if selection.get("selection", {}).get("winner") != record.get("candidate") or \
            record.get("selection_sha256") != sha256_file(selection_path):
        problems.append("selection ledger")
    if record.get("candidate") != f"{record.get('G')}{record.get('L')}" or record.get("candidate") != recipe:
        problems.append(f"candidate {record.get('candidate')} vs frozen recipe {recipe}")
    if sha256_file(root / f"stage1/frozen_map_h{h:02d}.csv") != record.get("map_sha256"):
        problems.append("map bytes")
    if record.get("accepted_split") is not True:
        problems.append("no accepted split")
    if problems:
        raise ContractError(f"frozen map H{h} lineage rejected: {problems}")
    return record, {"schema": schema_identity, "prepared_manifest_sha256": manifest_sha,
                    "source_contract_version": orig_contract["contract_version"]}


@dataclass
class Horizon:
    h: int
    keys: pd.DataFrame
    X: np.ndarray
    region_of: dict
    x_sha256: str
    keys_sha256: str
    map_sha256: str
    lineage: dict = field(default_factory=dict)

    def __post_init__(self):
        self.t = self.keys["target_ord"].to_numpy(dtype=np.int64)
        self.area = self.keys["admin_code"].to_numpy(dtype=np.int64)
        self.Y = self.keys[list(TARGETS)].to_numpy(dtype=np.float64)
        self.observed_months = np.unique(self.t)
        self.node = np.array([self.region_of.get(int(a), "") for a in self.area], dtype=object)
        nodes: dict = {}
        for area, node in self.region_of.items():
            nodes.setdefault(node, []).append(int(area))
        self.regions = {n: np.array(sorted(a), dtype=np.int64) for n, a in sorted(nodes.items())}

    def rows_at(self, month: int) -> np.ndarray:
        return np.flatnonzero(self.t == int(month))

    def pool_rows(self, origin: int) -> np.ndarray:
        """All valid rows with target month t <= O (no lower bound), ascending row order."""
        return np.flatnonzero(self.t <= int(origin))

    def region_mask(self, rows: np.ndarray, node: str) -> np.ndarray:
        return np.isin(self.area[rows], self.regions[node])


def load_horizon(h: int, recipe: str, inputs: dict | None = None, staged: Path | None = None) -> Horizon:
    inputs = inputs or load_inputs()
    root = roots(inputs, staged)[0]
    record, lineage = frozen_lineage(h, inputs, recipe, staged)
    keys = pd.read_csv(root / f"prepared/keys_h{h:02d}.csv.gz")
    X = np.load(root / f"prepared/X_rich561_h{h:02d}.npy", mmap_mode="r")
    if len(keys) != X.shape[0] or X.shape[1] != 561:
        raise TechnicalError(f"H{h}: keys/X shape mismatch")
    if keys.duplicated(["admin_code", "target_ord"]).any():
        raise TechnicalError(f"H{h}: duplicated keys")
    fmap = pd.read_csv(root / f"stage1/frozen_map_h{h:02d}.csv", dtype={"node_id": str})
    if fmap["admin_code"].duplicated().any():
        raise TechnicalError(f"H{h}: frozen map has duplicate areas")
    region_of = dict(zip(fmap["admin_code"].astype(int), fmap["node_id"].astype(str)))
    files = inputs["run_files"]
    return Horizon(h=h, keys=keys, X=X, region_of=region_of,
                   x_sha256=files[f"prepared/X_rich561_h{h:02d}.npy"]["sha256"],
                   keys_sha256=files[f"prepared/keys_h{h:02d}.csv.gz"]["sha256"],
                   map_sha256=record["map_sha256"], lineage=lineage)


def load_calendar(inputs: dict | None = None, staged: Path | None = None) -> pd.DataFrame:
    inputs = inputs or load_inputs()
    return pd.read_csv(roots(inputs, staged)[0] / "prepared/fold_calendar.csv")


def load_p6_predictions(h: int, inputs: dict | None = None, staged: Path | None = None) -> pd.DataFrame:
    inputs = inputs or load_inputs()
    return pd.read_csv(roots(inputs, staged)[0] / f"stage3/h{h:02d}/predictions.csv.gz", float_precision="round_trip")
