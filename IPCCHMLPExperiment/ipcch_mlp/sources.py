"""Read-only access to the verified p6-formal-20261004b inputs (design section 2-3).

Every consumer first re-hashes the 28 pinned files (``verify``). Calendar and
support helpers are attributed copies from ipcch_geoxgb (schedule.py
``training_window``/``historical_gate_dates``, stage1.py ``support``/``meets``)
at 6798df2 with unchanged logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_mlp.artifacts import sha256_file
from ipcch_mlp.contract import load_inputs, source_root
from ipcch_mlp.errors import ContractError, TechnicalError

TARGETS = ("q2", "q3", "q4", "q5")


def verify(inputs: dict | None = None) -> dict:
    inputs = inputs or load_inputs()
    root = source_root(inputs)
    observed = {}
    for rel, entry in inputs["files"].items():
        path = root / rel
        if not path.is_file():
            raise ContractError(f"pinned input missing: {path}")
        size = path.stat().st_size
        if size != entry["bytes"]:
            raise ContractError(f"{rel}: {size} bytes, pinned {entry['bytes']}")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ContractError(f"{rel}: sha256 {digest} != pinned {entry['sha256']}")
        observed[rel] = digest
    return {"root": str(root), "files": observed}


# ------------------------------------------------------------- calendar/support (copied)

def training_window(origin_ord: int, months: int = 36) -> tuple[int, int]:
    """Closed target-month window [O - 35, O] (copy of schedule.training_window)."""
    return origin_ord - (months - 1), origin_ord


def historical_gate_dates(observed_months, origin_ord: int, max_dates: int = 6) -> np.ndarray:
    """Latest up to six distinct observed target months U < O, newest first (copy)."""
    months = np.unique(np.asarray(observed_months, dtype=np.int64))
    earlier = months[months < origin_ord]
    return earlier[::-1][:max_dates]


def support(frame: pd.DataFrame) -> dict:
    """Original-key support of a pool (copy of stage1.support)."""
    crisis = int(frame["crisis_truth"].sum())
    return {"keys": int(len(frame)), "areas": int(frame["admin_code"].nunique()),
            "target_months": int(frame["target_ord"].nunique()), "crisis_keys": crisis,
            "noncrisis_keys": int(len(frame) - crisis)}


def meets(record: dict, floor: dict) -> bool:
    return all(record[k] >= v for k, v in floor.items())


# ------------------------------------------------------------- data

@dataclass
class Horizon:
    """One H: keys aligned with X rows, the frozen map and month index."""

    h: int
    keys: pd.DataFrame
    X: np.ndarray  # float64 memmap (n, 561)
    region_of: dict  # area -> node id (string)
    x_sha256: str
    keys_sha256: str
    map_sha256: str
    rows_by_month: dict = field(default_factory=dict)
    regions: dict = field(default_factory=dict)

    def __post_init__(self):
        months = self.keys["target_ord"].to_numpy(dtype=np.int64)
        order = np.argsort(months, kind="mergesort")
        for m in np.unique(months):
            self.rows_by_month[int(m)] = np.sort(order[months[order] == m])
        nodes: dict = {}
        for area, node in self.region_of.items():
            nodes.setdefault(node, []).append(int(area))
        self.regions = {n: np.array(sorted(a), dtype=np.int64) for n, a in sorted(nodes.items())}
        self.area = self.keys["admin_code"].to_numpy(dtype=np.int64)
        self.Y = self.keys[list(TARGETS)].to_numpy(dtype=np.float64)
        self.observed_months = np.unique(months)
        self.node = np.array([self.region_of.get(int(a), "") for a in self.area], dtype=object)

    def rows_at(self, month: int) -> np.ndarray:
        return self.rows_by_month.get(int(month), np.zeros(0, dtype=np.int64))

    def window_rows(self, origin: int, months: int = 36) -> np.ndarray:
        lo, hi = training_window(origin, months)
        parts = [self.rows_at(m) for m in range(lo, hi + 1)]
        return np.sort(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)

    def region_rows(self, rows: np.ndarray, node: str) -> np.ndarray:
        return rows[np.isin(self.area[rows], self.regions[node])]

    def raw(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(self.X[np.asarray(rows, dtype=np.int64)], dtype=np.float64)


def load_horizon(h: int, inputs: dict | None = None) -> Horizon:
    inputs = inputs or load_inputs()
    root = source_root(inputs)
    keys_rel, x_rel = f"prepared/keys_h{h:02d}.csv.gz", f"prepared/X_rich561_h{h:02d}.npy"
    map_rel, frozen_rel = f"stage1/frozen_map_h{h:02d}.csv", f"stage1/frozen_h{h:02d}.json"
    keys = pd.read_csv(root / keys_rel)
    X = np.load(root / x_rel, mmap_mode="r")
    if X.shape != (len(keys), 561) or X.dtype != np.float64:
        raise TechnicalError(f"H{h}: keys/X shape mismatch {X.shape} vs {len(keys)}")
    if keys.duplicated(["admin_code", "target_ord"]).any():
        raise TechnicalError(f"H{h}: duplicated keys")
    frozen = json.loads((root / frozen_rel).read_text(encoding="utf-8"))
    map_sha = inputs["files"][map_rel]["sha256"]
    if frozen.get("map_sha256") != map_sha or frozen.get("H") != h:
        raise TechnicalError(f"H{h}: frozen record does not bind the pinned map")
    if frozen.get("accepted_split") is not True:
        raise TechnicalError(f"H{h}: the pinned map has no accepted split (unexpected for this experiment)")
    fmap = pd.read_csv(root / map_rel, dtype={"node_id": str})
    if fmap["admin_code"].duplicated().any():
        raise TechnicalError(f"H{h}: duplicate areas in the frozen map")
    region_of = dict(zip(fmap["admin_code"].astype(int), fmap["node_id"].astype(str)))
    return Horizon(h=h, keys=keys, X=X, region_of=region_of, x_sha256=inputs["files"][x_rel]["sha256"],
                   keys_sha256=inputs["files"][keys_rel]["sha256"], map_sha256=map_sha)


def load_split(inputs: dict | None = None) -> pd.DataFrame:
    inputs = inputs or load_inputs()
    return pd.read_csv(source_root(inputs) / "prepared/stage1_split.csv.gz")


def split_roles(hz: Horizon, split: pd.DataFrame) -> np.ndarray:
    """Per keys row: 'fit' / 'validation' / 'singleton' / '' (outside 2014-2022)."""
    role = dict(zip(zip(split["admin_code"].astype(int), split["month_ord"].astype(int)), split["split_role"]))
    return np.array([role.get((int(a), int(t)), "") for a, t in
                     zip(hz.keys["admin_code"], hz.keys["target_ord"])], dtype=object)


def load_calendar(inputs: dict | None = None) -> pd.DataFrame:
    inputs = inputs or load_inputs()
    return pd.read_csv(source_root(inputs) / "prepared/fold_calendar.csv")


def load_p6_predictions(h: int, inputs: dict | None = None) -> pd.DataFrame:
    inputs = inputs or load_inputs()
    return pd.read_csv(source_root(inputs) / f"stage3/h{h:02d}/predictions.csv.gz", float_precision="round_trip")
