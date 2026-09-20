"""Utilities for assembling terminal partition assignments."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Sequence, Tuple

import pandas as pd

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is expected in runtime env
    np = None  # type: ignore

try:
    from config_visual import (
        PARTITIONING_VALIDATE_TERMINAL_LABELS,
        VISUALIZE_HIDE_UNASSIGNED,
    )
except ImportError:  # pragma: no cover
    from config import (
        PARTITIONING_VALIDATE_TERMINAL_LABELS,
        VISUALIZE_HIDE_UNASSIGNED,
    )  # type: ignore


def _as_array(values: Sequence[object]) -> Sequence[object]:
    if np is None:
        return values
    try:
        return np.asarray(values)
    except Exception:
        return values


def build_terminal(
    X_group: Sequence[object],
    X_branch_id: Sequence[object],
    *,
    adoption_map: Dict[str, str] | None = None,
    keep_branch_id: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Create a correspondence dataframe preserving terminal lineage.

    Parameters
    ----------
    X_group : sequence
        Iterable of group identifiers aligned with ``X_branch_id`` records.
    X_branch_id : sequence
        Iterable of hierarchical branch labels (e.g., '0101').
    adoption_map : dict, optional
        Mapping of branch labels to the labels they should inherit from when a
        child adopted its parent (e.g., {'0100': '010'}).
    keep_branch_id : bool
        Retain the original branch label alongside ``partition_id`` for
        diagnostics.

    Returns
    -------
    (DataFrame, diagnostics)
        DataFrame contains ``FEWSNET_admin_code``, ``partition_id`` and
        optionally ``branch_id`` columns. Diagnostics surface duplicates and
        adoption corrections applied.
    """
    if len(X_group) != len(X_branch_id):
        raise ValueError("X_group and X_branch_id must have the same length")

    group_values = _as_array(X_group)
    branch_values = _as_array(X_branch_id)

    adoption_map = adoption_map or {}

    assignments: "OrderedDict[str, Dict[str, str]]" = OrderedDict()
    collisions: list[tuple[str, str, str]] = []

    for idx, raw_branch in enumerate(branch_values):
        uid_raw = group_values[idx]
        uid = str(uid_raw)
        branch_label = str(raw_branch) if raw_branch not in ("", None) else "root"
        adopted_label = adoption_map.get(branch_label, branch_label)

        entry = assignments.get(uid)
        if entry is None:
            assignments[uid] = {
                "partition_id": adopted_label,
                "branch_id": branch_label,
            }
        else:
            if entry["partition_id"] != adopted_label:
                collisions.append((uid, entry["partition_id"], adopted_label))

    data = {
        "FEWSNET_admin_code": list(assignments.keys()),
        "partition_id": [entry["partition_id"] for entry in assignments.values()],
    }
    if keep_branch_id:
        data["branch_id"] = [entry["branch_id"] for entry in assignments.values()]

    df = pd.DataFrame(data)
    df["FEWSNET_admin_code"] = df["FEWSNET_admin_code"].astype(str)
    df["partition_id"] = df["partition_id"].astype(str)

    if PARTITIONING_VALIDATE_TERMINAL_LABELS:
        valid_mask = df["partition_id"].str.fullmatch(r"[01]+|root")
        if not valid_mask.all():
            df.loc[~valid_mask, "partition_id"] = "root"

    if VISUALIZE_HIDE_UNASSIGNED:
        df = df[df["partition_id"].notna()]

    diagnostics = {
        "n_assignments": len(df),
        "n_collisions": len(collisions),
        "collisions": collisions[:20],
        "adoption_applied": adoption_map,
    }

    df = df.sort_values("FEWSNET_admin_code").reset_index(drop=True)
    return df, diagnostics
