"""Scoped partition map helpers."""

from __future__ import annotations

import os
import tempfile
from typing import Iterable, Optional, Sequence

import pandas as pd

try:
    from config_visual import (
        VISUALIZE_ENFORCE_PARENT_SCOPE,
        VISUALIZE_HIDE_UNASSIGNED,
        PARTITIONING_VALIDATE_TERMINAL_LABELS,
    )
except ImportError:  # pragma: no cover - fallback for minimal environments
    from config import (
        VISUALIZE_ENFORCE_PARENT_SCOPE,
        VISUALIZE_HIDE_UNASSIGNED,
        PARTITIONING_VALIDATE_TERMINAL_LABELS,
    )

from src.vis.visualization import plot_partition_map


def _to_scope_set(parent_scope_uids: Iterable[object]) -> set[str]:
    scope = set()
    for uid in parent_scope_uids:
        if uid is None:
            continue
        scope.add(str(uid))
    return scope


def render_round_map(
    df_round: pd.DataFrame,
    parent_scope_uids: Iterable[object],
    parent_label: str,
    round_id: int,
    save_path: str,
    *,
    shapefile_path: Optional[str] = None,
    title: Optional[str] = None,
    missing_color: str = "#dddddd",
    VIS_DEBUG_MODE: Optional[bool] = None,
) -> Optional[str]:
    """Render a per-round partition map constrained to the parent scope.

    Parameters
    ----------
    df_round : pandas.DataFrame
        DataFrame containing at least ``FEWSNET_admin_code`` and ``partition_id``
        columns for the two child partitions produced this round.
    parent_scope_uids : Iterable
        Collection of UID values that belong to the parent branch being split.
    parent_label : str
        Branch identifier ('' for root) currently being split.
    round_id : int
        Zero-based round index (for annotation only).
    save_path : str
        Destination PNG path.
    shapefile_path : str, optional
        Optional override for the boundary polygons.
    title : str, optional
        Custom title; defaults to a scoped description.
    missing_color : str
        Hex color used by downstream plotting for NoData areas.
    VIS_DEBUG_MODE : bool, optional
        Explicit visualization flag. If ``None`` we defer to configuration.
    """
    if VIS_DEBUG_MODE is None:
        try:
            from config_visual import VIS_DEBUG_MODE as _VIS
        except ImportError:
            from config import VIS_DEBUG_MODE as _VIS  # type: ignore
        VIS_DEBUG_MODE = bool(_VIS)
    if not VIS_DEBUG_MODE:
        return None

    required_cols = {"FEWSNET_admin_code", "partition_id"}
    missing_cols = required_cols - set(map(str, df_round.columns))
    if missing_cols:
        raise ValueError(f"df_round missing required columns: {sorted(missing_cols)}")

    scope = _to_scope_set(parent_scope_uids)
    if not scope:
        raise ValueError("Parent scope is empty; nothing to render")

    df = df_round.copy()
    df["FEWSNET_admin_code"] = df["FEWSNET_admin_code"].astype(str)
    # retain strings; avoid coercing to numeric to preserve hierarchical labels
    df["partition_id"] = df["partition_id"].astype(str)

    if VISUALIZE_ENFORCE_PARENT_SCOPE:
        df = df[df["FEWSNET_admin_code"].isin(scope)]

    # Ensure every UID in scope appears exactly once
    seen = set(df["FEWSNET_admin_code"].tolist())
    missing_uids = sorted(scope - seen)
    if missing_uids:
        padding = pd.DataFrame({
            "FEWSNET_admin_code": missing_uids,
            "partition_id": [pd.NA] * len(missing_uids),
        })
        df = pd.concat([df, padding], ignore_index=True)

    base = "" if parent_label in ("", "root") else parent_label
    expected_children = {f"{base}0", f"{base}1"}

    if PARTITIONING_VALIDATE_TERMINAL_LABELS:
        unexpected_mask = ~df["partition_id"].isin(expected_children) & df["partition_id"].notna()
        df.loc[unexpected_mask, "partition_id"] = pd.NA

    df = df.drop_duplicates(subset=["FEWSNET_admin_code"], keep="first")
    df = df.sort_values("FEWSNET_admin_code")

    if VISUALIZE_HIDE_UNASSIGNED:
        filtered_df = df[df["partition_id"].notna()].copy()
    else:
        filtered_df = df.copy()

    scoped_title = title or (
        f"Round {round_id} • Branch {parent_label or 'root'}"
        f" • scope={len(scope)}"
    )

    temp_path = None
    diag_path = os.path.splitext(save_path)[0] + "_uids.csv"
    try:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
        temp_path = tmp.name
        filtered_df.to_csv(tmp, index=False)
        tmp.close()
        plot_partition_map(
            temp_path,
            shapefile_path=shapefile_path,
            save_path=save_path,
            title=scoped_title,
            VIS_DEBUG_MODE=VIS_DEBUG_MODE,
        )
        filtered_df[["FEWSNET_admin_code", "partition_id"]].to_csv(diag_path, index=False)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    return save_path
