"""Utility functions for train/validation splitting with group-level constraints."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict


def group_aware_train_val_split(
    groups: np.ndarray,
    val_ratio: float,
    min_val_per_group: int = 1,
    random_state: int | None = None,
    skip_singleton_groups: bool = True,
) -> Dict[str, Any]:
    """Create a train/validation assignment that guarantees validation coverage per group.

    Parameters
    ----------
    groups : array-like
        Group identifier for each sample.
    val_ratio : float
        Desired validation ratio across the dataset.
    min_val_per_group : int, default=1
        Minimum number of validation samples per group when the group has enough members.
    random_state : int, optional
        Seed for deterministic shuffling within each group.
    skip_singleton_groups : bool, default=True
        If True, groups with a single sample remain fully in the training split.

    Returns
    -------
    dict
        Mapping with keys:
        - ``X_set`` (np.ndarray): indicator array where 0=train, 1=validation.
        - ``coverage`` (pd.DataFrame): per-group coverage summary.
        - ``val_groups`` (np.ndarray): ordered unique group IDs present in validation.
    """
    groups = np.asarray(groups)
    n_samples = groups.shape[0]
    X_set = np.zeros(n_samples, dtype=int)

    rng = np.random.RandomState(random_state)

    unique_groups, inverse = np.unique(groups, return_inverse=True)

    group_indices: Dict[int, np.ndarray] = {}
    for idx, group_position in enumerate(inverse):
        if group_position not in group_indices:
            group_indices[group_position] = []
        group_indices[group_position].append(idx)

    coverage_records = []
    val_indices = []

    for pos, gid in enumerate(unique_groups):
        member_indices = np.asarray(group_indices.get(pos, []), dtype=int)
        total = member_indices.size

        if total == 0:
            continue

        if skip_singleton_groups and total <= 1:
            coverage_records.append((gid, total, total, 0))
            continue

        # Ensure at least one validation sample (subject to available members).
        desired_val = max(min_val_per_group, int(np.ceil(total * val_ratio)))
        if total > 1:
            desired_val = min(desired_val, total - 1)
        else:
            desired_val = 0

        if desired_val <= 0:
            coverage_records.append((gid, total, total, 0))
            continue

        shuffled_indices = member_indices.copy()
        rng.shuffle(shuffled_indices)
        chosen_val = shuffled_indices[:desired_val]
        val_indices.extend(chosen_val.tolist())

        coverage_records.append((gid, total, total - desired_val, desired_val))

    if val_indices:
        X_set[np.asarray(val_indices, dtype=int)] = 1

    coverage_df = pd.DataFrame(
        coverage_records,
        columns=["FEWSNET_admin_code", "total_count", "train_count", "val_count"],
    ).sort_values("FEWSNET_admin_code").reset_index(drop=True)

    val_groups = coverage_df.loc[coverage_df["val_count"] > 0, "FEWSNET_admin_code"].to_numpy()

    return {"X_set": X_set, "coverage": coverage_df, "val_groups": val_groups}


def _month_labels(months) -> list:
    return [f"{int(m) // 12:04d}-{int(m) % 12 + 1:02d}" for m in months]


def time_block_split(groups, months, origin: int, n_months: int, expected_months=None) -> Dict[str, Any]:
    """D27 time block: the latest ``n_months`` distinct label months of the pool are
    validation (X_set=1) for every area alike; all earlier rows are fitting (X_set=0).

    ``months`` are integer month indices (year*12 + month - 1) of the root's legal pool,
    all strictly before ``origin``. No per-area reassignment: an area observed only in
    the block keeps validation rows and gets no fitting rows. Raises (never falls back
    to a random split) if the pool reaches the origin, has fewer than ``n_months``
    validation months plus at least one earlier fitting month, or the block differs from
    ``expected_months`` (YYYY-MM labels).
    """
    groups = np.asarray(groups)
    months = np.asarray(months, dtype=np.int64)
    if groups.shape != months.shape:
        raise ValueError("groups and months must be aligned")
    if months.size and months.max() >= origin:
        raise ValueError("a pool row is at or after the forecast origin (target month in the history pool)")
    distinct = np.unique(months)
    if distinct.size < n_months + 1:
        raise ValueError(f"incomplete time block: {distinct.size} observed label months, need {n_months} "
                         "validation months plus earlier fitting months")
    block = distinct[-n_months:]
    X_set = np.isin(months, block).astype(int)
    fitting_months = np.unique(months[X_set == 0])
    if fitting_months.max() >= block.min():
        raise ValueError("a fitting month is not strictly earlier than every validation month")
    validation_labels = _month_labels(block)
    if expected_months is not None and validation_labels != list(expected_months):
        raise ValueError(f"time-block validation months {validation_labels} differ from the frozen plan "
                         f"{list(expected_months)}")
    fit_groups = set(np.unique(groups[X_set == 0]).tolist())
    val_groups = np.unique(groups[X_set == 1])
    return {"X_set": X_set, "validation_months": validation_labels,
            "fitting_months": _month_labels(fitting_months), "val_groups": val_groups,
            "groups_with_validation": int(val_groups.size),
            "validation_only_groups": int(sum(g not in fit_groups for g in val_groups.tolist())),
            "fitting_only_groups": int(len(fit_groups - set(val_groups.tolist())))}



def confirmation_split(groups, months, seed: int = 42) -> np.ndarray:
    """D29 / A4: label-blind split of the ORIGINAL validation rows into S (0) and C (1).

    ``groups``/``months`` are the original validation rows only. A fresh
    ``random.Random(seed)``; areas numeric ascending, months ascending within area.
    Odd-count areas (ascending) are shuffled; the first floor(n_odd/2) give S the extra
    row, the rest give C. Then for every area in ascending order its month indices are
    shuffled with the same rng: the first floor(n/2)+extra rows are S, the rest C.
    Returns 0/1 per input row (input order). Duplicate (area, month) keys are refused.
    """
    import random

    groups = np.asarray(groups, dtype=np.int64)
    months = np.asarray(months, dtype=np.int64)
    if len(groups) != len(months):
        raise ValueError("groups and months differ in length")
    keys = pd.DataFrame({"g": groups, "m": months})
    if keys.duplicated().any():
        raise ValueError("duplicate (area, month) keys in the original validation rows")
    rng = random.Random(seed)
    by_area = {}
    for idx in np.lexsort((months, groups)):          # area ascending, month ascending
        by_area.setdefault(int(groups[idx]), []).append(int(idx))
    areas = sorted(by_area)
    odd = [a for a in areas if len(by_area[a]) % 2 == 1]
    rng.shuffle(odd)
    s_extra = {a: int(i < len(odd) // 2) for i, a in enumerate(odd)}
    role = np.ones(len(groups), dtype=int)
    for a in areas:
        rows = by_area[a]
        order = list(range(len(rows)))
        rng.shuffle(order)
        n_s = len(rows) // 2 + s_extra.get(a, 0)
        for j in order[:n_s]:
            role[rows[j]] = 0
    return role
