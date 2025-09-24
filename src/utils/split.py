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

