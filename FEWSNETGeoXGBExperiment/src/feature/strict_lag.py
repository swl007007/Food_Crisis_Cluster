"""Feature slicing for forecast-time-safe lag-only experiments."""

from typing import Sequence

import numpy as np


def select_strict_lag_features(
    X: np.ndarray,
    feature_columns: Sequence[str],
    l1_index: Sequence[int],
    l2_index: Sequence[int],
    lag_months: int,
) -> tuple[np.ndarray, list[int], list[int], list[str]]:
    """Keep static L1 columns and only the active lagged L2 columns."""
    if X.shape[1] != len(feature_columns):
        raise ValueError("Feature matrix width does not match feature column count")

    l1 = set(l1_index)
    l2 = set(l2_index)
    suffix = f"_lag{lag_months}m"
    keep = [
        index
        for index, name in enumerate(feature_columns)
        if index in l1 or (index in l2 and str(name).endswith(suffix))
    ]
    active_lag = [index for index in keep if index in l2]
    if not active_lag:
        raise ValueError(f"No active lag columns found for lag {lag_months} months")

    selected_columns = [str(feature_columns[index]) for index in keep]
    selected_l1 = [position for position, index in enumerate(keep) if index in l1]
    selected_l2 = [position for position, index in enumerate(keep) if index in l2]
    return X[:, keep], selected_l1, selected_l2, selected_columns
