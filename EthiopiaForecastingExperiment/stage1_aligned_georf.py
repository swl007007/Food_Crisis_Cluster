#!/usr/bin/env python3
"""Learn one Ethiopia GeoRF partition directly from an aligned 88-feature snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.split import group_aware_train_val_split
from EthiopiaForecastingExperiment.aligned_refit import (
    KEY,
    MODEL_PREDICTORS,
    SCOPE_HORIZONS,
    TARGET,
    apply_fold_medians,
    fit_fold_medians,
    select_rolling_fold,
    validate_aligned_frame,
)


def partition_with_nonempty_guard(partition_module, *args, **kwargs):
    """Reject empty candidate branches without changing production defaults."""
    original = partition_module.MIN_BRANCH_SAMPLE_SIZE
    partition_module.MIN_BRANCH_SAMPLE_SIZE = max(1, original)
    try:
        return partition_module.partition(*args, **kwargs)
    finally:
        partition_module.MIN_BRANCH_SAMPLE_SIZE = original


def prepare_stage1_fold(
    frame: pd.DataFrame,
    *,
    target_month: str,
    horizon: int,
    validation_fraction: float = 0.2,
) -> dict[str, object]:
    """Create finite Stage 1 matrices with imputation fitted before validation."""
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be in (0, 1)")
    train_indices, test_indices = select_rolling_fold(
        frame,
        target_month=target_month,
        horizon=horizon,
        window_months=36,
    )
    test_groups = frame.iloc[test_indices][KEY].unique()
    train_indices = train_indices[
        np.isin(frame.iloc[train_indices][KEY].to_numpy(), test_groups)
    ]
    train_dates = pd.to_datetime(frame.iloc[train_indices]["target_month"])
    split = group_aware_train_val_split(
        frame.iloc[train_indices][KEY].to_numpy(),
        val_ratio=validation_fraction,
        min_val_per_group=1,
        random_state=5,
    )
    validation_mask = split["X_set"].astype(bool)
    fit_mask = ~validation_mask
    X_raw = frame.iloc[train_indices][list(MODEL_PREDICTORS)].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=float)
    X_test_raw = frame.iloc[test_indices][list(MODEL_PREDICTORS)].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=float)
    medians = fit_fold_medians(X_raw[fit_mask])
    return {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "X_train": apply_fold_medians(X_raw, medians),
        "X_test": apply_fold_medians(X_test_raw, medians),
        "y_train": pd.to_numeric(frame.iloc[train_indices][TARGET], errors="raise").astype(int).to_numpy(),
        "y_test": pd.to_numeric(frame.iloc[test_indices][TARGET], errors="coerce").to_numpy(dtype=float),
        "groups_train": frame.iloc[train_indices][KEY].to_numpy(dtype=np.int32),
        "groups_test": frame.iloc[test_indices][KEY].to_numpy(dtype=np.int32),
        "X_set": validation_mask.astype(int),
        "medians": medians,
        "fit_months": int(train_dates[fit_mask].dt.to_period("M").nunique()),
        "validation_months": int(train_dates[validation_mask].dt.to_period("M").nunique()),
        "fit_rows": int(fit_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
    }


def _polygon_contract(frame: pd.DataFrame, row_indices: np.ndarray) -> dict[str, object]:
    locations = frame.iloc[row_indices][[KEY, "lat", "lon"]].drop_duplicates()
    if locations.duplicated(KEY).any():
        raise ValueError("Admin coordinates vary within the Stage 1 fold")
    locations = locations.sort_values(KEY).reset_index(drop=True)
    return {
        "polygon_centroids": locations[["lat", "lon"]].to_numpy(dtype=float),
        "polygon_group_mapping": {
            index: [int(code)] for index, code in enumerate(locations[KEY])
        },
        "neighbor_distance_threshold": 0.8,
        "adjacency_dict": None,
    }


def run_stage1(args: argparse.Namespace) -> Path:
    """Execute one Stage 1 scope-month cell and write the Stage 2 contract."""
    from config import MAX_DEPTH, MIN_DEPTH, N_JOBS
    from src.helper.helper import get_X_branch_id_by_group
    from src.merge.terminal import build_terminal
    from src.model.model_RF import RFmodel
    import src.partition.transformation as transformation

    if args.start_year != args.end_year:
        raise ValueError("Aligned Stage 1 accepts one year per invocation")
    scope = args.forecasting_scope
    if scope not in SCOPE_HORIZONS:
        raise ValueError(f"Unsupported forecasting scope: {scope}")
    target = pd.Period(args.desired_terms, freq="M")
    if target.year != args.start_year:
        raise ValueError("desired_terms year must equal start_year")

    frame = pd.read_csv(args.data, low_memory=False)
    validate_aligned_frame(frame, f"fs{scope}", SCOPE_HORIZONS[scope], MODEL_PREDICTORS)
    fold = prepare_stage1_fold(
        frame,
        target_month=str(target),
        horizon=SCOPE_HORIZONS[scope],
    )

    model_dir = Path.cwd() / "result_GeoRF"
    space_dir = model_dir / "space_partitions"
    checkpoint_dir = model_dir / "checkpoints"
    space_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir()
    model = RFmodel(
        str(checkpoint_dir),
        100,
        max_depth=None,
        random_state=args.random_seed,
        n_jobs=N_JOBS,
        max_model_depth=MAX_DEPTH,
        use_smote=True,
    )
    groups_train = fold["groups_train"]
    X_branch_id = np.full(len(groups_train), "", dtype=f"U{MAX_DEPTH + 1}")
    _, branch_table, s_branch = partition_with_nonempty_guard(
        transformation,
        model,
        fold["X_train"],
        fold["y_train"],
        groups_train,
        fold["X_set"],
        np.arange(len(groups_train)),
        X_branch_id,
        min_depth=MIN_DEPTH,
        max_depth=MAX_DEPTH,
        contiguity_type="polygon",
        polygon_contiguity_info=_polygon_contract(frame, fold["train_indices"]),
        track_partition_metrics=False,
        model_dir=str(model_dir),
        VIS_DEBUG_MODE=False,
    )
    X_branch_id = get_X_branch_id_by_group(groups_train, s_branch)
    s_branch.to_pickle(space_dir / "s_branch.pkl")
    np.save(space_dir / "branch_table.npy", branch_table)
    np.save(space_dir / "X_branch_id.npy", X_branch_id)

    correspondence, diagnostics = build_terminal(
        groups_train,
        X_branch_id,
        keep_branch_id=False,
    )
    expected_groups = set(map(str, np.unique(fold["groups_test"])))
    if diagnostics["n_collisions"] or set(correspondence[KEY]) != expected_groups:
        raise ValueError("Stage 1 correspondence does not cover the exact test cohort")
    correspondence_path = model_dir / f"correspondence_table_{target}.csv"
    correspondence.to_csv(correspondence_path, index=False)

    observed_test = np.isfinite(fold["y_test"])
    model.load("")
    pooled = model.predict(fold["X_test"][observed_test])
    partitioned = model.predict_georf(
        fold["X_test"][observed_test],
        fold["groups_test"][observed_test],
        s_branch,
    )
    metrics = pd.DataFrame(
        [
            {
                "year": target.year,
                "month": target.month,
                "f1(1)": f1_score(fold["y_test"][observed_test], partitioned, zero_division=0),
                "f1_base(1)": f1_score(fold["y_test"][observed_test], pooled, zero_division=0),
                "train_rows": len(fold["train_indices"]),
                "fit_rows": fold["fit_rows"],
                "validation_rows": fold["validation_rows"],
                "test_rows": len(fold["test_indices"]),
                "training_all_null_columns": json.dumps(
                    [
                        feature
                        for feature, median in zip(MODEL_PREDICTORS, fold["medians"])
                        if median == 0
                        and frame.iloc[fold["train_indices"]][feature].isna().all()
                    ]
                ),
            }
        ]
    )
    metrics.to_csv(
        Path.cwd() / f"results_df_gp_fs{scope}_{args.start_year}_{args.end_year}.csv",
        index=False,
    )
    (model_dir / "feature_names.txt").write_text(
        "\n".join(MODEL_PREDICTORS) + "\n",
        encoding="utf-8",
    )
    return correspondence_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--end_year", type=int, required=True)
    parser.add_argument("--forecasting_scope", type=int, required=True)
    parser.add_argument("--desired_terms", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    output = run_stage1(parse_args())
    print(f"Stage 1 correspondence: {output}")
