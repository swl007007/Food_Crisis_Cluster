from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import logit
from sklearn.metrics.pairwise import haversine_distances


N_ADMIN_CODES = 5718
SIGMA_DEGREES = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute similarity matrices from linked tables.")
    parser.add_argument("--experiment-dir", required=True, help="Experiment directory path")
    parser.add_argument(
        "--month",
        type=int,
        choices=[2, 6, 10],
        default=None,
        help="Optional month filter (2, 6, 10); omit for general matrix",
    )
    return parser.parse_args()


def resolve_experiment_dir(experiment_dir: str) -> Path:
    exp_dir = Path(experiment_dir).expanduser().resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    return exp_dir


def compute_plan_weights(main_df: pd.DataFrame) -> pd.DataFrame:
    required = {"f1(1)", "f1_base(1)"}
    missing = required - set(main_df.columns)
    if missing:
        raise ValueError(f"main_index.csv is missing required columns: {sorted(missing)}")

    eps = 1e-6
    f1 = np.clip(main_df["f1(1)"].to_numpy(dtype=np.float64), eps, 1 - eps)
    f1_base = np.clip(main_df["f1_base(1)"].to_numpy(dtype=np.float64), eps, 1 - eps)
    weights = logit(f1) - logit(f1_base)
    weights = np.maximum(weights, 0.0)

    out = main_df.copy()
    out["weight"] = weights
    return out


def load_partition_labels(partition_file: Path, n_admin_codes: int) -> NDArray[np.int32]:
    if not partition_file.exists():
        raise FileNotFoundError(f"Partition file not found: {partition_file}")

    partition_df = pd.read_csv(partition_file, dtype={"partition_id": "str"})
    required = {"FEWSNET_admin_code", "partition_id"}
    missing = required - set(partition_df.columns)
    if missing:
        raise ValueError(f"Partition file {partition_file} missing columns: {sorted(missing)}")

    partition_df = partition_df.sort_values("FEWSNET_admin_code").reset_index(drop=True)
    labels = np.full(n_admin_codes, -1, dtype=np.int32)

    unique_partitions = partition_df["partition_id"].astype(str).unique()
    partition_to_int: dict[str, int] = {
        pid: idx for idx, pid in enumerate(unique_partitions) if pid != "s-1"
    }

    for _, row in partition_df.iterrows():
        admin_code = int(row["FEWSNET_admin_code"])
        if admin_code < 0 or admin_code >= n_admin_codes:
            continue
        pid = str(row["partition_id"])
        if pid == "s-1":
            labels[admin_code] = -1
        else:
            labels[admin_code] = partition_to_int[pid]

    return labels


def accumulate_similarity(plans: list[dict[str, Any]], n_admin_codes: int) -> NDArray[np.float32]:
    similarity = np.zeros((n_admin_codes, n_admin_codes), dtype=np.float32)
    print(f"Computing co-grouping similarity matrix: {n_admin_codes} x {n_admin_codes}")

    for idx, plan in enumerate(plans, start=1):
        labels = plan["labels"]
        weight = float(plan["weight"])
        if weight <= 0:
            continue

        valid_labels = labels[labels != -1]
        if valid_labels.size == 0:
            continue

        unique_labels = np.unique(valid_labels)
        for label in unique_labels:
            member_idx = np.where(labels == label)[0]
            if member_idx.size == 0:
                continue
            similarity[np.ix_(member_idx, member_idx)] += weight

        if idx % 10 == 0 or idx == len(plans):
            print(f"  Processed {idx}/{len(plans)} plans")

    return similarity


def load_lat_lon(exp_dir: Path, n_admin_codes: int) -> NDArray[np.float64]:
    latlon_path = Path(r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_admin_code_lat_lon.csv")
    if not latlon_path.exists():
        raise FileNotFoundError(f"Missing lat/lon file: {latlon_path}")

    latlon_df = pd.read_csv(latlon_path)
    required = {"FEWSNET_admin_code", "lat", "lon"}
    missing = required - set(latlon_df.columns)
    if missing:
        raise ValueError(f"Lat/lon file missing columns: {sorted(missing)}")

    latlon_df = latlon_df.sort_values("FEWSNET_admin_code").reset_index(drop=True)
    coords = latlon_df[["lat", "lon"]].to_numpy(dtype=np.float64)
    if coords.shape[0] < n_admin_codes:
        raise ValueError(
            f"Lat/lon file has {coords.shape[0]} rows, expected at least {n_admin_codes}"
        )
    return coords[:n_admin_codes]


def gaussian_spatial_weight(
    lat_lon_data: NDArray[np.float64], sigma_degrees: float
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    coords_radians = np.radians(lat_lon_data)
    distances_radians = haversine_distances(coords_radians, coords_radians)
    distances_degrees = np.degrees(distances_radians)
    spatial_weight = np.exp(-(distances_degrees ** 2) / (2.0 * sigma_degrees ** 2))
    np.fill_diagonal(spatial_weight, 1.0)
    return spatial_weight.astype(np.float32), distances_degrees.astype(np.float32)


def normalize_matrix(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    max_value = float(matrix.max())
    if max_value <= 0:
        return matrix
    return (matrix / max_value).astype(np.float32)


def build_output_dir(exp_dir: Path, month: int | None) -> Path:
    if month is None:
        return exp_dir / "similarity_matrices"
    return exp_dir / f"similarity_matrices_m{month:02d}"


def main() -> int:
    args = parse_args()
    try:
        exp_dir = resolve_experiment_dir(args.experiment_dir)
        linked_dir = exp_dir / "linked_tables"
        main_index_path = linked_dir / "main_index.csv"
        partition_dir = linked_dir / "partitions"

        if not main_index_path.exists():
            raise FileNotFoundError(f"Missing main index: {main_index_path}")
        if not partition_dir.exists():
            raise FileNotFoundError(f"Missing partition directory: {partition_dir}")

        main_df = pd.read_csv(main_index_path)
        if "month" not in main_df.columns or "name" not in main_df.columns:
            raise ValueError("main_index.csv must contain columns: name, month")

        weighted_df = compute_plan_weights(main_df)
        if args.month is not None:
            weighted_df = weighted_df[weighted_df["month"].astype(int) == int(args.month)]
            print(f"Filtered plans to month {args.month:02d}: {len(weighted_df)} plans")
        else:
            print(f"Using all plans: {len(weighted_df)} plans")

        if weighted_df.empty:
            raise RuntimeError("No plans available after filtering")

        plans: list[dict[str, Any]] = []
        for idx, row in weighted_df.iterrows():
            name = str(row["name"])
            part_file = partition_dir / f"{name}_partition.csv"
            labels = load_partition_labels(part_file, N_ADMIN_CODES)
            plans.append({"name": name, "weight": float(row["weight"]), "labels": labels})
            if (len(plans) % 10 == 0) or (len(plans) == len(weighted_df)):
                print(f"  Loaded partition labels: {len(plans)}/{len(weighted_df)}")

        similarity_matrix = accumulate_similarity(plans, N_ADMIN_CODES)
        lat_lon_data = load_lat_lon(exp_dir, N_ADMIN_CODES)
        spatial_weight, distance_matrix = gaussian_spatial_weight(lat_lon_data, SIGMA_DEGREES)
        final_matrix = similarity_matrix * spatial_weight
        final_matrix = normalize_matrix(final_matrix)

        output_dir = build_output_dir(exp_dir, args.month)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_npz = (
            output_dir / "similarity_matrices.npz"
            if args.month is None
            else output_dir / f"similarity_matrices_m{args.month:02d}.npz"
        )

        np.savez_compressed(
            output_npz,
            similarity_matrix=similarity_matrix,
            spatial_weight=spatial_weight,
            final_matrix=final_matrix,
            distance_matrix=distance_matrix,
            lat_lon_data=lat_lon_data.astype(np.float32),
            sigma_degrees=np.float32(SIGMA_DEGREES),
            month_filter=-1 if args.month is None else int(args.month),
        )

        summary = {
            "n_admin_codes": int(N_ADMIN_CODES),
            "n_plans": int(len(plans)),
            "month_filter": None if args.month is None else int(args.month),
            "sigma_degrees": float(SIGMA_DEGREES),
            "similarity_nonzero": int(np.count_nonzero(similarity_matrix)),
            "final_nonzero": int(np.count_nonzero(final_matrix)),
            "final_max": float(final_matrix.max()),
            "final_mean": float(final_matrix.mean()),
        }
        summary_path = (
            output_dir / "summary_statistics.json"
            if args.month is None
            else output_dir / f"summary_statistics_m{args.month:02d}.json"
        )
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved matrix file: {output_npz}")
        print(f"Saved summary file: {summary_path}")
        print("Step 4 similarity matrix completed")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
