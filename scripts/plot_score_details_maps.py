#!/usr/bin/env python3
"""Generate choropleth maps for GeoRF score detail CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")  # Headless environments
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

import config_visual


def _coerce_path(path_str: str) -> Path:
    """Return a Path that works on both Windows and POSIX mounts."""
    path = Path(path_str)
    if path.exists():
        return path
    if len(path_str) >= 2 and path_str[1] == ":":
        drive = path_str[0].lower()
        rest = path_str[2:].replace("\\", "/")
        alt = Path("/mnt") / drive / rest.lstrip("/")
        if alt.exists():
            return alt
    return path


def _normalize_codes(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def load_polygons() -> gpd.GeoDataFrame:
    shape_path = _coerce_path(config_visual.ADJACENCY_SHAPEFILE_PATH)
    if not shape_path.exists():
        raise FileNotFoundError(f"Polygon shapefile not found at {shape_path}")
    polygons = gpd.read_file(shape_path)
    id_col = config_visual.ADJACENCY_POLYGON_ID_COLUMN
    if id_col not in polygons.columns:
        raise KeyError(
            f"Column '{id_col}' not found in shapefile. Available columns: {sorted(polygons.columns)}"
        )
    polygons = polygons.copy()
    polygons[id_col] = _normalize_codes(polygons[id_col])
    return polygons


def iter_score_files(vis_dir: Path) -> Iterable[Path]:
    pattern = "score_details_round_*_branch_*.csv"
    for path in sorted(vis_dir.glob(pattern)):
        if path.is_file():
            yield path


def format_title(csv_path: Path) -> str:
    stem = csv_path.stem
    return stem.replace("_", " ")


def plot_scores(polygons: gpd.GeoDataFrame, csv_path: Path, output_dir: Path, metrics: List[str]) -> Path:
    df = pd.read_csv(csv_path)
    if "FEWSNET_admin_code" not in df.columns:
        raise KeyError(
            f"Column 'FEWSNET_admin_code' required in {csv_path}. Found columns: {df.columns.tolist()}"
        )
    df = df.copy()
    df["FEWSNET_admin_code"] = _normalize_codes(df["FEWSNET_admin_code"])
    merged = polygons.merge(
        df,
        how="left",
        left_on=config_visual.ADJACENCY_POLYGON_ID_COLUMN,
        right_on="FEWSNET_admin_code",
        validate="1:1",
    )

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6), constrained_layout=True)
    if n_metrics == 1:
        axes = [axes]

    for metric, ax in zip(metrics, axes):
        if metric not in merged.columns:
            ax.axis("off")
            ax.set_title(f"Missing metric: {metric}")
            continue
        plot = merged.plot(
            column=metric,
            ax=ax,
            cmap="RdBu",
            legend=True,
            missing_kwds={
                "color": "lightgrey",
                "edgecolor": "white",
                "hatch": "///",
                "label": "No data",
            },
            linewidth=0.2,
            edgecolor="black",
        )
        plot.set_axis_off()
        ax.set_title(metric)

    fig.suptitle(format_title(csv_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{csv_path.stem}_map.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot score detail choropleths per round/branch.")
    parser.add_argument(
        "vis_dir",
        type=Path,
        nargs="?",
        default=Path("deliverables/GeoRF_Deliverables/result_GeoRF_minimal_dataset(high_depth)/vis"),
        help="Directory containing score_details_round_*_branch_*.csv files.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["gscore"],
        help="Columns to visualize from the score detail files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for saving plots (defaults to vis_dir/plots_score_details).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vis_dir = args.vis_dir.resolve()
    if not vis_dir.exists():
        print(f"Input directory not found: {vis_dir}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve() if args.output_dir else vis_dir / "plots_score_details"

    polygons = load_polygons()
    metrics = args.metrics

    generated = []
    for csv_path in iter_score_files(vis_dir):
        try:
            output_path = plot_scores(polygons, csv_path, output_dir, metrics)
            generated.append(output_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to plot {csv_path.name}: {exc}", file=sys.stderr)

    if not generated:
        print("No plots generated. Check input directory and metrics.", file=sys.stderr)
        return 1

    print("Generated plots:")
    for path in generated:
        print(f" - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
