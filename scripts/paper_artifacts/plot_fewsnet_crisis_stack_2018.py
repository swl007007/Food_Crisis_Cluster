#!/usr/bin/env python3
"""Plot FEWSNET crisis/non-crisis class counts by quarter from 2018 onward."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path(
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data"
    r"\FEWSNET_forecast_unadjusted_bm.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "scripts" / "fewsnet_crisis_stack_2018.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stacked FEWSNET crisis/non-crisis counts by quarter for 2018 onward."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to FEWSNET_forecast_unadjusted_bm.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args()


def load_quarterly_counts(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, usecols=["date", "fews_ipc_crisis"])
    df = df.dropna(subset=["fews_ipc_crisis"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["fews_ipc_crisis"] = pd.to_numeric(df["fews_ipc_crisis"], errors="coerce")
    df = df.dropna(subset=["date", "fews_ipc_crisis"])
    df = df[df["fews_ipc_crisis"].isin([0, 1])].copy()
    df = df[df["date"].dt.year >= 2018].copy()
    if df.empty:
        raise ValueError("No non-missing fews_ipc_crisis records found for 2018 onward.")

    df["period"] = df["date"].dt.to_period("Q")
    counts = (
        df.groupby(["period", "fews_ipc_crisis"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0.0: "non_crisis", 1.0: "crisis"})
        .sort_index()
    )
    for column in ("crisis", "non_crisis"):
        if column not in counts.columns:
            counts[column] = 0
    return counts[["crisis", "non_crisis"]]


def plot_stack(counts: pd.DataFrame, output_path: Path, dpi: int) -> None:
    x = counts.index.to_timestamp("Q")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.stackplot(
        x,
        counts["crisis"],
        counts["non_crisis"],
        labels=["Crisis (value=1)", "Non-crisis (value=0)"],
        colors=["#d73027", "#1a9850"],
        alpha=0.95,
    )
    ax.set_title("Stacked FEWSNET crisis/non-crisis counts by quarter (2018 onward)")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left")
    ax.margins(x=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{period.year}-Q{period.quarter}" for period in counts.index], rotation=45, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    counts = load_quarterly_counts(args.input)
    plot_stack(counts, args.output, args.dpi)
    print(f"Rows plotted: {int(counts.sum().sum())}")
    print(f"Quarters: {counts.index.min()} to {counts.index.max()}")
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
