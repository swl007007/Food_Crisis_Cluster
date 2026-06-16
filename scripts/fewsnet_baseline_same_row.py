#!/usr/bin/env python3
"""
Recompute FEWSNET baseline results with same-row projection comparison.

This variant treats FEWSNET near- and medium-term projections in each row as
predictions for that row's target month:
- fs1 compares pred_near with crisis_actual
- fs2 compares pred_med with crisis_actual

It intentionally does not shift projections by 4 or 8 months.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_FEWSNET_DATA_PATH = (
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis"
    r"\1.Source Data\Outcome\FEWSNET_IPC\FEWSNET.csv"
)
DEFAULT_OUTPUT_DIR = "fewsnet_baseline_results_backup"


def month_to_quarter(month: int) -> int:
    """Convert a month number to a quarter number."""
    return (int(month) - 1) // 3 + 1


def crisis_indicator(values: pd.Series) -> pd.Series:
    """Convert FEWSNET IPC phase values to binary crisis indicators."""
    return (pd.to_numeric(values, errors="coerce") >= 3).astype(int)


def prepare_fewsnet_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare FEWSNET rows for same-row baseline evaluation."""
    df = raw_df.copy()

    if "FEWSNET_admin_code" not in df.columns and "admin_code" in df.columns:
        df["FEWSNET_admin_code"] = df["admin_code"]

    required_columns = {"year", "month", "fews_ipc", "fews_proj_near", "fews_proj_med"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required FEWSNET columns: {missing}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df = df.dropna(subset=["year", "month"]).copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    df["quarter"] = df["month"].apply(month_to_quarter)
    df["crisis_actual"] = crisis_indicator(df["fews_ipc"])
    df["pred_near"] = crisis_indicator(df["fews_proj_near"])
    df["pred_med"] = crisis_indicator(df["fews_proj_med"])

    sort_columns = [column for column in ["FEWSNET_admin_code", "year", "month"] if column in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns).reset_index(drop=True)

    return df


def evaluate_predictions_class1_only(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    """Calculate precision, recall, and F1 for crisis class 1 only."""
    valid = pd.notna(y_true) & pd.notna(y_pred)
    y_true_clean = pd.Series(y_true[valid]).astype(int)
    y_pred_clean = pd.Series(y_pred[valid]).astype(int)

    if len(y_true_clean) == 0:
        return {
            "precision(1)": 0.0,
            "recall(1)": 0.0,
            "f1(1)": 0.0,
            "num_samples(1)": 0,
        }

    true_positive = int(((y_true_clean == 1) & (y_pred_clean == 1)).sum())
    false_positive = int(((y_true_clean == 0) & (y_pred_clean == 1)).sum())
    false_negative = int(((y_true_clean == 1) & (y_pred_clean == 0)).sum())
    num_samples_1 = int((y_true_clean == 1).sum())

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "precision(1)": precision,
        "recall(1)": recall,
        "f1(1)": f1,
        "num_samples(1)": num_samples_1,
    }


def generate_quarters(start_year: int, end_year: int) -> list[tuple[int, int]]:
    """Generate inclusive year-quarter pairs."""
    return [(year, quarter) for year in range(start_year, end_year + 1) for quarter in range(1, 5)]


def evaluate_scope(
    df: pd.DataFrame,
    forecasting_scope: int,
    start_year: int = 2013,
    end_year: int = 2024,
) -> pd.DataFrame:
    """Evaluate same-row FEWSNET predictions for one forecasting scope."""
    if forecasting_scope == 1:
        pred_col = "pred_near"
    elif forecasting_scope == 2:
        pred_col = "pred_med"
    else:
        raise ValueError(f"FEWSNET same-row baseline supports scopes 1 and 2, got {forecasting_scope}")

    results = []
    for year, quarter in generate_quarters(start_year, end_year):
        quarter_data = df[(df["year"] == year) & (df["quarter"] == quarter)]
        if quarter_data.empty:
            continue

        metrics = evaluate_predictions_class1_only(
            quarter_data["crisis_actual"],
            quarter_data[pred_col],
        )
        metrics["year"] = year
        metrics["quarter"] = quarter
        results.append(metrics)

    columns = ["precision(1)", "recall(1)", "f1(1)", "num_samples(1)", "year", "quarter"]
    return pd.DataFrame(results, columns=columns)


def write_results(
    df: pd.DataFrame,
    output_dir: Path,
    scopes: list[int],
    start_year: int,
    end_year: int,
) -> None:
    """Write same-row baseline CSV files for the requested scopes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for scope in scopes:
        results_df = evaluate_scope(df, scope, start_year=start_year, end_year=end_year)
        output_file = output_dir / f"fewsnet_baseline_results_fs{scope}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Wrote {len(results_df)} rows to {output_file}")
        if not results_df.empty:
            print(f"  Average F1(1): {results_df['f1(1)'].mean():.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute FEWSNET baseline using same-row near/medium projections."
    )
    parser.add_argument("--input", default=DEFAULT_FEWSNET_DATA_PATH, help="Path to FEWSNET.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for output CSV files")
    parser.add_argument("--start-year", type=int, default=2013, help="First evaluation year")
    parser.add_argument("--end-year", type=int, default=2024, help="Last evaluation year")
    parser.add_argument("--forecasting-scope", type=int, choices=[1, 2], help="Run only one scope")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    scopes = [args.forecasting_scope] if args.forecasting_scope else [1, 2]

    print("=== FEWSNET Same-Row Baseline Evaluation ===")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Years: {args.start_year}-{args.end_year}")
    print("Comparison: fs1 pred_near vs crisis_actual; fs2 pred_med vs crisis_actual")

    raw_df = pd.read_csv(input_path)
    prepared_df = prepare_fewsnet_data(raw_df)
    prepared_df = prepared_df[prepared_df["year"] >= args.start_year]
    prepared_df = prepared_df[prepared_df["year"] <= args.end_year]

    write_results(
        prepared_df,
        output_dir=output_dir,
        scopes=scopes,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    print("=== Complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
