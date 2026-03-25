"""
GeoDT Results Visualization
============================
Visualizes precision, recall, and F1 metrics from results_df_dt_gp_fs*_2024_2024.csv
and prediction-level analysis from y_pred_test_dt_gp_fs*_2024_2024.csv.

Generates interactive HTML plots via Plotly:
1. Metric comparison (model vs baseline) across year-month, per scope
2. Heatmap of F1 scores across year × month × scope
3. Confusion matrix summary from y_pred_test files
4. Delta (model − baseline) bar chart

Usage:
    python visualize_dt_gp_fs_results.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "viz_results_output"

SCOPES = ["fs1", "fs2", "fs3"]
SCOPE_LABELS = {"fs1": "FS1 (4-month)", "fs2": "FS2 (8-month)", "fs3": "FS3 (12-month)"}
MONTH_LABELS = {2: "Feb", 6: "Jun", 10: "Oct"}

COLORS = {
    "precision": "#3b82f6",   # blue
    "recall": "#ef4444",      # red
    "f1": "#10b981",          # green
    "precision_base": "#93c5fd",  # light blue
    "recall_base": "#fca5a5",     # light red
    "f1_base": "#6ee7b7",         # light green
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results() -> dict[str, pd.DataFrame]:
    """Load results_df_dt_gp_fs*_2024_2024.csv files."""
    data = {}
    for scope in SCOPES:
        path = DATA_DIR / f"results_df_dt_gp_{scope}_2024_2024.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["year_month"] = df["year"].astype(str) + "-" + df["month"].map(MONTH_LABELS)
            data[scope] = df
    return data


def load_predictions() -> dict[str, pd.DataFrame]:
    """Load y_pred_test_dt_gp_fs*_2024_2024.csv files."""
    data = {}
    for scope in SCOPES:
        path = DATA_DIR / f"y_pred_test_dt_gp_{scope}_2024_2024.csv"
        if path.exists():
            data[scope] = pd.read_csv(path)
    return data


# ---------------------------------------------------------------------------
# 1. Metric Line Charts (Model vs Baseline) per scope
# ---------------------------------------------------------------------------

def build_metric_line_charts(all_results: dict[str, pd.DataFrame]) -> go.Figure:
    """3-row line chart: one row per scope, showing precision/recall/F1 for model & baseline."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[SCOPE_LABELS[s] for s in SCOPES],
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    for row_idx, scope in enumerate(SCOPES):
        if scope not in all_results:
            continue
        df = all_results[scope]
        r = row_idx + 1
        show_legend = (row_idx == 0)

        # Model metrics
        for metric, col, color in [
            ("Precision", "precision(1)", COLORS["precision"]),
            ("Recall", "recall(1)", COLORS["recall"]),
            ("F1", "f1(1)", COLORS["f1"]),
        ]:
            fig.add_trace(go.Scatter(
                x=df["year_month"], y=df[col],
                name=f"{metric} (Model)", mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=6),
                legendgroup=f"model_{metric}",
                showlegend=show_legend,
                hovertemplate=f"{metric}: %{{y:.3f}}<extra>{scope}</extra>",
            ), row=r, col=1)

        # Baseline metrics
        for metric, col, color in [
            ("Precision", "precision_base(1)", COLORS["precision_base"]),
            ("Recall", "recall_base(1)", COLORS["recall_base"]),
            ("F1", "f1_base(1)", COLORS["f1_base"]),
        ]:
            fig.add_trace(go.Scatter(
                x=df["year_month"], y=df[col],
                name=f"{metric} (Base)", mode="lines+markers",
                line=dict(color=color, width=1.5, dash="dash"),
                marker=dict(size=4, symbol="diamond"),
                legendgroup=f"base_{metric}",
                showlegend=show_legend,
                hovertemplate=f"{metric} (Base): %{{y:.3f}}<extra>{scope}</extra>",
            ), row=r, col=1)

    fig.update_layout(
        title=dict(text="GeoDT Model vs Baseline — Precision, Recall, F1 (Crisis Class)", font_size=16),
        height=900,
        width=1400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=60, r=30, t=120, b=60),
    )
    fig.update_yaxes(range=[0, 1], dtick=0.1)
    fig.update_xaxes(tickangle=45, tickfont_size=9)

    return fig


# ---------------------------------------------------------------------------
# 2. F1 Heatmap (year × month × scope)
# ---------------------------------------------------------------------------

def build_f1_heatmap(all_results: dict[str, pd.DataFrame]) -> go.Figure:
    """Side-by-side heatmaps: Model F1 vs Baseline F1 across year × month for each scope."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Model F1", "Baseline F1"],
        horizontal_spacing=0.12,
    )

    years = sorted(set().union(*(df["year"].unique() for df in all_results.values())))
    months = [2, 6, 10]

    for panel_idx, (f1_col, title) in enumerate(
        [("f1(1)", "Model"), ("f1_base(1)", "Baseline")]
    ):
        # Build matrix: rows = scope×year, cols = month
        row_labels = []
        matrix = []
        for scope in SCOPES:
            if scope not in all_results:
                continue
            df = all_results[scope]
            for year in years:
                row_labels.append(f"{scope.upper()} {year}")
                row_vals = []
                for month in months:
                    mask = (df["year"] == year) & (df["month"] == month)
                    val = df.loc[mask, f1_col].values
                    row_vals.append(val[0] if len(val) > 0 else np.nan)
                matrix.append(row_vals)

        col_labels = [MONTH_LABELS[m] for m in months]
        z = np.array(matrix)

        fig.add_trace(go.Heatmap(
            z=z,
            x=col_labels,
            y=row_labels,
            colorscale="RdYlGn",
            zmin=0.2, zmax=0.8,
            text=np.round(z, 3).astype(str),
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorbar=dict(
                title="F1",
                x=0.45 if panel_idx == 0 else 1.02,
                len=0.9,
            ),
            hovertemplate="F1: %{z:.3f}<br>%{y}<br>%{x}<extra></extra>",
        ), row=1, col=panel_idx + 1)

    fig.update_layout(
        title=dict(text="F1 Score Heatmap — Model vs Baseline (Crisis Class)", font_size=16),
        height=600,
        width=1100,
        margin=dict(l=120, r=80, t=80, b=40),
    )
    fig.update_yaxes(autorange="reversed", tickfont_size=10)

    return fig


# ---------------------------------------------------------------------------
# 3. Delta (Model − Baseline) Bar Chart
# ---------------------------------------------------------------------------

def build_delta_chart(all_results: dict[str, pd.DataFrame]) -> go.Figure:
    """Grouped bar chart showing F1 delta (model − baseline) per year-month, grouped by scope."""
    fig = go.Figure()

    bar_colors = {"fs1": "#3b82f6", "fs2": "#f59e0b", "fs3": "#8b5cf6"}

    for scope in SCOPES:
        if scope not in all_results:
            continue
        df = all_results[scope]
        delta_f1 = df["f1(1)"] - df["f1_base(1)"]

        fig.add_trace(go.Bar(
            x=df["year_month"],
            y=delta_f1,
            name=SCOPE_LABELS[scope],
            marker_color=bar_colors[scope],
            hovertemplate=f"{SCOPE_LABELS[scope]}<br>ΔF1: %{{y:+.3f}}<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    fig.update_layout(
        title=dict(text="F1 Delta (GeoDT Model − Baseline) — Positive = Model Better", font_size=16),
        xaxis=dict(title="Year-Month", tickangle=45),
        yaxis=dict(title="ΔF1", zeroline=True),
        barmode="group",
        height=500,
        width=1400,
        margin=dict(l=60, r=30, t=80, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    return fig


# ---------------------------------------------------------------------------
# 4. Confusion Matrix from y_pred_test
# ---------------------------------------------------------------------------

def build_confusion_matrices(all_preds: dict[str, pd.DataFrame]) -> go.Figure:
    """3×4 confusion matrix grid: rows = scope, cols = year."""
    years = sorted(set().union(*(df["year"].unique() for df in all_preds.values())))

    fig = make_subplots(
        rows=len(SCOPES), cols=len(years),
        subplot_titles=[
            f"{SCOPE_LABELS[s]} — {y}" for s in SCOPES for y in years
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    for row_idx, scope in enumerate(SCOPES):
        if scope not in all_preds:
            continue
        df = all_preds[scope]
        for col_idx, year in enumerate(years):
            sub = df[df["year"] == year]
            if sub.empty:
                continue

            y_true = sub["fews_ipc_crisis_true"].values
            y_pred = sub["fews_ipc_crisis_pred"].values

            # Build 2×2 confusion matrix
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            cm = np.array([[tn, fp], [fn, tp]])
            total = cm.sum()

            # Annotations
            text = [
                [f"TN={tn}<br>({tn/total*100:.1f}%)", f"FP={fp}<br>({fp/total*100:.1f}%)"],
                [f"FN={fn}<br>({fn/total*100:.1f}%)", f"TP={tp}<br>({tp/total*100:.1f}%)"],
            ]

            fig.add_trace(go.Heatmap(
                z=cm,
                x=["Pred 0", "Pred 1"],
                y=["True 0", "True 1"],
                colorscale=[[0, "#dbeafe"], [1, "#1e40af"]],
                showscale=False,
                text=text,
                texttemplate="%{text}",
                textfont=dict(size=9),
                hovertemplate="Count: %{z}<extra></extra>",
            ), row=row_idx + 1, col=col_idx + 1)

    fig.update_layout(
        title=dict(text="Confusion Matrices — GeoDT Model by Scope and Year", font_size=16),
        height=250 * len(SCOPES) + 100,
        width=300 * len(years) + 100,
        margin=dict(l=80, r=30, t=100, b=30),
    )
    fig.update_yaxes(autorange="reversed")

    return fig


# ---------------------------------------------------------------------------
# 5. Per-year Accuracy from y_pred_test
# ---------------------------------------------------------------------------

def build_accuracy_summary(all_preds: dict[str, pd.DataFrame],
                           all_results: dict[str, pd.DataFrame]) -> go.Figure:
    """Grouped bar chart of overall accuracy per year-month, per scope."""
    fig = go.Figure()

    bar_colors = {"fs1": "#3b82f6", "fs2": "#f59e0b", "fs3": "#8b5cf6"}

    for scope in SCOPES:
        if scope not in all_preds:
            continue
        df = all_preds[scope]
        x_labels = []
        accuracies = []

        for (year, month), grp in df.groupby(["year", "month"]):
            y_true = grp["fews_ipc_crisis_true"].values
            y_pred = grp["fews_ipc_crisis_pred"].values
            acc = (y_true == y_pred).mean()
            x_labels.append(f"{year}-{MONTH_LABELS[month]}")
            accuracies.append(acc)

        fig.add_trace(go.Bar(
            x=x_labels,
            y=accuracies,
            name=SCOPE_LABELS[scope],
            marker_color=bar_colors[scope],
            hovertemplate=f"{SCOPE_LABELS[scope]}<br>Accuracy: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Overall Accuracy by Year-Month and Scope", font_size=16),
        xaxis=dict(title="Year-Month", tickangle=45),
        yaxis=dict(title="Accuracy", range=[0, 1]),
        barmode="group",
        height=500,
        width=1400,
        margin=dict(l=60, r=30, t=80, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    return fig


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

def _write_index_html(n_scopes: int):
    """Generate an index.html linking to all visualizations."""
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>GeoDT Results Visualization</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
    h1 {{ color: #1e40af; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; }}
    h2 {{ color: #374151; margin-top: 30px; }}
    a {{ color: #2563eb; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .stats {{ background: #f3f4f6; padding: 12px 18px; border-radius: 8px; margin: 10px 0; }}
    ul {{ line-height: 1.8; }}
</style>
</head><body>
<h1>GeoDT Results Visualization</h1>
<div class="stats">
    <b>{n_scopes}</b> forecast scopes (fs1, fs2, fs3) &nbsp;|&nbsp;
    Years: 2021-2024 &nbsp;|&nbsp;
    Months: Feb, Jun, Oct
</div>

<h2>Metric Charts</h2>
<ul>
    <li><a href="metric_line_charts.html">Precision / Recall / F1 — Model vs Baseline (Line Charts)</a></li>
    <li><a href="f1_heatmap.html">F1 Score Heatmap — Model vs Baseline</a></li>
    <li><a href="delta_f1.html">F1 Delta (Model - Baseline) Bar Chart</a></li>
</ul>

<h2>Prediction Analysis</h2>
<ul>
    <li><a href="confusion_matrices.html">Confusion Matrices by Scope and Year</a></li>
    <li><a href="accuracy_summary.html">Overall Accuracy by Year-Month and Scope</a></li>
</ul>

<h2>Data Files</h2>
<ul>
    <li><a href="results_summary.csv">Consolidated Results Summary (CSV)</a></li>
</ul>
</body></html>"""

    out_path = OUTPUT_DIR / "index.html"
    with open(str(out_path), "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {DATA_DIR}")
    all_results = load_results()
    all_preds = load_predictions()
    print(f"Loaded {len(all_results)} result files, {len(all_preds)} prediction files")

    for scope, df in all_results.items():
        print(f"  {scope}: {len(df)} rows, years {sorted(df['year'].unique())}")

    # --- 1. Metric line charts ---
    print("\n[1/5] Building metric line charts...")
    fig = build_metric_line_charts(all_results)
    out_path = OUTPUT_DIR / "metric_line_charts.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  -> {out_path}")

    # --- 2. F1 Heatmap ---
    print("[2/5] Building F1 heatmap...")
    fig = build_f1_heatmap(all_results)
    out_path = OUTPUT_DIR / "f1_heatmap.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  -> {out_path}")

    # --- 3. Delta chart ---
    print("[3/5] Building delta F1 chart...")
    fig = build_delta_chart(all_results)
    out_path = OUTPUT_DIR / "delta_f1.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  -> {out_path}")

    # --- 4. Confusion matrices ---
    if all_preds:
        print("[4/5] Building confusion matrices...")
        fig = build_confusion_matrices(all_preds)
        out_path = OUTPUT_DIR / "confusion_matrices.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        print(f"  -> {out_path}")

        # --- 5. Accuracy summary ---
        print("[5/5] Building accuracy summary...")
        fig = build_accuracy_summary(all_preds, all_results)
        out_path = OUTPUT_DIR / "accuracy_summary.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        print(f"  -> {out_path}")
    else:
        print("[4-5/5] No prediction files found, skipping confusion matrix and accuracy.")

    # --- Summary CSV ---
    summary_rows = []
    for scope, df in all_results.items():
        df_copy = df.copy()
        df_copy["scope"] = scope
        df_copy["delta_f1"] = df_copy["f1(1)"] - df_copy["f1_base(1)"]
        summary_rows.append(df_copy)
    if summary_rows:
        summary = pd.concat(summary_rows, ignore_index=True)
        out_path = OUTPUT_DIR / "results_summary.csv"
        summary.to_csv(str(out_path), index=False)
        print(f"\n  Summary CSV -> {out_path}")

    # --- Index ---
    _write_index_html(len(all_results))

    print(f"\n[OK] All outputs in: {OUTPUT_DIR}")
    print(f"  Open {OUTPUT_DIR / 'index.html'} in a browser to navigate all visualizations.")


if __name__ == "__main__":
    main()
