"""
Decision Tree Rules Visualization Suite
========================================
Generates three interactive HTML visualizations from the dt_rules folder:

1. Feature Importance Heatmap  — cross-file comparison (36 files)
2. Sankey Top-N Paths          — per-file decision flow diagrams
3. Small-Multiple Dashboard    — year×month grid per forecasting scope

Usage:
    python visualize_dt_rules.py
    python visualize_dt_rules.py --top-n 30  --sankey-file dt_rules_2023_fs1_2023-06.csv
"""
from __future__ import annotations

import argparse
import ast
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RULES_DIR = Path(__file__).parent / "dt_rules"
OUTPUT_DIR = Path(__file__).parent / "viz_output"

YEARS = [2021, 2022, 2023, 2024]
SCOPES = ["fs1", "fs2", "fs3"]
MONTHS = ["02", "06", "10"]
MONTH_LABELS = {"02": "Feb", "06": "Jun", "10": "Oct"}

# Feature categories for grouping in heatmap
FEATURE_CATEGORIES = {
    "IPC Lags": re.compile(r"^fews_ipc"),
    "Climate - Temperature": re.compile(r"^Tair_"),
    "Climate - Rainfall": re.compile(r"^Rainf_"),
    "Vegetation - EVI": re.compile(r"^EVI"),
    "Vegetation - GPP": re.compile(r"^gpp_mean"),
    "Conflict - Fatalities": re.compile(r"^sum_fatalities"),
    "Conflict - Events": re.compile(r"^event_count"),
    "Conflict - Distance": re.compile(r"^distance_to_nearest_acled"),
    "Economic - WFP Price": re.compile(r"^WFP_Price"),
    "Economic - FAO Price": re.compile(r"^FAO_price"),
    "Economic - CPI/GDP": re.compile(r"^(Food_CPI|CPI|GDP|Food_food_inflation)"),
    "Geographic": re.compile(r"^(lat|lon|elevation|slope|ruggedness|distance_to_river)"),
    "Soil": re.compile(r"^sg_"),
    "Nightlight": re.compile(r"^nightlight"),
    "Land/Market": re.compile(r"^(market_|AEZ_|CC|crop|pop|range)"),
    "Other": re.compile(r".*"),  # catch-all, must be last
}

# Colour palette
CLASS_COLORS = {0: "#3b82f6", 1: "#ef4444"}       # blue=no-crisis, red=crisis
CLASS_LABELS = {0: "No Crisis (0)", 1: "Crisis (1)"}


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------

def parse_condition(cond_str: str) -> tuple[str, str, float]:
    """Parse a single condition like 'feature <= 0.5' → (feature, op, value)."""
    m = re.match(r"^(.+?)\s*(<=|>=|<|>|==)\s*(.+)$", cond_str.strip())
    if not m:
        return cond_str.strip(), "?", 0.0
    return m.group(1).strip(), m.group(2).strip(), float(m.group(3))


def extract_feature_name(cond_str: str) -> str:
    """Extract just the feature name from a condition string."""
    feat, _, _ = parse_condition(cond_str)
    return feat


def categorize_feature(feat: str) -> str:
    """Map a feature name to its category."""
    for cat, pattern in FEATURE_CATEGORIES.items():
        if pattern.match(feat):
            return cat
    return "Other"


def load_rules_file(filepath: Path) -> pd.DataFrame:
    """Load a single dt_rules CSV and parse conditions."""
    df = pd.read_csv(filepath)
    # Parse Conditions_List from JSON string to Python list
    df["conditions"] = df["Conditions_List"].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else []
    )
    df["n_conditions"] = df["conditions"].apply(len)
    df["features"] = df["conditions"].apply(
        lambda conds: [extract_feature_name(c) for c in conds]
    )
    return df


def load_all_rules() -> dict[str, pd.DataFrame]:
    """Load all 36 dt_rules CSVs. Key = filename stem."""
    data = {}
    for f in sorted(RULES_DIR.glob("dt_rules_*.csv")):
        if "manifest" in f.name:
            continue
        df = load_rules_file(f)
        data[f.stem] = df
    return data


def file_meta(stem: str) -> dict[str, Any]:
    """Extract year, scope, month from a filename stem like 'dt_rules_2021_fs1_2021-02'."""
    m = re.match(r"dt_rules_(\d{4})_(fs\d)_\d{4}-(\d{2})", stem)
    if m:
        return {"year": int(m.group(1)), "scope": m.group(2), "month": m.group(3)}
    return {"year": 0, "scope": "", "month": ""}


# ---------------------------------------------------------------------------
# 1. Feature Importance Heatmap
# ---------------------------------------------------------------------------

def build_feature_importance_heatmap(all_data: dict[str, pd.DataFrame]) -> go.Figure:
    """
    Heatmap: rows = features (grouped by category), cols = 36 files.
    Cell value = sample-weighted frequency of feature across rules.
    """
    # Gather all features across every file
    global_features: Counter = Counter()
    per_file_feature_counts: dict[str, Counter] = {}

    for stem, df in all_data.items():
        counts: Counter = Counter()
        for _, row in df.iterrows():
            weight = row["Samples"]
            for feat in row["features"]:
                counts[feat] += weight
                global_features[feat] += weight
        per_file_feature_counts[stem] = counts

    # Select top 60 features by total weighted count
    top_features = [f for f, _ in global_features.most_common(60)]

    # Categorize and sort
    cat_order = list(FEATURE_CATEGORIES.keys())
    top_features.sort(key=lambda f: (cat_order.index(categorize_feature(f)), f))

    # Build matrix
    file_stems = sorted(all_data.keys(), key=lambda s: (
        file_meta(s)["scope"], file_meta(s)["year"], file_meta(s)["month"]
    ))

    matrix = np.zeros((len(top_features), len(file_stems)))
    for j, stem in enumerate(file_stems):
        counts = per_file_feature_counts[stem]
        total_samples = sum(counts.values()) or 1
        for i, feat in enumerate(top_features):
            # Normalize: proportion of total weighted feature appearances
            matrix[i, j] = counts.get(feat, 0) / total_samples

    # Build column labels
    col_labels = []
    for stem in file_stems:
        meta = file_meta(stem)
        col_labels.append(f"{meta['year']}-{MONTH_LABELS[meta['month']]} {meta['scope']}")

    # Add category annotations to row labels
    row_labels = []
    prev_cat = ""
    for feat in top_features:
        cat = categorize_feature(feat)
        if cat != prev_cat:
            row_labels.append(f"【{cat}】 {feat}")
            prev_cat = cat
        else:
            row_labels.append(feat)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=col_labels,
        y=row_labels,
        colorscale="YlOrRd",
        colorbar=dict(title="Weighted<br>Proportion"),
        hovertemplate="Feature: %{y}<br>File: %{x}<br>Proportion: %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Feature Importance Across All DT Models (Top 60, Sample-Weighted)", font_size=16),
        xaxis=dict(title="Model Iteration (scope → year → month)", tickangle=45, tickfont_size=8),
        yaxis=dict(title="", tickfont_size=9, autorange="reversed"),
        height=max(900, len(top_features) * 18),
        width=1600,
        margin=dict(l=350, r=50, t=80, b=150),
    )

    return fig


def build_feature_importance_by_class(all_data: dict[str, pd.DataFrame]) -> go.Figure:
    """
    Side-by-side heatmaps: feature importance split by Class 0 vs Class 1.
    """
    global_features: Counter = Counter()
    per_class: dict[int, Counter] = {0: Counter(), 1: Counter()}
    per_file_class: dict[str, dict[int, Counter]] = {}

    for stem, df in all_data.items():
        per_file_class[stem] = {0: Counter(), 1: Counter()}
        for _, row in df.iterrows():
            cls = int(row["Class"])
            weight = row["Samples"]
            for feat in row["features"]:
                per_class[cls][feat] += weight
                per_file_class[stem][cls][feat] += weight
                global_features[feat] += weight

    top_features = [f for f, _ in global_features.most_common(40)]
    cat_order = list(FEATURE_CATEGORIES.keys())
    top_features.sort(key=lambda f: (cat_order.index(categorize_feature(f)), f))

    file_stems = sorted(all_data.keys(), key=lambda s: (
        file_meta(s)["scope"], file_meta(s)["year"], file_meta(s)["month"]
    ))
    col_labels = []
    for stem in file_stems:
        meta = file_meta(stem)
        col_labels.append(f"{meta['year']}-{MONTH_LABELS[meta['month']]} {meta['scope']}")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["No Crisis (Class 0)", "Crisis (Class 1)"],
        horizontal_spacing=0.05,
    )

    for cls_idx, cls in enumerate([0, 1]):
        matrix = np.zeros((len(top_features), len(file_stems)))
        for j, stem in enumerate(file_stems):
            counts = per_file_class[stem][cls]
            total = sum(counts.values()) or 1
            for i, feat in enumerate(top_features):
                matrix[i, j] = counts.get(feat, 0) / total

        fig.add_trace(go.Heatmap(
            z=matrix,
            x=col_labels,
            y=top_features,
            colorscale="YlOrRd" if cls == 0 else "YlGnBu",
            colorbar=dict(
                title="Prop.",
                x=0.47 if cls == 0 else 1.02,
                len=0.8,
            ),
            hovertemplate="Feature: %{y}<br>File: %{x}<br>Proportion: %{z:.4f}<extra></extra>",
        ), row=1, col=cls_idx + 1)

    fig.update_layout(
        title=dict(text="Feature Importance by Class (Top 40, Sample-Weighted)", font_size=16),
        height=max(800, len(top_features) * 20),
        width=2200,
        margin=dict(l=250, r=80, t=80, b=150),
    )
    fig.update_xaxes(tickangle=45, tickfont_size=7)
    fig.update_yaxes(tickfont_size=9, autorange="reversed")

    return fig


# ---------------------------------------------------------------------------
# 2. Sankey Top-N Paths
# ---------------------------------------------------------------------------

class TrieNode:
    """Node in a prefix tree for reconstructing the decision tree."""
    __slots__ = ("condition", "children", "leaf_class", "leaf_samples", "total_samples")

    def __init__(self, condition: str = "ROOT"):
        self.condition = condition
        self.children: dict[str, TrieNode] = {}
        self.leaf_class: int | None = None
        self.leaf_samples: int = 0
        self.total_samples: int = 0


def build_trie(df: pd.DataFrame, top_n: int = 30) -> TrieNode:
    """Build a prefix trie from the top N rules by sample count."""
    root = TrieNode("ROOT")
    df_top = df.nlargest(top_n, "Samples")

    for _, row in df_top.iterrows():
        node = root
        node.total_samples += row["Samples"]
        for cond in row["conditions"]:
            if cond not in node.children:
                node.children[cond] = TrieNode(cond)
            node = node.children[cond]
            node.total_samples += row["Samples"]
        # Mark leaf
        node.leaf_class = int(row["Class"])
        node.leaf_samples = row["Samples"]

    return root


def trie_to_sankey(root: TrieNode, max_depth: int = 8) -> go.Figure:
    """Convert a trie into a Sankey diagram up to max_depth levels."""
    labels: list[str] = []
    label_to_idx: dict[str, int] = {}
    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    colors: list[str] = []
    node_colors: list[str] = []

    def get_or_create(label: str, depth: int) -> int:
        # Use depth-prefixed key for uniqueness
        key = f"d{depth}:{label}"
        if key not in label_to_idx:
            label_to_idx[key] = len(labels)
            # Shorten condition for display
            display = _shorten_condition(label) if label != "ROOT" else "All Samples"
            labels.append(display)
            node_colors.append("rgba(100,100,100,0.5)")
        return label_to_idx[key]

    def traverse(node: TrieNode, parent_idx: int, depth: int):
        if depth >= max_depth:
            return
        for cond, child in sorted(node.children.items(),
                                   key=lambda x: -x[1].total_samples):
            child_idx = get_or_create(cond, depth)
            sources.append(parent_idx)
            targets.append(child_idx)
            values.append(child.total_samples)

            # Color the link by dominant class in subtree
            if child.leaf_class is not None:
                link_color = (
                    "rgba(239,68,68,0.4)" if child.leaf_class == 1
                    else "rgba(59,130,246,0.3)"
                )
            else:
                link_color = "rgba(180,180,180,0.25)"
            colors.append(link_color)

            # Update node color if leaf
            if child.leaf_class is not None:
                node_colors[child_idx] = (
                    "rgba(239,68,68,0.8)" if child.leaf_class == 1
                    else "rgba(59,130,246,0.7)"
                )
                # Append class label
                labels[child_idx] += f" >> {'CRISIS' if child.leaf_class == 1 else 'No Crisis'} ({child.leaf_samples:,})"

            traverse(child, child_idx, depth + 1)

    root_idx = get_or_create("ROOT", 0)
    traverse(root, root_idx, 1)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=labels,
            color=node_colors,
            hovertemplate="<b>%{label}</b><br>Samples: %{value:,}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors,
            hovertemplate="From: %{source.label}<br>To: %{target.label}<br>Samples: %{value:,}<extra></extra>",
        ),
    )])

    return fig


def _shorten_condition(cond: str, max_len: int = 45) -> str:
    """Shorten a condition string for display."""
    if len(cond) <= max_len:
        return cond
    feat, op, val = parse_condition(cond)
    # Truncate feature name if very long
    if len(feat) > 30:
        feat = feat[:27] + "..."
    return f"{feat} {op} {val:.2f}"


def build_sankey_for_file(df: pd.DataFrame, stem: str, top_n: int = 30,
                          max_depth: int = 8) -> go.Figure:
    """Build a Sankey diagram for a specific file's rules."""
    trie = build_trie(df, top_n=top_n)
    fig = trie_to_sankey(trie, max_depth=max_depth)

    meta = file_meta(stem)
    title_text = (
        f"Decision Tree Paths — {meta['year']} {meta['scope'].upper()} "
        f"{MONTH_LABELS[meta['month']]} (Top {top_n} rules, depth ≤ {max_depth})"
    )
    fig.update_layout(
        title=dict(text=title_text, font_size=15),
        font_size=10,
        height=800,
        width=1400,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Small-Multiple Dashboard
# ---------------------------------------------------------------------------

def build_small_multiples(all_data: dict[str, pd.DataFrame], scope: str) -> go.Figure:
    """
    4×3 grid (years × months) for one scope.
    Each cell shows:
      - Stacked bar: Class 0 vs Class 1 by rule count and sample count
      - Top-5 features at the first split level
    """
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=[
            f"{y} - {MONTH_LABELS[m]}" for y in YEARS for m in MONTHS
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    for row_idx, year in enumerate(YEARS):
        for col_idx, month in enumerate(MONTHS):
            stem = f"dt_rules_{year}_{scope}_{year}-{month}"
            r = row_idx + 1
            c = col_idx + 1

            if stem not in all_data:
                fig.add_annotation(
                    text="No data", row=r, col=c,
                    xref=f"x{(row_idx * 3 + col_idx + 1)}", yref=f"y{(row_idx * 3 + col_idx + 1)}",
                )
                continue

            df = all_data[stem]

            # Top-10 features by first-split frequency (level 1 conditions)
            first_splits: Counter = Counter()
            for _, row in df.iterrows():
                if row["conditions"]:
                    feat = extract_feature_name(row["conditions"][0])
                    first_splits[feat] += row["Samples"]

            top_feats = first_splits.most_common(10)
            feat_names = [f[0] for f in top_feats][::-1]
            feat_vals = [f[1] for f in top_feats][::-1]

            # Class breakdown
            class_counts = df.groupby("Class")["Samples"].sum()
            c0_samples = class_counts.get(0, 0)
            c1_samples = class_counts.get(1, 0)
            total = c0_samples + c1_samples
            c1_pct = (c1_samples / total * 100) if total > 0 else 0

            # Shorten feature names for display
            short_names = [n[:25] + "…" if len(n) > 25 else n for n in feat_names]

            fig.add_trace(
                go.Bar(
                    y=short_names,
                    x=feat_vals,
                    orientation="h",
                    marker_color=["#3b82f6"] * len(feat_names),
                    showlegend=False,
                    hovertemplate="%{y}: %{x:,} samples<extra></extra>",
                ),
                row=r, col=c,
            )

            # Add crisis % annotation
            fig.add_annotation(
                text=f"Crisis: {c1_pct:.1f}%<br>Rules: {len(df)}<br>Samples: {total:,}",
                xref=f"x{row_idx * 3 + col_idx + 1}" if (row_idx * 3 + col_idx) > 0 else "x",
                yref=f"y{row_idx * 3 + col_idx + 1}" if (row_idx * 3 + col_idx) > 0 else "y",
                x=max(feat_vals) * 0.95 if feat_vals else 0,
                y=0,
                showarrow=False,
                font=dict(size=9, color="#ef4444"),
                align="right",
                xanchor="right",
            )

    fig.update_layout(
        title=dict(
            text=f"Decision Tree Summary — {scope.upper()} (Top-10 Root Features per Model)",
            font_size=16,
        ),
        height=1200,
        width=1400,
        showlegend=False,
        margin=dict(l=180, r=30, t=80, b=30),
    )
    fig.update_xaxes(tickfont_size=8)
    fig.update_yaxes(tickfont_size=8)

    return fig


# ---------------------------------------------------------------------------
# Bonus: Cross-file summary statistics
# ---------------------------------------------------------------------------

def build_summary_stats(all_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a summary table across all files."""
    rows = []
    for stem, df in sorted(all_data.items()):
        meta = file_meta(stem)
        class_counts = df.groupby("Class")["Samples"].sum()
        c0 = class_counts.get(0, 0)
        c1 = class_counts.get(1, 0)
        total = c0 + c1

        # Top-3 features overall (weighted)
        feat_counter: Counter = Counter()
        for _, row in df.iterrows():
            for feat in row["features"]:
                feat_counter[feat] += row["Samples"]
        top3 = [f for f, _ in feat_counter.most_common(3)]

        rows.append({
            "Year": meta["year"],
            "Scope": meta["scope"],
            "Month": MONTH_LABELS[meta["month"]],
            "Rules": len(df),
            "Total Samples": total,
            "Crisis Samples": c1,
            "Crisis %": round(c1 / total * 100, 1) if total else 0,
            "Avg Depth": round(df["n_conditions"].mean(), 1),
            "Max Depth": df["n_conditions"].max(),
            "Top Feature 1": top3[0] if len(top3) > 0 else "",
            "Top Feature 2": top3[1] if len(top3) > 1 else "",
            "Top Feature 3": top3[2] if len(top3) > 2 else "",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DT Rules Visualization Suite")
    parser.add_argument("--top-n", type=int, default=30,
                        help="Number of top rules for Sankey diagram (default: 30)")
    parser.add_argument("--sankey-depth", type=int, default=8,
                        help="Max tree depth for Sankey diagram (default: 8)")
    parser.add_argument("--sankey-file", type=str, default=None,
                        help="Specific CSV file for Sankey (default: all files)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading rules from: {RULES_DIR}")
    all_data = load_all_rules()
    print(f"Loaded {len(all_data)} files, {sum(len(df) for df in all_data.values()):,} total rules")

    # --- 1. Feature Importance Heatmap ---
    print("\n[1/4] Building feature importance heatmap...")
    fig_heatmap = build_feature_importance_heatmap(all_data)
    out_path = OUTPUT_DIR / "feature_importance_heatmap.html"
    fig_heatmap.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  -> {out_path}")

    print("[1b/4] Building class-split heatmap...")
    fig_class = build_feature_importance_by_class(all_data)
    out_path = OUTPUT_DIR / "feature_importance_by_class.html"
    fig_class.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  -> {out_path}")

    # --- 2. Sankey Diagrams ---
    print(f"\n[2/4] Building Sankey diagrams (top_n={args.top_n}, depth={args.sankey_depth})...")
    if args.sankey_file:
        # Single file
        stem = Path(args.sankey_file).stem
        if stem in all_data:
            fig_s = build_sankey_for_file(all_data[stem], stem,
                                          top_n=args.top_n, max_depth=args.sankey_depth)
            out_path = OUTPUT_DIR / f"sankey_{stem}.html"
            fig_s.write_html(str(out_path), include_plotlyjs="cdn")
            print(f"  -> {out_path}")
        else:
            print(f"  X File not found: {stem}")
    else:
        # Generate for all files
        sankey_dir = OUTPUT_DIR / "sankey"
        sankey_dir.mkdir(parents=True, exist_ok=True)
        for stem, df in sorted(all_data.items()):
            fig_s = build_sankey_for_file(df, stem,
                                          top_n=args.top_n, max_depth=args.sankey_depth)
            out_path = sankey_dir / f"sankey_{stem}.html"
            fig_s.write_html(str(out_path), include_plotlyjs="cdn")
        print(f"  -> {len(all_data)} Sankey diagrams in {sankey_dir}/")

    # --- 3. Small-Multiple Dashboards ---
    print("\n[3/4] Building small-multiple dashboards...")
    for scope in SCOPES:
        fig_sm = build_small_multiples(all_data, scope)
        out_path = OUTPUT_DIR / f"dashboard_{scope}.html"
        fig_sm.write_html(str(out_path), include_plotlyjs="cdn")
        print(f"  -> {out_path}")

    # --- 4. Summary Statistics ---
    print("\n[4/4] Building summary statistics...")
    summary_df = build_summary_stats(all_data)
    out_path = OUTPUT_DIR / "summary_stats.csv"
    summary_df.to_csv(str(out_path), index=False)
    print(f"  -> {out_path}")
    print("\n" + summary_df.to_string(index=False))

    # --- Index page ---
    _write_index_html(all_data, args)

    print(f"\n[OK] All outputs in: {OUTPUT_DIR}")
    print(f"  Open {OUTPUT_DIR / 'index.html'} in a browser to navigate all visualizations.")


def _write_index_html(all_data: dict[str, pd.DataFrame], args):
    """Generate an index.html that links to all generated visualizations."""
    sankey_links = []
    for stem in sorted(all_data.keys()):
        meta = file_meta(stem)
        label = f"{meta['year']} {meta['scope'].upper()} {MONTH_LABELS[meta['month']]}"
        sankey_links.append(f'<li><a href="sankey/sankey_{stem}.html">{label}</a></li>')

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>DT Rules Visualization Suite</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
    h1 {{ color: #1e40af; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; }}
    h2 {{ color: #374151; margin-top: 30px; }}
    a {{ color: #2563eb; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .stats {{ background: #f3f4f6; padding: 12px 18px; border-radius: 8px; margin: 10px 0; }}
    ul {{ line-height: 1.8; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; }}
    .grid li {{ list-style: none; }}
</style>
</head><body>
<h1>🌍 DT Rules Visualization Suite</h1>
<div class="stats">
    <b>{len(all_data)}</b> model iterations &nbsp;|&nbsp;
    <b>{sum(len(df) for df in all_data.values()):,}</b> total rules &nbsp;|&nbsp;
    Years: {min(YEARS)}–{max(YEARS)} &nbsp;|&nbsp;
    Scopes: fs1, fs2, fs3 &nbsp;|&nbsp;
    Months: Feb, Jun, Oct
</div>

<h2>📊 Feature Importance Heatmaps</h2>
<ul>
    <li><a href="feature_importance_heatmap.html">Overall Feature Importance (Top 60)</a></li>
    <li><a href="feature_importance_by_class.html">Feature Importance by Class (Top 40, Side-by-Side)</a></li>
</ul>

<h2>🔀 Sankey Decision Paths (Top {args.top_n} rules, depth ≤ {args.sankey_depth})</h2>
<ul class="grid">
{"".join(sankey_links)}
</ul>

<h2>📋 Small-Multiple Dashboards (Year × Month)</h2>
<ul>
    <li><a href="dashboard_fs1.html">FS1 — 4-month forecast</a></li>
    <li><a href="dashboard_fs2.html">FS2 — 8-month forecast</a></li>
    <li><a href="dashboard_fs3.html">FS3 — 12-month forecast</a></li>
</ul>

<h2>📈 Summary</h2>
<ul>
    <li><a href="summary_stats.csv">Summary Statistics (CSV)</a></li>
</ul>
</body></html>"""

    out_path = OUTPUT_DIR / "index.html"
    with open(str(out_path), "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
