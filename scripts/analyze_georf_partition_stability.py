#!/usr/bin/env python3
"""Analyze GeoRF Stage 1 partition stability across years, months, and horizons."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = REPO_ROOT / "GeoRFExperiment" / "linked_tables" / "main_index.csv"
DEFAULT_PARTITION_DIR = REPO_ROOT / "GeoRFExperiment" / "linked_tables" / "partitions"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final_artifacts_in_paper_updated"

INVALID_PARTITION_LABELS = {"", "nan", "none", "null", "na", "n/a", "s-1"}
HORIZON_MONTHS = {"fs1": 4, "fs2": 8, "fs3": 12}
COMPARISON_ORDER = ["across_years", "across_horizons", "across_months", "mixed"]


def normalize_identifier(series: pd.Series) -> pd.Series:
    """Normalize FEWSNET identifiers or partition labels for joins and filtering."""
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def valid_partition_mask(series: pd.Series) -> pd.Series:
    labels = normalize_identifier(series).str.lower()
    return ~labels.isin(INVALID_PARTITION_LABELS)


def normalize_partition_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"FEWSNET_admin_code", "partition_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Partition file missing required columns: {missing}")

    normalized = df[list(required)].copy()
    normalized["FEWSNET_admin_code"] = normalize_identifier(normalized["FEWSNET_admin_code"])
    normalized["partition_id"] = normalize_identifier(normalized["partition_id"])
    normalized = normalized[
        normalized["FEWSNET_admin_code"].ne("") & valid_partition_mask(normalized["partition_id"])
    ]
    return normalized.drop_duplicates("FEWSNET_admin_code").reset_index(drop=True)


def horizon_months(scope: object) -> int | None:
    return HORIZON_MONTHS.get(str(scope).strip())


def comparison_group(meta_a: dict, meta_b: dict) -> str:
    same_year = int(meta_a["year"]) == int(meta_b["year"])
    same_month = int(meta_a["month"]) == int(meta_b["month"])
    same_horizon = str(meta_a["forecasting_scope"]) == str(meta_b["forecasting_scope"])

    if same_month and same_horizon and not same_year:
        return "across_years"
    if same_year and same_month and not same_horizon:
        return "across_horizons"
    if same_year and same_horizon and not same_month:
        return "across_months"
    return "mixed"


def pairwise_stability_row(
    meta_a: dict,
    meta_b: dict,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> dict:
    left = normalize_partition_frame(df_a).rename(columns={"partition_id": "partition_a"})
    right = normalize_partition_frame(df_b).rename(columns={"partition_id": "partition_b"})
    merged = left.merge(right, on="FEWSNET_admin_code", how="inner")
    n_common = int(len(merged))

    if n_common < 2:
        ari = float("nan")
        nmi = float("nan")
    else:
        ari = float(adjusted_rand_score(merged["partition_a"], merged["partition_b"]))
        nmi = float(normalized_mutual_info_score(merged["partition_a"], merged["partition_b"]))

    return {
        "plan_a": meta_a["name"],
        "year_a": int(meta_a["year"]),
        "month_a": int(meta_a["month"]),
        "forecasting_horizon_months_a": horizon_months(meta_a["forecasting_scope"]),
        "source_scope_a": meta_a["forecasting_scope"],
        "plan_b": meta_b["name"],
        "year_b": int(meta_b["year"]),
        "month_b": int(meta_b["month"]),
        "forecasting_horizon_months_b": horizon_months(meta_b["forecasting_scope"]),
        "source_scope_b": meta_b["forecasting_scope"],
        "comparison_group": comparison_group(meta_a, meta_b),
        "n_common_valid": n_common,
        "adjusted_rand_index": ari,
        "normalized_mutual_information": nmi,
    }


def cluster_size_summary_row(meta: dict, df: pd.DataFrame) -> dict:
    normalized = normalize_partition_frame(df)
    sizes = normalized["partition_id"].value_counts().sort_values()
    n_polygons = int(sizes.sum())

    if sizes.empty:
        stats = {
            "n_clusters": 0,
            "min_cluster_size": 0,
            "p25_cluster_size": float("nan"),
            "median_cluster_size": float("nan"),
            "mean_cluster_size": float("nan"),
            "p75_cluster_size": float("nan"),
            "max_cluster_size": 0,
            "largest_cluster_share": float("nan"),
        }
    else:
        stats = {
            "n_clusters": int(len(sizes)),
            "min_cluster_size": int(sizes.min()),
            "p25_cluster_size": float(sizes.quantile(0.25)),
            "median_cluster_size": float(sizes.median()),
            "mean_cluster_size": float(sizes.mean()),
            "p75_cluster_size": float(sizes.quantile(0.75)),
            "max_cluster_size": int(sizes.max()),
            "largest_cluster_share": float(sizes.max() / n_polygons) if n_polygons else float("nan"),
        }

    return {
        "plan": meta["name"],
        "year": int(meta["year"]),
        "month": int(meta["month"]),
        "forecasting_horizon_months": horizon_months(meta["forecasting_scope"]),
        "source_scope": meta["forecasting_scope"],
        "n_polygons": n_polygons,
        **stats,
    }


def load_plan_index(index_path: Path) -> list[dict]:
    index = pd.read_csv(index_path)
    required = {"name", "variant", "year", "month", "forecasting_scope"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"{index_path} missing required columns: {missing}")

    index = index[index["variant"].eq("GeoRF")].copy()
    index["year"] = pd.to_numeric(index["year"], errors="raise").astype(int)
    index["month"] = pd.to_numeric(index["month"], errors="raise").astype(int)
    index["forecasting_horizon_months"] = index["forecasting_scope"].map(horizon_months)
    index = index.sort_values(["year", "month", "forecasting_horizon_months", "name"])
    return index.to_dict("records")


def load_partition_table(partition_dir: Path, plan_name: str) -> pd.DataFrame:
    path = partition_dir / f"{plan_name}_partition.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing partition file: {path}")
    return pd.read_csv(path)


def build_pairwise_table(plans: list[dict], partitions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [
        pairwise_stability_row(plan_a, plan_b, partitions[plan_a["name"]], partitions[plan_b["name"]])
        for plan_a, plan_b in itertools.combinations(plans, 2)
    ]
    return pd.DataFrame(rows)


def build_cluster_size_table(plans: list[dict], partitions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [cluster_size_summary_row(plan, partitions[plan["name"]]) for plan in plans]
    return pd.DataFrame(rows)


def summarize_pairwise_stability(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in COMPARISON_ORDER:
        sub = pairwise[pairwise["comparison_group"].eq(group)]
        if sub.empty:
            continue
        metric_sub = sub[sub["adjusted_rand_index"].notna() & sub["normalized_mutual_information"].notna()]
        row = {
            "comparison_group": group,
            "n_pairs_total": int(len(sub)),
            "n_pairs_with_metric": int(len(metric_sub)),
            "n_pairs_without_common_valid": int((sub["n_common_valid"] < 2).sum()),
            "n_common_valid_min": int(metric_sub["n_common_valid"].min()) if not metric_sub.empty else 0,
            "n_common_valid_median": float(metric_sub["n_common_valid"].median())
            if not metric_sub.empty
            else float("nan"),
            "n_common_valid_max": int(metric_sub["n_common_valid"].max()) if not metric_sub.empty else 0,
        }
        for metric in ("adjusted_rand_index", "normalized_mutual_information"):
            values = metric_sub[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else float("nan")
            row[f"{metric}_median"] = float(values.median()) if not values.empty else float("nan")
            row[f"{metric}_min"] = float(values.min()) if not values.empty else float("nan")
            row[f"{metric}_p25"] = float(values.quantile(0.25)) if not values.empty else float("nan")
            row[f"{metric}_p75"] = float(values.quantile(0.75)) if not values.empty else float("nan")
            row[f"{metric}_max"] = float(values.max()) if not values.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_size_records(plans: list[dict], partitions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for plan in plans:
        normalized = normalize_partition_frame(partitions[plan["name"]])
        sizes = normalized["partition_id"].value_counts().sort_index()
        for partition_id, size in sizes.items():
            records.append(
                {
                    "plan": plan["name"],
                    "year": int(plan["year"]),
                    "month": int(plan["month"]),
                    "forecasting_horizon_months": horizon_months(plan["forecasting_scope"]),
                    "source_scope": plan["forecasting_scope"],
                    "partition_id": partition_id,
                    "cluster_size": int(size),
                }
            )
    return pd.DataFrame(records)


def ordered_group_values(df: pd.DataFrame, value_column: str) -> list[pd.Series]:
    return [
        df.loc[df["comparison_group"].eq(group), value_column].dropna()
        for group in COMPARISON_ORDER
        if group in set(df["comparison_group"])
    ]


def render_figure(
    pairwise: pd.DataFrame,
    cluster_sizes: pd.DataFrame,
    output_path: Path,
    dpi: int = 300,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [group for group in COMPARISON_ORDER if group in set(pairwise["comparison_group"])]
    labels = [group.replace("_", "\n") for group in groups]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, metric, title in (
        (axes[0], "adjusted_rand_index", "Adjusted Rand index"),
        (axes[1], "normalized_mutual_information", "Normalized mutual information"),
    ):
        values = [pairwise.loc[pairwise["comparison_group"].eq(group), metric].dropna() for group in groups]
        ax.boxplot(values, tick_labels=labels, showmeans=True)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.25)

    horizons = sorted(cluster_sizes["forecasting_horizon_months"].dropna().unique())
    size_values = [
        cluster_sizes.loc[cluster_sizes["forecasting_horizon_months"].eq(horizon), "cluster_size"]
        for horizon in horizons
    ]
    axes[2].boxplot(size_values, tick_labels=[f"{int(h)} mo" for h in horizons], showmeans=True)
    axes[2].set_title("Stage 1 cluster-size distribution", fontweight="bold")
    axes[2].set_xlabel("Forecasting horizon / lag")
    axes[2].set_ylabel("Polygons per cluster")
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("GeoRF Stage 1 Partition Stability", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small GitHub-style Markdown table without optional dependencies."""
    if df.empty:
        return "_No rows._"

    text = df.copy()
    for column in text.columns:
        text[column] = text[column].map(
            lambda value: ""
            if pd.isna(value)
            else f"{value:.6g}"
            if isinstance(value, float)
            else str(value)
        )

    headers = [str(column) for column in text.columns]
    rows = text.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def metric_iqr_text(row: pd.Series, prefix: str) -> str:
    median = row[f"{prefix}_median"]
    p25 = row[f"{prefix}_p25"]
    p75 = row[f"{prefix}_p75"]
    return f"{median:.3f} [{p25:.3f}, {p75:.3f}]"


def build_appendix_stability_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Build a compact appendix-ready version of the partition-stability table."""
    labels = {
        "across_years": "Across years",
        "across_horizons": "Across horizons",
        "across_months": "Across months",
        "mixed": "Mixed pairings",
    }
    rows = []
    for _, row in summary.iterrows():
        total_pairs = int(row["n_pairs_total"])
        metric_pairs = int(row["n_pairs_with_metric"])
        rows.append(
            {
                "Comparison axis": labels.get(row["comparison_group"], str(row["comparison_group"])),
                "Total pairs": total_pairs,
                "Pairs used for metrics": f"{metric_pairs}/{total_pairs}",
                "Median common polygons": f"{row['n_common_valid_median']:.0f}",
                "ARI median [IQR]": metric_iqr_text(row, "adjusted_rand_index"),
                "NMI median [IQR]": metric_iqr_text(row, "normalized_mutual_information"),
            }
        )
    return pd.DataFrame(rows)


def write_note(output_dir: Path, summary: pd.DataFrame, cluster_summary: pd.DataFrame) -> Path:
    note_path = output_dir / "georf_stage1_partition_stability_note.md"
    summary_lines = markdown_table(summary)
    all_invalid = cluster_summary.loc[
        cluster_summary["n_polygons"].eq(0),
        ["plan", "year", "month", "forecasting_horizon_months", "source_scope", "n_polygons", "n_clusters"],
    ]
    all_invalid_lines = markdown_table(all_invalid)
    cluster_lines = (
        markdown_table(
            cluster_summary[
                [
                    "forecasting_horizon_months",
                    "n_clusters",
                    "n_polygons",
                    "median_cluster_size",
                    "largest_cluster_share",
                ]
            ]
            .groupby("forecasting_horizon_months", as_index=False)
            .agg(
                n_plans=("n_clusters", "size"),
                median_n_clusters=("n_clusters", "median"),
                median_n_polygons=("n_polygons", "median"),
                median_cluster_size=("median_cluster_size", "median"),
                median_largest_cluster_share=("largest_cluster_share", "median"),
            )
        )
    )
    note_path.write_text(
        "\n".join(
            [
                "# GeoRF Stage 1 Partition Stability Note",
                "",
                "中文审查说明：",
                "",
                "本诊断仅覆盖当前论文主模型 GeoRF 的第一阶段 partition plans，",
                "不纳入 GeoDT。GeoDT 在本研究中主要用于展开和检查 branch differences，",
                "不作为后续 partition stability 结论的默认模型。",
                "",
                "稳定性基于 `GeoRFExperiment/linked_tables/main_index.csv` 中列出的",
                "2018-2020 Stage 1 linked partition plans 计算。每一对 partition 先按",
                "`FEWSNET_admin_code` 取共同且有效的 polygon；`s-1`、空值和缺失标签",
                "被视为 out-of-scope，不进入 ARI/NMI 或 cluster-size 统计。",
                "",
                "Pairwise comparison 分为三类主轴：同一月份和同一 forecasting horizon / lag",
                "但不同年份为 across years；同一年和同一月份但不同 horizon 为 across horizons；",
                "同一年和同一 horizon 但不同月份为 across months。其他组合保留为 mixed，",
                "用于透明报告但不作为主要稳定性解释。",
                "",
                "Adjusted Rand index (ARI) 和 normalized mutual information (NMI) 都对",
                "cluster label permutation 不敏感，因此适合比较不同年度、月份和 horizon 下",
                "重新学习得到的 partition labels。Cluster-size distribution 用于检查是否存在",
                "少数超大 cluster 或大量极小 cluster 驱动的表观稳定性。",
                "",
                "两个 Stage 1 plans 在当前 linked partition 表中全部为 `s-1`，",
                "因此涉及这些 plans 的 pairwise ARI/NMI 被标记为不可计算，而不是解释为",
                "低稳定性。",
                "",
                "## Pairwise Stability Summary",
                "",
                summary_lines,
                "",
                "## All-Out-of-Scope Stage 1 Plans",
                "",
                all_invalid_lines,
                "",
                "## Cluster-Size Summary by Forecasting Horizon / Lag",
                "",
                cluster_lines,
                "",
                "Appendix text (English):",
                "",
                "We assessed the stability of the GeoRF Stage 1 partition plans using pairwise",
                "adjusted Rand index (ARI), normalized mutual information (NMI), and cluster-size",
                "distributions. For each pair of Stage 1 partition plans, polygons were aligned by",
                "FEWSNET administrative code, and invalid or out-of-scope assignments (`s-1`,",
                "blank, or missing labels) were excluded. Pairwise comparisons were grouped as",
                "across-year comparisons when month and forecasting horizon were fixed,",
                "across-horizon comparisons when year and month were fixed, and across-month",
                "comparisons when year and forecasting horizon were fixed. Remaining pairings",
                "were retained as mixed comparisons for transparency. Cluster-size summaries were",
                "computed within each Stage 1 plan after applying the same validity filter.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return note_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--partition-dir", type=Path, default=DEFAULT_PARTITION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plans = load_plan_index(args.index)
    partitions = {
        plan["name"]: load_partition_table(args.partition_dir, plan["name"])
        for plan in plans
    }

    pairwise = build_pairwise_table(plans, partitions)
    summary = summarize_pairwise_stability(pairwise)
    cluster_summary = build_cluster_size_table(plans, partitions)
    cluster_records = cluster_size_records(plans, partitions)

    pairwise.to_csv(output_dir / "georf_stage1_partition_stability_pairwise.csv", index=False)
    summary.to_csv(output_dir / "georf_stage1_partition_stability_summary.csv", index=False)
    cluster_summary.to_csv(output_dir / "georf_stage1_partition_cluster_sizes.csv", index=False)
    cluster_records.to_csv(output_dir / "georf_stage1_partition_cluster_size_distribution.csv", index=False)
    appendix = build_appendix_stability_table(summary)
    appendix.to_csv(output_dir / "georf_stage1_partition_stability_appendix_table.csv", index=False)
    (output_dir / "georf_stage1_partition_stability_appendix_table.md").write_text(
        "\n".join(
            [
                "# Appendix Table: GeoRF Stage 1 Partition Stability",
                "",
                markdown_table(appendix),
                "",
                "Note: ARI and NMI are computed on common FEWSNET administrative polygons with valid",
                "Stage 1 partition assignments; out-of-scope `s-1` assignments are excluded.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    render_figure(pairwise, cluster_records, output_dir / "georf_stage1_partition_stability.png", args.dpi)
    write_note(output_dir, summary, cluster_summary)

    print(f"Wrote GeoRF Stage 1 partition stability artifacts to {output_dir}")
    print(f"Plans analyzed: {len(plans)}")
    print(f"Pairwise comparisons: {len(pairwise)}")


if __name__ == "__main__":
    main()
