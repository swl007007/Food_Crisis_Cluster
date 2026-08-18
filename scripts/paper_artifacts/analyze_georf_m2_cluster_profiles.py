"""Helpers for GeoRF m2 cluster profile diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = (
    REPO_ROOT
    / "result_partition_k40_compare_GF_fs1"
    / "refined"
    / "cluster_mapping_k40_nc13_m2_refined_contig3.csv"
)
DEFAULT_PREDICTIONS = REPO_ROOT / "result_partition_k40_compare_GF_fs1" / "predictions_monthly.csv"
DEFAULT_PANEL = Path(
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data"
    r"\FEWSNET_forecast_unadjusted_bm.csv"
)
DEFAULT_SHAPEFILE = Path(
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data"
    r"\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final_artifacts_in_paper_updated"

MARKET_COLUMNS = ["market_access", "market_distance"]
CONFLICT_COLUMNS = [
    "distance_to_nearest_acled",
    "event_count_battles",
    "event_count_explosions",
    "event_count_violence",
    "sum_fatalities_battles",
    "sum_fatalities_explosions",
    "sum_fatalities_violence",
    "event_count_battles_w5",
    "event_count_explosions_w5",
    "event_count_violence_w5",
    "sum_fatalities_battles_w5",
    "sum_fatalities_explosions_w5",
    "sum_fatalities_violence_w5",
    "event_count_battles_w10",
    "event_count_explosions_w10",
    "event_count_violence_w10",
    "sum_fatalities_battles_w10",
    "sum_fatalities_explosions_w10",
    "sum_fatalities_violence_w10",
]
ERROR_SUMMARY_COLUMNS = [
    "cluster_id",
    "n_observations",
    "crisis_prevalence",
    "tp_count",
    "fp_count",
    "fn_count",
    "tn_count",
    "main_error_mode",
    "tp_share",
    "fp_share",
    "fn_share",
    "tn_share",
]
DOMINANT_AEZ_COLUMNS = ["cluster_id", "dominant_aez", "dominant_aez_share"]
COUNTRY_REGION_SUMMARY_COLUMNS = [
    "cluster_id",
    "n_polygons",
    "n_countries",
    "dominant_region",
    "countries_or_regions_included",
]
INTERCLUSTER_SIMILARITY_COLUMNS = ["profile_type", "cluster_i", "cluster_j", "similarity"]
COHESION_COLUMNS = ["profile_type", "cluster_id", "cohesion"]
ERROR_SHARE_COLUMNS = ["tp_share", "fp_share", "fn_share", "tn_share"]


def resolve_path(path: Path) -> Path:
    """Convert paths between WSL and Windows Python conventions."""
    raw_path = str(path)

    if os.name == "nt":
        match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", raw_path)
        if match:
            drive, rest = match.groups()
            windows_rest = rest.replace("/", "\\")
            return Path(f"{drive.upper()}:\\{windows_rest}")
        return Path(raw_path)

    match = re.match(r"^([a-zA-Z]):[\\/](.*)$", raw_path)
    if match:
        drive, rest = match.groups()
        posix_rest = rest.replace("\\", "/")
        return Path("/mnt") / drive.lower() / posix_rest

    return Path(raw_path)


def normalize_admin_code(series: pd.Series) -> pd.Series:
    """Return FEWSNET admin codes as stripped string identifiers."""
    normalized = series.astype("string").str.strip()
    return normalized.str.replace(r"\.0$", "", regex=True)


def load_cluster_mapping(path: Path) -> pd.DataFrame:
    """Load and normalize a cluster mapping CSV."""
    mapping = pd.read_csv(resolve_path(path))
    mapping = mapping.copy()
    mapping["FEWSNET_admin_code"] = normalize_admin_code(mapping["FEWSNET_admin_code"])
    return mapping


def attach_clusters(records: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Attach cluster IDs to records using normalized FEWSNET admin codes."""
    records_norm = records.copy()
    mapping_norm = mapping.copy()
    records_norm["FEWSNET_admin_code"] = normalize_admin_code(records_norm["FEWSNET_admin_code"])
    mapping_norm["FEWSNET_admin_code"] = normalize_admin_code(mapping_norm["FEWSNET_admin_code"])
    return records_norm.merge(mapping_norm, on="FEWSNET_admin_code", how="inner")


def available_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return candidate columns present in the DataFrame, preserving candidate order."""
    return [column for column in candidates if column in df.columns]


def aez_columns(df: pd.DataFrame) -> list[str]:
    """Return AEZ one-hot columns in stable order."""
    return sorted(column for column in df.columns if column.startswith("AEZ_"))


def build_dominant_aez(panel_with_clusters: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Find the dominant AEZ one-hot column for each cluster and its within-cluster share."""
    present = available_columns(panel_with_clusters, columns)
    clustered = panel_with_clusters.loc[panel_with_clusters["cluster_id"].notna()].copy()
    if clustered.empty or not present:
        return pd.DataFrame(columns=DOMINANT_AEZ_COLUMNS)

    rows = []
    for cluster_id, sub in clustered.groupby("cluster_id"):
        means = sub[present].apply(pd.to_numeric, errors="coerce").mean().fillna(0.0)
        dominant = str(means.sort_values(ascending=False, kind="mergesort").index[0])
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "dominant_aez": dominant,
                "dominant_aez_share": float(means[dominant]),
            }
        )
    return pd.DataFrame(rows, columns=DOMINANT_AEZ_COLUMNS).sort_values("cluster_id").reset_index(drop=True)


def tertile_label(value: float) -> str:
    """Map a percentile-like value to low, moderate, or high."""
    if pd.isna(value):
        return "unknown"
    if value >= 2 / 3:
        return "high"
    if value >= 1 / 3:
        return "moderate"
    return "low"


def cluster_percentile(series: pd.Series) -> pd.Series:
    """Return 0..1 percentile positions for cluster-level values."""
    numeric = pd.to_numeric(series, errors="coerce")
    values = sorted(numeric.dropna().unique())
    if len(values) <= 1:
        return pd.Series(0.5, index=series.index, dtype="float64")

    denominator = len(values) - 1
    value_to_percentile = {value: idx / denominator for idx, value in enumerate(values)}
    return numeric.map(value_to_percentile).fillna(0.5).astype("float64")


def build_market_profile(panel_with_clusters: pd.DataFrame) -> pd.DataFrame:
    """Summarize available market-access variables by cluster."""
    return build_cluster_feature_profile(panel_with_clusters, MARKET_COLUMNS, "market")


def build_conflict_profile(panel_with_clusters: pd.DataFrame) -> pd.DataFrame:
    """Summarize available conflict variables by cluster."""
    return build_cluster_feature_profile(panel_with_clusters, CONFLICT_COLUMNS, "conflict")


def build_cluster_feature_profile(panel_with_clusters: pd.DataFrame, columns: list[str], prefix: str) -> pd.DataFrame:
    """Build mean, percentile, and tertile labels for cluster-level feature columns."""
    clustered = panel_with_clusters.loc[panel_with_clusters["cluster_id"].notna()].copy()
    base_columns = ["cluster_id", "n_observations"]
    if clustered.empty:
        return pd.DataFrame(columns=base_columns)

    present = available_columns(clustered, columns)
    for column in present:
        clustered[column] = pd.to_numeric(clustered[column], errors="coerce")
    counts = clustered.groupby("cluster_id").size().rename("n_observations").reset_index()
    profile = counts.sort_values("cluster_id").reset_index(drop=True)
    if present:
        means = clustered.groupby("cluster_id")[present].mean().reset_index()
        profile = profile.merge(means, on="cluster_id", how="left")
    profile["cluster_id"] = profile["cluster_id"].astype(int)

    for column in columns:
        mean_column = f"{prefix}_{column}_mean"
        percentile_column = f"{prefix}_{column}_percentile"
        label_column = f"{prefix}_{column}_label"
        if column in profile.columns:
            profile = profile.rename(columns={column: mean_column})
            profile[percentile_column] = cluster_percentile(profile[mean_column])
        else:
            profile[mean_column] = np.nan
            profile[percentile_column] = 0.5
        profile[label_column] = profile[percentile_column].map(tertile_label)

    return profile


def load_region_map() -> dict[str, str]:
    """Load REGION_MAP from the region prevalence script using an absolute file path."""
    module_path = REPO_ROOT / "scripts" / "paper_artifacts" / "plot_region_class_prevalence.py"
    spec = importlib.util.spec_from_file_location("plot_region_class_prevalence_for_profiles", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import region map from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.REGION_MAP)


def load_region_lookup(shapefile_path: Path) -> pd.DataFrame:
    """Load FEWSNET admin-code to country and region lookup from the boundary shapefile."""
    import geopandas as gpd

    region_map = load_region_map()

    gdf = gpd.read_file(resolve_path(shapefile_path))
    for column in ("FEWSNET_admin_code", "uid", "admin_code", "adm_code", "FNID"):
        if column in gdf.columns:
            gdf = gdf.rename(columns={column: "FEWSNET_admin_code"})
            break
    if "FEWSNET_admin_code" not in gdf.columns:
        raise ValueError(f"No FEWSNET admin-code column found in {shapefile_path}")
    if "ADMIN0" not in gdf.columns:
        raise ValueError(f"ADMIN0 column not found in {shapefile_path}")

    lookup = gdf[["FEWSNET_admin_code", "ADMIN0"]].copy()
    lookup["FEWSNET_admin_code"] = normalize_admin_code(lookup["FEWSNET_admin_code"])
    lookup["region"] = lookup["ADMIN0"].map(region_map).fillna("Other")
    return lookup.drop_duplicates("FEWSNET_admin_code").reset_index(drop=True)


def build_country_region_summary(mapping: pd.DataFrame, region_lookup: pd.DataFrame) -> pd.DataFrame:
    """Summarize country or region composition for each cluster."""
    attached = attach_clusters(region_lookup, mapping)
    attached = attached.loc[attached["cluster_id"].notna()].copy()
    if attached.empty:
        return pd.DataFrame(columns=COUNTRY_REGION_SUMMARY_COLUMNS)

    rows = []
    for cluster_id, sub in attached.groupby("cluster_id"):
        countries = sorted(str(country) for country in sub["ADMIN0"].dropna().unique())
        region_counts = sub["region"].fillna("Other").value_counts().sort_index()
        dominant_region = str(region_counts.sort_values(ascending=False, kind="mergesort").index[0])
        if len(countries) <= 8:
            summary = ", ".join(countries)
        else:
            summary = "; ".join(f"{region}: {int(count)}" for region, count in region_counts.items())
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_polygons": int(len(sub)),
                "n_countries": int(len(countries)),
                "dominant_region": dominant_region,
                "countries_or_regions_included": summary,
            }
        )
    return (
        pd.DataFrame(rows, columns=COUNTRY_REGION_SUMMARY_COLUMNS)
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )


def filter_february_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep February target months and standardize label columns."""
    filtered = predictions.copy()
    filtered["month_start"] = pd.to_datetime(filtered["month_start"])
    filtered = filtered.loc[filtered["month_start"].dt.month == 2].copy()
    filtered["target_month"] = filtered["month_start"].dt.strftime("%Y-%m")
    filtered["y_true"] = pd.to_numeric(filtered["y_true"], errors="coerce")
    filtered["y_pred_partitioned"] = pd.to_numeric(filtered["y_pred_partitioned"], errors="coerce")
    return filtered


def add_error_modes(df: pd.DataFrame) -> pd.DataFrame:
    """Label binary prediction outcomes as TP, FP, FN, or TN."""
    out = df.copy()
    y_true = pd.to_numeric(out["y_true"], errors="coerce")
    y_pred = pd.to_numeric(out["y_pred_partitioned"], errors="coerce")
    valid_binary = y_true.isin([0, 1]) & y_pred.isin([0, 1])
    conditions = [
        valid_binary & y_true.eq(1) & y_pred.eq(1),
        valid_binary & y_true.eq(0) & y_pred.eq(1),
        valid_binary & y_true.eq(1) & y_pred.eq(0),
        valid_binary & y_true.eq(0) & y_pred.eq(0),
    ]
    out["error_mode"] = np.select(conditions, ["TP", "FP", "FN", "TN"], default="invalid")
    return out


def dominant_non_tn_mode(counts: dict[str, int]) -> str:
    """Return the dominant non-TN mode, with sorted slash-joined ties."""
    non_tn = {key: counts.get(key, 0) for key in ("FN", "FP", "TP")}
    max_count = max(non_tn.values()) if non_tn else 0
    if max_count == 0:
        return "TN-dominant"
    winners = sorted([key for key, value in non_tn.items() if value == max_count])
    return "/".join(winners)


def build_error_summary(predictions_with_clusters: pd.DataFrame) -> pd.DataFrame:
    """Summarize prediction error modes by cluster."""
    labeled = add_error_modes(predictions_with_clusters)
    labeled = labeled.loc[labeled["cluster_id"].notna()].copy()
    if labeled.empty:
        return pd.DataFrame(columns=ERROR_SUMMARY_COLUMNS)

    rows = []
    for cluster_id, sub in labeled.groupby("cluster_id"):
        counts = sub["error_mode"].value_counts().to_dict()
        n = int(len(sub))
        y_true = pd.to_numeric(sub["y_true"], errors="coerce")
        row = {
            "cluster_id": int(cluster_id),
            "n_observations": n,
            "crisis_prevalence": float(y_true.mean()) if n else 0.0,
            "tp_count": int(counts.get("TP", 0)),
            "fp_count": int(counts.get("FP", 0)),
            "fn_count": int(counts.get("FN", 0)),
            "tn_count": int(counts.get("TN", 0)),
            "main_error_mode": dominant_non_tn_mode(counts),
        }
        for mode in ("TP", "FP", "FN", "TN"):
            row[f"{mode.lower()}_share"] = row[f"{mode.lower()}_count"] / n if n else 0.0
        rows.append(row)
    return pd.DataFrame(rows, columns=ERROR_SUMMARY_COLUMNS).sort_values("cluster_id").reset_index(drop=True)


def standardized_feature_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return numeric feature columns centered and scaled with safe missing-value handling."""
    present = available_columns(df, columns)
    if df.empty or not present:
        return pd.DataFrame(columns=present, index=df.index, dtype="float64")

    values = df[present].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median(numeric_only=True)).fillna(0.0)
    std = values.std(ddof=0).replace(0, 1.0)
    return ((values - values.mean()) / std).astype("float64")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity, treating identical zero vectors as perfectly similar."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    denom = norm_a * norm_b
    if denom == 0:
        return 1.0 if norm_a == 0 and norm_b == 0 else 0.0
    value = float(np.dot(a, b) / denom)
    return max(-1.0, min(1.0, value))


def build_intercluster_similarity(df: pd.DataFrame, columns: list[str], profile_type: str) -> pd.DataFrame:
    """Build long-format pairwise cosine similarity between cluster centroids."""
    present = available_columns(df, columns)
    if "cluster_id" not in df.columns or not present:
        return pd.DataFrame(columns=INTERCLUSTER_SIMILARITY_COLUMNS)

    clustered = df.loc[df["cluster_id"].notna()].copy()
    if clustered.empty:
        return pd.DataFrame(columns=INTERCLUSTER_SIMILARITY_COLUMNS)

    features = standardized_feature_frame(clustered, present)
    tmp = pd.concat([clustered[["cluster_id"]].reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    centroids = tmp.groupby("cluster_id")[present].mean().sort_index()
    if centroids.empty:
        return pd.DataFrame(columns=INTERCLUSTER_SIMILARITY_COLUMNS)

    rows = []
    for cluster_i, vec_i in centroids.iterrows():
        for cluster_j, vec_j in centroids.iterrows():
            rows.append(
                {
                    "profile_type": profile_type,
                    "cluster_i": int(cluster_i),
                    "cluster_j": int(cluster_j),
                    "similarity": cosine(vec_i.to_numpy(dtype=float), vec_j.to_numpy(dtype=float)),
                }
            )
    return pd.DataFrame(rows, columns=INTERCLUSTER_SIMILARITY_COLUMNS)


def build_feature_cohesion(df: pd.DataFrame, columns: list[str], profile_type: str) -> pd.DataFrame:
    """Build cluster-level average within-cluster cosine cohesion on a 0..1 scale."""
    present = available_columns(df, columns)
    if "cluster_id" not in df.columns or not present:
        return pd.DataFrame(columns=COHESION_COLUMNS)

    clustered = df.loc[df["cluster_id"].notna()].copy()
    if clustered.empty:
        return pd.DataFrame(columns=COHESION_COLUMNS)

    features = standardized_feature_frame(clustered, present)
    tmp = pd.concat([clustered[["cluster_id"]].reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    centroids = tmp.groupby("cluster_id")[present].mean()
    if centroids.empty:
        return pd.DataFrame(columns=COHESION_COLUMNS)

    rows = []
    for cluster_id, sub in tmp.groupby("cluster_id"):
        centroid = centroids.loc[cluster_id].to_numpy(dtype=float)
        similarities = [
            max(0.0, min(1.0, (cosine(row.to_numpy(dtype=float), centroid) + 1.0) / 2.0))
            for _, row in sub[present].iterrows()
        ]
        rows.append(
            {
                "profile_type": profile_type,
                "cluster_id": int(cluster_id),
                "cohesion": float(np.mean(similarities)) if similarities else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=COHESION_COLUMNS).sort_values("cluster_id").reset_index(drop=True)


def build_error_similarity(error_summary: pd.DataFrame) -> pd.DataFrame:
    """Build long-format pairwise cosine similarity between cluster error-share vectors."""
    present = available_columns(error_summary, ERROR_SHARE_COLUMNS)
    if error_summary.empty or "cluster_id" not in error_summary.columns or present != ERROR_SHARE_COLUMNS:
        return pd.DataFrame(columns=INTERCLUSTER_SIMILARITY_COLUMNS)

    data = error_summary.loc[error_summary["cluster_id"].notna(), ["cluster_id", *ERROR_SHARE_COLUMNS]].copy()
    if data.empty:
        return pd.DataFrame(columns=INTERCLUSTER_SIMILARITY_COLUMNS)
    for column in ERROR_SHARE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    data = data.set_index("cluster_id")[ERROR_SHARE_COLUMNS].sort_index()

    rows = []
    for cluster_i, vec_i in data.iterrows():
        for cluster_j, vec_j in data.iterrows():
            rows.append(
                {
                    "profile_type": "error_mode",
                    "cluster_i": int(cluster_i),
                    "cluster_j": int(cluster_j),
                    "similarity": cosine(vec_i.to_numpy(dtype=float), vec_j.to_numpy(dtype=float)),
                }
            )
    return pd.DataFrame(rows, columns=INTERCLUSTER_SIMILARITY_COLUMNS)


def build_error_cohesion(labeled_predictions: pd.DataFrame) -> pd.DataFrame:
    """Build cluster-level cohesion as the dominant error-mode share."""
    required = {"cluster_id", "error_mode"}
    if labeled_predictions.empty or not required.issubset(labeled_predictions.columns):
        return pd.DataFrame(columns=COHESION_COLUMNS)

    labeled = labeled_predictions.loc[labeled_predictions["cluster_id"].notna()].copy()
    if labeled.empty:
        return pd.DataFrame(columns=COHESION_COLUMNS)

    rows = []
    for cluster_id, sub in labeled.groupby("cluster_id"):
        shares = sub["error_mode"].value_counts(normalize=True)
        rows.append(
            {
                "profile_type": "error_mode",
                "cluster_id": int(cluster_id),
                "cohesion": float(shares.max()) if not shares.empty else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=COHESION_COLUMNS).sort_values("cluster_id").reset_index(drop=True)


def upper_triangle_mask(matrix: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean mask for duplicate cells above a square matrix diagonal."""
    mask = np.triu(np.ones(matrix.shape, dtype=bool), k=1)
    return pd.DataFrame(mask, index=matrix.index, columns=matrix.columns)


def build_market_access_profile_label(market: pd.DataFrame) -> pd.Series:
    """Build compact market-access labels from the detailed market profile."""
    access = market.get("market_market_access_label", pd.Series("unknown", index=market.index))
    distance = market.get("market_market_distance_label", pd.Series("unknown", index=market.index))
    return "access: " + access.astype(str) + "; distance: " + distance.astype(str)


def build_conflict_exposure_profile_label(conflict: pd.DataFrame) -> pd.Series:
    """Build compact conflict-exposure labels from detailed conflict percentiles."""
    percentile_columns = [
        column
        for column in conflict.columns
        if column.startswith("conflict_") and column.endswith("_percentile")
    ]
    if not percentile_columns:
        return pd.Series("moderate", index=conflict.index)
    exposure_rank = conflict[percentile_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(0.5)
    return exposure_rank.map(tertile_label)


def build_cluster_profile_table(
    mapping: pd.DataFrame,
    region_lookup: pd.DataFrame,
    panel_with_clusters: pd.DataFrame,
    predictions_with_clusters: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the reviewer-facing cluster profile table from helper summaries."""
    country_region = build_country_region_summary(mapping, region_lookup)
    error_summary = build_error_summary(predictions_with_clusters)
    dominant_aez = build_dominant_aez(panel_with_clusters, aez_columns(panel_with_clusters))
    market = build_market_profile(panel_with_clusters)
    conflict = build_conflict_profile(panel_with_clusters)

    if not market.empty:
        market = market.copy()
        market["market_access_profile"] = build_market_access_profile_label(market)
    else:
        market = pd.DataFrame(columns=["cluster_id", "market_access_profile"])

    if not conflict.empty:
        conflict = conflict.copy()
        conflict["conflict_exposure_profile"] = build_conflict_exposure_profile_label(conflict)
    else:
        conflict = pd.DataFrame(columns=["cluster_id", "conflict_exposure_profile"])

    table = country_region.merge(error_summary, on="cluster_id", how="left")
    table = table.merge(dominant_aez, on="cluster_id", how="left")
    table = table.merge(market[["cluster_id", "market_access_profile"]], on="cluster_id", how="left")
    table = table.merge(conflict[["cluster_id", "conflict_exposure_profile"]], on="cluster_id", how="left")

    numeric_defaults = {
        "n_observations": 0,
        "crisis_prevalence": 0.0,
        "tp_count": 0,
        "fp_count": 0,
        "fn_count": 0,
        "tn_count": 0,
        "tp_share": 0.0,
        "fp_share": 0.0,
        "fn_share": 0.0,
        "tn_share": 0.0,
        "dominant_aez_share": 0.0,
    }
    for column, default in numeric_defaults.items():
        if column in table.columns:
            table[column] = table[column].fillna(default)
    for column in (
        "main_error_mode",
        "dominant_aez",
        "market_access_profile",
        "conflict_exposure_profile",
    ):
        if column in table.columns:
            table[column] = table[column].fillna("unknown")

    return table.sort_values("cluster_id").reset_index(drop=True)


def plot_similarity_figure(similarities: pd.DataFrame, cohesion: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """Plot market, conflict, and error-mode similarity heatmaps with cohesion bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except ModuleNotFoundError:
        sns = None

    profile_order = [
        ("market_access", "Market access"),
        ("conflict_exposure", "Conflict exposure"),
        ("error_mode", "Error mode"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 14), gridspec_kw={"width_ratios": [3.2, 1.2]})
    for row_idx, (profile_type, title) in enumerate(profile_order):
        if "profile_type" in similarities.columns:
            sub = similarities.loc[similarities["profile_type"].eq(profile_type)]
        else:
            sub = pd.DataFrame()
        if sub.empty:
            axes[row_idx, 0].text(0.5, 0.5, "not available", ha="center", va="center")
            axes[row_idx, 0].set_xticks([])
            axes[row_idx, 0].set_yticks([])
        else:
            matrix = (
                sub.pivot(index="cluster_i", columns="cluster_j", values="similarity")
                .sort_index()
                .sort_index(axis=1)
            )
            mask = upper_triangle_mask(matrix)
            if matrix.empty:
                axes[row_idx, 0].text(0.5, 0.5, "not available", ha="center", va="center")
                axes[row_idx, 0].set_xticks([])
                axes[row_idx, 0].set_yticks([])
            elif sns is not None:
                sns.heatmap(
                    matrix,
                    ax=axes[row_idx, 0],
                    vmin=-1,
                    vmax=1,
                    cmap="vlag",
                    square=True,
                    cbar=row_idx == 0,
                    mask=mask,
                )
            else:
                cmap = plt.get_cmap("coolwarm").copy()
                cmap.set_bad(color="white")
                masked_values = np.ma.array(matrix.to_numpy(dtype=float), mask=mask.to_numpy(dtype=bool))
                image = axes[row_idx, 0].imshow(masked_values, vmin=-1, vmax=1, cmap=cmap)
                axes[row_idx, 0].set_xticks(range(len(matrix.columns)), labels=matrix.columns.astype(str), rotation=90)
                axes[row_idx, 0].set_yticks(range(len(matrix.index)), labels=matrix.index.astype(str))
                axes[row_idx, 0].set_aspect("equal")
                if row_idx == 0:
                    fig.colorbar(image, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)
        axes[row_idx, 0].set_title(f"{title}: inter-cluster similarity", fontweight="bold")
        axes[row_idx, 0].set_xlabel("Cluster")
        axes[row_idx, 0].set_ylabel("Cluster")

        if "profile_type" in cohesion.columns:
            coh = cohesion.loc[cohesion["profile_type"].eq(profile_type)].sort_values("cluster_id")
        else:
            coh = pd.DataFrame()
        if coh.empty:
            axes[row_idx, 1].text(0.5, 0.5, "not available", ha="center", va="center")
            axes[row_idx, 1].set_yticks([])
        else:
            axes[row_idx, 1].barh(coh["cluster_id"].astype(str), coh["cohesion"], color="#4c78a8")
            axes[row_idx, 1].invert_yaxis()
        axes[row_idx, 1].set_xlim(0, 1)
        axes[row_idx, 1].set_title("Within-cluster\ncohesion", fontweight="bold")
        axes[row_idx, 1].set_xlabel("Cohesion")
        axes[row_idx, 1].set_ylabel("Cluster")

    fig.text(
        0.5,
        0.012,
        "Similarity matrices are symmetric; only the lower triangle is shown.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_note(output_dir: Path, profile_table: pd.DataFrame) -> Path:
    """Write reviewer-facing Chinese note and English appendix text."""
    output_dir.mkdir(parents=True, exist_ok=True)
    note_path = output_dir / "georf_m2_cluster_profile_note.md"
    note_path.write_text(
        "\n".join(
            [
                "# GeoRF m2 Cluster-Level Profile Diagnostics",
                "",
                "中文审查说明：",
                "",
                "该 appendix 只汇报一个代表性 GeoRF m2 refined local-model partition 的描述性 cluster profiles。",
                "每个 cluster 的 market access、conflict exposure、AEZ、region/country composition、crisis prevalence 和 error mode 均从现有数据与 Stage 3 prediction outputs 汇总。",
                "Similarity heatmaps 使用标准化描述性 profile 或 observed error composition 计算，并不来自 model internals。",
                "由于 similarity matrices 是 symmetric，figure 中只显示 lower triangle 以避免重复信息。",
                "因此这些结果用于刻画 local model domains，不解释为 feature importance 或 causal drivers。",
                "",
                f"该表共包含 {len(profile_table)} 个 GeoRF m2 clusters。",
                "",
                "Appendix text (English):",
                "",
                "We summarize cluster-level descriptive profiles for one representative GeoRF m2 local-model partition.",
                "The profiles include country/region composition, crisis prevalence, dominant AEZ, market-access characteristics, conflict exposure, and observed error modes.",
                "Inter-cluster similarities are computed from standardized descriptive profiles or observed error-mode composition, while the paired cohesion bars report within-cluster consistency.",
                "Because the similarity matrices are symmetric, the figure displays only the lower triangle.",
                "These summaries characterize the local-model domains and should not be interpreted as feature importance or causal attribution.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return note_path


def parse_args(argv=None):
    """Parse command-line arguments for future diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--shapefile", type=Path, default=DEFAULT_SHAPEFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def filter_february_panel(panel: pd.DataFrame, target_months=None) -> pd.DataFrame:
    """Keep February rows, optionally aligned to evaluated prediction target months."""
    filtered = panel.copy()
    date_column = next((column for column in ("date", "month_start", "target_month") if column in filtered.columns), None)
    if date_column is None:
        raise ValueError("Panel must contain one of: date, month_start, target_month")
    parsed = pd.to_datetime(filtered[date_column], errors="coerce")
    keep = parsed.dt.month.eq(2)
    if target_months is not None:
        target_month_set = {str(month) for month in pd.Series(target_months).dropna().astype(str)}
        month_labels = parsed.dt.strftime("%Y-%m")
        keep = keep & month_labels.isin(target_month_set)
    return filtered.loc[keep].copy()


def main(argv=None) -> None:
    """Build and write GeoRF m2 cluster profile artifacts."""
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_cluster_mapping(args.mapping)
    predictions = pd.read_csv(resolve_path(args.predictions))
    feb_predictions = filter_february_predictions(predictions)
    predictions_with_clusters = attach_clusters(feb_predictions, mapping)
    labeled_predictions = add_error_modes(predictions_with_clusters)

    panel = pd.read_csv(resolve_path(args.panel))
    target_months = feb_predictions["target_month"].dropna().unique()
    panel_with_clusters = attach_clusters(filter_february_panel(panel, target_months), mapping)

    region_lookup = load_region_lookup(args.shapefile)
    profile_table = build_cluster_profile_table(
        mapping,
        region_lookup,
        panel_with_clusters,
        predictions_with_clusters,
    )

    market_columns = available_columns(panel_with_clusters, MARKET_COLUMNS)
    conflict_columns = available_columns(panel_with_clusters, CONFLICT_COLUMNS)
    error_summary = build_error_summary(predictions_with_clusters)
    similarities = pd.concat(
        [
            build_intercluster_similarity(panel_with_clusters, market_columns, "market_access"),
            build_intercluster_similarity(panel_with_clusters, conflict_columns, "conflict_exposure"),
            build_error_similarity(error_summary),
        ],
        ignore_index=True,
    )
    cohesion = pd.concat(
        [
            build_feature_cohesion(panel_with_clusters, market_columns, "market_access"),
            build_feature_cohesion(panel_with_clusters, conflict_columns, "conflict_exposure"),
            build_error_cohesion(labeled_predictions),
        ],
        ignore_index=True,
    )

    profile_table.to_csv(output_dir / "georf_m2_cluster_profile_table.csv", index=False)
    similarities.to_csv(output_dir / "georf_m2_cluster_profile_similarity_matrices.csv", index=False)
    cohesion.to_csv(output_dir / "georf_m2_cluster_profile_cohesion.csv", index=False)
    plot_similarity_figure(similarities, cohesion, output_dir / "georf_m2_cluster_profile_similarity.png", args.dpi)
    write_note(output_dir, profile_table)
    (output_dir / "artifact_source_manifest.json").write_text(
        json.dumps(
            {
                "artifact_group": "06_cluster_profiles",
                "panel_path": str(args.panel),
                "mapping_path": str(args.mapping),
                "predictions_path": str(args.predictions),
                "shapefile_path": str(args.shapefile),
                "n_profile_clusters": int(profile_table["cluster_id"].nunique()),
                "n_profile_rows": int(len(profile_table)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote GeoRF m2 cluster profile artifacts to {output_dir}")
    print(f"Clusters: {profile_table['cluster_id'].nunique()}")


if __name__ == "__main__":
    main()
