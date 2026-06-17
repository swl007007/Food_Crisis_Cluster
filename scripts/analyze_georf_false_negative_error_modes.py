#!/usr/bin/env python3
"""Analyze GeoRF partitioned-model false-negative crisis error modes."""

from __future__ import annotations

import argparse
import os
import pickle
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data"
    r"\FEWSNET_forecast_unadjusted_bm.csv"
)
DEFAULT_SEASONAL_CRISIS = (
    REPO_ROOT / "final_artifacts_in_paper_updated" / "04_error_analysis" / "error_rate_seasonal_crisis.csv"
)
DEFAULT_ADJACENCY_CACHE = REPO_ROOT / "src" / "adjacency" / "polygon_adjacency_cache.pkl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final_artifacts_in_paper_updated" / "10_false_negative_error_modes"

HORIZONS = {
    "fs1": "4-month lag",
    "fs2": "8-month lag",
    "fs3": "12-month lag",
}
LAG_PHASE_COLUMNS = {
    "fs1": "fews_overall_phase_lagone",
    "fs2": "fews_overall_phase_lagtwo",
    "fs3": "fews_overall_phase_lagthree",
}
HOTSPOT_COUNTRIES = ["Sudan", "South Sudan", "Somalia", "Chad", "Niger"]
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
PRICE_COLUMNS = ["FAO_price", "WFP_Price", "WFP_Price_std", "CPI", "Food_CPI", "Food_food_inflation"]
PANEL_COLUMNS = [
    "unit_name",
    "ADMIN0",
    "ADMIN1",
    "ADMIN2",
    "FEWSNET_admin_code",
    "lat",
    "lon",
    "date",
    "month",
    "pop",
    "fews_ipc",
    "fews_ipc_crisis",
    "fews_overall_phase_lagone",
    "fews_overall_phase_lagtwo",
    "fews_overall_phase_lagthree",
    "market_distance",
    "market_access",
    "Rainf_zscore",
    "Tair_zscore",
    "dry_index",
] + CONFLICT_COLUMNS + PRICE_COLUMNS


def resolve_path(path: Path, platform_name: str | None = None) -> Path:
    """Resolve Windows paths when running under WSL."""
    platform = os.name if platform_name is None else platform_name
    raw = str(path)
    if platform == "nt":
        wsl_match = re.match(r"^/mnt/([A-Za-z])/(.*)$", raw)
        if wsl_match:
            drive, rest = wsl_match.groups()
            return Path(f"{drive.upper()}:\\{rest.replace('/', '\\')}")
        return Path(raw)

    match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
    if match:
        drive, rest = match.groups()
        return Path("/mnt") / drive.lower() / rest.replace("\\", "/")
    path = path.expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def normalize_admin_code(series: pd.Series) -> pd.Series:
    """Normalize FEWSNET admin codes to stable strings."""
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def load_predictions(source_dir: Path, scopes: list[str]) -> pd.DataFrame:
    """Load GeoRF Stage 3 prediction files."""
    frames = []
    required = {
        "FEWSNET_admin_code",
        "month_start",
        "partition_id",
        "y_true",
        "y_pred_partitioned",
        "y_prob_partitioned",
    }
    for scope in scopes:
        path = source_dir / f"result_partition_k40_compare_GF_{scope}" / "predictions_monthly.csv"
        df = pd.read_csv(path)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        df["scope"] = scope
        df["forecasting_horizon"] = HORIZONS.get(scope, scope)
        df["FEWSNET_admin_code"] = normalize_admin_code(df["FEWSNET_admin_code"])
        df["month_start"] = pd.to_datetime(df["month_start"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_panel_features(data_path: Path) -> pd.DataFrame:
    """Load selected model-panel columns used as false-negative error-mode proxies."""
    path = resolve_path(data_path)
    header = pd.read_csv(path, nrows=0)
    usecols = [column for column in PANEL_COLUMNS if column in header.columns]
    panel = pd.read_csv(path, usecols=usecols)
    panel["FEWSNET_admin_code"] = normalize_admin_code(panel["FEWSNET_admin_code"])
    panel["month_start"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["pop"] = pd.to_numeric(panel["pop"], errors="coerce")
    key_cols = ["FEWSNET_admin_code", "month_start"]
    panel = panel.dropna(subset=key_cols).copy()
    duplicated = panel.duplicated(key_cols, keep=False)
    if bool(duplicated.any()):
        examples = panel.loc[duplicated, key_cols].head(5)
        raise ValueError(f"Duplicate model-panel keys: {examples.to_dict('records')}")
    return panel


def join_panel(predictions: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Join predictions to model-panel features with full coverage required."""
    merged = predictions.merge(panel, on=["FEWSNET_admin_code", "month_start"], how="left", validate="many_to_one")
    missing = merged["ADMIN0"].isna()
    if bool(missing.any()):
        examples = merged.loc[missing, ["FEWSNET_admin_code", "month_start"]].head(10)
        raise ValueError(f"Missing model-panel features for {int(missing.sum())} rows: {examples.to_dict('records')}")
    return merged


def assign_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    """Assign reviewer-requested hotspot labels."""
    out = df.copy()
    out["hotspot"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
    for country in HOTSPOT_COUNTRIES:
        out.loc[out["ADMIN0"].eq(country), "hotspot"] = country

    afghan = out["ADMIN0"].eq("Afghanistan")
    if bool(afghan.any()):
        afghan_median = out.loc[afghan, "lat"].median()
        out.loc[afghan & (out["lat"] >= afghan_median), "hotspot"] = "Northern Afghanistan"

    mozambique = out["ADMIN0"].eq("Mozambique")
    if bool(mozambique.any()):
        mozambique_median = out.loc[mozambique, "lat"].median()
        out.loc[mozambique & (out["lat"] <= mozambique_median), "hotspot"] = "Central/southern Mozambique"
    return out


def filter_partitioned_false_negatives(df: pd.DataFrame) -> pd.DataFrame:
    """Return crisis observations missed by the GeoRF partitioned model."""
    return df[(df["y_true"] == 1) & (df["y_pred_partitioned"] == 0)].copy()


def add_lag_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Attach active lag phase and lagged non-crisis/missing proxy columns."""
    out = df.copy()
    values = []
    for _, row in out.iterrows():
        col = LAG_PHASE_COLUMNS.get(row.get("scope"))
        values.append(row.get(col, np.nan) if col else np.nan)
    out["active_lag_phase"] = pd.to_numeric(pd.Series(values, index=out.index), errors="coerce")
    out["lagged_missing"] = out["active_lag_phase"].isna()
    out["lagged_noncrisis"] = out["active_lag_phase"].lt(3)
    return out


def load_adjacency_context(cache_path: Path) -> tuple[dict[int, list[int]], dict[str, int]]:
    """Load optional adjacency context from cache."""
    path = resolve_path(cache_path)
    with open(path, "rb") as handle:
        cache = pickle.load(handle)
    adjacency_raw = cache["adjacency_dict"]
    adjacency = {int(k): [int(v) for v in values] for k, values in adjacency_raw.items()}
    code_to_index = {str(k).replace(".0", ""): int(v) for k, v in cache["polygon_id_mapping"].items()}
    return adjacency, code_to_index


def add_neighbor_context(
    df: pd.DataFrame,
    adjacency: dict[int, Iterable[int]] | None,
    code_to_index: dict[str, int] | None,
) -> pd.DataFrame:
    """Compute adjacent polygon actual/predicted crisis context for each row."""
    out = df.copy()
    if not adjacency or not code_to_index:
        out["neighbor_actual_crisis_share"] = np.nan
        out["neighbor_predicted_crisis_share"] = np.nan
        out["mixed_neighbor_actual_state"] = False
        return out

    index_to_code = {idx: code for code, idx in code_to_index.items()}
    out["FEWSNET_admin_code"] = normalize_admin_code(out["FEWSNET_admin_code"])
    key_cols = ["scope", "month_start"]
    actual_values: dict[tuple[str, pd.Timestamp], dict[str, int]] = {}
    pred_values: dict[tuple[str, pd.Timestamp], dict[str, int]] = {}
    for key, group in out.groupby(key_cols, sort=False):
        actual_values[key] = dict(zip(group["FEWSNET_admin_code"], group["y_true"].astype(int)))
        pred_values[key] = dict(zip(group["FEWSNET_admin_code"], group["y_pred_partitioned"].astype(int)))

    actual_shares = []
    pred_shares = []
    mixed_flags = []
    for _, row in out.iterrows():
        code = str(row["FEWSNET_admin_code"])
        poly_idx = code_to_index.get(code)
        key = (row["scope"], row["month_start"])
        neighbor_indices = list(adjacency.get(poly_idx, [])) if poly_idx is not None else []
        neighbor_codes = [index_to_code[idx] for idx in neighbor_indices if idx in index_to_code]
        actual = [actual_values.get(key, {}).get(ncode) for ncode in neighbor_codes]
        pred = [pred_values.get(key, {}).get(ncode) for ncode in neighbor_codes]
        actual = [value for value in actual if value is not None]
        pred = [value for value in pred if value is not None]
        if actual:
            actual_share = float(np.mean(actual))
            actual_shares.append(actual_share)
            mixed_flags.append(0 < actual_share < 1)
        else:
            actual_shares.append(np.nan)
            mixed_flags.append(False)
        pred_shares.append(float(np.mean(pred)) if pred else np.nan)

    out["neighbor_actual_crisis_share"] = actual_shares
    out["neighbor_predicted_crisis_share"] = pred_shares
    out["mixed_neighbor_actual_state"] = mixed_flags
    return out


def add_error_mode_proxies(df: pd.DataFrame, crisis_reference: pd.DataFrame) -> pd.DataFrame:
    """Add descriptive proxy flags for error-mode summaries."""
    out = add_lag_proxy(df)
    available_conflict = [col for col in CONFLICT_COLUMNS if col in out.columns]
    event_cols = [col for col in available_conflict if col != "distance_to_nearest_acled"]
    if event_cols:
        out["conflict_activity"] = out[event_cols].fillna(0).sum(axis=1).gt(0)
        ref_conflict = crisis_reference[event_cols].fillna(0).sum(axis=1)
        threshold = float(ref_conflict.quantile(0.75)) if len(ref_conflict) else np.nan
        out["high_conflict_proxy"] = out[event_cols].fillna(0).sum(axis=1).gt(threshold) if not np.isnan(threshold) else False
    else:
        out["conflict_activity"] = False
        out["high_conflict_proxy"] = False
    out["conflict_missing"] = out[available_conflict].isna().all(axis=1) if available_conflict else True

    available_price = [col for col in PRICE_COLUMNS if col in out.columns]
    out["price_missing"] = out[available_price].isna().all(axis=1) if available_price else True
    price_high = pd.Series(False, index=out.index)
    if "WFP_Price_std" in out.columns:
        price_high = price_high | pd.to_numeric(out["WFP_Price_std"], errors="coerce").ge(1.0).fillna(False)
    price_level_cols = [col for col in ["FAO_price", "WFP_Price", "CPI", "Food_CPI", "Food_food_inflation"] if col in out.columns]
    for col in price_level_cols:
        threshold = pd.to_numeric(crisis_reference[col], errors="coerce").quantile(0.75)
        if pd.notna(threshold):
            price_high = price_high | pd.to_numeric(out[col], errors="coerce").gt(threshold).fillna(False)
    out["high_price_proxy"] = price_high
    out["near_threshold"] = pd.to_numeric(out["y_prob_partitioned"], errors="coerce").between(0.35, 0.5, inclusive="left")
    return out


def mode_or_blank(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return ""
    return str(values.mode().iloc[0])


def build_hotspot_summary(all_rows: pd.DataFrame, false_negative: pd.DataFrame) -> pd.DataFrame:
    """Build hotspot-by-horizon false-negative summary."""
    rows = []
    scoped = all_rows[all_rows["hotspot"].notna()].copy()
    for (hotspot, scope, horizon), crisis_group in scoped[scoped["y_true"] == 1].groupby(
        ["hotspot", "scope", "forecasting_horizon"],
        sort=True,
    ):
        fn = false_negative[
            (false_negative["hotspot"] == hotspot)
            & (false_negative["scope"] == scope)
            & (false_negative["forecasting_horizon"] == horizon)
        ]
        actual_crisis_population = float(pd.to_numeric(crisis_group["pop"], errors="coerce").sum())
        fn_population = float(pd.to_numeric(fn["pop"], errors="coerce").sum())
        rows.append(
            {
                "hotspot": hotspot,
                "scope": scope,
                "forecasting_horizon": horizon,
                "actual_crisis_observations": int(len(crisis_group)),
                "false_negative_observations": int(len(fn)),
                "actual_crisis_population": actual_crisis_population,
                "false_negative_population": fn_population,
                "missed_crisis_population_share": fn_population / actual_crisis_population
                if actual_crisis_population
                else np.nan,
                "mean_false_negative_probability": float(pd.to_numeric(fn["y_prob_partitioned"], errors="coerce").mean())
                if len(fn)
                else np.nan,
                "near_threshold_share": float(fn["y_prob_partitioned"].between(0.35, 0.5, inclusive="left").mean())
                if len(fn)
                else np.nan,
                "dominant_month": mode_or_blank(fn["month_start"].dt.strftime("%Y-%m")),
            }
        )
    return pd.DataFrame(rows)


def build_error_mode_table(false_negative: pd.DataFrame) -> pd.DataFrame:
    """Build hotspot-level descriptive error-mode evidence table."""
    rows = []
    for hotspot, group in false_negative.groupby("hotspot", sort=True):
        population = float(pd.to_numeric(group["pop"], errors="coerce").sum())
        by_horizon = group.groupby("forecasting_horizon")["pop"].sum().sort_values(ascending=False)
        rows.append(
            {
                "hotspot": hotspot,
                "false_negative_observations": int(len(group)),
                "false_negative_population": population,
                "dominant_horizon_by_population": str(by_horizon.index[0]) if len(by_horizon) else "",
                "dominant_month": mode_or_blank(group["month_start"].dt.strftime("%Y-%m")),
                "mean_false_negative_probability": float(pd.to_numeric(group["y_prob_partitioned"], errors="coerce").mean()),
                "near_threshold_share": float(group["near_threshold"].mean()),
                "conflict_activity_share": float(group["conflict_activity"].mean()),
                "high_conflict_proxy_share": float(group["high_conflict_proxy"].mean()),
                "conflict_missing_share": float(group["conflict_missing"].mean()),
                "high_price_proxy_share": float(group["high_price_proxy"].mean()),
                "price_missing_share": float(group["price_missing"].mean()),
                "lagged_noncrisis_share": float(group["lagged_noncrisis"].mean()),
                "lagged_missing_share": float(group["lagged_missing"].mean()),
                "mixed_neighbor_actual_state_share": float(group["mixed_neighbor_actual_state"].mean()),
                "neighbor_actual_crisis_share_mean": float(group["neighbor_actual_crisis_share"].mean()),
                "neighbor_predicted_crisis_share_mean": float(group["neighbor_predicted_crisis_share"].mean()),
            }
        )
    return pd.DataFrame(rows)


def add_summary_context_to_error_modes(error_modes: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Attach dominant seasonal-error context from the hotspot-by-horizon table."""
    rows = []
    for hotspot, group in summary.groupby("hotspot", sort=False):
        valid = group.dropna(subset=["seasonal_crisis_error_rate_weighted"])
        if valid.empty:
            rows.append(
                {
                    "hotspot": hotspot,
                    "max_seasonal_crisis_error_rate": np.nan,
                    "dominant_error_season": "",
                    "dominant_error_horizon": "",
                }
            )
            continue
        row = valid.sort_values("seasonal_crisis_error_rate_weighted", ascending=False).iloc[0]
        rows.append(
            {
                "hotspot": hotspot,
                "max_seasonal_crisis_error_rate": row["seasonal_crisis_error_rate_weighted"],
                "dominant_error_season": row.get("highest_error_season", ""),
                "dominant_error_horizon": row.get("forecasting_horizon", ""),
            }
        )
    context = pd.DataFrame(rows)
    return error_modes.merge(context, on="hotspot", how="left")


def add_seasonal_context(summary: pd.DataFrame, seasonal_csv: Path, panel: pd.DataFrame) -> pd.DataFrame:
    """Attach weighted seasonal crisis error-rate context if the CSV is available."""
    path = resolve_path(seasonal_csv)
    if not path.exists():
        return summary
    seasonal = pd.read_csv(path)
    required = {"forecasting_horizon", "season", "uid", "n", "err_rate"}
    if not required.issubset(seasonal.columns):
        return summary
    lookup = panel[["FEWSNET_admin_code", "hotspot"]].dropna(subset=["hotspot"]).drop_duplicates("FEWSNET_admin_code")
    lookup = lookup.rename(columns={"FEWSNET_admin_code": "uid"})
    lookup["uid"] = normalize_admin_code(lookup["uid"])
    seasonal["uid"] = normalize_admin_code(seasonal["uid"])
    merged = seasonal.merge(lookup, on="uid", how="inner")
    if merged.empty:
        return summary
    rows = []
    for (hotspot, horizon), group in merged.groupby(["hotspot", "forecasting_horizon"], sort=False):
        weighted_error = np.average(group["err_rate"], weights=group["n"]) if group["n"].sum() else np.nan
        by_season = group.groupby("season").apply(
            lambda g: np.average(g["err_rate"], weights=g["n"]) if g["n"].sum() else np.nan,
            include_groups=False,
        )
        rows.append(
            {
                "hotspot": hotspot,
                "forecasting_horizon": horizon,
                "seasonal_crisis_error_rate_weighted": float(weighted_error),
                "highest_error_season": str(by_season.idxmax()) if len(by_season.dropna()) else "",
            }
        )
    context = pd.DataFrame(rows)
    return summary.merge(context, on=["hotspot", "forecasting_horizon"], how="left")


def format_compact_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format selected error-mode columns for appendix Markdown."""
    columns = [
        "hotspot",
        "false_negative_population",
        "dominant_error_season",
        "near_threshold_share",
        "lagged_noncrisis_share",
        "high_conflict_proxy_share",
        "high_price_proxy_share",
        "mixed_neighbor_actual_state_share",
    ]
    out = table[columns].copy()
    out["false_negative_population"] = out["false_negative_population"].map(lambda value: f"{value:,.0f}")
    for col in columns[3:]:
        out[col] = out[col].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    return out


def write_markdown_table(table: pd.DataFrame, output_path: Path) -> None:
    """Write a Markdown table without optional pandas dependencies."""
    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = ["" if pd.isna(row[column]) else str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def describe_hotspot(row: pd.Series) -> str:
    """Build a cautious English evidence sentence for one hotspot."""
    signals = []
    if row.get("dominant_error_season"):
        signals.append(
            f"seasonal crisis-error context peaks in {row['dominant_error_season']} "
            f"({row.get('dominant_error_horizon', '')})"
        )
    if row["lagged_noncrisis_share"] >= 0.5:
        signals.append("many missed crises had lagged non-crisis phase values")
    elif row["lagged_noncrisis_share"] < 0.5:
        signals.append("lagged non-crisis states are not the dominant proxy")
    if row["high_conflict_proxy_share"] >= 0.25:
        signals.append("high conflict-intensity proxies were common")
    elif row["conflict_activity_share"] >= 0.5:
        signals.append("basic conflict activity proxies were present, but high-conflict intensity was limited")
    if row["high_price_proxy_share"] >= 0.5:
        signals.append("market-price stress proxies were common")
    if row["price_missing_share"] >= 0.5:
        signals.append("price covariates were frequently missing")
    elif row["price_missing_share"] < 0.1 and row["conflict_missing_share"] < 0.1:
        signals.append("covariate missingness is not a dominant explanation")
    if row["near_threshold_share"] >= 0.25:
        signals.append("many false negatives were near the hard classification threshold")
    if row["mixed_neighbor_actual_state_share"] >= 0.5:
        signals.append("adjacent polygons often had mixed crisis/non-crisis states")
    elif row["mixed_neighbor_actual_state_share"] < 0.25:
        signals.append("the adjacency boundary proxy is limited")
    if not signals:
        signals.append("no single proxy dominates the missed-crisis pattern")
    return "; ".join(signals)


def write_note(output_dir: Path, error_modes: pd.DataFrame) -> Path:
    """Write reviewer-facing Chinese note and English appendix text."""
    lines = [
        "# GeoRF Partitioned False-Negative Error Modes",
        "",
        "中文审查说明：",
        "",
        "该 appendix 只分析 GeoRF partitioned/local RF 模型的 crisis false negatives，即 `y_true = 1` 且 `y_pred_partitioned = 0`。",
        "分析对象包括 Sudan、South Sudan、Somalia、Chad、Niger、Northern Afghanistan，以及 Central/southern Mozambique。",
        "这些结果是 descriptive error-mode analysis，不是 causal attribution。",
        "Conflict、market price、lagged outcome、seasonal timing 和 boundary context 都是现有 covariates 或 adjacency 的 proxy evidence。",
        "Missing covariates 用 conflict / price proxy columns 的缺失率表示；如果缺失率低，则不把 missingness 写成主要解释。",
        "Northern Afghanistan 与 Central/southern Mozambique 使用该国 evaluated polygons 的纬度中位数定义为透明的 spatial proxy。",
        "",
    ]
    for _, row in error_modes.sort_values("false_negative_population", ascending=False).iterrows():
        lines.append(
            f"- {row['hotspot']}: false-negative population-months = {row['false_negative_population']:,.0f}; "
            f"{describe_hotspot(row)}."
        )
    lines.extend(
        [
            "",
            "Appendix text (English):",
            "",
            "We summarize descriptive error modes for crisis observations missed by the GeoRF partitioned model.",
            "The analysis is restricted to observations with actual crisis status and a partitioned-model non-crisis prediction.",
            "The evidence columns should be interpreted as proxies: conflict exposure is measured from available ACLED-derived columns, market stress from food-price columns, lagged outcomes from prior FEWSNET phase columns, and boundary context from adjacent polygon crisis states.",
            "Missing covariates are represented by missingness in the conflict and price proxy columns; when missingness is low, the note does not treat missingness as the dominant explanation.",
            "These summaries identify plausible mechanisms for missed crises but do not establish causal attribution.",
            "",
        ]
    )
    for _, row in error_modes.sort_values("false_negative_population", ascending=False).iterrows():
        lines.append(
            f"- {row['hotspot']}: {row['false_negative_population']:,.0f} false-negative population-months; "
            f"{describe_hotspot(row)}."
        )
    path = output_dir / "georf_partitioned_false_negative_note.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seasonal-crisis", type=Path, default=DEFAULT_SEASONAL_CRISIS)
    parser.add_argument("--adjacency-cache", type=Path, default=DEFAULT_ADJACENCY_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scopes", nargs="+", default=["fs1", "fs2", "fs3"], choices=["fs1", "fs2", "fs3"])
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(args.source_dir, args.scopes)
    panel = load_panel_features(args.data)
    joined = join_panel(predictions, panel)
    joined = assign_hotspots(joined)

    try:
        adjacency, code_to_index = load_adjacency_context(args.adjacency_cache)
    except Exception:
        adjacency, code_to_index = None, None
    joined = add_neighbor_context(joined, adjacency, code_to_index)
    crisis_reference = joined[joined["y_true"] == 1].copy()
    joined = add_error_mode_proxies(joined, crisis_reference)

    false_negative = filter_partitioned_false_negatives(joined)
    false_negative = false_negative[false_negative["hotspot"].notna()].copy()
    summary = build_hotspot_summary(joined, false_negative)
    summary = add_seasonal_context(summary, args.seasonal_crisis, joined[["FEWSNET_admin_code", "hotspot"]])
    error_modes = build_error_mode_table(false_negative)
    error_modes = add_summary_context_to_error_modes(error_modes, summary)
    compact = format_compact_table(error_modes)

    summary.to_csv(args.output_dir / "georf_partitioned_false_negative_hotspot_summary.csv", index=False)
    error_modes.to_csv(args.output_dir / "georf_partitioned_false_negative_error_modes.csv", index=False)
    write_markdown_table(compact, args.output_dir / "georf_partitioned_false_negative_hotspot_compact_table.md")
    write_note(args.output_dir, error_modes)

    print(f"Wrote GeoRF false-negative error-mode artifacts to {args.output_dir}")
    print(f"Rows: predictions={len(predictions)}, joined={len(joined)}, selected_false_negatives={len(false_negative)}")


if __name__ == "__main__":
    main()
