#!/usr/bin/env python3
"""Create region-level partitioned vs pooled vs FEWSNET expert comparison table."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "main_ablation_results" / "march2026_main_backup_month_ind_cont3"
DEFAULT_FEWSNET = Path(
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome"
    r"\FEWSNET_IPC\FEWSNET.csv"
)
DEFAULT_SHAPEFILE = Path(
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome"
    r"\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp"
)
DEFAULT_OUTPUT = (
    DEFAULT_SOURCE_DIR
    / "seasonal_performance_GF"
    / "table2_region_performance_partitioned_pooled_fewsnet.csv"
)
SCOPE_TO_HORIZON = {"fs1": 4, "fs2": 8, "fs3": 12}


def resolve_local_path(path: Path) -> Path:
    if path.exists():
        return path
    raw = str(path)
    if len(raw) >= 2 and raw[1] == ":":
        alt = Path("/mnt") / raw[0].lower() / raw[2:].replace("\\", "/").lstrip("/")
        if alt.exists():
            return alt
    return path


def normalize_admin_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def scope_label(scope: str) -> str:
    horizon = SCOPE_TO_HORIZON.get(str(scope))
    return f"{horizon}-month lag" if horizon is not None else str(scope)


def load_seasonal_helpers():
    script = REPO_ROOT / "scripts" / "plot_seasonal_performance.py"
    spec = importlib.util.spec_from_file_location("plot_seasonal_performance", script)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load module from {script}")
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate region-level partitioned vs pooled vs FEWSNET expert comparison table."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--fewsnet", type=Path, default=DEFAULT_FEWSNET)
    parser.add_argument("--shapefile", type=Path, default=DEFAULT_SHAPEFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-token", default="GF", help="Stage-3 result folder token, e.g. GF.")
    parser.add_argument("--scopes", nargs="+", default=["fs1", "fs2", "fs3"], choices=["fs1", "fs2", "fs3"])
    parser.add_argument(
        "--extend-fewsnet",
        dest="extend_fewsnet",
        action="store_true",
        default=False,
        help="Reuse FEWSNET 8-month expert predictions for the 12-month lag as a labeled diagnostic.",
    )
    return parser.parse_args(argv)


def load_model_predictions(source_dir: Path, model_token: str, scopes: list[str]) -> pd.DataFrame:
    frames = []
    for scope in scopes:
        path = source_dir / f"result_partition_k40_compare_{model_token}_{scope}" / "predictions_monthly.csv"
        df = pd.read_csv(path)
        required = {"FEWSNET_admin_code", "month_start", "y_true", "y_pred_partitioned", "y_pred_pooled"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        df["scope"] = scope
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["FEWSNET_admin_code"] = normalize_admin_code(combined["FEWSNET_admin_code"])
    combined["date"] = pd.to_datetime(combined["month_start"])
    return combined


def load_fewsnet_expert_predictions(
    fewsnet_path: Path,
    scopes: list[str],
    extend_fewsnet: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(
        resolve_local_path(fewsnet_path),
        usecols=["country", "admin_code", "year", "month", "fews_proj_near", "fews_proj_med"],
    )
    df = df.dropna(subset=["admin_code", "year", "month"]).copy()
    df["FEWSNET_admin_code"] = normalize_admin_code(df["admin_code"])
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df["pred_near"] = np.where(df["fews_proj_near"].notna(), (df["fews_proj_near"] >= 3).astype(int), np.nan)
    df["pred_med"] = np.where(df["fews_proj_med"].notna(), (df["fews_proj_med"] >= 3).astype(int), np.nan)
    df = df.sort_values(["FEWSNET_admin_code", "year", "month"])
    df["fewsnet_expert_fs1"] = df.groupby("FEWSNET_admin_code")["pred_near"].shift(4)
    df["fewsnet_expert_fs2"] = df.groupby("FEWSNET_admin_code")["pred_med"].shift(8)

    parts = []
    if "fs1" in scopes:
        parts.append(
            df[["FEWSNET_admin_code", "date", "fewsnet_expert_fs1"]]
            .rename(columns={"fewsnet_expert_fs1": "y_pred_fewsnet"})
            .assign(scope="fs1")
        )
    if "fs2" in scopes:
        parts.append(
            df[["FEWSNET_admin_code", "date", "fewsnet_expert_fs2"]]
            .rename(columns={"fewsnet_expert_fs2": "y_pred_fewsnet"})
            .assign(scope="fs2")
        )
    if "fs3" in scopes and extend_fewsnet:
        parts.append(
            df[["FEWSNET_admin_code", "date", "fewsnet_expert_fs2"]]
            .rename(columns={"fewsnet_expert_fs2": "y_pred_fewsnet"})
            .assign(scope="fs3")
        )
    return pd.concat(parts, ignore_index=True)


def load_region_lookup(shapefile_path: Path, region_map: dict[str, str]) -> pd.DataFrame:
    gdf = gpd.read_file(resolve_local_path(shapefile_path))
    for col in ["FEWSNET_admin_code", "uid", "admin_code", "adm_code"]:
        if col in gdf.columns:
            gdf = gdf.rename(columns={col: "FEWSNET_admin_code"})
            break
    if "FEWSNET_admin_code" not in gdf.columns:
        raise ValueError(f"No admin-code column found in shapefile: {list(gdf.columns)}")
    if "ADMIN0" not in gdf.columns:
        raise ValueError(f"ADMIN0 column not found in shapefile: {list(gdf.columns)}")
    gdf["FEWSNET_admin_code"] = normalize_admin_code(gdf["FEWSNET_admin_code"])
    lookup = gdf[["FEWSNET_admin_code", "ADMIN0"]].copy()
    lookup["region"] = lookup["ADMIN0"].map(region_map).fillna("Other")
    return lookup


def metrics_for(sub: pd.DataFrame, pred_col: str, helpers) -> dict[str, float]:
    valid = sub[pred_col].notna() & sub["y_true"].notna()
    if int(valid.sum()) == 0:
        return {"support": 0, "precision": np.nan, "recall": np.nan, "f1": np.nan, "accuracy": np.nan}
    c = helpers._confusion(sub.loc[valid, "y_true"].values, sub.loc[valid, pred_col].values)
    precision, recall, f1, accuracy = helpers._prf(c["tp"], c["fp"], c["fn"], c["tn"])
    return {"support": c["support"], "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def build_table(
    model_df: pd.DataFrame,
    fewsnet_df: pd.DataFrame,
    region_lookup: pd.DataFrame,
    scopes: list[str],
    helpers,
) -> pd.DataFrame:
    merged = model_df.merge(fewsnet_df, on=["FEWSNET_admin_code", "date", "scope"], how="left")
    merged = merged.merge(region_lookup, on="FEWSNET_admin_code", how="left")
    merged = merged[merged["ADMIN0"].notna()].copy()
    native_fewsnet_scopes = set(fewsnet_df["scope"].dropna().astype(str).unique())

    rows = []
    for scope in scopes:
        for region in sorted(merged["region"].unique()):
            sub = merged[(merged["scope"] == scope) & (merged["region"] == region)]
            if sub.empty:
                continue
            partitioned = metrics_for(sub, "y_pred_partitioned", helpers)
            pooled = metrics_for(sub, "y_pred_pooled", helpers)
            if scope in native_fewsnet_scopes:
                fewsnet = metrics_for(sub, "y_pred_fewsnet", helpers)
            else:
                fewsnet = {
                    "support": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "accuracy": np.nan,
                }

            row = {
                "forecasting_horizon": scope_label(scope),
                "region": region,
                "support": partitioned["support"],
                "fewsnet_valid_support": fewsnet["support"],
            }
            for prefix, values in [
                ("partitioned", partitioned),
                ("pooled", pooled),
                ("fewsnet_expert", fewsnet),
            ]:
                for metric in ["precision", "recall", "f1", "accuracy"]:
                    value = values[metric]
                    row[f"{prefix}_{metric}"] = round(value, 4) if not np.isnan(value) else np.nan
            for baseline in ["pooled", "fewsnet_expert"]:
                for metric in ["precision", "recall", "f1", "accuracy"]:
                    partitioned_value = row[f"partitioned_{metric}"]
                    baseline_value = row[f"{baseline}_{metric}"]
                    row[f"delta_partitioned_minus_{baseline}_{metric}"] = (
                        round(partitioned_value - baseline_value, 4)
                        if not (pd.isna(partitioned_value) or pd.isna(baseline_value))
                        else np.nan
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    args = parse_args(argv)
    helpers = load_seasonal_helpers()
    model_df = load_model_predictions(args.source_dir, args.model_token, args.scopes)
    fewsnet_df = load_fewsnet_expert_predictions(
        args.fewsnet,
        args.scopes,
        extend_fewsnet=args.extend_fewsnet,
    )
    region_lookup = load_region_lookup(args.shapefile, helpers.REGION_MAP)
    result = build_table(model_df, fewsnet_df, region_lookup, args.scopes, helpers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    fewsnet_source = pd.read_csv(resolve_local_path(args.fewsnet), usecols=["country", "admin_code"]).dropna(subset=["admin_code"])
    print(f"FEWSNET source rows with admin_code: {len(fewsnet_source)}")
    print(f"FEWSNET unique admin codes: {fewsnet_source['admin_code'].nunique()}")
    print(f"FEWSNET countries with admin_code: {fewsnet_source['country'].nunique()}")
    print(f"Model unique admin codes: {model_df['FEWSNET_admin_code'].nunique()}")
    print(f"FEWSNET 12-month baseline extended from 8-month lag: {args.extend_fewsnet}")
    print(f"Wrote: {args.output}")
    print(f"Rows: {len(result)}, columns: {len(result.columns)}")


if __name__ == "__main__":
    main()
