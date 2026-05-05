#!/usr/bin/env python3
"""Standalone pure-prediction pipeline for the (Apr 2026 / Apr 2027) deliverables.

Substitutes "April 2026" with **Jun 2026** (next FEWSNET publication after the
2026-02 release; fs1 / lag = 4 months) and "April 2027" with **Feb 2027** (the
12-month forecast horizon from the 2026-02 publication; fs3 / lag = 12 months).
Both target months are FEWSNET non-publication months relative to the input
data, so labels are unavailable; this is a pure-prediction (no validation) flow.

Outputs (under --out-dir, default deliverables/predict_2026_2027/):
- predictions_2026-06_2027-02_georf.xlsx  (D1)
- map_phase3plus_2026-06_georf.png        (D2)
- map_phase3plus_2027-02_georf.png        (D2)
- run_manifest.json
- intermediate per-target CSVs

Forks scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py for feature
preparation, partition assignment, and the choropleth-rendering helpers.
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Repo imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess.preprocess import load_and_preprocess_data
from src.feature.feature import prepare_features
from src.utils.lag_schedules import forecasting_scope_to_lag
from config import (
    LAGS_MONTHS,
    PREDICTION_THRESHOLD,
    IMPUTE_FEWSNET_GAPS,
    ADJACENCY_SHAPEFILE_PATH,
)

warnings.filterwarnings('ignore')

RANDOM_STATE = 5
RF_PARAMS = dict(n_estimators=100, max_depth=None, random_state=RANDOM_STATE, n_jobs=1)
MIN_PARTITION_SAMPLES = 50

# Two-target deliverable spec.
DEFAULT_TARGETS = [
    {'label': 'Jun 2026 (nowcast)',  'target_month': '2026-06', 'forecasting_scope': 1},
    {'label': 'Feb 2027 (12-month)', 'target_month': '2027-02', 'forecasting_scope': 3},
]
DEFAULT_DATA_PATH = (
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data"
    r"\assembled_FEWSNET\FEWSNET_forecast_unadjusted_bm_2025_combined.csv"
)
DEFAULT_OUT_DIR = r"deliverables/predict_2026_2027"
# Forward-fill the FEWSNET non-publication slots that fall inside training
# windows so lag features that reference them are not silently NaN-dropped.
DEFAULT_GAP_MONTHS = ('2025-02', '2025-06')


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _sha256_head(path: str, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:n]


def _git_state(repo_root: Path) -> Dict[str, str]:
    def _run(args: List[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=repo_root, text=True).strip()
        except Exception:
            return ''
    head = _run(['git', 'rev-parse', 'HEAD'])
    dirty = _run(['git', 'status', '--porcelain'])
    return {'git_commit': head, 'git_dirty': bool(dirty)}


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def _materialize_target_rows(src_csv: str, target_months: List[str], dest_csv: str) -> Dict[str, int]:
    """Append synthetic rows for `target_months` if they are absent from src_csv.

    For each (polygon, target_month) missing from the CSV, we copy the
    polygon's latest existing row, set `date` to the target month start, set
    `fews_ipc_crisis` to null (they're predict-only targets), and write the
    extended dataframe to dest_csv. Returns per-month counts of synthesized
    rows.

    Rationale: the existing pipeline requires rows at the prediction dates so
    that prepare_features can compute lag features against them. The combined
    CSV currently ends at 2026-04 — to predict Jun 2026 / Feb 2027 we need
    placeholder rows there with feature snapshots forward-filled from the
    polygon's most recent observation. Time-variant features held at "latest
    observed" values is a documented modeling assumption.
    """
    src = pd.read_csv(src_csv)
    src['date'] = pd.to_datetime(src['date'])
    poly_col = 'FEWSNET_admin_code'
    if poly_col not in src.columns:
        raise ValueError(f"{src_csv} missing {poly_col} column")
    existing_pairs = set(
        (str(p), d.strftime('%Y-%m-01'))
        for p, d in zip(src[poly_col].astype(str), src['date'])
    )

    synth_frames = []
    counts: Dict[str, int] = {}
    for ym in target_months:
        target_start = pd.Period(ym, freq='M').start_time
        target_iso = target_start.strftime('%Y-%m-01')
        # Polygons that already have a row at the target month — nothing to do.
        polys_present = {
            p for (p, d) in existing_pairs if d == target_iso
        }
        all_polys = src[poly_col].astype(str).unique()
        polys_missing = [p for p in all_polys if p not in polys_present]
        if not polys_missing:
            counts[ym] = 0
            continue
        # Each missing polygon: copy its latest pre-existing row and rewrite
        # the date / null the label. Drop duplicate (poly, date) rows from the
        # source first — the combined CSV has 2 known duplicates that would
        # otherwise multiply the synthetic frame.
        src_dedup = src.drop_duplicates(subset=[poly_col, 'date'], keep='first')
        latest_per_poly = (
            src_dedup.sort_values('date')
                     .drop_duplicates(subset=[poly_col], keep='last')
                     .copy()
        )
        latest_per_poly[poly_col] = latest_per_poly[poly_col].astype(str)
        synth = latest_per_poly[latest_per_poly[poly_col].isin(polys_missing)].copy()
        synth['date'] = target_start
        synth['fews_ipc_crisis'] = np.nan
        if 'fews_ipc' in synth.columns:
            synth['fews_ipc'] = np.nan
        if 'year' in synth.columns:
            synth['year'] = target_start.year
        if 'month' in synth.columns:
            synth['month'] = target_start.month
        synth_frames.append(synth)
        counts[ym] = len(synth)

    if synth_frames:
        out = pd.concat([src] + synth_frames, ignore_index=True, sort=False)
    else:
        out = src
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out.to_csv(dest_csv, index=False)
    return counts


def _resolve_train_window(df: pd.DataFrame,
                          target_month: pd.Period,
                          lag_months: int,
                          train_window_months: int) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Period]:
    """Return (train_window_start, requested_train_end, actual_labeled_train_end_period).

    requested_train_end = target_month_start - lag_months
    actual_labeled_train_end = max labeled month <= requested_train_end
    train_window_start = actual_labeled_train_end_start - (train_window_months - 1) months
    """
    target_start = target_month.start_time
    requested_end = target_start - pd.DateOffset(months=lag_months)
    labeled_dates = df.loc[df['fews_ipc_crisis'].notna(), 'date']
    eligible = labeled_dates[labeled_dates < requested_end + pd.DateOffset(months=1)]
    if eligible.empty:
        raise RuntimeError(
            f"No labeled rows on or before {requested_end:%Y-%m} — cannot train for target {target_month}"
        )
    actual_end_period = eligible.dt.to_period('M').max()
    actual_end_start = actual_end_period.start_time
    window_start = actual_end_start - pd.DateOffset(months=train_window_months - 1)
    return window_start, requested_end, actual_end_period


def _load_partition_mapping(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'FEWSNET_admin_code', 'cluster_id'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Partition map {path} missing columns: {missing}')
    return df[['FEWSNET_admin_code', 'cluster_id']].copy()


def _attach_clusters(df: pd.DataFrame, partition_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    merged = df.merge(partition_df, on='FEWSNET_admin_code', how='left')
    merged['cluster_id'] = merged['cluster_id'].fillna(-1).astype(int)
    return merged['cluster_id'].to_numpy(), merged


def _train_partitioned(X_train: np.ndarray, y_train: np.ndarray,
                       cluster_train: np.ndarray) -> Dict[int, Optional[RandomForestClassifier]]:
    models: Dict[int, Optional[RandomForestClassifier]] = {}
    fallback = 0
    for cid in np.unique(cluster_train):
        if cid < 0:
            continue
        m = cluster_train == cid
        Xc, yc = X_train[m], y_train[m]
        classes = np.unique(yc)
        if len(yc) < MIN_PARTITION_SAMPLES or classes.size < 2:
            models[int(cid)] = None
            fallback += 1
            print(f"  cluster {cid}: pooled fallback (n={len(yc)}, classes={classes.tolist()})")
            continue
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(Xc, yc)
        models[int(cid)] = clf
    print(f"  partitioned models trained; {fallback} clusters falling back to pooled")
    return models


def _predict_class1(model: RandomForestClassifier, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (hard_pred, prob_class1) handling one-class fallback."""
    proba = model.predict_proba(X)
    classes = model.classes_
    if proba.shape[1] == 1:
        only_class = int(classes[0])
        prob1 = np.full(X.shape[0], 1.0 if only_class == 1 else 0.0)
    else:
        idx = np.where(classes == 1)[0]
        prob1 = proba[:, idx[0]] if idx.size else np.zeros(X.shape[0])
    pred = (prob1 >= PREDICTION_THRESHOLD).astype(int)
    return pred, prob1


def _qc_assert_probs(prob: np.ndarray, name: str) -> None:
    if prob.size == 0:
        return
    if not np.all((prob >= 0.0) & (prob <= 1.0)):
        raise AssertionError(f"{name}: probabilities out of [0, 1]")


def _qc_assert_threshold(pred: np.ndarray, prob: np.ndarray, name: str) -> None:
    expected = (prob >= PREDICTION_THRESHOLD).astype(int)
    if not np.array_equal(pred, expected):
        raise AssertionError(f"{name}: pred != (prob >= {PREDICTION_THRESHOLD})")


# ----------------------------------------------------------------------------
# Per-target prediction
# ----------------------------------------------------------------------------

def _predict_target(args: argparse.Namespace, target: Dict[str, Any]) -> pd.DataFrame:
    target_month = pd.Period(target['target_month'], freq='M')
    scope = int(target['forecasting_scope'])
    lag_months = forecasting_scope_to_lag(scope, LAGS_MONTHS)
    feature_month = (target_month.start_time - pd.DateOffset(months=lag_months)).to_period('M')

    print('=' * 80)
    print(f"Target: {target['label']}  ->  {target_month}, fs{scope}, lag={lag_months}mo")
    print(f"Feature month: {feature_month}")
    print('=' * 80)

    # Step 1 — load with predict-only whitelist + gap forward-fill.
    df = load_and_preprocess_data(
        args.data,
        predict_target_months=[str(target_month)],
        impute_gap_months=tuple(args.gap_months) if args.gap_months else None,
    )
    print(f"  loaded df: {len(df)} rows, {df['FEWSNET_admin_code'].nunique()} polygons")

    # Step 2 — partitions. Resolve scope-specific override if provided.
    scope_override = getattr(args, f'partition_map_fs{scope}', None)
    pmap_path = scope_override or args.partition_map
    if pmap_path is None:
        raise ValueError(
            f"No partition map for fs{scope}. Provide --partition-map or "
            f"--partition-map-fs{scope}."
        )
    print(f"  partition map (fs{scope}): {pmap_path}")
    partition_df = _load_partition_mapping(pmap_path)
    cluster_arr, df = _attach_clusters(df, partition_df)
    n_unmapped = int((cluster_arr < 0).sum())
    if n_unmapped:
        print(f"  WARNING: {n_unmapped} rows have no cluster_id (will use pooled fallback)")

    # Step 3 — features. prepare_features sorts by (poly, date) internally
    # and returns df_sorted['date'], so all aligned arrays must be in the
    # same sort order. Sort df (which already carries cluster_id from
    # _attach_clusters) + derive aligned X_loc / cluster_arr from it.
    df = df.sort_values(['FEWSNET_admin_code', 'date']).reset_index(drop=True)
    cluster_arr = df['cluster_id'].fillna(-1).astype(int).to_numpy()
    if {'lat', 'lon'}.issubset(df.columns):
        X_loc = df[['lat', 'lon']].to_numpy()
    elif {'latitude', 'longitude'}.issubset(df.columns):
        X_loc = df[['latitude', 'longitude']].to_numpy()
    else:
        raise ValueError('df missing latitude/longitude columns')
    X, y, _, _, _, _, dates, _ = prepare_features(df, cluster_arr, X_loc, forecasting_scope=scope)
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    admin_codes = df['FEWSNET_admin_code'].to_numpy()
    if len(dates) != len(admin_codes):
        raise AssertionError(
            f"dates ({len(dates)}) and admin_codes ({len(admin_codes)}) length mismatch"
        )

    # Step 4 — train/test split. train_window: labeled rows in
    # [actual_labeled_train_end - 35 months, actual_labeled_train_end]; test:
    # rows whose date == target_month_start.
    train_start, requested_end, actual_end_period = _resolve_train_window(
        df, target_month, lag_months, args.train_window
    )
    train_end_excl = actual_end_period.start_time + pd.DateOffset(months=1)
    target_start = target_month.start_time
    target_end_excl = target_start + pd.DateOffset(months=1)

    is_train = (dates >= train_start) & (dates < train_end_excl) & pd.Series(y).notna().values
    is_test = (dates >= target_start) & (dates < target_end_excl)

    # Drop training rows with any NaN feature.
    feature_nan = np.isnan(X).any(axis=1)
    is_train = is_train & ~feature_nan
    print(
        f"  train window: [{train_start:%Y-%m}, {actual_end_period}] "
        f"({int(is_train.sum())} rows after NaN-feature drop)"
    )
    if is_test.sum() == 0:
        raise RuntimeError(
            f"No rows for target {target_month} — verify the CSV contains rows there with null labels"
        )

    Xtr, ytr = X[is_train], y[is_train].astype(int)
    Xtg = X[is_test]
    cluster_tr = cluster_arr[is_train]
    cluster_tg = cluster_arr[is_test]
    admin_tg = admin_codes[is_test]
    test_feature_nan = np.isnan(Xtg).any(axis=1)

    if test_feature_nan.any():
        # Replace NaN test features with column means from the training set so
        # sklearn doesn't refuse to predict. Track and surface affected rows.
        col_means = np.nanmean(Xtr, axis=0)
        Xtg = np.where(np.isnan(Xtg), col_means, Xtg)
        print(f"  WARNING: {int(test_feature_nan.sum())} test rows had NaN features; "
              "imputed with training-set column means.")

    # Step 5 — train pooled + partitioned.
    print(f"  training pooled RF on {len(ytr)} rows ...")
    pooled = RandomForestClassifier(**RF_PARAMS)
    pooled.fit(Xtr, ytr)
    print(f"  training partitioned RFs ...")
    partitioned = _train_partitioned(Xtr, ytr, cluster_tr)

    # Step 6 — predict + probabilities for the target month.
    pred_pool, prob_pool = _predict_class1(pooled, Xtg)
    _qc_assert_probs(prob_pool, 'pooled')
    _qc_assert_threshold(pred_pool, prob_pool, 'pooled')

    pred_part = np.zeros(Xtg.shape[0], dtype=int)
    prob_part = np.zeros(Xtg.shape[0], dtype=float)
    pred_source = np.empty(Xtg.shape[0], dtype=object)

    handled = np.zeros(Xtg.shape[0], dtype=bool)
    for cid, model in partitioned.items():
        m = (cluster_tg == cid)
        if not m.any():
            continue
        if model is None:
            pred_part[m] = pred_pool[m]
            prob_part[m] = prob_pool[m]
            pred_source[m] = 'pooled_fallback'
        else:
            p, pr = _predict_class1(model, Xtg[m])
            pred_part[m] = p
            prob_part[m] = pr
            pred_source[m] = 'partitioned'
        handled |= m
    # Unmapped or never-seen cluster ids -> pooled fallback.
    unhandled = ~handled
    if unhandled.any():
        pred_part[unhandled] = pred_pool[unhandled]
        prob_part[unhandled] = prob_pool[unhandled]
        pred_source[unhandled] = 'pooled_fallback'

    _qc_assert_probs(prob_part, 'partitioned')
    _qc_assert_threshold(pred_part, prob_part, 'partitioned')

    # Step 7 — assemble per-target frame.
    pred_final = pred_part
    prob_final = prob_part
    uncertainty = 1.0 - np.abs(prob_final - 0.5) * 2.0
    confidence = 1.0 - uncertainty
    confidence_band = np.where(
        confidence >= 0.7, 'high',
        np.where(confidence >= 0.4, 'medium', 'low')
    )

    out = pd.DataFrame({
        'FEWSNET_admin_code': admin_tg,
        'target_month': str(target_month),
        'feature_month': str(feature_month),
        'lag_months': lag_months,
        'forecasting_scope': scope,
        'pred_pooled': pred_pool,
        'prob_pooled': prob_pool,
        'pred_partitioned': pred_part,
        'prob_partitioned': prob_part,
        'pred_source': pred_source.astype(str),
        'pred_final': pred_final,
        'prob_final': prob_final,
        'agreement_with_pooled': pred_pool == pred_part,
        'uncertainty_prob': uncertainty,
        'confidence_score': confidence,
        'confidence_band': confidence_band,
        'cluster_id': cluster_tg,
        'feature_imputed': test_feature_nan,
    })

    # Coverage QC: one row per polygon, no duplicates, no nulls in pred_final.
    if out['FEWSNET_admin_code'].duplicated().any():
        dup = out.loc[out['FEWSNET_admin_code'].duplicated(), 'FEWSNET_admin_code'].head().tolist()
        raise AssertionError(f'Duplicate polygons in target output: {dup}')
    if out['pred_final'].isna().any():
        raise AssertionError('Null pred_final encountered')

    # Stash window info for the manifest.
    out.attrs['window'] = {
        'train_window_start': train_start.strftime('%Y-%m'),
        'requested_train_end': requested_end.strftime('%Y-%m'),
        'actual_labeled_train_end': str(actual_end_period),
        'n_train_rows': int(is_train.sum()),
        'n_target_rows': int(is_test.sum()),
        'n_pooled_fallback': int((out['pred_source'] == 'pooled_fallback').sum()),
    }
    return out


# ----------------------------------------------------------------------------
# Deliverables
# ----------------------------------------------------------------------------

def _write_d1_xlsx(per_target: List[pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        readme_rows = [
            ('FEWSNET_admin_code', 'Polygon identifier (matches Nigeria.shp admin_code)'),
            ('target_month', 'YYYY-MM the prediction is for'),
            ('feature_month', 'Most recent labeled month used as features (target - lag_months)'),
            ('lag_months', 'Forecast horizon in months'),
            ('forecasting_scope', '1 = fs1 (4mo), 3 = fs3 (12mo)'),
            ('pred_pooled', '0/1 from a single RF trained on the full train window'),
            ('prob_pooled', 'P(class=1) under the pooled model'),
            ('pred_partitioned', '0/1 from per-cluster RFs (using staged Stage 2 cluster mapping)'),
            ('prob_partitioned', 'P(class=1) under the partitioned model'),
            ('pred_source', 'partitioned (per-cluster RF) or pooled_fallback'),
            ('pred_final', 'pred_partitioned (or pooled if pred_source=pooled_fallback)'),
            ('prob_final', 'Probability associated with pred_final'),
            ('agreement_with_pooled', 'pred_pooled == pred_partitioned'),
            ('uncertainty_prob', '1 - |prob_final - 0.5| * 2  (1 = at threshold, 0 = certain)'),
            ('confidence_score', '1 - uncertainty_prob'),
            ('confidence_band', 'high (>=0.7), medium (0.4-0.7), low (<0.4) — high == high confidence'),
            ('cluster_id', 'Cluster id from Stage 2 mapping (or -1 if unmapped)'),
            ('feature_imputed', 'True if any test-time feature was NaN and replaced with train mean'),
        ]
        readme_df = pd.DataFrame(readme_rows, columns=['column', 'description'])
        meta_lines = [
            'Standalone pure-prediction deliverable.',
            'April 2026 / April 2027 are not FEWSNET publication months;',
            'these have been substituted with Jun 2026 (nowcast, fs1=lag4) and Feb 2027 (12-mo, fs3=lag12).',
            f'Threshold for class-1 prediction: {PREDICTION_THRESHOLD}.',
            'Forward-filled FEWSNET non-publication slots: ' + (
                ', '.join(IMPUTE_FEWSNET_GAPS) if IMPUTE_FEWSNET_GAPS else '(none)'
            ),
        ]
        meta_df = pd.DataFrame({'note': meta_lines})
        readme_df.to_excel(writer, sheet_name='README', index=False)
        meta_df.to_excel(writer, sheet_name='README', index=False,
                         startrow=len(readme_df) + 3, header=True)

        for tdf in per_target:
            target_str = str(tdf['target_month'].iloc[0])
            sheet = target_str.replace('-', '_')
            tdf.sort_values('FEWSNET_admin_code').to_excel(writer, sheet_name=sheet, index=False)
    print(f"  D1 spreadsheet -> {path}")


def _render_d2_maps(per_target: List[pd.DataFrame], shapefile: str, out_dir: Path) -> List[Path]:
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import Patch
    except ImportError as exc:
        print(f"  D2 maps skipped: {exc}")
        return []

    gdf = gpd.read_file(shapefile)
    admin_col = next(
        (c for c in ('admin_code', 'FEWSNET_admin_code', 'adm_code', 'area_id') if c in gdf.columns),
        None
    )
    if admin_col is None:
        raise ValueError('Shapefile is missing an admin code column')
    if admin_col != 'FEWSNET_admin_code':
        gdf = gdf.rename(columns={admin_col: 'FEWSNET_admin_code'})
    gdf = gdf[['FEWSNET_admin_code', 'geometry']].copy()
    gdf['FEWSNET_admin_code'] = gdf['FEWSNET_admin_code'].astype(str)

    paths: List[Path] = []
    for tdf in per_target:
        tdf = tdf.copy()
        tdf['FEWSNET_admin_code'] = tdf['FEWSNET_admin_code'].astype(str)
        merged = gdf.merge(tdf, on='FEWSNET_admin_code', how='left')
        merged_gdf = gpd.GeoDataFrame(merged, geometry='geometry')
        target_str = str(tdf['target_month'].iloc[0])

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Panel 1 — predicted class.
        cmap = mcolors.ListedColormap(['#2e8b57', '#d73027'])  # green = 1/2, red = 3+
        norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
        merged_gdf.plot(column='pred_final', ax=axes[0], cmap=cmap, norm=norm,
                        edgecolor='black', linewidth=0.2,
                        missing_kwds={'color': 'white', 'hatch': '////', 'edgecolor': 'gray'})
        axes[0].set_title(f'Predicted IPC class — {target_str}\n(red = Phase 3+, green = Phase 1/2)',
                          fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        legend_handles = [
            Patch(facecolor='#d73027', edgecolor='black', label='Phase 3+'),
            Patch(facecolor='#2e8b57', edgecolor='black', label='Phase 1/2'),
            Patch(facecolor='white', edgecolor='gray', hatch='////', label='No prediction'),
        ]
        axes[0].legend(handles=legend_handles, loc='lower left')

        # Panel 2 — uncertainty.
        merged_gdf.plot(column='uncertainty_prob', ax=axes[1], cmap='Greys',
                        vmin=0, vmax=1, edgecolor='black', linewidth=0.2,
                        legend=True, legend_kwds={'label': 'Uncertainty', 'shrink': 0.7},
                        missing_kwds={'color': 'white', 'hatch': '////', 'edgecolor': 'gray'})
        axes[1].set_title(f'Prediction uncertainty — {target_str}\n(0 = certain, 1 = at threshold)',
                          fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')

        plt.tight_layout()
        out_path = out_dir / f'map_phase3plus_{target_str}_georf.png'
        plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        paths.append(out_path)
        print(f"  D2 map -> {out_path}")
    return paths


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description='Pure-prediction pipeline for Jun 2026 + Feb 2027')
    parser.add_argument('--data', default=DEFAULT_DATA_PATH, help='Combined FEWSNET CSV')
    parser.add_argument('--partition-map', required=False, default=None,
                        help='Default cluster mapping CSV (used for any target without a scope-specific override).')
    parser.add_argument('--partition-map-fs0', default=None, help='Override cluster mapping for fs0 targets')
    parser.add_argument('--partition-map-fs1', default=None, help='Override cluster mapping for fs1 targets')
    parser.add_argument('--partition-map-fs2', default=None, help='Override cluster mapping for fs2 targets')
    parser.add_argument('--partition-map-fs3', default=None, help='Override cluster mapping for fs3 targets')
    parser.add_argument('--polygons', default=ADJACENCY_SHAPEFILE_PATH, help='Admin boundary shapefile')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--train-window', type=int, default=36)
    parser.add_argument('--gap-months', nargs='*', default=list(DEFAULT_GAP_MONTHS),
                        help='FEWSNET non-publication months to forward-fill (default 2025-02 2025-06)')
    parser.add_argument('--smoke', action='store_true',
                        help='Diagnostic mode: process only the first target')
    parser.add_argument('--target', action='append', default=None,
                        metavar='YYYY-MM:SCOPE',
                        help='Override default targets. Repeatable. Format: 2024-06:1')
    args = parser.parse_args()

    _seed_everything(RANDOM_STATE)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 80)
    print('STANDALONE PURE-PREDICTION PIPELINE (GeoRF)')
    print('=' * 80)
    print(f'Data:           {args.data}')
    print(f'Partition map:  {args.partition_map}')
    print(f'Shapefile:      {args.polygons}')
    print(f'Out dir:        {out_dir}')
    print(f'Gap months:     {args.gap_months}')
    print(f'Threshold:      {PREDICTION_THRESHOLD}')
    print('=' * 80)

    # Materialize placeholder rows for any target month missing from the CSV
    # (the combined input ends at 2026-04 but deliverables target Jun 2026 +
    # Feb 2027). Synthetic rows hold the polygon's latest-observed feature
    # snapshot; downstream prepare_features computes lag features as usual.
    requested_targets = (
        [t.split(':', 1)[0] for t in args.target] if args.target
        else [t['target_month'] for t in (DEFAULT_TARGETS[:1] if args.smoke else DEFAULT_TARGETS)]
    )
    extended_csv = out_dir / 'input_with_target_placeholders.csv'
    print(f'Extending CSV with placeholder rows for: {requested_targets}')
    synth_counts = _materialize_target_rows(args.data, requested_targets, str(extended_csv))
    for ym, n in synth_counts.items():
        print(f'  +{n} synthetic rows for {ym}')
    args.data = str(extended_csv)

    if args.target:
        targets = []
        for spec in args.target:
            if ':' not in spec:
                raise SystemExit(f"--target expects YYYY-MM:SCOPE, got {spec!r}")
            ym, sc = spec.split(':', 1)
            targets.append({'label': f'{ym} (override)', 'target_month': ym, 'forecasting_scope': int(sc)})
    else:
        targets = DEFAULT_TARGETS[:1] if args.smoke else DEFAULT_TARGETS
    per_target = []
    windows = []
    for target in targets:
        tdf = _predict_target(args, target)
        per_target.append(tdf)
        # intermediate CSV
        csv_path = out_dir / f"predictions_{target['target_month']}_georf.csv"
        tdf.to_csv(csv_path, index=False)
        print(f"  intermediate -> {csv_path}")
        windows.append({'target_month': target['target_month'], **tdf.attrs.get('window', {})})

    # D1
    target_tags = '_'.join(t['target_month'] for t in targets)
    d1_path = out_dir / f"predictions_{target_tags}_georf.xlsx"
    _write_d1_xlsx(per_target, d1_path)

    # D2
    map_paths = _render_d2_maps(per_target, args.polygons, out_dir)

    # Manifest
    repo_root = Path(__file__).resolve().parents[1]
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'data_path': args.data,
        'data_sha256_16': _sha256_head(args.data),
        'partition_map': args.partition_map,
        'shapefile': args.polygons,
        'targets': targets,
        'gap_months': args.gap_months,
        'prediction_threshold': PREDICTION_THRESHOLD,
        'random_state': RANDOM_STATE,
        'rf_params': RF_PARAMS,
        'windows': windows,
        'd1_path': str(d1_path),
        'd2_paths': [str(p) for p in map_paths],
        **_git_state(repo_root),
        'python_version': sys.version.split()[0],
    }
    manifest_path = out_dir / 'run_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nmanifest -> {manifest_path}")
    print('Done.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
