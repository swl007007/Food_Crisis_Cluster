#!/usr/bin/env python3
"""Scenario-based prediction overlay (Iran-oil + El-Nino) for Jun 2026 + Feb 2027.

Stand-alone wrapper around scripts/predict_partitioned_2026_2027.py. Re-runs
the standard training procedure on real (non-mutated) training data, then on
the prediction step copies the target feature matrix, injects price + weather
shocks for Greater-Horn-of-Africa polygons only, predicts at a lower threshold
(0.40 by default), and emits a separate set of deliverables that preserves
base predictions side-by-side.

This is a SYNTHETIC SCENARIO OVERLAY, NOT a probabilistic forecast. It is not
part of the standard pipeline. config.py, the standard script, and standard
deliverables are not modified.

Usage:
    python scripts/predict_scenario_2026_2027.py
    python scripts/predict_scenario_2026_2027.py --smoke
    python scripts/predict_scenario_2026_2027.py \
        --partition-map-fs1 .../fs1_general.csv \
        --partition-map-fs3 .../fs3_general.csv
"""

import argparse
import hashlib
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Repo + scripts path setup so we can import both src.* and the sibling script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from src.preprocess.preprocess import load_and_preprocess_data
from src.feature.feature import prepare_features
from src.utils.lag_schedules import forecasting_scope_to_lag
from config import LAGS_MONTHS, ADJACENCY_SHAPEFILE_PATH

# Helpers from the standard predict script (not a fork; explicit re-use).
from predict_partitioned_2026_2027 import (
    _materialize_target_rows,
    _resolve_train_window,
    _load_partition_mapping,
    _attach_clusters,
    _train_partitioned,
    _predict_class1,
    _qc_assert_probs,
    _render_d2_maps,
    _sha256_head,
    _git_state,
    _seed_everything,
    RANDOM_STATE,
    RF_PARAMS,
    DEFAULT_DATA_PATH,
    DEFAULT_GAP_MONTHS,
)

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

DEFAULT_OUT_DIR = r"deliverables/predict_scenario_jun2026_feb2027"

# Greater Horn of Africa + Uganda (per user choice). ISO3 codes.
# Note: ERI (Eritrea) and DJI (Djibouti) are not in the FEWSNET combined dataset
# and are dropped automatically with a runtime info-print rather than aborting.
DEFAULT_HORN_ISO = ('ETH', 'SOM', 'ERI', 'DJI', 'SDN', 'SSD', 'KEN', 'UGA')
ISO_CANDIDATE_COLS = ('ISO3', 'ISO', 'iso3', 'iso', 'ADMIN0_ISO', 'country_iso')

# Two scenario targets — same as standard pipeline.
DEFAULT_TARGETS = [
    {'label': 'Jun 2026 (nowcast)',  'target_month': '2026-06', 'forecasting_scope': 1},
    {'label': 'Feb 2027 (12-month)', 'target_month': '2027-02', 'forecasting_scope': 3},
]

# Probability tolerance for attribution (delta ~ 0 means "no shock effect").
ATTRIB_EPS = 1e-9

# Per-horizon graduation factor for graduated multipliers.
# multiplier_at_horizon = 1 + (M - 1) * factor
HORIZON_FACTORS = {
    'current': 1.00,
    'm4':      1.00,   # rolling-mean-of-recent-past treated as full shock
    'm12':     1.00,
    'lag4m':   0.75,
    'lag8m':   0.50,
    'lag12m':  0.25,
}

# Price columns to shock (current + rolling means + lag of any horizon present).
PRICE_BASE_COLS = ('WFP_Price', 'FAO_price')
PRICE_ROLLING_SUFFIXES = ('_m4', '_m12')
PRICE_LAG_SUFFIXES = ('_lag4m', '_lag8m', '_lag12m')

# Weather z-score columns to shift.
WEATHER_TEMP_BASES = ('Tair_zscore',)
WEATHER_RAIN_BASES = ('Rainf_zscore',)
WEATHER_LAG_SUFFIXES = ('_lag4m', '_lag8m', '_lag12m')


# ----------------------------------------------------------------------------
# Mutation helpers
# ----------------------------------------------------------------------------

def _detect_iso_column(source_df: pd.DataFrame) -> str:
    iso_col = next((c for c in ISO_CANDIDATE_COLS if c in source_df.columns), None)
    if iso_col is None:
        raise KeyError(
            f"No ISO-like column in source CSV. Tried: {list(ISO_CANDIDATE_COLS)}. "
            f"Available: {sorted(source_df.columns.tolist())}"
        )
    return iso_col


def _build_polygon_iso_map(source_df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame ['FEWSNET_admin_code' (str), 'iso' (str)] with uniqueness assertions."""
    iso_col = _detect_iso_column(source_df)
    src = source_df[['FEWSNET_admin_code', iso_col]].dropna().copy()
    src['FEWSNET_admin_code'] = src['FEWSNET_admin_code'].astype(str)
    src['iso'] = src[iso_col].astype(str).str.upper().str.strip()
    src = src[['FEWSNET_admin_code', 'iso']].drop_duplicates()
    n_multi = src.groupby('FEWSNET_admin_code')['iso'].nunique()
    if (n_multi > 1).any():
        bad = n_multi[n_multi > 1].head(10).to_dict()
        raise ValueError(f"Admin codes mapped to multiple ISOs: {bad}")
    return src


def _resolve_mutation_columns(
    feature_columns: List[str],
    lag_months: int,
    mutate_price_std: bool,
    force_price_std: bool,
    allow_missing: bool,
) -> Dict[str, List[Tuple[str, int, str]]]:
    """Resolve the actual feature-column names + indices to mutate, grouped by (kind, horizon).

    Returns dict with keys:
      'price': list of (col_name, col_index, horizon_key)
      'temp':  list of (col_name, col_index, horizon_key)
      'rain':  list of (col_name, col_index, horizon_key)
    """
    name_to_idx = {c: i for i, c in enumerate(feature_columns)}

    def add_strict(group, col_name, horizon_key):
        """Required column — abort (or warn under --allow-missing-scenario-cols) if absent."""
        if col_name in name_to_idx:
            group.append((col_name, name_to_idx[col_name], horizon_key))
        elif not allow_missing:
            raise KeyError(
                f"Required scenario column missing from feature matrix: {col_name!r}. "
                f"Pass --allow-missing-scenario-cols to demote to warning."
            )
        else:
            print(f"  WARNING: scenario column missing, skipping: {col_name}")

    def add_optional(group, col_name, horizon_key):
        """Optional column — silently add if present, skip otherwise."""
        if col_name in name_to_idx:
            group.append((col_name, name_to_idx[col_name], horizon_key))

    price: List[Tuple[str, int, str]] = []
    for base in PRICE_BASE_COLS:
        # Current-month price columns are strict requirements.
        add_strict(price, base, 'current')
        # Rolling means (_m4, _m12) and lag columns (_lag*m) are optional —
        # not every base has them, and prepare_features only adds the scope's
        # specific lag dynamically.
        for suf in PRICE_ROLLING_SUFFIXES:
            add_optional(price, base + suf, suf.lstrip('_'))
        for suf in PRICE_LAG_SUFFIXES:
            add_optional(price, base + suf, suf.lstrip('_'))

    # WFP_Price_std handling (default OFF, --mutate-price-std opt-in).
    std_col = 'WFP_Price_std'
    std_decision = 'not_mutated_default'
    if std_col in name_to_idx and mutate_price_std:
        # Heuristic: if it looks standardized (negatives, abs(mean) small),
        # we refuse unless --force-price-std-mutation is also set.
        # We can't actually inspect values here (only have names); the caller
        # will run a runtime guard with the matrix. Mark intent here and let
        # the caller decide.
        std_decision = ('mutated_opt_in_forced' if force_price_std
                        else 'mutated_opt_in_with_runtime_guard')
        price.append((std_col, name_to_idx[std_col], 'current'))

    temp: List[Tuple[str, int, str]] = []
    for base in WEATHER_TEMP_BASES:
        add_strict(temp, base, 'current')
        for suf in WEATHER_LAG_SUFFIXES:
            add_optional(temp, base + suf, suf.lstrip('_'))

    rain: List[Tuple[str, int, str]] = []
    for base in WEATHER_RAIN_BASES:
        add_strict(rain, base, 'current')
        for suf in WEATHER_LAG_SUFFIXES:
            add_optional(rain, base + suf, suf.lstrip('_'))

    return {'price': price, 'temp': temp, 'rain': rain, 'wfp_price_std_decision': std_decision}


def _multiplier(M: float, horizon_key: str) -> float:
    factor = HORIZON_FACTORS.get(horizon_key, 1.0)
    return 1.0 + (M - 1.0) * factor


def _zshift_for_horizon(z: float, horizon_key: str) -> float:
    return z * HORIZON_FACTORS.get(horizon_key, 1.0)


def _apply_price_shock(
    X_scenario: np.ndarray,
    horn_mask: np.ndarray,
    price_cols: List[Tuple[str, int, str]],
    M: float,
    std_decision: str,
    log_rows: List[Dict[str, Any]],
    target_month: str,
) -> None:
    """Mutate X_scenario in place on horn rows for price columns (graduated)."""
    if not horn_mask.any():
        return
    for col_name, col_idx, horizon_key in price_cols:
        if col_name == 'WFP_Price_std':
            # Runtime guard: looks-standardized => abort unless forced.
            col_view = X_scenario[horn_mask, col_idx]
            looks_standardized = (
                np.nanmin(col_view) < 0
                or abs(float(np.nanmean(col_view))) < 5.0
            )
            if looks_standardized and std_decision != 'mutated_opt_in_forced':
                raise RuntimeError(
                    f"WFP_Price_std looks standardized (min<0 or |mean|<5). "
                    f"Pass --force-price-std-mutation to override, or omit "
                    f"--mutate-price-std to leave it untouched."
                )
        mult = _multiplier(M, horizon_key)
        before = X_scenario[horn_mask, col_idx].copy()
        X_scenario[horn_mask, col_idx] = before * mult
        log_rows.append({
            'target_month': target_month,
            'kind': 'price',
            'column': col_name,
            'horizon': horizon_key,
            'multiplier': mult,
            'n_rows_mutated': int(horn_mask.sum()),
            'before_p50': float(np.nanmedian(before)),
            'after_p50': float(np.nanmedian(X_scenario[horn_mask, col_idx])),
            'before_max': float(np.nanmax(before)),
            'after_max': float(np.nanmax(X_scenario[horn_mask, col_idx])),
        })


def _apply_weather_shock(
    X_scenario: np.ndarray,
    horn_mask: np.ndarray,
    temp_cols: List[Tuple[str, int, str]],
    rain_cols: List[Tuple[str, int, str]],
    zshift: float,
    log_rows: List[Dict[str, Any]],
    target_month: str,
) -> None:
    """Mutate X_scenario in place on horn rows for weather z-score columns."""
    if not horn_mask.any():
        return

    # Temperature: always-positive additive shift (heat scenario), graduated.
    for col_name, col_idx, horizon_key in temp_cols:
        shift = _zshift_for_horizon(zshift, horizon_key)
        before = X_scenario[horn_mask, col_idx].copy()
        X_scenario[horn_mask, col_idx] = before + shift
        log_rows.append({
            'target_month': target_month,
            'kind': 'weather_temp',
            'column': col_name,
            'horizon': horizon_key,
            'shift': shift,
            'n_rows_mutated': int(horn_mask.sum()),
            'before_p50': float(np.nanmedian(before)),
            'after_p50': float(np.nanmedian(X_scenario[horn_mask, col_idx])),
            'before_p99': float(np.nanpercentile(before, 99)),
            'after_p99': float(np.nanpercentile(X_scenario[horn_mask, col_idx], 99)),
        })

    # Rainfall: sign-preserving amplification — drier gets drier, wetter wetter.
    for col_name, col_idx, horizon_key in rain_cols:
        shift_mag = _zshift_for_horizon(zshift, horizon_key)
        before = X_scenario[horn_mask, col_idx].copy()
        sign = np.where(before >= 0, 1.0, -1.0)  # z == 0 -> +1
        X_scenario[horn_mask, col_idx] = before + sign * shift_mag
        log_rows.append({
            'target_month': target_month,
            'kind': 'weather_rain',
            'column': col_name,
            'horizon': horizon_key,
            'shift_magnitude': shift_mag,
            'n_rows_mutated': int(horn_mask.sum()),
            'before_p50': float(np.nanmedian(before)),
            'after_p50': float(np.nanmedian(X_scenario[horn_mask, col_idx])),
        })


# ----------------------------------------------------------------------------
# Distribution diagnostics (resolved Issue #7 — separate price vs z-score)
# ----------------------------------------------------------------------------

def _summarize_col(arr: np.ndarray) -> Dict[str, float]:
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {'min': None, 'p1': None, 'p25': None, 'p50': None,
                'p75': None, 'p99': None, 'max': None, 'mean': None, 'std': None, 'n': 0}
    return {
        'min': float(np.min(arr)),
        'p1':  float(np.percentile(arr, 1)),
        'p25': float(np.percentile(arr, 25)),
        'p50': float(np.percentile(arr, 50)),
        'p75': float(np.percentile(arr, 75)),
        'p99': float(np.percentile(arr, 99)),
        'max': float(np.max(arr)),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'n': int(arr.size),
    }


def _diagnose_distributions(
    X_train: np.ndarray,
    X_target_base: np.ndarray,
    X_target_scenario: np.ndarray,
    cols: List[Tuple[str, int, str]],
    kind: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col_name, col_idx, horizon_key in cols:
        diag = {
            'horizon': horizon_key,
            'train': _summarize_col(X_train[:, col_idx]),
            'target_base': _summarize_col(X_target_base[:, col_idx]),
            'target_scenario': _summarize_col(X_target_scenario[:, col_idx]),
        }
        warns = []
        if kind == 'price':
            tmax = diag['train']['max']
            sp99 = diag['target_scenario']['p99']
            smax = diag['target_scenario']['max']
            if tmax is not None and smax is not None and smax > tmax * 5:
                warns.append(f"scenario_max ({smax:.1f}) > 5x train_max ({tmax:.1f})")
            if tmax is not None and sp99 is not None and sp99 > tmax:
                warns.append(f"scenario_p99 ({sp99:.1f}) > train_max ({tmax:.1f})")
        elif kind == 'zscore':
            sp99 = diag['target_scenario']['p99']
            if sp99 is not None and abs(sp99) > 4.0:
                warns.append(f"|scenario_p99| ({abs(sp99):.2f}) > 4 sigma")
        diag['warnings'] = warns
        if warns:
            print(f"  DIST-WARN {col_name}: {'; '.join(warns)}")
        out[col_name] = diag
    return out


# ----------------------------------------------------------------------------
# Attribution (resolved Issue #2 — explicit truth table)
# ----------------------------------------------------------------------------

def _attribute(
    pred_base_050: np.ndarray,
    pred_base_040: np.ndarray,
    pred_scenario_040: np.ndarray,
    delta_prob: np.ndarray,
) -> np.ndarray:
    b50 = pred_base_050.astype(int)
    b40 = pred_base_040.astype(int)
    s40 = pred_scenario_040.astype(int)
    dp = delta_prob

    abs_dp_small = np.abs(dp) < ATTRIB_EPS
    dp_pos = dp > ATTRIB_EPS
    dp_neg = dp < -ATTRIB_EPS

    conds = [
        (b50 == 0) & (s40 == 0),
        (b50 == 1) & (s40 == 1) & ~dp_neg,                       # unchanged_pos (incl. dp~=0 and dp>=0)
        (b50 == 0) & (b40 == 1) & (s40 == 1) & abs_dp_small,     # threshold-only (delta ~ 0)
        (b50 == 0) & (b40 == 1) & (s40 == 1) & dp_neg,           # threshold-only with weak counter-shock
        (b50 == 0) & (b40 == 0) & (s40 == 1) & dp_pos,           # shock-only
        (b50 == 0) & (b40 == 1) & (s40 == 1) & dp_pos,           # both
        (b50 == 1) & (b40 == 0) & (s40 == 0) & dp_neg,           # impossible (b40<b50 means prob<0.4 but >=0.5?), guard anyway
        (b50 == 1) & (b40 == 1) & (s40 == 0) & dp_neg,           # flipped_down_at_threshold_change
        (b50 == 1) & (s40 == 0) & dp_neg,                        # flipped_down_shock (catch-all for b40 path)
        (b50 == 1) & (s40 == 1) & dp_neg,                        # weakened_but_still_positive
    ]
    labels = [
        'unchanged_neg',
        'unchanged_pos',
        'flipped_up_threshold_only',
        'flipped_up_threshold_despite_weak_shock',
        'flipped_up_shock_only',
        'flipped_up_both',
        'attribution_uncategorized',  # the impossible branch
        'flipped_down_at_threshold_change',
        'flipped_down_shock',
        'weakened_but_still_positive',
    ]
    attribution = np.select(conds, labels, default='attribution_uncategorized')
    n_uncat = int((attribution == 'attribution_uncategorized').sum())
    if n_uncat > 0:
        print(f"  WARNING: {n_uncat} rows in attribution_uncategorized — review.")
    return attribution


# ----------------------------------------------------------------------------
# Per-target scenario prediction
# ----------------------------------------------------------------------------

def _scenario_predict_target(
    args: argparse.Namespace,
    target: Dict[str, Any],
    polygon_iso_map: pd.DataFrame,
    horn_iso: Tuple[str, ...],
    scenario_threshold: float,
    base_threshold: float,
    price_mult: float,
    zshift: float,
    mutation_log: List[Dict[str, Any]],
    diagnostics: Dict[str, Any],
) -> pd.DataFrame:
    target_month = pd.Period(target['target_month'], freq='M')
    scope = int(target['forecasting_scope'])
    lag_months = forecasting_scope_to_lag(scope, LAGS_MONTHS)
    feature_month = (target_month.start_time - pd.DateOffset(months=lag_months)).to_period('M')

    print('=' * 80)
    print(f"SCENARIO Target: {target['label']}  ->  {target_month}, fs{scope}, lag={lag_months}mo")
    print(f"  price_mult={price_mult}, zshift={zshift}, scenario_threshold={scenario_threshold}")
    print('=' * 80)

    # Step 1 — load with predict-only whitelist + gap forward-fill.
    df = load_and_preprocess_data(
        args.data,
        predict_target_months=[str(target_month)],
        impute_gap_months=tuple(args.gap_months) if args.gap_months else None,
    )
    print(f"  loaded df: {len(df)} rows, {df['FEWSNET_admin_code'].nunique()} polygons")

    # Step 2 — partitions (scope-specific override or default).
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

    # Step 3 — features.
    df = df.sort_values(['FEWSNET_admin_code', 'date']).reset_index(drop=True)
    cluster_arr = df['cluster_id'].fillna(-1).astype(int).to_numpy()
    if {'lat', 'lon'}.issubset(df.columns):
        X_loc = df[['lat', 'lon']].to_numpy()
    elif {'latitude', 'longitude'}.issubset(df.columns):
        X_loc = df[['latitude', 'longitude']].to_numpy()
    else:
        raise ValueError('df missing latitude/longitude columns')
    X, y, _, _, _, _, dates, feature_columns = prepare_features(
        df, cluster_arr, X_loc, forecasting_scope=scope
    )
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    admin_codes = df['FEWSNET_admin_code'].to_numpy()
    if len(dates) != len(admin_codes):
        raise AssertionError(
            f"dates ({len(dates)}) and admin_codes ({len(admin_codes)}) length mismatch"
        )

    # Step 4 — train/test split.
    train_start, requested_end, actual_end_period = _resolve_train_window(
        df, target_month, lag_months, args.train_window
    )
    train_end_excl = actual_end_period.start_time + pd.DateOffset(months=1)
    target_start = target_month.start_time
    target_end_excl = target_start + pd.DateOffset(months=1)

    is_train = (dates >= train_start) & (dates < train_end_excl) & pd.Series(y).notna().values
    is_test = (dates >= target_start) & (dates < target_end_excl)
    feature_nan = np.isnan(X).any(axis=1)
    is_train = is_train & ~feature_nan
    print(
        f"  train window: [{train_start:%Y-%m}, {actual_end_period}] "
        f"({int(is_train.sum())} rows)"
    )
    if is_test.sum() == 0:
        raise RuntimeError(f"No rows for target {target_month}")

    Xtr, ytr = X[is_train], y[is_train].astype(int)
    Xtg = X[is_test]
    cluster_tr = cluster_arr[is_train]
    cluster_tg = cluster_arr[is_test]
    admin_tg = admin_codes[is_test]
    test_feature_nan = np.isnan(Xtg).any(axis=1)
    if test_feature_nan.any():
        col_means = np.nanmean(Xtr, axis=0)
        Xtg = np.where(np.isnan(Xtg), col_means, Xtg)
        print(f"  WARNING: {int(test_feature_nan.sum())} test rows had NaN features; imputed.")

    # Step 5 — train pooled + partitioned (same training data, no synthetic).
    print(f"  training pooled RF on {len(ytr)} rows ...")
    pooled = RandomForestClassifier(**RF_PARAMS)
    pooled.fit(Xtr, ytr)
    print(f"  training partitioned RFs ...")
    partitioned = _train_partitioned(Xtr, ytr, cluster_tr)

    # ------------------------------------------------------------------
    # SCENARIO FORK begins here.
    # ------------------------------------------------------------------

    # Step 6a — build polygon -> ISO mapping for target rows.
    target_admin_str = pd.Series(admin_tg).astype(str)
    iso_lookup = polygon_iso_map.set_index('FEWSNET_admin_code')['iso']
    iso_for_target = target_admin_str.map(iso_lookup).to_numpy()
    n_missing_iso = int(pd.isna(iso_for_target).sum())
    if n_missing_iso > 0:
        raise ValueError(
            f"{n_missing_iso} target polygons have no ISO mapping. "
            f"Sample missing: {target_admin_str[pd.isna(iso_for_target)].head(5).tolist()}"
        )
    horn_set = set(s.upper() for s in horn_iso)
    available_iso = set(str(x).upper() for x in iso_for_target.tolist())
    requested_in_data = sorted(horn_set & available_iso)
    requested_missing = sorted(horn_set - available_iso)
    if requested_missing:
        print(f"  INFO: requested Horn ISOs absent from data (skipping): {requested_missing}")
    horn_mask = np.array([str(x).upper() in horn_set for x in iso_for_target])
    if horn_mask.sum() == 0:
        raise ValueError(
            f"No Horn polygons matched. Horn ISOs={horn_iso}. "
            f"Available ISOs in target: {sorted(available_iso)}"
        )
    print(f"  Horn polygons in target: {int(horn_mask.sum())} / {len(horn_mask)} "
          f"(active ISOs: {requested_in_data})")

    # Step 6b — resolve mutation columns.
    mut = _resolve_mutation_columns(
        feature_columns,
        lag_months,
        mutate_price_std=args.mutate_price_std,
        force_price_std=args.force_price_std_mutation,
        allow_missing=args.allow_missing_scenario_cols,
    )
    price_cols = mut['price']
    temp_cols = mut['temp']
    rain_cols = mut['rain']
    print(f"  mutation cols — price: {len(price_cols)}, temp: {len(temp_cols)}, rain: {len(rain_cols)}")
    print(f"  WFP_Price_std decision: {mut['wfp_price_std_decision']}")
    diagnostics.setdefault('wfp_price_std_decisions', {})[str(target_month)] = mut['wfp_price_std_decision']

    # Step 6c — copy + mutate.
    X_target_base = Xtg.copy()
    X_target_scenario = Xtg.copy()
    _apply_price_shock(
        X_target_scenario, horn_mask, price_cols, price_mult,
        mut['wfp_price_std_decision'], mutation_log, str(target_month),
    )
    _apply_weather_shock(
        X_target_scenario, horn_mask, temp_cols, rain_cols, zshift,
        mutation_log, str(target_month),
    )

    # Step 6d — non-Horn isolation assertion (only on mutated columns).
    mutated_idx = (
        [i for _, i, _ in price_cols]
        + [i for _, i, _ in temp_cols]
        + [i for _, i, _ in rain_cols]
    )
    if mutated_idx:
        non_horn = ~horn_mask
        np.testing.assert_array_equal(
            X_target_scenario[np.ix_(non_horn, mutated_idx)],
            X_target_base[np.ix_(non_horn, mutated_idx)],
        )
        print(f"  non-Horn isolation OK ({non_horn.sum()} rows × {len(mutated_idx)} cols, exact equal)")

    # Step 6e — distribution diagnostics (price + zscore separately).
    target_diag = diagnostics.setdefault('per_target', {}).setdefault(str(target_month), {})
    target_diag['price_columns'] = _diagnose_distributions(
        Xtr, X_target_base, X_target_scenario, price_cols, kind='price'
    )
    target_diag['zscore_columns'] = _diagnose_distributions(
        Xtr, X_target_base, X_target_scenario, temp_cols + rain_cols, kind='zscore'
    )

    # Step 7 — predict probabilities (base + scenario, pooled + partitioned).
    def _per_cluster_prob(model_pool: RandomForestClassifier,
                          parts: Dict[int, Optional[RandomForestClassifier]],
                          Xmat: np.ndarray) -> np.ndarray:
        _, prob_pool = _predict_class1(model_pool, Xmat)
        _qc_assert_probs(prob_pool, 'pooled')
        prob_part = np.zeros(Xmat.shape[0], dtype=float)
        handled = np.zeros(Xmat.shape[0], dtype=bool)
        for cid, m in parts.items():
            sel = (cluster_tg == cid)
            if not sel.any():
                continue
            if m is None:
                prob_part[sel] = prob_pool[sel]
            else:
                _, pr = _predict_class1(m, Xmat[sel])
                prob_part[sel] = pr
            handled |= sel
        unhandled = ~handled
        if unhandled.any():
            prob_part[unhandled] = prob_pool[unhandled]
        _qc_assert_probs(prob_part, 'partitioned')
        return prob_part

    prob_base = _per_cluster_prob(pooled, partitioned, X_target_base)
    prob_scenario = _per_cluster_prob(pooled, partitioned, X_target_scenario)

    # Hard guarantee: non-Horn rows get identical probs. (Not strictly required
    # if RF is deterministic and inputs are identical, but verify.)
    if (~horn_mask).any():
        max_non_horn_diff = float(np.max(np.abs(prob_scenario[~horn_mask] - prob_base[~horn_mask])))
        if max_non_horn_diff > 1e-10:
            raise AssertionError(
                f"Non-Horn rows differ in scenario vs base: max |delta| = {max_non_horn_diff}"
            )

    # Step 8 — apply thresholds + attribution.
    pred_base_050 = (prob_base >= base_threshold).astype(int)
    pred_base_040 = (prob_base >= scenario_threshold).astype(int)
    pred_scenario_040 = (prob_scenario >= scenario_threshold).astype(int)
    delta_prob = prob_scenario - prob_base
    attribution = _attribute(pred_base_050, pred_base_040, pred_scenario_040, delta_prob)

    pred_final = pred_scenario_040
    prob_final = prob_scenario
    uncertainty = 1.0 - np.abs(prob_final - scenario_threshold) * 2.0
    uncertainty = np.clip(uncertainty, 0.0, 1.0)
    confidence = 1.0 - uncertainty
    confidence_band = np.where(
        confidence >= 0.7, 'high',
        np.where(confidence >= 0.4, 'medium', 'low')
    )

    out = pd.DataFrame({
        'FEWSNET_admin_code': admin_tg,
        'iso': iso_for_target,
        'is_horn': horn_mask,
        'target_month': str(target_month),
        'feature_month': str(feature_month),
        'lag_months': lag_months,
        'forecasting_scope': scope,
        'cluster_id': cluster_tg,
        'prob_base': prob_base,
        'prob_scenario': prob_scenario,
        'pred_base_050': pred_base_050,
        'pred_base_040': pred_base_040,
        'pred_scenario_040': pred_scenario_040,
        'delta_prob': delta_prob,
        'attribution': attribution,
        'pred_final': pred_final,
        'prob_final': prob_final,
        'uncertainty_prob': uncertainty,
        'confidence_score': confidence,
        'confidence_band': confidence_band,
        'feature_imputed': test_feature_nan,
    })
    if out['FEWSNET_admin_code'].duplicated().any():
        raise AssertionError("Duplicate polygons in scenario output")

    # Soft-check reports.
    n_b50 = int(pred_base_050.sum())
    n_b40 = int(pred_base_040.sum())
    n_s40 = int(pred_scenario_040.sum())
    horn_dp = delta_prob[horn_mask]
    non_horn_dp = delta_prob[~horn_mask]
    print(
        f"  REPORT  base@0.50={n_b50}  base@{scenario_threshold:.2f}={n_b40}  "
        f"scenario@{scenario_threshold:.2f}={n_s40}"
    )
    print(
        f"  REPORT  delta_prob horn: mean={horn_dp.mean():.4f} median={np.median(horn_dp):.4f} "
        f"max|.|={np.max(np.abs(horn_dp)):.4f}"
    )
    if abs(float(non_horn_dp.mean() if non_horn_dp.size else 0.0)) > 1e-12:
        raise AssertionError("Non-Horn mean delta_prob != 0")
    if horn_dp.mean() <= 0:
        print(f"  WARNING  mean delta_prob_horn ({horn_dp.mean():.4f}) <= 0 — investigate shock direction")
    if n_s40 < n_b50:
        print(f"  WARNING  scenario crisis count ({n_s40}) < base@0.50 ({n_b50}) — unexpected")

    # Stash window info for the manifest.
    out.attrs['window'] = {
        'train_window_start': train_start.strftime('%Y-%m'),
        'requested_train_end': requested_end.strftime('%Y-%m'),
        'actual_labeled_train_end': str(actual_end_period),
        'n_train_rows': int(is_train.sum()),
        'n_target_rows': int(is_test.sum()),
        'n_horn': int(horn_mask.sum()),
        'n_non_horn': int((~horn_mask).sum()),
        'price_mult': price_mult,
        'zshift': zshift,
        'scenario_threshold': scenario_threshold,
        'counts': {
            'base_phase3_at_050': n_b50,
            'base_phase3_at_040': n_b40,
            'scenario_phase3_at_040': n_s40,
        },
    }
    return out


# ----------------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------------

def _write_attribution_csv(per_target_df: pd.DataFrame, out_path: Path) -> None:
    target_month = str(per_target_df['target_month'].iloc[0])
    horn = per_target_df[per_target_df['is_horn']]
    non_horn = per_target_df[~per_target_df['is_horn']]
    cat_counts = per_target_df['attribution'].value_counts().to_dict()

    # Build a flat summary.
    rows = [
        ('target_month', target_month),
        ('n_polygons', len(per_target_df)),
        ('n_horn', len(horn)),
        ('n_non_horn', len(non_horn)),
        ('base_phase3_at_050', int(per_target_df['pred_base_050'].sum())),
        ('base_phase3_at_040', int(per_target_df['pred_base_040'].sum())),
        ('scenario_phase3_at_040', int(per_target_df['pred_scenario_040'].sum())),
        ('mean_delta_prob_horn', float(horn['delta_prob'].mean()) if len(horn) else 0.0),
        ('median_delta_prob_horn', float(horn['delta_prob'].median()) if len(horn) else 0.0),
        ('max_abs_delta_prob_horn', float(horn['delta_prob'].abs().max()) if len(horn) else 0.0),
        ('mean_delta_prob_non_horn', float(non_horn['delta_prob'].mean()) if len(non_horn) else 0.0),
        ('n_horn_flipped_down', int(
            horn['attribution'].isin(
                ['flipped_down_shock', 'flipped_down_at_threshold_change',
                 'weakened_but_still_positive']
            ).sum()
        )),
    ]
    for label in [
        'unchanged_neg', 'unchanged_pos',
        'flipped_up_threshold_only', 'flipped_up_threshold_despite_weak_shock',
        'flipped_up_shock_only', 'flipped_up_both',
        'flipped_down_shock', 'flipped_down_at_threshold_change',
        'weakened_but_still_positive', 'attribution_uncategorized',
    ]:
        rows.append((f'count_{label}', int(cat_counts.get(label, 0))))

    pd.DataFrame(rows, columns=['metric', 'value']).to_csv(out_path, index=False)
    print(f"  attribution -> {out_path}")


def _write_d1_xlsx(
    per_target: List[pd.DataFrame],
    path: Path,
    scenario_threshold: float,
    base_threshold: float,
    horn_iso: Tuple[str, ...],
    price_mult_fs1: float,
    price_mult_fs3: float,
    zshift_fs1: float,
    zshift_fs3: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        readme_rows = [
            ('FEWSNET_admin_code', 'Polygon identifier'),
            ('iso', 'ISO country code (joined from source CSV)'),
            ('is_horn', 'True if polygon ISO is in the Horn-of-Africa scenario set'),
            ('target_month', 'YYYY-MM the prediction is for'),
            ('feature_month', 'Most recent labeled month used as features'),
            ('lag_months', 'Forecast horizon in months'),
            ('forecasting_scope', '1 = fs1 (4mo), 3 = fs3 (12mo)'),
            ('cluster_id', 'Cluster id from Stage 2 mapping (or -1 if unmapped)'),
            ('prob_base', 'Class-1 probability with NO scenario mutation (threshold-independent)'),
            ('prob_scenario', 'Class-1 probability AFTER scenario mutation; equals prob_base for non-Horn rows'),
            ('pred_base_050', '(prob_base >= 0.50) — what standard pipeline emits'),
            ('pred_base_040', f'(prob_base >= {scenario_threshold:.2f}) — threshold-only change'),
            ('pred_scenario_040', f'(prob_scenario >= {scenario_threshold:.2f}) — what the D2 map shows'),
            ('delta_prob', 'prob_scenario - prob_base (zero for non-Horn)'),
            ('attribution', 'See attribution truth table in scenario manifest'),
            ('pred_final', 'Same as pred_scenario_040 (the deliverable column)'),
            ('prob_final', 'Same as prob_scenario'),
            ('uncertainty_prob', '1 - |prob_final - scenario_threshold| * 2'),
            ('confidence_score', '1 - uncertainty_prob'),
            ('confidence_band', 'high (>=0.7), medium (0.4-0.7), low (<0.4)'),
            ('feature_imputed', 'True if any test-time feature was NaN and replaced with train mean'),
        ]
        readme_df = pd.DataFrame(readme_rows, columns=['column', 'description'])
        meta_lines = [
            '*** SYNTHETIC SCENARIO OVERLAY — NOT A PROBABILISTIC FORECAST ***',
            'Standard training procedure executed on real (non-mutated) training data.',
            'Scenario mutation applied only to target-row feature matrix at predict time.',
            'Hand-set Iran-oil and El-Nino shocks per user domain knowledge.',
            'NOT part of the standard pipeline. Do not redistribute alongside standard',
            'production output without this disclaimer.',
            '',
            f'Scenario threshold: {scenario_threshold} (base threshold for reference: {base_threshold})',
            f'Horn-of-Africa ISOs: {", ".join(horn_iso)}',
            f'Price multiplier — fs1 (Jun 2026): {price_mult_fs1}x; fs3 (Feb 2027): {price_mult_fs3}x',
            f'Weather z-shift — fs1: +{zshift_fs1}; fs3: +{zshift_fs3}',
            'Price columns mutated: WFP_Price* and FAO_price* (graduated by lag horizon).',
            'Weather: Tair_zscore additively (always positive, "more heat"); Rainf_zscore',
            '  sign-preservingly amplified ("dry gets drier, wet gets wetter").',
        ]
        meta_df = pd.DataFrame({'note': meta_lines})
        readme_df.to_excel(writer, sheet_name='README', index=False)
        meta_df.to_excel(writer, sheet_name='README', index=False,
                         startrow=len(readme_df) + 3, header=True)
        for tdf in per_target:
            sheet = str(tdf['target_month'].iloc[0]).replace('-', '_')
            tdf.sort_values('FEWSNET_admin_code').to_excel(
                writer, sheet_name=sheet, index=False
            )
    print(f"  D1 spreadsheet -> {path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description='SCENARIO predict overlay (Iran-oil + El-Nino)')
    # Standard-pipeline plumbing
    parser.add_argument('--data', default=DEFAULT_DATA_PATH)
    parser.add_argument('--partition-map', default=None)
    parser.add_argument('--partition-map-fs0', default=None)
    parser.add_argument('--partition-map-fs1', default=None)
    parser.add_argument('--partition-map-fs2', default=None)
    parser.add_argument('--partition-map-fs3', default=None)
    parser.add_argument('--polygons', default=ADJACENCY_SHAPEFILE_PATH)
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--train-window', type=int, default=36)
    parser.add_argument('--gap-months', nargs='*', default=list(DEFAULT_GAP_MONTHS))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--target', action='append', default=None,
                        metavar='YYYY-MM:SCOPE')
    # Scenario knobs
    parser.add_argument('--scenario-threshold', type=float, default=0.40)
    parser.add_argument('--base-threshold', type=float, default=0.50,
                        help='Reference threshold for base prediction column')
    parser.add_argument('--price-mult-fs1', type=float, default=2.0)
    parser.add_argument('--price-mult-fs3', type=float, default=3.0)
    parser.add_argument('--zshift-fs1', type=float, default=1.0)
    parser.add_argument('--zshift-fs3', type=float, default=1.5)
    parser.add_argument('--horn-iso', default=','.join(DEFAULT_HORN_ISO),
                        help='Comma-separated ISO codes (default: ET,SO,ER,DJ,SD,SS,KE,UG)')
    parser.add_argument('--allow-missing-scenario-cols', action='store_true')
    parser.add_argument('--mutate-price-std', action='store_true',
                        help='Opt-in: mutate WFP_Price_std (default OFF for safety)')
    parser.add_argument('--force-price-std-mutation', action='store_true',
                        help='Override the looks-standardized guard for WFP_Price_std')
    args = parser.parse_args()

    horn_iso = tuple(s.strip().upper() for s in args.horn_iso.split(',') if s.strip())
    if len(horn_iso) == 0:
        raise SystemExit("--horn-iso must be a non-empty comma-separated list")

    _seed_everything(RANDOM_STATE)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 80)
    print('SCENARIO PREDICTION OVERLAY (synthetic; not part of standard pipeline)')
    print('=' * 80)
    print(f'Data:           {args.data}')
    print(f'Out dir:        {out_dir}')
    print(f'Horn ISOs:      {horn_iso}')
    print(f'Scenario thr:   {args.scenario_threshold}')
    print(f'fs1 mult / z:   {args.price_mult_fs1} / +{args.zshift_fs1}')
    print(f'fs3 mult / z:   {args.price_mult_fs3} / +{args.zshift_fs3}')
    print(f'mutate_price_std: {args.mutate_price_std}  force: {args.force_price_std_mutation}')
    print('=' * 80)

    # Materialize placeholder rows for any target month missing from CSV.
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

    # Build polygon -> ISO map ONCE from the extended CSV.
    print('Building polygon -> ISO map from source CSV ...')
    src_for_iso = pd.read_csv(extended_csv, usecols=lambda c: (
        c == 'FEWSNET_admin_code' or c in ISO_CANDIDATE_COLS
    ))
    polygon_iso_map = _build_polygon_iso_map(src_for_iso)
    print(f'  polygon_iso_map: {len(polygon_iso_map)} unique polygons')
    print(f'  ISO codes present: {sorted(polygon_iso_map["iso"].unique().tolist())[:20]}')

    # Resolve targets list.
    if args.target:
        targets = []
        for spec in args.target:
            if ':' not in spec:
                raise SystemExit(f"--target expects YYYY-MM:SCOPE, got {spec!r}")
            ym, sc = spec.split(':', 1)
            targets.append({'label': f'{ym} (override)', 'target_month': ym, 'forecasting_scope': int(sc)})
    else:
        targets = DEFAULT_TARGETS[:1] if args.smoke else DEFAULT_TARGETS

    mutation_log: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {}
    per_target: List[pd.DataFrame] = []
    windows: List[Dict[str, Any]] = []

    for target in targets:
        scope = int(target['forecasting_scope'])
        # Per-scope multipliers.
        if scope == 1:
            pmult, zsh = args.price_mult_fs1, args.zshift_fs1
        elif scope == 3:
            pmult, zsh = args.price_mult_fs3, args.zshift_fs3
        else:
            # Use fs1 defaults for any unsupported scope and warn.
            print(f"  WARNING: scope {scope} not in {{1, 3}}; using fs1 multipliers")
            pmult, zsh = args.price_mult_fs1, args.zshift_fs1

        tdf = _scenario_predict_target(
            args, target, polygon_iso_map, horn_iso,
            scenario_threshold=args.scenario_threshold,
            base_threshold=args.base_threshold,
            price_mult=pmult,
            zshift=zsh,
            mutation_log=mutation_log,
            diagnostics=diagnostics,
        )
        per_target.append(tdf)
        # Per-target CSV.
        csv_path = out_dir / f"predictions_scenario_{target['target_month']}_georf.csv"
        tdf.to_csv(csv_path, index=False)
        print(f"  intermediate -> {csv_path}")
        # Attribution CSV.
        attrib_path = out_dir / f"attribution_{target['target_month']}.csv"
        _write_attribution_csv(tdf, attrib_path)
        windows.append({'target_month': target['target_month'], **tdf.attrs.get('window', {})})

    # D1 xlsx.
    target_tags = '_'.join(t['target_month'] for t in targets)
    d1_path = out_dir / f"predictions_scenario_{target_tags}_georf.xlsx"
    _write_d1_xlsx(
        per_target, d1_path,
        scenario_threshold=args.scenario_threshold,
        base_threshold=args.base_threshold,
        horn_iso=horn_iso,
        price_mult_fs1=args.price_mult_fs1,
        price_mult_fs3=args.price_mult_fs3,
        zshift_fs1=args.zshift_fs1,
        zshift_fs3=args.zshift_fs3,
    )

    # D2 maps (renamed for scenario).
    import shutil
    import time
    map_paths: List[Path] = []
    rendered = _render_d2_maps(per_target, args.polygons, out_dir)
    # Standard renderer writes map_phase3plus_<target>_georf.png. Copy to a
    # scenario-tagged name (rename can fail on Windows when matplotlib leaves
    # a brief file handle); retry copy a couple of times if needed.
    for p in rendered:
        scenario_p = p.with_name(p.name.replace('map_phase3plus_', 'map_phase3plus_scenario_'))
        last_err = None
        for attempt in range(3):
            try:
                shutil.copyfile(p, scenario_p)
                last_err = None
                break
            except OSError as exc:
                last_err = exc
                time.sleep(0.5)
        if last_err is not None:
            print(f"  WARNING: could not copy {p} -> {scenario_p}: {last_err}")
            map_paths.append(p)
            continue
        try:
            p.unlink()
        except OSError:
            pass  # leftover original is harmless
        map_paths.append(scenario_p)
        print(f"  D2 map renamed -> {scenario_p}")

    # Mutation log + diagnostics.
    if mutation_log:
        mlog_df = pd.DataFrame(mutation_log)
        for ym in mlog_df['target_month'].unique():
            sub = mlog_df[mlog_df['target_month'] == ym]
            mlog_path = out_dir / f"feature_mutation_log_{ym}.csv"
            sub.to_csv(mlog_path, index=False)
            print(f"  mutation log -> {mlog_path}")
    diag_path = out_dir / 'feature_distribution_diagnostics.json'
    with open(diag_path, 'w') as f:
        json.dump(diagnostics, f, indent=2, default=str)
    print(f"  diagnostics -> {diag_path}")

    # Manifest.
    repo_root = Path(__file__).resolve().parents[1]
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'kind': 'scenario_overlay',
        'disclaimer': (
            'SYNTHETIC SCENARIO OVERLAY. Not a probabilistic forecast. Standard '
            'training procedure executed on real (non-mutated) training data; '
            'scenario mutation applied only to target-row feature matrix at '
            'predict time. Hand-set Iran-oil and El-Nino shocks per user '
            'domain knowledge. Not part of the standard pipeline. Do not '
            'redistribute alongside standard production output without this '
            'disclaimer.'
        ),
        'data_path': args.data,
        'data_sha256_16': _sha256_head(args.data),
        'partition_map': args.partition_map,
        'partition_maps_per_scope': {
            'fs0': args.partition_map_fs0,
            'fs1': args.partition_map_fs1,
            'fs2': args.partition_map_fs2,
            'fs3': args.partition_map_fs3,
        },
        'shapefile': args.polygons,
        'targets': targets,
        'gap_months': args.gap_months,
        'horn_iso': list(horn_iso),
        'scenario_threshold': args.scenario_threshold,
        'base_threshold': args.base_threshold,
        'price_mult_fs1': args.price_mult_fs1,
        'price_mult_fs3': args.price_mult_fs3,
        'zshift_fs1': args.zshift_fs1,
        'zshift_fs3': args.zshift_fs3,
        'horizon_factors': HORIZON_FACTORS,
        'wfp_price_std_mutated': args.mutate_price_std,
        'wfp_price_std_force_flag': args.force_price_std_mutation,
        'wfp_price_std_decisions': diagnostics.get('wfp_price_std_decisions', {}),
        'random_state': RANDOM_STATE,
        'rf_params': RF_PARAMS,
        'windows': windows,
        'd1_path': str(d1_path),
        'd2_paths': [str(p) for p in map_paths],
        'attribution_files': [
            str(out_dir / f"attribution_{t['target_month']}.csv") for t in targets
        ],
        'mutation_log_files': [
            str(out_dir / f"feature_mutation_log_{t['target_month']}.csv") for t in targets
        ],
        'diagnostics_file': str(diag_path),
        **_git_state(repo_root),
        'python_version': sys.version.split()[0],
    }
    manifest_path = out_dir / 'run_manifest_scenario.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nmanifest -> {manifest_path}")
    print('Done. SCENARIO overlay deliverables ready.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
