"""rich561 feature construction at each row's own origin O = T - H (R21, R22).

Bounded local copies, adapted to the >=0.20 crisis truth:

* original93 -- ``IPCCHGeoRFExperiment/prepare_data.py`` raw whitelist,
  ``load_covariate_panel``, ``_PanelGrid``, ``_WindowSummer``, ``_as_of_index``
  and ``assemble_feature_matrix``. The three label-history and two recency
  columns read the new binary crisis truth (phase >= 3).
* history468 -- ``IPCCHPopulationHistoryExperiment/prepare_data.py``
  ``compute_series``, ``build_history_index``, ``_window_statistics`` and
  ``build_history_block``. Binary states are the new crisis truth; q2..q5 are
  the ledger's exact-derived normalized shares.

Infinity policy (implement.md clarification 3): raw covariate and assembled
original93 infinities become NaN with a column audit; any engineered history
infinity stops preparation. Missing history stays NaN for native XGBoost
handling; nothing is imputed, interpolated or forward-filled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ipcch_geoxgb.errors import ContractError

# --------------------------------------------------------------------------
# original93 inventory (order is checked against config/feature-schema.json)
# --------------------------------------------------------------------------

RAW_CONFLICT_COLUMNS = (
    "distance_to_nearest_acled",
    "event_count_battles",
    "event_count_battles_w5",
    "event_count_battles_w10",
    "event_count_explosions",
    "event_count_explosions_w5",
    "event_count_explosions_w10",
    "event_count_violence",
    "event_count_violence_w5",
    "event_count_violence_w10",
    "sum_fatalities_battles",
    "sum_fatalities_battles_w5",
    "sum_fatalities_battles_w10",
    "sum_fatalities_explosions",
    "sum_fatalities_explosions_w5",
    "sum_fatalities_explosions_w10",
    "sum_fatalities_violence",
    "sum_fatalities_violence_w5",
    "sum_fatalities_violence_w10",
)
RAW_PRICE_MACRO_COLUMNS = (
    "FAO_price", "WFP_Price", "WFP_Price_std", "CPI", "GDP", "CC", "gini", "Food_CPI", "Food_food_inflation",
)
RAW_VEGETATION_WEATHER_COLUMNS = (
    "EVI_mean", "GPP_mean", "Rainf_f_tavg_mean", "Tair_f_tavg_mean", "nightlight_mean", "nightlight_std",
)
RAW_LAND_ACCESS_COLUMNS = (
    "crop", "range", "distance_to_river", "elevation", "market_distance", "market_access", "ruggedness",
    "slope", "sg_cec_5-15cm", "sg_cfvo_5-15cm", "sg_nitrogen_5-15cm", "sg_phh2o_5-15cm", "sg_soc_5-15cm",
)
RAW_AEZ_COLUMNS = tuple(
    f"AEZ_{code}"
    for code in (4000, 7000, 9000, 10000, 12000, 17000, 19000, 20000, 25000, 28000, 30000,
                 31000, 32000, 33000, 34000, 35000, 36000, 38000, 40000, 42000, 43000)
)
RAW_COORDINATE_COLUMNS = ("lat", "lon")
RAW_FEATURE_COLUMNS = (
    RAW_CONFLICT_COLUMNS
    + RAW_PRICE_MACRO_COLUMNS
    + RAW_VEGETATION_WEATHER_COLUMNS
    + RAW_LAND_ACCESS_COLUMNS
    + RAW_AEZ_COLUMNS
    + RAW_COORDINATE_COLUMNS
)
#: ``(output_name, source_column, window_length)``; windows end at O inclusive.
SUM_DERIVATIVES = (
    ("WFP_Price_sum4_asof", "WFP_Price", 4),
    ("WFP_Price_sum12_asof", "WFP_Price", 12),
    ("nightlight_mean_sum12_asof", "nightlight_mean", 12),
)
LAG_DERIVATIVE_SOURCE = "EVI_mean"
LAG_DERIVATIVE_MAX = 12
DERIVED_FEATURE_COLUMNS = tuple(name for name, _, _ in SUM_DERIVATIVES) + tuple(
    f"{LAG_DERIVATIVE_SOURCE}_lag{k}_asof" for k in range(1, LAG_DERIVATIVE_MAX + 1)
)
CALENDAR_FEATURE_COLUMNS = ("target_month_sin", "target_month_cos")
#: Latest observed binary crisis status at O (including 0), its age, no-history flag.
HISTORY_FEATURE_COLUMNS = ("last_observed_label", "last_observed_label_age_months", "no_observed_label_history")
#: Recency of the latest observed crisis (phase >= 3).
RECENCY_FEATURE_COLUMNS = ("months_since_last_observed_crisis", "no_prior_observed_crisis")
HORIZON_FEATURE_COLUMNS = ("horizon_months",)
ORIGINAL_FEATURES = (
    RAW_FEATURE_COLUMNS
    + DERIVED_FEATURE_COLUMNS
    + CALENDAR_FEATURE_COLUMNS
    + HISTORY_FEATURE_COLUMNS
    + RECENCY_FEATURE_COLUMNS
    + HORIZON_FEATURE_COLUMNS
)
assert len(RAW_FEATURE_COLUMNS) == 70 and len(DERIVED_FEATURE_COLUMNS) == 15
assert len(ORIGINAL_FEATURES) == 93 == len(set(ORIGINAL_FEATURES))

_KEY_SCALE = 1_000_000


# --------------------------------------------------------------------------
# Calendar arithmetic
# --------------------------------------------------------------------------


def month_ordinal(year, month):
    """``year * 12 + (month - 1)``: ordinal differences are calendar months."""
    year_arr = np.asarray(year, dtype=np.int64)
    month_arr = np.asarray(month, dtype=np.int64)
    if np.any((month_arr < 1) | (month_arr > 12)):
        raise ContractError("month outside 1..12 while computing ordinals")
    return year_arr * 12 + (month_arr - 1)


def month_label(ordinal) -> np.ndarray:
    """``YYYY-MM`` strings; ``-1`` renders empty."""
    ordinal = np.asarray(ordinal, dtype=np.int64)
    out = np.array([f"{y:04d}-{m:02d}" for y, m in zip(ordinal // 12, ordinal % 12 + 1)], dtype=object)
    out[ordinal < 0] = ""
    return out


def parse_month(label: str) -> int:
    year, month = label.split("-")
    return int(month_ordinal(int(year), int(month)))


# --------------------------------------------------------------------------
# Covariate panel and dense calendar addressing
# --------------------------------------------------------------------------


def load_covariate_panel(path: Path | str) -> tuple[pd.DataFrame, dict]:
    """Keys plus the 70 raw whitelist columns for every source area-month.

    Non-numeric tokens become NaN and are counted; infinities become NaN with
    an audit. Unlabeled scaffold months stay: they supply covariates only.
    """
    usecols = ["admin_code", "year", "month"] + list(RAW_FEATURE_COLUMNS)
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    panel = pd.DataFrame(
        {
            "admin_code": frame["admin_code"].astype(np.int64),
            "month_ord": month_ordinal(frame["year"], frame["month"]),
        }
    )
    coerced: dict[str, int] = {}
    infinite: dict[str, int] = {}
    for name in RAW_FEATURE_COLUMNS:
        column = frame[name]
        if column.dtype == object:
            numeric = pd.to_numeric(column, errors="coerce")
            lost = int((numeric.isna() & column.notna() & (column != "")).sum())
            if lost:
                coerced[name] = lost
        else:
            numeric = column
        values = numeric.to_numpy(dtype=np.float64, copy=True)
        mask = np.isinf(values)
        if mask.any():
            values[mask] = np.nan
            infinite[name] = int(mask.sum())
        panel[name] = values
    if panel.duplicated(["admin_code", "month_ord"]).any():
        raise ContractError("(admin_code, year, month) is not unique in the panel")
    panel = panel.sort_values(["admin_code", "month_ord"], kind="mergesort").reset_index(drop=True)
    audit = {
        "panel_rows": int(len(panel)),
        "panel_areas": int(panel["admin_code"].nunique()),
        "panel_month_min": month_label([int(panel["month_ord"].min())])[0],
        "panel_month_max": month_label([int(panel["month_ord"].max())])[0],
        "non_numeric_coerced_to_nan": coerced,
        "source_infinities_converted": infinite,
    }
    return panel, audit


@dataclass(frozen=True)
class PanelGrid:
    """Dense (area, calendar month) -> panel row position, -1 where absent."""

    areas: np.ndarray
    month_lo: int
    positions: np.ndarray

    @property
    def n_months(self) -> int:
        return int(self.positions.shape[1])

    def area_index(self, admin_code: np.ndarray) -> np.ndarray:
        codes = np.asarray(admin_code, dtype=np.int64)
        guess = np.clip(np.searchsorted(self.areas, codes), 0, len(self.areas) - 1)
        return np.where(self.areas[guess] == codes, guess, -1)

    def positions_at(self, area_idx: np.ndarray, month_ord: np.ndarray) -> np.ndarray:
        month_idx = np.asarray(month_ord, dtype=np.int64) - self.month_lo
        inside = (area_idx >= 0) & (month_idx >= 0) & (month_idx < self.n_months)
        out = np.full(len(month_idx), -1, dtype=np.int64)
        if inside.any():
            out[inside] = self.positions[area_idx[inside], month_idx[inside]]
        return out


def build_panel_grid(panel: pd.DataFrame) -> PanelGrid:
    admin = panel["admin_code"].to_numpy(dtype=np.int64)
    ords = panel["month_ord"].to_numpy(dtype=np.int64)
    areas = np.unique(admin)
    month_lo, month_hi = int(ords.min()), int(ords.max())
    positions = np.full((len(areas), month_hi - month_lo + 1), -1, dtype=np.int64)
    positions[np.searchsorted(areas, admin), ords - month_lo] = np.arange(len(panel), dtype=np.int64)
    return PanelGrid(areas=areas, month_lo=month_lo, positions=positions)


def _gather(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    safe = np.where(positions >= 0, positions, 0)
    taken = values[safe]
    if taken.ndim == 1:
        return np.where(positions >= 0, taken, np.nan)
    taken = taken.astype(np.float64, copy=True)
    taken[positions < 0, :] = np.nan
    return taken


class WindowSummer:
    """Inclusive calendar window sums that require every month to be observed."""

    def __init__(self, grid: PanelGrid, values: np.ndarray):
        dense = np.full(grid.positions.shape, np.nan, dtype=np.float64)
        present = grid.positions >= 0
        dense[present] = values[grid.positions[present]]
        self._observed = ~np.isnan(dense)
        self._filled = np.where(self._observed, dense, 0.0)
        self._grid = grid

    def sum_ending_at(self, area_idx: np.ndarray, end_ord: np.ndarray, length: int) -> np.ndarray:
        end_idx = np.asarray(end_ord, dtype=np.int64) - self._grid.month_lo
        total = np.zeros(len(end_idx), dtype=np.float64)
        count = np.zeros(len(end_idx), dtype=np.int64)
        for offset in range(length - 1, -1, -1):  # oldest month first
            month_idx = end_idx - offset
            inside = (area_idx >= 0) & (month_idx >= 0) & (month_idx < self._grid.n_months)
            rows, cols = area_idx[inside], month_idx[inside]
            total[inside] += self._filled[rows, cols]
            count[inside] += self._observed[rows, cols]
        return np.where(count == length, total, np.nan)


def as_of_index(hist_area, hist_ord, query_area, query_ord) -> np.ndarray:
    """Index of the latest same-area entry with month <= query month, else -1."""
    hist_area = np.asarray(hist_area, dtype=np.int64)
    hist_ord = np.asarray(hist_ord, dtype=np.int64)
    query_area = np.asarray(query_area, dtype=np.int64)
    query_ord = np.asarray(query_ord, dtype=np.int64)
    if len(hist_area) == 0:
        return np.full(len(query_area), -1, dtype=np.int64)
    if np.any(hist_ord < 0) or np.any(hist_ord >= _KEY_SCALE):
        raise ContractError("month ordinal outside the packed as-of key range")
    order = np.lexsort((hist_ord, hist_area))
    sorted_area = hist_area[order]
    sorted_key = sorted_area * _KEY_SCALE + hist_ord[order]
    query_key = query_area * _KEY_SCALE + np.maximum(query_ord, 0)
    slot = np.searchsorted(sorted_key, query_key, side="right") - 1
    safe = np.clip(slot, 0, None)
    hit = (slot >= 0) & (sorted_area[safe] == query_area) & (query_ord >= 0)
    return np.where(hit, order[safe], -1)


# --------------------------------------------------------------------------
# original93
# --------------------------------------------------------------------------


def assemble_original93(
    labels: pd.DataFrame, panel: pd.DataFrame, horizon: int, grid: PanelGrid | None = None
) -> tuple[np.ndarray, dict]:
    """93 columns for every valid target row at origin O = T - horizon.

    ``labels`` is the complete valid ledger (admin_code, month_ord, crisis_truth),
    sorted by (admin_code, month_ord). Rows come out in that order.
    """
    grid = grid or build_panel_grid(panel)
    raw_values = panel[list(RAW_FEATURE_COLUMNS)].to_numpy(dtype=np.float64, copy=True)
    infinite_cells = np.isinf(raw_values)
    input_infinities = {
        name: int(n) for name, n in zip(RAW_FEATURE_COLUMNS, infinite_cells.sum(axis=0)) if n
    }
    raw_values[infinite_cells] = np.nan
    column_index = {name: i for i, name in enumerate(RAW_FEATURE_COLUMNS)}

    admin = labels["admin_code"].to_numpy(dtype=np.int64)
    target_ord = labels["month_ord"].to_numpy(dtype=np.int64)
    label_y = labels["crisis_truth"].to_numpy(dtype=np.int64)
    if not np.isin(label_y, (0, 1)).all():
        raise ContractError("crisis truth must be binary")
    origin_ord = target_ord - int(horizon)
    area_idx = grid.area_index(admin)
    out: dict[str, np.ndarray] = {}

    origin_positions = grid.positions_at(area_idx, origin_ord)
    raw_at_origin = _gather(raw_values, origin_positions)
    for i, name in enumerate(RAW_FEATURE_COLUMNS):
        out[name] = raw_at_origin[:, i]

    summers: dict[str, WindowSummer] = {}
    for name, source, length in SUM_DERIVATIVES:
        if source not in summers:
            summers[source] = WindowSummer(grid, raw_values[:, column_index[source]])
        out[name] = summers[source].sum_ending_at(area_idx, origin_ord, length)
    lag_source = raw_values[:, column_index[LAG_DERIVATIVE_SOURCE]]
    for k in range(1, LAG_DERIVATIVE_MAX + 1):
        out[f"{LAG_DERIVATIVE_SOURCE}_lag{k}_asof"] = _gather(
            lag_source, grid.positions_at(area_idx, origin_ord - k)
        )

    angle = 2.0 * np.pi * (target_ord % 12) / 12.0  # target month (pre-known calendar)
    out["target_month_sin"] = np.sin(angle)
    out["target_month_cos"] = np.cos(angle)

    slot = as_of_index(admin, target_ord, admin, origin_ord)
    has = slot >= 0
    safe = np.clip(slot, 0, None)
    last_ord = np.where(has, target_ord[safe], -1)
    out["last_observed_label"] = np.where(has, label_y[safe].astype(np.float64), np.nan)
    out["last_observed_label_age_months"] = np.where(has, (origin_ord - last_ord).astype(np.float64), np.nan)
    out["no_observed_label_history"] = (~has).astype(np.float64)

    positive = label_y == 1
    crisis_ord = target_ord[positive]
    crisis_slot = as_of_index(admin[positive], crisis_ord, admin, origin_ord)
    has_crisis = crisis_slot >= 0
    found = crisis_ord[np.clip(crisis_slot, 0, None)] if len(crisis_ord) else np.zeros(len(admin), np.int64)
    out["months_since_last_observed_crisis"] = np.where(
        has_crisis, (origin_ord - found).astype(np.float64), np.nan
    )
    out["no_prior_observed_crisis"] = (~has_crisis).astype(np.float64)
    out["horizon_months"] = np.full(len(admin), float(horizon))

    matrix = np.column_stack([out[name] for name in ORIGINAL_FEATURES]).astype(np.float64)
    assembled_inf = np.isinf(matrix)
    assembled_infinities = {
        name: int(n) for name, n in zip(ORIGINAL_FEATURES, assembled_inf.sum(axis=0)) if n
    }
    matrix[assembled_inf] = np.nan
    audit = {
        "rows": int(len(admin)),
        "rows_with_missing_origin_panel_row": int((origin_positions < 0).sum()),
        "rows_without_label_history": int((~has).sum()),
        "rows_without_prior_crisis": int((~has_crisis).sum()),
        "panel_infinities_converted": input_infinities,
        "assembled_infinities_converted": assembled_infinities,
        "last_observed_month": month_label(last_ord),
    }
    return matrix, audit


# --------------------------------------------------------------------------
# history468
# --------------------------------------------------------------------------

SERIES_ORDER = ("q2", "q3", "q4", "q5", "severity_index", "entropy", "concentration", "severe_fraction")
COMPLETE_SERIES = SERIES_ORDER[:7]
RATIO_SERIES = "severe_fraction"
WINDOWS: tuple[tuple[str, int | None], ...] = (("m06", 6), ("m12", 12), ("m24", 24), ("m36", 36), ("all", None))
STATISTICS = ("mean", "std", "min", "max", "latest_minus_mean", "slope")
N_SLOTS = 6
CRISIS_THRESHOLD = 0.20
BLOCK_ORDER = (
    "observation_levels",
    "observation_timing",
    "changes",
    "observation_trends",
    "window_statistics",
    "window_support",
    "prior_binary_states",
    "threshold_distance",
    "window_crisis",
    "event_and_run",
)


def compute_series(valid: pd.DataFrame) -> pd.DataFrame:
    """Eight history series per valid observation (technical-contract.md 1).

    q2..q5 are the ledger's exact-derived normalized cumulative shares (the
    same values used as regression targets); the other four use p1..p5.
    """
    p = valid[["p1", "p2", "p3", "p4", "p5"]].to_numpy(dtype=np.float64)
    q = valid[["q2", "q3", "q4", "q5"]].to_numpy(dtype=np.float64)
    if not (np.isfinite(p).all() and np.isfinite(q).all()):
        raise ContractError("non-finite normalized share on a valid row")
    severity = p @ np.arange(1, 6, dtype=np.float64)
    concentration = (p * p).sum(axis=1)
    safe = np.where(p > 0.0, p, 1.0)
    entropy = -(np.where(p > 0.0, p * np.log(safe), 0.0)).sum(axis=1) / np.log(5.0)
    q3, q4 = q[:, 1], q[:, 2]
    severe_fraction = np.where(q3 > 0.0, q4 / np.where(q3 > 0.0, q3, 1.0), np.nan)
    out = pd.DataFrame(
        {
            "q2": q[:, 0],
            "q3": q3,
            "q4": q4,
            "q5": q[:, 3],
            "severity_index": severity,
            "entropy": entropy,
            "concentration": concentration,
            "severe_fraction": severe_fraction,
        },
        index=valid.index,
    )
    for name in COMPLETE_SERIES:
        if not np.isfinite(out[name].to_numpy()).all():
            raise ContractError(f"series {name} is not finite on every valid row")
    return out


@dataclass
class HistoryIndex:
    """The full valid ledger, indexed for own-origin history queries."""

    keys: np.ndarray
    admin: np.ndarray
    months: np.ndarray
    months_f: np.ndarray
    states: np.ndarray
    series: dict
    last_crisis: np.ndarray
    last_noncrisis: np.ndarray
    last_entry_newer: np.ndarray
    last_exit_newer: np.ndarray
    run_start: np.ndarray
    max_area_span: int
    n: int


def build_history_index(valid: pd.DataFrame) -> HistoryIndex:
    """Index the complete valid ledger (never a fitting or evaluation subset)."""
    frame = valid.sort_values(["admin_code", "month_ord"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["admin_code", "month_ord"]).any():
        raise ContractError("valid ledger has duplicate (admin_code, month)")
    series = compute_series(frame)
    admin = frame["admin_code"].to_numpy(dtype=np.int64)
    months = frame["month_ord"].to_numpy(dtype=np.int64)
    if len(months) and (months.min() < 0 or months.max() >= _KEY_SCALE):
        raise ContractError("month ordinal outside the packing range")
    keys = admin * _KEY_SCALE + months
    if not np.all(np.diff(keys) > 0):
        raise ContractError("packed observation keys are not strictly increasing")
    states = frame["crisis_truth"].to_numpy(dtype=np.int64)
    if not np.isin(states, (0, 1)).all():
        raise ContractError("valid ledger carries a non-binary crisis state")

    n = len(frame)
    last_crisis = np.full(n, -1, dtype=np.int64)
    last_noncrisis = np.full(n, -1, dtype=np.int64)
    last_entry_newer = np.full(n, -1, dtype=np.int64)
    last_exit_newer = np.full(n, -1, dtype=np.int64)
    run_start = np.zeros(n, dtype=np.int64)
    current_crisis = current_noncrisis = current_entry = current_exit = -1
    for i in range(n):
        if i == 0 or admin[i] != admin[i - 1]:
            current_crisis = current_noncrisis = current_entry = current_exit = -1
            run_start[i] = i
        else:
            if states[i] == 1 and states[i - 1] == 0:
                current_entry = i
            elif states[i] == 0 and states[i - 1] == 1:
                current_exit = i
            run_start[i] = run_start[i - 1] if states[i] == states[i - 1] else i
        if states[i] == 0:
            current_noncrisis = i
        else:
            current_crisis = i
        last_crisis[i] = current_crisis
        last_noncrisis[i] = current_noncrisis
        last_entry_newer[i] = current_entry
        last_exit_newer[i] = current_exit
    _, area_sizes = np.unique(admin, return_counts=True)
    return HistoryIndex(
        keys=keys,
        admin=admin,
        months=months,
        months_f=months.astype(np.float64),
        states=states,
        series={name: series[name].to_numpy(dtype=np.float64) for name in SERIES_ORDER},
        last_crisis=last_crisis,
        last_noncrisis=last_noncrisis,
        last_entry_newer=last_entry_newer,
        last_exit_newer=last_exit_newer,
        run_start=run_start,
        max_area_span=int(area_sizes.max()) if n else 0,
        n=n,
    )


def _gather_window(lo: np.ndarray, hi: np.ndarray, span: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(span, dtype=np.int64)
    index = lo[:, None] + offsets[None, :]
    mask = index < hi[:, None]
    return np.where(mask, index, 0), mask


def _window_statistics(values: np.ndarray, months_f: np.ndarray, index: np.ndarray, mask: np.ndarray) -> dict:
    """mean/min/max/latest need 1 finite value, std 2, slope 3; else NaN."""
    x = np.where(mask, values[index], np.nan)
    finite = np.isfinite(x)
    n = finite.sum(axis=1)
    rows = x.shape[0]
    has_one = n >= 1
    total = np.where(finite, x, 0.0).sum(axis=1)
    mean = np.where(has_one, total / np.where(has_one, n, 1), np.nan)
    deviation = np.where(finite, x - mean[:, None], 0.0)
    sq = (deviation * deviation).sum(axis=1)
    std = np.where(n >= 2, np.sqrt(np.maximum(sq / np.where(n >= 2, n, 1), 0.0)), np.nan)
    minimum = np.where(has_one, np.where(finite, x, np.inf).min(axis=1), np.nan)
    maximum = np.where(has_one, np.where(finite, x, -np.inf).max(axis=1), np.nan)
    positions = np.where(finite, np.arange(x.shape[1])[None, :], -1)
    latest = np.where(has_one, x[np.arange(rows), np.maximum(positions.max(axis=1), 0)], np.nan)
    t = np.where(mask, months_f[index], np.nan)
    t_mean = np.where(has_one, np.where(finite, t, 0.0).sum(axis=1) / np.where(has_one, n, 1), np.nan)
    t_dev = np.where(finite, t - t_mean[:, None], 0.0)
    stt = (t_dev * t_dev).sum(axis=1)
    stx = (t_dev * deviation).sum(axis=1)
    enough = (n >= 3) & (stt > 0.0)
    return {
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "latest_minus_mean": np.where(has_one, latest - mean, np.nan),
        "slope": np.where(enough, stx / np.where(enough, stt, 1.0), np.nan),
    }


def _pick(values: np.ndarray, index: np.ndarray, ok: np.ndarray) -> np.ndarray:
    out = np.full(index.shape, np.nan, dtype=np.float64)
    if ok.any():
        out[ok] = values[index[ok]]
    return out


def build_history_block(
    index: HistoryIndex, admin_code: np.ndarray, origin_ord: np.ndarray, feature_names: Sequence[str]
) -> np.ndarray:
    """Appended history columns per (area, own origin); months > origin invisible."""
    admin_code = np.asarray(admin_code, dtype=np.int64)
    origin_ord = np.asarray(origin_ord, dtype=np.int64)
    rows = admin_code.size
    area_start = np.searchsorted(index.keys, admin_code * _KEY_SCALE, side="left")
    hi = np.searchsorted(index.keys, admin_code * _KEY_SCALE + origin_ord, side="right")
    if np.any(hi < area_start):
        raise ContractError("history range end precedes its area block")
    span = max(index.max_area_span, 1)
    if np.any(hi - area_start > span):
        raise ContractError("an area block is wider than the recorded maximum")

    columns: dict[str, np.ndarray] = {}
    origin_f = origin_ord.astype(np.float64)
    months_f = index.months_f
    states_f = index.states.astype(np.float64)

    slot_index = np.empty((N_SLOTS, rows), dtype=np.int64)
    slot_ok = np.empty((N_SLOTS, rows), dtype=bool)
    slot_month = np.empty((N_SLOTS, rows), dtype=np.float64)
    for j in range(N_SLOTS):
        idx = hi - 1 - j
        ok = idx >= area_start
        slot_index[j] = np.where(ok, idx, 0)
        slot_ok[j] = ok
        slot_month[j] = _pick(months_f, slot_index[j], ok)

    slot_value: dict[str, np.ndarray] = {}
    for name in SERIES_ORDER:
        stacked = np.empty((N_SLOTS, rows), dtype=np.float64)
        for j in range(N_SLOTS):
            stacked[j] = _pick(index.series[name], slot_index[j], slot_ok[j])
            columns[f"hist_{name}_obs{j + 1}"] = stacked[j]
        slot_value[name] = stacked

    for j in range(1, N_SLOTS):
        columns[f"hist_age_obs{j + 1}"] = origin_f - slot_month[j]
    for j in range(N_SLOTS - 1):
        columns[f"hist_gap_obs{j + 1}_obs{j + 2}"] = slot_month[j] - slot_month[j + 1]

    for name in SERIES_ORDER:
        stacked = slot_value[name]
        for j in range(N_SLOTS - 1):
            gap = slot_month[j] - slot_month[j + 1]
            difference = stacked[j] - stacked[j + 1]
            usable = np.isfinite(gap) & (gap > 0)
            columns[f"hist_{name}_change_obs{j + 1}_obs{j + 2}"] = difference
            columns[f"hist_{name}_rate_obs{j + 1}_obs{j + 2}"] = np.where(
                usable, difference / np.where(usable, gap, 1.0), np.nan
            )

    for label, depth in (("last3", 3), ("last6", 6)):
        lo = np.maximum(area_start, hi - depth)
        gathered, mask = _gather_window(lo, hi, depth)
        for name in SERIES_ORDER:
            columns[f"hist_{name}_slope_{label}"] = _window_statistics(
                index.series[name], months_f, gathered, mask
            )["slope"]

    window_ranges = {}
    for label, width in WINDOWS:
        if width is None:
            lo = area_start
        else:
            lo = np.maximum(
                np.searchsorted(index.keys, admin_code * _KEY_SCALE + (origin_ord - width + 1), side="left"),
                area_start,
            )
        gathered, mask = _gather_window(lo, hi, span)
        window_ranges[label] = (lo, gathered, mask)

    for name in SERIES_ORDER:
        for label, _width in WINDOWS:
            _lo, gathered, mask = window_ranges[label]
            stats = _window_statistics(index.series[name], months_f, gathered, mask)
            for statistic in STATISTICS:
                columns[f"hist_{name}_{label}_{statistic}"] = stats[statistic]

    for label, _width in WINDOWS:
        lo, gathered, mask = window_ranges[label]
        count = (hi - lo).astype(np.float64)
        has = count >= 1
        newest_month = _pick(months_f, np.where(has, hi - 1, 0), has)
        oldest_month = _pick(months_f, np.where(has, lo, 0), has)
        columns[f"hist_support_common_{label}_count"] = count
        columns[f"hist_support_common_{label}_span"] = newest_month - oldest_month
        if label != "all":
            columns[f"hist_support_common_{label}_age"] = origin_f - newest_month
        ratio_finite = np.isfinite(np.where(mask, index.series[RATIO_SERIES][gathered], np.nan))
        ratio_count = ratio_finite.sum(axis=1)
        ratio_has = ratio_count >= 1
        positions = np.where(ratio_finite, np.arange(mask.shape[1])[None, :], -1)
        newest_column = positions.max(axis=1)
        oldest_column = np.where(ratio_finite, np.arange(mask.shape[1])[None, :], mask.shape[1]).min(axis=1)
        row_index = np.arange(rows)
        ratio_newest = np.where(
            ratio_has, months_f[gathered[row_index, np.maximum(newest_column, 0)]], np.nan
        )
        ratio_oldest = np.where(
            ratio_has, months_f[gathered[row_index, np.minimum(oldest_column, mask.shape[1] - 1)]], np.nan
        )
        columns[f"hist_support_{RATIO_SERIES}_{label}_count"] = ratio_count.astype(np.float64)
        columns[f"hist_support_{RATIO_SERIES}_{label}_span"] = ratio_newest - ratio_oldest
        columns[f"hist_support_{RATIO_SERIES}_{label}_age"] = origin_f - ratio_newest

    for j in range(1, N_SLOTS):
        columns[f"hist_crisis_obs{j + 1}"] = _pick(states_f, slot_index[j], slot_ok[j])

    q3_obs1 = slot_value["q3"][0]
    columns["hist_q3_margin_obs1"] = q3_obs1 - CRISIS_THRESHOLD
    columns["hist_q3_abs_margin_obs1"] = np.abs(q3_obs1 - CRISIS_THRESHOLD)
    abs_margin = np.abs(index.series["q3"] - CRISIS_THRESHOLD)
    for label, _width in WINDOWS:
        _lo, gathered, mask = window_ranges[label]
        stats = _window_statistics(abs_margin, months_f, gathered, mask)
        columns[f"hist_q3_{label}_abs_margin_mean"] = stats["mean"]
        columns[f"hist_q3_{label}_abs_margin_min"] = stats["min"]

    for label, _width in WINDOWS:
        lo, gathered, mask = window_ranges[label]
        count = (hi - lo).astype(np.float64)
        gathered_states = np.where(mask, states_f[gathered], np.nan)
        live = count >= 1
        fraction = np.where(
            live, np.where(mask, gathered_states, 0.0).sum(axis=1) / np.where(live, count, 1.0), np.nan
        )
        pair = mask[:, :-1] & mask[:, 1:]
        older = np.where(pair, np.where(mask[:, :-1], gathered_states[:, :-1], 0.0), np.nan)
        newer = np.where(pair, np.where(mask[:, 1:], gathered_states[:, 1:], 0.0), np.nan)
        entries = ((older == 0.0) & (newer == 1.0) & pair).sum(axis=1).astype(np.float64)
        exits = ((older == 1.0) & (newer == 0.0) & pair).sum(axis=1).astype(np.float64)
        enough = count >= 2
        columns[f"hist_crisis_{label}_fraction"] = fraction
        columns[f"hist_crisis_{label}_entries"] = np.where(enough, entries, np.nan)
        columns[f"hist_crisis_{label}_exits"] = np.where(enough, exits, np.nan)
        columns[f"hist_crisis_{label}_pairs"] = np.maximum(count - 1.0, 0.0)

    newest_index = np.maximum(hi - 1, 0)
    any_history = hi > area_start
    noncrisis_idx = np.where(any_history, index.last_noncrisis[newest_index], -1)
    noncrisis_ok = any_history & (noncrisis_idx >= area_start)
    columns["hist_noncrisis_age"] = origin_f - _pick(months_f, np.where(noncrisis_ok, noncrisis_idx, 0), noncrisis_ok)
    columns["hist_no_noncrisis"] = (~noncrisis_ok).astype(np.float64)
    for kind, table in (("entry", index.last_entry_newer), ("exit", index.last_exit_newer)):
        idx = np.where(any_history, table[newest_index], -1)
        ok = any_history & (idx >= area_start) & (idx <= hi - 1)
        columns[f"hist_{kind}_age"] = origin_f - _pick(months_f, np.where(ok, idx, 0), ok)
        columns[f"hist_no_{kind}"] = (~ok).astype(np.float64)
    run_begin = np.where(any_history, index.run_start[newest_index], 0)
    columns["hist_current_run_count"] = np.where(any_history, (hi - run_begin).astype(np.float64), np.nan)
    columns["hist_current_run_span"] = np.where(
        any_history,
        _pick(months_f, newest_index, any_history) - _pick(months_f, run_begin, any_history),
        np.nan,
    )

    missing = [name for name in feature_names if name not in columns]
    if missing:
        raise ContractError(f"history block did not produce: {missing[:10]}")
    extra = sorted(set(columns) - set(feature_names))
    if extra:
        raise ContractError(f"history block produced unlisted columns: {extra[:10]}")
    block = np.empty((rows, len(feature_names)), dtype=np.float64)
    for position, name in enumerate(feature_names):
        values = columns[name]
        if np.isinf(values).any():
            raise ContractError(f"engineered history column {name} produced an infinity")
        block[:, position] = values
    return block


def check_aliases(original_X, original_names, aliases, index, admin_code, origin_ord) -> dict:
    """Each aliased history name must equal the original93 column it reuses."""
    position = {name: i for i, name in enumerate(original_names)}
    area_start = np.searchsorted(index.keys, admin_code * _KEY_SCALE, side="left")
    hi = np.searchsorted(index.keys, admin_code * _KEY_SCALE + origin_ord, side="right")
    newest = np.maximum(hi - 1, 0)
    has = hi > area_start
    origin_f = origin_ord.astype(np.float64)
    expected = {
        "hist_age_obs1": origin_f - _pick(index.months_f, newest, has),
        "hist_crisis_obs1": _pick(index.states.astype(np.float64), newest, has),
    }
    expected["hist_support_common_all_age"] = expected["hist_age_obs1"]
    crisis_index = np.where(has, index.last_crisis[newest], -1)
    crisis_ok = has & (crisis_index >= area_start)
    expected["hist_crisis_age"] = origin_f - _pick(index.months_f, np.where(crisis_ok, crisis_index, 0), crisis_ok)
    expected["hist_no_crisis"] = (~crisis_ok).astype(np.float64)
    checked = {}
    for alias, target in aliases.items():
        if alias not in expected:
            raise ContractError(f"alias {alias}: no recomputation implemented")
        actual = original_X[:, position[target]]
        want = expected[alias]
        same_nan = np.array_equal(np.isnan(actual), np.isnan(want))
        same_val = np.array_equal(actual[~np.isnan(actual)], want[~np.isnan(want)]) if same_nan else False
        if not (same_nan and same_val):
            raise ContractError(f"alias {alias} disagrees with original93 column {target}")
        checked[alias] = target
    return checked
