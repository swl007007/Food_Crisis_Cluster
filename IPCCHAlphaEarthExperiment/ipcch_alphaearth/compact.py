"""compact296 non-oracle feature recipe rebuilt at each row origin O = T - H for H in {1, 3, 6, 12}.

Vendored (bounded copy, no dynamic import of the sibling repository) from ../IPCCH at HEAD 9b41d7a; the bytes of
the sources equal the code_sha256 recorded in the compact_climate_weather_oracle_v1 manifest:

- src/ipcch/compact_features.py      923cae26335f9994043c178a0920a7736e7e80dc9c51a5fffa8d406f07fdf324
  (names, trailing, same_month_z, ordinary/z/stress arrays, monthly_block, season_major_flags, completed_seasons,
  interim_grid/composite_frame)
- src/ipcch/climate2015_features.py  e8da9de9224b1f5c42706d50fccd12f2173886079c4c7a711f152f900e9c22d5
  (Grid, shift, safe_divide, months_since, longest_run, stress rules, gather)
- src/ipcch/retained_feature_recipes.py cf1557e405d0aef3a62ee9af4286680465e8fae3f7661c959d321870423465ed
  (commodity_members, stress_signal, composites, static-name pattern)
- src/ipcch/origin_safe.py           629936a8de4973399a3dfa798b4a212bb8fefe9838d16a032c072c446101ea03
  (safe reported-phase history, national IDP, independent reference replays)
- src/ipcch/forecasting_weight_decay.py 1a9f20e2... (add_identifier_features: lat/lon at the target key + target
  month/year dummies)

Adaptations (design section 4, D15/D29):

1. Horizons are {1, 3, 6, 12}. H1 is an explicit row-origin rebuild at O = T - 1; H0 values are never used.
   ``label_cutoff`` keeps the sibling ``T - max(H, 1)`` rule, which equals O for every H >= 1.
2. Rows are this task's population-derived canonical keys, not the sibling reported-phase cohort; no row is
   dropped because a feature is missing.
3. Static29 is read per admin from the four pinned compact baseline tables (H0/H3/H6/H12) after a within- and
   across-table invariance check; largely missing popdensity is retained, never refilled from another snapshot.
4. Identifier year dummies are the fixed contract columns year_2014..year_2026 (the sibling built them with
   pd.get_dummies over its cohort, which yields the same 13 columns; an explicit list cannot drift with the cohort).
5. The weather-oracle arm and its helpers are not copied.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ipcch_alphaearth.months import month_ord

HORIZONS = (1, 3, 6, 12)
KEYS = ("area_id", "year", "month")
VALID_PHASES = (1, 2, 3, 4, 5)

# --------------------------------------------------------------------------- climate2015_features constants
CLIMATE_BASES_ALL = (
    "prcp_anom", "prcp_z", "rainy_days", "cdd", "tmean_anom", "tmax_anom", "hot_days_p95",
    "gdd", "edd", "sm_z", "spi03", "spei03", "ndvi_anom", "evi_anom",
)
STRESS_MIN_PERIODS = 6
DEFICIT = "spi03_month_ensmean__deficit_stress"
HOT = "tmean_anom_month_ensmean__hot_stress"
VEGETATION = "evi_anom_month_ensmean__vegetation_stress"
STRESS_RULES: Tuple[Tuple[str, str, Callable[[np.ndarray], np.ndarray]], ...] = (
    (DEFICIT, "spi03_month_ensmean", lambda x: x <= -1.0),
    (HOT, "tmean_anom_month_ensmean", lambda x: x >= 1.0),
    (VEGETATION, "evi_anom_month_ensmean", lambda x: x <= -0.015),
)

# --------------------------------------------------------------------------- retained_feature_recipes constants
ENSO = "nino34_anom"
CONFLICT = ("event_count_violence", "sum_fatalities_violence")
COMPOSITE_GROUPS = ("bbg_staple_food", "bbg_oilgas", "bbg_soybean_oil", "bbg_global_food")
STATIC_COLUMNS_PATTERN = r"^(AEZ_\d+|crop|elevation|market_access|popdensity|range|ruggedness|slope|coastline_dist)$"

# --------------------------------------------------------------------------- compact_features constants
COMPOSITES = tuple(f"{g}__composite_mean" for g in COMPOSITE_GROUPS)
INTERIM_SOURCES = ("GPP_mean", "nightlight_mean", "event_count_violence", "sum_fatalities_violence", "WFP_Price",
                   "food_price_index_WB", ENSO, *COMPOSITES)
CLIMATE_BASES = tuple(b for b in CLIMATE_BASES_ALL if b != "ndvi_anom")
CLIMATE_SOURCES = tuple(f"{b}_month_ensmean" for b in CLIMATE_BASES)
ORDINARY_SOURCES = INTERIM_SOURCES + CLIMATE_SOURCES
Z_SOURCES = ("GPP_mean", "nightlight_mean", "WFP_Price", "food_price_index_WB", ENSO,
             *(f"{b}_month_ensmean" for b in ("prcp_anom", "rainy_days", "cdd", "tmean_anom", "tmax_anom", "hot_days_p95",
                                              "gdd", "edd", "evi_anom")))
INTERIM_STRESS_SOURCES = ("GPP_mean", "event_count_violence", "WFP_Price", ENSO)
STRESS_STEMS = ("GPP_mean__vegetation_stress", "event_count_violence__nonzero_stress", "WFP_Price__price_shock_stress",
                "nino34_anom__enso_stress", DEFICIT, HOT, VEGETATION)
SEASON_METRICS = tuple(f"{b}_gs_ensmean" for b in CLIMATE_BASES)
MA_WINDOWS = (3, 6, 12)
MIN_PERIODS = {3: 2, 6: 3, 12: 6}
SD_WINDOW, SD_MIN_PERIODS = 12, 6
N_SEASONS = 2
MAJOR = "gs_last1__is_major_at_origin"

# --------------------------------------------------------------------------- origin_safe constants
HISTORY_VALUES = tuple(f"overall_phase_history_{k}" for k in (1, 2, 3))
HISTORY_CHANGES = ("overall_phase_history_change_1_2", "overall_phase_history_change_1_3")
HISTORY_FEATURES = HISTORY_VALUES + HISTORY_CHANGES
IDP_FEATURES = ("idp_admin0_latest_stock", "idp_admin0_latest_age_months")

IDENTIFIER_FEATURES = ("lat", "lon", *(f"month_{m}" for m in range(1, 13)), *(f"year_{y}" for y in range(2014, 2027)))
EXPECTED_COUNTS = {"static": 29, "ordinary": 120, "same_month_z": 56, "stress": 28, "growing_season": 29,
                   "identifier": 27, "history": 5, "idp": 2}


def validate_horizon(horizon: int) -> int:
    if int(horizon) not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}; received {horizon}")
    return int(horizon)


def label_cutoff(target_ord, horizon: int) -> np.ndarray:
    """Latest reported-phase month usable as a history input: ``min(T - H, T - 1)`` (= O for H >= 1)."""
    horizon = validate_horizon(horizon)
    return np.asarray(target_ord, dtype=np.int64) - max(horizon, 1)


def is_static_name(column: str) -> bool:
    return re.match(STATIC_COLUMNS_PATTERN, column) is not None


# --------------------------------------------------------------------------- names and schema


def ordinary_names(source: str) -> List[str]:
    return [f"{source}__value_at_origin", *(f"{source}__ma{w}_at_origin" for w in MA_WINDOWS), f"{source}__sd12_at_origin"]


def z_names(source: str) -> List[str]:
    return [f"{source}__hist_same_month_z_at_origin", *(f"{source}__hist_same_month_z_ma{w}_at_origin" for w in MA_WINDOWS)]


def stress_names(stem: str) -> List[str]:
    return [f"{stem}__{s}_at_origin" for s in ("share12", "months_since", "longest_run12", "any12")]


def season_names() -> List[str]:
    out = []
    for k in range(1, N_SEASONS + 1):
        out += [f"gs_last{k}__{m}__at_origin" for m in SEASON_METRICS] + [f"gs_last{k}__months_since_end_at_origin"]
    return out + [MAJOR]


def dynamic_features() -> List[str]:
    """Ordinary (source-major) -> same-month z -> stress (signal-major) -> seasons + major dummy (233 columns)."""
    return ([n for s in ORDINARY_SOURCES for n in ordinary_names(s)] + [n for s in Z_SOURCES for n in z_names(s)]
            + [n for s in STRESS_STEMS for n in stress_names(s)] + season_names())


def feature_groups(frozen_order: Sequence[str]) -> Dict[str, List[str]]:
    """Split the frozen 296 order into its eight groups and check it equals this module's recipe order."""
    frozen = list(frozen_order)
    static = frozen[:29]
    dynamic = frozen[29:262]
    rest = frozen[262:]
    groups = {
        "static": static,
        "ordinary": dynamic[:120],
        "same_month_z": dynamic[120:176],
        "stress": dynamic[176:204],
        "growing_season": dynamic[204:233],
        "identifier": rest[:27],
        "history": rest[27:32],
        "idp": rest[32:34],
    }
    for name, cols in groups.items():
        if len(cols) != EXPECTED_COUNTS[name]:
            raise ValueError(f"{name} block has {len(cols)} columns, expected {EXPECTED_COUNTS[name]}")
    if not all(is_static_name(c) for c in static):
        raise ValueError("frozen static block contains a non-static name")
    if dynamic != dynamic_features():
        raise ValueError("frozen dynamic block differs from the vendored recipe order")
    if groups["identifier"] != list(IDENTIFIER_FEATURES):
        raise ValueError("frozen identifier block differs from lat/lon + month_1..12 + year_2014..2026")
    if groups["history"] != list(HISTORY_FEATURES) or groups["idp"] != list(IDP_FEATURES):
        raise ValueError("frozen history/IDP literals differ from the origin-safe definitions")
    return groups


# --------------------------------------------------------------------------- grid primitives (climate2015_features)


@dataclass
class Grid:
    """Complete area x month panel; ``values[v]`` is an (n_areas, n_months) float array."""

    area_ids: np.ndarray
    first_ord: int
    n_months: int
    values: Dict[str, np.ndarray]

    @classmethod
    def from_long(cls, df: pd.DataFrame, variables: Sequence[str], area_col: str = "area_id", require_complete: bool = True) -> "Grid":
        ords = month_ord(df["year"], df["month"])
        area_ids = np.sort(df[area_col].unique())
        first, last = int(ords.min()), int(ords.max())
        n_months = last - first + 1
        if df.duplicated([area_col, "year", "month"]).any():
            raise ValueError("panel has duplicate area-month keys")
        if require_complete and len(df) != len(area_ids) * n_months:
            raise ValueError("climate panel is not a complete area x month grid")
        if not require_complete:
            span = pd.Series(ords).groupby(df[area_col].to_numpy()).agg(["min", "max", "count"])
            if ((span["max"] - span["min"] + 1) != span["count"]).any():
                raise ValueError("panel has internal month gaps; row shifts would not equal month shifts")
        ai = np.searchsorted(area_ids, df[area_col].to_numpy())
        mi = ords - first
        values = {}
        for v in variables:
            arr = np.full((len(area_ids), n_months), np.nan)
            arr[ai, mi] = pd.to_numeric(df[v], errors="coerce").to_numpy(dtype=float)
            values[v] = arr
        return cls(area_ids, first, n_months, values)

    def index_of(self, area_ids, ords) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        area_ids = np.asarray(area_ids)
        ai = np.searchsorted(self.area_ids, area_ids)
        ai_clip = np.clip(ai, 0, len(self.area_ids) - 1)
        mi = np.asarray(ords, dtype=np.int64) - self.first_ord
        ok = (self.area_ids[ai_clip] == area_ids) & (mi >= 0) & (mi < self.n_months)
        return ai_clip, np.clip(mi, 0, self.n_months - 1), ok


def shift(x: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return x.copy()
    out = np.full_like(x, np.nan)
    out[:, k:] = x[:, :-k]
    return out


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    den = np.where(den == 0, np.nan, den)
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def months_since(signal_at_anchor: np.ndarray) -> np.ndarray:
    out = np.full_like(signal_at_anchor, np.nan)
    last = np.full(signal_at_anchor.shape[0], np.nan)
    for j in range(signal_at_anchor.shape[1]):
        value = signal_at_anchor[:, j]
        hit = value == 1
        last = np.where(hit, 0.0, last + 1.0)  # NaN + 1 stays NaN until the first stress month
        out[:, j] = last
    return out


def longest_run(signal_at_anchor: np.ndarray, window: int = 12, min_periods: int = STRESS_MIN_PERIODS) -> np.ndarray:
    n_areas, n_months = signal_at_anchor.shape
    out = np.full_like(signal_at_anchor, np.nan)
    for j in range(n_months):
        lo = max(0, j - window + 1)
        win = signal_at_anchor[:, lo : j + 1]
        enough = np.sum(~np.isnan(win), axis=1) >= min_periods
        best = np.zeros(n_areas)
        current = np.zeros(n_areas)
        for k in range(win.shape[1]):
            current = np.where(win[:, k] == 1, current + 1, 0)
            best = np.maximum(best, current)
        out[:, j] = np.where(enough, best, np.nan)
    return out


def climate_stress_signals(grid: Grid) -> Dict[str, np.ndarray]:
    out = {}
    for signal, source, rule in STRESS_RULES:
        x = grid.values[source]
        with np.errstate(invalid="ignore"):
            out[signal] = np.where(np.isnan(x), np.nan, rule(x).astype(float))
    return out


def gather(values: np.ndarray, index: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    ai, mi, ok = index
    out = values[ai, mi]
    return np.where(ok, out, np.nan)


# --------------------------------------------------------------------------- retained_feature_recipes


def commodity_members(columns: Sequence[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {name: [] for name in COMPOSITE_GROUPS}
    for column in columns:
        lower = column.lower()
        if not lower.startswith("bbg_"):
            continue
        if "staple_food" in lower:
            groups["bbg_staple_food"].append(column)
        elif "oilgas" in lower or "ngushhub" in lower:
            groups["bbg_oilgas"].append(column)
        elif "soybean_oil" in lower:
            groups["bbg_soybean_oil"].append(column)
        else:
            groups["bbg_global_food"].append(column)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def stress_signal(source: str, x: np.ndarray) -> Tuple[str, np.ndarray]:
    prev = shift(x, 12)
    with np.errstate(invalid="ignore", divide="ignore"):
        if source in CONFLICT:
            name, hit = "nonzero_stress", np.nan_to_num(x, nan=0.0) > 0
        elif source == "GPP_mean":
            name, hit = "vegetation_stress", x < prev * 0.9
        elif source == ENSO:
            name, hit = "enso_stress", np.abs(x) > 0.5
        else:
            name, hit = "price_shock_stress", safe_divide(x - prev, prev) > 0.10
    return f"{source}__{name}", np.where(np.isnan(x), np.nan, hit.astype(float))


# --------------------------------------------------------------------------- source grids (compact_features)


def composite_frame(panel: pd.DataFrame, members: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    if tuple(members) != COMPOSITE_GROUPS:
        raise ValueError(f"composite groups {list(members)} differ from {COMPOSITE_GROUPS}")
    out = panel.loc[:, list(KEYS)].copy()
    for group, cols in members.items():
        out[f"{group}__composite_mean"] = panel[list(cols)].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
    return out


def interim_grid(panel: pd.DataFrame, members: Mapping[str, Sequence[str]]) -> Grid:
    work = pd.concat([panel.loc[:, list(KEYS) + list(INTERIM_SOURCES[:7])].reset_index(drop=True),
                      composite_frame(panel, members).drop(columns=list(KEYS)).reset_index(drop=True)], axis=1)
    return Grid.from_long(work, INTERIM_SOURCES, require_complete=False)


# --------------------------------------------------------------------------- monthly recipes (compact_features)


def trailing(x: np.ndarray, window: int, min_periods: int, reducer: str) -> np.ndarray:
    n_areas, n_months = x.shape
    padded = np.concatenate([np.full((n_areas, window - 1), np.nan), x], axis=1)
    win = np.lib.stride_tricks.sliding_window_view(padded, window, axis=1)
    count = np.sum(~np.isnan(win), axis=2)
    enough = count >= min_periods
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if reducer == "max":
            out = np.nanmax(win, axis=2)
        elif reducer in ("mean", "std"):
            ref = np.nanmin(win, axis=2)
            d = win - ref[..., None]
            mean_d = np.nansum(d, axis=2) / count
            if reducer == "mean":
                out = ref + mean_d
            else:
                out = np.sqrt(np.nansum((d - mean_d[..., None]) ** 2, axis=2) / (count - 1))
        else:
            raise ValueError(f"unknown reducer {reducer}")
    return np.where(enough, out, np.nan)


def same_month_z(x: np.ndarray, first_ord: int) -> np.ndarray:
    lead = first_ord % 12
    n_areas, n_months = x.shape
    n_years = -(-(lead + n_months) // 12)
    padded = np.full((n_areas, n_years * 12), np.nan)
    padded[:, lead : lead + n_months] = x
    cube = padded.reshape(n_areas, n_years, 12)
    has = ~np.isnan(cube)
    first = np.argmax(has, axis=1)
    ref = np.take_along_axis(cube, first[:, None, :], axis=1)[:, 0, :]
    d = cube - ref[:, None, :]
    z = np.full_like(cube, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for y in range(1, n_years):
            prior = d[:, :y, :]
            count = np.sum(~np.isnan(prior), axis=1)
            mean_d = np.nansum(prior, axis=1) / count
            sd = np.sqrt(np.nansum((prior - mean_d[:, None, :]) ** 2, axis=1) / (count - 1))
            constant = np.nanmax(prior, axis=1) == np.nanmin(prior, axis=1)
            ok = (count >= 2) & ~constant & (sd > 0) & ~np.isnan(d[:, y, :])
            z[:, y, :] = np.where(ok, (d[:, y, :] - mean_d) / sd, np.nan)
    return z.reshape(n_areas, n_years * 12)[:, lead : lead + n_months]


def ordinary_arrays(source: str, x: np.ndarray) -> Iterator[Tuple[str, np.ndarray]]:
    names = ordinary_names(source)
    yield names[0], x
    for name, w in zip(names[1:4], MA_WINDOWS):
        yield name, trailing(x, w, MIN_PERIODS[w], "mean")
    yield names[4], trailing(x, SD_WINDOW, SD_MIN_PERIODS, "std")


def z_arrays(source: str, x: np.ndarray, first_ord: int) -> Iterator[Tuple[str, np.ndarray]]:
    z = same_month_z(x, first_ord)
    names = z_names(source)
    yield names[0], z
    for name, w in zip(names[1:], MA_WINDOWS):
        yield name, trailing(z, w, MIN_PERIODS[w], "mean")


def stress_signal_arrays(interim: Grid, climate: Grid) -> Dict[str, Tuple[Grid, np.ndarray]]:
    out: Dict[str, Tuple[Grid, np.ndarray]] = {}
    for source in INTERIM_STRESS_SOURCES:
        name, s = stress_signal(source, interim.values[source])
        out[name] = (interim, s)
    for name, s in climate_stress_signals(climate).items():
        out[name] = (climate, s)
    if tuple(out) != STRESS_STEMS:
        raise AssertionError(f"stress signal order {list(out)} differs from {STRESS_STEMS}")
    return out


def stress_arrays(stem: str, s: np.ndarray) -> Iterator[Tuple[str, np.ndarray]]:
    names = stress_names(stem)
    yield names[0], trailing(s, 12, STRESS_MIN_PERIODS, "mean")
    yield names[1], months_since(s)
    yield names[2], longest_run(s)
    yield names[3], trailing(s, 12, STRESS_MIN_PERIODS, "max")


def monthly_feature_arrays(interim: Grid, climate: Grid) -> Iterator[Tuple[str, Grid, np.ndarray]]:
    """(name, grid, full-grid array) in contract order; the value at column ``u`` uses source months ``<= u`` only."""
    grids = {s: (interim if s in INTERIM_SOURCES else climate) for s in ORDINARY_SOURCES}
    for s in ORDINARY_SOURCES:
        for name, arr in ordinary_arrays(s, grids[s].values[s]):
            yield name, grids[s], arr
    for s in Z_SOURCES:
        for name, arr in z_arrays(s, grids[s].values[s], grids[s].first_ord):
            yield name, grids[s], arr
    for stem, (grid, s) in stress_signal_arrays(interim, climate).items():
        for name, arr in stress_arrays(stem, s):
            yield name, grid, arr


def monthly_block(interim: Grid, climate: Grid, keys: pd.DataFrame, horizons: Sequence[int]) -> Dict[int, pd.DataFrame]:
    """Ordinary, z and stress columns for each horizon's rows, gathered at the calendar month O = T - H."""
    target = month_ord(keys["year"], keys["month"])
    area = keys["area_id"].to_numpy()
    index = {(id(g), h): g.index_of(area, target - h) for g in (interim, climate) for h in horizons}
    cols: Dict[int, Dict[str, np.ndarray]] = {h: {} for h in horizons}
    for name, grid, arr in monthly_feature_arrays(interim, climate):
        for h in horizons:
            cols[h][name] = gather(arr, index[(id(grid), h)])
    return {h: pd.DataFrame(c) for h, c in cols.items()}


# --------------------------------------------------------------------------- growing seasons (compact_features)


def season_major_flags(seasons: pd.DataFrame) -> pd.Series:
    start = pd.to_datetime(seasons["gs_start_date"], errors="coerce")
    end = pd.to_datetime(seasons["gs_end_date_exclusive"], errors="coerce")
    days = (end - start).dt.days.astype(float)
    days = days.where(days > 0)
    frame = pd.DataFrame({"admin_code": seasons["admin_code"].to_numpy(), "season_year": seasons["season_year"].to_numpy(),
                          "season": seasons["season"].to_numpy(), "days": days.to_numpy()})
    grp = frame.groupby(["admin_code", "season_year"], sort=False)
    pair = (grp["season"].transform("size") == 2) & (grp["season"].transform("nunique") == 2) \
        & frame.assign(ok=frame["season"].isin(["s1", "s2"])).groupby(["admin_code", "season_year"], sort=False)["ok"].transform("all")
    other = grp["days"].transform("sum") - frame["days"]
    comparable = pair & (grp["days"].transform("count") == 2)
    flag = np.where(frame["days"] > other, 1.0, np.where(frame["days"] < other, 0.0, np.nan))
    return pd.Series(np.where(comparable, flag, np.nan), index=seasons.index, name="is_major")


def completed_seasons(seasons: pd.DataFrame, area_ids, origin_ords) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Latest two completed seasons (``end_exclusive <= first day of O+1``) per row: features and selection ledger."""
    s = seasons.copy()
    s["is_major"] = season_major_flags(s).to_numpy()
    s["end"] = pd.to_datetime(s["gs_end_date_exclusive"])
    s["start"] = pd.to_datetime(s["gs_start_date"])
    s["duration_days"] = (s["end"] - s["start"]).dt.days
    s = s.sort_values(["admin_code", "end", "start"], kind="mergesort").reset_index(drop=True)
    s["pos"] = s.groupby("admin_code").cumcount()
    end_month = s["end"] - pd.Timedelta(days=1)
    s["end_ord"] = month_ord(end_month.dt.year, end_month.dt.month)
    origin = np.asarray(origin_ords, dtype=np.int64)
    keys = pd.DataFrame({"row": np.arange(len(origin)), "admin_code": np.asarray(area_ids), "origin": origin})
    cut_ord = keys["origin"] + 1
    keys["cutoff"] = pd.to_datetime({"year": cut_ord // 12, "month": cut_ord % 12 + 1, "day": 1})
    keys = keys.sort_values("cutoff", kind="mergesort")
    right = s[["admin_code", "end", "pos"]].sort_values(["end", "pos"], kind="mergesort")
    hit = pd.merge_asof(keys, right, left_on="cutoff", right_on="end", by="admin_code", direction="backward", allow_exact_matches=True)
    hit = hit.sort_values("row").reset_index(drop=True)
    lookup = s.set_index(["admin_code", "pos"])
    feats: Dict[str, np.ndarray] = {}
    ledger = pd.DataFrame({"origin_ord": origin, "season_cutoff_date": hit["cutoff"].dt.strftime("%Y-%m-%d").to_numpy()})
    for k in range(1, N_SEASONS + 1):
        pos = hit["pos"] - (k - 1)
        ok = (pos.notna() & (pos >= 0)).to_numpy()
        idx = pd.MultiIndex.from_arrays([hit.loc[ok, "admin_code"], pos[ok].astype(int)])
        picked = lookup.reindex(idx)
        for m in SEASON_METRICS:
            col = np.full(len(origin), np.nan)
            col[ok] = picked[m].to_numpy(dtype=float)
            feats[f"gs_last{k}__{m}__at_origin"] = col
        since = np.full(len(origin), np.nan)
        since[ok] = origin[ok] - picked["end_ord"].to_numpy()
        feats[f"gs_last{k}__months_since_end_at_origin"] = since
        for field, src, kind in (("season_year", "season_year", float), ("season", "season", object), ("start", "gs_start_date", object),
                                 ("end_exclusive", "gs_end_date_exclusive", object), ("end_ord", "end_ord", float),
                                 ("duration_days", "duration_days", float), ("is_major", "is_major", float)):
            col = np.full(len(origin), np.nan if kind is float else None, dtype=kind)
            col[ok] = picked[src].to_numpy()
            ledger[f"gs_last{k}_{field}"] = col
    feats[MAJOR] = ledger["gs_last1_is_major"].to_numpy(dtype=float)
    out = pd.DataFrame(feats)
    if list(out.columns) != season_names():
        raise AssertionError("season block order differs from the contract")
    return out, ledger


def assert_season_ledger(frame: pd.DataFrame, ledger: pd.DataFrame, origin: np.ndarray) -> Dict[str, int]:
    """Season ages/major come from the ledger's selected seasons, all completed by the first day of O+1."""
    origin = np.asarray(origin, dtype=np.int64)
    if len(ledger) != len(frame) or not np.array_equal(ledger["origin_ord"].to_numpy(dtype=np.int64), origin):
        raise ValueError("season ledger rows do not align with the dataset rows or origins")
    cutoff = pd.to_datetime(ledger["season_cutoff_date"])
    expected_cut = pd.to_datetime({"year": (origin + 1) // 12, "month": (origin + 1) % 12 + 1, "day": 1})
    if not (cutoff.to_numpy() == expected_cut.to_numpy()).all():
        raise ValueError("season ledger cutoff is not the first day of O+1")
    for k in range(1, N_SEASONS + 1):
        has = ledger[f"gs_last{k}_end_ord"].notna().to_numpy()
        end = pd.to_datetime(ledger.loc[has, f"gs_last{k}_end_exclusive"])
        if (end.to_numpy() > cutoff[has].to_numpy()).any():
            raise ValueError(f"gs_last{k}: selected season ends after the first day of O+1")
        since = frame[f"gs_last{k}__months_since_end_at_origin"].to_numpy(dtype=float)
        if not np.array_equal(np.isfinite(since), has) or not np.array_equal(
                since[has], origin[has] - ledger.loc[has, f"gs_last{k}_end_ord"].to_numpy(dtype=float)):
            raise ValueError(f"gs_last{k} months_since_end differs from the season ledger")
        for m in SEASON_METRICS:
            if frame.loc[~has, f"gs_last{k}__{m}__at_origin"].notna().any():
                raise ValueError(f"gs_last{k}__{m}: value without a selected season")
    if not np.array_equal(frame[MAJOR].to_numpy(dtype=float), ledger["gs_last1_is_major"].to_numpy(dtype=float), equal_nan=True):
        raise ValueError(f"{MAJOR} differs from the selected latest season's ledger classification")
    return {"rows": len(frame)}


# --------------------------------------------------------------------------- safe reported-phase history (origin_safe)


def valid_phase_observations(labels: pd.DataFrame) -> pd.DataFrame:
    """Reported phase 1-5 observations (area_id, ord, overall_phase); duplicates fail."""
    for column in ("area_id", "year", "month", "overall_phase"):
        if column not in labels.columns:
            raise ValueError(f"labels missing {column}")
    if labels.duplicated(["area_id", "year", "month"]).any():
        raise ValueError("labels have duplicate area_id/year/month keys")
    phase = pd.to_numeric(labels["overall_phase"], errors="coerce")
    keep = phase.isin(VALID_PHASES)
    out = pd.DataFrame({
        "area_id": labels.loc[keep, "area_id"].to_numpy(),
        "ord": month_ord(labels.loc[keep, "year"], labels.loc[keep, "month"]),
        "overall_phase": phase[keep].astype(float).to_numpy(),
    })
    return out.sort_values(["area_id", "ord"], kind="mergesort").reset_index(drop=True)


def build_safe_history(observations: pd.DataFrame, rows: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Latest three reported phases of the same area with month ``<= min(O, T-1)``; missing stays NaN."""
    horizon = validate_horizon(horizon)
    if observations.duplicated(["area_id", "ord"]).any():
        raise ValueError("history observations have duplicate area/month keys")
    if not observations["overall_phase"].isin(VALID_PHASES).all():
        raise ValueError("history observations must be reported phases 1-5")
    work = rows.loc[:, ["area_id", "year", "month"]].reset_index(drop=True)
    target = month_ord(work["year"], work["month"])
    cutoff = label_cutoff(target, horizon)
    values = np.full((len(work), 3), np.nan)
    sources = np.full((len(work), 3), np.nan)
    obs_groups = {area: idx for area, idx in observations.groupby("area_id", sort=False).indices.items()}
    obs_ord = observations["ord"].to_numpy(dtype=np.int64)
    obs_phase = observations["overall_phase"].to_numpy(dtype=float)
    for area, indices in work.groupby("area_id", sort=False).indices.items():
        if area not in obs_groups:
            continue
        available = obs_groups[area]
        available = available[np.argsort(obs_ord[available], kind="mergesort")]
        times = obs_ord[available]
        positions = np.searchsorted(times, cutoff[indices], side="right")
        for k in range(3):
            previous = positions - 1 - k
            ok = previous >= 0
            source = available[previous[ok]]
            values[indices[ok], k] = obs_phase[source]
            sources[indices[ok], k] = obs_ord[source]
    block = pd.DataFrame(values, columns=list(HISTORY_VALUES))
    block[HISTORY_CHANGES[0]] = block[HISTORY_VALUES[0]] - block[HISTORY_VALUES[1]]
    block[HISTORY_CHANGES[1]] = block[HISTORY_VALUES[0]] - block[HISTORY_VALUES[2]]
    ledger = work.copy()
    ledger["horizon"] = horizon
    ledger["target_ord"] = target
    ledger["origin_ord"] = target - horizon
    ledger["label_cutoff_ord"] = cutoff
    for k in range(3):
        ledger[f"history_{k + 1}_source_ord"] = sources[:, k]
        ledger[f"history_{k + 1}_value"] = values[:, k]
    return block, ledger


def reference_history_check(observations: pd.DataFrame, ledger: pd.DataFrame, block: pd.DataFrame) -> Dict[str, int]:
    """Independent pure-Python latest-three replay over every ledger row; raises on any disagreement."""
    records: Dict[object, List[Tuple[int, float]]] = {}
    for area, ord_, phase in observations.loc[:, ["area_id", "ord", "overall_phase"]].itertuples(index=False, name=None):
        records.setdefault(area, []).append((int(ord_), float(phase)))
    for area in records:
        records[area].sort()
    present = 0
    for i, row in enumerate(ledger.itertuples(index=False)):
        cutoff = int(row.label_cutoff_ord)
        if cutoff != int(row.target_ord) - max(int(row.horizon), 1):
            raise AssertionError(f"row {i}: cutoff is not min(O, T-1)")
        allowed = [r for r in records.get(row.area_id, []) if r[0] <= cutoff]
        expected = list(reversed(allowed[-3:]))
        for k in range(3):
            src = getattr(row, f"history_{k + 1}_source_ord")
            val = block.iat[i, k]
            if k < len(expected):
                if src != expected[k][0] or val != expected[k][1]:
                    raise AssertionError(f"row {i} history_{k + 1}: got ({src}, {val}), expected {expected[k]}")
                if not src <= cutoff < row.target_ord:
                    raise AssertionError(f"row {i} history_{k + 1}: source month after cutoff")
                present += 1
            elif not (np.isnan(src) and np.isnan(val)):
                raise AssertionError(f"row {i} history_{k + 1}: expected missing history")
        h1, h2, h3 = block.iat[i, 0], block.iat[i, 1], block.iat[i, 2]
        for got, want in ((block.iat[i, 3], h1 - h2), (block.iat[i, 4], h1 - h3)):
            if not ((np.isnan(got) and np.isnan(want)) or got == want):
                raise AssertionError(f"row {i}: history change mismatch")
    return {"rows_checked": len(ledger), "history_values_present": present}


# --------------------------------------------------------------------------- national IDP (origin_safe)


def idp_observations(idp_monthly: pd.DataFrame) -> pd.DataFrame:
    """Observed non-null DTM admin0 stocks (iso3, report_ord, stock); validates one report per country-month."""
    required = {"admin0Pcode", "year", "month", "idp_ind", "observed", "reportingDate"}
    missing = sorted(required - set(idp_monthly.columns))
    if missing:
        raise ValueError(f"IDP admin0 table missing {missing}")
    if idp_monthly.duplicated(["admin0Pcode", "year", "month"]).any():
        raise ValueError("IDP admin0 table has duplicate country-month rows")
    obs = idp_monthly[idp_monthly["idp_ind"].notna()].copy()
    if (obs["observed"] != 1).any():
        raise ValueError("non-null IDP stock on a row not flagged observed")
    report = pd.to_datetime(obs["reportingDate"], errors="raise")
    if ((report.dt.year != obs["year"]) | (report.dt.month != obs["month"])).any():
        raise ValueError("IDP reportingDate month disagrees with the row month")
    if (obs["idp_ind"] < 0).any():
        raise ValueError("negative IDP stock")
    out = pd.DataFrame({
        "iso3": obs["admin0Pcode"].astype(str).str.strip().str.upper().to_numpy(),
        "report_ord": month_ord(obs["year"], obs["month"]),
        "stock": obs["idp_ind"].astype(float).to_numpy(),
    })
    return out.sort_values(["iso3", "report_ord"], kind="mergesort").reset_index(drop=True)


def build_idp_features(observations: pd.DataFrame, rows: pd.DataFrame, iso3: Sequence[Optional[str]], horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Latest observed national stock with report month ``<= O`` and its age ``O - report``; missing stays NaN."""
    horizon = validate_horizon(horizon)
    work = rows.loc[:, ["area_id", "year", "month"]].reset_index(drop=True)
    origin = month_ord(work["year"], work["month"]) - horizon
    codes = pd.Series(list(iso3), dtype="object").where(pd.notna(pd.Series(list(iso3), dtype="object")), None)
    stock = np.full(len(work), np.nan)
    report = np.full(len(work), np.nan)
    groups = observations.groupby("iso3", sort=False).indices
    obs_ord = observations["report_ord"].to_numpy(dtype=np.int64)
    obs_stock = observations["stock"].to_numpy(dtype=float)
    for code, indices in pd.Series(np.arange(len(work))).groupby(codes.to_numpy(), dropna=True).indices.items():
        if code not in groups:
            continue
        available = groups[code]
        times = obs_ord[available]
        positions = np.searchsorted(times, origin[indices], side="right") - 1
        ok = positions >= 0
        stock[indices[ok]] = obs_stock[available[positions[ok]]]
        report[indices[ok]] = times[positions[ok]]
    block = pd.DataFrame({IDP_FEATURES[0]: stock, IDP_FEATURES[1]: origin - report})
    ledger = work.copy()
    ledger["iso3"] = codes.to_numpy()
    ledger["horizon"] = horizon
    ledger["origin_ord"] = origin
    ledger["idp_report_ord"] = report
    ledger["idp_stock"] = stock
    return block, ledger


def reference_idp_check(observations: pd.DataFrame, ledger: pd.DataFrame, block: pd.DataFrame) -> Dict[str, int]:
    """Independent replay of the national IDP lookup on unique (iso3, origin) pairs."""
    by_code: Dict[str, List[Tuple[int, float]]] = {}
    for code, ord_, value in observations.loc[:, ["iso3", "report_ord", "stock"]].itertuples(index=False, name=None):
        by_code.setdefault(code, []).append((int(ord_), float(value)))
    cache: Dict[Tuple[object, int], Tuple[float, float]] = {}
    present = 0
    for i, row in enumerate(ledger.itertuples(index=False)):
        key = (row.iso3, int(row.origin_ord))
        if key not in cache:
            allowed = [r for r in by_code.get(row.iso3, []) if r[0] <= key[1]] if row.iso3 is not None else []
            cache[key] = (float(allowed[-1][0]), allowed[-1][1]) if allowed else (np.nan, np.nan)
        want_report, want_stock = cache[key]
        got_stock, got_age = block.iat[i, 0], block.iat[i, 1]
        if np.isnan(want_report):
            if not (np.isnan(got_stock) and np.isnan(got_age) and np.isnan(row.idp_report_ord)):
                raise AssertionError(f"row {i}: IDP should be missing")
            continue
        if row.idp_report_ord != want_report or got_stock != want_stock or got_age != key[1] - want_report or got_age < 0:
            raise AssertionError(f"row {i}: IDP mismatch")
        present += 1
    return {"rows_checked": len(ledger), "idp_present": present}


# --------------------------------------------------------------------------- identifiers and static context


def identifier_block(rows: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """lat/lon at the target key (raw panel) + target-calendar month_1..12 and year_2014..2026 dummies.

    Same semantics as the sibling ``add_identifier_features`` (validate many-to-one, missing lat/lon stops).
    ``coords``: admin_code/area_id, year, month, lat, lon at target keys.
    """
    work = rows.loc[:, ["area_id", "year", "month"]].reset_index(drop=True)
    look = coords.loc[:, ["area_id", "year", "month", "lat", "lon"]].copy()
    look["lat"] = pd.to_numeric(look["lat"], errors="coerce")
    look["lon"] = pd.to_numeric(look["lon"], errors="coerce")
    look = look.dropna(subset=["lat", "lon"]).drop_duplicates(["area_id", "year", "month"])
    merged = work.merge(look, on=["area_id", "year", "month"], how="left", validate="many_to_one")
    if merged[["lat", "lon"]].isna().any().any():
        raise ValueError(f"identifier merge produced missing lat/lon for {int(merged[['lat', 'lon']].isna().any(axis=1).sum())} rows")
    years = work["year"].to_numpy()
    if years.min() < 2014 or years.max() > 2026:
        raise ValueError("target year outside the fixed year_2014..year_2026 identifier columns")
    out = {"lat": merged["lat"].to_numpy(dtype=float), "lon": merged["lon"].to_numpy(dtype=float)}
    for m in range(1, 13):
        out[f"month_{m}"] = (work["month"].to_numpy() == m).astype(float)
    for y in range(2014, 2027):
        out[f"year_{y}"] = (years == y).astype(float)
    block = pd.DataFrame(out)
    if list(block.columns) != list(IDENTIFIER_FEATURES):
        raise AssertionError("identifier order")
    return block


def static_per_admin(tables: Sequence[pd.DataFrame], static_names: Sequence[str]) -> Tuple[pd.DataFrame, dict]:
    """One static29 row per admin from the compact tables; any within/across-table variation stops.

    ``tables`` carry ``area_id`` plus the static columns. A value missing everywhere stays NaN.
    """
    stacked = pd.concat([t.loc[:, ["area_id", *static_names]] for t in tables], ignore_index=True)
    nunique = stacked.groupby("area_id")[list(static_names)].nunique(dropna=True)
    varying = {c: int((nunique[c] > 1).sum()) for c in static_names if (nunique[c] > 1).any()}
    if varying:
        raise ValueError(f"static columns vary within an admin across the pinned compact tables: {varying}")
    # A table where an admin's value is missing while another table has it would be a source inconsistency too.
    present = stacked.groupby("area_id")[list(static_names)].count()
    rows_per = stacked.groupby("area_id").size()
    partial = {c: int(((present[c] > 0) & (present[c] < rows_per)).sum()) for c in static_names}
    partial = {c: n for c, n in partial.items() if n}
    if partial:
        raise ValueError(f"static value present in some rows/tables but missing in others for the same admin: {partial}")
    first = stacked.groupby("area_id")[list(static_names)].first()
    audit = {"admins": int(len(first)), "all_missing_admins_by_column": {c: int(first[c].isna().sum()) for c in static_names}}
    return first, audit
