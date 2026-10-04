"""Origin-aligned predictors for the four-class baseline (feature-contract.md).

Every row is a key (area a, target month T, horizon H) with origin O = T - H.
Covariates are read from the complete monthly scaffold at calendar offsets from
O, and outcome history uses only the area's own observed assessments at months
<= O. Nothing here is lagged by row position, rolled across areas, interpolated
or imputed; missing stays NaN until each estimator fits its own imputer.

Months are integer indices ``year * 12 + (month - 1)``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_PATH_ENV = "FOURCLASS_FEATURE_SCHEMA"
HISTORY_OFFSETS = (0, 4, 8, 12)
CHANGE_PAIRS = ((0, 4), (4, 8), (8, 12), (0, 12))
WINDOWS = (12, 24, 36)
N_CLASSES = 4


def month_index(dates) -> np.ndarray:
    stamps = pd.to_datetime(pd.Series(dates))
    if (stamps.dt.day != 1).any():
        raise ValueError("months must start on day 1")
    return (stamps.dt.year * 12 + stamps.dt.month - 1).to_numpy(dtype=np.int64)


def month_label(index) -> np.ndarray:
    index = np.asarray(index, dtype=np.int64)
    return np.array([f"{i // 12:04d}-{i % 12 + 1:02d}" for i in index])


def load_schema(path: Path) -> dict:
    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    ordered = schema["ordered_features"]
    history = [c for block in schema["history_blocks"].values() for c in block]
    declared = (schema["static_sources"] + schema["dynamic_sources_at_origin"]
                + schema["legacy_covariate_derived"] + schema["known_calendar"] + history)
    if ordered != declared or len(set(ordered)) != len(ordered):
        raise ValueError("feature schema order does not match its declared blocks")
    if len(ordered) != schema["proposed_feature_count"]:
        raise ValueError("feature schema count mismatch")
    return schema


# --------------------------------------------------------------------------------------
# Covariates
# --------------------------------------------------------------------------------------

class Scaffold:
    """Area x month arrays over the complete monthly panel."""

    def __init__(self, panel: pd.DataFrame, columns):
        months = month_index(panel["date"])
        areas = panel["FEWSNET_admin_code"].to_numpy(dtype=np.int64)
        self.areas = np.unique(areas)
        self.first_month = int(months.min())
        self.n_months = int(months.max()) - self.first_month + 1
        if len(panel) != self.areas.size * self.n_months:
            raise ValueError("panel is not a complete area x month scaffold")
        self.area_pos = {int(a): i for i, a in enumerate(self.areas)}
        row = np.searchsorted(self.areas, areas)
        col = months - self.first_month
        if len(np.unique(row * self.n_months + col)) != len(panel):
            raise ValueError("duplicate area-month keys in scaffold")
        self.arrays = {}
        for name in columns:
            grid = np.full((self.areas.size, self.n_months), np.nan)
            grid[row, col] = panel[name].to_numpy(dtype=float)
            self.arrays[name] = grid

    def at(self, name, area_rows, months):
        """Exact value at a calendar month; NaN outside the scaffold."""
        col = np.asarray(months, dtype=np.int64) - self.first_month
        inside = (col >= 0) & (col < self.n_months)
        out = np.full(col.shape, np.nan)
        out[inside] = self.arrays[name][area_rows[inside], col[inside]]
        return out

    def trailing_sum(self, name, area_rows, origins, width):
        """Sum over [O-width, O-1]; NaN unless all ``width`` real values exist."""
        total = np.zeros(len(origins))
        for k in range(1, width + 1):
            total = total + self.at(name, area_rows, origins - k)
        return total


#: Legacy derived covariate -> its source column (aligned with that source's rule).
LEGACY_BASE = {"WFP_Price_m4": "WFP_Price", "WFP_Price_m12": "WFP_Price", "nightlight_m12": "nightlight",
               **{f"EVI_l{k}": "EVI" for k in range(1, 13)}}
ALIGNMENT_KINDS = ("static", "monthly", "annual", "excluded")
ALIGNMENT_STATUS = ("verified_vintage", "reconstructed", "synthetic")


def _release_months(rule: dict) -> dict:
    """Optional actual-release evidence: reference (month index, or year for annual) -> release month."""
    out = {}
    for ref, date in (rule.get("releases") or {}).items():
        stamp = pd.Timestamp(date)
        key = int(ref) if rule["kind"] == "annual" else int(month_index([f"{ref}-01"])[0])
        out[key] = stamp.year * 12 + stamp.month - 1
    return out


def check_alignment(schema: dict, alignment: dict, real: bool = False) -> dict:
    """Validate a per-source availability table (interruption design D1, D7).

    ``alignment[name]`` = {"kind": static|monthly|annual|excluded, "status": verified_vintage|
    reconstructed|synthetic, "evidence": citation, plus "lag" (monthly, months >= 0) or
    "release_delay_years"/"release_month"/"value_month" (annual)}, optionally "releases": actual
    release dates per reference ("YYYY-MM" monthly, year annual -> "YYYY-MM-DD"). Every static/
    dynamic source needs an entry; there are no defaults, so this code adopts no lag. ``real``
    refuses synthetic entries."""
    if not isinstance(alignment, dict):
        raise ValueError("alignment must be a dict of per-source rules")
    sources = schema["static_sources"] + schema["dynamic_sources_at_origin"]
    if set(alignment) != set(sources):
        raise ValueError(f"alignment must cover exactly the schema sources: "
                         f"missing {sorted(set(sources) - set(alignment))}, extra {sorted(set(alignment) - set(sources))}")
    for name, rule in alignment.items():
        kind = rule.get("kind")
        if kind not in ALIGNMENT_KINDS:
            raise ValueError(f"{name}: unknown kind {kind!r}")
        if rule.get("status") not in ALIGNMENT_STATUS or not rule.get("evidence"):
            raise ValueError(f"{name}: status and evidence are required")
        if real and rule["status"] == "synthetic":
            raise ValueError(f"{name}: synthetic availability rule in a real run")
        if kind == "monthly" and not (isinstance(rule.get("lag"), int) and rule["lag"] >= 0):
            raise ValueError(f"{name}: monthly sources need an integer lag >= 0")
        if kind == "annual":
            ok = (isinstance(rule.get("release_delay_years"), int) and rule["release_delay_years"] >= 0
                  and rule.get("release_month") in range(1, 13) and rule.get("value_month") in range(1, 13))
            if not ok:
                raise ValueError(f"{name}: annual sources need release_delay_years, release_month, value_month")
        if rule.get("releases"):
            if kind not in ("monthly", "annual"):
                raise ValueError(f"{name}: release evidence applies to monthly/annual sources only")
            try:
                released = _release_months(rule)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{name}: malformed release evidence ({exc})") from None
            for ref, month in released.items():
                if month < (ref * 12 if kind == "annual" else ref):
                    raise ValueError(f"{name}: a release precedes its reference period")
    for derived, base in LEGACY_BASE.items():
        if derived in schema["legacy_covariate_derived"] and alignment[base]["kind"] not in ("monthly", "excluded"):
            raise ValueError(f"{derived}: its source {base} must be monthly or excluded")
    return alignment


def aligned_feature_names(schema: dict, alignment: dict) -> list:
    """Schema order without excluded sources and the legacy columns derived from them."""
    dropped = {n for n, r in alignment.items() if r["kind"] == "excluded"}
    dropped |= {d for d, b in LEGACY_BASE.items() if b in dropped}
    return [c for c in schema["ordered_features"] if c not in dropped]


def annual_reference_year(origins, rule: dict) -> np.ndarray:
    """Latest eligible annual reference year at each origin cutoff (float; NaN if none).

    With actual release evidence: the latest evidenced reference year released by the origin
    month (a delayed release is not exposed). Otherwise the documented calendar: reference year r
    is released in ``release_month`` of year r + ``release_delay_years``."""
    origins = np.asarray(origins, dtype=np.int64)
    released = _release_months(rule)
    if not released:
        return ((origins - (rule["release_month"] - 1)) // 12 - rule["release_delay_years"]).astype(float)
    refs = np.array(sorted(released), dtype=np.int64)
    months = np.array([released[r] for r in refs], dtype=np.int64)
    out = np.full(origins.shape, np.nan)
    for i, o in enumerate(origins):
        eligible = refs[months <= o]
        if eligible.size:
            out[i] = eligible.max()
    return out


def source_release_month(rule: dict, refs) -> np.ndarray:
    """Evidenced release month of each reference (month index or year); NaN without evidence."""
    released = _release_months(rule)
    return np.array([released.get(int(r), np.nan) if np.isfinite(r) else np.nan
                     for r in np.asarray(refs, dtype=float)], dtype=float)


def covariate_features(scaffold: Scaffold, schema: dict, areas, targets, origins,
                       alignment: dict | None = None) -> pd.DataFrame:
    """Covariates at calendar offsets from O.

    ``alignment`` None (default) keeps the frozen exact-origin values. Any other value is first
    validated by ``check_alignment``: monthly sources and their legacy lags/sums are read relative
    to the exact source month O - lag; annual sources take the value of their latest eligible
    reference year (``annual_reference_year``) at ``value_month``; static sources stay at O;
    excluded sources and their derived columns are omitted. Where actual release evidence is
    given, a selected source month/year not released by the origin cutoff stays NaN: the
    eligible publication is chosen first and a missing value is never replaced by searching back."""
    origins = np.asarray(origins, dtype=np.int64)
    rows = np.array([scaffold.area_pos[int(a)] for a in areas], dtype=np.int64)
    rule = {} if alignment is None else check_alignment(schema, alignment)

    def read(name, months):
        """Value at source months, NaN where release evidence says it is not out by the origin."""
        months = np.asarray(months, dtype=np.int64)
        values = scaffold.at(name, rows, months)
        r = rule.get(name)
        if r is not None and r.get("releases"):
            released = source_release_month(r, months)
            values = np.where(released <= origins, values, np.nan)   # NaN release -> not released
        return values

    def end(name):  # source-month endpoint replacing O
        r = rule.get(name)
        return origins if r is None or r["kind"] == "static" else origins - r["lag"]

    out = {}
    for name in schema["static_sources"] + schema["dynamic_sources_at_origin"]:
        r = rule.get(name)
        if r is not None and r["kind"] == "excluded":
            continue
        if r is not None and r["kind"] == "annual":
            ref = annual_reference_year(origins, r)
            month = np.where(np.isfinite(ref), np.nan_to_num(ref) * 12 + r["value_month"] - 1, -10 ** 9)
            out[name] = np.where(np.isfinite(ref), scaffold.at(name, rows, month.astype(np.int64)), np.nan)
        else:
            out[name] = read(name, end(name))

    def keep(base):  # frozen default: always; aligned: only a retained monthly source
        return alignment is None or rule.get(base, {}).get("kind") == "monthly"

    for derived, width in (("WFP_Price_m4", 4), ("WFP_Price_m12", 12), ("nightlight_m12", 12)):
        base = LEGACY_BASE[derived]
        if keep(base):
            total = np.zeros(len(origins))
            for k in range(1, width + 1):   # same [E-width, E-1] window as Scaffold.trailing_sum
                total = total + read(base, end(base) - k)
            out[derived] = total
    if keep("EVI"):
        for k in range(1, 13):
            out[f"EVI_l{k}"] = read("EVI", end("EVI") - k)
    targets = np.asarray(targets, dtype=np.int64)
    angle = 2 * np.pi * (targets % 12) / 12
    out["target_year"] = (targets // 12).astype(float)
    out["target_month_sin"] = np.sin(angle)
    out["target_month_cos"] = np.cos(angle)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------
# Outcome history
# --------------------------------------------------------------------------------------

def _area_history(om, op, origins):
    """History features for one area. ``om``/``op`` are sorted observed months and
    merged phases 1..4; ``origins`` the origin months of this area's keys."""
    n_keys = origins.size
    feats = {}
    lookup = dict(zip(om.tolist(), op.tolist()))
    for k in HISTORY_OFFSETS:
        phase = np.array([lookup.get(int(o) - k, np.nan) for o in origins], dtype=float)
        feats[f"hist_phase_o{k:02d}"] = phase
        feats[f"hist_crisis_o{k:02d}"] = np.where(np.isnan(phase), np.nan, (phase >= 3).astype(float))
    for newer, older in CHANGE_PAIRS:
        delta = feats[f"hist_phase_o{newer:02d}"] - feats[f"hist_phase_o{older:02d}"]
        feats[f"hist_delta_o{newer:02d}_o{older:02d}"] = delta
        feats[f"hist_direction_o{newer:02d}_o{older:02d}"] = np.sign(delta)

    n_obs = om.size
    onehot = np.zeros((n_obs + 1, N_CLASSES))
    if n_obs:
        onehot[1:] = np.cumsum(np.eye(N_CLASSES)[op.astype(int) - 1], axis=0)
    up = np.zeros(max(n_obs, 1))
    down = np.zeros(max(n_obs, 1))
    if n_obs > 1:
        up[1:n_obs] = np.cumsum(op[1:] > op[:-1])
        down[1:n_obs] = np.cumsum(op[1:] < op[:-1])
    # Running index of the latest qualifying observation/event at or before record i.
    last_crisis = np.full(n_obs, -1)
    last_severe = np.full(n_obs, -1)
    last_change = np.full(n_obs, -1)
    run_start = np.zeros(n_obs, dtype=np.int64)
    for i in range(n_obs):
        last_crisis[i] = i if op[i] >= 3 else (last_crisis[i - 1] if i else -1)
        last_severe[i] = i if op[i] == 4 else (last_severe[i - 1] if i else -1)
        changed = i > 0 and op[i] != op[i - 1]
        last_change[i] = i if changed else (last_change[i - 1] if i else -1)
        run_start[i] = i if (i == 0 or changed) else run_start[i - 1]

    hi_all = np.searchsorted(om, origins, side="right")
    for W in WINDOWS:
        lo = np.searchsorted(om, origins - W + 1, side="left")
        n = hi_all - lo
        prefix = f"hist_w{W}_"
        cols = {name: np.full(n_keys, np.nan) for name in (
            "frac1", "frac2", "frac3", "frac4or5", "min_phase", "max_phase",
            "up_rate", "down_rate", "span_months", "latest_age")}
        for j in range(n_keys):
            a, b = lo[j], hi_all[j]
            if b <= a:
                continue
            counts = onehot[b] - onehot[a]
            fracs = counts / (b - a)
            for name, value in zip(("frac1", "frac2", "frac3", "frac4or5"), fracs):
                cols[name][j] = value
            window = op[a:b]
            cols["min_phase"][j] = window.min()
            cols["max_phase"][j] = window.max()
            cols["span_months"][j] = om[b - 1] - om[a]
            cols["latest_age"][j] = origins[j] - om[b - 1]
            if b - a >= 2:
                cols["up_rate"][j] = (up[b - 1] - up[a]) / (b - a - 1)
                cols["down_rate"][j] = (down[b - 1] - down[a]) / (b - a - 1)
        for name in ("frac1", "frac2", "frac3", "frac4or5", "min_phase", "max_phase",
                     "up_rate", "down_rate"):
            feats[prefix + name] = cols[name]
        feats[prefix + "n_obs"] = n.astype(float)
        feats[prefix + "n_pairs"] = np.maximum(n - 1, 0).astype(float)
        feats[prefix + "span_months"] = cols["span_months"]
        feats[prefix + "latest_age"] = cols["latest_age"]

    latest = hi_all - 1
    has = latest >= 0
    safe = np.where(has, latest, 0)

    def event_age(index_array):
        idx = np.where(has, index_array[safe] if n_obs else -1, -1)
        present = idx >= 0
        age = np.where(present, origins - om[np.where(present, idx, 0)] if n_obs else np.nan, np.nan)
        return age.astype(float), (~present).astype(float)

    feats["hist_latest_observed_phase"] = np.where(has, op[safe] if n_obs else np.nan, np.nan).astype(float)
    feats["hist_latest_observed_age"] = np.where(has, origins - om[safe] if n_obs else np.nan, np.nan).astype(float)
    feats["hist_no_history"] = (~has).astype(float)
    feats["hist_crisis_age"], feats["hist_crisis_absent"] = event_age(last_crisis)
    feats["hist_phase4or5_age"], feats["hist_phase4or5_absent"] = event_age(last_severe)
    feats["hist_change_age"], feats["hist_change_absent"] = event_age(last_change)
    if n_obs:
        start = run_start[safe]
        feats["hist_run_n"] = np.where(has, safe - start + 1, np.nan).astype(float)
        feats["hist_run_span"] = np.where(has, om[safe] - om[start], np.nan).astype(float)
    else:
        feats["hist_run_n"] = np.full(n_keys, np.nan)
        feats["hist_run_span"] = np.full(n_keys, np.nan)

    origin_phase = feats["hist_phase_o00"]
    for k in range(1, N_CLASSES + 1):
        indicator = np.where(np.isnan(origin_phase), np.nan, (origin_phase == k).astype(float))
        for summary in ("up_rate", "down_rate", "frac4or5"):
            feats[f"hist_origin_phase{k}_x_w12_{summary}"] = indicator * feats[f"hist_w12_{summary}"]
    return feats


def history_features(observations: pd.DataFrame, areas, origins) -> pd.DataFrame:
    """``observations`` holds valid records only: area, month (index), phase (merged 1..4).

    Returns features aligned to the (areas, origins) key order. Observations after a
    key's origin are never visible to it.
    """
    areas = np.asarray(areas, dtype=np.int64)
    origins = np.asarray(origins, dtype=np.int64)
    obs = observations.sort_values(["area", "month"])
    if obs.duplicated(["area", "month"]).any():
        raise ValueError("duplicate observed area-months")
    if not obs["phase"].isin([1, 2, 3, 4]).all():
        raise ValueError("history phases must be merged classes 1..4")
    grouped = {int(a): (g["month"].to_numpy(np.int64), g["phase"].to_numpy(float))
               for a, g in obs.groupby("area", sort=False)}
    empty = (np.array([], dtype=np.int64), np.array([], dtype=float))
    order = np.argsort(areas, kind="stable")
    result = None
    for area in np.unique(areas):
        pos = order[np.searchsorted(areas[order], area, "left"):np.searchsorted(areas[order], area, "right")]
        om, op = grouped.get(int(area), empty)
        feats = _area_history(om, op, origins[pos])
        if result is None:
            result = {name: np.full(areas.size, np.nan) for name in feats}
        for name, values in feats.items():
            result[name][pos] = values
    return pd.DataFrame(result)
