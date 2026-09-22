"""R1-R3 preparation for the population-history experiment.

Builds, from the pinned IPCCH source, the two feature matrices the task
compares: ``original93`` exactly as the earlier IPCCH experiment defines it,
and ``rich561`` = original93 plus the 468 continuous phase-distribution history
columns frozen in ``config/feature-schema.json``.

Three properties this module is responsible for, because nothing downstream can
recover them:

* Every feature for a row is computed at that row's **own** origin ``o = T - h``
  (R3). A refit later in the calendar never re-dates a training row's history.
* History is read from the **full** valid ledger, never from a fitting or test
  subset (technical-contract.md 2). A window may reach back before the fitting
  window; it may never reach past ``o``.
* A missing derived value stays missing. No interpolation, no forward fill, no
  zero fill, and no row is dropped for lacking history (R2, R4).

The 93-column block, the target QC and the country lookup are reused from
``IPCCHGeoRFExperiment.prepare_data`` rather than reimplemented, so the two
experiments cannot drift apart on what a valid outcome is.

    python -B IPCCHPopulationHistoryExperiment/prepare_data.py \
        --source-root SOURCE_ROOT --run-dir NEW_RUN
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
CONFIG_DIR = PACKAGE_DIR / "config"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IPCCHGeoRFExperiment import prepare_data as ipcch  # noqa: E402


class PreparationError(RuntimeError):
    """Raised when a pinned input or contract invariant does not hold."""


# --------------------------------------------------------------------------
# Pinned runtime (technical-contract.md 4)
# --------------------------------------------------------------------------

#: Compared, not merely logged. A numerically different stack would silently
#: move F1 in the fourth decimal and make the frozen thresholds unreplayable.
PINNED_RUNTIME = {
    "python": "3.12.10",
    "numpy": "2.2.6",
    "pandas": "2.2.3",
    "sklearn": "1.6.1",
    "xgboost": "3.0.0",
    "scipy": "1.15.2",
    "geopandas": "1.0.1",
    "shapely": "2.1.0",
}

SOURCE_RELATIVE_PATH = "raw/IPCCH_2026_completed.csv"
COUNTRY_LOOKUP_RELATIVE_PATH = "country_area_id_lookup.csv"

#: Same convention as the inherited dataset, used for reporting strata only.
CH_ADMIN_CODE_FLOOR = 100000


def runtime_identity() -> dict:
    import numpy
    import pandas
    import scipy
    import sklearn
    import xgboost

    found = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "scipy": scipy.__version__,
    }
    # The baseline import path pulls these in; record them even though this
    # module never touches geometry.
    for name in ("geopandas", "shapely"):
        try:
            module = __import__(name)
            found[name] = getattr(module, "__version__", "")
        except Exception as exc:  # pragma: no cover - reported, not raised here
            found[name] = f"<import failed: {exc}>"
    return found


def verify_runtime(strict: bool = True) -> dict:
    found = runtime_identity()
    mismatches = {
        name: {"expected": expected, "actual": found.get(name)}
        for name, expected in PINNED_RUNTIME.items()
        if found.get(name) != expected
    }
    report = {
        "expected": dict(PINNED_RUNTIME),
        "actual": found,
        "mismatches": mismatches,
        "pass": not mismatches,
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    if strict and mismatches:
        raise PreparationError(f"runtime does not match the pinned stack: {mismatches}")
    return report


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Frozen inventories (feature-schema.json, candidate-configs.json)
# --------------------------------------------------------------------------

SERIES_ORDER = (
    "q2",
    "q3",
    "q4",
    "q5",
    "severity_index",
    "entropy",
    "concentration",
    "severe_fraction",
)
#: The seven series defined for every valid observation. ``severe_fraction``
#: is undefined at q3 == 0 and therefore carries its own support everywhere.
COMPLETE_SERIES = SERIES_ORDER[:7]
RATIO_SERIES = "severe_fraction"

WINDOWS: tuple[tuple[str, int | None], ...] = (
    ("m06", 6),
    ("m12", 12),
    ("m24", 24),
    ("m36", 36),
    ("all", None),
)
STATISTICS = ("mean", "std", "min", "max", "latest_minus_mean", "slope")

N_SLOTS = 6
CRISIS_THRESHOLD = 0.20

#: Block order; the appended 468 columns are emitted in exactly this sequence.
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


@dataclass(frozen=True)
class FrozenSpec:
    """The two JSON inventories plus their hashes, bound to one run."""

    schema: dict
    configs: dict
    schema_sha256: str
    configs_sha256: str
    original_features: tuple[str, ...]
    additional_features: tuple[str, ...]
    rich_features: tuple[str, ...]
    aliases: dict

    def identity(self) -> dict:
        return {
            "feature_schema_sha256": self.schema_sha256,
            "candidate_configs_sha256": self.configs_sha256,
            "original_count": len(self.original_features),
            "additional_count": len(self.additional_features),
            "rich_count": len(self.rich_features),
        }


def load_frozen_spec(config_dir: Path | str = CONFIG_DIR) -> FrozenSpec:
    """Load and validate the frozen inventories.

    Validation is structural, not cosmetic: the 93 original names must be the
    ones the reused builder actually emits, the appended names must be unique,
    disjoint from the original block, and count exactly what the schema claims.
    A silent mismatch here would produce a matrix whose columns do not mean what
    the contract says they mean.
    """
    config_dir = Path(config_dir)
    schema_path = config_dir / "feature-schema.json"
    configs_path = config_dir / "candidate-configs.json"
    for path in (schema_path, configs_path):
        if not path.is_file():
            raise PreparationError(f"frozen inventory missing: {path}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    configs = json.loads(configs_path.read_text(encoding="utf-8"))

    original = tuple(schema["original_features"])
    if original != tuple(ipcch.FEATURE_COLUMNS):
        raise PreparationError(
            "feature-schema.json original_features do not match "
            "IPCCHGeoRFExperiment.prepare_data.FEATURE_COLUMNS"
        )
    if tuple(schema["series_order"]) != SERIES_ORDER:
        raise PreparationError("series_order drifted from the implementation")
    if tuple(schema["windows"]) != tuple(name for name, _ in WINDOWS):
        raise PreparationError("windows drifted from the implementation")
    if tuple(schema["statistic_order"]) != STATISTICS:
        raise PreparationError("statistic_order drifted from the implementation")

    blocks = schema["additional_blocks"]
    if tuple(blocks) != BLOCK_ORDER:
        raise PreparationError(
            f"additional_blocks order {tuple(blocks)} != {BLOCK_ORDER}"
        )
    additional: list[str] = []
    for block in BLOCK_ORDER:
        additional.extend(blocks[block])
    additional_t = tuple(additional)

    if len(additional_t) != int(schema["additional_count"]):
        raise PreparationError(
            f"additional_blocks hold {len(additional_t)} names, "
            f"additional_count says {schema['additional_count']}"
        )
    if len(set(additional_t)) != len(additional_t):
        raise PreparationError("duplicate name in the appended inventory")
    overlap = sorted(set(additional_t) & set(original))
    if overlap:
        raise PreparationError(f"appended names collide with original93: {overlap}")

    rich = original + additional_t
    if len(rich) != int(schema["rich_count"]):
        raise PreparationError(
            f"rich width {len(rich)} != declared {schema['rich_count']}"
        )

    # The aliases exist so the same quantity is not materialised twice; every
    # target must be a real original93 column.
    aliases = dict(schema.get("aliases", {}))
    unknown = sorted(set(aliases.values()) - set(original))
    if unknown:
        raise PreparationError(f"alias targets are not original93 columns: {unknown}")
    shadowed = sorted(set(aliases) & set(additional_t))
    if shadowed:
        raise PreparationError(
            f"aliased names were also materialised as columns: {shadowed}"
        )

    return FrozenSpec(
        schema=schema,
        configs=configs,
        schema_sha256=sha256_file(schema_path),
        configs_sha256=sha256_file(configs_path),
        original_features=original,
        additional_features=additional_t,
        rich_features=rich,
        aliases=aliases,
    )


# --------------------------------------------------------------------------
# The eight series
# --------------------------------------------------------------------------


def compute_series(valid: pd.DataFrame) -> pd.DataFrame:
    """Derive the eight history series from one valid-ledger frame.

    Inputs are the exact normalized shares the ledger preserved as strings; the
    contract asks for their float64 representations here, while the binary
    classification history keeps using the ledger's exact label.
    """
    parts = []
    for column in ipcch.NORMALIZED_PHASE_COLUMNS:
        values = valid[column].to_numpy()
        blank = values == ""
        if blank.any():
            raise PreparationError(
                f"{column} is blank on {int(blank.sum())} valid rows; a valid "
                "row always reached normalization"
            )
        parts.append(values.astype(np.float64))
    p = np.column_stack(parts)  # (n, 5) ordered p1..p5

    if not np.isfinite(p).all():
        raise PreparationError("non-finite normalized share on a valid row")

    q2 = p[:, 1:].sum(axis=1)
    q3 = p[:, 2:].sum(axis=1)
    q4 = p[:, 3:].sum(axis=1)
    q5 = p[:, 4]

    weights = np.arange(1, 6, dtype=np.float64)
    severity = p @ weights
    concentration = (p * p).sum(axis=1)

    # 0*ln(0) = 0 by contract; np.where alone would still evaluate log(0).
    safe = np.where(p > 0.0, p, 1.0)
    entropy = -(np.where(p > 0.0, p * np.log(safe), 0.0)).sum(axis=1) / np.log(5.0)

    severe_fraction = np.where(q3 > 0.0, q4 / np.where(q3 > 0.0, q3, 1.0), np.nan)

    out = pd.DataFrame(
        {
            "q2": q2,
            "q3": q3,
            "q4": q4,
            "q5": q5,
            "severity_index": severity,
            "entropy": entropy,
            "concentration": concentration,
            "severe_fraction": severe_fraction,
        },
        index=valid.index,
    )
    for name in COMPLETE_SERIES:
        column = out[name].to_numpy()
        if not np.isfinite(column).all():
            raise PreparationError(f"series {name} is not finite on every valid row")
    return out


# --------------------------------------------------------------------------
# Observation index and window arithmetic
#
# Every window is a contiguous slice of one area's month-sorted observations,
# and no area in the pinned source has more than a couple of dozen of them. So
# each window is materialised as a padded (rows x span) gather and reduced
# directly, rather than through prefix sums.
#
# That choice is numerical, not stylistic. The one-pass prefix identity
# ``Var = E[x^2] - mean^2`` cancels catastrophically on the near-constant
# windows this data is full of: on a constant q3 window around 0.5 it returned
# std = 1e-7 where the true value is 0. A direct two-pass reduction returns 0.
# --------------------------------------------------------------------------

_KEY_SCALE = 1_000_000


@dataclass
class HistoryIndex:
    """The full valid ledger, indexed for own-origin history queries."""

    keys: np.ndarray  # packed (admin_code, month), strictly increasing
    admin: np.ndarray
    months: np.ndarray
    months_f: np.ndarray
    states: np.ndarray  # exact ledger binary label
    series: dict  # name -> float64 values, NaN where undefined
    last_crisis: np.ndarray
    last_noncrisis: np.ndarray
    last_entry_newer: np.ndarray
    last_exit_newer: np.ndarray
    run_start: np.ndarray
    max_area_span: int
    n: int


def build_history_index(valid: pd.DataFrame) -> HistoryIndex:
    """Index the full valid ledger.

    ``valid`` must be the complete set of R1-valid outcomes, not a fitting or
    evaluation subset: the contract reads history from all of it.
    """
    frame = valid.copy()
    frame["month_ord"] = ipcch.month_ordinal(frame["year"], frame["month"])
    frame["admin_code"] = frame["admin_code"].astype(np.int64)
    if frame.duplicated(["admin_code", "month_ord"]).any():
        raise PreparationError("valid ledger has duplicate (admin_code, month)")
    frame = frame.sort_values(["admin_code", "month_ord"], kind="mergesort").reset_index(
        drop=True
    )

    series = compute_series(frame)

    admin = frame["admin_code"].to_numpy(dtype=np.int64)
    months = frame["month_ord"].to_numpy(dtype=np.int64)
    if months.min() < 0 or months.max() >= _KEY_SCALE:
        raise PreparationError("month ordinal outside the packing range")
    keys = admin * _KEY_SCALE + months
    if not np.all(np.diff(keys) > 0):
        raise PreparationError("packed observation keys are not strictly increasing")

    states = pd.to_numeric(frame["ipcch_food_crisis"]).to_numpy(dtype=np.int64)
    if not np.isin(states, (0, 1)).all():
        raise PreparationError("valid ledger carries a non-binary label")

    n = len(frame)
    # Most recent record of each kind at or before index i, reset at every area
    # boundary so a neighbouring area can never supply an event.
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
    max_area_span = int(area_sizes.max()) if n else 0

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
        max_area_span=max_area_span,
        n=n,
    )


def _gather_window(lo: np.ndarray, hi: np.ndarray, span: int) -> tuple[np.ndarray, np.ndarray]:
    """Padded oldest-to-newest indices for each ``[lo, hi)`` plus an in-range mask."""
    offsets = np.arange(span, dtype=np.int64)
    index = lo[:, None] + offsets[None, :]
    mask = index < hi[:, None]
    return np.where(mask, index, 0), mask


def _window_statistics(
    values: np.ndarray, months_f: np.ndarray, index: np.ndarray, mask: np.ndarray
) -> dict:
    """mean/std/min/max/latest-minus-mean/slope over each gathered window.

    Support rules, applied per statistic rather than per window: mean, extrema
    and latest-minus-mean need one finite value, std needs two, slope needs
    three. Insufficient support is missing, never zero.
    """
    x = np.where(mask, values[index], np.nan)
    finite = np.isfinite(x)
    n = finite.sum(axis=1)
    rows = x.shape[0]

    nan = np.full(rows, np.nan, dtype=np.float64)
    has_one = n >= 1
    total = np.where(finite, x, 0.0).sum(axis=1)
    mean = np.where(has_one, total / np.where(has_one, n, 1), np.nan)

    deviation = np.where(finite, x - mean[:, None], 0.0)
    sq = (deviation * deviation).sum(axis=1)
    std = np.where(n >= 2, np.sqrt(np.maximum(sq / np.where(n >= 2, n, 1), 0.0)), np.nan)

    big = np.where(finite, x, np.inf)
    small = np.where(finite, x, -np.inf)
    minimum = np.where(has_one, big.min(axis=1), np.nan)
    maximum = np.where(has_one, small.max(axis=1), np.nan)

    # Columns run oldest to newest, so the latest finite value is the finite
    # column with the largest position.
    positions = np.where(finite, np.arange(x.shape[1])[None, :], -1)
    latest_column = positions.max(axis=1)
    latest = np.where(
        has_one, x[np.arange(rows), np.maximum(latest_column, 0)], np.nan
    )

    t = np.where(mask, months_f[index], np.nan)
    t_mean = np.where(
        has_one, np.where(finite, t, 0.0).sum(axis=1) / np.where(has_one, n, 1), np.nan
    )
    t_dev = np.where(finite, t - t_mean[:, None], 0.0)
    stt = (t_dev * t_dev).sum(axis=1)
    stx = (t_dev * deviation).sum(axis=1)
    enough = (n >= 3) & (stt > 0.0)
    slope = np.where(enough, stx / np.where(enough, stt, 1.0), np.nan)

    return {
        "n": n,
        "finite": finite,
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "latest": latest,
        "latest_minus_mean": np.where(has_one, latest - mean, nan),
        "slope": slope,
    }


# --------------------------------------------------------------------------
# The 468 appended columns
# --------------------------------------------------------------------------


def _pick(values: np.ndarray, index: np.ndarray, ok: np.ndarray) -> np.ndarray:
    out = np.full(index.shape, np.nan, dtype=np.float64)
    if ok.any():
        out[ok] = values[index[ok]]
    return out


def build_history_block(
    index: HistoryIndex,
    admin_code: np.ndarray,
    origin_ord: np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    """Compute the appended history columns for each ``(area, origin)`` row.

    ``origin_ord`` is the row's own ``o = T - h``. Observations at months
    strictly after ``o`` are invisible by construction: the range end comes
    from a right-insertion of ``o`` itself into that area's block.
    """
    admin_code = np.asarray(admin_code, dtype=np.int64)
    origin_ord = np.asarray(origin_ord, dtype=np.int64)
    if admin_code.shape != origin_ord.shape:
        raise PreparationError("admin_code and origin_ord lengths differ")
    rows = admin_code.size

    area_start = np.searchsorted(index.keys, admin_code * _KEY_SCALE, side="left")
    hi = np.searchsorted(index.keys, admin_code * _KEY_SCALE + origin_ord, side="right")
    if np.any(hi < area_start):
        raise PreparationError("history range end precedes its area block")
    span = index.max_area_span
    if np.any(hi - area_start > span):
        raise PreparationError("an area block is wider than the recorded maximum")

    columns: dict[str, np.ndarray] = {}
    origin_f = origin_ord.astype(np.float64)
    months_f = index.months_f
    states_f = index.states.astype(np.float64)

    # --- observation slots, newest first ----------------------------------
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
        values = index.series[name]
        stacked = np.empty((N_SLOTS, rows), dtype=np.float64)
        for j in range(N_SLOTS):
            stacked[j] = _pick(values, slot_index[j], slot_ok[j])
            columns[f"hist_{name}_obs{j + 1}"] = stacked[j]
        slot_value[name] = stacked

    # observation_timing: ages of slots 2..6 (slot 1's age is an alias of an
    # original93 column) and the five adjacent gaps.
    for j in range(1, N_SLOTS):
        columns[f"hist_age_obs{j + 1}"] = origin_f - slot_month[j]
    for j in range(N_SLOTS - 1):
        columns[f"hist_gap_obs{j + 1}_obs{j + 2}"] = slot_month[j] - slot_month[j + 1]

    # changes: raw difference newer-older and its per-month rate
    for name in SERIES_ORDER:
        stacked = slot_value[name]
        for j in range(N_SLOTS - 1):
            gap = slot_month[j] - slot_month[j + 1]
            difference = stacked[j] - stacked[j + 1]
            usable = np.isfinite(gap) & (gap > 0)
            rate = np.where(usable, difference / np.where(usable, gap, 1.0), np.nan)
            columns[f"hist_{name}_change_obs{j + 1}_obs{j + 2}"] = difference
            columns[f"hist_{name}_rate_obs{j + 1}_obs{j + 2}"] = rate

    # --- trends over the last 3 and last 6 slots --------------------------
    for label, depth in (("last3", 3), ("last6", 6)):
        lo = np.maximum(area_start, hi - depth)
        gathered, mask = _gather_window(lo, hi, depth)
        for name in SERIES_ORDER:
            stats = _window_statistics(index.series[name], months_f, gathered, mask)
            columns[f"hist_{name}_slope_{label}"] = stats["slope"]

    # --- calendar windows --------------------------------------------------
    window_ranges: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, width in WINDOWS:
        if width is None:
            lo = area_start
        else:
            lo = np.maximum(
                np.searchsorted(
                    index.keys,
                    admin_code * _KEY_SCALE + (origin_ord - width + 1),
                    side="left",
                ),
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

    # --- window support ----------------------------------------------------
    # The seven complete series share one mask; severe_fraction carries its own,
    # because a q3 == 0 observation contributes to neither its count nor span.
    for label, _width in WINDOWS:
        lo, gathered, mask = window_ranges[label]
        count = (hi - lo).astype(np.float64)
        has = count >= 1
        newest_month = _pick(months_f, np.where(has, hi - 1, 0), has)
        oldest_month = _pick(months_f, np.where(has, lo, 0), has)
        columns[f"hist_support_common_{label}_count"] = count
        columns[f"hist_support_common_{label}_span"] = newest_month - oldest_month
        if label != "all":  # the all-window age is an alias of an original column
            columns[f"hist_support_common_{label}_age"] = origin_f - newest_month

        ratio_finite = np.isfinite(np.where(mask, index.series[RATIO_SERIES][gathered], np.nan))
        ratio_count = ratio_finite.sum(axis=1)
        ratio_has = ratio_count >= 1
        positions = np.where(ratio_finite, np.arange(mask.shape[1])[None, :], -1)
        newest_column = positions.max(axis=1)
        oldest_column = np.where(
            ratio_finite, np.arange(mask.shape[1])[None, :], mask.shape[1]
        ).min(axis=1)
        row_index = np.arange(rows)
        ratio_newest = np.where(
            ratio_has,
            months_f[gathered[row_index, np.maximum(newest_column, 0)]],
            np.nan,
        )
        ratio_oldest = np.where(
            ratio_has,
            months_f[gathered[row_index, np.minimum(oldest_column, mask.shape[1] - 1)]],
            np.nan,
        )
        columns[f"hist_support_{RATIO_SERIES}_{label}_count"] = ratio_count.astype(np.float64)
        columns[f"hist_support_{RATIO_SERIES}_{label}_span"] = ratio_newest - ratio_oldest
        columns[f"hist_support_{RATIO_SERIES}_{label}_age"] = origin_f - ratio_newest

    # --- prior binary states ----------------------------------------------
    for j in range(1, N_SLOTS):
        columns[f"hist_crisis_obs{j + 1}"] = _pick(states_f, slot_index[j], slot_ok[j])

    # --- distance to the .20 threshold ------------------------------------
    q3_obs1 = slot_value["q3"][0]
    columns["hist_q3_margin_obs1"] = q3_obs1 - CRISIS_THRESHOLD
    columns["hist_q3_abs_margin_obs1"] = np.abs(q3_obs1 - CRISIS_THRESHOLD)
    abs_margin = np.abs(index.series["q3"] - CRISIS_THRESHOLD)
    for label, _width in WINDOWS:
        _lo, gathered, mask = window_ranges[label]
        stats = _window_statistics(abs_margin, months_f, gathered, mask)
        columns[f"hist_q3_{label}_abs_margin_mean"] = stats["mean"]
        columns[f"hist_q3_{label}_abs_margin_min"] = stats["min"]

    # --- observed crisis dynamics per window -------------------------------
    for label, _width in WINDOWS:
        lo, gathered, mask = window_ranges[label]
        count = (hi - lo).astype(np.float64)
        gathered_states = np.where(mask, states_f[gathered], np.nan)
        live = count >= 1
        fraction = np.where(
            live,
            np.where(mask, gathered_states, 0.0).sum(axis=1) / np.where(live, count, 1.0),
            np.nan,
        )
        # A pair exists only where BOTH endpoints are inside the window.
        pair = mask[:, :-1] & mask[:, 1:]
        older = np.where(pair, np.where(mask[:, :-1], gathered_states[:, :-1], 0.0), np.nan)
        newer = np.where(pair, np.where(mask[:, 1:], gathered_states[:, 1:], 0.0), np.nan)
        entries = ((older == 0.0) & (newer == 1.0) & pair).sum(axis=1).astype(np.float64)
        exits = ((older == 1.0) & (newer == 0.0) & pair).sum(axis=1).astype(np.float64)
        enough = count >= 2
        columns[f"hist_crisis_{label}_fraction"] = fraction
        columns[f"hist_crisis_{label}_entries"] = np.where(enough, entries, np.nan)
        columns[f"hist_crisis_{label}_exits"] = np.where(enough, exits, np.nan)
        # Eligible pairs are reported even when the counts are not, so a long
        # gap can never be mistaken for continuous observation.
        columns[f"hist_crisis_{label}_pairs"] = np.maximum(count - 1.0, 0.0)

    # --- events and the current run ---------------------------------------
    newest_index = np.maximum(hi - 1, 0)
    any_history = hi > area_start

    noncrisis_idx = np.where(any_history, index.last_noncrisis[newest_index], -1)
    noncrisis_ok = any_history & (noncrisis_idx >= area_start)
    columns["hist_noncrisis_age"] = origin_f - _pick(
        months_f, np.where(noncrisis_ok, noncrisis_idx, 0), noncrisis_ok
    )
    columns["hist_no_noncrisis"] = (~noncrisis_ok).astype(np.float64)

    for kind, table in (("entry", index.last_entry_newer), ("exit", index.last_exit_newer)):
        idx = np.where(any_history, table[newest_index], -1)
        # The change is dated at its later observed endpoint, which must itself
        # lie inside this row's visible history.
        ok = any_history & (idx >= area_start) & (idx <= hi - 1)
        columns[f"hist_{kind}_age"] = origin_f - _pick(months_f, np.where(ok, idx, 0), ok)
        columns[f"hist_no_{kind}"] = (~ok).astype(np.float64)

    run_begin = np.where(any_history, index.run_start[newest_index], 0)
    columns["hist_current_run_count"] = np.where(
        any_history, (hi - run_begin).astype(np.float64), np.nan
    )
    columns["hist_current_run_span"] = np.where(
        any_history,
        _pick(months_f, newest_index, any_history) - _pick(months_f, run_begin, any_history),
        np.nan,
    )

    # --- assemble in the frozen order -------------------------------------
    missing = [name for name in feature_names if name not in columns]
    if missing:
        raise PreparationError(f"history block did not produce: {missing[:10]}")
    extra = sorted(set(columns) - set(feature_names))
    if extra:
        raise PreparationError(f"history block produced unlisted columns: {extra[:10]}")

    block = np.empty((rows, len(feature_names)), dtype=np.float64)
    for position, name in enumerate(feature_names):
        values = columns[name]
        if np.isinf(values).any():
            raise PreparationError(f"engineered column {name} produced an infinity")
        block[:, position] = values
    return block


# --------------------------------------------------------------------------
# Matrices, keys and folds
# --------------------------------------------------------------------------

ACTIVE_HORIZONS = (1, 3, 6, 12)

DEVELOPMENT_TARGETS = ((2020, 1), (2022, 12))
MAIN_SCHEDULE = {
    1: ((2023, 2), (2025, 12)),
    3: ((2023, 4), (2025, 12)),
    6: ((2023, 7), (2025, 12)),
    12: ((2024, 1), (2025, 12)),
}
TRAIN_WINDOW_MONTHS = 36


def month_range(start: tuple[int, int], end: tuple[int, int]) -> list[int]:
    lo = int(ipcch.month_ordinal(start[0], start[1]))
    hi = int(ipcch.month_ordinal(end[0], end[1]))
    if hi < lo:
        raise PreparationError(f"empty month range {start}..{end}")
    return list(range(lo, hi + 1))


def build_fold_calendar() -> pd.DataFrame:
    """The complete scheduled fold list, empty months included (R12)."""
    rows = []
    for horizon in ACTIVE_HORIZONS:
        for target in month_range(*DEVELOPMENT_TARGETS):
            rows.append(("development", horizon, target, target - horizon))
    for horizon, (start, end) in MAIN_SCHEDULE.items():
        for target in month_range(start, end):
            rows.append(("main", horizon, target, target - horizon))
    frame = pd.DataFrame(rows, columns=["stage", "horizon_months", "target_ord", "origin_ord"])
    frame["target_month"] = ipcch.month_label(frame["target_ord"].to_numpy())
    frame["origin_month"] = ipcch.month_label(frame["origin_ord"].to_numpy())
    frame["fold_id"] = [
        f"{stage[:3]}_h{h:02d}_{month}"
        for stage, h, month in zip(frame["stage"], frame["horizon_months"], frame["target_month"])
    ]
    return frame


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def prepare(source_root: Path | str, run_dir: Path | str, strict_runtime: bool = True) -> dict:
    """Build every artifact the fitting stages read, into a fresh run root."""
    started = time.time()
    source_root = Path(source_root)
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise PreparationError(f"{run_dir} already exists and is not empty")

    source_path = source_root / SOURCE_RELATIVE_PATH
    country_path = source_root / COUNTRY_LOOKUP_RELATIVE_PATH
    for path in (source_path, country_path):
        if not path.is_file():
            raise PreparationError(f"pinned input missing: {path}")

    runtime = verify_runtime(strict=strict_runtime)
    spec = load_frozen_spec()

    # --- R1: targets -------------------------------------------------------
    ledger = ipcch.build_target_ledger(source_path, verify_hash=True)
    gate = ipcch.check_target_gate(ledger)
    if not gate["gate_pass"]:
        raise PreparationError(f"target gate failed: {gate['gate_mismatches']}")
    valid = ledger.valid().reset_index(drop=True)

    countries, country_audit = ipcch.load_country_lookup(country_path)
    covered = set(countries[ipcch.REFERENCE_ID_COLUMN].to_numpy(dtype=np.int64))
    needed = set(valid["admin_code"].astype(np.int64).to_numpy())
    uncovered = sorted(needed - covered)
    if uncovered:
        raise PreparationError(
            f"{len(uncovered)} areas with valid outcomes are absent from the "
            f"country lookup (first: {uncovered[:5]})"
        )

    # --- R2/R3: original93, then the appended history ---------------------
    matrix = ipcch.build_feature_matrix(ledger, source_path, horizons=ACTIVE_HORIZONS)
    frame = matrix.frame
    if tuple(matrix.feature_columns) != spec.original_features:
        raise PreparationError("original93 order does not match the frozen schema")

    target_ord = ipcch.month_ordinal(
        [int(value[:4]) for value in frame["target_month"]],
        [int(value[5:7]) for value in frame["target_month"]],
    )
    origin_ord = ipcch.month_ordinal(
        [int(value[:4]) for value in frame["origin_month"]],
        [int(value[5:7]) for value in frame["origin_month"]],
    )
    admin_code = frame["admin_code"].to_numpy(dtype=np.int64)
    horizon = frame["horizon_months"].to_numpy(dtype=np.int64)
    if not np.array_equal(target_ord - horizon, origin_ord):
        raise PreparationError("origin is not target minus horizon on every row")

    index = build_history_index(valid)
    block = build_history_block(index, admin_code, origin_ord, spec.additional_features)

    original_X = frame[list(spec.original_features)].to_numpy(dtype=np.float64)
    rich_X = np.concatenate([original_X, block], axis=1)
    if rich_X.shape[1] != len(spec.rich_features):
        raise PreparationError(
            f"rich matrix has {rich_X.shape[1]} columns, expected {len(spec.rich_features)}"
        )
    alias_audit = check_aliases(original_X, spec, index, admin_code, origin_ord)

    # --- keys, persistence and the regression target ----------------------
    share_lookup = pd.DataFrame(
        {
            "admin_code": valid["admin_code"].astype(np.int64).to_numpy(),
            "target_ord": ipcch.month_ordinal(valid["year"], valid["month"]),
            "q3_target": valid["normalized_p3plus_str"].to_numpy().astype(np.float64),
        }
    )
    country_map = dict(
        zip(
            countries[ipcch.REFERENCE_ID_COLUMN].to_numpy(dtype=np.int64),
            countries["country_key"].to_numpy(),
        )
    )

    keys = pd.DataFrame(
        {
            "admin_code": admin_code,
            "target_month": frame["target_month"].to_numpy(),
            "horizon_months": horizon,
            "target_ord": target_ord,
            "origin_ord": origin_ord,
            "origin_month": frame["origin_month"].to_numpy(),
            "ipcch_food_crisis": frame["ipcch_food_crisis"].to_numpy(dtype=np.int64),
            "persistence_b": frame["last_observed_label"].to_numpy(dtype=np.float64),
            "has_history": 1 - frame["no_observed_label_history"].to_numpy(dtype=np.int64),
            "country_key": [country_map[int(code)] for code in admin_code],
            "cohort": np.where(admin_code >= CH_ADMIN_CODE_FLOOR, "CH", "IPC"),
        }
    )
    keys = keys.merge(
        share_lookup, on=["admin_code", "target_ord"], how="left", validate="many_to_one"
    )
    if keys["q3_target"].isna().any():
        raise PreparationError("a supervised row has no normalized P3+ outcome")
    if len(keys) != len(frame):
        raise PreparationError("key join changed the row count")

    # b must exist exactly where history exists; the whole matched/full split
    # rests on these two agreeing.
    has_b = np.isfinite(keys["persistence_b"].to_numpy())
    if not np.array_equal(has_b, keys["has_history"].to_numpy().astype(bool)):
        raise PreparationError("persistence availability disagrees with has_history")
    observed_b = keys.loc[has_b, "persistence_b"].to_numpy()
    if not np.isin(observed_b, (0.0, 1.0)).all():
        raise PreparationError("persistence carries a non-binary value")

    # Every appended history column is computed at or before the row's origin;
    # the slot-1 age is the age of the newest observation any of them used.
    newest_age = frame["last_observed_label_age_months"].to_numpy(dtype=np.float64)
    if np.nanmin(newest_age) < 0:
        raise PreparationError("an observation newer than the row origin was used")

    # --- write ------------------------------------------------------------
    data_dir = run_dir / "data"
    inputs_dir = run_dir / "inputs"
    folds_dir = run_dir / "folds"
    for directory in (data_dir, inputs_dir, folds_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # original93 is stored as the first 93 columns of the rich matrix rather
    # than as a second file: the two can then never disagree, and the rich arms
    # are guaranteed to see byte-identical values in that block.
    np.save(data_dir / "rich561_X.npy", np.ascontiguousarray(rich_X))
    keys.to_csv(data_dir / "keys.csv.gz", index=False)
    ledger.frame.to_csv(data_dir / "target_ledger.csv.gz", index=False)
    valid.to_csv(data_dir / "target_ledger_valid.csv.gz", index=False)
    countries.to_csv(data_dir / "country_lookup.csv.gz", index=False)

    # The six slots each row actually used, so a window can be rebuilt by hand.
    slot_keys = _slot_provenance(index, admin_code, origin_ord)
    slot_keys.insert(0, "horizon_months", horizon)
    slot_keys.insert(0, "target_month", frame["target_month"].to_numpy())
    slot_keys.insert(0, "admin_code", admin_code)
    slot_keys.to_csv(data_dir / "history_source_keys.csv.gz", index=False)

    schema_frame = pd.DataFrame(
        {
            "position": np.arange(len(spec.rich_features)),
            "name": list(spec.rich_features),
            "block": ["original93"] * len(spec.original_features) + _block_labels(spec),
        }
    )
    schema_frame.to_csv(data_dir / "feature_schema.csv", index=False)

    calendar = build_fold_calendar()
    counts = _fold_support(calendar, keys)
    counts.to_csv(folds_dir / "calendar.csv", index=False)

    audit = _feature_audit(rich_X, spec.rich_features)
    _write_json(audit, data_dir / "feature_audit.json")

    manifest = {
        "run_dir": str(run_dir),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        "runtime": runtime,
        "spec": spec.identity(),
        "inputs": {
            "source_path": str(source_path),
            "source_sha256": ledger.source_sha256,
            "country_lookup": country_audit,
        },
        "target_gate": gate,
        "matrix": {
            "rows": int(len(keys)),
            "original_columns": len(spec.original_features),
            "rich_columns": len(spec.rich_features),
            "horizons": list(ACTIVE_HORIZONS),
            "rows_with_history": int(keys["has_history"].sum()),
            "rows_without_history": int((1 - keys["has_history"]).sum()),
            "positives": int(keys["ipcch_food_crisis"].sum()),
            "max_observations_per_area": index.max_area_span,
        },
        "original93_audit": matrix.audit,
        "alias_audit": alias_audit,
        "folds": {
            "scheduled": int(len(calendar)),
            "development_scheduled": int((calendar["stage"] == "development").sum()),
            "main_scheduled": int((calendar["stage"] == "main").sum()),
            "development_nonempty": int(
                ((counts["stage"] == "development") & (counts["test_rows"] > 0)).sum()
            ),
            "main_nonempty": int(((counts["stage"] == "main") & (counts["test_rows"] > 0)).sum()),
        },
        "feature_audit_summary": {
            "all_nan_columns": audit["all_nan_columns"],
            "constant_columns": audit["constant_columns"],
        },
    }
    _write_json(manifest, run_dir / "manifest.json")
    for name in ("feature-schema.json", "candidate-configs.json"):
        (inputs_dir / name).write_bytes((CONFIG_DIR / name).read_bytes())
    return manifest


def check_aliases(
    original_X: np.ndarray,
    spec: FrozenSpec,
    index: HistoryIndex,
    admin_code: np.ndarray,
    origin_ord: np.ndarray,
) -> dict:
    """Prove each aliased name really is the original93 column it reuses.

    Five history names are deliberately not materialised because an original93
    column already holds that exact quantity. That deduplication is only sound
    if the two definitions agree on every row, so each aliased quantity is
    recomputed here from this module's own index and compared. A disagreement
    would mean the rich matrix is silently missing a feature it claims to have.
    """
    position = {name: i for i, name in enumerate(spec.original_features)}
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
    expected["hist_crisis_age"] = origin_f - _pick(
        index.months_f, np.where(crisis_ok, crisis_index, 0), crisis_ok
    )
    expected["hist_no_crisis"] = (~crisis_ok).astype(np.float64)

    problems = []
    for alias, target in spec.aliases.items():
        if alias not in expected:
            problems.append(f"{alias}: no recomputation is implemented")
            continue
        actual = original_X[:, position[target]]
        want = expected[alias]
        if not np.array_equal(np.isnan(actual), np.isnan(want)):
            problems.append(f"{alias} vs {target}: missingness differs")
            continue
        live = ~np.isnan(want)
        if live.any() and not np.allclose(actual[live], want[live], rtol=0, atol=0):
            worst = float(np.abs(actual[live] - want[live]).max())
            problems.append(f"{alias} vs {target}: values differ, max |delta| {worst}")
    if problems:
        raise PreparationError("alias deduplication is unsound: " + "; ".join(problems))
    return {
        "aliases_checked": sorted(spec.aliases),
        "rule": "each aliased history name was recomputed and compared exactly "
        "against the original93 column it reuses",
    }


def _block_labels(spec: FrozenSpec) -> list[str]:
    labels: list[str] = []
    for block in BLOCK_ORDER:
        labels.extend([block] * len(spec.schema["additional_blocks"][block]))
    return labels


def _slot_provenance(
    index: HistoryIndex, admin_code: np.ndarray, origin_ord: np.ndarray
) -> pd.DataFrame:
    """The months of the six observation slots actually used by each row."""
    area_start = np.searchsorted(index.keys, admin_code * _KEY_SCALE, side="left")
    hi = np.searchsorted(index.keys, admin_code * _KEY_SCALE + origin_ord, side="right")
    out = {"origin_month": ipcch.month_label(origin_ord)}
    for j in range(N_SLOTS):
        idx = hi - 1 - j
        ok = idx >= area_start
        month = np.full(idx.shape, -1, dtype=np.int64)
        month[ok] = index.months[idx[ok]]
        out[f"obs{j + 1}_month"] = ipcch.month_label(month)
    out["observations_available"] = (hi - area_start).astype(np.int64)
    return pd.DataFrame(out)


def _fold_support(calendar: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Per-fold test/full/matched counts, so an empty fold is visible up front."""
    target = keys["target_ord"].to_numpy()
    horizon = keys["horizon_months"].to_numpy()
    history = keys["has_history"].to_numpy().astype(bool)

    rows = []
    for record in calendar.itertuples(index=False):
        h = int(record.horizon_months)
        T = int(record.target_ord)
        O = int(record.origin_ord)
        same_h = horizon == h
        test = same_h & (target == T)
        full = same_h & (target >= O - TRAIN_WINDOW_MONTHS + 1) & (target <= O)
        rows.append(
            {
                "fold_id": record.fold_id,
                "stage": record.stage,
                "horizon_months": h,
                "target_month": record.target_month,
                "origin_month": record.origin_month,
                "target_ord": T,
                "origin_ord": O,
                "test_rows": int(test.sum()),
                "test_rows_with_history": int((test & history).sum()),
                "test_rows_without_history": int((test & ~history).sum()),
                "full_pool_rows": int(full.sum()),
                "matched_pool_rows": int((full & history).sum()),
            }
        )
    return pd.DataFrame(rows)


def _feature_audit(X: np.ndarray, names: Sequence[str]) -> dict:
    nan_rate = np.isnan(X).mean(axis=0)
    finite_min = np.where(np.isnan(X), np.inf, X).min(axis=0)
    finite_max = np.where(np.isnan(X), -np.inf, X).max(axis=0)
    all_nan = [names[i] for i in np.where(nan_rate >= 1.0)[0]]
    constant = [
        names[i]
        for i in range(len(names))
        if nan_rate[i] < 1.0 and finite_min[i] == finite_max[i]
    ]
    return {
        "rows": int(X.shape[0]),
        "columns": int(X.shape[1]),
        "infinities": int(np.isinf(X).sum()),
        "all_nan_columns": all_nan,
        "constant_columns": constant,
        "nan_rate": {names[i]: float(nan_rate[i]) for i in range(len(names))},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--allow-runtime-drift",
        action="store_true",
        help="record a runtime mismatch instead of stopping (diagnostics only)",
    )
    args = parser.parse_args(argv)

    manifest = prepare(
        args.source_root, args.run_dir, strict_runtime=not args.allow_runtime_drift
    )
    print(json.dumps({k: manifest[k] for k in ("matrix", "folds", "spec")}, indent=2))
    print(f"prepared: {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
