"""Calendars: Stage1 F/S split (R35), rolling fold schedule (R47), windows (R23, R37).

The F/S split is adapted from ``IPCCHGeoRFExperiment/prepare_data.py``
``build_stage1_split``: same within-area earliest-floor(n/2) rule on original
outcomes, now over the >=0.20 truth and with no inherited count gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ipcch_climate_geoxgb.errors import ContractError
from ipcch_climate_geoxgb.features import month_label, parse_month


def stage1_split(valid: pd.DataFrame, first_month: str, last_month: str, universe) -> tuple[pd.DataFrame, dict]:
    """One row per original 2014-01..2022-12 valid outcome with role fit/validation/singleton.

    Within an area with n >= 2 outcomes the earliest floor(n/2) are F and the
    rest S (odd n puts the extra one in S). n == 1 is a singleton: no fit,
    scan or gate role. Split before horizon expansion, so all four H views of
    one outcome share a side.
    """
    lo, hi = parse_month(first_month), parse_month(last_month)
    pool = valid[(valid["month_ord"] >= lo) & (valid["month_ord"] <= hi)]
    pool = pool.sort_values(["admin_code", "month_ord"], kind="mergesort").reset_index(drop=True)
    size = pool.groupby("admin_code")["month_ord"].transform("size").to_numpy()
    rank = pool.groupby("admin_code").cumcount().to_numpy()
    role = np.where(size == 1, "singleton", np.where(rank < size // 2, "fit", "validation"))
    out = pool[["admin_code", "month_ord", "phase_truth", "crisis_truth"]].copy()
    out["target_month"] = month_label(out["month_ord"].to_numpy())
    out["split_role"] = role
    if out.duplicated(["admin_code", "month_ord"]).any():
        raise ContractError("an original outcome appears twice in the Stage1 split")
    # within-area ordering: every F month precedes every S month of that area
    fit_max = out[out.split_role == "fit"].groupby("admin_code")["month_ord"].max()
    val_min = out[out.split_role == "validation"].groupby("admin_code")["month_ord"].min()
    joined = pd.concat([fit_max, val_min], axis=1, keys=["fit_max", "val_min"]).dropna()
    if (joined["fit_max"] >= joined["val_min"]).any():
        raise ContractError("a Stage1 F outcome is not earlier than its area's S outcomes")
    with_outcomes = set(out["admin_code"].tolist())
    counts = out["split_role"].value_counts().to_dict()
    audit = {
        "window": [first_month, last_month],
        "original_outcomes": int(len(out)),
        "fit": int(counts.get("fit", 0)),
        "validation": int(counts.get("validation", 0)),
        "singleton_areas": int(counts.get("singleton", 0)),
        "multi_outcome_areas": int(out.loc[out.split_role != "singleton", "admin_code"].nunique()),
        "zero_outcome_areas": int(sum(1 for a in universe if int(a) not in with_outcomes)),
        "crisis_fit": int(out.loc[out.split_role == "fit", "crisis_truth"].sum()),
        "crisis_validation": int(out.loc[out.split_role == "validation", "crisis_truth"].sum()),
    }
    return out, audit


def main_fold_calendar(contract: dict) -> pd.DataFrame:
    """All scheduled main H x target-month folds (122), origins >= 2023-01."""
    cal = contract["calendar"]
    last = parse_month(cal["main_evaluation_last_target_month"])
    rows = []
    for h in cal["horizons_months"]:
        first = parse_month(cal["main_first_target_month"][str(h)])
        for target in range(first, last + 1):
            rows.append(("main", h, target, target - h))
    frame = pd.DataFrame(rows, columns=["period", "horizon_months", "target_ord", "origin_ord"])
    if len(frame) != cal["main_fold_total"]:
        raise ContractError(f"main calendar has {len(frame)} folds, contract says {cal['main_fold_total']}")
    if (frame["origin_ord"] < parse_month("2023-01")).any():
        raise ContractError("a main fold origin precedes 2023-01")
    return _label(frame)


def supplementary_calendar(contract: dict, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2026 months that have >= 1 QC-valid outcome in the frozen source, x all H.

    Also returns the 12-month coverage ledger; absent months are recorded as
    having no QC-valid outcome in this source (not as forecast failures).
    """
    cal = contract["calendar"]
    year = int(cal["supplementary_year"])
    coverage = []
    observed = []
    for month in range(1, 13):
        ordinal = year * 12 + month - 1
        n = int((valid["month_ord"] == ordinal).sum())
        coverage.append(
            {
                "target_month": f"{year:04d}-{month:02d}",
                "target_ord": ordinal,
                "valid_outcomes": n,
                "status": "observed" if n else "no_qc_valid_outcome_in_source",
            }
        )
        if n:
            observed.append(ordinal)
    rows = [("supplementary", h, t, t - h) for h in cal["horizons_months"] for t in observed]
    frame = pd.DataFrame(rows, columns=["period", "horizon_months", "target_ord", "origin_ord"])
    return _label(frame), pd.DataFrame(coverage)


def _label(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["target_month"] = month_label(frame["target_ord"].to_numpy())
    frame["origin_month"] = month_label(frame["origin_ord"].to_numpy())
    frame["fold_id"] = [
        f"{p[:4]}_h{h:02d}_{t}" for p, h, t in zip(frame["period"], frame["horizon_months"], frame["target_month"])
    ]
    return frame


def attach_fold_support(folds: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Number of QC-valid evaluation keys per fold; empty folds are kept (R47)."""
    counts = valid.groupby("month_ord").size()
    out = folds.copy()
    out["eval_keys"] = out["target_ord"].map(counts).fillna(0).astype(np.int64)
    out["status"] = np.where(out["eval_keys"] > 0, "scheduled", "no_valid_target")
    return out


def training_window(origin_ord: int, months: int = 36) -> tuple[int, int]:
    """Closed target-month window [O - 35, O] (R23)."""
    return origin_ord - (months - 1), origin_ord


def historical_gate_dates(observed_months: np.ndarray, origin_ord: int, max_dates: int = 6) -> np.ndarray:
    """Latest up to six distinct observed target months U < O (R37), newest first."""
    months = np.unique(np.asarray(observed_months, dtype=np.int64))
    earlier = months[months < origin_ord]
    return earlier[::-1][:max_dates]
