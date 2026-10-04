"""Population-target QC ledger and exact share-derived phase truth (R11-R13, R20).

Adapted from ``IPCCHGeoRFExperiment/prepare_data.py`` (``_to_decimal``,
``_exact_context``, ``_exact_sum``, ``_classify_row``, ``build_target_ledger``).
The QC steps and their order are unchanged. What changed:

* Truth is the five-level phase ``max({1} | {k : q_k >= .20})`` with the
  inclusive boundary, decided exactly as ``5 * (P_k + ... + P5) >= S`` for each
  k = 2..5 (never from a rounded quotient). The old strict ``> .20`` binary
  label is not produced.
* The ledger also carries the four cumulative regression targets q2..q5
  (normalized ``Q_k / S`` at the declared precision, then float64).
* Invalid truth stays missing; ``overall_phase`` is preserved raw and never
  used as truth or to fill a gap.
"""

from __future__ import annotations

from decimal import Context, Decimal, Inexact, InvalidOperation, localcontext
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd

from ipcch_geoxgb.errors import ContractError

PHASE_COLUMNS = (
    "phase1_percent",
    "phase2_percent",
    "phase3_percent",
    "phase4_percent",
    "phase5_percent",
)
KEY_COLUMNS = ("admin_code", "year", "month")
TARGETS = ("q2", "q3", "q4", "q5")

SUM_LOWER = Decimal("0.90")
SUM_UPPER = Decimal("1.10")
ZERO = Decimal(0)
ONE = Decimal(1)
FIVE = Decimal(5)  # 5 * Q >= S  <=>  Q / S >= 0.20, without division

#: Provenance precision for Q_k / S. Decisions never depend on it.
NORMALIZATION_PRECISION = 100
_NORMALIZATION_CONTEXT = Context(prec=NORMALIZATION_PRECISION)

#: Evaluation mapping (R13): four-class 1, 2, 3, 4/5; binary crisis = phase >= 3.
FOUR_CLASS_LABELS = ("1", "2", "3", "4/5")


def four_class(phase: np.ndarray) -> np.ndarray:
    """Phase 1..5 -> class index 0..3 (phase 4 and 5 merge)."""
    phase = np.asarray(phase, dtype=np.int64)
    if phase.size and (phase.min() < 1 or phase.max() > 5):
        raise ContractError("phase outside 1..5")
    return np.minimum(phase, 4) - 1


def binary_crisis(phase: np.ndarray) -> np.ndarray:
    return (np.asarray(phase, dtype=np.int64) >= 3).astype(np.int64)


def to_decimal(raw: object) -> Decimal | None:
    """Parse a source cell to Decimal at its written precision; None if missing."""
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() in {"na", "nan", "none", "null", "<na>"}:
        return None
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    if value.is_nan():
        return None
    return value


def _exact_context(values: Sequence[Decimal], guard: int = 3) -> Context:
    """A context wide enough that adding/scaling ``values`` cannot round."""
    integer_digits = 1
    fraction_digits = 0
    for value in values:
        _sign, digits, exponent = value.as_tuple()
        if not isinstance(exponent, int):
            raise ContractError(f"non-finite decimal in exact arithmetic: {value}")
        integer_digits = max(integer_digits, len(digits) + exponent)
        fraction_digits = max(fraction_digits, -exponent if exponent < 0 else 0)
    context = Context(prec=integer_digits + fraction_digits + guard)
    context.traps[Inexact] = True
    return context


def exact_sum(values: Sequence[Decimal]) -> Decimal:
    try:
        with localcontext(_exact_context(values)):
            total = ZERO
            for value in values:
                total = total + value
    except Inexact as error:  # pragma: no cover - precision is sized to fit
        raise ContractError(f"inexact decimal sum of {list(values)}") from error
    return total


def reaches_threshold(cumulative: Decimal, total: Decimal) -> bool:
    """Exact ``cumulative / total >= 0.20`` as ``5 * cumulative >= total``."""
    try:
        with localcontext(_exact_context((cumulative, total))):
            scaled = cumulative * FIVE
    except Inexact as error:  # pragma: no cover
        raise ContractError(f"inexact scaling of {cumulative}") from error
    return scaled >= total


class RowVerdict(NamedTuple):
    valid: int
    reason: str
    p5_filled: bool
    total: Decimal | None
    cumulative: tuple[Decimal, ...] | None  # normalized q2..q5
    components: tuple[Decimal, ...] | None  # normalized p1..p5
    phase: int | None


def classify_row(phases: Sequence[Decimal | None], population: Decimal | None) -> RowVerdict:
    """R20 steps in order; a row failing an earlier step is never rescued."""
    p1, p2, p3, p4, p5 = phases
    if any(value is None for value in (p1, p2, p3, p4)):
        return RowVerdict(0, "missing_phase_1_to_4", False, None, None, None, None)
    p5_filled = p5 is None
    if p5_filled:
        p5 = ZERO
    filled = (p1, p2, p3, p4, p5)
    for value in filled:
        if value < ZERO or value > ONE:
            return RowVerdict(0, "phase_share_out_of_bounds", p5_filled, None, None, None, None)
    if population is None:
        return RowVerdict(0, "population_missing", p5_filled, None, None, None, None)
    if population <= ZERO:
        return RowVerdict(0, "population_not_positive", p5_filled, None, None, None, None)
    total = exact_sum(filled)
    if total < SUM_LOWER or total > SUM_UPPER:
        return RowVerdict(0, "sum_out_of_bounds", p5_filled, total, None, None, None)

    phase = 1
    cumulative = []
    for k in range(2, 6):  # q_k = P_k + ... + P5
        q_exact = exact_sum(filled[k - 1:])
        if reaches_threshold(q_exact, total):
            phase = k
        cumulative.append(_NORMALIZATION_CONTEXT.divide(q_exact, total))
    components = tuple(_NORMALIZATION_CONTEXT.divide(value, total) for value in filled)
    return RowVerdict(1, "", p5_filled, total, tuple(cumulative), components, phase)


def build_target_ledger(path: Path | str) -> pd.DataFrame:
    """One row per source area-month, valid or not, sorted by (area, month).

    The caller is responsible for verifying the source identity first
    (``preflight.verify_identities``); this function only reads.
    """
    needed = list(KEY_COLUMNS) + list(PHASE_COLUMNS) + [
        "estimated_population",
        "overall_phase",
    ]
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, usecols=needed)
    for column in KEY_COLUMNS:
        if (raw[column].str.strip() == "").any():
            raise ContractError(f"{column} has blank values; keys must be complete")
    out = pd.DataFrame(
        {
            "admin_code": raw["admin_code"].astype(np.int64),
            "year": raw["year"].astype(np.int64),
            "month": raw["month"].astype(np.int64),
            "overall_phase_raw": raw["overall_phase"],
        }
    )
    if not out["month"].between(1, 12).all():
        raise ContractError("month outside 1..12")
    if out.duplicated(["admin_code", "year", "month"]).any():
        raise ContractError("(admin_code, year, month) is not unique in the source")
    out["month_ord"] = out["year"] * 12 + (out["month"] - 1)

    phase_decimals = [raw[column].map(to_decimal) for column in PHASE_COLUMNS]
    population = raw["estimated_population"].map(to_decimal)
    verdicts = [
        classify_row([p.iat[i] for p in phase_decimals], population.iat[i]) for i in range(len(raw))
    ]
    out["target_valid"] = np.array([v.valid for v in verdicts], dtype=np.int64)
    out["target_invalid_reason"] = [v.reason for v in verdicts]
    out["p5_missing_filled"] = np.array([int(v.p5_filled) for v in verdicts], dtype=np.int64)
    out["phase_sum_S_str"] = ["" if v.total is None else str(v.total) for v in verdicts]
    for index, name in enumerate(TARGETS):
        exact = ["" if v.cumulative is None else str(v.cumulative[index]) for v in verdicts]
        out[f"{name}_str"] = exact
        out[name] = np.array([np.nan if s == "" else float(s) for s in exact], dtype=np.float64)
    for index in range(5):
        exact = ["" if v.components is None else str(v.components[index]) for v in verdicts]
        out[f"p{index + 1}_str"] = exact
        out[f"p{index + 1}"] = np.array([np.nan if s == "" else float(s) for s in exact], dtype=np.float64)
    phase = pd.Series([v.phase for v in verdicts], dtype="Int64")
    out["phase_truth"] = phase
    out["crisis_truth"] = (phase >= 3).astype("Int64")
    for column, series in zip(PHASE_COLUMNS, raw[list(PHASE_COLUMNS)].items()):
        out[f"raw_{column}"] = series[1]
    out["raw_estimated_population"] = raw["estimated_population"]
    return out.sort_values(["admin_code", "month_ord"], kind="mergesort").reset_index(drop=True)


def valid_targets(ledger: pd.DataFrame) -> pd.DataFrame:
    """The supervised/evaluable subset: rows that passed QC, nothing filled."""
    valid = ledger[ledger["target_valid"] == 1].copy()
    if valid[list(TARGETS)].isna().any().any() or valid["phase_truth"].isna().any():
        raise ContractError("a valid ledger row lacks a target or phase")
    valid["phase_truth"] = valid["phase_truth"].astype(np.int64)
    valid["crisis_truth"] = valid["crisis_truth"].astype(np.int64)
    return valid.reset_index(drop=True)


def ledger_summary(ledger: pd.DataFrame) -> dict:
    valid = ledger[ledger["target_valid"] == 1]
    phase = valid["phase_truth"].astype(np.int64)
    exact_boundary = {}
    for name in TARGETS:
        exact_boundary[name] = int((valid[f"{name}_str"].map(Decimal) == Decimal("0.2")).sum())
    return {
        "rows": int(len(ledger)),
        "valid": int(len(valid)),
        "areas_total": int(ledger["admin_code"].nunique()),
        "areas_with_valid": int(valid["admin_code"].nunique()),
        "p5_filled_valid": int(valid["p5_missing_filled"].sum()),
        "invalid_reasons": ledger.loc[ledger["target_valid"] == 0, "target_invalid_reason"]
        .value_counts()
        .to_dict(),
        "phase_counts": {int(k): int(v) for k, v in phase.value_counts().sort_index().items()},
        "crisis": int((phase >= 3).sum()),
        "noncrisis": int((phase < 3).sum()),
        "normalized_q_exactly_0_20": exact_boundary,
    }
