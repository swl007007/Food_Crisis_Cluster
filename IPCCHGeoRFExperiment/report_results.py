"""IPCCH reporting: cohorts, confusion-count metrics and country bootstrap.

This module **never fits a model and never changes a map** (design.md
"Reporting and verification boundary"). It reads saved Stage3 prediction rows
and writes to a specified *fresh* output directory so a third party can
reconstruct every number independently.

Run standalone::

    python3.12.exe -B IPCCHGeoRFExperiment/report_results.py \
        --run-dir <run> --out-dir <fresh-report-check>

or call :func:`generate_report` from the runner.

``--run-dir`` also carries the R1 valid-label ledger that defines persistence
availability, so E_persist is built from the history the data actually has
(Q4/Q4b) rather than from whichever rows the runner filled in.

Requirement anchors are cited inline and match
`.trellis/tasks/09-19-ipcch-binary-georf-pipeline/prd.md` (R5's Q4b, all of R6)
and `research/evaluation.md`, which carry the authoritative metric and
bootstrap contracts.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Frozen reporting constants (R6 / evaluation.md)
# --------------------------------------------------------------------------

ACTIVE_HORIZONS = (1, 3, 6, 12)

#: R4's main target schedule. A target outside its horizon's window is either
#: partial-2026 (a separate period) or a contract error -- never silently
#: folded into the main cohort.
MAIN_TARGET_SCHEDULE = {
    1: ("2023-02", "2025-12"),
    3: ("2023-04", "2025-12"),
    6: ("2023-07", "2025-12"),
    12: ("2024-01", "2025-12"),
}
PARTIAL_PERIOD_YEARS = (2026,)
PERIOD_MAIN = "main"
PERIOD_PARTIAL = "partial_2026"

#: Q7a. p1 > .5 gives 1; exactly .5 gives 0. Used only to *verify* the saved
#: hard labels -- the reporter never re-derives a decision it was not given.
DECISION_THRESHOLD = 0.5

#: Q9b. Fixed; deliberately not exposed on the CLI.
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 42
CI_LOWER_PERCENTILE = 2.5
CI_UPPER_PERCENTILE = 97.5
#: NumPy's standard linear interpolation, recorded in reporting_config.json
#: as design.md "Reporting and verification boundary" requires.
PERCENTILE_METHOD = "linear"
MIN_COUNTRIES_FOR_CI = 2
MIN_DEFINED_REPLICATES_FOR_CI = 2

COHORT_E_ALL = "E_all"
COHORT_E_PERSIST = "E_persist"

#: Q4/Q4b. Persistence is the latest valid same-area R1 label at or before the
#: origin, and E_persist is the *history-available* subset of E_all. Both are
#: properties of the data, so the reporter reconciles what the runner wrote
#: against the valid as-of label history before any cohort exists.
#:
#: ``valid_label_ledger``: the R1 valid-label ledger is available, so
#: availability, source month and value are verified exactly.
#: ``saved_prediction_truth_only``: no ledger was supplied, so the only
#: reconstructible history is the valid truth inside the saved rows. That is a
#: *subset* of the real history, which still refutes a wrongly-missing or
#: wrongly-valued persistence but cannot confirm a legitimately missing one.
HISTORY_MODE_COMPLETE = "valid_label_ledger"
HISTORY_MODE_PARTIAL = "saved_prediction_truth_only"

#: Written by the runner's R1 step; searched under ``--run-dir`` in order.
HISTORY_FILE_CANDIDATES = (
    "data/target_ledger_valid.csv.gz",
    "data/target_ledger_valid.csv",
)

HISTORY_PARTIAL_LIMITATION = (
    "Persistence availability was reconciled only against the valid truth "
    "inside the saved prediction rows (no R1 valid-label ledger was supplied), "
    "so a persistence value whose history predates the test window could not "
    "be confirmed (Q4/Q4b)."
)

#: Month ordinals stay far below this, so ``area * SCALE + ordinal`` is a
#: collision-free sort key for the as-of search.
_AS_OF_KEY_SCALE = 1 << 20

#: R4. Only these three values may appear; anything else is a contract error.
ALLOWED_ASSIGNMENT_SOURCES = ("learned", "nearest_donor", "unresolved")
UNRESOLVED_PARTITION_CODE = -1

UNCERTAINTY_SCOPE_SENTENCE = (
    "Intervals describe country-composition uncertainty conditional on saved "
    "predictions - not future prediction, not partition or training uncertainty."
)

LIMITATIONS = (
    UNCERTAINTY_SCOPE_SENTENCE,
    "No p-values, significance stars, simultaneous coverage or multiplicity "
    "correction are reported; no favourable horizon/cohort/contrast may be "
    "selected as evidence of general superiority (Q9b).",
    "A subset persistence F1 must never be paired against a full-sample "
    "learned F1: the four-arm comparison lives only on E_persist (Q4b).",
    "Raw F1 differences between FEWSNET and IPCCH are not improvement on the "
    "same task (R6).",
    "Observation-month timing and unverified source/geographic provenance "
    "remain limitations of every number in this report (R6).",
    "Success is a valid reproducible comparison, including a null or negative "
    "result; no winning-model condition is imposed (A7).",
)


class ReportContractError(RuntimeError):
    """Raised when saved predictions violate the reporting input contract."""


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One comparison arm and the saved columns that carry its decisions."""

    name: str
    pred_column: str
    prob_column: str | None  # persistence is a history lookup, not a model

    @property
    def is_learned(self) -> bool:
        return self.prob_column is not None


ARM_PARTITIONED = Arm("partitioned_rf", "pred_partitioned_rf", "prob_partitioned_rf")
ARM_POOLED = Arm("pooled_rf", "pred_pooled_rf", "prob_pooled_rf")
ARM_XGB = Arm("xgb", "pred_xgb", "prob_xgb")
ARM_PERSISTENCE = Arm("persistence", "persistence_pred", None)

LEARNED_ARMS = (ARM_PARTITIONED, ARM_POOLED, ARM_XGB)
ALL_ARMS = LEARNED_ARMS + (ARM_PERSISTENCE,)

#: Q9a: paired deltas are partitioned RF minus each eligible baseline.
REFERENCE_ARM = ARM_PARTITIONED

#: Q4b: the three learned arms on E_all; all four on E_persist's identical keys.
COHORT_ARMS = {
    COHORT_E_ALL: LEARNED_ARMS,
    COHORT_E_PERSIST: ALL_ARMS,
}


# --------------------------------------------------------------------------
# Input contract
# --------------------------------------------------------------------------

KEY_COLUMNS = ("admin_code", "target_month", "horizon_months")

REQUIRED_PREDICTION_COLUMNS = (
    "admin_code",
    "country_en",
    "target_month",
    "origin_month",
    "horizon_months",
    "ipcch_food_crisis",
    "prob_partitioned_rf",
    "pred_partitioned_rf",
    "prob_pooled_rf",
    "pred_pooled_rf",
    "prob_xgb",
    "pred_xgb",
    "persistence_pred",
    "persistence_source_month",
    "persistence_age_months",
    "branch_id",
    "partition_code",
    "assignment_source",
    "donor_admin_code",
    "donor_distance_km",
    "model_route",
    "fold_id",
)

#: Read but not required; ``country_id`` is derived from ``country_en`` when
#: absent, because the keyed lookup's ISO3/country_code are known-incomplete
#: (convergence-check.md) while the country name is complete for all 53.
OPTIONAL_PREDICTION_COLUMNS = (
    "country_id",
    "ISO3",
    "period",
    "model_fallback_reason",
)

#: Columns that must keep their exact written text (leading zeros in branch
#: strings are learned lineage, not a number).
_STRING_COLUMNS = (
    "country_id",
    "country_en",
    "ISO3",
    "target_month",
    "origin_month",
    "persistence_source_month",
    "branch_id",
    "assignment_source",
    "model_route",
    "model_fallback_reason",
    "fold_id",
    "period",
)

_MISSING_TOKENS = {"", "na", "nan", "none", "null", "<na>"}

#: Stage3 prediction file names searched under ``--run-dir``, in order.
PREDICTION_FILE_CANDIDATES = (
    "stage3/predictions.csv.gz",
    "stage3/predictions.csv",
    "stage3/stage3_predictions.csv.gz",
    "stage3/stage3_predictions.csv",
    "predictions.csv.gz",
    "predictions.csv",
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def month_ordinal_from_label(labels: Iterable[object], field: str) -> np.ndarray:
    """Parse ``YYYY-MM`` into a dense month index; ``-1`` for a missing token.

    ``year * 12 + (month - 1)`` makes an ordinal difference exactly the number
    of calendar months, which is how every R3 window and age is defined. This
    mirrors ``prepare_data.month_ordinal`` without importing it, so the
    reporter stays runnable against nothing but saved rows.
    """
    out = []
    for raw in labels:
        text = "" if raw is None else str(raw).strip()
        if text.lower() in _MISSING_TOKENS:
            out.append(-1)
            continue
        parts = text.split("-")
        if len(parts) != 2:
            raise ReportContractError(f"{field}: {text!r} is not YYYY-MM")
        try:
            year, month = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ReportContractError(f"{field}: {text!r} is not YYYY-MM") from exc
        if not 1 <= month <= 12:
            raise ReportContractError(f"{field}: month outside 1..12 in {text!r}")
        out.append(year * 12 + (month - 1))
    return np.asarray(out, dtype=np.int64)


def month_label_from_ordinal(ordinal) -> str:
    ordinal = int(ordinal)
    if ordinal < 0:
        return ""
    return f"{ordinal // 12:04d}-{ordinal % 12 + 1:02d}"


def _is_missing(series: pd.Series) -> np.ndarray:
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin(_MISSING_TOKENS).to_numpy()


@dataclass(frozen=True)
class _Numeric:
    """A parsed numeric column with *declared missing* kept apart from *malformed*."""

    values: np.ndarray  # NaN where missing or malformed
    missing: np.ndarray  # declared missing token
    malformed: np.ndarray  # present but not a finite number

    @property
    def usable(self) -> np.ndarray:
        return ~self.missing & ~self.malformed


def _parse_numeric(
    series: pd.Series,
    field: str,
    failures: list[str],
    *,
    allow_missing: bool = True,
    integral: bool = False,
) -> _Numeric:
    """Parse a written-text column, separating declared missing from malformed.

    A declared missing token (``""``, ``NA``, ``NaN``, ``None``, ``null``)
    means "this row carries no value". Anything else that does not parse as a
    *finite* number is a contract error and never a silently missing value:
    ``persistence_pred="BROKEN"`` must not drop a row out of E_persist, and
    ``admin_code="2.7"`` must not become area 2 (finding 2).

    The missing mask is applied positionally rather than through
    ``Series.replace``, whose downcasting behaviour differs between pandas
    2.2 and 3.0. Every problem is appended to ``failures`` so one pass reports
    the whole picture.
    """
    text = series.fillna("").astype(str).str.strip().to_numpy()
    missing = _is_missing(series)
    values = np.full(len(text), np.nan, dtype=np.float64)
    present = ~missing
    if present.any():
        values[present] = pd.to_numeric(
            pd.Series(text[present]), errors="coerce"
        ).to_numpy(dtype=np.float64)
    # NaN (unparseable) or +-inf among the present rows.
    malformed = present & ~np.isfinite(values)
    if malformed.any():
        examples = sorted({str(v) for v in text[malformed]})[:3]
        failures.append(
            f"{int(malformed.sum())} rows have a malformed {field} (not a finite "
            f"number and not a declared missing token {sorted(_MISSING_TOKENS)}): "
            f"{examples}"
        )
    if not allow_missing and missing.any():
        failures.append(f"{int(missing.sum())} rows have no {field}")
    if integral:
        usable = present & ~malformed
        non_integral = usable & (values != np.floor(values))
        if non_integral.any():
            examples = sorted({str(v) for v in text[non_integral]})[:3]
            failures.append(
                f"{int(non_integral.sum())} rows have a non-integral {field}; an "
                f"identifier must be a whole number, never truncated: {examples}"
            )
    values = np.where(malformed, np.nan, values)
    return _Numeric(values=values, missing=missing, malformed=malformed)


def _as_int(values: np.ndarray, fill: int = -1) -> np.ndarray:
    """Safe int64 view of a parsed numeric column (NaN -> ``fill``)."""
    return np.where(np.isfinite(values), values, float(fill)).astype(np.int64)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _concat_frames(frames: Sequence[pd.DataFrame], columns: Sequence[str]) -> pd.DataFrame:
    """Concatenate, skipping empty pieces.

    A suppressed cohort contributes an empty replicate/draw frame; feeding it
    to ``pd.concat`` changes result dtypes across pandas versions, so the
    empties are dropped and an explicit schema is used when nothing is left.
    """
    non_empty = [frame for frame in frames if len(frame)]
    if not non_empty:
        return pd.DataFrame({name: pd.Series(dtype=object) for name in columns})
    return pd.concat(non_empty, ignore_index=True)


def _write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 keeps the gzip container byte-identical across runs, which the
    # determinism check depends on.
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        compression={"method": "gzip", "mtime": 0},
    )


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------


def find_predictions(run_dir: Path | str) -> Path:
    """Locate the saved Stage3 prediction rows inside a run directory."""
    run_dir = Path(run_dir)
    for relative in PREDICTION_FILE_CANDIDATES:
        candidate = run_dir / relative
        if candidate.is_file():
            return candidate
    raise ReportContractError(
        f"no Stage3 prediction file under {run_dir}; looked for "
        f"{list(PREDICTION_FILE_CANDIDATES)}"
    )


def read_prediction_rows(path: Path | str) -> pd.DataFrame:
    """Read saved prediction rows with every field as written text.

    ``dtype=str`` plus ``keep_default_na=False`` is deliberate: a branch string
    such as ``"01"`` is learned lineage and must not become the integer 1, and
    a country code must not be swallowed by pandas' default NA tokens.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    if frame.empty and not list(frame.columns):
        raise ReportContractError(f"{path} has no header")
    return frame


# --------------------------------------------------------------------------
# Q4/Q4b -- the valid as-of label history that defines persistence
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelHistory:
    """Valid R1 labels, sorted by ``(area, month)``, for as-of lookups.

    ``mode`` records whether this is the real ledger or the subset that can be
    rebuilt from the saved rows themselves; the verification strength differs
    and is reported rather than assumed.
    """

    area: np.ndarray
    month_ord: np.ndarray
    label: np.ndarray
    mode: str
    source: str

    @property
    def is_complete(self) -> bool:
        return self.mode == HISTORY_MODE_COMPLETE

    @property
    def key(self) -> np.ndarray:
        return self.area * _AS_OF_KEY_SCALE + self.month_ord

    def as_of(self, area: np.ndarray, month_ord: np.ndarray) -> np.ndarray:
        """Index of the latest same-area entry with month <= ``month_ord``.

        ``side="right"`` makes the bound inclusive, so a label dated exactly at
        O counts (R3: "source/history/training-label month <= O"). Returns -1
        where the area has no qualifying entry. This mirrors
        ``prepare_data._as_of_index`` without importing it, so the reporter
        stays runnable against nothing but saved rows.
        """
        area = np.asarray(area, dtype=np.int64)
        month_ord = np.asarray(month_ord, dtype=np.int64)
        if len(self.area) == 0:
            return np.full(len(area), -1, dtype=np.int64)
        query = area * _AS_OF_KEY_SCALE + np.maximum(month_ord, 0)
        slot = np.searchsorted(self.key, query, side="right") - 1
        safe = np.clip(slot, 0, None)
        hit = (slot >= 0) & (self.area[safe] == area) & (month_ord >= 0)
        return np.where(hit, safe, -1)

    def at(self, area: np.ndarray, month_ord: np.ndarray) -> np.ndarray:
        """Index of the entry at exactly ``(area, month_ord)``; -1 if absent."""
        area = np.asarray(area, dtype=np.int64)
        month_ord = np.asarray(month_ord, dtype=np.int64)
        if len(self.area) == 0:
            return np.full(len(area), -1, dtype=np.int64)
        query = area * _AS_OF_KEY_SCALE + np.maximum(month_ord, 0)
        slot = np.searchsorted(self.key, query, side="left")
        safe = np.clip(slot, 0, len(self.key) - 1)
        hit = (slot < len(self.key)) & (self.key[safe] == query) & (month_ord >= 0)
        return np.where(hit, safe, -1)


def normalize_label_history(
    frame: pd.DataFrame, *, mode: str, source: str
) -> LabelHistory:
    """Build a :class:`LabelHistory` from any valid-label table.

    Accepts either ``(year, month)`` or a ``target_month`` label column, and
    keeps only rows whose R1 target is valid and binary -- invalid truth is
    never history (R1: "Invalid truth stays missing").
    """
    if "admin_code" not in frame.columns:
        raise ReportContractError("label history is missing admin_code")
    if "ipcch_food_crisis" not in frame.columns:
        raise ReportContractError("label history is missing ipcch_food_crisis")

    keep = pd.Series(True, index=frame.index)
    if "target_valid" in frame.columns:
        keep &= pd.to_numeric(frame["target_valid"], errors="coerce") == 1
    label = pd.to_numeric(frame["ipcch_food_crisis"], errors="coerce")
    keep &= label.isin((0.0, 1.0))
    subset = frame[keep.to_numpy()]
    label = label[keep.to_numpy()].to_numpy(dtype=np.int64)

    area = pd.to_numeric(subset["admin_code"], errors="coerce")
    if not np.isfinite(area.to_numpy(dtype=np.float64)).all():
        raise ReportContractError("label history has a non-numeric admin_code")
    area = area.to_numpy(dtype=np.int64)

    if "year" in subset.columns and "month" in subset.columns:
        year = pd.to_numeric(subset["year"], errors="coerce").to_numpy(dtype=np.int64)
        month = pd.to_numeric(subset["month"], errors="coerce").to_numpy(dtype=np.int64)
        if month.size and ((month < 1) | (month > 12)).any():
            raise ReportContractError("label history has a month outside 1..12")
        month_ord = year * 12 + (month - 1)
    elif "target_month" in subset.columns:
        month_ord = month_ordinal_from_label(
            subset["target_month"].astype(str).str.slice(0, 7), "history target_month"
        )
        if (month_ord < 0).any():
            raise ReportContractError("label history has a missing target_month")
    else:
        raise ReportContractError(
            "label history needs either (year, month) or target_month"
        )

    order = np.lexsort((month_ord, area))
    area, month_ord, label = area[order], month_ord[order], label[order]

    key = area * _AS_OF_KEY_SCALE + month_ord
    if key.size > 1:
        collision = key[1:] == key[:-1]
        if collision.any() and (label[1:][collision] != label[:-1][collision]).any():
            raise ReportContractError(
                "label history holds two different labels for one area-month"
            )
    return LabelHistory(area=area, month_ord=month_ord, label=label, mode=mode, source=source)


def load_label_history(path: Path | str) -> LabelHistory:
    """Read a saved R1 valid-label ledger (the runner's target_ledger_valid)."""
    path = Path(path)
    return normalize_label_history(
        pd.read_csv(path, low_memory=False),
        mode=HISTORY_MODE_COMPLETE,
        source=str(path),
    )


def find_label_history(run_dir: Path | str) -> LabelHistory:
    """Locate the R1 valid-label ledger for a run.

    A real run must be able to prove its persistence availability, so this
    raises rather than quietly downgrading to the saved-row subset. The pinned
    source CSV recorded in the run manifest is the documented fallback, read
    through ``prepare_data.build_target_ledger`` so R1 is never reimplemented.
    """
    run_dir = Path(run_dir)
    for relative in HISTORY_FILE_CANDIDATES:
        candidate = run_dir / relative
        if candidate.is_file():
            return load_label_history(candidate)

    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError as exc:  # pragma: no cover - unreadable manifest
            raise ReportContractError(f"{manifest_path} is not valid JSON") from exc
        source_csv = str(manifest.get("source", {}).get("csv", ""))
        if source_csv and Path(source_csv).is_file():
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from prepare_data import build_target_ledger  # noqa: PLC0415 - lazy, read-only

            ledger = build_target_ledger(source_csv, verify_hash=False)
            return normalize_label_history(
                ledger.valid(),
                mode=HISTORY_MODE_COMPLETE,
                source=f"prepare_data.build_target_ledger({source_csv})",
            )

    raise ReportContractError(
        f"no R1 valid-label ledger under {run_dir}; looked for "
        f"{list(HISTORY_FILE_CANDIDATES)} and the manifest's pinned source CSV. "
        "E_persist is defined by the available history (Q4b), so it cannot be "
        "built from whatever the runner happened to write."
    )


def history_from_predictions(
    area: np.ndarray, month_ord: np.ndarray, truth: np.ndarray
) -> LabelHistory:
    """The valid-label subset that the saved rows themselves already prove.

    One area-month appears once per horizon with the same label, so duplicate
    keys are collapsed here; a genuine disagreement is reported separately by
    :func:`validate_predictions` rather than raised from inside this helper.
    """
    frame = pd.DataFrame(
        {
            "admin_code": np.asarray(area, dtype=np.int64),
            "target_month": [month_label_from_ordinal(v) for v in month_ord],
            "ipcch_food_crisis": np.asarray(truth, dtype=np.int64),
        }
    ).drop_duplicates(subset=["admin_code", "target_month"], keep="first")
    return normalize_label_history(
        frame,
        mode=HISTORY_MODE_PARTIAL,
        source="valid truth inside the saved prediction rows",
    )


def _resolve_country_id(frame: pd.DataFrame, failures: list[str]) -> tuple[pd.Series, str]:
    """Return the stable bootstrap cluster key and where it came from.

    design.md: "Use keyed country names/IDs from the complete lookup even when
    ISO3 is missing; no country row can disappear before bootstrap." ISO3 and
    country_code are both known-incomplete in the pinned lookup, so the country
    *name* is the fallback -- and completeness is enforced either way.
    """
    if "country_id" in frame.columns and not _is_missing(frame["country_id"]).all():
        source = "country_id"
        series = frame["country_id"]
    else:
        source = "country_en"
        series = frame["country_en"]
    missing = int(_is_missing(series).sum())
    if missing:
        failures.append(
            f"{missing} rows have no {source}; a missing country mapping is a "
            "failure, never a dropped row (Q9b)"
        )
    return series.astype(str).str.strip(), source


def validate_predictions(
    frame: pd.DataFrame, history: LabelHistory | None = None
) -> tuple[pd.DataFrame, dict]:
    """Validate keys, binary fields, country mapping and prob/pred consistency.

    Every check runs before scoring (design.md), and *all* failures are
    collected so one pass reports the whole problem rather than the first row.
    ``history`` is the R1 valid-label ledger; when it is omitted the subset
    provable from the saved rows themselves is used instead, and the weaker
    verification is recorded rather than assumed away (Q4/Q4b).
    Returns ``(prepared, report)``; raises :class:`ReportContractError` on any
    failure.
    """
    failures: list[str] = []

    missing_columns = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in frame.columns]
    if missing_columns:
        raise ReportContractError(
            f"saved predictions are missing required columns: {missing_columns}"
        )

    n = len(frame)
    if n == 0:
        raise ReportContractError("saved predictions are empty; nothing to report")

    out = pd.DataFrame(index=pd.RangeIndex(n))

    # --- keys ------------------------------------------------------------
    # Identifiers must be finite whole numbers: "2.7" is a malformed area, not
    # area 2 (finding 2).
    admin = _parse_numeric(
        frame["admin_code"], "admin_code", failures, allow_missing=False, integral=True
    )
    out["admin_code"] = _as_int(admin.values)

    horizon_col = _parse_numeric(
        frame["horizon_months"],
        "horizon_months",
        failures,
        allow_missing=False,
        integral=True,
    )
    horizon = horizon_col.values
    bad_horizon = horizon_col.usable & ~np.isin(horizon, ACTIVE_HORIZONS)
    if bad_horizon.any():
        failures.append(
            f"{int(bad_horizon.sum())} rows have a horizon outside {ACTIVE_HORIZONS}; "
            "IPCCH uses actual calendar months, never a legacy scope index (R3)"
        )
    horizon = np.where(
        horizon_col.usable & ~bad_horizon, horizon, float(ACTIVE_HORIZONS[0])
    )
    out["horizon_months"] = horizon.astype(np.int64)

    target_ord = month_ordinal_from_label(frame["target_month"], "target_month")
    origin_ord = month_ordinal_from_label(frame["origin_month"], "origin_month")
    if (target_ord < 0).any():
        failures.append("target_month is missing on some rows")
    if (origin_ord < 0).any():
        failures.append("origin_month is missing on some rows")
    out["target_month"] = [month_label_from_ordinal(v) for v in target_ord]
    out["origin_month"] = [month_label_from_ordinal(v) for v in origin_ord]
    out["target_month_ord"] = target_ord
    out["origin_month_ord"] = origin_ord
    out["target_year"] = np.where(target_ord >= 0, target_ord // 12, -1)

    # R3: O = T - H, month-end inclusive. A saved row that disagrees is a
    # timing failure, not a rounding difference.
    bad_origin = (target_ord >= 0) & (origin_ord >= 0) & (
        (target_ord - origin_ord) != out["horizon_months"].to_numpy()
    )
    if bad_origin.any():
        failures.append(
            f"{int(bad_origin.sum())} rows violate O = T - H (R3 horizon contract)"
        )

    duplicated = out.duplicated(list(KEY_COLUMNS))
    if duplicated.any():
        failures.append(
            f"{int(duplicated.sum())} duplicate (admin_code, target_month, "
            "horizon_months) keys; the cohort key must be unique"
        )

    # --- country mapping --------------------------------------------------
    country_id, country_id_source = _resolve_country_id(frame, failures)
    out["country_id"] = country_id
    out["country_en"] = frame["country_en"].astype(str).str.strip()
    out["ISO3"] = (
        frame["ISO3"].astype(str).str.strip() if "ISO3" in frame.columns else ""
    )
    per_area = out.groupby("admin_code")["country_id"].nunique()
    inconsistent = per_area[per_area > 1]
    if len(inconsistent):
        failures.append(
            f"{len(inconsistent)} areas map to more than one country_id; the "
            "bootstrap cluster key must be unambiguous"
        )

    # --- truth ------------------------------------------------------------
    truth_col = _parse_numeric(
        frame["ipcch_food_crisis"], "ipcch_food_crisis", failures, allow_missing=False
    )
    truth = truth_col.values
    bad_truth = ~truth_col.usable | ~np.isin(truth, (0.0, 1.0))
    non_binary_truth = truth_col.usable & ~np.isin(truth, (0.0, 1.0))
    if non_binary_truth.any():
        failures.append(
            f"{int(non_binary_truth.sum())} rows have non-binary truth; "
            "invalid truth is never imputed (R1)"
        )
    truth = np.where(bad_truth, 0.0, truth)
    out["ipcch_food_crisis"] = truth.astype(np.int64)
    truth_usable = ~bad_truth

    # One area-month carries one R1 label, whatever horizon reads it; the saved
    # rows must agree before they can stand in for the label history.
    if truth_usable.any():
        agreement = (
            pd.DataFrame(
                {
                    "admin_code": out["admin_code"].to_numpy()[truth_usable],
                    "month_ord": target_ord[truth_usable],
                    "truth": truth[truth_usable],
                }
            )
            .groupby(["admin_code", "month_ord"])["truth"]
            .nunique()
        )
        conflicting = int((agreement > 1).sum())
        if conflicting:
            failures.append(
                f"{conflicting} area-months carry two different truth values across "
                "horizons; one area-month has exactly one R1 label"
            )

    # --- learned arms: presence, range and p1 > .5 consistency ------------
    for arm in LEARNED_ARMS:
        prob_col = _parse_numeric(
            frame[arm.prob_column], arm.prob_column, failures, allow_missing=False
        )
        pred_col = _parse_numeric(
            frame[arm.pred_column], arm.pred_column, failures, allow_missing=True
        )
        prob, pred = prob_col.values, pred_col.values

        missing_pred = ~pred_col.usable
        if pred_col.missing.any():
            failures.append(
                f"{int(pred_col.missing.sum())} rows have no {arm.pred_column}; a "
                "missing learned prediction is a failure, never a silently "
                "shrunk cohort (Q4b)"
            )
        bad_pred = pred_col.usable & ~np.isin(pred, (0.0, 1.0))
        if bad_pred.any():
            failures.append(f"{int(bad_pred.sum())} rows have non-binary {arm.pred_column}")

        missing_prob = ~prob_col.usable
        out_of_range = prob_col.usable & ((prob < 0.0) | (prob > 1.0))
        if out_of_range.any():
            failures.append(
                f"{int(out_of_range.sum())} rows have {arm.prob_column} outside [0, 1]"
            )

        # Q7a: p1 > .5 gives 1, exactly .5 gives 0. Verified, never re-derived.
        checkable = ~missing_prob & ~missing_pred & ~bad_pred & ~out_of_range
        expected = (prob > DECISION_THRESHOLD).astype(np.float64)
        mismatch = checkable & (expected != pred)
        if mismatch.any():
            failures.append(
                f"{int(mismatch.sum())} rows where {arm.pred_column} does not equal "
                f"{arm.prob_column} > {DECISION_THRESHOLD} (Q7a: ties give 0)"
            )

        out[arm.prob_column] = prob
        out[arm.pred_column] = np.where(missing_pred | bad_pred, -1, pred).astype(np.int64)

    # --- persistence: reconciled against the valid as-of history ----------
    # Q4/Q4b. What the runner wrote is *checked*, not believed: availability,
    # source month and value all come from the data (finding 1).
    persistence_col = _parse_numeric(
        frame["persistence_pred"], "persistence_pred", failures, allow_missing=True
    )
    persistence = persistence_col.values
    supplied_available = persistence_col.usable
    bad_persistence = supplied_available & ~np.isin(persistence, (0.0, 1.0))
    if bad_persistence.any():
        failures.append(
            f"{int(bad_persistence.sum())} rows have a non-binary persistence_pred"
        )
        supplied_available = supplied_available & ~bad_persistence

    persistence_source_ord = month_ordinal_from_label(
        frame["persistence_source_month"], "persistence_source_month"
    )
    out["persistence_source_month"] = [
        month_label_from_ordinal(v) for v in persistence_source_ord
    ]
    missing_source = supplied_available & (persistence_source_ord < 0)
    if missing_source.any():
        failures.append(
            f"{int(missing_source.sum())} rows carry a persistence label with no "
            "source month; Q4 requires the exported source month and age"
        )
    stray_source = ~supplied_available & (persistence_source_ord >= 0)
    if stray_source.any():
        failures.append(
            f"{int(stray_source.sum())} rows have a persistence source month but no "
            "persistence label"
        )

    if history is None:
        history = history_from_predictions(
            out["admin_code"].to_numpy()[truth_usable & (target_ord >= 0)],
            target_ord[truth_usable & (target_ord >= 0)],
            truth[truth_usable & (target_ord >= 0)].astype(np.int64),
        )
    verified_available, history_report = _verify_persistence(
        area=out["admin_code"].to_numpy(),
        origin_ord=origin_ord,
        target_ord=target_ord,
        truth=out["ipcch_food_crisis"].to_numpy(),
        truth_usable=truth_usable,
        supplied_available=supplied_available,
        supplied_value=persistence,
        supplied_source_ord=persistence_source_ord,
        history=history,
        failures=failures,
    )
    out["persistence_available"] = verified_available.astype(np.int64)
    out[ARM_PERSISTENCE.pred_column] = np.where(
        verified_available, np.nan_to_num(persistence, nan=-1.0), -1
    ).astype(np.int64)

    age_col = _parse_numeric(
        frame["persistence_age_months"], "persistence_age_months", failures
    )
    age = age_col.values
    out["persistence_age_months"] = age
    available = supplied_available
    with_age = available & (persistence_source_ord >= 0) & (origin_ord >= 0)
    expected_age = np.where(with_age, origin_ord - persistence_source_ord, np.nan)
    age_mismatch = with_age & (np.isnan(age) | (age != expected_age))
    if age_mismatch.any():
        failures.append(
            f"{int(age_mismatch.sum())} rows where persistence_age_months does not "
            "equal O - source month (Q4)"
        )
    stale_history = with_age & (expected_age < 0)
    if stale_history.any():
        failures.append(
            f"{int(stale_history.sum())} rows use a persistence label dated after "
            "their own origin (R3 own-origin cutoff)"
        )

    # --- provenance --------------------------------------------------------
    assignment = frame["assignment_source"].astype(str).str.strip()
    unknown = ~assignment.isin(ALLOWED_ASSIGNMENT_SOURCES)
    if unknown.any():
        failures.append(
            f"{int(unknown.sum())} rows have an assignment_source outside "
            f"{list(ALLOWED_ASSIGNMENT_SOURCES)}: "
            f"{sorted(set(assignment[unknown]))[:5]}"
        )
    out["assignment_source"] = assignment

    partition_col = _parse_numeric(
        frame["partition_code"],
        "partition_code",
        failures,
        allow_missing=False,
        integral=True,
    )
    out["partition_code"] = _as_int(
        partition_col.values, fill=UNRESOLVED_PARTITION_CODE
    )
    is_unresolved = (assignment == "unresolved").to_numpy()
    code_disagrees = is_unresolved != (out["partition_code"].to_numpy() == UNRESOLVED_PARTITION_CODE)
    if code_disagrees.any():
        failures.append(
            f"{int(code_disagrees.sum())} rows where partition_code == "
            f"{UNRESOLVED_PARTITION_CODE} disagrees with assignment_source "
            "'unresolved'"
        )

    out["branch_id"] = frame["branch_id"].astype(str)  # leading zeros preserved
    # Kept as text (a donor ID is provenance, not a number) but still parsed so
    # a malformed or fractional donor code cannot pass as provenance.
    _parse_numeric(
        frame["donor_admin_code"], "donor_admin_code", failures, integral=True
    )
    out["donor_admin_code"] = frame["donor_admin_code"].astype(str).str.strip()
    out["donor_distance_km"] = _parse_numeric(
        frame["donor_distance_km"], "donor_distance_km", failures
    ).values
    out["model_route"] = frame["model_route"].astype(str).str.strip()
    if (out["model_route"] == "").any():
        failures.append("model_route is blank on some rows")
    out["model_fallback_reason"] = (
        frame["model_fallback_reason"].astype(str).str.strip()
        if "model_fallback_reason" in frame.columns
        else ""
    )
    out["fold_id"] = frame["fold_id"].astype(str).str.strip()
    if (out["fold_id"] == "").any():
        failures.append("fold_id is blank on some rows")

    # --- period (R4 schedule) ---------------------------------------------
    period, period_failures = _classify_period(
        out["target_month_ord"].to_numpy(), out["horizon_months"].to_numpy()
    )
    failures.extend(period_failures)
    out["period"] = period
    if "period" in frame.columns:
        stated = frame["period"].astype(str).str.strip()
        stated_known = stated != ""
        disagree = stated_known.to_numpy() & (stated.to_numpy() != period)
        if disagree.any():
            failures.append(
                f"{int(disagree.sum())} rows where the saved period column "
                "disagrees with the R4 schedule"
            )

    if failures:
        raise ReportContractError(
            "saved predictions violate the reporting input contract:\n  - "
            + "\n  - ".join(failures)
        )

    report = {
        "rows": int(n),
        "areas": int(out["admin_code"].nunique()),
        "countries": int(out["country_id"].nunique()),
        "country_id_source": country_id_source,
        "horizons": sorted(int(h) for h in out["horizon_months"].unique()),
        "target_month_min": min(m for m in out["target_month"] if m),
        "target_month_max": max(m for m in out["target_month"] if m),
        "rows_by_period": {
            str(k): int(v) for k, v in out["period"].value_counts().items()
        },
        "persistence_available": int(out["persistence_available"].sum()),
        "history_verification": history_report,
        "checks_passed": [
            "required_columns",
            "unique_keys",
            "numeric_fields_well_formed_not_silently_missing",
            "identifiers_finite_and_integral",
            "horizon_in_1_3_6_12",
            "origin_equals_target_minus_horizon",
            "binary_truth_present",
            "one_label_per_area_month",
            "learned_predictions_present_and_binary",
            "probabilities_in_unit_interval",
            "hard_label_equals_p1_gt_half",
            "persistence_binary_or_missing",
            "persistence_source_month_and_age_consistent",
            "persistence_reconciled_with_valid_as_of_history",
            "country_mapping_complete_and_unambiguous",
            "assignment_source_and_partition_code_agree",
            "period_matches_r4_schedule",
        ],
    }
    return out, report


def _take(values: np.ndarray, index: np.ndarray, fill: int = -1) -> np.ndarray:
    """``values[index]`` with -1 meaning "no entry"; safe on an empty history."""
    index = np.asarray(index, dtype=np.int64)
    if len(values) == 0:
        return np.full(len(index), fill, dtype=np.int64)
    return np.where(index >= 0, values[np.clip(index, 0, None)], fill)


def _verify_persistence(
    *,
    area: np.ndarray,
    origin_ord: np.ndarray,
    target_ord: np.ndarray,
    truth: np.ndarray,
    truth_usable: np.ndarray,
    supplied_available: np.ndarray,
    supplied_value: np.ndarray,
    supplied_source_ord: np.ndarray,
    history: LabelHistory,
    failures: list[str],
) -> tuple[np.ndarray, dict]:
    """Reconcile the saved persistence column against the as-of label history.

    Q4 defines persistence as the latest valid same-area R1 label with source
    month <= O, and Q4b defines E_persist as the *history-available* subset of
    E_all. Both are properties of the data, so a row whose origin does have
    available history cannot be pushed out of E_persist by leaving the column
    blank, and a value that contradicts the label at its own stated source
    month is a failure rather than a silently scored prediction (finding 1).

    Returns ``(verified_available, report)``.
    """
    slot = history.as_of(area, origin_ord)
    found = slot >= 0
    history_source_ord = _take(history.month_ord, slot)
    history_label = _take(history.label, slot)

    report: dict = {
        "mode": history.mode,
        "source": history.source,
        "history_rows": int(len(history.area)),
        "history_areas": int(len(np.unique(history.area))) if len(history.area) else 0,
    }

    if history.is_complete:
        # Exact verification: availability, source month and value.
        expected_available = found
        missing_but_available = expected_available & ~supplied_available
        if missing_but_available.any():
            examples = _persistence_examples(
                area, target_ord, missing_but_available, history_source_ord
            )
            failures.append(
                f"{int(missing_but_available.sum())} rows have no persistence_pred "
                "although a valid same-area label exists at or before their own "
                f"origin; E_persist is the history-available subset (Q4b): {examples}"
            )
        present_but_unavailable = ~expected_available & supplied_available
        if present_but_unavailable.any():
            examples = _persistence_examples(
                area, target_ord, present_but_unavailable, history_source_ord
            )
            failures.append(
                f"{int(present_but_unavailable.sum())} rows carry a persistence_pred "
                "although no valid same-area label exists at or before their own "
                f"origin (Q4): {examples}"
            )
        both = expected_available & supplied_available
        wrong_source = both & (supplied_source_ord >= 0) & (
            supplied_source_ord != history_source_ord
        )
        if wrong_source.any():
            examples = _persistence_examples(
                area, target_ord, wrong_source, history_source_ord
            )
            failures.append(
                f"{int(wrong_source.sum())} rows name a persistence_source_month that "
                "is not the latest valid same-area label at or before O (Q4): "
                f"{examples}"
            )
        wrong_value = both & (supplied_value != history_label.astype(np.float64))
        if wrong_value.any():
            examples = _persistence_examples(
                area, target_ord, wrong_value, history_source_ord
            )
            failures.append(
                f"{int(wrong_value.sum())} rows have a persistence_pred that "
                f"contradicts the valid label it claims to carry (Q4): {examples}"
            )

        # The ledger and the saved truth must describe the same R1 labels.
        at_target = history.at(area, target_ord)
        checkable = truth_usable & (target_ord >= 0)
        absent = checkable & (at_target < 0)
        if absent.any():
            failures.append(
                f"{int(absent.sum())} scored area-months have no valid label in the "
                "R1 ledger; scored truth must be valid observed truth (R1)"
            )
        matched = checkable & (at_target >= 0)
        disagree = matched & (_take(history.label, at_target) != truth.astype(np.int64))
        if disagree.any():
            failures.append(
                f"{int(disagree.sum())} scored area-months disagree with the R1 "
                "ledger's label for the same area-month"
            )
        report["verified"] = "availability_source_month_and_value"
        verified_available = expected_available
    else:
        # The saved truth is a *subset* of the history, so it can only refute.
        missing_but_available = found & ~supplied_available
        if missing_but_available.any():
            examples = _persistence_examples(
                area, target_ord, missing_but_available, history_source_ord
            )
            failures.append(
                f"{int(missing_but_available.sum())} rows have no persistence_pred "
                "although the saved rows themselves hold a valid same-area label at "
                f"or before their own origin (Q4b): {examples}"
            )
        too_old = (
            supplied_available
            & found
            & (supplied_source_ord >= 0)
            & (supplied_source_ord < history_source_ord)
        )
        if too_old.any():
            examples = _persistence_examples(
                area, target_ord, too_old, history_source_ord
            )
            failures.append(
                f"{int(too_old.sum())} rows name a persistence_source_month earlier "
                "than a valid same-area label that the saved rows already prove is "
                f"available at or before O (Q4): {examples}"
            )
        at_source = history.at(area, np.maximum(supplied_source_ord, -1))
        known_source = supplied_available & (at_source >= 0)
        wrong_value = known_source & (
            supplied_value != _take(history.label, at_source).astype(np.float64)
        )
        if wrong_value.any():
            examples = _persistence_examples(
                area, target_ord, wrong_value, supplied_source_ord
            )
            failures.append(
                f"{int(wrong_value.sum())} rows have a persistence_pred that "
                "contradicts the saved truth at their own stated source month (Q4): "
                f"{examples}"
            )
        report["verified"] = (
            "refutation_only: wrongly missing history, a stale source month and a "
            "value contradicting the saved truth are all caught; a legitimately "
            "missing history cannot be confirmed"
        )
        report["limitation"] = HISTORY_PARTIAL_LIMITATION
        verified_available = supplied_available

    report["rows_with_history"] = int(verified_available.sum())
    report["rows_without_history"] = int((~verified_available).sum())
    return verified_available, report


def _persistence_examples(
    area: np.ndarray,
    target_ord: np.ndarray,
    mask: np.ndarray,
    source_ord: np.ndarray,
    limit: int = 3,
) -> list[str]:
    """``area@target(history=source)`` strings for the first few offending rows."""
    positions = np.flatnonzero(mask)[:limit]
    return [
        f"{int(area[i])}@{month_label_from_ordinal(target_ord[i])}"
        f"(history={month_label_from_ordinal(source_ord[i]) or 'none'})"
        for i in positions
    ]


def _classify_period(
    target_ord: np.ndarray, horizon: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Label each row ``main`` or ``partial_2026`` per R4's target schedule."""
    period = np.full(len(target_ord), "", dtype=object)
    failures: list[str] = []
    out_of_schedule = 0
    for h in ACTIVE_HORIZONS:
        lo = month_ordinal_from_label([MAIN_TARGET_SCHEDULE[h][0]], "schedule")[0]
        hi = month_ordinal_from_label([MAIN_TARGET_SCHEDULE[h][1]], "schedule")[0]
        rows = horizon == h
        if not rows.any():
            continue
        t = target_ord[rows]
        labels = np.full(len(t), "", dtype=object)
        labels[(t >= lo) & (t <= hi)] = PERIOD_MAIN
        labels[(t > hi) & np.isin(t // 12, PARTIAL_PERIOD_YEARS)] = PERIOD_PARTIAL
        out_of_schedule += int((labels == "").sum())
        period[rows] = labels
    if out_of_schedule:
        failures.append(
            f"{out_of_schedule} rows fall outside the approved R4 target schedule "
            f"{MAIN_TARGET_SCHEDULE} and are not partial-{PARTIAL_PERIOD_YEARS[0]}"
        )
    return period, failures


# --------------------------------------------------------------------------
# Q9a metrics -- aggregate counts first, then the ratio
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricResult:
    """Class-1 metrics from aggregated confusion counts, with reasons."""

    tp: int
    fp: int
    fn: int
    tn: int
    n: int
    f1: float
    precision: float
    recall: float
    f1_reason: str
    precision_reason: str
    recall_reason: str

    def as_row(self) -> dict:
        return {
            "n_observations": self.n,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "f1_reason": self.f1_reason,
            "precision_reason": self.precision_reason,
            "recall_reason": self.recall_reason,
        }


def confusion_counts(truth: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    """Observation-level TP/FP/FN/TN with unit weight per row (Q9a)."""
    truth = np.asarray(truth, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    if truth.shape != pred.shape:
        raise ReportContractError("truth and prediction lengths differ")
    if truth.size and (not np.isin(truth, (0, 1)).all() or not np.isin(pred, (0, 1)).all()):
        raise ReportContractError("confusion counts require binary truth and predictions")
    t1 = truth == 1
    p1 = pred == 1
    return (
        int(np.count_nonzero(t1 & p1)),
        int(np.count_nonzero(~t1 & p1)),
        int(np.count_nonzero(t1 & ~p1)),
        int(np.count_nonzero(~t1 & ~p1)),
    )


def class1_metrics(tp: int, fp: int, fn: int, tn: int) -> MetricResult:
    """F1 = 2TP/(2TP+FP+FN), precision = TP/(TP+FP), recall = TP/(TP+FN).

    Zero denominator gives NaN plus a reason; zero numerator with a positive
    denominator gives 0 (Q9a). No score is ever borrowed from class 0 -- that
    is exactly the legacy ``nan_option='mean'`` behaviour evaluation.md warns
    about, where an all-negative sample reports class-1 F1 = 1.
    """
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    n = tp + fp + fn + tn
    empty = n == 0

    f1_den = 2 * tp + fp + fn
    if f1_den == 0:
        f1, f1_reason = (
            float("nan"),
            "empty_cohort" if empty else "no_positive_truth_and_no_positive_prediction",
        )
    else:
        f1, f1_reason = 2.0 * tp / f1_den, ""

    p_den = tp + fp
    if p_den == 0:
        precision, precision_reason = (
            float("nan"),
            "empty_cohort" if empty else "no_predicted_positives",
        )
    else:
        precision, precision_reason = float(tp) / p_den, ""

    r_den = tp + fn
    if r_den == 0:
        recall, recall_reason = (
            float("nan"),
            "empty_cohort" if empty else "no_observed_positives",
        )
    else:
        recall, recall_reason = float(tp) / r_den, ""

    return MetricResult(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        n=n,
        f1=f1,
        precision=precision,
        recall=recall,
        f1_reason=f1_reason,
        precision_reason=precision_reason,
        recall_reason=recall_reason,
    )


def evaluate_arm(frame: pd.DataFrame, arm: Arm) -> MetricResult:
    truth = frame["ipcch_food_crisis"].to_numpy(dtype=np.int64)
    pred = frame[arm.pred_column].to_numpy(dtype=np.int64)
    if pred.size and (pred == -1).any():
        raise ReportContractError(
            f"{arm.name} has a missing prediction inside a scored cohort; a "
            "missing prediction is a failure, never a per-arm complete-case mask"
        )
    return class1_metrics(*confusion_counts(truth, pred))


# --------------------------------------------------------------------------
# Q4b cohorts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Cohort:
    """One (period, horizon, cohort) slice and the arms eligible on it."""

    period: str
    horizon: int
    name: str
    frame: pd.DataFrame

    @property
    def arms(self) -> tuple[Arm, ...]:
        return COHORT_ARMS[self.name]

    @property
    def label(self) -> str:
        return f"{self.period}/h{self.horizon}/{self.name}"


def build_cohorts(prepared: pd.DataFrame) -> list[Cohort]:
    """E_all and its history-available subset E_persist, per period/horizon.

    E_persist is a *mask on the same rows* (Q4b): predictions are reused, never
    recomputed, and training is never filtered for the subset.

    Every approved horizon of the main period gets a cohort even when it holds
    no rows at all. Q9a requires empty cohorts to stay explicit, so an entire
    horizon with no saved predictions must still appear -- with zero support
    and undefined metrics -- rather than vanishing from the report (finding 3).
    The partial-2026 period is a separate, opportunistic period and is emitted
    only when the run actually reaches it.
    """
    cohorts: list[Cohort] = []
    empty = prepared.iloc[0:0]
    for period in (PERIOD_MAIN, PERIOD_PARTIAL):
        in_period = (prepared["period"] == period).to_numpy()
        if period != PERIOD_MAIN and not in_period.any():
            continue
        for horizon in ACTIVE_HORIZONS:
            rows = prepared[in_period & (prepared["horizon_months"] == horizon).to_numpy()]
            if rows.empty:
                cohorts.append(Cohort(period, horizon, COHORT_E_ALL, empty))
                cohorts.append(Cohort(period, horizon, COHORT_E_PERSIST, empty))
                continue
            e_all = rows.reset_index(drop=True)
            e_persist = rows[rows["persistence_available"] == 1].reset_index(drop=True)
            cohorts.append(Cohort(period, horizon, COHORT_E_ALL, e_all))
            cohorts.append(Cohort(period, horizon, COHORT_E_PERSIST, e_persist))
    return cohorts


def cohort_keys_table(prepared: pd.DataFrame) -> pd.DataFrame:
    """Row-level cohort membership, sufficient to rebuild both cohorts."""
    return pd.DataFrame(
        {
            "period": prepared["period"],
            "horizon_months": prepared["horizon_months"],
            "admin_code": prepared["admin_code"],
            "target_month": prepared["target_month"],
            "origin_month": prepared["origin_month"],
            "country_id": prepared["country_id"],
            "ipcch_food_crisis": prepared["ipcch_food_crisis"],
            "in_e_all": 1,
            "in_e_persist": prepared["persistence_available"],
        }
    ).sort_values(
        ["period", "horizon_months", "admin_code", "target_month"], kind="mergesort"
    ).reset_index(drop=True)


def _support(frame: pd.DataFrame) -> dict:
    n = len(frame)
    positives = int((frame["ipcch_food_crisis"] == 1).sum()) if n else 0
    return {
        "n_observations": n,
        "n_areas": int(frame["admin_code"].nunique()) if n else 0,
        "n_countries": int(frame["country_id"].nunique()) if n else 0,
        "n_positive_truth": positives,
        "prevalence": (positives / n) if n else float("nan"),
    }


def scheduled_target_months(horizon: int) -> list[str]:
    """R4's main target months for one horizon, inclusive at both ends."""
    lo = month_ordinal_from_label([MAIN_TARGET_SCHEDULE[horizon][0]], "schedule")[0]
    hi = month_ordinal_from_label([MAIN_TARGET_SCHEDULE[horizon][1]], "schedule")[0]
    return [month_label_from_ordinal(v) for v in range(int(lo), int(hi) + 1)]


def cohort_support_table(cohorts: Sequence[Cohort], prepared: pd.DataFrame) -> pd.DataFrame:
    """Support, prevalence, persistence coverage and fallback counts (Q9a).

    The main-period rows also reconcile against R4's target schedule, so an
    empty or thin cohort can be read as "no valid truth that month" versus
    "predictions are missing" instead of silently disappearing (finding 3).
    """
    routes = sorted(prepared["model_route"].unique())
    assignments = list(ALLOWED_ASSIGNMENT_SOURCES)
    rows = []
    for cohort in cohorts:
        frame = cohort.frame
        e_all_n = int(
            (
                (prepared["period"] == cohort.period)
                & (prepared["horizon_months"] == cohort.horizon)
            ).sum()
        )
        available = int(frame["persistence_available"].sum()) if len(frame) else 0
        months_present = (
            set(frame["target_month"].unique()) if len(frame) else set()
        )
        if cohort.period == PERIOD_MAIN:
            scheduled = scheduled_target_months(cohort.horizon)
            n_scheduled = float(len(scheduled))
            n_scheduled_present = float(len(months_present & set(scheduled)))
            n_scheduled_absent = n_scheduled - n_scheduled_present
        else:
            # Partial-2026 has no approved schedule to reconcile against.
            n_scheduled = n_scheduled_present = n_scheduled_absent = float("nan")
        row = {
            "period": cohort.period,
            "horizon_months": cohort.horizon,
            "cohort": cohort.name,
            **_support(frame),
            "n_target_months_present": len(months_present),
            "n_scheduled_target_months": n_scheduled,
            "n_scheduled_target_months_present": n_scheduled_present,
            "n_scheduled_target_months_absent": n_scheduled_absent,
            "n_e_all_reference": e_all_n,
            "n_persistence_available": available,
            # Coverage is always reported against E_all, so the E_persist row
            # shows the share of the full schedule it actually covers.
            "persistence_coverage": (available / e_all_n) if e_all_n else float("nan"),
            "n_persistence_missing": (
                int((frame["persistence_available"] == 0).sum()) if len(frame) else 0
            ),
        }
        for value in assignments:
            row[f"assignment_{value}"] = (
                int((frame["assignment_source"] == value).sum()) if len(frame) else 0
            )
        for value in routes:
            row[f"route_{value}"] = (
                int((frame["model_route"] == value).sum()) if len(frame) else 0
            )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Metric and delta tables
# --------------------------------------------------------------------------


def metrics_table(cohorts: Sequence[Cohort]) -> pd.DataFrame:
    rows = []
    for cohort in cohorts:
        support = _support(cohort.frame)
        for arm in cohort.arms:
            result = evaluate_arm(cohort.frame, arm)
            predicted_positives = (
                int((cohort.frame[arm.pred_column] == 1).sum()) if len(cohort.frame) else 0
            )
            rows.append(
                {
                    "period": cohort.period,
                    "horizon_months": cohort.horizon,
                    "cohort": cohort.name,
                    "arm": arm.name,
                    **support,
                    "predicted_positives": predicted_positives,
                    **{k: v for k, v in result.as_row().items() if k != "n_observations"},
                }
            )
    return pd.DataFrame(rows)


def deltas_table(cohorts: Sequence[Cohort]) -> pd.DataFrame:
    """Partitioned RF minus each eligible baseline, within one horizon/cohort.

    Q9a: if either metric is undefined its delta stays undefined -- and a
    full-sample learned score is never subtracted from a persistence subsample
    score, which is why the reference and baseline always come from the *same*
    cohort object.
    """
    rows = []
    for cohort in cohorts:
        reference = evaluate_arm(cohort.frame, REFERENCE_ARM)
        for arm in cohort.arms:
            if arm.name == REFERENCE_ARM.name:
                continue
            baseline = evaluate_arm(cohort.frame, arm)
            row = {
                "period": cohort.period,
                "horizon_months": cohort.horizon,
                "cohort": cohort.name,
                "reference_arm": REFERENCE_ARM.name,
                "baseline_arm": arm.name,
                "n_observations": len(cohort.frame),
            }
            for metric in ("f1", "precision", "recall"):
                a = getattr(reference, metric)
                b = getattr(baseline, metric)
                row[f"{metric}_reference"] = a
                row[f"{metric}_baseline"] = b
                row[f"delta_{metric}"] = (
                    float("nan") if (np.isnan(a) or np.isnan(b)) else a - b
                )
                reasons = []
                if np.isnan(a):
                    reasons.append(f"{REFERENCE_ARM.name}_{metric}_undefined")
                if np.isnan(b):
                    reasons.append(f"{arm.name}_{metric}_undefined")
                row[f"delta_{metric}_reason"] = "|".join(reasons)
            rows.append(row)
    return pd.DataFrame(rows)


#: Q9a: reported separately within each horizon/cohort, never as an automatic
#: cross-product of groupings.
BREAKDOWN_DIMENSIONS = {
    "target_month": ("target_month",),
    "target_year": ("target_year_label",),
    "country": ("country_id", "country_en"),
    "assignment_provenance": ("assignment_source",),
}


def breakdown_table(cohorts: Sequence[Cohort], dimension: str) -> pd.DataFrame:
    """One grouping dimension, on matching keys for every eligible arm."""
    if dimension not in BREAKDOWN_DIMENSIONS:
        raise ReportContractError(f"unknown breakdown dimension: {dimension}")
    columns = BREAKDOWN_DIMENSIONS[dimension]
    rows = []
    for cohort in cohorts:
        frame = cohort.frame
        if frame.empty:
            continue
        frame = frame.copy()
        frame["target_year_label"] = (frame["target_month_ord"] // 12).astype(int).astype(str)
        for keys, group in frame.groupby(list(columns), sort=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            support = _support(group)
            for arm in cohort.arms:
                result = evaluate_arm(group, arm)
                rows.append(
                    {
                        "period": cohort.period,
                        "horizon_months": cohort.horizon,
                        "cohort": cohort.name,
                        "dimension": dimension,
                        **dict(zip(columns, keys)),
                        "arm": arm.name,
                        **support,
                        "predicted_positives": int((group[arm.pred_column] == 1).sum()),
                        **{
                            k: v
                            for k, v in result.as_row().items()
                            if k != "n_observations"
                        },
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Q9b country-cluster bootstrap
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    summary: pd.DataFrame
    replicates: pd.DataFrame
    draws: pd.DataFrame
    config: dict


def _f1_from_count_arrays(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """Vectorised Q9a F1 over many replicates; NaN where 2TP+FP+FN == 0."""
    den = 2.0 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, 2.0 * tp / np.where(den > 0, den, 1.0), np.nan)
    return out


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(
        float(v)
        for v in np.percentile(
            values,
            [CI_LOWER_PERCENTILE, CI_UPPER_PERCENTILE],
            method=PERCENTILE_METHOD,
        )
    )


def bootstrap_cohort(
    cohort: Cohort,
    *,
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapResult:
    """Paired country-cluster bootstrap for one horizon/cohort (Q9b).

    Each draw samples K countries uniformly with replacement and keeps *every*
    row of each copy, preserving multiplicity. Because Q9a's confusion counts
    are additive over rows, summing a country's counts with its multiplicity is
    arithmetically identical to materialising the duplicated rows -- and the
    exported multiplicity table lets a third party check that.

    All arms share the identical sampled rows in a replicate, and each paired
    difference is taken *within* that same replicate.
    """
    frame = cohort.frame
    arms = cohort.arms

    countries = np.array(sorted(frame["country_id"].astype(str).unique()), dtype=object)
    n_countries = len(countries)
    truth = frame["ipcch_food_crisis"].to_numpy(dtype=np.int64)

    per_country: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if n_countries:
        code = np.searchsorted(countries, frame["country_id"].astype(str).to_numpy())
        for arm in arms:
            pred = frame[arm.pred_column].to_numpy(dtype=np.int64)
            if (pred == -1).any():
                raise ReportContractError(
                    f"{arm.name} has a missing prediction inside {cohort.label}"
                )
            t1, p1 = truth == 1, pred == 1
            counts = []
            for mask in (t1 & p1, ~t1 & p1, t1 & ~p1):
                counts.append(
                    np.bincount(code, weights=mask.astype(np.float64), minlength=n_countries)
                )
            per_country[arm.name] = tuple(counts)

    # Point estimates: the full cohort is the all-ones multiplicity vector, so
    # these agree with metrics_table by construction.
    point = {arm.name: evaluate_arm(frame, arm).f1 for arm in arms}
    for arm in arms:
        if arm.name == REFERENCE_ARM.name:
            continue
        a, b = point[REFERENCE_ARM.name], point[arm.name]
        point[f"delta_f1__{REFERENCE_ARM.name}_minus_{arm.name}"] = (
            float("nan") if (np.isnan(a) or np.isnan(b)) else a - b
        )

    statistic_names = [f"f1__{arm.name}" for arm in arms] + [
        f"delta_f1__{REFERENCE_ARM.name}_minus_{arm.name}"
        for arm in arms
        if arm.name != REFERENCE_ARM.name
    ]
    point_by_statistic = {
        f"f1__{arm.name}": point[arm.name] for arm in arms
    }
    point_by_statistic.update(
        {name: point[name] for name in statistic_names if name.startswith("delta_f1__")}
    )

    config = {
        "requested_draws": int(n_draws),
        "seed": int(seed),
        "rng": "numpy.random.default_rng",
        "sampler": (
            "Generator.integers(0, K, size=(draws, K)) as indices into the "
            "sorted country list; K countries with replacement per draw"
        ),
        "percentile_method": PERCENTILE_METHOD,
        "percentiles": [CI_LOWER_PERCENTILE, CI_UPPER_PERCENTILE],
        "n_countries": int(n_countries),
        "cohort": cohort.label,
    }

    # An empty cohort has nothing to resample; that is the only case where no
    # replicate exists. K = 1 still draws -- Q9b suppresses the *interval*
    # below two countries, it does not cancel the sampling (finding 4).
    if n_countries == 0:
        summary = pd.DataFrame(
            [
                {
                    "period": cohort.period,
                    "horizon_months": cohort.horizon,
                    "cohort": cohort.name,
                    "statistic": name,
                    "point": point_by_statistic[name],
                    "n_countries": n_countries,
                    "requested_draws": int(n_draws),
                    "defined_replicates": 0,
                    "undefined_replicates": 0,
                    "ci_lower": float("nan"),
                    "ci_upper": float("nan"),
                    "ci_available": 0,
                    "uses_defined_replicates_only": 0,
                    "suppression_reason": "empty_cohort_no_countries_to_resample",
                    "undefined_reason": "",
                }
                for name in statistic_names
            ]
        )
        empty_reps = pd.DataFrame(columns=["draw_index", "statistic", "value"])
        empty_draws = pd.DataFrame(columns=["draw_index", "country_id", "multiplicity"])
        return BootstrapResult(summary, empty_reps, empty_draws, config)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_countries, size=(n_draws, n_countries))
    # Per-draw multiplicity over the sorted country list.
    flat = idx + (np.arange(n_draws) * n_countries)[:, None]
    multiplicity = np.bincount(
        flat.ravel(), minlength=n_draws * n_countries
    ).reshape(n_draws, n_countries)

    replicate_values: dict[str, np.ndarray] = {}
    for arm in arms:
        tp, fp, fn = per_country[arm.name]
        replicate_values[f"f1__{arm.name}"] = _f1_from_count_arrays(
            multiplicity @ tp, multiplicity @ fp, multiplicity @ fn
        )
    for arm in arms:
        if arm.name == REFERENCE_ARM.name:
            continue
        # Same draw for both arms; NaN in either side propagates.
        replicate_values[
            f"delta_f1__{REFERENCE_ARM.name}_minus_{arm.name}"
        ] = replicate_values[f"f1__{REFERENCE_ARM.name}"] - replicate_values[
            f"f1__{arm.name}"
        ]

    summary_rows = []
    for name in statistic_names:
        values = replicate_values[name]
        defined = ~np.isnan(values)
        n_defined = int(defined.sum())
        n_undefined = int(n_draws - n_defined)
        point_value = point_by_statistic[name]
        reason = ""
        lower = upper = float("nan")
        available = 0
        # Q9b's three non-degeneracy gates, in a fixed order so the reported
        # reason is deterministic. Only the interval is suppressed; the
        # replicate statistics below are kept whatever the reason.
        if np.isnan(point_value):
            reason = "point_estimate_undefined"
        elif n_countries < MIN_COUNTRIES_FOR_CI:
            reason = f"fewer_than_{MIN_COUNTRIES_FOR_CI}_countries"
        elif n_defined < MIN_DEFINED_REPLICATES_FOR_CI:
            reason = f"fewer_than_{MIN_DEFINED_REPLICATES_FOR_CI}_defined_replicates"
        else:
            lower, upper = _percentile_interval(values[defined])
            available = 1
        summary_rows.append(
            {
                "period": cohort.period,
                "horizon_months": cohort.horizon,
                "cohort": cohort.name,
                "statistic": name,
                "point": point_value,
                "n_countries": n_countries,
                "requested_draws": int(n_draws),
                "defined_replicates": n_defined,
                "undefined_replicates": n_undefined,
                "ci_lower": lower,
                "ci_upper": upper,
                "ci_available": available,
                # Explicit label when quantiles used only the defined draws.
                "uses_defined_replicates_only": int(available and n_undefined > 0),
                "suppression_reason": reason,
                "undefined_reason": (
                    "replicate_f1_denominator_zero" if n_undefined else ""
                ),
            }
        )

    replicates = pd.DataFrame(
        {
            "draw_index": np.tile(np.arange(n_draws), len(statistic_names)),
            "statistic": np.repeat(statistic_names, n_draws),
            "value": np.concatenate([replicate_values[n] for n in statistic_names]),
        }
    )

    draw_rows, country_rows = np.nonzero(multiplicity)
    draws = pd.DataFrame(
        {
            "draw_index": draw_rows,
            "country_id": countries[country_rows],
            "multiplicity": multiplicity[draw_rows, country_rows],
        }
    )

    return BootstrapResult(pd.DataFrame(summary_rows), replicates, draws, config)


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


def reporting_config() -> dict:
    return {
        "decision_threshold_note": (
            "p1 > 0.5 gives 1, exactly 0.5 gives 0 (Q7a); the reporter verifies "
            "the saved hard labels and never re-derives a decision."
        ),
        "decision_threshold": DECISION_THRESHOLD,
        "cohorts": {
            COHORT_E_ALL: "every valid test key in the approved R4 schedule; "
            "three learned arms",
            COHORT_E_PERSIST: "the history-available subset of E_all; all four "
            "arms on identical keys",
            "membership": "history availability is read from the valid as-of "
            "label history and reconciled against the saved persistence "
            "columns, not taken from whichever rows the runner filled in "
            "(Q4/Q4b)",
            "empty_cohorts": "every approved horizon of the main period is "
            "emitted even with zero rows, with undefined metrics and a "
            "schedule reconciliation, so an absent horizon stays visible (Q9a)",
        },
        "arms": {
            COHORT_E_ALL: [a.name for a in COHORT_ARMS[COHORT_E_ALL]],
            COHORT_E_PERSIST: [a.name for a in COHORT_ARMS[COHORT_E_PERSIST]],
        },
        "reference_arm": REFERENCE_ARM.name,
        "metrics": {
            "f1": "2TP/(2TP+FP+FN)",
            "precision": "TP/(TP+FP)",
            "recall": "TP/(TP+FN)",
            "aggregation": "observation-level counts aggregated first, unit row "
            "weight, then the ratio; no population weighting and no "
            "country/month mean headline",
            "zero_denominator": "NaN plus a reason",
            "zero_numerator_positive_denominator": "0",
        },
        "main_target_schedule": MAIN_TARGET_SCHEDULE,
        "periods": [PERIOD_MAIN, PERIOD_PARTIAL],
        "bootstrap": {
            "scope": "main Stage3 horizon/cohort summaries only; no subgroup, "
            "partial-2026 or Stage1 intervals",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "rng": "numpy.random.default_rng, re-seeded per horizon/cohort",
            "cluster": "country_id, sorted before sampling",
            "percentile_method": PERCENTILE_METHOD,
            "percentiles": [CI_LOWER_PERCENTILE, CI_UPPER_PERCENTILE],
            "min_countries": MIN_COUNTRIES_FOR_CI,
            "min_defined_replicates": MIN_DEFINED_REPLICATES_FOR_CI,
            "suppression_scope": "only the interval is suppressed; the draws "
            "and replicate statistics are still produced and exported "
            "whenever the cohort has at least one country",
            "suppression_reason_order": [
                "point_estimate_undefined",
                f"fewer_than_{MIN_COUNTRIES_FOR_CI}_countries",
                f"fewer_than_{MIN_DEFINED_REPLICATES_FOR_CI}_defined_replicates",
            ],
            "no_redraw_until_valid": True,
            "nan_to_zero": False,
            "scope_sentence": UNCERTAINTY_SCOPE_SENTENCE,
        },
        "breakdown_dimensions": sorted(BREAKDOWN_DIMENSIONS),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": sys.version.split()[0],
        },
    }


def generate_report(
    out_dir: Path | str,
    *,
    run_dir: Path | str | None = None,
    predictions_path: Path | str | None = None,
    predictions: pd.DataFrame | None = None,
    history: LabelHistory | None = None,
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Read saved predictions, write every report artifact, return the manifest.

    Exactly one input source is used, in order of precedence: an in-memory
    ``predictions`` frame, an explicit ``predictions_path``, or the Stage3 file
    discovered under ``run_dir``. ``out_dir`` must be fresh.

    ``history`` is the R1 valid-label ledger that defines persistence
    availability (Q4/Q4b). When ``run_dir`` is given the ledger is discovered
    inside the run and is *required*: a real run must be able to prove which
    rows have available history. Without a run directory the reporter falls
    back to the subset provable from the saved rows and records that weaker
    verification in ``validation.json`` and ``limitations.txt``.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ReportContractError(
            f"{out_dir} already exists and is not empty; the reporter writes to a "
            "fresh directory so a third party can reconstruct independently"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    source_path = None
    source_sha256 = ""
    if predictions is not None:
        raw = predictions.astype(str)
        source_description = "in-memory frame"
    else:
        if predictions_path is None:
            if run_dir is None:
                raise ReportContractError("one of predictions, predictions_path or run_dir is required")
            predictions_path = find_predictions(run_dir)
        source_path = Path(predictions_path)
        source_sha256 = sha256_file(source_path)
        raw = read_prediction_rows(source_path)
        source_description = str(source_path)

    if history is None and run_dir is not None:
        history = find_label_history(run_dir)

    config = reporting_config()
    _write_json(config, out_dir / "reporting_config.json")
    limitations = list(LIMITATIONS)
    if history is None:
        limitations.append(HISTORY_PARTIAL_LIMITATION)
    (out_dir / "limitations.txt").write_text(
        "\n".join(f"- {line}" for line in limitations) + "\n", encoding="utf-8"
    )

    try:
        prepared, validation = validate_predictions(raw, history)
    except ReportContractError as exc:
        _write_json(
            {
                "status": "failed",
                "stage": "validation",
                "source": source_description,
                "source_sha256": source_sha256,
                "error": str(exc),
            },
            out_dir / "validation.json",
        )
        raise

    validation["status"] = "passed"
    validation["source"] = source_description
    validation["source_sha256"] = source_sha256
    _write_json(validation, out_dir / "validation.json")

    _write_csv_gz(cohort_keys_table(prepared), out_dir / "cohort_keys.csv.gz")

    cohorts = build_cohorts(prepared)
    written: list[str] = [
        "reporting_config.json",
        "limitations.txt",
        "validation.json",
        "cohort_keys.csv.gz",
    ]
    periods_present = []

    for period in (PERIOD_MAIN, PERIOD_PARTIAL):
        slice_ = [c for c in cohorts if c.period == period]
        if not slice_:
            continue
        periods_present.append(period)
        period_dir = out_dir / period

        _write_csv(cohort_support_table(slice_, prepared), period_dir / "cohort_support.csv")
        _write_csv(metrics_table(slice_), period_dir / "metrics.csv")
        _write_csv(deltas_table(slice_), period_dir / "deltas.csv")
        written += [
            f"{period}/cohort_support.csv",
            f"{period}/metrics.csv",
            f"{period}/deltas.csv",
        ]
        for dimension in sorted(BREAKDOWN_DIMENSIONS):
            _write_csv(
                breakdown_table(slice_, dimension),
                period_dir / f"breakdown_{dimension}.csv",
            )
            written.append(f"{period}/breakdown_{dimension}.csv")

        if period != PERIOD_MAIN:
            # Q9b: no subgroup, partial-2026 or Stage1 intervals in this baseline.
            _write_json(
                {
                    "bootstrap": "suppressed",
                    "reason": "Q9b applies only to main Stage3 horizon/cohort "
                    "summaries; partial-2026 keeps point estimates and support.",
                },
                period_dir / "bootstrap_suppressed.json",
            )
            written.append(f"{period}/bootstrap_suppressed.json")
            continue

        summaries, replicates, draws, configs = [], [], [], {}
        for cohort in slice_:
            result = bootstrap_cohort(cohort, n_draws=n_draws, seed=seed)
            summaries.append(result.summary)
            reps = result.replicates.copy()
            drw = result.draws.copy()
            for frame in (reps, drw):
                frame.insert(0, "cohort", cohort.name)
                frame.insert(0, "horizon_months", cohort.horizon)
            replicates.append(reps)
            draws.append(drw)
            configs[cohort.label] = result.config

        _write_csv(pd.concat(summaries, ignore_index=True), period_dir / "bootstrap_summary.csv")
        _write_csv_gz(
            _concat_frames(
                replicates,
                ("horizon_months", "cohort", "draw_index", "statistic", "value"),
            ),
            period_dir / "bootstrap_replicates.csv.gz",
        )
        _write_csv_gz(
            _concat_frames(
                draws,
                ("horizon_months", "cohort", "draw_index", "country_id", "multiplicity"),
            ),
            period_dir / "bootstrap_draws.csv.gz",
        )
        _write_json(configs, period_dir / "bootstrap_config.json")
        written += [
            f"{period}/bootstrap_summary.csv",
            f"{period}/bootstrap_replicates.csv.gz",
            f"{period}/bootstrap_draws.csv.gz",
            f"{period}/bootstrap_config.json",
        ]

    manifest = {
        "status": "complete",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "reporter_sha256": sha256_file(Path(__file__)),
        "source": source_description,
        "source_sha256": source_sha256,
        "run_dir": str(run_dir) if run_dir is not None else "",
        "out_dir": str(out_dir),
        "rows": validation["rows"],
        "periods": periods_present,
        "cohorts": [c.label for c in cohorts],
        "history_verification": validation["history_verification"],
        "files": sorted(written),
        "notes": limitations,
        "fitting_performed": False,
        "map_modified": False,
    }
    _write_json(manifest, out_dir / "report_manifest.json")
    return manifest


# --------------------------------------------------------------------------
# CLI -- reads existing run outputs only; no scientific knobs
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct IPCCH Stage3 metrics and country-bootstrap intervals "
            "from saved prediction rows. Never fits a model or changes a map."
        )
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "run directory holding stage3 predictions and the R1 valid-label "
            "ledger that defines persistence availability"
        ),
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help=(
            "explicit prediction file; without a run directory persistence "
            "availability can only be checked against the saved rows' own truth"
        ),
    )
    parser.add_argument("--out-dir", required=True, help="fresh output directory")
    args = parser.parse_args(argv)

    if args.run_dir is None and args.predictions is None:
        parser.error("one of --run-dir or --predictions is required")

    try:
        manifest = generate_report(
            args.out_dir, run_dir=args.run_dir, predictions_path=args.predictions
        )
    except ReportContractError as exc:
        print(f"REPORT FAILED: {exc}", file=sys.stderr)
        return 2

    print(f"report complete: {manifest['out_dir']}")
    print(f"  rows      : {manifest['rows']}")
    print(f"  periods   : {', '.join(manifest['periods'])}")
    print(f"  cohorts   : {len(manifest['cohorts'])}")
    print(f"  files     : {len(manifest['files'])}")
    print(f"  {UNCERTAINTY_SCOPE_SENTENCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
