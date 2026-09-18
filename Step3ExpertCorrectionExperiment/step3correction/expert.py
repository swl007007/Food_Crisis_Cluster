"""FEWS NET expert reconstruction for the Step 3 correction experiment.

Two *separate* series are built from the same source, and they must never be
confused with one another.

**Experiment expert (calendar-aligned, R1 revised 2026-09-18).**
    The expert estimate for target month ``T`` is the projection *published* at
    origin ``O = T - H``: ``fews_proj_near`` at ``T-4`` for fs1 and
    ``fews_proj_med`` at ``T-8`` for fs2.  A near projection published in month
    ``D`` targets ``D+4``; a medium projection published in ``D`` targets
    ``D+8``.  This is the only series the correction layer may consume.

**Legacy record-shift series (pipeline-validation artifact only).**
    The historical evaluator
    (``archived/release_20260624_nonpaper_pipelines/legacy_misc/app_final/
    fewsnet_baseline_evaluation.py:67-75``) applied a per-admin **record**
    ``shift(4)``/``shift(8)``.  ``FEWSNET.csv`` is not monthly - each admin has
    exactly 53 records, quarterly 2009-07..2015-10 then tri-annual
    (Feb/Jun/Oct) 2016-02..2024-10 - so those record shifts resolve to 12-16
    (fs1) and 24-32 (fs2) calendar months.  The series is retained *solely* to
    prove this loader reproduces the archived baselines bit-for-bit; it is
    prefixed with :data:`LEGACY_COLUMN_PREFIX` and is structurally excluded from
    :meth:`ExpertTable.for_scope`, which is the only path into the correction
    layer.

Both series share the unchanged historical conventions: per-admin ordering,
binarisation to IPC Phase 3+ **before** any join or shift, and the raw-missing
phase -> 0 rule.  Missing or absent publications stay missing; nothing is ever
imputed.  The Ethiopia calendar anchors (including its fs3 ``T-12`` medium
anchor) must never be substituted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .protected import FEWSNET_SOURCE, archived_expert_baseline_path

KEYS = ["admin_code", "month_start"]
SCOPE_PHASE_FIELD: Dict[int, str] = {1: "fews_proj_near", 2: "fews_proj_med"}
SCOPE_HORIZON_MONTHS: Dict[int, int] = {1: 4, 2: 8}

#: Historical record shifts, kept only for the pipeline-validation artifact.
LEGACY_RECORD_SHIFT: Dict[int, int] = {1: 4, 2: 8}
#: Every legacy column carries this prefix so the firewall can be enforced by name.
LEGACY_COLUMN_PREFIX = "legacy_record_shift_"

EXPERT_ALIGNMENT = "calendar"
EXPERT_CONVENTION = (
    "binarise phase>=3 with raw-missing->0, then calendar-join the projection "
    "published at origin O = T-H (fs1 fews_proj_near at T-4, fs2 fews_proj_med "
    "at T-8); absent publications stay missing and are never imputed"
)
LEGACY_EXPERT_CONVENTION = (
    "PIPELINE-VALIDATION ARTIFACT ONLY - not the experiment expert: binarise "
    "phase>=3 with raw-missing->0, then per-admin record shift (fs1 shift(4), "
    "fs2 shift(8)). On the tri-annual source this resolves to 12-16 (fs1) and "
    "24-32 (fs2) calendar months, which is why it was replaced."
)


class ExpertContractError(RuntimeError):
    """Raised when the expert data contract is violated; the run must halt."""


def legacy_column(scope: int) -> str:
    """Return the legacy record-shift column name for ``scope``."""
    return f"{LEGACY_COLUMN_PREFIX}expert_{scope}"


def assert_no_legacy_expert_columns(frame: pd.DataFrame, context: str) -> None:
    """Halt if a legacy record-shift column could reach the correction layer."""
    leaked = [name for name in frame.columns if str(name).startswith(LEGACY_COLUMN_PREFIX)]
    if leaked:
        raise ExpertContractError(
            f"{context}: legacy record-shift columns {leaked} must never reach the "
            "correction layer; only the calendar-aligned expert is permitted"
        )


@dataclass(frozen=True)
class ExpertTable:
    """Reconstructed expert history with per-row source provenance."""

    frame: pd.DataFrame
    source_path: Path

    def for_scope(self, scope: int) -> pd.DataFrame:
        """Return the **calendar-aligned** expert columns for one scope.

        This is the only supported entry point to the correction layer, and it
        deliberately cannot emit a legacy record-shift column.
        """
        if scope not in SCOPE_PHASE_FIELD:
            raise ExpertContractError(
                f"Expert correction is only defined for fs1/fs2, got fs{scope}"
            )
        columns = KEYS + [
            "source_truth",
            "fews_ipc",
            f"expert_{scope}",
            f"expert_{scope}_source_month",
            f"expert_{scope}_phase_raw",
            f"expert_{scope}_phase_missing",
        ]
        selected = self.frame[columns].copy()
        assert_no_legacy_expert_columns(selected, f"fs{scope} expert lookup")
        return selected

    def legacy_record_shift_series(self, scope: int) -> pd.Series:
        """Return the legacy record-shift series (pipeline validation only)."""
        if scope not in SCOPE_PHASE_FIELD:
            raise ExpertContractError(f"No legacy series defined for fs{scope}")
        return self.frame[legacy_column(scope)]


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Require integral admin IDs, first-of-month dates and unique area-month keys."""
    codes = pd.to_numeric(frame["admin_code"], errors="raise")
    if codes.isna().any() or not np.isfinite(codes).all() or (codes % 1 != 0).any():
        raise ExpertContractError("Invalid admin code in expert source")
    frame = frame.copy()
    frame["admin_code"] = codes.astype("int64")
    frame["month_start"] = pd.to_datetime(frame["month_start"], errors="raise")
    if frame[KEYS].isna().any().any():
        raise ExpertContractError("Null admin-month key in expert source")
    if frame.duplicated(KEYS).any():
        duplicates = frame.loc[frame.duplicated(KEYS, keep=False), KEYS]
        raise ExpertContractError(
            f"Duplicate valid admin-month keys in expert source:\n{duplicates.head(20)}"
        )
    if frame["month_start"].dt.day.ne(1).any():
        raise ExpertContractError("Expert source months must start on day 1")
    return frame.sort_values(KEYS).reset_index(drop=True)


def _calendar_align(raw: pd.DataFrame, scope: int) -> pd.DataFrame:
    """Join the projection published at ``O = T - H`` onto each target month.

    The already-binarised publication is re-keyed by the month it *targets*
    (``source month + H``) and left-joined onto the panel.  Targets whose origin
    month is absent from the source - every pre-2016 quarterly-era origin that
    misses the release grid - stay missing.
    """
    horizon = SCOPE_HORIZON_MONTHS[scope]
    phase = SCOPE_PHASE_FIELD[scope]
    # Historical convention: binarise (raw missing -> 0) BEFORE joining.
    published = pd.DataFrame(
        {
            "admin_code": raw["admin_code"].to_numpy(),
            "_target_month": (raw["month_start"] + pd.DateOffset(months=horizon)).to_numpy(),
            f"expert_{scope}": raw[phase].ge(3).astype(int).to_numpy(),
            f"expert_{scope}_source_month": raw["month_start"].to_numpy(),
            f"expert_{scope}_phase_raw": raw[phase].to_numpy(),
            f"expert_{scope}_phase_missing": raw[phase].isna().astype(int).to_numpy(),
        }
    )
    if published.duplicated(["admin_code", "_target_month"]).any():
        raise ExpertContractError(
            f"fs{scope} calendar alignment produced duplicate (admin, target month) keys"
        )
    merged = raw[KEYS].merge(
        published,
        left_on=KEYS,
        right_on=["admin_code", "_target_month"],
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(raw):
        raise ExpertContractError(
            f"fs{scope} calendar alignment changed the row count "
            f"({len(merged)} vs {len(raw)})"
        )
    return merged.drop(columns=["_target_month"] + KEYS)


def load_expert_history(source_path: Path | str = FEWSNET_SOURCE) -> ExpertTable:
    """Rebuild the full historical expert series before any date or cohort cut."""
    source_path = Path(source_path)
    raw = pd.read_csv(source_path)
    for name in ("admin_code", "year", "month"):
        raw[name] = pd.to_numeric(raw[name], errors="raise")

    invalid = raw[["admin_code", "year", "month"]].isna().any(axis=1)
    if invalid.any():
        if not raw.loc[invalid, ["admin_code", "year", "month"]].isna().all().all():
            raise ExpertContractError(
                "Source rows with partially missing admin/date keys are not interpretable"
            )
        raw = raw.loc[~invalid].copy()
    if (raw[["year", "month"]] % 1 != 0).any().any() or not raw["month"].between(1, 12).all():
        raise ExpertContractError("Invalid source year/month value")

    raw["month_start"] = pd.to_datetime(
        dict(year=raw["year"].astype(int), month=raw["month"].astype(int), day=1)
    )
    raw = _normalize_keys(raw)
    raw["source_truth"] = raw["fews_ipc"].ge(3).astype(int)
    raw["quarter"] = (raw["month"].astype(int) - 1) // 3 + 1

    for scope, phase in SCOPE_PHASE_FIELD.items():
        aligned = _calendar_align(raw, scope)
        for column in aligned.columns:
            raw[column] = aligned[column].to_numpy()
        # Pipeline-validation artifact only; prefixed so the firewall can see it.
        binary = raw[phase].ge(3).astype(int)
        raw[legacy_column(scope)] = binary.groupby(raw["admin_code"]).shift(
            LEGACY_RECORD_SHIFT[scope]
        )
    return ExpertTable(frame=raw, source_path=source_path)


def verify_archived_expert_baselines(
    table: ExpertTable, scopes=(1, 2), tolerance: float = 1e-12
) -> Dict[int, Dict[str, object]]:
    """Reproduce archived fs1/fs2 expert baselines from the **legacy** series.

    This is the pipeline-validation gate (AC1a): it proves the loader, the
    binarisation order and the raw-missing-phase convention are faithful to the
    frozen historical evaluator.  It deliberately uses the record-shift series,
    which the archived paper baseline inherits, and says nothing about the
    calendar-aligned series the experiment actually consumes.

    Mirrors the verification pattern already used by
    ``other_outputs/georf_country_performance/generate.py:105-145`` at its 1e-12
    tolerance, without invoking that document-writing entrypoint.
    """
    raw = table.frame
    summary: Dict[int, Dict[str, object]] = {}
    for scope in scopes:
        column = legacy_column(scope)
        reference = pd.read_csv(archived_expert_baseline_path(scope))
        if len(reference) != 39 or reference.duplicated(["year", "quarter"]).any():
            raise ExpertContractError(f"Unexpected archived fs{scope} baseline support")
        checked = 0
        for row in reference.to_dict("records"):
            quarter = raw.loc[raw["year"].eq(row["year"]) & raw["quarter"].eq(row["quarter"])]
            quarter = quarter.dropna(subset=[column])
            counts, scores = counts_and_scores(quarter["source_truth"], quarter[column])
            expected = [row["precision(1)"], row["recall(1)"], row["f1(1)"]]
            np.testing.assert_allclose(scores, expected, rtol=0, atol=tolerance)
            if counts[0] + counts[2] != row["num_samples(1)"]:
                raise ExpertContractError(
                    f"fs{scope} {row['year']}Q{row['quarter']} positive support mismatch: "
                    f"{counts[0] + counts[2]} != {row['num_samples(1)']}"
                )
            checked += 1
        summary[scope] = {
            "quarters_reproduced": checked,
            "series": "legacy record shift (pipeline validation only)",
            "record_shift": LEGACY_RECORD_SHIFT[scope],
        }
    return summary


def counts_and_scores(truth, prediction):
    """Positive-class counts and scores; zero denominators return 0.0 as archived."""
    y = np.asarray(truth)
    p = np.asarray(prediction)
    if not (np.isin(y, [0, 1]).all() and np.isin(p, [0, 1]).all()):
        raise ExpertContractError("Expected complete binary labels for metric computation")
    tp, fp, fn, tn = (
        int(mask.sum())
        for mask in (
            (y == 1) & (p == 1),
            (y == 0) & (p == 1),
            (y == 1) & (p == 0),
            (y == 0) & (p == 0),
        )
    )
    scores = [
        a / b if b else 0.0
        for a, b in ((tp, tp + fp), (tp, tp + fn), (2 * tp, 2 * tp + fp + fn))
    ]
    return [tp, fp, fn, tn], scores


def coverage_report(frame: pd.DataFrame, scope: int, *, label: str = "all rows") -> Dict[str, object]:
    """Report calendar-aligned expert coverage over ``frame``'s rows."""
    available = pd.to_datetime(frame[f"expert_{scope}_source_month"]).notna()
    rows = int(len(frame))
    return {
        "scope": scope,
        "support": label,
        "rows": rows,
        "rows_with_expert": int(available.sum()),
        "coverage": (float(available.sum()) / rows) if rows else 0.0,
    }


def audit_source_alignment(frame: pd.DataFrame, scope: int) -> Dict[str, object]:
    """Report the observed source-row lag versus the declared origin ``target - H``.

    Under calendar alignment every available row must sit at exactly ``T - H``,
    so the observed lag distribution is expected to be a single bucket at ``H``.
    Anything else is reported, never silently repaired.
    """
    horizon = SCOPE_HORIZON_MONTHS[scope]
    source_month = pd.to_datetime(frame[f"expert_{scope}_source_month"])
    target_month = pd.to_datetime(frame["month_start"])
    available = source_month.notna()
    lag = (
        (target_month.dt.year - source_month.dt.year) * 12
        + (target_month.dt.month - source_month.dt.month)
    )
    lag_available = lag[available]
    declared_origin_rows = int((lag_available == horizon).sum())
    return {
        "scope": scope,
        "alignment": EXPERT_ALIGNMENT,
        "declared_horizon_months": horizon,
        "phase_field": SCOPE_PHASE_FIELD[scope],
        "rows": int(len(frame)),
        "rows_with_expert": int(available.sum()),
        "rows_expert_unavailable": int((~available).sum()),
        "rows_at_declared_origin": declared_origin_rows,
        "rows_off_declared_origin": int(available.sum()) - declared_origin_rows,
        "rows_with_nonpositive_lag": int((lag_available <= 0).sum()),
        "observed_lag_value_counts": {
            int(k): int(v) for k, v in lag_available.value_counts().sort_index().items()
        },
        "availability_assumption": (
            "within-month publication timing assumed; the source carries publication "
            "month, not day, and no actual release dates were supplied"
        ),
    }


def require_source_alignment(frame: pd.DataFrame, scope: int) -> Dict[str, object]:
    """Gate the run on an exact ``H``-month publication lag.

    Under the calendar-aligned contract there is no acceptable deviation and no
    override:

    ``leakage``
        Some expert source row is dated *after* the declared origin, so the
        expert could not have been available at forecast time.  Halts
        unconditionally.

    ``stale``
        Some expert source row is older than ``target - H``.  The declared
        forecast horizon is then not what the data says, so this halts too.

    Neither branch may be resolved by reintroducing the legacy record shift.
    """
    audit = audit_source_alignment(frame, scope)
    horizon = SCOPE_HORIZON_MONTHS[scope]
    observed = audit["observed_lag_value_counts"]
    leaking = {lag: count for lag, count in observed.items() if lag < horizon}
    stale = {lag: count for lag, count in observed.items() if lag > horizon}
    audit["leakage_direction_lag_counts"] = leaking
    audit["stale_direction_lag_counts"] = stale
    audit["expert_horizon_verified"] = not leaking and not stale

    if leaking:
        raise ExpertContractError(
            f"fs{scope} expert source rows post-date the declared origin "
            f"(target - {horizon} months): lag->rows {leaking}. The expert could not "
            "have been available at forecast time; halting unconditionally."
        )
    if stale:
        raise ExpertContractError(
            f"fs{scope} expert source dates are older than the declared "
            f"{horizon}-month origin: observed calendar lags (months -> rows) "
            f"{observed}. The calendar-aligned contract admits exactly one lag "
            f"({horizon}); halting rather than relabelling the horizon or "
            "reintroducing the legacy record shift."
        )
    return audit
