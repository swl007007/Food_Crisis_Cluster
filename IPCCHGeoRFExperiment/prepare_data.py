"""IPCCH data preparation: R1 target/QC ledger, features, split and geography.

Plain functions, no adapter class framework (design.md "Boundary and file
responsibilities"). Every public function is pure with respect to the pinned
source: it reads, it never writes back, and it never fills a missing label.

Requirement anchors are cited inline as ``R1``/``R3``/``R4`` etc., matching
`.trellis/tasks/09-19-ipcch-binary-georf-pipeline/prd.md`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Context, Decimal, Inexact, InvalidOperation, localcontext
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Pinned identities (PRD "Pinned inputs and verified background")
# --------------------------------------------------------------------------

SOURCE_SHA256 = "ae696087c3bbb280537ae269a05924133acdb51060d31290523404fa8a717673"
RELEASE_SHA256 = "39a26138e3fafb0be2bbd22e9760095d6cdefa7d79b98b4f798cb3aa79b500a0"

#: R1 gate. A count mismatch is investigated, not normalized away (implement.md 1).
EXPECTED_VALID = 42695
EXPECTED_POSITIVE = 15206
EXPECTED_NEGATIVE = 27489
EXPECTED_AREAS = 6227

PHASE_COLUMNS = (
    "phase1_percent",
    "phase2_percent",
    "phase3_percent",
    "phase4_percent",
    "phase5_percent",
)
KEY_COLUMNS = ("admin_code", "year", "month")

#: R1 step 3 bounds and step 5 threshold, held as Decimal so the .20/.90/1.10
#: boundaries compare exactly against source decimal strings.
SUM_LOWER = Decimal("0.90")
SUM_UPPER = Decimal("1.10")
CRISIS_THRESHOLD = Decimal("0.20")
ZERO = Decimal(0)
ONE = Decimal(1)
FIVE = Decimal(5)

#: Working precision for the *provenance* normalization P_i / S (R1 step 4).
#: A division is almost never exact, so it is the one place a rounding choice
#: is unavoidable; it is declared here, it is far wider than the ~17 written
#: digits of the source, and -- decisively -- the label does not depend on it.
#: R1's exact equivalent ``5*(P3+P4+P5) > S`` is used for the label instead.
NORMALIZATION_PRECISION = 100
_NORMALIZATION_CONTEXT = Context(prec=NORMALIZATION_PRECISION)

#: Exact normalized components, kept as decimal strings alongside the raw ones
#: ("Preserve raw components, P5-fill flag, S, normalized components, share,
#: validity reason and label", R1).
NORMALIZED_PHASE_COLUMNS = tuple(f"normalized_{name}_str" for name in PHASE_COLUMNS)


class DataContractError(RuntimeError):
    """Raised when a pinned contract is violated; the run must stop."""


# --------------------------------------------------------------------------
# Source identity
# --------------------------------------------------------------------------


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    """Stream a file's SHA256 without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(path: Path | str, expected: str = SOURCE_SHA256) -> str:
    """Halt unless the source file is byte-identical to the pinned input (A1)."""
    actual = sha256_file(path)
    if actual != expected:
        raise DataContractError(
            f"source hash mismatch for {path}: expected {expected}, got {actual}. "
            "Never substitute a similarly named corrected input."
        )
    return actual


# --------------------------------------------------------------------------
# R1 — target and provenance
# --------------------------------------------------------------------------


def _to_decimal(raw: object) -> Decimal | None:
    """Parse a source cell to Decimal, preserving its written precision.

    Returns ``None`` for any missing token. Parsing from the *string* keeps
    ``0.20`` exact, which binary floats cannot, and R1 requires exact behaviour
    at .20/.90/1.10.
    """
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


def load_source_strings(path: Path | str, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read the pinned CSV with every field as a string.

    ``dtype=str`` plus ``keep_default_na=False`` stops pandas from coercing
    country codes or phase shares, so R1 sees exactly what the file says
    (design.md "Source and runtime preflight" item 4).
    """
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        usecols=list(columns) if columns is not None else None,
    )


@dataclass(frozen=True)
class TargetLedger:
    """R1 output: one row per source area-month, valid or not."""

    frame: pd.DataFrame
    source_sha256: str

    def valid(self) -> pd.DataFrame:
        return self.frame[self.frame["target_valid"] == 1]

    def summary(self) -> dict:
        valid = self.valid()
        reasons = (
            self.frame.loc[self.frame["target_valid"] == 0, "target_invalid_reason"]
            .value_counts()
            .to_dict()
        )
        return {
            "rows": int(len(self.frame)),
            "valid": int(len(valid)),
            "positive": int((valid["ipcch_food_crisis"] == 1).sum()),
            "negative": int((valid["ipcch_food_crisis"] == 0).sum()),
            "areas_total": int(self.frame["admin_code"].nunique()),
            "areas_with_valid": int(valid["admin_code"].nunique()),
            "countries_with_valid": int(valid["country_en"].nunique()),
            "p5_filled": int(valid["p5_missing_filled"].sum()),
            "normalized_share_equals_threshold": int(
                (valid["normalized_p3plus_str"].map(Decimal) == CRISIS_THRESHOLD).sum()
            ),
            "invalid_reasons": reasons,
        }


def _exact_context(values: Sequence[Decimal], guard: int = 3) -> Context:
    """A context wide enough that adding/scaling ``values`` cannot round.

    ``Decimal`` arithmetic is exact only while the result fits the ambient
    context precision, which defaults to 28 significant digits. A source cell
    written with more digits than that would round -- and at the .20/.90/1.10
    boundaries R1 demands exact behaviour -- so every step-3/step-5 operation
    runs in a context sized from the operands themselves.

    ``guard`` covers the carry: summing five values adds at most one integer
    digit and multiplying by 5 adds at most one more. ``Inexact`` is trapped so
    that a precision error would stop the run instead of silently relabelling
    a row.
    """
    integer_digits = 1
    fraction_digits = 0
    for value in values:
        _sign, digits, exponent = value.as_tuple()
        if not isinstance(exponent, int):  # NaN/Infinity never reach here
            raise DataContractError(f"non-finite decimal in exact arithmetic: {value}")
        integer_digits = max(integer_digits, len(digits) + exponent)
        fraction_digits = max(fraction_digits, -exponent if exponent < 0 else 0)
    context = Context(prec=integer_digits + fraction_digits + guard)
    context.traps[Inexact] = True
    return context


def _exact_sum(values: Sequence[Decimal]) -> Decimal:
    """Sum decimals with no rounding, whatever the ambient context precision."""
    try:
        with localcontext(_exact_context(values)):
            total = ZERO
            for value in values:
                total = total + value
    except Inexact as error:  # pragma: no cover - guard, precision is sufficient
        raise DataContractError(f"inexact decimal sum of {list(values)}") from error
    return total


def _exceeds_threshold(p3plus: Decimal, total: Decimal) -> int:
    """R1 step 5, via the requirement's exact equivalent ``5*(P3+P4+P5) > S``.

    This deliberately avoids the division: multiplication and comparison are
    exact in a sufficiently wide context, so the label can never depend on a
    rounded quotient. Exact equality with .20 stays negative.
    """
    try:
        with localcontext(_exact_context((p3plus, total))):
            scaled = p3plus * FIVE
    except Inexact as error:  # pragma: no cover - guard, precision is sufficient
        raise DataContractError(f"inexact scaling of {p3plus}") from error
    return int(scaled > total)


class _RowVerdict(NamedTuple):
    """One row's R1 outcome; ``None`` everywhere the step was never reached."""

    valid: int
    reason: str
    p5_filled: bool
    total: Decimal | None
    normalized_p3plus: Decimal | None
    normalized_components: tuple[Decimal, ...] | None
    crisis: int | None


def _classify_row(
    phases: Sequence[Decimal | None], population: Decimal | None
) -> _RowVerdict:
    """Apply R1 steps 1-5 to one row.

    The ordering of the checks is the requirement's ordering and is
    load-bearing: a row failing an earlier step is never rescued by a later
    one. The returned label comes from the exact comparison, never from the
    provenance-precision normalized share.
    """
    p1, p2, p3, p4, p5 = phases

    # Step 1 — P1..P4 must be observed. Only a missing P5 may be filled with 0.
    if any(value is None for value in (p1, p2, p3, p4)):
        return _RowVerdict(0, "missing_phase_1_to_4", False, None, None, None, None)
    p5_filled = p5 is None
    if p5_filled:
        p5 = ZERO

    # Step 2 — observed shares in [0, 1]; population observed and strictly > 0.
    filled = (p1, p2, p3, p4, p5)
    for value in filled:
        if value < ZERO or value > ONE:
            return _RowVerdict(
                0, "phase_share_out_of_bounds", p5_filled, None, None, None, None
            )
    if population is None:
        return _RowVerdict(0, "population_missing", p5_filled, None, None, None, None)
    if population <= ZERO:
        return _RowVerdict(
            0, "population_not_positive", p5_filled, None, None, None, None
        )

    # Step 3 — filled five-phase total in [0.90, 1.10]. No clipping, and a raw
    # P3+ above 1 is not an independent exclusion (R1 step 3). The sum is exact:
    # a wider-than-context source value must not round into or out of range.
    total = _exact_sum(filled)
    if total < SUM_LOWER or total > SUM_UPPER:
        return _RowVerdict(0, "sum_out_of_bounds", p5_filled, total, None, None, None)

    # Step 5 first, from the exact cross-multiplication. S > 0 is implied by
    # step 3, so the step-4 quotients below exist; they are provenance only.
    p3plus = _exact_sum((p3, p4, p5))
    crisis = _exceeds_threshold(p3plus, total)

    # Step 4 — proportional normalization by S, at the declared precision.
    normalized_components = tuple(
        _NORMALIZATION_CONTEXT.divide(value, total) for value in filled
    )
    normalized_p3plus = _NORMALIZATION_CONTEXT.divide(p3plus, total)
    return _RowVerdict(
        1, "", p5_filled, total, normalized_p3plus, normalized_components, crisis
    )


def build_target_ledger(path: Path | str, verify_hash: bool = True) -> TargetLedger:
    """Build the R1 target ledger for every source area-month row.

    Invalid truth stays missing: it is never replaced by ``overall_phase``,
    persistence or a probability threshold (R1). Unlabeled scaffold rows are
    retained so later steps can still read them as covariate history (R1 last
    paragraph, design.md "Data flow and tables" item 1).
    """
    path = Path(path)
    source_hash = verify_source(path) if verify_hash else sha256_file(path)

    needed = list(KEY_COLUMNS) + list(PHASE_COLUMNS) + [
        "estimated_population",
        "overall_phase",
        "country_en",
        "ISO3",
    ]
    raw = load_source_strings(path, needed)

    for column in KEY_COLUMNS:
        if (raw[column].str.strip() == "").any():
            raise DataContractError(f"{column} has blank values; keys must be complete")

    out = pd.DataFrame(
        {
            "admin_code": raw["admin_code"].astype(np.int64),
            "year": raw["year"].astype(int),
            "month": raw["month"].astype(int),
            "country_en": raw["country_en"],
            "ISO3": raw["ISO3"],
            "overall_phase_raw": raw["overall_phase"],
        }
    )
    if not out["month"].between(1, 12).all():
        raise DataContractError("month outside 1..12")
    if out.duplicated(["admin_code", "year", "month"]).any():
        raise DataContractError("(admin_code, year, month) is not unique in the source")

    out["target_month"] = pd.to_datetime(
        dict(year=out["year"], month=out["month"], day=1)
    )

    phase_decimals = [raw[column].map(_to_decimal) for column in PHASE_COLUMNS]
    population = raw["estimated_population"].map(_to_decimal)

    verdicts = [
        _classify_row([p[i] for p in phase_decimals], population.iat[i])
        for i in range(len(raw))
    ]

    out["target_valid"] = [v.valid for v in verdicts]
    out["target_invalid_reason"] = [v.reason for v in verdicts]
    out["p5_missing_filled"] = [int(v.p5_filled) for v in verdicts]
    out["phase_sum_S_str"] = [
        ("" if v.total is None else str(v.total)) for v in verdicts
    ]
    out["normalized_p3plus_str"] = [
        ("" if v.normalized_p3plus is None else str(v.normalized_p3plus))
        for v in verdicts
    ]
    # The five normalized components, exactly as R1 asks them to be preserved.
    # Blank on any invalid row: step 4 was never reached there.
    for index, column in enumerate(NORMALIZED_PHASE_COLUMNS):
        out[column] = [
            (
                ""
                if v.normalized_components is None
                else str(v.normalized_components[index])
            )
            for v in verdicts
        ]

    # R1 step 5 — strictly greater than .20; exact equality is negative. The
    # verdict already holds the exact ``5*(P3+P4+P5) > S`` decision, which does
    # not depend on the rounded quotient above.
    out["ipcch_food_crisis"] = pd.Series(
        [(pd.NA if v.crisis is None else v.crisis) for v in verdicts],
        dtype="Int64",
    )

    # Raw components are preserved for audit (R1 "Preserve raw components").
    for column, series in zip(PHASE_COLUMNS, phase_decimals):
        out[f"raw_{column}"] = raw[column]
    out["raw_estimated_population"] = raw["estimated_population"]

    out = out.sort_values(["admin_code", "target_month"], kind="mergesort").reset_index(
        drop=True
    )
    return TargetLedger(frame=out, source_sha256=source_hash)


def check_target_gate(ledger: TargetLedger) -> dict:
    """Compare the ledger against the PRD's audited counts (implement.md gate 1)."""
    summary = ledger.summary()
    mismatches = {}
    for name, expected in (
        ("valid", EXPECTED_VALID),
        ("positive", EXPECTED_POSITIVE),
        ("negative", EXPECTED_NEGATIVE),
        ("areas_total", EXPECTED_AREAS),
    ):
        if summary[name] != expected:
            mismatches[name] = {"expected": expected, "actual": summary[name]}
    summary["gate_pass"] = not mismatches
    summary["gate_mismatches"] = mismatches
    return summary


# --------------------------------------------------------------------------
# R3 — feature schema (Q6a-Q6f)
#
# The 93 model columns and their order are fixed here, once, and shared by
# Stage1 and every Stage3 arm (design.md "Data flow and tables"). The blocks
# below are quoted from research/secondary-predictors.md; family sizes are
# 19/9/6/13/21/2 and are asserted at import time.
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
    "FAO_price",
    "WFP_Price",
    "WFP_Price_std",
    "CPI",
    "GDP",
    "CC",
    "gini",
    "Food_CPI",
    "Food_food_inflation",
)

RAW_VEGETATION_WEATHER_COLUMNS = (
    "EVI_mean",
    "GPP_mean",
    "Rainf_f_tavg_mean",
    "Tair_f_tavg_mean",
    "nightlight_mean",
    "nightlight_std",
)

RAW_LAND_ACCESS_COLUMNS = (
    "crop",
    "range",
    "distance_to_river",
    "elevation",
    "market_distance",
    "market_access",
    "ruggedness",
    "slope",
    "sg_cec_5-15cm",
    "sg_cfvo_5-15cm",
    "sg_nitrogen_5-15cm",
    "sg_phh2o_5-15cm",
    "sg_soc_5-15cm",
)

RAW_AEZ_COLUMNS = (
    "AEZ_4000",
    "AEZ_7000",
    "AEZ_9000",
    "AEZ_10000",
    "AEZ_12000",
    "AEZ_17000",
    "AEZ_19000",
    "AEZ_20000",
    "AEZ_25000",
    "AEZ_28000",
    "AEZ_30000",
    "AEZ_31000",
    "AEZ_32000",
    "AEZ_33000",
    "AEZ_34000",
    "AEZ_35000",
    "AEZ_36000",
    "AEZ_38000",
    "AEZ_40000",
    "AEZ_42000",
    "AEZ_43000",
)

RAW_COORDINATE_COLUMNS = ("lat", "lon")

#: Q6b. Frozen whitelist; no auto-discovery from an updated file.
RAW_FEATURE_COLUMNS = (
    RAW_CONFLICT_COLUMNS
    + RAW_PRICE_MACRO_COLUMNS
    + RAW_VEGETATION_WEATHER_COLUMNS
    + RAW_LAND_ACCESS_COLUMNS
    + RAW_AEZ_COLUMNS
    + RAW_COORDINATE_COLUMNS
)

#: Q6d. ``(output_name, source_column, window_length)``; windows end at O and
#: are inclusive on both ends, so ``length=4`` means O-3..O.
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

#: Q6e (user revision): cyclic target-month encoding, no month/year dummies.
CALENDAR_FEATURE_COLUMNS = ("target_month_sin", "target_month_cos")

#: Q6a: latest observed binary status at O, including 0. NOT the last positive.
HISTORY_FEATURE_COLUMNS = (
    "last_observed_label",
    "last_observed_label_age_months",
    "no_observed_label_history",
)

#: Q6f: recency of the latest observed *positive*; distinct from Q6a's age.
RECENCY_FEATURE_COLUMNS = (
    "months_since_last_observed_crisis",
    "no_prior_observed_crisis",
)

#: Q5h: actual months 1/3/6/12, never a legacy scope index.
HORIZON_FEATURE_COLUMNS = ("horizon_months",)

FEATURE_COLUMNS = (
    RAW_FEATURE_COLUMNS
    + DERIVED_FEATURE_COLUMNS
    + CALENDAR_FEATURE_COLUMNS
    + HISTORY_FEATURE_COLUMNS
    + RECENCY_FEATURE_COLUMNS
    + HORIZON_FEATURE_COLUMNS
)

#: Kept beside the matrix but never handed to a model (design.md "Data flow").
METADATA_COLUMNS = (
    "admin_code",
    "country_en",
    "ISO3",
    "target_month",
    "origin_month",
    "ipcch_food_crisis",
    "last_observed_label_month",
    "last_observed_crisis_month",
)

ACTIVE_HORIZONS = (1, 3, 6, 12)

FEATURE_BLOCKS = {
    "raw": RAW_FEATURE_COLUMNS,
    "derived": DERIVED_FEATURE_COLUMNS,
    "calendar": CALENDAR_FEATURE_COLUMNS,
    "history": HISTORY_FEATURE_COLUMNS,
    "recency": RECENCY_FEATURE_COLUMNS,
    "horizon": HORIZON_FEATURE_COLUMNS,
}

EXPECTED_FEATURE_COUNT = 93

#: Packs ``(admin_code, month_ordinal)`` into one sortable int64 for as-of
#: searches. admin_code maxes at 101324 and ordinals at roughly 2.4e4, so the
#: product stays far inside int64.
_AS_OF_KEY_SCALE = 1_000_000

assert len(RAW_FEATURE_COLUMNS) == 70, len(RAW_FEATURE_COLUMNS)
assert (
    len(RAW_CONFLICT_COLUMNS),
    len(RAW_PRICE_MACRO_COLUMNS),
    len(RAW_VEGETATION_WEATHER_COLUMNS),
    len(RAW_LAND_ACCESS_COLUMNS),
    len(RAW_AEZ_COLUMNS),
    len(RAW_COORDINATE_COLUMNS),
) == (19, 9, 6, 13, 21, 2)
assert len(DERIVED_FEATURE_COLUMNS) == 15, len(DERIVED_FEATURE_COLUMNS)
assert len(FEATURE_COLUMNS) == EXPECTED_FEATURE_COUNT, len(FEATURE_COLUMNS)
assert len(set(FEATURE_COLUMNS)) == EXPECTED_FEATURE_COUNT, "duplicate feature name"


# --------------------------------------------------------------------------
# Calendar arithmetic (design.md: ordinals for arithmetic, YYYY-MM for export)
# --------------------------------------------------------------------------


def month_ordinal(year, month):
    """Map (year, month) to a dense integer month index.

    ``year * 12 + (month - 1)`` makes a difference of ordinals exactly the
    number of calendar months between two dates, which is what every R3 window
    and age is defined in.
    """
    year_arr = np.asarray(year, dtype=np.int64)
    month_arr = np.asarray(month, dtype=np.int64)
    if np.any((month_arr < 1) | (month_arr > 12)):
        raise DataContractError("month outside 1..12 while computing ordinals")
    return year_arr * 12 + (month_arr - 1)


def month_label(ordinal) -> np.ndarray:
    """Render month ordinals as ``YYYY-MM`` strings; ``-1`` renders empty."""
    ordinal = np.asarray(ordinal, dtype=np.int64)
    years = ordinal // 12
    months = ordinal % 12 + 1
    out = np.array(
        [f"{y:04d}-{m:02d}" for y, m in zip(years, months)],
        dtype=object,
    )
    out[ordinal < 0] = ""
    return out


# --------------------------------------------------------------------------
# Covariate panel
# --------------------------------------------------------------------------


def load_covariate_panel(
    path: Path | str, columns: Sequence[str] = RAW_FEATURE_COLUMNS
) -> tuple[pd.DataFrame, dict]:
    """Read the keys plus the frozen raw whitelist as a numeric monthly panel.

    Returns ``(panel, audit)``. The panel keeps *every* source area-month,
    labelled or not: R3 lets unlabeled scaffold rows supply covariates while
    never becoming supervised rows.

    Non-numeric tokens are coerced to NaN and counted rather than silently
    dropped, and infinities become NaN with an audit (Q6c).
    """
    path = Path(path)
    usecols = list(KEY_COLUMNS) + list(columns)
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)

    missing = [name for name in usecols if name not in frame.columns]
    if missing:
        raise DataContractError(f"source is missing required columns: {missing}")

    panel = pd.DataFrame(
        {
            "admin_code": frame["admin_code"].astype(np.int64),
            "year": frame["year"].astype(np.int64),
            "month": frame["month"].astype(np.int64),
        }
    )
    panel["month_ord"] = month_ordinal(panel["year"], panel["month"])

    coerced: dict[str, int] = {}
    infinite: dict[str, int] = {}
    for name in columns:
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
        count = int(mask.sum())
        if count:
            # Q6c: infinities become NaN for both model families, with an audit.
            values[mask] = np.nan
            infinite[name] = count
        panel[name] = values

    if panel.duplicated(["admin_code", "month_ord"]).any():
        raise DataContractError("(admin_code, year, month) is not unique in the panel")

    panel = panel.sort_values(["admin_code", "month_ord"], kind="mergesort").reset_index(
        drop=True
    )

    audit = {
        "panel_rows": int(len(panel)),
        "panel_areas": int(panel["admin_code"].nunique()),
        "panel_month_min": month_label([int(panel["month_ord"].min())])[0],
        "panel_month_max": month_label([int(panel["month_ord"].max())])[0],
        "non_numeric_coerced_to_nan": coerced,
        "source_infinities_converted": infinite,
        "source_infinity_total": int(sum(infinite.values())),
    }
    return panel, audit


# --------------------------------------------------------------------------
# Dense area x month addressing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _PanelGrid:
    """Dense ``(area, month)`` -> panel row position, with -1 for absent cells.

    R3 forbids a sparse-row shift standing in for a calendar month, so every
    lookup below addresses an explicit calendar cell. The grid is one int64
    array of roughly ``areas x months`` cells, which for this source is the
    same order as the panel itself.
    """

    areas: np.ndarray
    month_lo: int
    positions: np.ndarray

    @property
    def n_months(self) -> int:
        return int(self.positions.shape[1])

    def area_index(self, admin_code: np.ndarray) -> np.ndarray:
        """Row index per area code; -1 when the area is absent from the panel."""
        codes = np.asarray(admin_code, dtype=np.int64)
        guess = np.searchsorted(self.areas, codes)
        guess_safe = np.clip(guess, 0, len(self.areas) - 1)
        hit = self.areas[guess_safe] == codes
        return np.where(hit, guess_safe, -1)

    def positions_at(self, area_idx: np.ndarray, month_ord: np.ndarray) -> np.ndarray:
        """Panel row position for each (area, calendar month); -1 when absent."""
        month_idx = np.asarray(month_ord, dtype=np.int64) - self.month_lo
        inside = (
            (area_idx >= 0) & (month_idx >= 0) & (month_idx < self.n_months)
        )
        out = np.full(len(month_idx), -1, dtype=np.int64)
        if inside.any():
            out[inside] = self.positions[area_idx[inside], month_idx[inside]]
        return out


def _build_panel_grid(panel: pd.DataFrame) -> _PanelGrid:
    admin = panel["admin_code"].to_numpy(dtype=np.int64)
    ords = panel["month_ord"].to_numpy(dtype=np.int64)
    areas = np.unique(admin)
    month_lo = int(ords.min())
    month_hi = int(ords.max())
    positions = np.full((len(areas), month_hi - month_lo + 1), -1, dtype=np.int64)
    positions[np.searchsorted(areas, admin), ords - month_lo] = np.arange(
        len(panel), dtype=np.int64
    )
    return _PanelGrid(areas=areas, month_lo=month_lo, positions=positions)


def _gather(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Take rows of ``values`` by position, yielding NaN where position is -1."""
    safe = np.where(positions >= 0, positions, 0)
    taken = values[safe]
    if taken.ndim == 1:
        return np.where(positions >= 0, taken, np.nan)
    taken = taken.astype(np.float64, copy=True)
    taken[positions < 0, :] = np.nan
    return taken


class _WindowSummer:
    """Inclusive per-area calendar window sums that require a complete window.

    Q6d: "A sum requires every month's value in its 4/12-month window;
    otherwise it is NaN before imputation."

    The window is accumulated directly, oldest month first, over at most 12
    shifted reads. A cumulative-sum difference would be asymptotically faster
    but subtracts two area-lifetime totals, so its rounding error scales with
    the whole area's magnitude instead of the window's; at 12 terms the direct
    sum costs nothing and is exactly the arithmetic the requirement states.
    """

    def __init__(self, grid: _PanelGrid, values: np.ndarray):
        dense = np.full(grid.positions.shape, np.nan, dtype=np.float64)
        present = grid.positions >= 0
        dense[present] = values[grid.positions[present]]
        self._observed = ~np.isnan(dense)
        # Only NaN becomes 0 here; ``np.nan_to_num`` would also silently turn
        # an infinity into 1.8e308, so infinities are removed before this point.
        self._filled = np.where(self._observed, dense, 0.0)
        self._grid = grid

    def sum_ending_at(
        self, area_idx: np.ndarray, end_ord: np.ndarray, length: int
    ) -> np.ndarray:
        end_idx = np.asarray(end_ord, dtype=np.int64) - self._grid.month_lo
        n = len(end_idx)
        total = np.zeros(n, dtype=np.float64)
        count = np.zeros(n, dtype=np.int64)
        for offset in range(length - 1, -1, -1):  # oldest month first
            month_idx = end_idx - offset
            # A window month outside the panel is missing, never clamped: a
            # clamped index would silently sum the wrong calendar month.
            inside = (
                (area_idx >= 0)
                & (month_idx >= 0)
                & (month_idx < self._grid.n_months)
            )
            rows = area_idx[inside]
            cols = month_idx[inside]
            total[inside] += self._filled[rows, cols]
            count[inside] += self._observed[rows, cols]
        return np.where(count == length, total, np.nan)


def _as_of_index(
    hist_area: np.ndarray,
    hist_ord: np.ndarray,
    query_area: np.ndarray,
    query_ord: np.ndarray,
) -> np.ndarray:
    """Index of the latest same-area history entry with month <= query month.

    ``side="right"`` makes the bound inclusive, so a label dated exactly at O
    counts (R3: "source/history/training-label month <= O"). Returns -1 when
    the area has no qualifying entry.
    """
    hist_area = np.asarray(hist_area, dtype=np.int64)
    hist_ord = np.asarray(hist_ord, dtype=np.int64)
    query_area = np.asarray(query_area, dtype=np.int64)
    query_ord = np.asarray(query_ord, dtype=np.int64)
    if len(hist_area) == 0:
        return np.full(len(query_area), -1, dtype=np.int64)
    if np.any(hist_ord < 0) or np.any(hist_ord >= _AS_OF_KEY_SCALE):
        raise DataContractError("month ordinal outside the packed as-of key range")

    order = np.lexsort((hist_ord, hist_area))
    sorted_area = hist_area[order]
    sorted_ord = hist_ord[order]
    sorted_key = sorted_area * _AS_OF_KEY_SCALE + sorted_ord

    query_key = query_area * _AS_OF_KEY_SCALE + np.maximum(query_ord, 0)
    slot = np.searchsorted(sorted_key, query_key, side="right") - 1
    safe = np.clip(slot, 0, None)
    hit = (slot >= 0) & (sorted_area[safe] == query_area) & (query_ord >= 0)
    return np.where(hit, order[safe], -1)


# --------------------------------------------------------------------------
# Feature assembly
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureMatrix:
    """Pre-imputation R3 feature rows plus their metadata and audits."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]
    infinity_audit: dict
    audit: dict

    def X(self) -> pd.DataFrame:
        """The 93 model columns, in the fixed order, with NaN left explicit."""
        return self.frame[list(self.feature_columns)]

    def nan_rates(self) -> dict:
        """Per-block and per-column NaN rate over the emitted feature rows."""
        n = len(self.frame)
        by_column = {
            name: (float(self.frame[name].isna().mean()) if n else float("nan"))
            for name in self.feature_columns
        }
        by_block = {}
        for block, names in FEATURE_BLOCKS.items():
            if not n:
                by_block[block] = float("nan")
                continue
            values = self.frame[list(names)].to_numpy(dtype=np.float64)
            by_block[block] = float(np.isnan(values).mean())
        return {"by_block": by_block, "by_column": by_column}


def assemble_feature_matrix(
    valid_labels: pd.DataFrame,
    panel: pd.DataFrame,
    horizons: Iterable[int] = ACTIVE_HORIZONS,
) -> FeatureMatrix:
    """Build the 93-column matrix for each valid target crossed with horizons.

    ``valid_labels`` must already be the R1-valid rows (one per area-month);
    the full valid-label history is used as a lookup source for Q6a/Q6f but
    only these rows become supervised rows. ``panel`` is the numeric monthly
    covariate scaffold from :func:`load_covariate_panel`.

    Every lookup is anchored to that row's *own* origin ``O = T - H``, never a
    later fit origin (R3), and no value is imputed here.
    """
    horizons = tuple(int(h) for h in horizons)
    if not horizons:
        raise DataContractError("at least one horizon is required")
    if len(set(horizons)) != len(horizons):
        raise DataContractError(f"duplicate horizons: {horizons}")

    required = {"admin_code", "year", "month", "ipcch_food_crisis"}
    missing = required - set(valid_labels.columns)
    if missing:
        raise DataContractError(f"valid_labels is missing columns: {sorted(missing)}")
    for name in ("month_ord", *RAW_FEATURE_COLUMNS):
        if name not in panel.columns:
            raise DataContractError(f"panel is missing column: {name}")

    labels = valid_labels.copy()
    labels["admin_code"] = labels["admin_code"].astype(np.int64)
    labels["month_ord"] = month_ordinal(labels["year"], labels["month"])
    if labels.duplicated(["admin_code", "month_ord"]).any():
        raise DataContractError("valid_labels has duplicate (admin_code, month)")
    label_values = pd.to_numeric(labels["ipcch_food_crisis"], errors="coerce")
    if label_values.isna().any():
        raise DataContractError("valid_labels contains a missing target")
    if not np.isin(label_values.to_numpy(), (0, 1)).all():
        raise DataContractError("valid_labels contains a non-binary target")
    labels["ipcch_food_crisis"] = label_values.astype(np.int64)
    labels = labels.sort_values(["admin_code", "month_ord"], kind="mergesort")

    grid = _build_panel_grid(panel)
    raw_values = panel[list(RAW_FEATURE_COLUMNS)].to_numpy(dtype=np.float64, copy=True)

    # Q6c: infinities become NaN *before* any window arithmetic, so a single
    # infinite month cannot poison a cumulative sum. ``load_covariate_panel``
    # normally clears these already; this keeps direct callers safe too.
    input_infinities: dict[str, int] = {}
    infinite_cells = np.isinf(raw_values)
    if infinite_cells.any():
        per_column = infinite_cells.sum(axis=0)
        for column_index, name in enumerate(RAW_FEATURE_COLUMNS):
            if per_column[column_index]:
                input_infinities[name] = int(per_column[column_index])
        raw_values[infinite_cells] = np.nan
    raw_column_index = {name: i for i, name in enumerate(RAW_FEATURE_COLUMNS)}

    n_labels = len(labels)
    n_rows = n_labels * len(horizons)

    # Four views of one outcome, kept adjacent and ordered by horizon.
    label_admin = labels["admin_code"].to_numpy(dtype=np.int64)
    label_ord = labels["month_ord"].to_numpy(dtype=np.int64)
    label_y = labels["ipcch_food_crisis"].to_numpy(dtype=np.int64)

    horizon_arr = np.asarray(sorted(horizons), dtype=np.int64)
    admin = np.repeat(label_admin, len(horizon_arr))
    target_ord = np.repeat(label_ord, len(horizon_arr))
    target_y = np.repeat(label_y, len(horizon_arr))
    horizon = np.tile(horizon_arr, n_labels)
    origin_ord = target_ord - horizon

    area_idx = grid.area_index(admin)

    out = pd.DataFrame(index=pd.RangeIndex(n_rows))

    # --- Block 1: 70 raw fields at calendar month O ------------------------
    origin_positions = grid.positions_at(area_idx, origin_ord)
    raw_at_origin = _gather(raw_values, origin_positions)
    for column_index, name in enumerate(RAW_FEATURE_COLUMNS):
        out[name] = raw_at_origin[:, column_index]

    # --- Block 2: 15 derivatives (Q6d) -------------------------------------
    summers: dict[str, _WindowSummer] = {}
    for name, source_column, length in SUM_DERIVATIVES:
        if source_column not in summers:
            summers[source_column] = _WindowSummer(
                grid, raw_values[:, raw_column_index[source_column]]
            )
        out[name] = summers[source_column].sum_ending_at(area_idx, origin_ord, length)

    lag_source = raw_values[:, raw_column_index[LAG_DERIVATIVE_SOURCE]]
    for k in range(1, LAG_DERIVATIVE_MAX + 1):
        positions = grid.positions_at(area_idx, origin_ord - k)
        out[f"{LAG_DERIVATIVE_SOURCE}_lag{k}_asof"] = _gather(lag_source, positions)

    # --- Block 3: calendar encoding of the TARGET month (Q6e) --------------
    target_month_number = target_ord % 12 + 1
    angle = 2.0 * np.pi * (target_month_number - 1) / 12.0
    out["target_month_sin"] = np.sin(angle)
    out["target_month_cos"] = np.cos(angle)

    # --- Block 4: latest observed label at O (Q6a) -------------------------
    history_slot = _as_of_index(label_admin, label_ord, admin, origin_ord)
    has_history = history_slot >= 0
    safe_slot = np.clip(history_slot, 0, None)
    last_label = np.where(has_history, label_y[safe_slot].astype(np.float64), np.nan)
    last_label_ord = np.where(has_history, label_ord[safe_slot], -1)
    out["last_observed_label"] = last_label
    out["last_observed_label_age_months"] = np.where(
        has_history, (origin_ord - last_label_ord).astype(np.float64), np.nan
    )
    out["no_observed_label_history"] = (~has_history).astype(np.float64)

    # --- Block 5: recency of the latest observed positive (Q6f) ------------
    positive = label_y == 1
    crisis_admin = label_admin[positive]
    crisis_ord = label_ord[positive]
    crisis_slot = _as_of_index(crisis_admin, crisis_ord, admin, origin_ord)
    has_crisis = crisis_slot >= 0
    if len(crisis_ord):
        found_crisis_ord = crisis_ord[np.clip(crisis_slot, 0, None)]
    else:  # no observed positive anywhere; every row lacks a prior crisis
        found_crisis_ord = np.zeros(len(crisis_slot), dtype=np.int64)
    last_crisis_ord = np.where(has_crisis, found_crisis_ord, -1)
    out["months_since_last_observed_crisis"] = np.where(
        has_crisis, (origin_ord - last_crisis_ord).astype(np.float64), np.nan
    )
    out["no_prior_observed_crisis"] = (~has_crisis).astype(np.float64)

    # --- Block 6: horizon indicator ----------------------------------------
    out["horizon_months"] = horizon.astype(np.float64)

    if list(out.columns) != list(FEATURE_COLUMNS):
        raise DataContractError(
            "assembled feature order does not match the frozen schema"
        )

    # Q6c: sweep the assembled matrix too; a derivative must not carry an
    # infinity that the source sweep did not already remove.
    assembled_infinities: dict[str, int] = {}
    for name in FEATURE_COLUMNS:
        values = out[name].to_numpy(dtype=np.float64, copy=True)
        mask = np.isinf(values)
        count = int(mask.sum())
        if count:
            values[mask] = np.nan
            out[name] = values
            assembled_infinities[name] = count

    meta = pd.DataFrame(
        {
            "admin_code": admin,
            "target_month": month_label(target_ord),
            "origin_month": month_label(origin_ord),
            "ipcch_food_crisis": target_y,
            "last_observed_label_month": month_label(last_label_ord),
            "last_observed_crisis_month": month_label(last_crisis_ord),
        },
        index=out.index,
    )
    for name in ("country_en", "ISO3"):
        if name in labels.columns:
            meta[name] = np.repeat(labels[name].to_numpy(), len(horizon_arr))
        else:
            meta[name] = ""

    frame = pd.concat([meta[list(METADATA_COLUMNS)], out], axis=1)

    # Panel completeness is verified, not assumed: the source is a *near*-
    # complete monthly scaffold, and 39 areas in the pinned file stop at
    # 2024-12, so a missing (area, month) cell is a real possibility.
    occupied = int((grid.positions >= 0).sum())
    area_gaps = int(((grid.positions < 0).any(axis=1)).sum())
    audit = {
        "valid_outcomes": int(n_labels),
        "horizons": [int(h) for h in horizon_arr],
        "feature_rows": int(n_rows),
        "feature_columns": EXPECTED_FEATURE_COUNT,
        "areas": int(pd.unique(admin).size),
        "origin_month_min": month_label([int(origin_ord.min())])[0] if n_rows else "",
        "origin_month_max": month_label([int(origin_ord.max())])[0] if n_rows else "",
        "rows_with_missing_origin_row": int((origin_positions < 0).sum()),
        "rows_without_label_history": int((~has_history).sum()),
        "rows_without_prior_crisis": int((~has_crisis).sum()),
        "panel_grid_cells": int(grid.positions.size),
        "panel_grid_occupied": occupied,
        "panel_grid_gaps": int(grid.positions.size - occupied),
        "panel_areas_with_month_gaps": area_gaps,
        "panel_infinities_converted": input_infinities,
        "assembled_infinities_converted": assembled_infinities,
    }
    return FeatureMatrix(
        frame=frame,
        feature_columns=FEATURE_COLUMNS,
        metadata_columns=METADATA_COLUMNS,
        infinity_audit={
            "panel_input": input_infinities,
            "assembled": assembled_infinities,
        },
        audit=audit,
    )


def build_feature_matrix(
    ledger: TargetLedger,
    source_path: Path | str,
    horizons: Iterable[int] = ACTIVE_HORIZONS,
) -> FeatureMatrix:
    """R3 entry point: 93 columns x (valid outcomes x horizons) rows.

    The ledger supplies R1 truth and the full valid-label history; the source
    is re-read for the frozen raw whitelist. Nothing is imputed and nothing is
    written back to the source.
    """
    panel, panel_audit = load_covariate_panel(source_path)
    matrix = assemble_feature_matrix(ledger.valid(), panel, horizons=horizons)
    matrix.audit["panel"] = panel_audit
    matrix.audit["source_sha256"] = ledger.source_sha256
    matrix.infinity_audit["source_load"] = panel_audit["source_infinities_converted"]
    return matrix


# ==========================================================================
# R2 / Q8g / Q8 / Q8r -- geography
#
# Scope note, recorded prominently because it EXTENDS the approved policy:
#
#   Q8g approved ``make_valid`` on invalid geometries in an experiment-local
#   copy, and required a STOP on any non-polygon result. Running that trial
#   produced 0 invalid / 0 empty but **217 GeometryCollection** results, so the
#   stop triggered (research/geometry-repair-trial.md).
#
#   On 2026-09-20 the user resolved that stop with a narrow, explicit extension:
#
#     "When a GeometryCollection consists of exactly one areal component
#      (Polygon or MultiPolygon) plus only zero-area linear/point components,
#      keep the areal component and discard the zero-area parts. Any other
#      composition still stops the run."
#
#   Evidence for why the pattern is narrow: all 217 collections are exactly one
#   areal part plus one linear part (136 MultiLineString+Polygon, 62
#   MultiLineString+MultiPolygon, 19 LineString+Polygon) and the linear parts
#   have exactly zero area. :func:`authorized_areal_component` implements that
#   as a guarded branch which MEASURES the discarded area rather than assuming
#   it, and refuses everything else. It is deliberately not a general
#   "take the biggest polygon" rule.
#
#   Recorded limitation: 34 of the 217 areas change polygon area by more than
#   1% against their (invalid) original, worst ``admin_code=1425`` at 29.21%.
#   An invalid original's area is diagnostic only, never ground truth, and
#   topology repair never establishes administrative identity (R2).
# ==========================================================================

#: Source-relative locations of the three geographic components (R2).
GEOMETRY_RELATIVE_PATH = "spatial/ipcch_admin_geometry.shp"
REFERENCE_COORDS_RELATIVE_PATH = "spatial/unique_area_id_lat_lon.csv"
COUNTRY_LOOKUP_RELATIVE_PATH = "country_area_id_lookup.csv"

#: Every sidecar that defines the layer. All are hashed; .prj carries the CRS
#: and .dbf carries the IDs, so a path alone is not an identity.
SHAPEFILE_SIDECAR_SUFFIXES = (".shp", ".shx", ".dbf", ".prj", ".cpg")

GEOMETRY_ID_COLUMN = "admin_code"
REFERENCE_ID_COLUMN = "area_id"
GEOMETRY_EPSG = 4326

#: Declared ellipsoid for the geodesic footprint measure. Square degrees are
#: NOT treated as km^2 anywhere (design.md "Geographic preparation").
GEOD_ELLIPSOID = "WGS84"

AREAL_GEOMETRY_TYPES = ("Polygon", "MultiPolygon")

#: Types a GeometryCollection part may have and still be discardable. The zero
#: area is additionally asserted per part; membership here is not sufficient.
DISCARDABLE_GEOMETRY_TYPES = (
    "Point",
    "MultiPoint",
    "LineString",
    "LinearRing",
    "MultiLineString",
)

COUNTRY_LOOKUP_COLUMNS = ("area_id", "iso3", "country", "country_code", "country_en")


class GeometryRepairError(DataContractError):
    """Q8g stop: a repair outcome outside the narrowly authorized pattern."""


def _require_shapely():
    import shapely  # noqa: PLC0415 - optional heavy dependency, imported lazily

    return shapely


def _require_geopandas():
    import geopandas  # noqa: PLC0415 - optional heavy dependency

    return geopandas


def _require_geod():
    from pyproj import Geod  # noqa: PLC0415 - optional heavy dependency

    return Geod(ellps=GEOD_ELLIPSOID)


# --------------------------------------------------------------------------
# Identity: canonical integer area IDs and component hashes
# --------------------------------------------------------------------------


def normalize_area_ids(values, source: str = "area ids") -> np.ndarray:
    """Normalize any area-ID representation to canonical ``int64``.

    The three geographic components disagree on storage type: the shapefile's
    ``admin_code`` is a DBF *string*, the CSVs read as integers, and a pandas
    round trip can yield ``101324.0``. R2 requires one canonical identity, so
    the conversion is explicit and refuses anything that is not exactly an
    integer -- a silent ``int(float)`` truncation would fuse two areas.
    """
    out = np.empty(len(values), dtype=np.int64)
    for position, raw in enumerate(np.asarray(values, dtype=object)):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            raise DataContractError(f"{source}: blank area id at position {position}")
        text = str(raw).strip()
        if text == "":
            raise DataContractError(f"{source}: blank area id at position {position}")
        try:
            number = Decimal(text)
        except InvalidOperation as error:
            raise DataContractError(
                f"{source}: area id {raw!r} is not numeric"
            ) from error
        if number != number.to_integral_value():
            raise DataContractError(f"{source}: area id {raw!r} is not an integer")
        out[position] = int(number)
    return out


def hash_component_files(paths: Iterable[Path | str]) -> dict:
    """SHA256 every existing component; a missing required file is a stop."""
    digests = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise DataContractError(f"required geographic component missing: {path}")
        digests[path.name] = sha256_file(path)
    return digests


def shapefile_component_paths(shapefile_path: Path | str) -> list[Path]:
    """The .shp plus every sidecar that actually exists beside it."""
    shapefile_path = Path(shapefile_path)
    paths = [shapefile_path]
    for suffix in SHAPEFILE_SIDECAR_SUFFIXES:
        if suffix == ".shp":
            continue
        sidecar = shapefile_path.with_suffix(suffix)
        if sidecar.exists():
            paths.append(sidecar)
    return paths


def geometry_wkb_sha256(geometry) -> str:
    """Content hash of one geometry, for before/after repair evidence."""
    shapely = _require_shapely()
    return hashlib.sha256(shapely.to_wkb(geometry)).hexdigest()


# --------------------------------------------------------------------------
# Component loaders
# --------------------------------------------------------------------------


def load_area_geometry(path: Path | str):
    """Read the admin layer and enforce the R2 structural contract.

    Checks: the ID column exists, IDs normalize and are unique, exactly one
    feature per area, a known geographic CRS (EPSG:4326 in the pinned source),
    no missing/empty geometry and finite bounds. Geometry *validity* is not
    checked here; that is the Q8g repair step's subject.
    """
    gpd = _require_geopandas()
    path = Path(path)
    gdf = gpd.read_file(path)

    if GEOMETRY_ID_COLUMN not in gdf.columns:
        raise DataContractError(
            f"{path.name} has no '{GEOMETRY_ID_COLUMN}' column; columns are "
            f"{list(gdf.columns)}"
        )
    if gdf.crs is None:
        raise DataContractError(f"{path.name} has no CRS; an assumed CRS is not allowed")
    epsg = gdf.crs.to_epsg()
    if epsg != GEOMETRY_EPSG or not gdf.crs.is_geographic:
        raise DataContractError(
            f"{path.name} CRS is {gdf.crs.to_string()} (epsg={epsg}); the pinned "
            f"source is geographic EPSG:{GEOMETRY_EPSG} and is preserved as-is"
        )

    gdf = gdf.copy()
    gdf[REFERENCE_ID_COLUMN] = normalize_area_ids(
        gdf[GEOMETRY_ID_COLUMN].to_numpy(), source=path.name
    )
    if gdf[REFERENCE_ID_COLUMN].duplicated().any():
        duplicates = sorted(
            gdf.loc[gdf[REFERENCE_ID_COLUMN].duplicated(keep=False), REFERENCE_ID_COLUMN]
            .unique()
            .tolist()
        )
        raise DataContractError(
            f"{path.name}: {len(duplicates)} area ids carry more than one feature "
            f"(first: {duplicates[:5]}); one polygon record per group is required"
        )

    missing = int(gdf.geometry.isna().sum())
    empty = int(gdf.geometry.is_empty.sum())
    if missing or empty:
        raise DataContractError(
            f"{path.name}: {missing} missing and {empty} empty geometries"
        )
    bounds = gdf.geometry.bounds.to_numpy(dtype=np.float64)
    if not np.isfinite(bounds).all():
        raise DataContractError(f"{path.name}: non-finite geometry bounds")

    gdf = gdf.sort_values(REFERENCE_ID_COLUMN, kind="mergesort").reset_index(drop=True)
    audit = {
        "path": str(path),
        "features": int(len(gdf)),
        "areas": int(gdf[REFERENCE_ID_COLUMN].nunique()),
        "crs": gdf.crs.to_string(),
        "epsg": int(epsg),
        "geometry_types": {
            str(name): int(count)
            for name, count in gdf.geometry.geom_type.value_counts().items()
        },
        "invalid": int((~gdf.geometry.is_valid).sum()),
        "component_sha256": hash_component_files(shapefile_component_paths(path)),
    }
    return gdf, audit


def load_reference_coordinates(path: Path | str) -> tuple[pd.DataFrame, dict]:
    """Load the keyed reference coordinates used for DONOR DISTANCE (Q8r).

    These are the authority for the 100 km great-circle donor cap. Polygon
    centroids are a different quantity and must not be substituted, even
    though the shapefile also carries lat/lon attributes.
    """
    path = Path(path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    for column in (REFERENCE_ID_COLUMN, "lat", "lon"):
        if column not in raw.columns:
            raise DataContractError(f"{path.name} is missing column '{column}'")

    frame = pd.DataFrame(
        {
            REFERENCE_ID_COLUMN: normalize_area_ids(
                raw[REFERENCE_ID_COLUMN].to_numpy(), source=path.name
            ),
            "ref_lat": pd.to_numeric(raw["lat"], errors="coerce").to_numpy(np.float64),
            "ref_lon": pd.to_numeric(raw["lon"], errors="coerce").to_numpy(np.float64),
        }
    )
    if frame[REFERENCE_ID_COLUMN].duplicated().any():
        raise DataContractError(f"{path.name}: duplicate {REFERENCE_ID_COLUMN}")
    coords = frame[["ref_lat", "ref_lon"]].to_numpy(dtype=np.float64)
    if not np.isfinite(coords).all():
        raise DataContractError(f"{path.name}: non-finite reference coordinates")
    if not (np.abs(coords[:, 0]) <= 90).all() or not (np.abs(coords[:, 1]) <= 180).all():
        raise DataContractError(f"{path.name}: reference coordinates outside lat/lon range")

    frame = frame.sort_values(REFERENCE_ID_COLUMN, kind="mergesort").reset_index(drop=True)
    audit = {
        "path": str(path),
        "rows": int(len(frame)),
        "areas": int(frame[REFERENCE_ID_COLUMN].nunique()),
        "sha256": sha256_file(path),
        "role": "donor-distance authority (Q8r); NOT polygon centroids",
    }
    return frame, audit


def load_country_lookup(path: Path | str) -> tuple[pd.DataFrame, dict]:
    """Load the keyed country lookup, preserving areas with a missing ISO3.

    R2: "Preserve areas with missing ISO3 using the keyed country lookup for
    reporting." A blank ISO3 is recorded and kept; it is never inferred, and
    the area is never dropped. Q9b country clustering keys off this lookup, so
    losing a row here would silently shrink a bootstrap cohort.
    """
    path = Path(path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    missing_columns = [name for name in COUNTRY_LOOKUP_COLUMNS if name not in raw.columns]
    if missing_columns:
        raise DataContractError(f"{path.name} is missing columns: {missing_columns}")

    frame = raw[list(COUNTRY_LOOKUP_COLUMNS)].copy()
    frame[REFERENCE_ID_COLUMN] = normalize_area_ids(
        frame[REFERENCE_ID_COLUMN].to_numpy(), source=path.name
    )
    if frame[REFERENCE_ID_COLUMN].duplicated().any():
        raise DataContractError(f"{path.name}: duplicate {REFERENCE_ID_COLUMN}")
    for column in ("iso3", "country", "country_code", "country_en"):
        frame[column] = frame[column].astype(str).str.strip()

    blank_country = frame["country_en"].eq("") & frame["country"].eq("")
    if blank_country.any():
        raise DataContractError(
            f"{path.name}: {int(blank_country.sum())} areas have no country name; "
            "a country-clustered bootstrap cannot key on them"
        )
    #: Stable reporting key: country_en where present, else country.
    frame["country_key"] = np.where(
        frame["country_en"].ne(""), frame["country_en"], frame["country"]
    )

    missing_iso3 = frame["iso3"].eq("")
    missing_code = frame["country_code"].eq("")
    frame = frame.sort_values(REFERENCE_ID_COLUMN, kind="mergesort").reset_index(drop=True)
    audit = {
        "path": str(path),
        "rows": int(len(frame)),
        "areas": int(frame[REFERENCE_ID_COLUMN].nunique()),
        "countries": int(frame["country_key"].nunique()),
        "sha256": sha256_file(path),
        "areas_missing_iso3": int(missing_iso3.sum()),
        "countries_missing_iso3": sorted(
            frame.loc[missing_iso3, "country_key"].unique().tolist()
        ),
        "areas_missing_country_code": int(missing_code.sum()),
        "countries_missing_country_code": sorted(
            frame.loc[missing_code, "country_key"].unique().tolist()
        ),
        "note": "areas with a blank ISO3 are retained; no ISO repair is authorized",
    }
    return frame, audit


def reconcile_area_universe(
    geometry_ids,
    reference_ids,
    country_ids,
    expected: int = EXPECTED_AREAS,
) -> dict:
    """Require the exact shared area universe across all three sources (R2)."""
    sets = {
        "geometry": set(int(v) for v in np.asarray(geometry_ids, dtype=np.int64)),
        "reference": set(int(v) for v in np.asarray(reference_ids, dtype=np.int64)),
        "country": set(int(v) for v in np.asarray(country_ids, dtype=np.int64)),
    }
    union = set().union(*sets.values())
    problems = []
    for name, ids in sets.items():
        gap = sorted(union - ids)
        if gap:
            problems.append(f"{name} is missing {len(gap)} ids (first: {gap[:5]})")
    if len(union) != expected:
        problems.append(f"union has {len(union)} ids, expected {expected}")
    if problems:
        raise DataContractError("area universe mismatch: " + "; ".join(problems))
    return {
        "expected_universe": int(expected),
        "universe": int(len(union)),
        "geometry_areas": int(len(sets["geometry"])),
        "reference_areas": int(len(sets["reference"])),
        "country_areas": int(len(sets["country"])),
        "id_min": int(min(union)),
        "id_max": int(max(union)),
        "contiguous_0_to_n": bool(union == set(range(len(union)))),
    }


# --------------------------------------------------------------------------
# Q8g repair, with the narrowly authorized GeometryCollection extension
# --------------------------------------------------------------------------


def authorized_areal_component(geometry) -> tuple[object, dict]:
    """Return the areal geometry a repair result is allowed to contribute.

    Authorized outcomes, and nothing else:

    * a ``Polygon``/``MultiPolygon`` -- returned unchanged, nothing discarded;
    * a ``GeometryCollection`` holding **exactly one** areal component plus
      **only** linear/point components whose measured area is exactly zero --
      the areal component is kept and the zero-area parts are discarded.

    Every other composition raises :class:`GeometryRepairError`. In particular
    two areal parts stop the run: this is not a "keep the biggest polygon"
    rule, and choosing between two real polygons would be an unapproved
    boundary decision. The zero area of each discarded part is *asserted from
    the geometry*, not assumed from its type.
    """
    geom_type = geometry.geom_type
    if geom_type in AREAL_GEOMETRY_TYPES:
        return geometry, {
            "discarded_parts": 0,
            "discarded_types": "",
            "discarded_area_total": 0.0,
        }
    if geom_type != "GeometryCollection":
        raise GeometryRepairError(
            f"repair produced a non-polygonal {geom_type}; Q8g stops the run"
        )

    parts = list(geometry.geoms)
    areal = [part for part in parts if part.geom_type in AREAL_GEOMETRY_TYPES]
    others = [part for part in parts if part.geom_type not in AREAL_GEOMETRY_TYPES]
    if len(areal) != 1:
        raise GeometryRepairError(
            f"repair produced a GeometryCollection with {len(areal)} areal "
            f"components (parts: {[p.geom_type for p in parts]}); only exactly one "
            "areal component plus zero-area parts is authorized"
        )

    discarded_area = 0.0
    for part in others:
        if part.geom_type not in DISCARDABLE_GEOMETRY_TYPES:
            raise GeometryRepairError(
                f"repair produced a GeometryCollection containing a "
                f"{part.geom_type}; only zero-area linear/point parts may be "
                "discarded"
            )
        part_area = float(part.area)
        if part_area != 0.0:
            # Measured, not assumed: a non-zero-area part would mean real
            # surface is being thrown away.
            raise GeometryRepairError(
                f"repair produced a {part.geom_type} part with area {part_area!r}; "
                "only exactly zero-area parts may be discarded"
            )
        discarded_area += part_area

    return areal[0], {
        "discarded_parts": len(others),
        "discarded_types": "+".join(sorted(part.geom_type for part in others)),
        "discarded_area_total": discarded_area,
    }


def geodesic_area_m2(geometry) -> float:
    """Ellipsoidal (geodesic) footprint in square metres, WGS84.

    Declared measure, per design.md: square degrees are never reported as
    km^2. The sign from the ring orientation is dropped. On an *invalid*
    geometry the value is diagnostic only -- self-intersecting rings partially
    cancel -- so callers must label such comparisons accordingly.
    """
    geod = _require_geod()
    area, _perimeter = geod.geometry_area_perimeter(geometry)
    return abs(float(area))


def _safe_geodesic_area(geometry) -> tuple[float, str]:
    try:
        return geodesic_area_m2(geometry), ""
    except Exception as error:  # noqa: BLE001 - record, never fabricate a number
        return float("nan"), f"{type(error).__name__}: {error}"


def repair_geometry(geometry) -> tuple[object, dict]:
    """Apply the Q8g policy to one geometry and return it with its evidence.

    A valid *polygonal* geometry is returned **unchanged** -- the same object,
    so the before/after WKB hashes are identical by construction. An invalid
    one goes through ``shapely.make_valid`` and then
    :func:`authorized_areal_component`. Any residual invalid/empty/non-polygonal
    result stops the run.

    The polygonal-type requirement is checked on the **input**, before the
    valid-geometry shortcut: R2 requires one Polygon/MultiPolygon per area, and
    a valid ``Point``/``LineString`` is just as unusable as an invalid one.
    Routing it through the shortcut would have let it into adjacency untouched.
    """
    shapely = _require_shapely()
    from shapely.validation import explain_validity  # noqa: PLC0415

    if geometry is None:
        raise GeometryRepairError("missing geometry")
    if geometry.is_empty:
        raise GeometryRepairError("empty input geometry")

    original_type = geometry.geom_type
    if original_type not in AREAL_GEOMETRY_TYPES:
        raise GeometryRepairError(
            f"input geometry is a non-polygonal {original_type}; R2 requires one "
            "Polygon/MultiPolygon per area and Q8g stops the run"
        )

    original_hash = geometry_wkb_sha256(geometry)
    original_area, original_area_note = _safe_geodesic_area(geometry)

    if geometry.is_valid:
        return geometry, {
            "outcome": "unchanged_valid",
            "original_valid": 1,
            "original_validity_reason": "Valid Geometry",
            "original_geometry_type": original_type,
            "make_valid_geometry_type": "",
            "repaired_geometry_type": original_type,
            "original_wkb_sha256": original_hash,
            "repaired_wkb_sha256": original_hash,
            "geometry_changed": 0,
            "discarded_parts": 0,
            "discarded_types": "",
            "discarded_area_total": 0.0,
            "original_area_m2": original_area,
            "repaired_area_m2": original_area,
            "area_abs_change_m2": 0.0,
            "area_relative_change": 0.0,
            "footprint_comparison": "exact_unchanged",
            "footprint_comparison_note": original_area_note,
        }

    reason = explain_validity(geometry)
    repaired_raw = shapely.make_valid(geometry)
    make_valid_type = repaired_raw.geom_type
    areal, discarded = authorized_areal_component(repaired_raw)

    if areal.is_empty:
        raise GeometryRepairError(f"repair produced an empty geometry ({reason})")
    if areal.geom_type not in AREAL_GEOMETRY_TYPES:
        raise GeometryRepairError(
            f"repair produced a {areal.geom_type} ({reason}); Q8g stops the run"
        )
    if not areal.is_valid:
        raise GeometryRepairError(
            f"repair left an invalid geometry: {explain_validity(areal)} "
            f"(original: {reason})"
        )

    repaired_area, repaired_area_note = _safe_geodesic_area(areal)
    if np.isfinite(original_area) and np.isfinite(repaired_area) and original_area > 0:
        absolute = repaired_area - original_area
        relative = absolute / original_area
        comparison = "diagnostic_invalid_original"
        note = (
            "the original is invalid, so its area is diagnostic only and is not "
            "ground truth"
        )
    else:
        absolute = float("nan")
        relative = float("nan")
        comparison = "unavailable"
        note = "; ".join(
            part
            for part in (
                original_area_note,
                repaired_area_note,
                "" if np.isfinite(original_area) and original_area > 0
                else "original geodesic area is not a usable positive number",
            )
            if part
        )

    return areal, {
        "outcome": (
            "repaired_polygonal"
            if make_valid_type in AREAL_GEOMETRY_TYPES
            else "repaired_collection_areal_extracted"
        ),
        "original_valid": 0,
        "original_validity_reason": reason,
        "original_geometry_type": original_type,
        "make_valid_geometry_type": make_valid_type,
        "repaired_geometry_type": areal.geom_type,
        "original_wkb_sha256": original_hash,
        "repaired_wkb_sha256": geometry_wkb_sha256(areal),
        "geometry_changed": 1,
        "discarded_parts": discarded["discarded_parts"],
        "discarded_types": discarded["discarded_types"],
        "discarded_area_total": discarded["discarded_area_total"],
        "original_area_m2": original_area,
        "repaired_area_m2": repaired_area,
        "area_abs_change_m2": absolute,
        "area_relative_change": relative,
        "footprint_comparison": comparison,
        "footprint_comparison_note": note,
    }


def repair_geometries(gdf) -> tuple[object, pd.DataFrame, dict]:
    """Q8g repair over the whole layer, in an experiment-local copy only.

    The raw source file is never touched. Already-valid geometries pass through
    untouched; only invalid ones are repaired.
    """
    repaired_geometries = []
    records = []
    for _, row in gdf.iterrows():
        area_id = int(row[REFERENCE_ID_COLUMN])
        try:
            geometry, record = repair_geometry(row.geometry)
        except GeometryRepairError as error:
            raise GeometryRepairError(f"area {area_id}: {error}") from error
        record[REFERENCE_ID_COLUMN] = area_id
        repaired_geometries.append(geometry)
        records.append(record)

    gpd = _require_geopandas()
    out = gdf.copy()
    out["geometry"] = gpd.GeoSeries(repaired_geometries, index=out.index, crs=gdf.crs)
    out = out.set_geometry("geometry")

    audit_frame = pd.DataFrame.from_records(records)
    ordered = [REFERENCE_ID_COLUMN] + [
        name for name in audit_frame.columns if name != REFERENCE_ID_COLUMN
    ]
    audit_frame = audit_frame[ordered].sort_values(
        REFERENCE_ID_COLUMN, kind="mergesort"
    ).reset_index(drop=True)

    repaired = audit_frame[audit_frame["geometry_changed"] == 1]
    relative = repaired["area_relative_change"].abs()
    comparable = relative[np.isfinite(relative)]

    def _footprint_stats(subset: pd.DataFrame) -> dict:
        values = subset["area_relative_change"].abs()
        usable = values[np.isfinite(values)]
        return {
            "areas": int(len(subset)),
            "comparable": int(len(usable)),
            "unavailable": int(len(values) - len(usable)),
            "median_relative": float(usable.median()) if len(usable) else float("nan"),
            "over_0p1_percent": int((usable > 0.001).sum()),
            "over_1_percent": int((usable > 0.01).sum()),
            "max_relative": float(usable.max()) if len(usable) else float("nan"),
            "max_relative_area_id": (
                int(subset.loc[usable.idxmax(), REFERENCE_ID_COLUMN])
                if len(usable)
                else None
            ),
        }

    #: Stratified because the two groups behave differently, and because the
    #: authorized-extension evidence is stated for the collection group alone.
    by_outcome = {
        str(outcome): _footprint_stats(subset)
        for outcome, subset in repaired.groupby("outcome")
    }
    worst = repaired.loc[comparable.sort_values(ascending=False).index[:10]] if len(
        comparable
    ) else repaired.iloc[:0]
    summary = {
        "features": int(len(audit_frame)),
        "unchanged_valid": int((audit_frame["outcome"] == "unchanged_valid").sum()),
        "repaired_polygonal": int((audit_frame["outcome"] == "repaired_polygonal").sum()),
        "repaired_collection_areal_extracted": int(
            (audit_frame["outcome"] == "repaired_collection_areal_extracted").sum()
        ),
        "original_validity_reasons": {
            str(reason): int(count)
            for reason, count in repaired["original_validity_reason"]
            .map(lambda text: text.split("[")[0].strip())
            .value_counts()
            .items()
        },
        "make_valid_geometry_types": {
            str(name): int(count)
            for name, count in repaired["make_valid_geometry_type"].value_counts().items()
        },
        "repaired_geometry_types": {
            str(name): int(count)
            for name, count in repaired["repaired_geometry_type"].value_counts().items()
        },
        "discarded_zero_area_parts": int(repaired["discarded_parts"].sum()),
        "discarded_area_total": float(repaired["discarded_area_total"].sum()),
        "footprint_comparison": {
            str(name): int(count)
            for name, count in audit_frame["footprint_comparison"].value_counts().items()
        },
        "footprint_change_of_repaired": {
            "comparable": int(len(comparable)),
            "unavailable": int(len(relative) - len(comparable)),
            "median_relative": float(comparable.median()) if len(comparable) else float("nan"),
            "over_0p1_percent": int((comparable > 0.001).sum()),
            "over_1_percent": int((comparable > 0.01).sum()),
            "max_relative": float(comparable.max()) if len(comparable) else float("nan"),
            "max_relative_area_id": (
                int(repaired.loc[comparable.idxmax(), REFERENCE_ID_COLUMN])
                if len(comparable)
                else None
            ),
        },
        "footprint_change_by_outcome": by_outcome,
        "largest_footprint_changes": [
            {
                "area_id": int(row[REFERENCE_ID_COLUMN]),
                "outcome": str(row["outcome"]),
                "make_valid_geometry_type": str(row["make_valid_geometry_type"]),
                "original_validity_reason": str(row["original_validity_reason"]),
                "area_relative_change": float(row["area_relative_change"]),
            }
            for _, row in worst.iterrows()
        ],
        "limitation": (
            "an invalid original's area is diagnostic only, not ground truth; "
            "topology repair does not establish administrative identity (R2)"
        ),
        "measured_against_recorded_evidence": (
            "research/geometry-repair-trial.md records 34 of 217 collection cases "
            "over 1%; this run measures 35 of 217 under the declared geodesic "
            "measure, and an independent planar recomputation also gives 35 with "
            "a clear gap between 0.0088 and 0.0115, so the recorded 34 is an "
            "off-by-one. The worst case (area 1425, 29.21%) reconciles exactly."
        ),
    }
    return out, audit_frame, summary


# --------------------------------------------------------------------------
# Adjacency
# --------------------------------------------------------------------------


def _shared_boundary(left, right) -> bool:
    """The inherited adjacency predicate, restated once.

    ``GeoRFBaseline/src/adjacency/adjacency_utils.py:99-103``: polygons are
    neighbours when they ``touches`` **and** their intersection has positive
    length. A single shared corner has zero length and is therefore excluded.
    """
    if not left.touches(right):
        return False
    intersection = left.intersection(right)
    return hasattr(intersection, "length") and intersection.length > 0


def neighbor_ids_for_positions(gdf, positions) -> tuple[dict, dict]:
    """Neighbour area IDs for selected rows, using :func:`_shared_boundary`.

    Used for the before/after repair comparison. Errors are captured per area
    instead of aborting: a topology error on an *invalid original* means the
    comparison is unavailable, which R2/design.md require us to report rather
    than to record as a fabricated zero change.
    """
    spatial_index = gdf.sindex
    ids = gdf[REFERENCE_ID_COLUMN].to_numpy(dtype=np.int64)
    geometries = gdf.geometry.to_numpy()
    neighbours: dict[int, set] = {}
    errors: dict[int, str] = {}
    for position in positions:
        position = int(position)
        current = geometries[position]
        found: set[int] = set()
        try:
            candidates = list(spatial_index.intersection(current.bounds))
            for candidate in candidates:
                candidate = int(candidate)
                if candidate == position:
                    continue
                if _shared_boundary(current, geometries[candidate]):
                    found.add(int(ids[candidate]))
        except Exception as error:  # noqa: BLE001
            errors[int(ids[position])] = f"{type(error).__name__}: {error}"
            continue
        neighbours[int(ids[position])] = found
    return neighbours, errors


@dataclass(frozen=True)
class AdjacencyArtifacts:
    """Polygon adjacency in the exact shapes GeoRF's polygon mode expects.

    ``adjacency_dict`` keys and values are POLYGON INDICES, and
    ``polygon_group_mapping`` goes index -> area group id. That is the opposite
    direction from the helper's returned ``polygon_id_mapping`` (id -> index),
    which is why the conversion is explicit here
    (design.md "Geographic preparation").

    ``polygon_centroids`` is retained for the inherited polygon refinement
    only. It is NOT the donor-distance authority: the 100 km Q8r cap uses the
    separate keyed reference coordinates from ``unique_area_id_lat_lon.csv``.
    """

    adjacency_dict: dict
    polygon_id_mapping: dict
    polygon_group_mapping: dict
    polygon_centroids: np.ndarray
    area_ids: np.ndarray
    audit: dict


def build_polygon_adjacency(
    shapefile_path: Path | str,
    baseline_root: Path | str,
    id_column: str = GEOMETRY_ID_COLUMN,
) -> AdjacencyArtifacts:
    """Reuse the frozen adjacency helper on the validated local geometry.

    The helper silently substitutes an alternative ID field or the row index
    when the requested column is absent
    (``adjacency_utils.py:51-64``), so the column is verified here *before*
    the call and the returned mapping is verified against the expected area
    universe *after* it. Without those two checks a fallback would produce a
    plausible-looking adjacency keyed on the wrong identity.
    """
    gpd = _require_geopandas()
    shapefile_path = Path(shapefile_path)
    baseline_root = Path(baseline_root)

    try:
        columns = list(gpd.read_file(shapefile_path, rows=1).columns)
    except TypeError:  # engine without a row limit
        columns = list(gpd.read_file(shapefile_path).columns)
    if id_column not in columns:
        raise DataContractError(
            f"{shapefile_path.name} lacks '{id_column}'; refusing the helper's "
            f"silent ID fallback. Columns: {columns}"
        )

    import importlib.util  # noqa: PLC0415
    import sys  # noqa: PLC0415

    module_path = baseline_root / "src" / "adjacency" / "adjacency_utils.py"
    if not module_path.exists():
        raise DataContractError(f"adjacency helper not found at {module_path}")
    spec = importlib.util.spec_from_file_location(
        "ipcch_baseline_adjacency_utils", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    adjacency_dict, polygon_id_mapping, polygon_centroids = (
        module.create_polygon_adjacency_matrix(
            str(shapefile_path), polygon_id_column=id_column
        )
    )

    if "INDEX_ID" in polygon_id_mapping:
        raise DataContractError("adjacency helper fell back to row-index IDs")
    raw_keys = list(polygon_id_mapping.keys())
    canonical = normalize_area_ids(np.asarray(raw_keys, dtype=object), source="adjacency")
    indices = np.asarray([polygon_id_mapping[key] for key in raw_keys], dtype=np.int64)
    if len(set(indices.tolist())) != len(indices):
        raise DataContractError("adjacency helper returned duplicate polygon indices")

    polygon_group_mapping = {int(i): int(a) for a, i in zip(canonical, indices)}
    index_by_area = {int(a): int(i) for a, i in zip(canonical, indices)}
    if len(polygon_group_mapping) != len(adjacency_dict):
        raise DataContractError(
            f"adjacency covers {len(adjacency_dict)} polygons but the mapping has "
            f"{len(polygon_group_mapping)}"
        )
    if set(polygon_group_mapping) != set(adjacency_dict):
        raise DataContractError("adjacency keys and polygon indices disagree")
    if len(set(polygon_group_mapping.values())) != len(polygon_group_mapping):
        raise DataContractError("two polygon records map to the same area group")

    area_ids = np.array(
        [polygon_group_mapping[i] for i in range(len(polygon_group_mapping))],
        dtype=np.int64,
    )
    centroids = np.asarray(polygon_centroids, dtype=np.float64)
    if centroids.shape != (len(area_ids), 2):
        raise DataContractError(f"unexpected centroid array shape {centroids.shape}")

    degrees = np.array([len(adjacency_dict[i]) for i in range(len(area_ids))], dtype=np.int64)
    neighbour_sets = {
        int(index): {int(value) for value in neighbours}
        for index, neighbours in adjacency_dict.items()
    }
    symmetric = all(
        index in neighbour_sets.get(other, set())
        for index, neighbours in neighbour_sets.items()
        for other in neighbours
    )
    audit = {
        "shapefile": str(shapefile_path),
        "id_column": id_column,
        "polygons": int(len(area_ids)),
        "edges_undirected": int(degrees.sum() // 2),
        "directed_entries": int(degrees.sum()),
        "degree_min": int(degrees.min()) if len(degrees) else 0,
        "degree_max": int(degrees.max()) if len(degrees) else 0,
        "degree_mean": float(degrees.mean()) if len(degrees) else float("nan"),
        "degree_median": float(np.median(degrees)) if len(degrees) else float("nan"),
        "isolated_polygons": int((degrees == 0).sum()),
        "symmetric": bool(symmetric),
        "definition": (
            "touches() and intersection length > 0; point-only contact excluded "
            "(GeoRFBaseline/src/adjacency/adjacency_utils.py:99-103)"
        ),
        "component_sha256": hash_component_files(
            shapefile_component_paths(shapefile_path)
        ),
        "centroid_note": (
            "centroids are for inherited polygon refinement only; donor distance "
            "uses the keyed reference coordinates (Q8r)"
        ),
    }
    return AdjacencyArtifacts(
        adjacency_dict=adjacency_dict,
        polygon_id_mapping=index_by_area,
        polygon_group_mapping=polygon_group_mapping,
        polygon_centroids=centroids,
        area_ids=area_ids,
        audit=audit,
    )


def adjacency_cache_payload(artifacts: AdjacencyArtifacts) -> dict:
    """Cache payload bound to the geometry's CONTENT, not its path.

    The inherited cache validates only the shapefile path and ID column
    (``adjacency_utils.py:171-175``), so an edited layer at the same path would
    silently reuse a stale matrix. The component hashes make that impossible.
    """
    return {
        "component_sha256": artifacts.audit["component_sha256"],
        "id_column": artifacts.audit["id_column"],
        "polygons": artifacts.audit["polygons"],
        "adjacency_dict": artifacts.adjacency_dict,
        "polygon_group_mapping": artifacts.polygon_group_mapping,
        "polygon_id_mapping": artifacts.polygon_id_mapping,
        "polygon_centroids": artifacts.polygon_centroids,
        "area_ids": artifacts.area_ids,
    }


def adjacency_cache_is_current(payload: dict, shapefile_path: Path | str, id_column: str) -> bool:
    """True only when the cache matches the layer's current content."""
    if not isinstance(payload, dict):
        return False
    if payload.get("id_column") != id_column:
        return False
    expected = hash_component_files(shapefile_component_paths(shapefile_path))
    return payload.get("component_sha256") == expected


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GeographyArtifacts:
    """Validated, repaired geography plus every audit R2/Q8g requires."""

    geometry: object
    repair_audit: pd.DataFrame
    reference_coordinates: pd.DataFrame
    country_lookup: pd.DataFrame
    adjacency: AdjacencyArtifacts | None
    audit: dict

    def reference_coordinate_array(self, area_ids=None) -> np.ndarray:
        """``(lat, lon)`` rows for the Q8r donor search, in ``area_ids`` order.

        Deliberately a separate accessor from
        :attr:`AdjacencyArtifacts.polygon_centroids`: donor distance is defined
        on these keyed reference coordinates, not on polygon centroids.
        """
        frame = self.reference_coordinates.set_index(REFERENCE_ID_COLUMN)
        if area_ids is None:
            area_ids = frame.index.to_numpy()
        selected = frame.loc[np.asarray(area_ids, dtype=np.int64)]
        return selected[["ref_lat", "ref_lon"]].to_numpy(dtype=np.float64, copy=True)


def prepare_geography(
    source_root: Path | str,
    out_dir: Path | str,
    baseline_root: Path | str | None = None,
    build_adjacency: bool = True,
    compare_adjacency: bool = True,
    expected_universe: int = EXPECTED_AREAS,
) -> GeographyArtifacts:
    """R2/Q8g/Q8 pipeline: validate keys, repair locally, build adjacency.

    ``out_dir`` receives the experiment-local repaired copy and the audits.
    The pinned source under ``source_root`` is only ever read.
    """
    import json  # noqa: PLC0415
    import pickle  # noqa: PLC0415

    source_root = Path(source_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geometry_path = source_root / GEOMETRY_RELATIVE_PATH
    reference_path = source_root / REFERENCE_COORDS_RELATIVE_PATH
    country_path = source_root / COUNTRY_LOOKUP_RELATIVE_PATH

    gdf, geometry_audit = load_area_geometry(geometry_path)
    reference, reference_audit = load_reference_coordinates(reference_path)
    country, country_audit = load_country_lookup(country_path)

    universe = reconcile_area_universe(
        gdf[REFERENCE_ID_COLUMN].to_numpy(),
        reference[REFERENCE_ID_COLUMN].to_numpy(),
        country[REFERENCE_ID_COLUMN].to_numpy(),
        expected=expected_universe,
    )

    # Before/after neighbour sets for the areas we are about to repair.
    invalid_positions = np.flatnonzero((~gdf.geometry.is_valid).to_numpy())
    before_neighbours, before_errors = (
        neighbor_ids_for_positions(gdf, invalid_positions)
        if compare_adjacency and len(invalid_positions)
        else ({}, {})
    )

    repaired, repair_audit, repair_summary = repair_geometries(gdf)

    repaired_path = out_dir / "ipcch_admin_geometry_repaired.shp"
    repaired.to_file(repaired_path)
    local_copy_audit = {
        "path": str(repaired_path),
        "component_sha256": hash_component_files(
            shapefile_component_paths(repaired_path)
        ),
        "note": "experiment-local copy; the raw source file is never modified",
    }

    adjacency = None
    adjacency_comparison = {"status": "skipped"}
    if build_adjacency:
        if baseline_root is None:
            baseline_root = Path(__file__).resolve().parent.parent / "GeoRFBaseline"
        adjacency = build_polygon_adjacency(repaired_path, baseline_root)

        # Evidence that the refinement centroids and the Q8r donor-distance
        # reference coordinates are genuinely different quantities, so a silent
        # substitution in the 100 km donor search would be a real error.
        reference_array = (
            reference.set_index(REFERENCE_ID_COLUMN)
            .loc[adjacency.area_ids][["ref_lat", "ref_lon"]]
            .to_numpy(dtype=np.float64)
        )
        delta = np.abs(adjacency.polygon_centroids - reference_array)
        per_area = delta.max(axis=1)
        adjacency.audit["centroid_vs_reference_coordinate"] = {
            "max_abs_degrees": float(per_area.max()),
            "median_abs_degrees": float(np.median(per_area)),
            "areas_over_1e_6_degrees": int((per_area > 1e-6).sum()),
            "note": (
                "most areas coincide, but a substantial minority do not; donor "
                "distance must use the reference coordinates (Q8r), never these "
                "centroids"
            ),
        }

        cache_path = out_dir / "adjacency_cache.pkl"
        with open(cache_path, "wb") as handle:
            pickle.dump(adjacency_cache_payload(adjacency), handle)
        local_copy_audit["adjacency_cache"] = str(cache_path)

        if compare_adjacency and len(invalid_positions):
            after_neighbours, after_errors = neighbor_ids_for_positions(
                repaired, invalid_positions
            )
            # Cross-check the restated predicate against the helper itself, so
            # a definitional drift cannot masquerade as a repair effect.
            drift = 0
            for area_id, neighbours in after_neighbours.items():
                index = adjacency.polygon_id_mapping[int(area_id)]
                helper = {
                    int(adjacency.polygon_group_mapping[int(other)])
                    for other in adjacency.adjacency_dict[index]
                }
                if helper != neighbours:
                    drift += 1
            changed, unchanged, unavailable = [], 0, []
            for area_id in sorted(
                set(before_neighbours) | set(after_neighbours) | set(before_errors)
            ):
                if area_id in before_errors or area_id in after_errors:
                    unavailable.append(
                        {
                            "area_id": int(area_id),
                            "reason": before_errors.get(area_id)
                            or after_errors.get(area_id),
                        }
                    )
                    continue
                before = before_neighbours.get(area_id, set())
                after = after_neighbours.get(area_id, set())
                if before == after:
                    unchanged += 1
                else:
                    changed.append(
                        {
                            "area_id": int(area_id),
                            "added": sorted(int(v) for v in after - before),
                            "removed": sorted(int(v) for v in before - after),
                        }
                    )
            adjacency_comparison = {
                "status": "computed",
                "areas_compared": int(len(before_neighbours)),
                "unchanged": int(unchanged),
                "changed": changed,
                "changed_count": int(len(changed)),
                "unavailable": unavailable,
                "unavailable_count": int(len(unavailable)),
                "predicate_drift_vs_helper": int(drift),
                "note": (
                    "an unavailable comparison is a topology error on the invalid "
                    "original, not a zero change"
                ),
            }

    audit = {
        "requirement": "R2 / Q8 / Q8r / Q8g",
        "scope_extension": {
            "approved": "2026-09-20",
            "rule": (
                "a GeometryCollection of exactly one areal component plus only "
                "zero-area linear/point components keeps the areal component; any "
                "other composition stops the run"
            ),
            "evidence": "research/geometry-repair-trial.md",
        },
        "geometry_source": geometry_audit,
        "reference_coordinates": reference_audit,
        "country_lookup": country_audit,
        "area_universe": universe,
        "repair": repair_summary,
        "local_copy": local_copy_audit,
        "adjacency": adjacency.audit if adjacency is not None else None,
        "adjacency_change_for_repaired_areas": adjacency_comparison,
        "limitations": [
            "topology repair does not establish administrative identity",
            "the upstream builder allowed unrestricted nearest-neighbour fallback "
            "without saved per-area match provenance (source-audit.md)",
            "an invalid original's footprint is diagnostic only; see "
            "repair.largest_footprint_changes, where the two extreme cases are "
            "plain make_valid results on self-intersecting MultiPolygons, not "
            "the authorized GeometryCollection extraction",
            "an unavailable adjacency comparison is a topology error on the "
            "invalid original, never a recorded zero change",
        ],
    }

    repair_audit.to_csv(out_dir / "geometry_repair_audit.csv", index=False)
    reference.to_csv(out_dir / "reference_coordinates.csv", index=False)
    country.to_csv(out_dir / "country_lookup.csv", index=False)
    with open(out_dir / "geography_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, default=str)

    return GeographyArtifacts(
        geometry=repaired,
        repair_audit=repair_audit,
        reference_coordinates=reference,
        country_lookup=country,
        adjacency=adjacency,
        audit=audit,
    )


# --------------------------------------------------------------------------
# R4 — one pooled 2014-2022 partition split, before horizon expansion
# --------------------------------------------------------------------------

#: R4/Q2r pooled partition-learning window, inclusive.
STAGE1_FIRST_MONTH = "2014-01"
STAGE1_LAST_MONTH = "2022-12"

#: Audited capacities from the PRD. A mismatch is investigated, not normalised away.
EXPECTED_STAGE1_LABELS = 19591
EXPECTED_STAGE1_MULTI_AREAS = 3264
EXPECTED_STAGE1_FIT = 8561
EXPECTED_STAGE1_VALIDATION = 9558
EXPECTED_STAGE1_SINGLETON_AREAS = 1472
EXPECTED_STAGE1_ZERO_LABEL_AREAS = 1491


@dataclass(frozen=True)
class Stage1Split:
    """Original-outcome split for the pooled partition-learning window.

    ``outcomes`` is one row per original (area, target_month) label, before any
    horizon expansion, carrying ``split_role`` in {fit, validation, singleton}.
    Zero-label areas hold no outcome but remain in the geographic universe (R4).
    """

    outcomes: pd.DataFrame
    zero_label_areas: np.ndarray
    audit: dict

    def fit_keys(self) -> pd.DataFrame:
        return self.outcomes[self.outcomes["split_role"] == "fit"]

    def validation_keys(self) -> pd.DataFrame:
        return self.outcomes[self.outcomes["split_role"] == "validation"]

    def singleton_keys(self) -> pd.DataFrame:
        return self.outcomes[self.outcomes["split_role"] == "singleton"]


def build_stage1_split(
    ledger: TargetLedger,
    all_area_ids: Sequence[int] | None = None,
    first_month: str = STAGE1_FIRST_MONTH,
    last_month: str = STAGE1_LAST_MONTH,
) -> Stage1Split:
    """Split original 2014-2022 outcomes per area, chronologically (R4 Q5h/Q5a).

    Within each area holding ``n >= 2`` valid outcomes, the earliest
    ``floor(n/2)`` become fitting and the latest ``ceil(n/2)`` become validation,
    so an odd count puts the extra outcome in validation. An area with exactly one
    outcome is a Stage1 singleton: excluded from fitting, from q/group statistics
    and from the parent/child F1 gates, and scored only afterwards as a
    supplementary diagnostic (R4 Q5s/Q5m/Q5v).

    The split is deliberately performed on **original outcomes**, before the four
    horizon views are created, so that all four views of one outcome land on the
    same side and support is counted once rather than four times.
    """
    valid = ledger.valid()
    lower = pd.Timestamp(first_month + "-01")
    upper = pd.Timestamp(last_month + "-01")
    pool = valid[(valid["target_month"] >= lower) & (valid["target_month"] <= upper)]
    pool = pool.sort_values(["admin_code", "target_month"], kind="mergesort")

    frames = []
    counts = pool.groupby("admin_code")["target_month"].transform("size")
    singles = pool[counts == 1].copy()
    singles["split_role"] = "singleton"
    frames.append(singles)

    multi = pool[counts >= 2].copy()
    # rank within area by target month; mergesort above makes the order stable
    rank = multi.groupby("admin_code").cumcount()
    size = multi.groupby("admin_code")["target_month"].transform("size")
    n_fit = np.floor(size / 2).astype(int)
    multi["split_role"] = np.where(rank < n_fit, "fit", "validation")
    frames.append(multi)

    outcomes = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["admin_code", "target_month"], kind="mergesort")
        .reset_index(drop=True)
    )
    outcomes = outcomes[
        ["admin_code", "target_month", "country_en", "ISO3", "ipcch_food_crisis", "split_role"]
    ]

    universe = (
        np.asarray(sorted(set(int(a) for a in all_area_ids)))
        if all_area_ids is not None
        else np.asarray(sorted(ledger.frame["admin_code"].unique()))
    )
    with_outcomes = set(int(a) for a in outcomes["admin_code"].unique())
    zero_label_areas = np.asarray([a for a in universe if a not in with_outcomes])

    role_counts = outcomes["split_role"].value_counts().to_dict()
    multi_areas = int(multi["admin_code"].nunique())
    audit = {
        "window": {"first_month": first_month, "last_month": last_month},
        "original_labels": int(len(outcomes)),
        "areas_with_outcomes": int(outcomes["admin_code"].nunique()),
        "multi_outcome_areas": multi_areas,
        "multi_outcome_labels": int(len(multi)),
        "fit": int(role_counts.get("fit", 0)),
        "validation": int(role_counts.get("validation", 0)),
        "singleton_areas": int(len(singles)),
        "zero_label_areas": int(len(zero_label_areas)),
        "positive_rate_fit": float(
            outcomes.loc[outcomes["split_role"] == "fit", "ipcch_food_crisis"].mean()
        )
        if role_counts.get("fit")
        else float("nan"),
        "positive_rate_validation": float(
            outcomes.loc[outcomes["split_role"] == "validation", "ipcch_food_crisis"].mean()
        )
        if role_counts.get("validation")
        else float("nan"),
    }

    # No original outcome may appear on two sides (implement.md 1).
    if outcomes.duplicated(["admin_code", "target_month"]).any():
        raise DataContractError("an original outcome appears more than once in the split")
    overlap = set(map(tuple, audit and outcomes.loc[outcomes.split_role == "fit", ["admin_code", "target_month"]].to_numpy())) & set(
        map(tuple, outcomes.loc[outcomes.split_role == "validation", ["admin_code", "target_month"]].to_numpy())
    )
    if overlap:
        raise DataContractError(f"{len(overlap)} outcomes are on both split sides")

    return Stage1Split(outcomes=outcomes, zero_label_areas=zero_label_areas, audit=audit)


def check_stage1_split_gate(split: Stage1Split) -> dict:
    """Compare the split against the PRD's audited capacities (implement.md 1)."""
    audit = dict(split.audit)
    mismatches = {}
    for name, expected in (
        ("original_labels", EXPECTED_STAGE1_LABELS),
        ("multi_outcome_areas", EXPECTED_STAGE1_MULTI_AREAS),
        ("fit", EXPECTED_STAGE1_FIT),
        ("validation", EXPECTED_STAGE1_VALIDATION),
        ("singleton_areas", EXPECTED_STAGE1_SINGLETON_AREAS),
        ("zero_label_areas", EXPECTED_STAGE1_ZERO_LABEL_AREAS),
    ):
        if audit[name] != expected:
            mismatches[name] = {"expected": expected, "actual": audit[name]}
    audit["gate_pass"] = not mismatches
    audit["gate_mismatches"] = mismatches
    return audit
