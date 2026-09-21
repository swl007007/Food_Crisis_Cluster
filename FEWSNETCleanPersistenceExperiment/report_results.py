"""Recipe selection, final adjudication and robustness evidence (D22, D39-D45).

This module never fits a model, refits a calibrator or reselects a threshold. It reads
frozen Stage 3 predictions plus the frozen calibrators and thresholds, recomputes every
reported number from stored rows, and emits the tables and intervals the contracts
require.

Contract anchors:

* D22    - recipe score: mean over six (horizon, variant) cells of
           F1_2020(corrected) - F1_2020(persistence); pooled TP/FP/FN per cell, never a
           mean of per-month or per-area F1; identical evaluation keys across candidates.
* D36    - every paired metric uses the same labeled-target keys with a valid
           exact-origin persistence record within each horizon.
* D40    - horizon-specific final windows (fs1 2021-06, fs2 2021-10, fs3 2022-02;
           all ending 2024-10) plus a supplementary 2022-02..2024-10 common-calendar view.
* D41    - primary gain: the same equal-weight six-cell mean on the final windows, for
           the frozen winner; the reference arm and winner-minus-reference are secondary.
* D42    - +0.01 is an advisory reference, not a pass/fail cutoff. Signed values, full
           precision, and a negative difference stays negative.
* D43    - robust requires (1) a positive point estimate whose two-sided 95% interval
           has a lower endpoint strictly above zero, and (2) every leave-one-target-
           year-out recomputation strictly positive.
* D44    - one shared joint block bootstrap over the union of D40's target dates:
           2,000 valid draws, ``default_rng(5)``, multiplicities shared by every
           horizon/variant/comparator, linear-interpolated 2.5/97.5 percentiles.
* D45    - complete-pass / complete-fail / incomplete; incomplete is never a null.

Usage:

    python FEWSNETCleanPersistenceExperiment/report_results.py --run-dir RUN --verify
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "PersistenceCorrectionExperiment"))

import prepare_data as pdata  # noqa: E402
import run_pipeline as rp  # noqa: E402

HORIZONS: Tuple[int, ...] = pdata.HORIZONS
VARIANTS: Tuple[str, ...] = rp.PROBABILITY_VARIANTS
#: D44: 2,000 valid draws, seed 5, at most 20,000 attempts.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 5
BOOTSTRAP_MAX_ATTEMPTS = 20000
#: D42: advisory only. Never used as a gate.
ADVISORY_REFERENCE = 0.01
#: D40 supplementary common-calendar view.
COMMON_CALENDAR_START = pdata.SUPPLEMENTARY_COMMON_START
#: D43.2: exactly these four exclusions, always, regardless of which years the loaded
#: frame happens to contain.
LEAVE_ONE_YEAR_OUT_YEARS: Tuple[int, ...] = (2021, 2022, 2023, 2024)


def scheduled_target_dates() -> List[str]:
    """D44's fixed date universe: the sorted union of D40's scheduled target months.

    Derived from the frozen schedule, never from whichever dates happen to have
    support, so a fold with no eligible rows cannot silently shrink the bootstrap's
    resampling universe.
    """
    dates = {
        "%04d-%02d" % pair
        for horizon in HORIZONS
        for pair in pdata.final_target_dates(horizon)
    }
    return sorted(dates)


class ReportError(RuntimeError):
    """Missing or invalid required evidence. Under D45 this is *incomplete*, never a null."""


# --------------------------------------------------------------------------------------
# Pooled confusion counts and F1
# --------------------------------------------------------------------------------------


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "n": int(y_true.size),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
    }


def f1_from_counts(counts: Dict[str, int]) -> float:
    """Class-1 F1 from pooled counts, with the established F1=0 zero-division rule.

    D22/D41 require pooling TP/FP/FN across months *before* dividing. Averaging
    per-month or per-area F1 values gives a different number and is prohibited.
    """
    denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    return 0.0 if denominator == 0 else 2.0 * counts["tp"] / denominator


def apply_correction(
    persistence: np.ndarray, probability: np.ndarray, tau: Optional[float]
) -> np.ndarray:
    """D37's up-only correction, or unchanged persistence for a no-correction outcome."""
    persistence = np.asarray(persistence).astype(int)
    if tau is None:
        return persistence.copy()
    from persistencecorrection.override import apply_override  # noqa: PLC0415

    return apply_override(persistence, np.asarray(probability, dtype=float), float(tau))


# --------------------------------------------------------------------------------------
# Evidence loading
# --------------------------------------------------------------------------------------


def _verified_frame(fold_dir: Path, fold: "rp.Stage3Fold") -> Tuple[pd.DataFrame, str]:
    """Load one fold and check its *contents* against its declared identity.

    Hashing only proves the file has not changed since it was written. It does not
    prove the file holds the fold it claims to: a directory named for a 2020 selection
    fold could contain 2024 rows. That boundary is exactly the one preventing
    final-period labels from entering a development decision, so it is checked here.
    """
    path = fold_dir / "predictions.csv"
    if not path.is_file():
        raise ReportError(f"missing predictions: {path}")
    digest = rp.sha256_file(path)
    recorded = rp.read_json(fold_dir / "fold.json").get("predictions_sha256")
    if digest != recorded:
        raise ReportError(f"{path} was modified after its fold completed")

    frame = pd.read_csv(path)
    expected_month = f"{fold.year:04d}-{fold.month:02d}"
    problems = []
    for column, expected in (
        ("arm", fold.arm), ("map_role", fold.role),
        ("target_month", expected_month),
        ("horizon_months", fold.horizon),
        ("forecasting_scope", fold.scope),
    ):
        if column not in frame.columns:
            problems.append(f"missing column {column!r}")
            continue
        actual = set(frame[column].unique().tolist())
        if actual != {expected}:
            problems.append(f"{column}: expected {{{expected!r}}}, found {sorted(actual)[:4]}")
    duplicates = int(frame.duplicated(subset=["FEWSNET_admin_code", "target_month"]).sum())
    if duplicates:
        problems.append(f"{duplicates} duplicate (area, target_month) keys")
    if problems:
        raise ReportError(f"{path} does not match its declared identity: {problems}")
    return frame, digest


def load_scored_rows(
    run_dir: Path, arm: str, role: str, *, calibrators_arm: Optional[str] = None
) -> pd.DataFrame:
    """One arm/role's paired evaluation rows with both probability variants attached.

    Restricted to D36's valid exact-origin persistence support: that is the common
    paired cohort every method in a horizon must share.
    """
    context = rp.RunContext(run_dir, create=False)
    frames: List[pd.DataFrame] = []
    identities: Dict[str, str] = {}
    folds = rp.final_folds(arm) if role == "final" else rp.development_folds(arm)
    for fold in folds:
        if fold.role != role:
            continue
        loaded, digest = _verified_frame(context.root / "stage3" / arm / fold.name, fold)
        identities[fold.name] = digest
        frames.append(loaded)
    if not frames:
        raise ReportError(f"no {role} folds for arm {arm!r}")
    frame = pd.concat(frames, ignore_index=True)

    frame = frame[frame["persistence_available"]].copy()
    if frame.empty:
        raise ReportError(f"{arm}/{role}: no rows with valid exact-origin persistence")
    frame["persistence"] = frame["persistence"].astype(int)
    frame["calendar_month"] = pd.to_datetime(frame["target_month"] + "-01").dt.month

    frame["p_raw"] = frame["rf_prob_class1"].astype(float)
    calibrated = np.full(len(frame), np.nan)
    source_arm = calibrators_arm or arm
    for horizon in HORIZONS:
        mask = (frame["horizon_months"] == horizon).to_numpy()
        if not mask.any():
            continue
        calibrators = rp.load_calibrators(context, source_arm, horizon)
        values, _, _ = calibrators.transform(
            frame.loc[mask, "p_raw"].to_numpy(dtype=float),
            frame.loc[mask, "calendar_month"].to_numpy(),
            frame.loc[mask, "partition_id"].to_numpy().astype(int),
        )
        calibrated[mask] = values
    if np.isnan(calibrated).any():
        raise ReportError(f"{arm}/{role}: calibration left rows untransformed")
    frame["p_calibrated"] = calibrated
    frame.attrs["prediction_identities"] = identities
    return frame


def load_thresholds(
    run_dir: Path, arm: str, *, predictions: Optional[pd.DataFrame] = None
) -> Dict[str, object]:
    """Load the frozen thresholds and *enforce* the lineage they recorded.

    Recording input hashes is only half the protection. A Stage 3 rerun produces new
    predictions with new, internally consistent digests; without this check reporting
    would happily score them against thresholds selected from the old ones.
    """
    path = run_dir / "thresholds" / arm / "frozen_thresholds.json"
    if not path.is_file():
        raise ReportError(f"no frozen thresholds for {arm!r}: {path}")
    payload = rp.read_json(path)

    # Bindings are required, not optional. Defaulting a missing binding to {} and then
    # skipping the comparison turns "no lineage recorded" into "lineage verified".
    recorded_calibrators = payload.get("input_calibrators_sha256")
    if not recorded_calibrators:
        raise ReportError(
            f"{arm}: frozen thresholds record no calibrator identities; the lineage "
            "R42 requires is absent and cannot be verified"
        )
    expected_keys = {f"h{horizon}" for horizon in HORIZONS}
    if set(recorded_calibrators) != expected_keys:
        raise ReportError(
            f"{arm}: threshold freeze binds calibrators {sorted(recorded_calibrators)}, "
            f"expected {sorted(expected_keys)}"
        )
    for key, expected in recorded_calibrators.items():
        calibrator_path = run_dir / "calibration" / arm / f"calibrators_{key}.json"
        if not calibrator_path.is_file():
            raise ReportError(f"{arm}: calibrator {key} recorded in the freeze is missing")
        if rp.sha256_file(calibrator_path) != expected:
            raise ReportError(
                f"{arm}: calibrator {key} no longer matches the identity the frozen "
                "thresholds were selected against"
            )

    recorded = payload.get("input_predictions_sha256")
    if not recorded:
        raise ReportError(
            f"{arm}: frozen thresholds record no selection-prediction identities"
        )
    # A nonempty mapping is not a complete one. Checking only the entries the freeze
    # happens to declare would let a freeze that recorded 1 of 9 folds verify cleanly,
    # so the expected fold set is derived independently and must match exactly.
    expected_folds = {
        fold.name for fold in rp.development_folds(arm) if fold.role == "selection"
    }
    if set(recorded) != expected_folds:
        raise ReportError(
            f"{arm}: threshold freeze binds {len(recorded)} selection predictions, "
            f"expected all {len(expected_folds)}; missing "
            f"{sorted(expected_folds - set(recorded))}, unexpected "
            f"{sorted(set(recorded) - expected_folds)}"
        )

    # The selection lineage is checked against the *selection* predictions, always.
    # Final-window predictions legitimately differ and are never compared here.
    # Existence is checked before hashing so a missing bound file surfaces as a
    # ReportError rather than a raw FileNotFoundError from the hash call.
    missing = [name for name in recorded
               if not (run_dir / "stage3" / arm / name / "predictions.csv").is_file()]
    if missing:
        raise ReportError(
            f"{arm}: selection predictions recorded in the threshold freeze are missing "
            f"({missing[:5]})"
        )
    actual = {
        name: rp.sha256_file(run_dir / "stage3" / arm / name / "predictions.csv")
        for name in recorded
    }
    if actual != recorded:
        differing = sorted(name for name in recorded if actual.get(name) != recorded[name])
        raise ReportError(
            f"{arm}: selection predictions differ from those the thresholds were frozen "
            f"against ({differing[:5]}). An authorized repair must rebuild the whole "
            "chain, not re-report old thresholds against new predictions."
        )
    if predictions is not None:
        supplied = predictions.attrs.get("prediction_identities")
        if supplied is not None and supplied != recorded:
            differing = sorted(
                name for name in set(recorded) | set(supplied)
                if recorded.get(name) != supplied.get(name)
            )
            raise ReportError(
                f"{arm}: supplied selection rows do not carry the frozen identities "
                f"({differing[:5]})"
            )
    return payload["thresholds"]


# --------------------------------------------------------------------------------------
# D22 recipe score
# --------------------------------------------------------------------------------------


def six_cell_gains(
    frame: pd.DataFrame, thresholds: Dict[str, object]
) -> Tuple[Dict[str, Dict[str, object]], float]:
    """The six (horizon, variant) gains over persistence and their equal mean.

    Shared by D22's development score and D41's final adjudication: identical structure,
    different evaluation window. Every horizon and variant carries equal weight, so a
    horizon with more rows cannot dominate.
    """
    cells: Dict[str, Dict[str, object]] = {}
    gains: List[float] = []
    for horizon in HORIZONS:
        block = frame[frame["horizon_months"] == horizon]
        if block.empty:
            raise ReportError(f"horizon {horizon} has no paired support; cell required")
        truth = block["target_label"].to_numpy().astype(int)
        persistence = block["persistence"].to_numpy().astype(int)
        base_counts = confusion(truth, persistence)
        base_f1 = f1_from_counts(base_counts)
        for variant in VARIANTS:
            key = f"{variant}_h{horizon}"
            if key not in thresholds:
                raise ReportError(f"missing frozen threshold cell {key}")
            tau = thresholds[key]["tau"]
            corrected = apply_correction(
                persistence, block[f"p_{variant}"].to_numpy(dtype=float), tau
            )
            counts = confusion(truth, corrected)
            value = f1_from_counts(counts)
            gains.append(value - base_f1)
            cells[key] = {
                "horizon_months": horizon, "variant": variant, "tau": tau,
                "no_correction": tau is None,
                "rows": int(len(block)),
                "persistence_f1": base_f1, "corrected_f1": value,
                "gain": value - base_f1,
                "flips_0_to_1": int(((persistence == 0) & (corrected == 1)).sum()),
                "persistence_counts": base_counts, "corrected_counts": counts,
            }
    if len(gains) != 6:
        raise ReportError(f"expected six cells, computed {len(gains)}")
    return cells, float(np.mean(gains))


def _normalise_cohort(block: pd.DataFrame) -> pd.DataFrame:
    """The single canonical form for a cohort, used on both sides of the comparison."""
    cohort = block[[
        "FEWSNET_admin_code", "target_month", "target_label", "persistence",
    ]].copy()
    cohort["FEWSNET_admin_code"] = cohort["FEWSNET_admin_code"].astype("int64")
    cohort["target_month"] = cohort["target_month"].astype(str)
    cohort["target_label"] = cohort["target_label"].astype("int64")
    cohort["persistence"] = cohort["persistence"].astype("int64")
    return (
        cohort.sort_values(["FEWSNET_admin_code", "target_month"])
        .reset_index(drop=True)
    )


def evaluation_cohort(frame: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Per-horizon evaluation cohort: keys *and* the truth/persistence they carry.

    A frozenset of keys is not enough. Two candidates can hold identical key sets while
    one has duplicated a positive row or flipped a label, and D22's score would move
    without the check noticing. The comparator values travel with the keys.
    """
    out: Dict[int, pd.DataFrame] = {}
    for horizon, block in frame.groupby("horizon_months"):
        # Cast exactly as expected_selection_cohort does. A float 1.0 and an int 1
        # serialise differently and would produce different signatures for identical
        # data, so the normalisation has to be shared, not merely similar.
        cohort = _normalise_cohort(block)
        duplicates = int(cohort.duplicated(
            subset=["FEWSNET_admin_code", "target_month"]
        ).sum())
        if duplicates:
            raise ReportError(
                f"horizon {int(horizon)}: {duplicates} duplicate evaluation keys; D22 "
                "requires a unique predeclared cohort"
            )
        out[int(horizon)] = cohort.reset_index(drop=True)
    return out


def _cohort_signature(cohort: pd.DataFrame) -> str:
    return rp.sha256_bytes(
        cohort.to_csv(index=False).encode("utf-8")
    )


def expected_selection_cohort(run_dir: Path, role: str) -> Dict[int, pd.DataFrame]:
    """The predeclared paired cohort, derived from the bound prepared master.

    Comparing candidates only against each other establishes that they agree, not that
    they are right: every arm could omit the same difficult areas, or share a wrong
    label, and pass. D22's "predeclared" keys have to come from outside the candidates,
    so they are rebuilt here from the prepared sources and D36's exact-origin
    persistence rule.
    """
    context = rp.RunContext(run_dir, create=False)
    sources = pdata.load_prepared_sources(rp._prepared_cache(context))
    targets = pdata.MAP_ROLES[role]["prediction_targets"]
    if not targets:
        raise ReportError(f"map role {role!r} declares no prediction targets")

    out: Dict[int, pd.DataFrame] = {}
    for scope, horizon in enumerate(pdata.HORIZONS, start=1):
        del scope
        frames: List[pd.DataFrame] = []
        for year, month in targets:
            _, target_rows, _ = rp.select_stage3_rows(sources, horizon, int(year), int(month))
            frames.append(pdata.build_row_metadata(sources, horizon, target_rows))
        block = pd.concat(frames, ignore_index=True)
        block = block[block["persistence_available"]]
        out[int(horizon)] = _normalise_cohort(block)
    return out


def select_recipe(
    run_dir: Path, arms: Sequence[str], *, require_full_inventory: bool = True
) -> Dict[str, object]:
    """D22: score every candidate recipe on the 2020 selection window and rank.

    An authoritative selection must run the whole approved inventory. Scoring a subset
    would let a winner be declared without the competition D22 prescribes, so a partial
    run is explicitly marked diagnostic and cannot be frozen.
    """
    required = [name for name, _ in pdata.RECIPE_MANIFEST]
    if require_full_inventory:
        duplicates = sorted({arm for arm in arms if list(arms).count(arm) > 1})
        if duplicates:
            raise ReportError(
                f"duplicate candidate arms {duplicates}; an authoritative D22 "
                "selection scores each recipe exactly once"
            )
        missing = [arm for arm in required if arm not in arms]
        extra = [arm for arm in arms if arm not in required]
        if missing or extra:
            raise ReportError(
                "an authoritative D22 selection must score exactly the 12 approved "
                f"updated recipes; missing {missing}, unexpected {extra}. The "
                "original-feature reference is a separate comparator, not a candidate. "
                "Pass require_full_inventory=False for a diagnostic ranking."
            )

    scores: List[Dict[str, object]] = []
    cohorts_by_arm: Dict[str, Dict[int, pd.DataFrame]] = {}
    for arm in arms:
        frame = load_scored_rows(run_dir, arm, "selection")
        cells, score = six_cell_gains(
            frame, load_thresholds(run_dir, arm, predictions=frame)
        )
        cohorts_by_arm[arm] = evaluation_cohort(frame)
        scores.append({
            "arm": arm, "score": score,
            "feature_columns": len(pdata.recipe_columns(arm))
            if arm != pdata.REFERENCE_ARM else len(pdata.reference_columns()),
            "manifest_position": _manifest_position(arm),
            "cells": {key: cell["gain"] for key, cell in cells.items()},
            "cell_detail": cells,
        })

    # D22: "all candidates, both variants and persistence must use identical
    # predeclared evaluation keys; a candidate cannot improve its score by dropping
    # hard rows." The whole cohort - keys, labels and persistence - must match, not just
    # the key set, and every horizon must be present for every arm.
    anchor = expected_selection_cohort(run_dir, "selection")
    signatures = {
        arm: {horizon: _cohort_signature(cohort) for horizon, cohort in cohorts.items()}
        for arm, cohorts in cohorts_by_arm.items()
    }
    anchor_signature = {
        horizon: _cohort_signature(cohort) for horizon, cohort in anchor.items()
    }
    mismatches: Dict[str, object] = {}
    for arm, table in signatures.items():
        if set(table) != set(anchor_signature):
            mismatches[arm] = {
                "missing_horizons": sorted(set(anchor_signature) - set(table)),
                "unexpected_horizons": sorted(set(table) - set(anchor_signature)),
            }
            continue
        differing = [
            horizon for horizon, value in table.items()
            if value != anchor_signature[horizon]
        ]
        if differing:
            mismatches[arm] = {
                "horizons_with_different_cohort": sorted(differing),
                "rows_here": {
                    str(horizon): int(len(cohorts_by_arm[arm][horizon]))
                    for horizon in sorted(differing)
                },
                "rows_in_anchor": {
                    str(horizon): int(len(anchor[horizon]))
                    for horizon in sorted(differing)
                },
            }
    if mismatches:
        raise ReportError(
            "evaluation cohorts do not match the predeclared cohort derived from the "
            f"prepared master, which D22 requires: {mismatches}"
        )

    # Greatest score; exact ties prefer fewer feature columns, then the earlier
    # manifest position.
    ranked = sorted(
        scores,
        key=lambda item: (-item["score"], item["feature_columns"], item["manifest_position"]),
    )
    best = ranked[0]["score"]
    tied = [item["arm"] for item in ranked if item["score"] == best]
    return {
        "rule": "D22: mean of six (horizon, variant) F1 gains over persistence on 2020",
        "authoritative": bool(require_full_inventory),
        "required_inventory": required,
        "scored_arms": list(arms),
        "evaluation_cohort_identical_across_arms": True,
        "evaluation_cohort_sha256": {
            str(horizon): value for horizon, value in anchor_signature.items()
        },
        "candidates": len(scores),
        "winner": ranked[0]["arm"],
        "winner_score": best,
        "tied_on_score": tied,
        "tie_break_applied": len(tied) > 1,
        "ranking": [
            {
                "arm": item["arm"], "score": item["score"],
                "feature_columns": item["feature_columns"],
                "cells": item["cells"],
            }
            for item in ranked
        ],
        "cell_detail": {item["arm"]: item["cell_detail"] for item in ranked},
    }


def _manifest_position(arm: str) -> int:
    for position, (name, _) in enumerate(pdata.RECIPE_MANIFEST):
        if name == arm:
            return position
    return len(pdata.RECIPE_MANIFEST)


# --------------------------------------------------------------------------------------
# D44 joint block bootstrap
# --------------------------------------------------------------------------------------


def _cell_counts_by_date(
    frame: pd.DataFrame, thresholds: Dict[str, object]
) -> Dict[Tuple[str, str], Dict[str, Dict[str, int]]]:
    """Per target date, the pooled counts for persistence and each corrected variant.

    Aggregating once here is what lets a bootstrap draw reweight whole date blocks
    without recomputing predictions, exactly as D44 requires.
    """
    out: Dict[Tuple[str, str], Dict[str, Dict[str, int]]] = {}
    for horizon in HORIZONS:
        block = frame[frame["horizon_months"] == horizon]
        for date, rows in block.groupby("target_month"):
            truth = rows["target_label"].to_numpy().astype(int)
            persistence = rows["persistence"].to_numpy().astype(int)
            entry = {"persistence": confusion(truth, persistence)}
            for variant in VARIANTS:
                tau = thresholds[f"{variant}_h{horizon}"]["tau"]
                corrected = apply_correction(
                    persistence, rows[f"p_{variant}"].to_numpy(dtype=float), tau
                )
                entry[variant] = confusion(truth, corrected)
            out[(str(horizon), str(date))] = entry
    return out


def _weighted_gain(
    counts: Dict[Tuple[str, str], Dict[str, Dict[str, int]]],
    multiplicity: Dict[str, int],
) -> Optional[float]:
    """One draw's primary gain, or None when a required horizon drew zero observations."""
    gains: List[float] = []
    for horizon in HORIZONS:
        totals = {
            method: {"n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
            for method in ("persistence", *VARIANTS)
        }
        for (cell_horizon, date), entry in counts.items():
            if cell_horizon != str(horizon):
                continue
            weight = multiplicity.get(date, 0)
            if weight == 0:
                continue
            for method, table in entry.items():
                for field in totals[method]:
                    totals[method][field] += weight * table[field]
        if totals["persistence"]["n"] == 0:
            return None  # undefined draw: record and redraw, never score as zero
        base = f1_from_counts(totals["persistence"])
        for variant in VARIANTS:
            gains.append(f1_from_counts(totals[variant]) - base)
    return float(np.mean(gains)) if len(gains) == 6 else None


def joint_bootstrap(
    frame: pd.DataFrame, thresholds: Dict[str, object],
    target_dates: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """D44: one shared date-block draw reused by every horizon, variant and comparator.

    The date universe is D40's fixed schedule, not the dates with observed support. A
    horizon whose window legitimately excludes a date contributes nothing for that
    block, which is the structural absence D44 says to preserve rather than invent.
    """
    counts = _cell_counts_by_date(frame, thresholds)
    dates = sorted(set(target_dates)) if target_dates is not None else scheduled_target_dates()
    expected = scheduled_target_dates()
    if dates != expected:
        raise ReportError(
            f"D44 requires the {len(expected)} scheduled target dates "
            f"{expected}; got {len(dates)}: {dates}"
        )
    # Each horizon must also stay inside its OWN D40 window. The global union alone
    # would accept an fs3 row dated 2021-06, three quarters before its window opens.
    violations: Dict[str, List[str]] = {}
    for horizon in HORIZONS:
        allowed = {"%04d-%02d" % pair for pair in pdata.final_target_dates(horizon)}
        outside = sorted(
            date for cell_horizon, date in counts
            if cell_horizon == str(horizon) and date not in allowed
        )
        if outside:
            violations[f"h{horizon}"] = outside
    if violations:
        raise ReportError(
            f"predictions contain target dates outside their horizon's D40 window: "
            f"{violations}"
        )
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    gains: List[float] = []
    attempts = 0
    rejected = 0
    ledger: List[Dict[str, object]] = []
    while len(gains) < BOOTSTRAP_DRAWS and attempts < BOOTSTRAP_MAX_ATTEMPTS:
        attempts += 1
        drawn = rng.choice(len(dates), size=len(dates), replace=True)
        multiplicity: Dict[str, int] = {}
        for index in drawn:
            key = dates[index]
            multiplicity[key] = multiplicity.get(key, 0) + 1
        value = _weighted_gain(counts, multiplicity)
        # Rejection is driven solely by an empty required horizon. Gain sign and size
        # never influence acceptance.
        ledger.append({
            "attempt": attempts,
            "multiplicity": {date: multiplicity.get(date, 0) for date in dates},
            "accepted": value is not None,
            "rejection_reason": None if value is not None else "empty_required_horizon",
            "primary_gain": value,
        })
        if value is None:
            rejected += 1
            continue
        gains.append(value)

    complete = len(gains) == BOOTSTRAP_DRAWS
    interval = (
        [float(np.percentile(gains, 2.5, method="linear")),
         float(np.percentile(gains, 97.5, method="linear"))]
        if complete else None
    )
    return {
        "protocol": "D44 shared joint date-block bootstrap",
        "seed": BOOTSTRAP_SEED,
        "numpy_version": np.__version__,
        "generator": "numpy.random.default_rng",
        "target_dates": dates,
        "n_target_dates": len(dates),
        "requested_draws": BOOTSTRAP_DRAWS,
        "valid_draws": len(gains),
        "attempts": attempts,
        "rejected_undefined_draws": rejected,
        "rejection_rate": rejected / attempts if attempts else None,
        "complete": complete,
        "interval_95": interval,
        "percentile_method": "linear",
        # Retained even when the run is incomplete: partial evidence is what makes an
        # attempt-limit failure diagnosable, and the multiplicity ledger is what lets
        # the interval be recomputed independently.
        "gains": gains,
        "draw_ledger": ledger,
    }


# --------------------------------------------------------------------------------------
# D43 robustness
# --------------------------------------------------------------------------------------


def leave_one_year_out(
    frame: pd.DataFrame, thresholds: Dict[str, object]
) -> Dict[str, object]:
    """D43.2: drop each target year from every horizon/variant/comparator together."""
    frame = frame.copy()
    frame["target_year"] = frame["target_month"].str.slice(0, 4).astype(int)
    results: Dict[str, object] = {}
    # All four exclusions are required, not "whichever years happen to be present".
    # Iterating observed years would silently turn a frame missing 2021 into three
    # passing checks. Excluding 2021 legitimately removes no fs3 rows (its window
    # starts 2022), but the recomputation must still be performed and reported.
    for year in LEAVE_ONE_YEAR_OUT_YEARS:
        remaining = frame[frame["target_year"] != year]
        present = int((frame["target_year"] == year).sum())
        try:
            _, mean = six_cell_gains(remaining, thresholds)
            results[str(year)] = {
                "primary_gain": mean, "rows": int(len(remaining)),
                "rows_excluded": present,
            }
        except ReportError as error:
            results[str(year)] = {
                "primary_gain": None, "rows_excluded": present,
                "incomplete": str(error),
            }
    values = [entry["primary_gain"] for entry in results.values()]
    return {
        "rule": "D43.2: every leave-one-target-year-out mean strictly positive",
        "required_years": list(LEAVE_ONE_YEAR_OUT_YEARS),
        "by_excluded_year": results,
        "all_defined": bool(
            len(results) == len(LEAVE_ONE_YEAR_OUT_YEARS)
            and all(value is not None and bool(np.isfinite(value)) for value in values)
        ),
        "all_strictly_positive": bool(
            len(results) == len(LEAVE_ONE_YEAR_OUT_YEARS)
            and all(value is not None and bool(np.isfinite(value)) and value > 0
                    for value in values)
        ),
    }


def adjudicate(
    point_estimate: float, bootstrap: Dict[str, object], loyo: Dict[str, object]
) -> Dict[str, object]:
    """D43 conditions and the D45 status they imply.

    Completion flags are not trusted on their own: a NaN point estimate or a NaN
    interval endpoint is invalid evidence, which D45 classes as *incomplete*. Treating
    it as a completed negative (or, worse, a pass) would misreport a broken computation
    as a scientific result.
    """
    interval = bootstrap.get("interval_95")
    # bool(...) throughout: np.isfinite returns np.bool_, which is not JSON
    # serialisable and would fail only at report-writing time.
    interval_valid = bool(
        isinstance(interval, (list, tuple)) and len(interval) == 2
        and all(isinstance(value, (int, float)) and bool(np.isfinite(value))
                for value in interval)
        and interval[0] <= interval[1]
    )
    point_valid = bool(
        isinstance(point_estimate, (int, float)) and bool(np.isfinite(point_estimate))
    )
    # Recompute from the four required results rather than trusting the summary flags.
    # Trusting them let a dictionary missing 2024, an empty dictionary, and four
    # gains of -0.1 all reach complete_pass as long as the flags said so.
    by_year = loyo.get("by_excluded_year") or {}
    loyo_values = [
        (by_year.get(str(year)) or {}).get("primary_gain")
        for year in LEAVE_ONE_YEAR_OUT_YEARS
    ]
    loyo_valid = bool(
        set(by_year) == {str(year) for year in LEAVE_ONE_YEAR_OUT_YEARS}
        and all(value is not None and bool(np.isfinite(value)) for value in loyo_values)
    )

    condition_1 = bool(point_valid and interval_valid and point_estimate > 0
                       and interval[0] > 0)
    condition_2 = bool(loyo_valid and all(value > 0 for value in loyo_values))
    complete = bool(
        bootstrap.get("complete") and interval_valid and point_valid and loyo_valid
    )
    if not complete:
        status = "incomplete"
        interpretation = (
            "Required interval or leave-one-year-out evidence is missing or invalid. "
            "Under D45 this is NOT a scientific null: resolve the specific failure."
        )
    elif condition_1 and condition_2:
        status = "complete_pass"
        interpretation = (
            "Robust positive aggregate benefit under the stated retrospective, source "
            "and resampling assumptions. +0.01 remains advisory, not a cutoff."
        )
    else:
        status = "complete_fail"
        interpretation = (
            "Insufficient evidence of a robust aggregate benefit. This is not proof of "
            "zero effect."
        )
    return {
        "point_estimate": point_estimate,
        "point_estimate_valid": point_valid,
        "interval_95": interval,
        "interval_valid": interval_valid,
        "leave_one_year_out_valid": loyo_valid,
        "condition_1_positive_with_interval_above_zero": condition_1,
        "condition_2_all_leave_one_year_out_positive": condition_2,
        "advisory_reference": ADVISORY_REFERENCE,
        "meets_advisory_reference": point_estimate > ADVISORY_REFERENCE,
        "advisory_note": (
            "D42: +0.01 is descriptive. A smaller gain is not automatically a failure "
            "and a larger one is not automatically robust."
        ),
        "d45_status": status,
        "interpretation": interpretation,
    }


# --------------------------------------------------------------------------------------
# D40/D41/D45 final report
# --------------------------------------------------------------------------------------


def _printable(payload):
    """Drop the bulky arrays so the console summary stays readable."""
    if isinstance(payload, dict):
        return {
            key: _printable(value) for key, value in payload.items()
            if key not in ("draw_ledger", "gains", "cell_detail", "ranking")
        }
    if isinstance(payload, list):
        return [_printable(item) for item in payload]
    return payload


def paired_method_table(
    frame: pd.DataFrame, thresholds: Dict[str, object]
) -> Dict[str, object]:
    """D39's paired rows per horizon: persistence, pooled RF, partitioned RF and the
    two correction variants, all on D36's shared keys.

    Both RF rows use their *native* classifier prediction. D39 forbids giving them a
    separately tuned classification threshold or a standalone-calibration arm, so the
    correction rows are the only place a threshold appears.
    """
    out: Dict[str, object] = {}
    for horizon in HORIZONS:
        block = frame[frame["horizon_months"] == horizon]
        if block.empty:
            raise ReportError(f"horizon {horizon} has no paired support")
        truth = block["target_label"].to_numpy().astype(int)
        persistence = block["persistence"].to_numpy().astype(int)
        rows: Dict[str, Dict[str, object]] = {
            "persistence": {
                "source": "D36 valid exact-origin observed crisis label",
                **_scored(truth, persistence),
            },
            "pooled_rf": {
                "source": "released pooled RF native hard prediction",
                **_scored(truth, block["pooled_hard_prediction"].to_numpy().astype(int)),
            },
            "partitioned_rf": {
                "source": "released partitioned RF native hard prediction",
                "route_counts": {
                    str(key): int(value)
                    for key, value in block["model_route"].value_counts().to_dict().items()
                },
                **_scored(truth, block["rf_hard_prediction"].to_numpy().astype(int)),
            },
        }
        for variant in VARIANTS:
            tau = thresholds[f"{variant}_h{horizon}"]["tau"]
            corrected = apply_correction(
                persistence, block[f"p_{variant}"].to_numpy(dtype=float), tau
            )
            rows[f"{variant}_correction"] = {
                "source": (
                    f"D37 override on partitioned-RF {variant} probability, "
                    f"tau={tau!r}"
                ),
                "tau": tau,
                "no_correction_outcome": tau is None,
                "identical_to_persistence": bool(np.array_equal(corrected, persistence)),
                **_scored(truth, corrected),
            }
        out[f"h{horizon}"] = {"paired_rows": int(len(block)), "methods": rows}
    return out


def _scored(truth: np.ndarray, predicted: np.ndarray) -> Dict[str, object]:
    counts = confusion(truth, predicted)
    precision = (
        counts["tp"] / (counts["tp"] + counts["fp"])
        if counts["tp"] + counts["fp"] else 0.0
    )
    recall = (
        counts["tp"] / (counts["tp"] + counts["fn"])
        if counts["tp"] + counts["fn"] else 0.0
    )
    return {
        "class1_f1": f1_from_counts(counts),
        "class1_precision": precision,
        "class1_recall": recall,
        "counts": counts,
    }


def full_support_rf_table(run_dir: Path, arm: str) -> Dict[str, object]:
    """D39's separate standalone-RF table on the full eligible labeled-target support.

    These rows do *not* share a cohort with the paired comparison — they include areas
    whose exact-origin persistence is unavailable — so they are reported separately and
    can never be compared against persistence or substituted for the paired results.
    """
    context = rp.RunContext(run_dir, create=False)
    frames: List[pd.DataFrame] = []
    for fold in rp.final_folds(arm):
        loaded, _ = _verified_frame(context.root / "stage3" / arm / fold.name, fold)
        frames.append(loaded)
    frame = pd.concat(frames, ignore_index=True)

    out: Dict[str, object] = {
        "note": (
            "Full eligible labeled-target support, including rows without valid "
            "exact-origin persistence. Not comparable to persistence and not a "
            "substitute for the paired table (D39/R43)."
        ),
    }
    for horizon in HORIZONS:
        block = frame[frame["horizon_months"] == horizon]
        truth = block["target_label"].to_numpy().astype(int)
        out[f"h{horizon}"] = {
            "rows": int(len(block)),
            "rows_without_persistence": int((~block["persistence_available"]).sum()),
            "pooled_rf": _scored(
                truth, block["pooled_hard_prediction"].to_numpy().astype(int)
            ),
            "partitioned_rf": _scored(
                truth, block["rf_hard_prediction"].to_numpy().astype(int)
            ),
        }
    return out


def map_deployment(run_dir: Path, arm: str) -> Dict[str, object]:
    """What the frozen final map actually did for this arm.

    D19's valid single-partition outcome routes the whole partitioned stream to the
    pooled model. That is contract-compliant and it is also the single most important
    fact for interpreting an arm's final numbers, so it is reported, not buried.
    """
    evidence = rp.read_json(run_dir / "stage2" / arm / "final" / "consensus_evidence.json")
    ledger = evidence["candidate_ledger"]
    return {
        "outcome": evidence["outcome"],
        "outcome_reason": evidence["outcome_reason"],
        "n_clusters": evidence["n_clusters"],
        "eligible_plans": ledger["eligible_plans"],
        "positive_weight_plans": ledger["positive_weight_plans"],
        "total_positive_weight": ledger["total_positive_weight"],
        "coverage": evidence["coverage"],
        "partitioned_stream_is_entirely_pooled": bool(
            str(evidence["outcome"]).startswith("valid_unsplit")
        ),
    }


def final_report(run_dir: Path, *, verify: bool = False) -> Dict[str, object]:
    """D41-D45: adjudicate the frozen winner against persistence on D40's windows.

    The reference arm and the winner-minus-reference increment are computed too, but
    only as *secondary* evidence. D41 and D45 are explicit that the primary decision is
    the winner-versus-persistence comparison and that a secondary result may not be
    swapped into the primary role after the final numbers are seen.
    """
    selection_path = run_dir / "report" / "recipe_selection.json"
    if not selection_path.is_file():
        raise ReportError(f"no frozen recipe selection at {selection_path}")
    selection = rp.read_json(selection_path)
    if not selection.get("authoritative"):
        raise ReportError("the frozen recipe selection is not authoritative")
    winner = str(selection["winner"])

    arms = {"winner": winner, "reference": pdata.REFERENCE_ARM}
    frames: Dict[str, pd.DataFrame] = {}
    thresholds: Dict[str, Dict[str, object]] = {}
    for role_name, arm in arms.items():
        frame = load_scored_rows(run_dir, arm, "final", calibrators_arm=arm)
        frames[role_name] = frame
        # Thresholds were frozen on the 2020 selection window; their lineage is checked
        # against those selection predictions, never against these final rows.
        thresholds[role_name] = load_thresholds(run_dir, arm)

    report: Dict[str, object] = {
        "frozen_winner": winner,
        "winner_development_score": selection["winner_score"],
        "reference_arm": pdata.REFERENCE_ARM,
        "recipe_selection_sha256": rp.sha256_file(selection_path),
        "final_windows": {
            f"h{horizon}": ["%04d-%02d" % pair for pair in pdata.final_target_dates(horizon)]
            for horizon in HORIZONS
        },
        "paired_support": "D36 exact-origin persistence, identical within each horizon",
        "runtime": rp.runtime_identity(),
    }

    for role_name in ("winner", "reference"):
        arm = arms[role_name]
        cells, mean = six_cell_gains(frames[role_name], thresholds[role_name])
        report[role_name] = {
            "arm": arm,
            "primary_gain" if role_name == "winner" else "six_cell_mean": mean,
            "cells": {key: cell["gain"] for key, cell in cells.items()},
            "cell_detail": cells,
            "rows_by_horizon": {
                f"h{horizon}": int((frames[role_name]["horizon_months"] == horizon).sum())
                for horizon in HORIZONS
            },
            # R43/D39's full reporting inventory, not just the gains.
            "final_map_deployment": map_deployment(run_dir, arm),
            "paired_methods": paired_method_table(frames[role_name], thresholds[role_name]),
            "full_support_standalone_rf": full_support_rf_table(run_dir, arm),
        }

    # D41: the matched increment attributable to the updated sources/engineering.
    report["winner_minus_reference"] = {
        "note": (
            "Secondary evidence. Passing the persistence comparison does not by itself "
            "establish an incremental benefit over the corrected reference, and this "
            "increment may not replace the primary metric after the fact."
        ),
        "by_cell": {
            key: report["winner"]["cells"][key] - report["reference"]["cells"][key]
            for key in report["winner"]["cells"]
        },
    }
    report["winner_minus_reference"]["mean"] = float(np.mean(
        list(report["winner_minus_reference"]["by_cell"].values())
    ))

    primary = float(report["winner"]["primary_gain"])
    bootstrap = joint_bootstrap(frames["winner"], thresholds["winner"])
    loyo = leave_one_year_out(frames["winner"], thresholds["winner"])
    report["bootstrap"] = bootstrap
    report["leave_one_year_out"] = loyo
    report["adjudication"] = adjudicate(primary, bootstrap, loyo)

    # D40's supplementary common-calendar view: same stored predictions, aligned
    # calendars. It adds no fit and cannot change the primary verdict.
    start = "%04d-%02d" % COMMON_CALENDAR_START
    common = frames["winner"][frames["winner"]["target_month"] >= start]
    try:
        common_cells, common_mean = six_cell_gains(common, thresholds["winner"])
        report["supplementary_common_calendar"] = {
            "window": f"{start} .. 2024-10",
            "note": (
                "Aligns calendar windows, not area-target support: exact-origin IPC "
                "availability still differs by horizon. Not a primary window."
            ),
            "six_cell_mean": common_mean,
            "cells": {key: cell["gain"] for key, cell in common_cells.items()},
            "rows_by_horizon": {
                f"h{horizon}": int((common["horizon_months"] == horizon).sum())
                for horizon in HORIZONS
            },
        }
    except ReportError as error:
        report["supplementary_common_calendar"] = {"incomplete": str(error)}

    if verify:
        verification = _verify_final(run_dir, report, frames, thresholds)
        report["verification"] = verification
        if not verification["all_passed"]:
            # D45: metric evidence that fails its own reproduction check is invalid,
            # and invalid evidence is incomplete — never a completed negative.
            failed = sorted(
                key for key, value in verification.items()
                if isinstance(value, bool) and value is False
            )
            report["adjudication"]["d45_status"] = "incomplete"
            report["adjudication"]["interpretation"] = (
                "Reported metrics failed their own independent reproduction checks "
                f"({failed}). Under D45 this is NOT a scientific null: resolve the "
                "specific failure before interpreting any number here."
            )
            report["adjudication"]["failed_verification_checks"] = failed
    report["limitations"] = _limitations(report)
    return report


def _limitations(report: Dict[str, object]) -> List[str]:
    """Limitations built from *this* report's arms and roles, not generic text."""
    items = list(FINAL_LIMITATIONS)
    for role_name in ("winner", "reference"):
        deployment = report[role_name]["final_map_deployment"]
        arm = report[role_name]["arm"]
        if deployment["partitioned_stream_is_entirely_pooled"]:
            items.insert(0, (
                f"The {role_name} arm ({arm}) learned NO spatial split on the frozen "
                f"final map: outcome {deployment['outcome']}, nc="
                f"{deployment['n_clusters']}, "
                f"{deployment['positive_weight_plans']} of "
                f"{deployment['eligible_plans']} eligible candidates carried positive "
                "consensus weight. Its entire partitioned stream is therefore the "
                "pooled model (D19/D62), so its final numbers compare a POOLED RF "
                "correction against persistence, not a partitioned one."
            ))
        else:
            items.append(
                f"The {role_name} arm ({arm}) final map learned "
                f"{deployment['n_clusters']} clusters covering "
                f"{deployment['coverage']['assigned_areas']} of "
                f"{deployment['coverage']['master_areas']} master areas; the remainder "
                "routes to the pooled model."
            )
    return items


#: Carried into every final report. These are properties of the frozen design, not
#: defects, and omitting them would overstate what the result establishes.
FINAL_LIMITATIONS: Tuple[str, ...] = (
    "Development-map coverage (reference arm, calibration and selection roles): the "
    "consensus map gave a learned partition to 63.6% and 83.6% of the 5,718-area "
    "master cohort; on matched target-row denominators local models scored 64.5% and "
    "84.1% of rows. These figures describe the reference arm's DEVELOPMENT maps and "
    "are not the final deployment of either arm, which is reported per arm above.",
    "Deliberate source changes carried from the approved design: the inherited climate "
    "z-scores (Rainf_zscore/Tair_zscore) were removed from the updated BASE and the "
    "corrected reference (D46), and assistance signals were excluded (D47). Raw "
    "rainfall and temperature fields are retained.",
    "Consensus weighting: D56's 1e-6 clip gives a plan whose pooled baseline F1 is "
    "exactly 0 a weight an order of magnitude above well-behaved plans. Two such plans "
    "carry 93.7% of the calibration-window weight. R60 mandates the released formula.",
    "The Stage 1 split gate uses a random within-area validation split, not a temporal "
    "holdout, so its F1 runs well above the same candidate's target-month F1.",
    "Stage 1 retains the released filter restricting training rows to groups present in "
    "the target month; it affected 43 of 51 reference candidates (3,509 rows).",
    "Zero unresolved failures after one native-crash retry, not 'no failures occurred'. "
    "The ACCESS_VIOLATION crash on GeoRF-E_2015_10_fs2 remains undiagnosed.",
    "Intervals are conditional on the frozen pipeline and the observed geographic "
    "support. Date-block exchangeability limits coverage claims; this is a "
    "retrospective evaluation of already-inspected years under documented source "
    "assumptions, not fresh holdout inference.",
    "A D45 preservation exception was granted by the user: superseded reference "
    "artifacts from an earlier defective-code run were deleted rather than archived.",
)


def _verify_final(
    run_dir: Path, report: Dict[str, object], frames: Dict[str, pd.DataFrame],
    thresholds: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    """Recompute the headline numbers by an independent route.

    "Independent" is load-bearing here: taking each reported cell's own tau and
    re-running the same helper would restate the computation rather than check it. So
    the taus are read back from the frozen threshold files, and the bootstrap interval
    is rebuilt from the retained draw ledger's multiplicities rather than from the
    gains the bootstrap already returned.
    """
    checks: Dict[str, object] = {}

    # Every tau used in a reported cell must be the one in the frozen file.
    tau_mismatches = []
    for role_name in ("winner", "reference"):
        frozen = rp.read_json(
            run_dir / "thresholds" / report[role_name]["arm"] / "frozen_thresholds.json"
        )["thresholds"]
        for key, cell in report[role_name]["cell_detail"].items():
            if frozen[key]["tau"] != cell["tau"]:
                tau_mismatches.append(f"{role_name}:{key}")
    checks["reported_taus_match_frozen_files"] = not tau_mismatches
    checks["tau_mismatches"] = tau_mismatches
    del thresholds

    # The six-cell mean must equal the mean of the six reported cells.
    for role_name in ("winner", "reference"):
        cells = report[role_name]["cells"]
        key = "primary_gain" if role_name == "winner" else "six_cell_mean"
        checks[f"{role_name}_mean_equals_cell_mean"] = bool(
            abs(float(np.mean(list(cells.values()))) - report[role_name][key]) < 1e-12
        )
        checks[f"{role_name}_cell_count"] = len(cells)

    # Each cell's gain must equal corrected F1 minus persistence F1 recomputed from
    # the stored rows, independently of six_cell_gains' own bookkeeping.
    mismatches = []
    for role_name in ("winner", "reference"):
        frame = frames[role_name]
        for key, cell in report[role_name]["cell_detail"].items():
            horizon, variant = cell["horizon_months"], cell["variant"]
            block = frame[frame["horizon_months"] == horizon]
            truth = block["target_label"].to_numpy().astype(int)
            persistence = block["persistence"].to_numpy().astype(int)
            corrected = apply_correction(
                persistence, block[f"p_{variant}"].to_numpy(dtype=float), cell["tau"]
            )
            gain = (
                f1_from_counts(confusion(truth, corrected))
                - f1_from_counts(confusion(truth, persistence))
            )
            if abs(gain - cell["gain"]) > 1e-12:
                mismatches.append(f"{role_name}:{key}")
    checks["independent_cell_recomputation_matches"] = not mismatches
    checks["mismatched_cells"] = mismatches

    # Rebuild every bootstrap gain from the ledger's multiplicities and the stored
    # rows, then rebuild the interval from those. This is the check that would catch a
    # ledger that does not correspond to the gains the bootstrap reported.
    bootstrap = report["bootstrap"]
    if bootstrap.get("complete"):
        frozen_taus = rp.read_json(
            run_dir / "thresholds" / report["winner"]["arm"] / "frozen_thresholds.json"
        )["thresholds"]
        counts = _cell_counts_by_date(frames["winner"], frozen_taus)
        rebuilt = [
            _weighted_gain(counts, entry["multiplicity"])
            for entry in bootstrap["draw_ledger"] if entry["accepted"]
        ]
        checks["ledger_reconstructs_every_gain"] = (
            len(rebuilt) == len(bootstrap["gains"])
            and all(
                value is not None and abs(value - reported) < 1e-12
                for value, reported in zip(rebuilt, bootstrap["gains"])
            )
        )
        rebuilt_interval = [
            float(np.percentile(rebuilt, 2.5, method="linear")),
            float(np.percentile(rebuilt, 97.5, method="linear")),
        ] if rebuilt and all(v is not None for v in rebuilt) else None
        checks["interval_matches_reconstructed_gains"] = (
            bootstrap["interval_95"] == rebuilt_interval
        )
        checks["draw_ledger_length_equals_attempts"] = (
            len(bootstrap["draw_ledger"]) == bootstrap["attempts"]
        )
        checks["rejection_only_for_empty_support"] = all(
            entry["rejection_reason"] in (None, "empty_required_horizon")
            for entry in bootstrap["draw_ledger"]
        )

    # Every LOYO value must be reproducible from the stored rows.
    loyo_mismatches = []
    frame = frames["winner"].copy()
    frame["target_year"] = frame["target_month"].str.slice(0, 4).astype(int)
    frozen_taus = rp.read_json(
        run_dir / "thresholds" / report["winner"]["arm"] / "frozen_thresholds.json"
    )["thresholds"]
    for year, entry in report["leave_one_year_out"]["by_excluded_year"].items():
        if entry.get("primary_gain") is None:
            continue
        _, mean = six_cell_gains(
            frame[frame["target_year"] != int(year)], frozen_taus
        )
        if abs(mean - entry["primary_gain"]) > 1e-12:
            loyo_mismatches.append(year)
    checks["leave_one_year_out_reproduces"] = not loyo_mismatches
    checks["loyo_mismatches"] = loyo_mismatches

    checks["all_passed"] = all(
        value is True for key, value in checks.items()
        if isinstance(value, bool) and key != "all_passed"
    )
    return checks


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage", default="select-recipe",
                        choices=("select-recipe", "final"))
    parser.add_argument("--arms", default=None,
                        help="Comma-separated candidate arms (select-recipe).")
    parser.add_argument("--winner", default=None, help="Frozen winner arm (final stage).")
    parser.add_argument("--diagnostic-ranking", action="store_true",
                        help="Score an arbitrary arm subset. Writes a separate "
                             "non-authoritative file and never freezes a winner.")
    parser.add_argument("--verify", action="store_true",
                        help="Recompute every reported number from stored rows.")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "select-recipe":
        arms = (
            [a.strip() for a in args.arms.split(",")] if args.arms
            else [name for name, _ in pdata.RECIPE_MANIFEST]
        )
        result = select_recipe(
            run_dir, arms, require_full_inventory=not args.diagnostic_ranking
        )
        if args.diagnostic_ranking:
            # A partial ranking is never written to the authoritative path: the final
            # stage binds to that file, so overwriting it would move the winner after
            # thresholds and calibrators were already frozen against the competition.
            path = out_dir / "recipe_ranking_diagnostic.json"
            rp.write_json(path, result)
        else:
            path = out_dir / "recipe_selection.json"
            if path.exists():
                raise ReportError(
                    f"{path} exists; the recipe selection is frozen. Re-verify with "
                    "--diagnostic-ranking, which writes to a separate file, or use a "
                    "fresh run root for an authorized repair."
                )
            rp.write_json(path, result)
        result["frozen_at"] = str(path)
        print(json.dumps({
            key: value for key, value in result.items()
            if key not in ("cell_detail", "ranking")
        }, indent=2))
        return 0

    result = final_report(run_dir, verify=args.verify)
    rp.write_json(out_dir / "final_report.json", result)
    print(json.dumps(_printable(result), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
