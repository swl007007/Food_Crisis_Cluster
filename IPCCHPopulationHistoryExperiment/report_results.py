"""R9-R10 inference: metrics, shared-country intervals and the conjunctive verdicts.

Reads the frozen thresholds and the stored main-schedule scores, and never
fits anything. Everything here is recomputable from ``main/folds/*.csv.gz``,
``freeze.json`` and ``data/keys.csv.gz`` alone.

Three separate claims, reported whether or not they succeed (§6):

1. the selected primary family beats ``rich_rf`` AND ``persistence``;
2. if the primary is a reformulation, it additionally beats ``rich_direct_xgb``
   AND ``fullpool_xgb`` -- the formulation advantage;
3. ``rich_direct_xgb`` beats ``binary_history_xgb`` -- the information gain.

A claim is "stable" only when all four of its conditions hold at once: mean
delta > 0, the 95% lower bound > 0, no horizon with a negative point delta, and
a positive mean delta after omitting each target year in turn. Missing evidence
is reported as incomplete; a complete negative result is a valid outcome.

    python -B IPCCHPopulationHistoryExperiment/report_results.py --run-dir RUN --out-dir OUT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IPCCHPopulationHistoryExperiment import prepare_data as prep  # noqa: E402
from IPCCHPopulationHistoryExperiment import run_pipeline as pipe  # noqa: E402

HORIZONS = pipe.HORIZONS
LEARNED_ARMS = tuple(arm.name for arm in pipe.ARMS)
ALL_METHODS = LEARNED_ARMS + ("persistence",)

BOOTSTRAP_SEED = 42
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_MAX_ATTEMPTS = 20000
LEAVE_OUT_YEARS = (2023, 2024, 2025)
SHARE_FIXED_CUTOFF = 0.20


class ReportError(RuntimeError):
    """Raised when required evidence is absent or internally inconsistent."""


# --------------------------------------------------------------------------
# Confusion arithmetic
# --------------------------------------------------------------------------


def confusion(truth: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    truth = np.asarray(truth, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    if truth.shape != pred.shape:
        raise ReportError("truth and prediction lengths differ")
    if truth.size and not (np.isin(truth, (0, 1)).all() and np.isin(pred, (0, 1)).all()):
        raise ReportError("confusion counts require binary truth and predictions")
    t1, p1 = truth == 1, pred == 1
    return (
        int((t1 & p1).sum()),
        int((~t1 & p1).sum()),
        int((t1 & ~p1).sum()),
        int((~t1 & ~p1).sum()),
    )


def class1_f1(tp, fp, fn) -> float:
    denominator = 2 * tp + fp + fn
    return float("nan") if denominator == 0 else float(2 * tp / denominator)


def metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Class-1 metrics. An undefined denominator is NaN with a reason, never 0."""
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n": int(tp + fp + fn + tn),
        "f1": class1_f1(tp, fp, fn),
        "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "undefined": [
            name
            for name, denominator in (
                ("f1", 2 * tp + fp + fn),
                ("precision", tp + fp),
                ("recall", tp + fn),
            )
            if denominator == 0
        ],
    }


# --------------------------------------------------------------------------
# Assembling the scored main table
# --------------------------------------------------------------------------


def build_scored_table(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Join main-fold scores to their keys and apply the frozen thresholds.

    The frozen (t0, t1) pair is state dependent, so it is applied per row
    through the row's own ``b``; ``E_no_history`` never sees it, because it has
    no ``b`` and is routed to fullpool_xgb at the fixed cutoff instead.
    """
    freeze = json.loads((run_dir / "freeze.json").read_text())
    # Reporting never touches X; not requiring it keeps the whole report
    # reproducible from committed evidence in a fresh clone.
    context = pipe.load_context(run_dir, mmap=True, require_matrix=False)
    predictions = pipe.load_stage_predictions(run_dir, "main")

    key_columns = [
        "admin_code",
        "target_month",
        "horizon_months",
        "ipcch_food_crisis",
        "persistence_b",
        "has_history",
        "country_key",
        "cohort",
        "q3_target",
    ]
    joined = predictions.merge(
        context.keys[key_columns].reset_index(names="row_index"),
        on="row_index",
        how="left",
        validate="many_to_one",
    )
    if joined["ipcch_food_crisis"].isna().any():
        raise ReportError("a main prediction did not join back to its key")

    # A9: one row per (arm, area, target, horizon); a duplicate would double
    # count into every confusion table downstream.
    duplicated = joined.duplicated(["arm", "admin_code", "target_month", "horizon_months"])
    if duplicated.any():
        raise ReportError(f"{int(duplicated.sum())} duplicated (arm, area, target, horizon) rows")

    joined["target_year"] = joined["target_month"].str.slice(0, 4).astype(int)
    b = joined["persistence_b"].to_numpy(dtype=np.float64)
    horizon = joined["horizon_months"].to_numpy()
    crisis = joined["crisis_score"].to_numpy(dtype=np.float64)

    t0 = np.full(len(joined), np.nan)
    t1 = np.full(len(joined), np.nan)
    for arm in LEARNED_ARMS:
        for h in HORIZONS:
            mask = (joined["arm"].to_numpy() == arm) & (horizon == h)
            if not mask.any():
                continue
            chosen = freeze["selections"][arm][str(h)]
            t0[mask] = pipe._decode(chosen["t0"])
            t1[mask] = pipe._decode(chosen["t1"])
            declared = joined.loc[mask, "config_id"].unique()
            if list(declared) != [chosen["config_id"]]:
                raise ReportError(
                    f"{arm}/h{h}: scored config {declared} is not the frozen "
                    f"{chosen['config_id']}"
                )

    history = joined["support"].to_numpy() == "E_history"
    threshold = np.where(b == 0.0, t0, t1)
    decision = np.where(history, (crisis > threshold).astype(np.int64), -1)
    # E_no_history is fullpool_xgb's territory, at the fixed cutoff, shared by
    # every combined stream (§3).
    no_history = ~history
    decision = np.where(
        no_history, (crisis > pipe.NO_HISTORY_CUTOFF).astype(np.int64), decision
    )
    joined["decision"] = decision
    joined["threshold"] = np.where(history, threshold, pipe.NO_HISTORY_CUTOFF)

    if (joined.loc[history, "decision"] < 0).any():
        raise ReportError("an E_history row was left without a decision")
    if no_history.any() and set(joined.loc[no_history, "arm"]) != {"fullpool_xgb"}:
        raise ReportError("a method other than fullpool_xgb scored E_no_history rows")

    audit = _cohort_audit(joined, context.keys, freeze, run_dir)
    return joined, audit


def _cohort_audit(scored: pd.DataFrame, keys: pd.DataFrame, freeze: dict, run_dir: Path) -> dict:
    """Verify the evaluation cohorts before any metric is computed.

    Three things have to hold, and none of them is safe to assume: every arm
    covers exactly the same E_history keys at a horizon, truth and persistence
    agree across arms on those keys, and E_history and E_no_history partition
    E_all exactly.
    """
    problems: list[str] = []
    per_horizon = {}
    history = scored[scored["support"] == "E_history"]

    for h in HORIZONS:
        block = history[history["horizon_months"] == h]
        reference = None
        sizes = {}
        for arm in LEARNED_ARMS:
            arm_keys = set(
                map(tuple, block.loc[block["arm"] == arm, ["admin_code", "target_month"]].to_numpy())
            )
            sizes[arm] = len(arm_keys)
            if reference is None:
                reference = arm_keys
            elif arm_keys != reference:
                problems.append(
                    f"h{h}: {arm} covers {len(arm_keys)} E_history keys, "
                    f"the first arm covers {len(reference)}"
                )
        per_horizon[str(h)] = {"common_keys": len(reference or ()), "by_arm": sizes}

        # Truth and persistence must be identical across arms on shared keys.
        pivot = block.pivot_table(
            index=["admin_code", "target_month"],
            columns="arm",
            values=["ipcch_food_crisis", "persistence_b"],
            aggfunc="first",
        )
        for field in ("ipcch_food_crisis", "persistence_b"):
            values = pivot[field].to_numpy()
            if values.size and not np.allclose(values, values[:, [0]], equal_nan=True):
                problems.append(f"h{h}: {field} differs between arms on shared keys")

    # E_history and E_no_history must partition E_all exactly (§3). The
    # scheduled set is keyed on (horizon, target month) pairs, not on the month
    # alone: each horizon has its own main range, so 2023-02 is scheduled at
    # h=1 and at no other horizon.
    calendar = pd.read_csv(run_dir / "folds" / "calendar.csv")
    main_pairs = set(
        map(
            tuple,
            calendar.loc[
                (calendar["stage"] == "main") & (calendar["test_rows"] > 0),
                ["horizon_months", "target_month"],
            ].to_numpy(),
        )
    )
    in_schedule = [
        (int(h), str(m)) in main_pairs
        for h, m in zip(keys["horizon_months"], keys["target_month"])
    ]
    scheduled = set(
        map(
            tuple,
            keys.loc[in_schedule, ["admin_code", "target_month", "horizon_months"]].to_numpy(),
        )
    )
    fullpool = scored[scored["arm"] == "fullpool_xgb"]
    covered = set(
        map(tuple, fullpool[["admin_code", "target_month", "horizon_months"]].to_numpy())
    )
    union_gap = scheduled - covered
    if union_gap:
        problems.append(
            f"{len(union_gap)} scheduled evaluation keys are absent from the "
            "combined fullpool stream"
        )
    # The partition is per (area, target month, HORIZON). The same area-month
    # is routinely E_history at h=1 and E_no_history at h=12, when its first
    # observation falls between the two origins -- that is the supports working,
    # not overlapping.
    partition_columns = ["admin_code", "target_month", "horizon_months"]
    overlap = set(
        map(tuple, fullpool.loc[fullpool["support"] == "E_history", partition_columns].to_numpy())
    ) & set(
        map(
            tuple,
            fullpool.loc[fullpool["support"] == "E_no_history", partition_columns].to_numpy(),
        )
    )
    if overlap:
        problems.append(f"{len(overlap)} keys appear in both supports")

    if problems:
        raise ReportError("cohort audit failed: " + "; ".join(problems))
    return {
        "per_horizon": per_horizon,
        "e_all_rows": int(len(fullpool)),
        "e_history_rows": int((fullpool["support"] == "E_history").sum()),
        "e_no_history_rows": int((fullpool["support"] == "E_no_history").sum()),
        "primary_family": freeze["primary_family"],
    }


def add_persistence(scored: pd.DataFrame) -> pd.DataFrame:
    """Persistence as a seventh method on exactly the E_history comparison keys."""
    template = scored[
        (scored["arm"] == "fullpool_xgb") & (scored["support"] == "E_history")
    ].copy()
    template["arm"] = "persistence"
    template["config_id"] = "persistence"
    template["raw_score"] = template["persistence_b"]
    template["crisis_score"] = template["persistence_b"]
    template["decision"] = template["persistence_b"].astype(np.int64)
    template["threshold"] = np.nan
    template["route"] = "no_model"
    return pd.concat([scored, template], ignore_index=True)


# --------------------------------------------------------------------------
# Headline tables
# --------------------------------------------------------------------------


def metric_table(scored: pd.DataFrame, support: str = "E_history") -> pd.DataFrame:
    rows = []
    block = scored[scored["support"] == support] if support else scored
    for method in ALL_METHODS:
        for h in HORIZONS:
            cell = block[(block["arm"] == method) & (block["horizon_months"] == h)]
            if cell.empty:
                rows.append({"method": method, "horizon_months": h, "n": 0, "f1": float("nan")})
                continue
            values = metrics(
                *confusion(
                    cell["ipcch_food_crisis"].to_numpy(), cell["decision"].to_numpy()
                )
            )
            rows.append({"method": method, "horizon_months": h, **values})
    return pd.DataFrame(rows)


def delta_table(table: pd.DataFrame) -> pd.DataFrame:
    """Point deltas of every method against every other, per horizon."""
    wide = table.pivot(index="horizon_months", columns="method", values="f1")
    rows = []
    for method in ALL_METHODS:
        for baseline in ALL_METHODS:
            if method == baseline:
                continue
            for h in HORIZONS:
                rows.append(
                    {
                        "method": method,
                        "baseline": baseline,
                        "horizon_months": h,
                        "delta_f1": float(wide.loc[h, method] - wide.loc[h, baseline]),
                    }
                )
    frame = pd.DataFrame(rows)
    means = (
        frame.groupby(["method", "baseline"])["delta_f1"]
        .mean()
        .rename("mean_delta_f1")
        .reset_index()
    )
    return frame.merge(means, on=["method", "baseline"], how="left")


# --------------------------------------------------------------------------
# Shared-country bootstrap (§6)
# --------------------------------------------------------------------------


def country_confusion(scored: pd.DataFrame, countries: Sequence[str]) -> dict:
    """Per-country (tp, fp, fn) for every method and horizon, on E_history."""
    index = {name: i for i, name in enumerate(countries)}
    block = scored[scored["support"] == "E_history"]
    out = {}
    for method in ALL_METHODS:
        for h in HORIZONS:
            cell = block[(block["arm"] == method) & (block["horizon_months"] == h)]
            tp = np.zeros(len(countries))
            fp = np.zeros(len(countries))
            fn = np.zeros(len(countries))
            if len(cell):
                position = cell["country_key"].map(index).to_numpy()
                truth = cell["ipcch_food_crisis"].to_numpy()
                pred = cell["decision"].to_numpy()
                np.add.at(tp, position, ((truth == 1) & (pred == 1)).astype(float))
                np.add.at(fp, position, ((truth == 0) & (pred == 1)).astype(float))
                np.add.at(fn, position, ((truth == 1) & (pred == 0)).astype(float))
            out[(method, h)] = (tp, fp, fn)
    return out


def joint_bootstrap(scored: pd.DataFrame, pairs: Sequence[tuple[str, str]]) -> dict:
    """2,000 valid country draws shared by every method and horizon.

    One multiplicity vector per draw is reused everywhere, so a method's and its
    baseline's resampled cohorts are the same cohort. A draw that leaves any
    required method/horizon undefined is rejected in full, never kept for some
    contrasts and dropped for others.
    """
    countries = sorted(set(scored.loc[scored["support"] == "E_history", "country_key"]))
    if not countries:
        raise ReportError("no countries in the E_history cohort")
    cells = country_confusion(scored, countries)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    draws = {pair: [] for pair in pairs}
    accepted = 0
    attempts = 0
    rejections: list[dict] = []
    ledger = []

    while accepted < BOOTSTRAP_DRAWS and attempts < BOOTSTRAP_MAX_ATTEMPTS:
        attempts += 1
        picks = rng.integers(0, len(countries), size=len(countries))
        multiplicity = np.bincount(picks, minlength=len(countries)).astype(float)

        f1 = {}
        reason = None
        for (method, h), (tp, fp, fn) in cells.items():
            weighted = (
                float(multiplicity @ tp),
                float(multiplicity @ fp),
                float(multiplicity @ fn),
            )
            if sum(weighted) == 0:
                reason = f"{method}/h{h} has empty support"
                break
            value = class1_f1(*weighted)
            if not np.isfinite(value):
                reason = f"{method}/h{h} has an undefined F1"
                break
            f1[(method, h)] = value
        if reason is not None:
            rejections.append({"attempt": attempts, "reason": reason})
            continue

        for pair in pairs:
            method, baseline = pair
            draws[pair].append(
                float(np.mean([f1[(method, h)] - f1[(baseline, h)] for h in HORIZONS]))
            )
        accepted += 1
        if accepted <= 5 or accepted % 500 == 0:
            ledger.append(
                {"draw": accepted, "attempt": attempts, "multiplicity_sum": float(multiplicity.sum())}
            )

    intervals = {}
    for pair, values in draws.items():
        array = np.asarray(values, dtype=np.float64)
        if array.size < BOOTSTRAP_DRAWS:
            intervals[f"{pair[0]}_vs_{pair[1]}"] = {
                "complete": False,
                "valid_draws": int(array.size),
                "note": "fewer than 2,000 valid draws: interval evidence is incomplete",
            }
            continue
        intervals[f"{pair[0]}_vs_{pair[1]}"] = {
            "complete": True,
            "valid_draws": int(array.size),
            "mean": float(array.mean()),
            "lower_2.5": float(np.percentile(array, 2.5, method="linear")),
            "upper_97.5": float(np.percentile(array, 97.5, method="linear")),
        }

    return {
        "seed": BOOTSTRAP_SEED,
        "countries": len(countries),
        "country_axis": countries,
        "requested_draws": BOOTSTRAP_DRAWS,
        "valid_draws": accepted,
        "attempts": attempts,
        "rejections": len(rejections),
        "rejection_reasons": rejections[:50],
        "draw_ledger": ledger,
        "intervals": intervals,
        "conditioning": "conditional on the trained predictions; not a refit bootstrap",
        "raw_draws": {f"{a}_vs_{b}": draws[(a, b)] for a, b in pairs},
    }


# --------------------------------------------------------------------------
# Leave-one-target-year-out and the stability verdicts
# --------------------------------------------------------------------------


def leave_year_out(scored: pd.DataFrame, pairs: Sequence[tuple[str, str]]) -> dict:
    """Mean delta after omitting each target year, recomputed not refitted."""
    out: dict = {}
    block = scored[scored["support"] == "E_history"]
    for year in LEAVE_OUT_YEARS:
        remaining = block[block["target_year"] != year]
        f1 = {}
        for method in ALL_METHODS:
            for h in HORIZONS:
                cell = remaining[(remaining["arm"] == method) & (remaining["horizon_months"] == h)]
                if cell.empty:
                    f1[(method, h)] = float("nan")
                    continue
                tp, fp, fn, _tn = confusion(
                    cell["ipcch_food_crisis"].to_numpy(), cell["decision"].to_numpy()
                )
                f1[(method, h)] = class1_f1(tp, fp, fn)
        for method, baseline in pairs:
            deltas = [f1[(method, h)] - f1[(baseline, h)] for h in HORIZONS]
            out.setdefault(f"{method}_vs_{baseline}", {})[str(year)] = {
                "mean_delta": float(np.mean(deltas)) if np.isfinite(deltas).all() else float("nan"),
                "per_horizon": {str(h): float(d) for h, d in zip(HORIZONS, deltas)},
                "rows": int(len(remaining)),
            }
    return out


def stability_verdict(
    pair: tuple[str, str], deltas: pd.DataFrame, interval: dict, omitted: dict
) -> dict:
    """All four conditions, conjunctively; a missing input is incomplete, not a pass."""
    method, baseline = pair
    block = deltas[(deltas["method"] == method) & (deltas["baseline"] == baseline)]
    per_horizon = {int(r.horizon_months): float(r.delta_f1) for r in block.itertuples()}
    mean_delta = float(np.mean(list(per_horizon.values()))) if per_horizon else float("nan")

    conditions = {
        "mean_delta_positive": bool(mean_delta > 0) if np.isfinite(mean_delta) else None,
        "lower_bound_positive": (
            bool(interval["lower_2.5"] > 0) if interval.get("complete") else None
        ),
        "no_negative_horizon": (
            bool(all(value >= 0 for value in per_horizon.values()))
            if per_horizon and np.isfinite(list(per_horizon.values())).all()
            else None
        ),
        "positive_without_each_year": (
            bool(all(entry["mean_delta"] > 0 for entry in omitted.values()))
            if omitted and all(np.isfinite(entry["mean_delta"]) for entry in omitted.values())
            else None
        ),
    }
    if any(value is None for value in conditions.values()):
        verdict = "incomplete"
    elif all(conditions.values()):
        verdict = "stable_gain"
    else:
        verdict = "no_stable_gain"
    return {
        "method": method,
        "baseline": baseline,
        "mean_delta_f1": mean_delta,
        "per_horizon_delta": {str(k): v for k, v in per_horizon.items()},
        "interval": interval,
        "leave_year_out": omitted,
        "conditions": conditions,
        "verdict": verdict,
    }


def required_pairs(primary: str) -> tuple[list[tuple[str, str]], dict]:
    """The comparisons each claim requires (§6)."""
    claims = {
        "prediction_gain": [(primary, "rich_rf"), (primary, "persistence")],
        "information_gain": [("rich_direct_xgb", "binary_history_xgb")],
    }
    if primary in ("correction_xgb", "share_xgb"):
        claims["formulation_advantage"] = [
            (primary, "rich_direct_xgb"),
            (primary, "fullpool_xgb"),
        ]
    pairs: list[tuple[str, str]] = []
    for entries in claims.values():
        for pair in entries:
            if pair not in pairs:
                pairs.append(pair)
    return pairs, claims


# --------------------------------------------------------------------------
# Descriptive strata and per-arm diagnostics
# --------------------------------------------------------------------------


def stratified_tables(scored: pd.DataFrame, run_dir: Path) -> dict[str, pd.DataFrame]:
    block = scored[scored["support"] == "E_history"].copy()

    provenance = pd.read_csv(
        run_dir / "data" / "history_source_keys.csv.gz",
        usecols=[
            "admin_code",
            "target_month",
            "horizon_months",
            "observations_available",
            "obs1_month",
        ],
    )
    block = block.merge(
        provenance, on=["admin_code", "target_month", "horizon_months"], how="left"
    )
    # Age of the newest observation the row could see, in months before its
    # own origin: the natural axis for "does a stale history hurt?".
    origin = block["target_month"].map(lambda v: int(v[:4]) * 12 + int(v[5:7]) - 1) - block[
        "horizon_months"
    ]
    obs1 = block["obs1_month"].map(
        lambda v: int(str(v)[:4]) * 12 + int(str(v)[5:7]) - 1 if isinstance(v, str) and v else np.nan
    )
    block["history_age_months"] = origin.to_numpy() - obs1.to_numpy()
    block["history_age_bin"] = pd.cut(
        block["history_age_months"],
        bins=[-0.5, 3.5, 6.5, 12.5, 24.5, np.inf],
        labels=["0-3", "4-6", "7-12", "13-24", "25+"],
    )
    block["support_bin"] = pd.cut(
        block["observations_available"],
        bins=[-0.5, 1.5, 3.5, 5.5, np.inf],
        labels=["1", "2-3", "4-5", "6+"],
    )

    tables = {}
    for label, columns in (
        ("by_cohort", ["cohort"]),
        ("by_country", ["country_key"]),
        ("by_target_year", ["target_year"]),
        ("by_history_age", ["history_age_bin"]),
        ("by_source_support", ["support_bin"]),
    ):
        rows = []
        for keys, cell in block.groupby(["arm", "horizon_months", *columns], observed=True):
            tp, fp, fn, tn = confusion(
                cell["ipcch_food_crisis"].to_numpy(), cell["decision"].to_numpy()
            )
            entry = dict(zip(["method", "horizon_months", *columns], keys))
            entry.update(metrics(tp, fp, fn, tn))
            entry.pop("undefined", None)
            rows.append(entry)
        tables[label] = pd.DataFrame(rows)
    return tables


def correction_flips(scored: pd.DataFrame) -> pd.DataFrame:
    """Beneficial and harmful flips by direction, for the correction arm."""
    block = scored[(scored["arm"] == "correction_xgb") & (scored["support"] == "E_history")]
    rows = []
    for h in HORIZONS:
        cell = block[block["horizon_months"] == h]
        b = cell["persistence_b"].to_numpy()
        d = cell["decision"].to_numpy()
        y = cell["ipcch_food_crisis"].to_numpy()
        flipped = d != b
        for direction, mask in (("0_to_1", flipped & (b == 0)), ("1_to_0", flipped & (b == 1))):
            rows.append(
                {
                    "horizon_months": h,
                    "direction": direction,
                    "flips": int(mask.sum()),
                    "beneficial": int((mask & (d == y)).sum()),
                    "harmful": int((mask & (d != y)).sum()),
                }
            )
        rows.append(
            {
                "horizon_months": h,
                "direction": "no_flip",
                "flips": int((~flipped).sum()),
                "beneficial": int(((~flipped) & (d == y)).sum()),
                "harmful": int(((~flipped) & (d != y)).sum()),
            }
        )
    return pd.DataFrame(rows)


def share_diagnostics(scored: pd.DataFrame) -> pd.DataFrame:
    """Continuous error and the fixed predicted-share > .20 rule (§6).

    Only the share arm's own E_history outputs are used; a fullpool probability
    is never substituted for a population share.
    """
    block = scored[(scored["arm"] == "share_xgb") & (scored["support"] == "E_history")]
    rows = []
    for h in HORIZONS:
        cell = block[block["horizon_months"] == h]
        raw = cell["raw_score"].to_numpy(dtype=np.float64)
        clipped = cell["crisis_score"].to_numpy(dtype=np.float64)
        truth_share = cell["q3_target"].to_numpy(dtype=np.float64)
        fixed = (clipped > SHARE_FIXED_CUTOFF).astype(np.int64)
        tp, fp, fn, tn = confusion(cell["ipcch_food_crisis"].to_numpy(), fixed)
        rows.append(
            {
                "horizon_months": h,
                "n": int(len(cell)),
                "mae": float(np.mean(np.abs(clipped - truth_share))) if len(cell) else float("nan"),
                "rmse": float(np.sqrt(np.mean((clipped - truth_share) ** 2)))
                if len(cell)
                else float("nan"),
                "raw_below_zero": int((raw < 0).sum()),
                "raw_above_one": int((raw > 1).sum()),
                "fixed_rule_f1": class1_f1(tp, fp, fn),
                "fixed_rule_tp": tp,
                "fixed_rule_fp": fp,
                "fixed_rule_fn": fn,
            }
        )
    return pd.DataFrame(rows)


def combined_stream_table(scored: pd.DataFrame) -> pd.DataFrame:
    """E_all = each method on E_history plus the shared fullpool fallback.

    Reported separately and labelled, because it is a method-plus-fallback
    stream rather than the method itself, and persistence has no entry at all
    on E_no_history (§3).
    """
    fallback = scored[(scored["arm"] == "fullpool_xgb") & (scored["support"] == "E_no_history")]
    rows = []
    for method in ALL_METHODS:
        for h in HORIZONS:
            own = scored[
                (scored["arm"] == method)
                & (scored["support"] == "E_history")
                & (scored["horizon_months"] == h)
            ]
            share = fallback[fallback["horizon_months"] == h]
            if method == "persistence":
                rows.append(
                    {
                        "method": method,
                        "horizon_months": h,
                        "f1": float("nan"),
                        "n": int(len(own)),
                        "note": "persistence is undefined on E_no_history; not a zero prediction",
                    }
                )
                continue
            truth = np.concatenate(
                [own["ipcch_food_crisis"].to_numpy(), share["ipcch_food_crisis"].to_numpy()]
            )
            pred = np.concatenate([own["decision"].to_numpy(), share["decision"].to_numpy()])
            tp, fp, fn, tn = confusion(truth, pred)
            rows.append(
                {
                    "method": method,
                    "horizon_months": h,
                    **metrics(tp, fp, fn, tn),
                    "fallback_rows": int(len(share)),
                    "note": "method on E_history + shared fullpool_xgb@0.5 on E_no_history",
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def generate(run_dir: Path, out_dir: Path) -> dict:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    freeze = json.loads((run_dir / "freeze.json").read_text())
    primary = freeze["primary_family"]

    scored, cohort_audit = build_scored_table(run_dir)
    scored = add_persistence(scored)
    scored.to_csv(out_dir / "main_predictions.csv.gz", index=False)

    table = metric_table(scored)
    deltas = delta_table(table)
    pairs, claims = required_pairs(primary)
    bootstrap = joint_bootstrap(scored, pairs)
    omitted = leave_year_out(scored, pairs)

    verdicts = {}
    for pair in pairs:
        label = f"{pair[0]}_vs_{pair[1]}"
        verdicts[label] = stability_verdict(
            pair, deltas, bootstrap["intervals"].get(label, {}), omitted.get(label, {})
        )

    claim_results = {}
    for claim, entries in claims.items():
        labels = [f"{a}_vs_{b}" for a, b in entries]
        outcomes = [verdicts[label]["verdict"] for label in labels]
        claim_results[claim] = {
            "required_comparisons": labels,
            "verdicts": dict(zip(labels, outcomes)),
            # Conjunctive: every required baseline must show a stable gain.
            "result": (
                "incomplete"
                if "incomplete" in outcomes
                else ("supported" if all(o == "stable_gain" for o in outcomes) else "not_supported")
            ),
        }
    if primary == "rich_direct_xgb":
        claim_results["formulation_advantage"] = {
            "result": "not_applicable",
            "reason": "the selected primary family is the direct classifier",
        }

    table.to_csv(out_dir / "metrics_e_history.csv", index=False)
    deltas.to_csv(out_dir / "deltas_e_history.csv", index=False)
    combined_stream_table(scored).to_csv(out_dir / "metrics_e_all_combined.csv", index=False)
    correction_flips(scored).to_csv(out_dir / "correction_flips.csv", index=False)
    share_diagnostics(scored).to_csv(out_dir / "share_diagnostics.csv", index=False)
    for label, frame in stratified_tables(scored, run_dir).items():
        frame.to_csv(out_dir / f"stratified_{label}.csv", index=False)

    raw_draws = bootstrap.pop("raw_draws")
    pd.DataFrame(raw_draws).to_csv(out_dir / "bootstrap_draws.csv.gz", index=False)
    prep._write_json(bootstrap, out_dir / "bootstrap.json")
    prep._write_json(omitted, out_dir / "leave_year_out.json")

    summary = {
        "run_dir": str(run_dir),
        "primary_family": primary,
        "primary_mean_development_delta": freeze["primary_mean_delta"],
        "cohort_audit": cohort_audit,
        "headline_f1": {
            method: {
                str(h): float(
                    table[(table["method"] == method) & (table["horizon_months"] == h)][
                        "f1"
                    ].iloc[0]
                )
                for h in HORIZONS
            }
            for method in ALL_METHODS
        },
        "verdicts": verdicts,
        "claims": claim_results,
        "bootstrap": {
            "valid_draws": bootstrap["valid_draws"],
            "attempts": bootstrap["attempts"],
            "rejections": bootstrap["rejections"],
            "intervals": bootstrap["intervals"],
        },
        "limitations": [
            "retrospective: the 2023-2025 outcomes were already inspected before this design",
            "source-month alignment does not establish publication-time availability",
            "claim 3 contrasts a tuned feature pipeline, not a fixed-hyperparameter ablation; "
            "the rich schema adds continuous shares AND longer binary history together",
            "the interval is conditional on the trained predictions, not a refit bootstrap",
        ],
    }
    prep._write_json(summary, out_dir / "summary.json")
    return summary


def replay_check(run_dir: Path, first: Path, second: Path) -> dict:
    """Regenerate the whole report into a second directory and compare hashes.

    Everything the reporter produces is a pure function of the stored scores,
    the frozen thresholds and the key table, so two independent runs must be
    byte-identical. A difference would mean a hidden input -- a clock, a set
    iteration order, an uninitialised seed -- is leaking into a reported number.
    """
    generate(run_dir, second)
    files = sorted({p.name for p in first.iterdir()} | {p.name for p in second.iterdir()})
    rows = []
    for name in files:
        a, b = first / name, second / name
        # .csv.gz carries an mtime in its container, so compare the decoded
        # payload rather than the compressed bytes.
        if name.endswith(".gz"):
            digest_a = prep.sha256_file(a) if a.is_file() else ""
            digest_b = prep.sha256_file(b) if b.is_file() else ""
            same = (
                pd.read_csv(a).equals(pd.read_csv(b)) if a.is_file() and b.is_file() else False
            )
        else:
            digest_a = prep.sha256_file(a) if a.is_file() else ""
            digest_b = prep.sha256_file(b) if b.is_file() else ""
            same = bool(digest_a) and digest_a == digest_b
        rows.append(
            {"file": name, "identical": same, "sha256_first": digest_a, "sha256_second": digest_b}
        )
    mismatched = [r["file"] for r in rows if not r["identical"]]
    report = {
        "files": rows,
        "all_identical": not mismatched,
        "mismatched": mismatched,
        "note": "gzip payloads are compared decoded; their containers embed an mtime",
    }
    prep._write_json(report, run_dir / "validation" / "report_replay.json")
    if mismatched:
        raise ReportError(f"the reporter is not deterministic: {mismatched}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--replay-into",
        default=None,
        help="regenerate into this directory too and prove the two agree (A9)",
    )
    args = parser.parse_args(argv)

    summary = generate(Path(args.run_dir), Path(args.out_dir))
    if args.replay_into:
        replay = replay_check(
            Path(args.run_dir), Path(args.out_dir), Path(args.replay_into)
        )
        print(f"reporter replay: {len(replay['files'])} files identical")
    print(json.dumps({k: summary[k] for k in ("primary_family", "headline_f1", "claims")}, indent=2))
    print(f"report written: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
