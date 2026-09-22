"""R4-R8 fitting, finite search and the frozen final schedule.

Six learned arms, one persistence reference, four horizons:

===================  ==================  =======  ==============================
arm                  estimator/target    X        fitting support
===================  ==================  =======  ==============================
binary_history_xgb   XGBClassifier, y    93       matched
rich_direct_xgb      XGBClassifier, y    561      matched
correction_xgb       XGBClassifier, e    561      matched
share_xgb            XGBRegressor, q3    561      matched
rich_rf              RandomForest, y     561      matched
fullpool_xgb         XGBClassifier, y    561      full pool
persistence          b, no fitting       -        evaluation history available
===================  ==================  =======  ==============================

The five matched arms fit on byte-identical ordered keys, so a difference
between them is a difference of features or objective and nothing else.
``fullpool_xgb`` deliberately trains on the superset that includes rows without
persistence, and is the only source of predictions for evaluation rows that
have no persistence at all.

    python -B IPCCHPopulationHistoryExperiment/run_pipeline.py --run-dir RUN --stage pilot
    python -B IPCCHPopulationHistoryExperiment/run_pipeline.py --run-dir RUN --stage development
    python -B IPCCHPopulationHistoryExperiment/run_pipeline.py --run-dir RUN --stage select
    python -B IPCCHPopulationHistoryExperiment/run_pipeline.py --run-dir RUN --stage main
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IPCCHGeoRFExperiment import baseline_runtime as brt  # noqa: E402
from IPCCHPopulationHistoryExperiment import prepare_data as prep  # noqa: E402

DEFAULT_RELEASE_ZIP = REPO_ROOT / "GeoRFBaseline" / "releases" / "georf-baseline-v0.1.0.zip"

TRAIN_WINDOW_MONTHS = prep.TRAIN_WINDOW_MONTHS
HORIZONS = prep.ACTIVE_HORIZONS
#: E_no_history decisions are fixed at this cutoff and never tuned (§3).
NO_HISTORY_CUTOFF = 0.5


class PipelineError(RuntimeError):
    """Raised when a contract invariant fails; never downgraded to a warning."""


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    name: str
    family: str  # xgb_classifier | xgb_regressor | rf_classifier
    target: str  # y | e | q3
    features: str  # original93 | rich561
    pool: str  # matched | full

    @property
    def candidate_family(self) -> str:
        return "rf" if self.family == "rf_classifier" else "xgb"


ARMS: tuple[Arm, ...] = (
    Arm("binary_history_xgb", "xgb_classifier", "y", "original93", "matched"),
    Arm("rich_direct_xgb", "xgb_classifier", "y", "rich561", "matched"),
    Arm("correction_xgb", "xgb_classifier", "e", "rich561", "matched"),
    Arm("share_xgb", "xgb_regressor", "q3", "rich561", "matched"),
    Arm("rich_rf", "rf_classifier", "y", "rich561", "matched"),
    Arm("fullpool_xgb", "xgb_classifier", "y", "rich561", "full"),
)
ARMS_BY_NAME = {arm.name: arm for arm in ARMS}
PRIMARY_CANDIDATES = ("rich_direct_xgb", "correction_xgb", "share_xgb")


def resolve_config(configs: dict, family: str, candidate: dict, regressor: bool) -> dict:
    """Overlay one declared candidate on its common block (§5)."""
    resolved = dict(configs[f"{family}_common"])
    if family == "xgb":
        block = "xgb_regressor_only" if regressor else "xgb_classifier_only"
        resolved.update(configs[block])
    resolved.update({k: v for k, v in candidate.items() if k != "id"})
    return resolved


def candidates_for(configs: dict, arm: Arm) -> list[dict]:
    return list(configs[f"{arm.candidate_family}_candidates"])


# --------------------------------------------------------------------------
# Run context
# --------------------------------------------------------------------------


@dataclass
class RunContext:
    run_dir: Path
    keys: pd.DataFrame
    calendar: pd.DataFrame
    spec: prep.FrozenSpec
    X: np.ndarray
    baseline_root: Path

    @property
    def configs(self) -> dict:
        return self.spec.configs


def load_context(run_dir: Path | str, mmap: bool = True, require_matrix: bool = True) -> RunContext:
    """Load a run's keys, calendar, frozen spec and (optionally) its matrix.

    ``require_matrix=False`` exists for the replay paths. Selection and
    reporting read only stored scores and the key table, and the 731 MB feature
    matrix is too large to commit. Letting them run without it is what makes
    those results reproducible from a fresh clone rather than only on the
    machine that fitted them.
    """
    run_dir = Path(run_dir)
    data = run_dir / "data"
    spec = prep.load_frozen_spec(run_dir / "inputs")
    keys = pd.read_csv(data / "keys.csv.gz")
    calendar = pd.read_csv(run_dir / "folds" / "calendar.csv")

    matrix_path = data / "rich561_X.npy"
    if not matrix_path.is_file():
        if require_matrix:
            raise PipelineError(
                f"{run_dir} has no prepared matrix; run prepare_data.py first"
            )
        X = np.empty((len(keys), 0), dtype=np.float64)
        return RunContext(
            run_dir=run_dir,
            keys=keys,
            calendar=calendar,
            spec=spec,
            X=X,
            baseline_root=run_dir / "baseline" / "GeoRFBaseline",
        )

    X = np.load(matrix_path, mmap_mode="r" if mmap else None)
    if X.shape[1] != len(spec.rich_features):
        raise PipelineError("prepared matrix width disagrees with the frozen schema")
    if X.shape[0] != len(keys):
        raise PipelineError("prepared matrix and key table have different row counts")
    return RunContext(
        run_dir=run_dir,
        keys=keys,
        calendar=calendar,
        spec=spec,
        X=X,
        baseline_root=run_dir / "baseline" / "GeoRFBaseline",
    )


def ensure_baseline(run_dir: Path, release_zip: Path = DEFAULT_RELEASE_ZIP) -> dict:
    """Extract the pinned release once per run; later calls verify it in place."""
    destination = run_dir / "baseline"
    root = destination / "GeoRFBaseline"
    if root.is_dir():
        return verify_baseline(root)
    runtime = brt.extract_baseline(release_zip, destination)
    payload = runtime.to_dict()
    payload["note"] = "pristine; the polygon refinement patch is NOT applied here"
    prep._write_json(payload, run_dir / "inputs" / "baseline.json")
    return payload


def verify_baseline(root: Path) -> dict:
    """Re-verify an already-extracted copy against its own manifest."""
    manifest = json.loads((root / "MANIFEST.json").read_text())
    payload = manifest["files_sha256"]
    for relative, expected in payload.items():
        on_disk = root / relative
        if not on_disk.is_file() or prep.sha256_file(on_disk) != expected:
            raise PipelineError(f"extracted baseline payload changed: {relative}")
    return {
        "root": str(root),
        "manifest_version": manifest.get("version", ""),
        "payload_files_verified": len(payload),
        "patch_applied": False,
    }


def attach_baseline(root: Path) -> brt.BaselineRuntime:
    """A BaselineRuntime for an already-extracted, freshly re-verified copy."""
    verify_baseline(root)
    return brt.BaselineRuntime(
        root=root,
        release_sha256=brt.RELEASE_SHA256,
        manifest_version="",
        manifest_source_commit="",
        payload_files_verified=0,
        patch_applied=False,
        patch_diff="",
        pristine_target_sha256="",
        patched_target_sha256="",
    )


# --------------------------------------------------------------------------
# Fold masks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldSupport:
    fold_id: str
    stage: str
    horizon: int
    target_ord: int
    origin_ord: int
    test: np.ndarray
    test_history: np.ndarray
    test_no_history: np.ndarray
    full: np.ndarray
    matched: np.ndarray

    def counts(self) -> dict:
        return {
            "test_rows": int(self.test.sum()),
            "test_rows_with_history": int(self.test_history.sum()),
            "test_rows_without_history": int(self.test_no_history.sum()),
            "full_pool_rows": int(self.full.sum()),
            "matched_pool_rows": int(self.matched.sum()),
        }


def fold_support(keys: pd.DataFrame, record) -> FoldSupport:
    """Boolean masks for one fold, built from keys alone.

    Deliberately not derived from per-model NaN filtering: a matched arm's
    training rows must not depend on which columns happen to be missing.
    """
    horizon = int(record.horizon_months)
    target = int(record.target_ord)
    origin = int(record.origin_ord)

    same_h = keys["horizon_months"].to_numpy() == horizon
    target_ord = keys["target_ord"].to_numpy()
    history = keys["has_history"].to_numpy().astype(bool)

    test = same_h & (target_ord == target)
    full = same_h & (target_ord >= origin - TRAIN_WINDOW_MONTHS + 1) & (target_ord <= origin)
    matched = full & history

    # R3: a training target may not reach the evaluation month, and every
    # training row's own origin is strictly before the fitting origin.
    if (target_ord[full] > origin).any():
        raise PipelineError(f"{record.fold_id}: a training label is later than the origin")
    if np.any(test & full):
        raise PipelineError(f"{record.fold_id}: fitting and evaluation keys overlap")

    return FoldSupport(
        fold_id=record.fold_id,
        stage=record.stage,
        horizon=horizon,
        target_ord=target,
        origin_ord=origin,
        test=test,
        test_history=test & history,
        test_no_history=test & ~history,
        full=full,
        matched=matched,
    )


# --------------------------------------------------------------------------
# Estimators, constants and score orientation
# --------------------------------------------------------------------------


def _feature_slice(X: np.ndarray, arm: Arm, spec: prep.FrozenSpec, rows: np.ndarray) -> np.ndarray:
    width = len(spec.original_features) if arm.features == "original93" else X.shape[1]
    return np.asarray(X[rows, :width], dtype=np.float64)


def _targets(arm: Arm, keys: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    frame = keys.loc[rows]
    if arm.target == "y":
        return frame["ipcch_food_crisis"].to_numpy(dtype=np.float64)
    if arm.target == "q3":
        return frame["q3_target"].to_numpy(dtype=np.float64)
    if arm.target == "e":
        y = frame["ipcch_food_crisis"].to_numpy(dtype=np.float64)
        b = frame["persistence_b"].to_numpy(dtype=np.float64)
        if not np.isfinite(b).all():
            raise PipelineError("the correction target needs b on every fitting row")
        return (y != b).astype(np.float64)
    raise PipelineError(f"unknown target {arm.target}")


def class1_probability(model, X: np.ndarray) -> np.ndarray:
    """P(class 1) read through ``classes_`` rather than by column position."""
    proba = model.predict_proba(X)
    classes = list(np.asarray(model.classes_).ravel())
    if 1 in classes:
        return np.asarray(proba[:, classes.index(1)], dtype=np.float64)
    if 1.0 in classes:
        return np.asarray(proba[:, classes.index(1.0)], dtype=np.float64)
    return np.zeros(len(X), dtype=np.float64)


def fit_and_score(
    arm: Arm,
    resolved: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit one estimator and return its raw scores plus the route actually taken.

    A degenerate fitting target is answered with an explicit constant, not with
    a fabricated row or a silently skipped arm (§4).
    """
    if X_train.shape[0] == 0:
        raise PipelineError(f"{arm.name}: empty fitting pool reached the estimator")

    unique = np.unique(y_train)
    if arm.family == "xgb_regressor":
        if unique.size == 1:
            value = float(unique[0])
            return np.full(X_eval.shape[0], value), {
                "route": "constant_target",
                "constant": value,
                "reason": "the regression target is constant on the fitting pool",
            }
        from xgboost import XGBRegressor  # noqa: PLC0415

        model = XGBRegressor(**resolved)
        model.fit(X_train, y_train)
        scores = np.asarray(model.predict(X_eval), dtype=np.float64)
        return scores, {"route": "model", "estimator": type(model).__name__, **_booster_identity(model)}

    if unique.size == 1:
        value = float(unique[0])
        return np.full(X_eval.shape[0], value), {
            "route": "constant_single_class",
            "constant": value,
            "reason": f"the fitting pool holds only class {int(value)}",
        }

    labels = y_train.astype(np.int64)
    if arm.family == "xgb_classifier":
        from xgboost import XGBClassifier  # noqa: PLC0415

        model = XGBClassifier(**resolved)
        model.fit(X_train, labels)
        return class1_probability(model, X_eval), {
            "route": "model",
            "estimator": type(model).__name__,
            **_booster_identity(model),
        }

    if arm.family == "rf_classifier":
        from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

        model = RandomForestClassifier(**resolved)
        model.fit(X_train, labels)
        return class1_probability(model, X_eval), {
            "route": "model",
            "estimator": type(model).__name__,
            "n_estimators_fitted": int(len(model.estimators_)),
        }

    raise PipelineError(f"unknown family {arm.family}")


def _booster_identity(model) -> dict:
    try:
        booster = model.get_booster()
        return {
            "boosted_rounds": int(booster.num_boosted_rounds()),
            "booster_features": int(booster.num_features()),
        }
    except Exception:  # pragma: no cover - identity is evidence, not control flow
        return {}


def crisis_oriented(arm: Arm, raw: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Map every arm's output onto one score that increases toward crisis (§5)."""
    if arm.name == "correction_xgb":
        if not np.isfinite(b).all():
            raise PipelineError("correction scores need b on every evaluation row")
        return np.where(b == 0.0, raw, 1.0 - raw)
    if arm.family == "xgb_regressor":
        return np.clip(raw, 0.0, 1.0)
    return raw


# --------------------------------------------------------------------------
# One fold
# --------------------------------------------------------------------------

_WORKER: dict = {}


def _worker_init(run_dir: str) -> None:
    context = load_context(run_dir)
    _WORKER["context"] = context
    _WORKER["baseline"] = attach_baseline(context.baseline_root)


def _run_fold_entry(payload: dict) -> dict:
    """Process-pool entry point; failures come back as data, not a dead worker."""
    try:
        return run_fold(_WORKER["context"], _WORKER["baseline"], **payload)
    except Exception as exc:  # noqa: BLE001 - reported and re-raised by the parent
        return {
            "fold_id": payload.get("fold_id"),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def run_fold(
    context: RunContext,
    baseline: brt.BaselineRuntime,
    fold_id: str,
    stage: str,
    selections: dict | None = None,
) -> dict:
    """Fit every scheduled (arm, candidate) for one fold and write its predictions.

    ``selections`` is ``None`` during the search, when all six candidates run,
    and the frozen per-arm choice during the main schedule, when exactly one
    candidate per arm runs.
    """
    started = time.time()
    record = context.calendar[context.calendar["fold_id"] == fold_id]
    if len(record) != 1:
        raise PipelineError(f"fold {fold_id} is not in the calendar exactly once")
    record = record.iloc[0]
    support = fold_support(context.keys, record)
    counts = support.counts()

    out_dir = context.run_dir / stage / "folds"
    out_dir.mkdir(parents=True, exist_ok=True)

    if counts["test_rows"] == 0:
        result = {
            "fold_id": fold_id,
            "stage": stage,
            "status": "skipped_empty_test",
            "support": counts,
            "elapsed_seconds": 0.0,
        }
        prep._write_json(result, out_dir / f"{fold_id}.json")
        return result

    # A scheduled fold with evaluation rows but no fitting pool is an
    # incomplete run, never a quietly dropped fold or a model borrowed from
    # later labels (§4).
    if counts["full_pool_rows"] == 0:
        raise PipelineError(f"{fold_id}: {counts['test_rows']} test rows but an empty full pool")
    if counts["matched_pool_rows"] == 0:
        raise PipelineError(f"{fold_id}: {counts['test_rows']} test rows but an empty matched pool")

    keys = context.keys
    spec = context.spec
    matched_rows = np.where(support.matched)[0]
    full_rows = np.where(support.full)[0]
    eval_history = np.where(support.test_history)[0]
    eval_all = np.where(support.test)[0]

    # One imputer per fold, fitted on the matched training X only, reused by
    # every RF candidate in this fold (§4).
    imputed: dict[str, np.ndarray] = {}
    imputer_audit: dict = {}
    if len(eval_history):
        with brt.baseline_imports(baseline) as (_config, _georf):
            from src.customize.customize import OutOfRangeImputer  # noqa: PLC0415

            imputer = OutOfRangeImputer(strategy="max_plus", multiplier=100.0)
            train_X = _feature_slice(context.X, ARMS_BY_NAME["rich_rf"], spec, matched_rows)
            imputer.fit(train_X)
            imputed["train"] = np.asarray(imputer.transform(train_X), dtype=np.float64)
            eval_X = _feature_slice(context.X, ARMS_BY_NAME["rich_rf"], spec, eval_history)
            imputed["eval"] = np.asarray(imputer.transform(eval_X), dtype=np.float64)
            fills = getattr(imputer, "impute_values_", {})
            stats = getattr(imputer, "column_stats_", {})
            imputer_audit = {
                "strategy": "max_plus",
                "multiplier": 100.0,
                "fitted_rows": int(train_X.shape[0]),
                "columns": int(train_X.shape[1]),
                "all_missing_columns": int(
                    sum(1 for value in stats.values() if value.get("all_missing"))
                ),
                "negative_max_columns": int(
                    sum(1 for value in stats.values() if value.get("max", 0) < 0)
                ),
                "fill_values": {str(k): float(v) for k, v in list(fills.items())},
            }
            for name, matrix in imputed.items():
                if not np.isfinite(matrix).all():
                    raise PipelineError(f"{fold_id}: imputed {name} still holds non-finite values")

    frames = []
    routes = {}
    for arm in ARMS:
        chosen = candidates_for(context.configs, arm)
        if selections is not None:
            wanted = selections[arm.name][str(support.horizon)]["config_id"]
            chosen = [c for c in chosen if c["id"] == wanted]
            if len(chosen) != 1:
                raise PipelineError(f"{arm.name}: frozen candidate {wanted} is not declared")

        train_rows = matched_rows if arm.pool == "matched" else full_rows
        score_rows = eval_all if arm.name == "fullpool_xgb" else eval_history
        if len(score_rows) == 0:
            continue

        y_train = _targets(arm, keys, train_rows)
        b_eval = keys.loc[score_rows, "persistence_b"].to_numpy(dtype=np.float64)

        if arm.family == "rf_classifier":
            X_train = imputed["train"]
            X_eval = imputed["eval"]
        else:
            X_train = _feature_slice(context.X, arm, spec, train_rows)
            X_eval = _feature_slice(context.X, arm, spec, score_rows)

        for candidate in chosen:
            resolved = resolve_config(
                context.configs, arm.candidate_family, candidate, arm.family == "xgb_regressor"
            )
            fit_started = time.time()
            raw, route = fit_and_score(arm, resolved, X_train, y_train, X_eval)
            route["seconds"] = round(time.time() - fit_started, 3)
            route["train_rows"] = int(len(train_rows))
            route["effective_params"] = resolved
            routes[f"{arm.name}::{candidate['id']}"] = route

            crisis = crisis_oriented(arm, raw, b_eval)
            frames.append(
                pd.DataFrame(
                    {
                        "fold_id": fold_id,
                        "arm": arm.name,
                        "config_id": candidate["id"],
                        "row_index": score_rows,
                        "support": np.where(
                            keys.loc[score_rows, "has_history"].to_numpy() == 1,
                            "E_history",
                            "E_no_history",
                        ),
                        "raw_score": raw,
                        "crisis_score": crisis,
                        "route": route["route"],
                    }
                )
            )

    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    predictions.to_csv(out_dir / f"{fold_id}.csv.gz", index=False)

    result = {
        "fold_id": fold_id,
        "stage": stage,
        "status": "complete",
        "horizon_months": support.horizon,
        "target_month": record.target_month,
        "origin_month": record.origin_month,
        "support": counts,
        "imputer": imputer_audit,
        "routes": routes,
        "prediction_rows": int(len(predictions)),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    prep._write_json(result, out_dir / f"{fold_id}.json")
    return result


# --------------------------------------------------------------------------
# Stage drivers
# --------------------------------------------------------------------------


def scheduled_folds(context: RunContext, stage: str) -> pd.DataFrame:
    calendar = context.calendar
    return calendar[calendar["stage"] == stage].reset_index(drop=True)


def completed_folds(run_dir: Path | str, stage: str) -> set[str]:
    directory = Path(run_dir) / stage / "folds"
    if not directory.is_dir():
        return set()
    done = set()
    for path in directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if payload.get("status") in ("complete", "skipped_empty_test"):
            done.add(payload["fold_id"])
    return done


def execute_stage(
    run_dir: Path,
    stage: str,
    fold_ids: Sequence[str],
    workers: int,
    selections: dict | None = None,
) -> dict:
    """Run a list of folds, sequentially or across worker processes.

    Parallelism is over folds only. Every estimator keeps ``n_jobs=1`` from the
    frozen candidate block, so a fold's numbers do not depend on how many
    workers ran beside it; ``--stage verify`` re-runs folds sequentially and
    compares, rather than asserting that.
    """
    started = time.time()
    results: list[dict] = []
    payloads = [
        {"fold_id": fold_id, "stage": stage, "selections": selections} for fold_id in fold_ids
    ]

    if workers <= 1:
        context = load_context(run_dir)
        baseline = attach_baseline(context.baseline_root)
        for payload in payloads:
            results.append(run_fold(context, baseline, **payload))
            _log(run_dir, stage, results[-1])
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_init, initargs=(str(run_dir),)
        ) as pool:
            futures = {pool.submit(_run_fold_entry, payload): payload for payload in payloads}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                _log(run_dir, stage, result)

    failed = [r for r in results if r.get("status") == "failed"]
    if failed:
        raise PipelineError(
            f"{len(failed)} folds failed, first: {failed[0]['fold_id']} "
            f"{failed[0]['error']}\n{failed[0].get('traceback', '')}"
        )

    summary = {
        "stage": stage,
        "workers": workers,
        "folds_requested": len(fold_ids),
        "folds_complete": sum(1 for r in results if r["status"] == "complete"),
        "folds_skipped_empty": sum(1 for r in results if r["status"] == "skipped_empty_test"),
        "wall_seconds": round(time.time() - started, 1),
        "fit_seconds": round(
            sum(
                route.get("seconds", 0.0)
                for r in results
                for route in r.get("routes", {}).values()
            ),
            1,
        ),
        "fits": sum(len(r.get("routes", {})) for r in results),
    }
    return summary


def _log(run_dir: Path, stage: str, result: dict) -> None:
    line = (
        f"[{time.strftime('%H:%M:%S')}] {stage} {result.get('fold_id')} "
        f"{result.get('status')} {result.get('elapsed_seconds', 0)}s "
        f"fits={len(result.get('routes', {}))}"
    )
    print(line, flush=True)
    with open(run_dir / "run.log", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


# --------------------------------------------------------------------------
# Threshold search and the freeze (R7, R8, §5)
# --------------------------------------------------------------------------


def load_stage_predictions(run_dir: Path | str, stage: str) -> pd.DataFrame:
    directory = Path(run_dir) / stage / "folds"
    frames = []
    for path in sorted(directory.glob("*.csv.gz")):
        frame = pd.read_csv(path)
        if len(frame):
            frames.append(frame)
    if not frames:
        raise PipelineError(f"no {stage} predictions under {directory}")
    return pd.concat(frames, ignore_index=True)


def threshold_candidates(scores: np.ndarray, extra: float) -> np.ndarray:
    """101 empirical quantiles plus one fixed cutoff plus both infinities (§5)."""
    if scores.size == 0:
        return np.array([-np.inf], dtype=np.float64)
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, 101), method="linear")
    pool = np.concatenate([quantiles, [extra, -np.inf, np.inf]])
    return np.unique(pool)


def _subgroup_counts(scores: np.ndarray, truth: np.ndarray, thresholds: np.ndarray) -> dict:
    """TP/FP/FN and predicted-positive counts for every threshold, vectorised.

    ``decision = score > t``; equality is negative. Once the score is oriented
    toward crisis, "above the threshold" means "predict crisis" in both b
    subgroups, so one routine serves them both.
    """
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_truth = truth[order]
    # Number of rows strictly above each threshold, and how many of them are positive.
    above = len(scores) - np.searchsorted(sorted_scores, thresholds, side="right")
    positive_suffix = np.concatenate([np.cumsum(sorted_truth[::-1])[::-1], [0.0]])
    index = np.searchsorted(sorted_scores, thresholds, side="right")
    tp = positive_suffix[index]
    fp = above - tp
    total_positive = float(sorted_truth.sum())
    fn = total_positive - tp
    return {"tp": tp, "fp": fp, "fn": fn, "predicted_positive": above.astype(np.float64)}


def select_thresholds(frame: pd.DataFrame, extra: float) -> dict:
    """Joint (t0, t1) maximising pooled class-1 F1, with the declared tie order.

    The two b subgroups are scored with separate thresholds but one pooled
    confusion table, so the grid is an outer sum of per-subgroup counts rather
    than 10,816 independent evaluations.
    """
    result = {"supported_subgroups": [], "thresholds": {}}
    parts = {}
    for b_value in (0, 1):
        block = frame[frame["persistence_b"] == b_value]
        scores = block["crisis_score"].to_numpy(dtype=np.float64)
        truth = block["ipcch_food_crisis"].to_numpy(dtype=np.float64)
        if len(block) == 0:
            # No support: only the no-flip option exists for this subgroup.
            no_flip = np.inf if b_value == 0 else -np.inf
            parts[b_value] = {
                "thresholds": np.array([no_flip]),
                "counts": {
                    "tp": np.zeros(1),
                    "fp": np.zeros(1),
                    "fn": np.zeros(1),
                    "predicted_positive": np.zeros(1),
                },
                "n": 0,
                "supported": False,
            }
            continue
        thresholds = threshold_candidates(scores, extra)
        parts[b_value] = {
            "thresholds": thresholds,
            "counts": _subgroup_counts(scores, truth, thresholds),
            "n": int(len(block)),
            "positives": float(truth.sum()),
            "supported": True,
        }
        result["supported_subgroups"].append(b_value)

    t0 = parts[0]["thresholds"]
    t1 = parts[1]["thresholds"]
    c0, c1 = parts[0]["counts"], parts[1]["counts"]

    tp = c0["tp"][:, None] + c1["tp"][None, :]
    fp = c0["fp"][:, None] + c1["fp"][None, :]
    fn = c0["fn"][:, None] + c1["fn"][None, :]
    denominator = 2.0 * tp + fp + fn
    f1 = np.where(denominator > 0, 2.0 * tp / np.where(denominator > 0, denominator, 1.0), np.nan)

    if not np.isfinite(f1).any():
        raise PipelineError("every candidate threshold pair leaves F1 undefined")

    # Changes from b: in the b=0 subgroup a positive decision is a flip; in the
    # b=1 subgroup a NEGATIVE decision is.
    changes0 = c0["predicted_positive"]
    changes1 = parts[1]["n"] - c1["predicted_positive"]
    changes = changes0[:, None] + changes1[None, :]

    best = np.nanmax(f1)
    mask = np.isclose(f1, best, rtol=0.0, atol=0.0)
    fewest = changes[mask].min()
    mask &= changes == fewest
    # Extended-real lexicographic order on (t0, t1) among the remaining ties.
    rows, cols = np.where(mask)
    order = np.lexsort((t1[cols], t0[rows]))
    pick_row, pick_col = rows[order[0]], cols[order[0]]

    result["thresholds"] = {
        "t0": _encode(t0[pick_row]),
        "t1": _encode(t1[pick_col]),
    }
    result["f1"] = float(best)
    result["confusion"] = {
        "tp": int(tp[pick_row, pick_col]),
        "fp": int(fp[pick_row, pick_col]),
        "fn": int(fn[pick_row, pick_col]),
    }
    result["changes_from_b"] = int(changes[pick_row, pick_col])
    result["grid"] = {"t0_candidates": int(len(t0)), "t1_candidates": int(len(t1))}
    result["subgroups"] = {
        str(b): {
            "rows": parts[b]["n"],
            "positives": parts[b].get("positives"),
            "supported": parts[b]["supported"],
        }
        for b in (0, 1)
    }
    return result


def _encode(value: float) -> str | float:
    """Infinities are persisted as names; JSON has no literal for them (§5)."""
    if np.isposinf(value):
        return "+Infinity"
    if np.isneginf(value):
        return "-Infinity"
    return float(value)


def _decode(value) -> float:
    if isinstance(value, str):
        if value == "+Infinity":
            return np.inf
        if value == "-Infinity":
            return -np.inf
        raise PipelineError(f"unrecognised threshold token {value!r}")
    return float(value)


def apply_thresholds(crisis: np.ndarray, b: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """``decision = score > t_b``; equality is always negative (§5)."""
    threshold = np.where(b == 0.0, t0, t1)
    return (crisis > threshold).astype(np.int64)


def f1_from(truth: np.ndarray, pred: np.ndarray) -> float:
    tp = float(((truth == 1) & (pred == 1)).sum())
    fp = float(((truth == 0) & (pred == 1)).sum())
    fn = float(((truth == 1) & (pred == 0)).sum())
    denominator = 2 * tp + fp + fn
    return float("nan") if denominator == 0 else 2 * tp / denominator


def run_selection(run_dir: Path) -> dict:
    """Pick one candidate and threshold pair per arm/horizon, then one family.

    Every input is a 2020-2022 out-of-time development prediction that already
    exists on disk, so this step can be replayed without fitting anything.
    """
    # Selection reads only stored development scores and the key table, so it
    # must not demand the feature matrix: that is what lets it be replayed.
    context = load_context(run_dir, mmap=True, require_matrix=False)
    predictions = load_stage_predictions(run_dir, "development")
    keys = context.keys
    joined = predictions.merge(
        keys[
            [
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
        ].reset_index(names="row_index"),
        on="row_index",
        how="left",
        validate="many_to_one",
    )
    if joined["ipcch_food_crisis"].isna().any():
        raise PipelineError("a development prediction did not join back to its key")

    # Selection uses E_history only: that is where all seven methods share keys.
    history = joined[joined["support"] == "E_history"].copy()

    persistence = {}
    for horizon in HORIZONS:
        block = keys[
            (keys["horizon_months"] == horizon)
            & (keys["has_history"] == 1)
            & keys["target_month"].isin(_development_months(context))
        ]
        persistence[horizon] = f1_from(
            block["ipcch_food_crisis"].to_numpy(),
            block["persistence_b"].to_numpy(dtype=np.int64),
        )

    ledger_rows = []
    selections: dict = {arm.name: {} for arm in ARMS}
    for arm in ARMS:
        extra = 0.20 if arm.family == "xgb_regressor" else 0.5
        for horizon in HORIZONS:
            best = None
            for candidate in candidates_for(context.configs, arm):
                block = history[
                    (history["arm"] == arm.name)
                    & (history["config_id"] == candidate["id"])
                    & (history["horizon_months"] == horizon)
                ]
                if len(block) == 0:
                    raise PipelineError(
                        f"{arm.name}/{candidate['id']}/h{horizon}: no development predictions"
                    )
                outcome = select_thresholds(block, extra)
                outcome["config_id"] = candidate["id"]
                outcome["rows"] = int(len(block))
                ledger_rows.append(
                    {
                        "arm": arm.name,
                        "horizon_months": horizon,
                        "config_id": candidate["id"],
                        "rows": outcome["rows"],
                        "f1": outcome["f1"],
                        "t0": outcome["thresholds"]["t0"],
                        "t1": outcome["thresholds"]["t1"],
                        "changes_from_b": outcome["changes_from_b"],
                        "tp": outcome["confusion"]["tp"],
                        "fp": outcome["confusion"]["fp"],
                        "fn": outcome["confusion"]["fn"],
                    }
                )
                # Across configurations: best F1, ties broken by JSON order,
                # which is the order this loop already visits them in.
                if best is None or outcome["f1"] > best["f1"]:
                    best = outcome
            selections[arm.name][str(horizon)] = {
                "config_id": best["config_id"],
                "t0": best["thresholds"]["t0"],
                "t1": best["thresholds"]["t1"],
                "development_f1": best["f1"],
                "development_rows": best["rows"],
                "delta_vs_persistence": best["f1"] - persistence[horizon],
            }

    means = {
        name: float(
            np.mean([selections[name][str(h)]["delta_vs_persistence"] for h in HORIZONS])
        )
        for name in PRIMARY_CANDIDATES
    }
    # Exact tie order is the declared one, which is PRIMARY_CANDIDATES' order.
    primary = max(PRIMARY_CANDIDATES, key=lambda name: (means[name], -PRIMARY_CANDIDATES.index(name)))

    freeze = {
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "information_cutoff": "2022-12",
        "selection_cohort": "E_history development predictions, 2020-2022 targets",
        "persistence_development_f1": {str(h): persistence[h] for h in HORIZONS},
        "selections": selections,
        "primary_family": primary,
        "primary_mean_delta": means,
        "spec": context.spec.identity(),
        "matrix_sha256": prep.sha256_file(run_dir / "data" / "rich561_X.npy"),
        "code_sha256": {
            name: prep.sha256_file(PACKAGE_DIR / name)
            for name in ("prepare_data.py", "run_pipeline.py")
        },
    }
    prep._write_json(freeze, run_dir / "freeze.json")
    pd.DataFrame(ledger_rows).to_csv(
        run_dir / "development" / "selection_ledger.csv", index=False
    )
    return freeze


def _development_months(context: RunContext) -> set[str]:
    calendar = context.calendar
    return set(calendar.loc[calendar["stage"] == "development", "target_month"])


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def stage_pilot(run_dir: Path, workers: int) -> dict:
    """The first supported 2020 h1 development target, all six arms and configs."""
    # calendar.csv already carries each fold's support counts, so "supported"
    # is read from it rather than recomputed.
    folds = scheduled_folds(load_context(run_dir), "development")
    candidates = folds[
        (folds["horizon_months"] == 1)
        & (folds["target_month"].str.startswith("2020"))
        & (folds["test_rows"] > 0)
    ].sort_values("target_ord")
    if candidates.empty:
        raise PipelineError("no supported 2020 h1 development target exists")
    fold_id = candidates.iloc[0]["fold_id"]
    return execute_stage(run_dir, "development", [fold_id], workers)


def stage_folds(run_dir: Path, stage: str, workers: int, selections: dict | None) -> dict:
    context = load_context(run_dir)
    calendar = pd.read_csv(run_dir / "folds" / "calendar.csv")
    scheduled = calendar[calendar["stage"] == stage]
    done = completed_folds(run_dir, stage)
    pending = [f for f in scheduled["fold_id"] if f not in done]
    summary = execute_stage(run_dir, stage, pending, workers, selections)
    summary["folds_reused"] = len(done)
    summary["folds_scheduled"] = int(len(scheduled))
    del context
    return summary


def stage_verify(run_dir: Path) -> dict:
    """Replay selected main folds sequentially and compare them exactly (A9).

    Two things are established at once. The obvious one is that a stored
    prediction can be regenerated from the saved inputs. The second is that
    fold-level parallelism changed nothing: this replay runs with one worker
    and one fold at a time, so a byte-identical result is evidence that the
    12-worker main schedule produced the same numbers a sequential one would.

    The folds are chosen without looking at any score: the first and last
    non-empty main fold at each horizon, deduplicated.
    """
    freeze = json.loads((run_dir / "freeze.json").read_text())
    calendar = pd.read_csv(run_dir / "folds" / "calendar.csv")
    main = calendar[(calendar["stage"] == "main") & (calendar["test_rows"] > 0)]

    chosen: list[str] = []
    for horizon in HORIZONS:
        block = main[main["horizon_months"] == horizon].sort_values("target_ord")
        if block.empty:
            continue
        for fold_id in (block.iloc[0]["fold_id"], block.iloc[-1]["fold_id"]):
            if fold_id not in chosen:
                chosen.append(fold_id)

    summary = execute_stage(run_dir, "validation", chosen, workers=1, selections=freeze["selections"])

    comparisons = []
    for fold_id in chosen:
        original = pd.read_csv(run_dir / "main" / "folds" / f"{fold_id}.csv.gz")
        replay = pd.read_csv(run_dir / "validation" / "folds" / f"{fold_id}.csv.gz")
        sort_columns = ["arm", "config_id", "row_index"]
        original = original.sort_values(sort_columns).reset_index(drop=True)
        replay = replay.sort_values(sort_columns).reset_index(drop=True)
        identical = original.equals(replay)
        worst = float("nan")
        if not identical and len(original) == len(replay):
            worst = float(
                np.nanmax(np.abs(original["crisis_score"] - replay["crisis_score"]))
            )
        comparisons.append(
            {
                "fold_id": fold_id,
                "rows": int(len(original)),
                "identical": bool(identical),
                "max_abs_score_difference": worst,
            }
        )

    mismatched = [c for c in comparisons if not c["identical"]]
    report = {
        "folds": comparisons,
        "fold_count": len(chosen),
        "fits": summary["fits"],
        "all_identical": not mismatched,
        "workers_used_for_replay": 1,
        "note": "the replay is sequential; identity here also shows the parallel "
        "main schedule did not alter any prediction",
        "budget_note": f"{summary['fits']} verification fits, counted separately "
        "from the search and main budgets",
    }
    prep._write_json(report, run_dir / "validation" / "replay.json")
    if mismatched:
        raise PipelineError(
            f"{len(mismatched)} replayed folds differ from the stored main "
            f"predictions: {[c['fold_id'] for c in mismatched]}"
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--stage", required=True, choices=("pilot", "development", "select", "main", "verify")
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="fold-level worker processes; every estimator stays at n_jobs=1",
    )
    parser.add_argument("--release-zip", default=str(DEFAULT_RELEASE_ZIP))
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    prep.verify_runtime(strict=True)
    ensure_baseline(run_dir, Path(args.release_zip))

    if args.stage == "pilot":
        summary = stage_pilot(run_dir, args.workers)
    elif args.stage == "development":
        summary = stage_folds(run_dir, "development", args.workers, None)
    elif args.stage == "select":
        summary = run_selection(run_dir)
    elif args.stage == "verify":
        summary = stage_verify(run_dir)
    else:
        freeze_path = run_dir / "freeze.json"
        if not freeze_path.is_file():
            raise PipelineError("the main schedule requires freeze.json; run --stage select")
        freeze = json.loads(freeze_path.read_text())
        summary = stage_folds(run_dir, "main", args.workers, freeze["selections"])

    stage_path = run_dir / f"stage_{args.stage}.json"
    prep._write_json(summary, stage_path)
    print(json.dumps(summary, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
