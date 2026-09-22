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
import hashlib
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
) -> tuple[np.ndarray, dict, object]:
    """Fit one estimator, returning its scores, the route taken, and the model.

    A degenerate fitting target is answered with an explicit constant, not with
    a fabricated row or a silently skipped arm (§4); those routes return
    ``None`` for the model and carry the constant instead.

    The route dictionary records the estimator's **fitted** state -- the
    ``get_params()`` readback and, for XGB, ``save_config()`` and the realised
    round count -- rather than the dictionary that was passed in. The two can
    differ wherever the library resolves a default, and only the readback
    establishes what actually produced a prediction.
    """
    if X_train.shape[0] == 0:
        raise PipelineError(f"{arm.name}: empty fitting pool reached the estimator")

    unique = np.unique(y_train)
    if arm.family == "xgb_regressor":
        if unique.size == 1:
            value = float(unique[0])
            return (
                np.full(X_eval.shape[0], value),
                {
                    "route": "constant_target",
                    "constant": value,
                    "reason": "the regression target is constant on the fitting pool",
                },
                None,
            )
        from xgboost import XGBRegressor  # noqa: PLC0415

        model = XGBRegressor(**resolved)
        model.fit(X_train, y_train)
        scores = np.asarray(model.predict(X_eval), dtype=np.float64)
        return scores, _fitted_identity(model), model

    if unique.size == 1:
        value = float(unique[0])
        return (
            np.full(X_eval.shape[0], value),
            {
                "route": "constant_single_class",
                "constant": value,
                "reason": f"the fitting pool holds only class {int(value)}",
            },
            None,
        )

    labels = y_train.astype(np.int64)
    if arm.family == "xgb_classifier":
        from xgboost import XGBClassifier  # noqa: PLC0415

        model = XGBClassifier(**resolved)
        model.fit(X_train, labels)
        return class1_probability(model, X_eval), _fitted_identity(model), model

    if arm.family == "rf_classifier":
        from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

        model = RandomForestClassifier(**resolved)
        model.fit(X_train, labels)
        return class1_probability(model, X_eval), _fitted_identity(model), model

    raise PipelineError(f"unknown family {arm.family}")


def _jsonable(value):
    """Params can hold numpy scalars and None; JSON needs plain Python."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _fitted_identity(model) -> dict:
    """What the estimator actually is after fitting, read back from itself."""
    identity = {
        "route": "model",
        "estimator": type(model).__name__,
        "fitted_params": {k: _jsonable(v) for k, v in model.get_params().items()},
    }
    try:
        booster = model.get_booster()
        identity["boosted_rounds"] = int(booster.num_boosted_rounds())
        identity["booster_features"] = int(booster.num_features())
        # The full resolved booster configuration, as the library reports it.
        identity["booster_config"] = json.loads(booster.save_config())
    except AttributeError:
        pass
    if hasattr(model, "estimators_"):
        identity["n_estimators_fitted"] = int(len(model.estimators_))
        identity["classes"] = [int(c) for c in np.asarray(model.classes_).ravel()]
    return identity


#: Serialisation is per family: XGB has its own portable binary dump, while a
#: scikit-learn forest only round-trips through pickle.
def save_model(model, arm: Arm, path_stem: Path) -> dict:
    """Persist one fitted estimator and return its path, size and digest."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    if arm.family == "rf_classifier":
        import joblib  # noqa: PLC0415

        path = path_stem.with_suffix(".joblib")
        joblib.dump(model, path, compress=3)
    else:
        path = path_stem.with_suffix(".ubj")
        model.save_model(str(path))
    return {
        # POSIX separators so the record reads the same on either platform.
        "path": path.relative_to(path_stem.parents[2]).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": prep.sha256_file(path),
    }


def load_model(arm: Arm, path: Path):
    """Reload a persisted estimator for prediction-only replay."""
    if arm.family == "rf_classifier":
        import joblib  # noqa: PLC0415

        return joblib.load(path)
    if arm.family == "xgb_regressor":
        from xgboost import XGBRegressor  # noqa: PLC0415

        model = XGBRegressor()
    else:
        from xgboost import XGBClassifier  # noqa: PLC0415

        model = XGBClassifier()
    model.load_model(str(path))
    return model


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
        payload = dict(payload)
        persist = payload.pop("persist_models", None)
        return run_fold(
            _WORKER["context"],
            _WORKER["baseline"],
            persist_models=Path(persist) if persist else None,
            **payload,
        )
    except Exception as exc:  # noqa: BLE001 - reported and re-raised by the parent
        return {
            "fold_id": payload.get("fold_id"),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _key_digest(keys: pd.DataFrame, rows: np.ndarray) -> str:
    """A digest of the ordered (area, target, horizon) fitting keys.

    Binds a saved estimator to the exact rows, in the exact order, that
    produced it -- which is what makes a retained model auditable rather than
    merely present.
    """
    block = keys.loc[rows, ["admin_code", "target_month", "horizon_months"]]
    payload = "\n".join(f"{a}|{t}|{h}" for a, t, h in block.to_numpy())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_fold(
    context: RunContext,
    baseline: brt.BaselineRuntime,
    fold_id: str,
    stage: str,
    selections: dict | None = None,
    persist_models: Path | None = None,
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
            "identity": run_identity(context.run_dir, selections),
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
            raw, route, model = fit_and_score(arm, resolved, X_train, y_train, X_eval)
            route["seconds"] = round(time.time() - fit_started, 3)
            route["train_rows"] = int(len(train_rows))
            # Both are recorded: what was asked for, and what the fitted
            # estimator reports back. A gap between them is itself evidence.
            route["requested_params"] = resolved
            route["fitting_keys_sha256"] = _key_digest(keys, train_rows)
            if model is not None and persist_models is not None:
                route["model"] = save_model(
                    model, arm, persist_models / f"{fold_id}__{arm.name}__{candidate['id']}"
                )
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
        # What this fold's numbers are bound to. completed_folds refuses to
        # reuse a record whose identity no longer matches, so a continuation
        # after changed inputs, candidates or code cannot be mixed in silently.
        "identity": run_identity(context.run_dir, selections),
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


def completed_folds(
    run_dir: Path | str, stage: str, identity: dict | None = None
) -> tuple[set[str], list[str]]:
    """Folds already done, and the ones refused because their identity moved.

    Reuse is not "a file exists". A record produced under a different source,
    schema, candidate inventory, frozen selection or code revision describes a
    different experiment, and mixing it into this one under the same run id is
    exactly the failure the immutability rule exists to prevent.
    """
    directory = Path(run_dir) / stage / "folds"
    if not directory.is_dir():
        return set(), []
    done: set[str] = set()
    stale: list[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if payload.get("status") not in ("complete", "skipped_empty_test"):
            continue
        if identity is not None:
            recorded = payload.get("identity")
            if not recorded:
                stale.append(f"{payload['fold_id']}: no identity recorded")
                continue
            # A partial record proves nothing about the fields it omits, so
            # every field this run defines must be present before the values
            # are compared at all. (The legacy-freeze path is deliberately
            # different: there a human names the gap and accepts it. A fold
            # record silently joining a fitting queue gets no such option.)
            absent = missing_identity_fields(recorded, identity)
            if "code_sha256" in identity and "code_sha256" not in recorded:
                absent = absent + ["code_sha256"]
            if absent:
                stale.append(f"{payload['fold_id']}: identity omits {sorted(absent)}")
                continue
            problems = compare_identity(recorded, identity)
            if problems:
                stale.append(f"{payload['fold_id']}: {problems[0]}")
                continue
        done.add(payload["fold_id"])
    return done, stale


def execute_stage(
    run_dir: Path,
    stage: str,
    fold_ids: Sequence[str],
    workers: int,
    selections: dict | None = None,
    persist_models: Path | None = None,
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
        {
            "fold_id": fold_id,
            "stage": stage,
            "selections": selections,
            "persist_models": str(persist_models) if persist_models else None,
        }
        for fold_id in fold_ids
    ]

    if workers <= 1:
        context = load_context(run_dir)
        baseline = attach_baseline(context.baseline_root)
        for payload in payloads:
            payload = dict(payload)
            persist = payload.pop("persist_models", None)
            results.append(
                run_fold(
                    context,
                    baseline,
                    persist_models=Path(persist) if persist else None,
                    **payload,
                )
            )
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


def code_identity() -> dict:
    return {
        name: prep.sha256_file(PACKAGE_DIR / name)
        for name in ("prepare_data.py", "run_pipeline.py")
    }


def run_identity(run_dir: Path, selections: dict | None = None) -> dict:
    """What a stage's outputs are bound to: inputs, schema, choices and code.

    Written into every fold record and compared before any continuation, so a
    later command cannot silently mix folds produced under different inputs,
    candidate definitions or code into one run id.
    """
    spec = prep.load_frozen_spec(run_dir / "inputs")
    identity = {
        "spec": spec.identity(),
        "matrix_sha256": matrix_identity(run_dir),
        "keys_sha256": prep.sha256_file(run_dir / "data" / "keys.csv.gz"),
        "calendar_sha256": prep.sha256_file(run_dir / "folds" / "calendar.csv"),
        "code_sha256": code_identity(),
    }
    if selections is not None:
        payload = json.dumps(selections, sort_keys=True).encode("utf-8")
        identity["selections_sha256"] = hashlib.sha256(payload).hexdigest()
    return identity


SCIENTIFIC_IDENTITY_FIELDS = (
    "spec",
    "matrix_sha256",
    "keys_sha256",
    "calendar_sha256",
    "selections_sha256",
)


def compare_identity(
    recorded: dict, current: dict, scientific_only: bool = False
) -> list[str]:
    """Differences between two identities, code drift reported separately.

    A field absent from one side is *not* drift. Treating it as drift is how an
    older record ends up looking like a changed experiment, which would block
    every continuation; treating it as agreement is how a real change would slip
    through. Missing fields are reported separately by :func:`missing_identity_fields`.
    """
    fields = list(SCIENTIFIC_IDENTITY_FIELDS)
    if not scientific_only:
        fields.append("code_sha256")
    return [
        f"{field}: recorded {recorded[field]!r} != current {current[field]!r}"
        for field in fields
        if field in recorded and field in current and recorded[field] != current[field]
    ]


def missing_identity_fields(recorded: dict, current: dict) -> list[str]:
    """Scientific fields the current identity defines but the record does not."""
    return [
        field
        for field in SCIENTIFIC_IDENTITY_FIELDS
        if field in current and field not in recorded
    ]


def reconcile_development_cohort(
    run_dir: Path,
    keys: pd.DataFrame,
    calendar: pd.DataFrame,
    history: pd.DataFrame,
    identity: dict | None = None,
) -> dict:
    """Prove the development cohort is complete before anything is optimised.

    Selection maximises F1 over whatever predictions happen to be on disk,
    while persistence is scored from the full calendar. If a fold is missing --
    an interrupted run, a lost artifact -- the search silently optimises on a
    smaller cohort than its own baseline, and the resulting configurations,
    thresholds and primary family look perfectly valid. Nothing downstream can
    detect that, so it is checked here and the freeze is refused if it fails.

    Checked: every supported development fold has a completed record and
    contributes predictions; each (arm, candidate, horizon) covers exactly the
    expected evaluation keys, with no duplicates and nothing outside the
    development window; and every candidate of every arm has identical support.
    """
    scheduled = calendar[calendar["stage"] == "development"]
    supported = scheduled[scheduled["test_rows_with_history"] > 0]
    problems: list[str] = []

    # ``identity`` is supplied when a new authoritative freeze is being
    # created: every consumed fold must then prove it was produced under these
    # same inputs, keys, candidates and code. It is None only for an explicitly
    # labelled read-only replay of an existing run, where the folds are
    # historical evidence and the output is not authoritative.
    done, stale = completed_folds(run_dir, "development", identity)
    missing_records = sorted(set(supported["fold_id"]) - done)
    if missing_records:
        problems.append(
            f"{len(missing_records)} supported development folds have no usable "
            f"completed record: {missing_records[:5]}"
        )
    if stale:
        problems.append(
            f"{len(stale)} development records were not produced under this run's "
            f"identity: {stale[:3]}"
        )

    present = set(history["fold_id"].unique())
    silent = sorted(set(supported["fold_id"]) - present)
    if silent:
        problems.append(
            f"{len(silent)} supported development folds contributed no E_history "
            f"predictions: {silent[:5]}"
        )
    unexpected = sorted(present - set(scheduled["fold_id"]))
    if unexpected:
        problems.append(f"predictions from outside the development schedule: {unexpected[:5]}")

    # The evaluation keys every arm and candidate must cover, per horizon.
    development_months = set(scheduled["target_month"])
    expected: dict[int, set] = {}
    for horizon in HORIZONS:
        block = keys[
            (keys["horizon_months"] == horizon)
            & (keys["has_history"] == 1)
            & (keys["target_month"].isin(development_months))
        ]
        expected[horizon] = set(map(tuple, block[["admin_code", "target_month"]].to_numpy()))

    coverage = {}
    for (arm, config_id, horizon), block in history.groupby(
        ["arm", "config_id", "horizon_months"], sort=True
    ):
        actual = list(map(tuple, block[["admin_code", "target_month"]].to_numpy()))
        as_set = set(actual)
        label = f"{arm}/{config_id}/h{horizon}"
        if len(actual) != len(as_set):
            problems.append(f"{label}: {len(actual) - len(as_set)} duplicated evaluation keys")
        want = expected[int(horizon)]
        if as_set != want:
            problems.append(
                f"{label}: covers {len(as_set)} evaluation keys, expected {len(want)} "
                f"(missing {len(want - as_set)}, unexpected {len(as_set - want)})"
            )
        coverage.setdefault(f"h{horizon}", {})[f"{arm}/{config_id}"] = len(as_set)

    for horizon_label, by_candidate in coverage.items():
        sizes = set(by_candidate.values())
        if len(sizes) > 1:
            problems.append(
                f"{horizon_label}: candidates do not share one support: {sorted(sizes)}"
            )

    if problems:
        why = (
            "the development cohort cannot be bound to this run, so a freeze "
            "derived from it would attach these inputs to predictions that may "
            "not have been made under them"
            if identity is not None
            else "the development cohort is incomplete, so a freeze derived from "
            "it would not be comparable with the persistence baseline"
        )
        hint = (
            " (for an existing run whose folds predate identity recording, use "
            "--replay-into DIR, which produces a labelled read-only replay)"
            if identity is not None
            else ""
        )
        raise PipelineError(f"{why}: " + "; ".join(problems[:6]) + hint)

    return {
        "identity_enforced": identity is not None,
        "supported_folds": int(len(supported)),
        "folds_with_predictions": len(present & set(supported["fold_id"])),
        "consumed_predictions_sha256": _consumed_predictions_digest(run_dir, sorted(present)),
        "expected_keys_per_horizon": {str(h): len(expected[h]) for h in HORIZONS},
        "candidates_per_horizon": {k: len(v) for k, v in coverage.items()},
        "rule": (
            "every supported development fold has a completed record and "
            "contributes predictions; each arm/candidate/horizon covers exactly "
            "the expected evaluation keys with no duplicates; candidates share "
            "one support"
        ),
    }


def _consumed_predictions_digest(run_dir: Path, fold_ids: Sequence[str]) -> str:
    """Bind the freeze to the exact prediction files it was derived from."""
    digest = hashlib.sha256()
    for fold_id in sorted(fold_ids):
        path = run_dir / "development" / "folds" / f"{fold_id}.csv.gz"
        digest.update(fold_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(
            (prep.sha256_file(path) if path.is_file() else "absent").encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def matrix_identity(run_dir: Path) -> str:
    """The feature matrix's hash, from the file if present, else the manifest.

    The freeze has to name the exact matrix it was computed from. That file is
    731 MB and stays local, so on a replay the hash is read from the
    preparation manifest, which recorded it while the file was in hand. A run
    with neither is not replayable and says so.
    """
    path = run_dir / "data" / "rich561_X.npy"
    if path.is_file():
        return prep.sha256_file(path)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        recorded = json.loads(manifest_path.read_text()).get("matrix", {}).get("matrix_sha256")
        if recorded:
            return recorded
    raise PipelineError(
        "the feature matrix is absent and its hash was not recorded at "
        "preparation time; this run cannot be bound to its inputs"
    )


def run_selection(run_dir: Path, into: Path | None = None, refreeze: bool = False) -> dict:
    """Pick one candidate and threshold pair per arm/horizon, then one family.

    Every input is a 2020-2022 out-of-time development prediction that already
    exists on disk, so this step can be replayed without fitting anything.

    The freeze is immutable once written. Re-running selection into a run that
    already has one is refused: a second freeze with a fresh timestamp would
    silently relabel main-schedule predictions that were computed under the
    first. Pass ``into`` to replay selection into a separate directory, which
    is how the freeze gets checked without being overwritten.
    """
    replay = into is not None
    target = Path(into) if replay else Path(run_dir)
    if not replay and (target / "freeze.json").is_file() and not refreeze:
        raise PipelineError(
            f"{target / 'freeze.json'} already exists. Selection is frozen once "
            "written; use --replay-into DIR to re-derive it for comparison, or "
            "--refreeze only when deliberately discarding the existing freeze."
        )
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

    # Nothing is optimised until the cohort is shown to be complete.
    cohort = reconcile_development_cohort(
        run_dir,
        keys,
        context.calendar,
        history,
        identity=None if replay else run_identity(run_dir, None),
    )

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
        "cohort_reconciliation": cohort,
        # A replay re-derives an existing freeze for comparison. It is not an
        # input-bound experimental commitment and must never drive a main run.
        "authoritative": not replay,
        "kind": "read_only_replay" if replay else "authoritative_freeze",
        "persistence_development_f1": {str(h): persistence[h] for h in HORIZONS},
        "selections": selections,
        "primary_family": primary,
        "primary_mean_delta": means,
        "spec": context.spec.identity(),
        "matrix_sha256": matrix_identity(run_dir),
        # The complete scientific identity, so the main schedule can prove it
        # is running under the inputs that were frozen rather than comparing a
        # partial record and guessing about the rest.
        "identity": run_identity(run_dir),
        "code_sha256": {
            name: prep.sha256_file(PACKAGE_DIR / name)
            for name in ("prepare_data.py", "run_pipeline.py")
        },
    }
    target.mkdir(parents=True, exist_ok=True)
    prep._write_json(freeze, target / "freeze.json")
    ledger_path = (
        target / "selection_ledger.csv"
        if into is not None
        else run_dir / "development" / "selection_ledger.csv"
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ledger_rows).to_csv(ledger_path, index=False)
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
    folds = scheduled_folds(load_context(run_dir, require_matrix=False), "development")
    candidates = folds[
        (folds["horizon_months"] == 1)
        & (folds["target_month"].str.startswith("2020"))
        & (folds["test_rows"] > 0)
    ].sort_values("target_ord")
    if candidates.empty:
        raise PipelineError("no supported 2020 h1 development target exists")
    fold_id = candidates.iloc[0]["fold_id"]

    # The pilot is one development fold, not a stage of its own, so it obeys
    # the same rule as the rest: an exactly matching completed record is
    # reused, anything else refuses. Repeating the documented pilot command
    # must never quietly refit over a fold the search already depends on.
    identity = run_identity(run_dir, None)
    done, stale = completed_folds(run_dir, "development", identity)
    if fold_id in done:
        return {
            "stage": "development",
            "pilot_fold": fold_id,
            "folds_requested": 0,
            "folds_complete": 0,
            "folds_reused": 1,
            "fits": 0,
            "note": "the pilot fold is already complete under this identity; reused, not refitted",
        }
    blocking = [reason for reason in stale if reason.startswith(f"{fold_id}:")]
    if blocking:
        raise PipelineError(
            f"the pilot fold {fold_id} already has a completed record that does "
            f"not match this run: {blocking[0]}. It is immutable; prepare a "
            "fresh run directory rather than refitting over it."
        )
    summary = execute_stage(run_dir, "development", [fold_id], workers)
    summary["pilot_fold"] = fold_id
    return summary


def stage_folds(run_dir: Path, stage: str, workers: int, selections: dict | None) -> dict:
    """Schedule the folds of one stage, never over a completed artifact.

    A completed fold whose identity no longer matches is not re-fitted and not
    reused: the run is rejected. Overwriting it in place would destroy the
    evidence of what the earlier fold actually produced while keeping the same
    run id, freeze and reports -- which is the specific failure R12's
    fresh-run rule exists to prevent. The remedy is a fresh run directory.
    """
    calendar = pd.read_csv(run_dir / "folds" / "calendar.csv")
    scheduled = calendar[calendar["stage"] == stage]
    identity = run_identity(run_dir, selections)
    done, stale = completed_folds(run_dir, stage, identity)
    if stale:
        unprovable = [reason for reason in stale if "no identity recorded" in reason]
        detail = (
            f"{len(unprovable)} of them predate identity recording and therefore "
            "cannot be proved to match; "
            if unprovable
            else ""
        )
        raise PipelineError(
            f"{len(stale)} completed {stage} folds in {run_dir} do not match this "
            f"run's identity: {stale[:3]}. {detail}"
            "They are immutable, and continuing here would overwrite them under "
            "the same run id, freeze and reports. Prepare a fresh run directory "
            "instead. A stage that is already finished needs no continuation."
        )
    pending = [f for f in scheduled["fold_id"] if f not in done]

    # Belt and braces: whatever the reuse logic decided, nothing already
    # recorded as complete may enter the fitting queue.
    recorded, _ = completed_folds(run_dir, stage, None)
    collisions = sorted(set(pending) & recorded)
    if collisions:
        raise PipelineError(
            f"refusing to refit {len(collisions)} folds that already have "
            f"completed records: {collisions[:3]}"
        )
    summary = execute_stage(run_dir, stage, pending, workers, selections)
    summary["folds_reused"] = len(done)
    summary["folds_scheduled"] = int(len(scheduled))
    summary["identity"] = identity
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


def check_main_preconditions(
    run_dir: Path,
    freeze: dict,
    allow_code_drift: bool = False,
    allow_legacy_freeze: bool = False,
) -> dict:
    """Everything the main schedule must be true of before it fits anything.

    The scientific identity is compared field by field against the one the
    freeze recorded. A field the freeze never carried is an old schema, not
    drift, and is refused by name rather than by inventing a historical hash or
    by quietly passing: the two failure modes the check exists to avoid.
    """
    if freeze.get("authoritative") is False:
        raise PipelineError(
            f"this freeze is a {freeze.get('kind')}, re-derived for comparison "
            "from historical folds without proving their identity. It is not an "
            "input-bound commitment and cannot drive a main schedule."
        )
    current = run_identity(run_dir)
    recorded = freeze.get("identity")
    legacy = recorded is None
    if legacy:
        # Freezes written before identities existed carry only these two.
        recorded = {
            key: freeze[key] for key in ("spec", "matrix_sha256") if key in freeze
        }

    drift = compare_identity(recorded, current, scientific_only=True)
    if drift:
        raise PipelineError(
            "the frozen inputs no longer describe this run, so the main schedule "
            f"would not be the one that was frozen: {drift}"
        )

    missing = missing_identity_fields(recorded, current)
    if missing and not allow_legacy_freeze:
        raise PipelineError(
            f"this freeze predates {missing}, so those inputs cannot be proved "
            "unchanged. Re-freeze on the current schema, or pass "
            "--allow-legacy-freeze and record why that is acceptable."
        )

    code_drift = freeze.get("code_sha256") != code_identity()
    if code_drift and not allow_code_drift:
        raise PipelineError(
            "the code has changed since the freeze. Re-run on the frozen "
            "revision, or pass --allow-code-drift and justify it in the run record."
        )
    return {
        "legacy_freeze": legacy,
        "identity_fields_compared": sorted(
            set(recorded) & set(current) & set(SCIENTIFIC_IDENTITY_FIELDS)
        ),
        "identity_fields_unprovable": missing,
        "code_drift_from_freeze": bool(code_drift),
        "allow_code_drift": bool(allow_code_drift),
        "allow_legacy_freeze": bool(allow_legacy_freeze),
    }


def stage_persist(run_dir: Path, workers: int) -> dict:
    """Retain the selected final estimators, and prove they are the ones used.

    The main schedule scored and discarded its models. This refits every
    selected model at every main origin under the frozen choices, keeps the
    fitted object, records its digest, its ``get_params()`` readback and its
    booster configuration, and then requires the regenerated predictions to be
    identical to the stored ones -- fold by fold, across the whole schedule.

    Identity, not just presence: each saved model carries the digest of the
    ordered fitting keys it was built from, so a reviewer can tell which rows
    produced it rather than taking the filename's word for it.

    Predictions go to a scratch directory; ``main/folds`` is never rewritten.
    """
    freeze = json.loads((run_dir / "freeze.json").read_text())
    calendar = pd.read_csv(run_dir / "folds" / "calendar.csv")
    scheduled = calendar[(calendar["stage"] == "main") & (calendar["test_rows"] > 0)]
    fold_ids = list(scheduled["fold_id"])

    models_dir = run_dir / "main" / "models"
    summary = execute_stage(
        run_dir,
        "main_refit",
        fold_ids,
        workers,
        selections=freeze["selections"],
        persist_models=models_dir,
    )

    comparisons = []
    identity_dir = run_dir / "main" / "model_identity"
    identity_dir.mkdir(parents=True, exist_ok=True)
    for fold_id in fold_ids:
        stored = pd.read_csv(run_dir / "main" / "folds" / f"{fold_id}.csv.gz")
        refit = pd.read_csv(run_dir / "main_refit" / "folds" / f"{fold_id}.csv.gz")
        order = ["arm", "config_id", "row_index"]
        stored = stored.sort_values(order).reset_index(drop=True)
        refit = refit.sort_values(order).reset_index(drop=True)
        record = json.loads((run_dir / "main_refit" / "folds" / f"{fold_id}.json").read_text())
        prep._write_json(
            {
                "fold_id": fold_id,
                "identity": record["identity"],
                "support": record["support"],
                "imputer": record.get("imputer", {}),
                "models": {
                    key: {
                        "route": route["route"],
                        "estimator": route.get("estimator"),
                        "constant": route.get("constant"),
                        "fitting_keys_sha256": route.get("fitting_keys_sha256"),
                        "train_rows": route.get("train_rows"),
                        "fitted_params": route.get("fitted_params"),
                        "booster_config": route.get("booster_config"),
                        "boosted_rounds": route.get("boosted_rounds"),
                        "n_estimators_fitted": route.get("n_estimators_fitted"),
                        "model": route.get("model"),
                    }
                    for key, route in record["routes"].items()
                },
                "reproduces_stored_predictions": bool(stored.equals(refit)),
            },
            identity_dir / f"{fold_id}.json",
        )
        comparisons.append(
            {"fold_id": fold_id, "rows": int(len(stored)), "identical": bool(stored.equals(refit))}
        )

    # The 36 (arm, candidate) fitted readbacks behind the development search.
    # They do not vary by fold -- the candidate block fixes every parameter and
    # the seed -- so one real fit per pair characterises all 4,506 of them.
    development = _development_param_readback(run_dir)

    mismatched = [c for c in comparisons if not c["identical"]]
    report = {
        "folds": comparisons,
        "fold_count": len(fold_ids),
        "fits": summary["fits"],
        "all_reproduce_stored_predictions": not mismatched,
        "models_dir": str(models_dir),
        "development_param_readback": development,
        "budget_note": (
            f"{summary['fits']} refit-for-provenance fits plus "
            f"{development['fits']} development readback fits, both counted "
            "separately from the 4,506 search and 660 main budgets"
        ),
        "note": (
            "the retained estimators regenerate every stored main prediction "
            "exactly, across all folds, not only the eight verification folds"
        ),
    }
    prep._write_json(report, run_dir / "validation" / "model_persistence.json")
    if mismatched:
        raise PipelineError(
            f"{len(mismatched)} refitted folds do not reproduce the stored "
            f"predictions: {[c['fold_id'] for c in mismatched][:5]}"
        )
    return report


def _development_param_readback(run_dir: Path) -> dict:
    """Fit each (arm, candidate) once and dump what the estimator reports back."""
    context = load_context(run_dir)
    baseline = attach_baseline(context.baseline_root)
    calendar = context.calendar
    development = calendar[
        (calendar["stage"] == "development") & (calendar["test_rows_with_history"] > 0)
    ].sort_values("target_ord")
    record = development.iloc[0]
    support = fold_support(context.keys, record)

    matched_rows = np.where(support.matched)[0]
    full_rows = np.where(support.full)[0]
    eval_rows = np.where(support.test_history)[0][:1]

    with brt.baseline_imports(baseline) as (_config, _georf):
        from src.customize.customize import OutOfRangeImputer  # noqa: PLC0415

        imputer = OutOfRangeImputer(strategy="max_plus", multiplier=100.0)
        rf_train = _feature_slice(context.X, ARMS_BY_NAME["rich_rf"], context.spec, matched_rows)
        imputer.fit(rf_train)
        rf_train_imputed = np.asarray(imputer.transform(rf_train), dtype=np.float64)
        rf_eval = np.asarray(
            imputer.transform(
                _feature_slice(context.X, ARMS_BY_NAME["rich_rf"], context.spec, eval_rows)
            ),
            dtype=np.float64,
        )

    readback: dict = {}
    fits = 0
    for arm in ARMS:
        train_rows = matched_rows if arm.pool == "matched" else full_rows
        y_train = _targets(arm, context.keys, train_rows)
        if arm.family == "rf_classifier":
            X_train, X_eval = rf_train_imputed, rf_eval
        else:
            X_train = _feature_slice(context.X, arm, context.spec, train_rows)
            X_eval = _feature_slice(context.X, arm, context.spec, eval_rows)
        for candidate in candidates_for(context.configs, arm):
            resolved = resolve_config(
                context.configs, arm.candidate_family, candidate, arm.family == "xgb_regressor"
            )
            _raw, route, _model = fit_and_score(arm, resolved, X_train, y_train, X_eval)
            fits += 1
            readback[f"{arm.name}::{candidate['id']}"] = {
                "requested_params": resolved,
                "fitted_params": route.get("fitted_params"),
                "booster_config": route.get("booster_config"),
                "boosted_rounds": route.get("boosted_rounds"),
                "n_estimators_fitted": route.get("n_estimators_fitted"),
            }
    payload = {
        "fold_used": record.fold_id,
        "fits": fits,
        "combinations": len(readback),
        "rationale": (
            "candidate parameters and the seed are fixed by the frozen block and "
            "do not vary by fold, so one fitted readback per (arm, candidate) "
            "characterises every development fit of that pair"
        ),
        "readback": readback,
    }
    prep._write_json(payload, run_dir / "development" / "effective_params.json")
    return payload


def stage_replay_models(run_dir: Path, workers: int) -> dict:
    """Predict from the retained estimators without fitting anything.

    This is the check the refit cannot make: it loads each saved model off
    disk, verifies its digest, predicts, and requires the stored scores back.
    A model that was silently replaced, truncated or re-fitted under different
    data fails here even though a refit would have passed.
    """
    del workers
    freeze = json.loads((run_dir / "freeze.json").read_text())
    context = load_context(run_dir)
    identity_dir = run_dir / "main" / "model_identity"
    if not identity_dir.is_dir():
        raise PipelineError("no retained models; run --stage persist first")

    baseline = attach_baseline(context.baseline_root)
    stored_all = pipe_stored_predictions(run_dir)

    results = []
    for path in sorted(identity_dir.glob("*.json")):
        record = json.loads(path.read_text())
        fold_id = record["fold_id"]
        fold = context.calendar[context.calendar["fold_id"] == fold_id].iloc[0]
        support = fold_support(context.keys, fold)
        matched_rows = np.where(support.matched)[0]
        eval_history = np.where(support.test_history)[0]
        eval_all = np.where(support.test)[0]

        rf_eval = None
        if len(eval_history):
            with brt.baseline_imports(baseline) as (_config, _georf):
                from src.customize.customize import OutOfRangeImputer  # noqa: PLC0415

                imputer = OutOfRangeImputer(strategy="max_plus", multiplier=100.0)
                imputer.fit(
                    _feature_slice(
                        context.X, ARMS_BY_NAME["rich_rf"], context.spec, matched_rows
                    )
                )
                rf_eval = np.asarray(
                    imputer.transform(
                        _feature_slice(
                            context.X, ARMS_BY_NAME["rich_rf"], context.spec, eval_history
                        )
                    ),
                    dtype=np.float64,
                )

        stored = stored_all[stored_all["fold_id"] == fold_id]
        for key, entry in record["models"].items():
            arm_name, config_id = key.split("::")
            arm = ARMS_BY_NAME[arm_name]
            score_rows = eval_all if arm_name == "fullpool_xgb" else eval_history
            block = stored[(stored["arm"] == arm_name) & (stored["config_id"] == config_id)]
            block = block.sort_values("row_index")
            if entry["route"] != "model":
                results.append(
                    {
                        "fold_id": fold_id,
                        "key": key,
                        "route": entry["route"],
                        "identical": bool(
                            np.allclose(block["raw_score"].to_numpy(), entry["constant"])
                        ),
                    }
                )
                continue
            # save_model records the path relative to the run root. Only a
            # subset of the 533 MB of estimators is committed, so an absent
            # binary is reported as unchecked rather than treated as a pass --
            # and never as a failure, which would make the committed subset
            # unusable in a fresh clone.
            model_path = run_dir / entry["model"]["path"]
            if not model_path.is_file():
                results.append(
                    {
                        "fold_id": fold_id,
                        "key": key,
                        "route": "model",
                        "checked": False,
                        "reason": "binary not present; regenerate with --stage persist",
                        "identical": None,
                    }
                )
                continue
            digest = prep.sha256_file(model_path)
            X_eval = (
                rf_eval
                if arm.family == "rf_classifier"
                else _feature_slice(context.X, arm, context.spec, score_rows)
            )
            model = load_model(arm, model_path)
            if arm.family == "xgb_regressor":
                raw = np.asarray(model.predict(X_eval), dtype=np.float64)
            else:
                raw = class1_probability(model, X_eval)
            expected = block["raw_score"].to_numpy(dtype=np.float64)
            # Stored scores went through a CSV round trip, so compare at the
            # precision that survives it rather than demanding exact bits.
            identical = expected.shape == raw.shape and np.allclose(
                expected, raw, rtol=0.0, atol=5e-16 + 1e-12
            )
            results.append(
                {
                    "fold_id": fold_id,
                    "key": key,
                    "route": "model",
                    "digest_matches": digest == entry["model"]["sha256"],
                    "rows": int(len(expected)),
                    "max_abs_difference": float(np.max(np.abs(expected - raw)))
                    if expected.shape == raw.shape
                    else float("nan"),
                    "identical": bool(identical),
                }
            )

    checked = [r for r in results if r.get("checked", True)]
    skipped = [r for r in results if not r.get("checked", True)]
    failures = [
        r for r in checked if not r["identical"] or not r.get("digest_matches", True)
    ]
    report = {
        "models_checked": len(checked),
        "models_absent": len(skipped),
        "folds": len(set(r["fold_id"] for r in checked)),
        "all_checked_reproduce": not failures,
        "failures": failures[:20],
        "primary_family": freeze["primary_family"],
        "note": (
            "prediction-only replay: models were loaded from disk, not refitted. "
            "An absent binary is reported as unchecked, never as a pass"
        ),
    }
    prep._write_json(report, run_dir / "validation" / "model_replay.json")
    if failures:
        raise PipelineError(
            f"{len(failures)} retained models did not reproduce their scores"
        )
    if not checked:
        raise PipelineError(
            "no retained model binaries were present, so nothing was verified; "
            "run --stage persist to regenerate them"
        )
    return report


def pipe_stored_predictions(run_dir: Path) -> pd.DataFrame:
    return load_stage_predictions(run_dir, "main")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "pilot",
            "development",
            "select",
            "main",
            "verify",
            "persist",
            "replay-models",
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="fold-level worker processes; every estimator stays at n_jobs=1",
    )
    parser.add_argument("--release-zip", default=str(DEFAULT_RELEASE_ZIP))
    parser.add_argument(
        "--replay-into", default=None, help="select: re-derive the freeze here instead"
    )
    parser.add_argument(
        "--refreeze", action="store_true", help="select: deliberately discard an existing freeze"
    )
    parser.add_argument(
        "--allow-code-drift",
        action="store_true",
        help="main: continue although the code moved since the freeze",
    )
    parser.add_argument(
        "--allow-legacy-freeze",
        action="store_true",
        help="main: accept a freeze written before identities were recorded",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    prep.verify_runtime(strict=True)
    ensure_baseline(run_dir, Path(args.release_zip))

    if args.stage == "pilot":
        summary = stage_pilot(run_dir, args.workers)
    elif args.stage == "development":
        summary = stage_folds(run_dir, "development", args.workers, None)
    elif args.stage == "select":
        summary = run_selection(
            run_dir,
            into=Path(args.replay_into) if args.replay_into else None,
            refreeze=args.refreeze,
        )
    elif args.stage == "verify":
        summary = stage_verify(run_dir)
    elif args.stage == "persist":
        summary = stage_persist(run_dir, args.workers)
    elif args.stage == "replay-models":
        summary = stage_replay_models(run_dir, args.workers)
    else:
        freeze_path = run_dir / "freeze.json"
        if not freeze_path.is_file():
            raise PipelineError("the main schedule requires freeze.json; run --stage select")
        freeze = json.loads(freeze_path.read_text())
        preconditions = check_main_preconditions(
            run_dir,
            freeze,
            allow_code_drift=args.allow_code_drift,
            allow_legacy_freeze=args.allow_legacy_freeze,
        )
        summary = stage_folds(run_dir, "main", args.workers, freeze["selections"])
        summary["preconditions"] = preconditions

    stage_path = run_dir / f"stage_{args.stage}.json"
    prep._write_json(summary, stage_path)
    print(json.dumps(summary, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
