"""The q2..q5 quartet: four independent scalar ``reg:squarederror`` boosters (R14, R30, R31, R43).

Adapted from ``FEWSNETGeoXGBExperiment/src/model/native_xgb.py`` (``raw``,
``sha``, ``from_raw``, ``base_score``, ``resolved_config``, ``prefix_identity``,
``continue_booster``): fixed-round ``xgb.train``; a local model is a
continuation of a fresh copy of its matching immutable global booster that
appends exactly L rounds once; the global's bytes, trees, base score and prefix
margins are verified unchanged. Changes: scalar regression instead of
four-class softprob, unit weights, explicit global ``base_score=0.5``, constant
targets fitted normally (recorded), and an explicit finite/shape check on every
prediction (R41). Feature NaN is passed to XGBoost as missing; an infinity in X
is a technical error here because P1 already guarantees finite-or-NaN inputs.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import xgboost as xgb

from ipcch_climate_geoxgb.errors import ContractError, TechnicalError

TARGETS = ("q2", "q3", "q4", "q5")
PROBE_ROWS = 256


def global_params(contract: dict, gid: str) -> tuple[dict, int]:
    model = contract["model"]
    recipe = model["global_recipes"][gid]
    params = {
        **model["fixed"],
        "objective": model["objective"],
        **model["global_common"],
        "max_depth": recipe["max_depth"],
        "base_score": model["global_base_score"],
    }
    return params, int(recipe["rounds"])


def local_params(contract: dict, lid: str) -> tuple[dict, int]:
    model = contract["model"]
    recipe = model["local_recipes"][lid]
    params = {**model["fixed"], "objective": model["objective"], **model["local_common"], "max_depth": recipe["max_depth"]}
    return params, int(recipe["appended_rounds"])


def check_X(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise TechnicalError(f"model input must be 2-D, got {X.shape}")
    if np.isinf(X).any():
        raise TechnicalError("model input contains an infinity (P1 guarantees finite-or-NaN)")
    return X


def dmatrix(X, y=None, nthread: int = 4) -> xgb.DMatrix:
    if y is not None:
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1 or len(y) != len(X) or not np.isfinite(y).all():
            raise TechnicalError("target vector must be finite, 1-D and aligned with X")
    return xgb.DMatrix(check_X(X), label=y, missing=np.nan, nthread=nthread)


def raw(booster: xgb.Booster) -> bytes:
    return bytes(booster.save_raw("ubj"))


def sha(booster: xgb.Booster) -> str:
    return hashlib.sha256(raw(booster)).hexdigest()


def from_raw(payload: bytes) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(bytearray(payload))
    return booster


def base_score(booster: xgb.Booster) -> str:
    return json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"]


def resolved_config(booster: xgb.Booster) -> dict:
    """The booster's resolved configuration captured AT FIT TIME.

    A reloaded UBJ model can report reset non-model training defaults (e.g.
    max_depth 6), so the record keeps this fit-time snapshot and its digest
    rather than a later reload's view.
    """
    return json.loads(booster.save_config())


def config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prefix_identity(booster: xgb.Booster, rounds: int | None = None) -> dict:
    """Digest of the first ``rounds`` trees (structure, splits, missing directions,
    leaf values; positional ``id`` excluded) plus the base score."""
    model = json.loads(bytes(booster.save_raw("json")))["learner"]
    trees = model["gradient_booster"]["model"]["trees"]
    total = booster.num_boosted_rounds()
    rounds = total if rounds is None else int(rounds)
    if rounds > total or len(trees) != total:
        raise TechnicalError("tree count does not match boosted rounds for a scalar booster")
    payload = {
        "trees": [{k: v for k, v in t.items() if k != "id"} for t in trees[:rounds]],
        "base_score": model["learner_model_param"]["base_score"],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"rounds": rounds, "sha256": digest}


def predict_scalar(booster: xgb.Booster, X) -> np.ndarray:
    X = check_X(X)
    if len(X) == 0:
        return np.zeros(0, dtype=np.float64)
    out = np.asarray(booster.predict(dmatrix(X)), dtype=np.float64)
    if out.shape != (len(X),):
        raise TechnicalError(f"booster returned shape {out.shape}, expected ({len(X)},)")
    if not np.isfinite(out).all():
        raise TechnicalError("raw share prediction is NaN/Inf (R41)")
    return out


def _target_record(y: np.ndarray) -> dict:
    constant = bool(np.all(y == y[0]))
    return {
        "rows": int(len(y)),
        "constant": constant,
        "constant_value": float(y[0]) if constant else None,
        "mean": float(y.mean()),
        "y_sha256": hashlib.sha256(np.ascontiguousarray(y, dtype=np.float64).tobytes()).hexdigest(),
    }


def fit_global(X, y, params: dict, rounds: int) -> tuple[xgb.Booster, dict]:
    """Fresh scalar booster; unit weights; constant targets fitted normally (R31)."""
    y = np.asarray(y, dtype=np.float64)
    if len(y) == 0:
        raise ContractError("empty global fitting pool (R40 stop)")
    try:
        booster = xgb.train(params, dmatrix(X, y), num_boost_round=rounds)
    except xgb.core.XGBoostError as error:
        raise TechnicalError(f"global fit raised: {error}") from error
    if booster.num_boosted_rounds() != rounds:
        raise TechnicalError("global booster has the wrong number of rounds")
    record = {
        "kind": "global",
        "rounds_total": rounds,
        "params": params,
        "resolved_config": resolved_config(booster),
        "resolved_config_sha256": config_digest(resolved_config(booster)),
        "base_score": base_score(booster),
        "structure_sha256": prefix_identity(booster)["sha256"],
        "booster_sha256": sha(booster),
        "weights": "unit",
        **_target_record(y),
    }
    return booster, record


def continue_local(parent_bytes: bytes, X, y, params: dict, rounds: int) -> tuple[xgb.Booster, dict]:
    """Append exactly ``rounds`` rounds to a fresh copy of an immutable global booster."""
    if "base_score" in params or "process_type" in params or "updater" in params:
        raise ContractError("local continuation must not override base_score or refresh trees")
    y = np.asarray(y, dtype=np.float64)
    if len(y) == 0:
        raise ContractError("empty local fitting pool")
    parent = from_raw(parent_bytes)
    n0 = parent.num_boosted_rounds()
    parent_prefix = prefix_identity(parent)
    try:
        child = xgb.train(params, dmatrix(X, y), num_boost_round=rounds, xgb_model=from_raw(parent_bytes))
    except xgb.core.XGBoostError as error:
        raise TechnicalError(f"local continuation raised: {error}") from error
    problems = []
    if raw(parent) != parent_bytes:
        problems.append("global booster bytes changed")
    if child.num_boosted_rounds() != n0 + rounds:
        problems.append(f"rounds {child.num_boosted_rounds()} != {n0} + {rounds}")
    if base_score(child) != base_score(parent):
        problems.append("base score changed")
    child_prefix = prefix_identity(child, n0)
    if child_prefix["sha256"] != parent_prefix["sha256"]:
        problems.append("local model's first rounds differ from the global booster")
    probe = check_X(X)[:PROBE_ROWS]
    if len(probe):
        prefix_margin = child.predict(dmatrix(probe), output_margin=True, iteration_range=(0, n0))
        if not np.array_equal(prefix_margin, parent.predict(dmatrix(probe), output_margin=True)):
            problems.append("local prefix does not reproduce the global margins")
    if problems:
        raise TechnicalError(f"global prefix preservation violated: {problems}")
    record = {
        "kind": "local",
        "parent_booster_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "parent_rounds": n0,
        "rounds_added": rounds,
        "rounds_total": n0 + rounds,
        "params": params,
        "resolved_config": resolved_config(child),
        "resolved_config_sha256": config_digest(resolved_config(child)),
        "base_score": base_score(child),
        "parent_structure_sha256": parent_prefix["sha256"],
        "child_prefix_structure_sha256": child_prefix["sha256"],
        "structure_sha256": prefix_identity(child)["sha256"],
        "booster_sha256": sha(child),
        "weights": "unit",
        **_target_record(y),
    }
    return child, record


class Quartet:
    """Four boosters that are always used together (atomic routing)."""

    def __init__(self, boosters: dict[str, bytes], records: dict[str, dict]):
        if set(boosters) != set(TARGETS) or set(records) != set(TARGETS):
            raise TechnicalError(f"a quartet needs exactly {TARGETS}, got {sorted(boosters)}")
        self.payloads = dict(boosters)
        self.records = dict(records)
        self._loaded = {q: from_raw(boosters[q]) for q in TARGETS}

    def predict_raw(self, X) -> np.ndarray:
        """(n, 4) raw q2..q5 predictions; any NaN/Inf or shape error stops (R41)."""
        X = check_X(X)
        if not len(X):
            return np.zeros((0, 4))
        columns = []
        for q in TARGETS:
            try:
                columns.append(predict_scalar(self._loaded[q], X))
            except Exception as error:
                error.add_note(f"target {q} prediction")
                raise
        return np.column_stack(columns)

    def booster_shas(self) -> dict:
        return {q: hashlib.sha256(self.payloads[q]).hexdigest() for q in TARGETS}


def fit_global_quartet(X, Y, params: dict, rounds: int) -> Quartet:
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[1] != 4:
        raise TechnicalError(f"targets must be (n, 4), got {Y.shape}")
    boosters, records = {}, {}
    for k, q in enumerate(TARGETS):
        try:
            booster, record = fit_global(X, Y[:, k], params, rounds)
        except Exception as error:
            error.add_note(f"target {q} global fit")
            raise
        boosters[q], records[q] = raw(booster), {**record, "target": q}
    return Quartet(boosters, records)


def continue_local_quartet(global_quartet: Quartet, X, Y, params: dict, rounds: int) -> Quartet:
    """Each target continues ONLY its own global booster (no cross-target sharing).

    The input must be a true global quartet: continuing an already-local
    quartet would accumulate increments (e.g. 220 -> 240 rounds), which the
    shared-root contract forbids.
    """
    kinds = {q: global_quartet.records[q].get("kind") for q in TARGETS}
    if set(kinds.values()) != {"global"}:
        raise TechnicalError(f"local continuation requires a global quartet, got kinds {kinds}")
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[1] != 4:
        raise TechnicalError(f"targets must be (n, 4), got {Y.shape}")
    boosters, records = {}, {}
    for k, q in enumerate(TARGETS):
        try:
            child, record = continue_local(global_quartet.payloads[q], X, Y[:, k], params, rounds)
        except Exception as error:
            error.add_note(f"target {q} local continuation")
            raise
        boosters[q], records[q] = raw(child), {**record, "target": q}
    return Quartet(boosters, records)


def check_aligned(predicted_keys: np.ndarray, requested_keys: np.ndarray) -> None:
    """Prediction rows must carry exactly the requested keys in the requested order."""
    if not np.array_equal(np.asarray(predicted_keys), np.asarray(requested_keys)):
        raise TechnicalError("prediction keys are misaligned with the requested keys (R41)")
