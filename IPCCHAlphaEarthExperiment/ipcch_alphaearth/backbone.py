"""Fixed pooled XGB quartets (B0/B1) and an exact-identity model store (design sections 6-7, 13; D21).

One scalar XGBRegressor per q (inherited compact recipes, seed 42, CPU hist, no early stopping), weighted by the
unnormalized D7 weights. A fit identity binds arm/H/block/fit origin, ordered fitting row refs and keys, the
actual X/Y/weight slices, feature schema, resolved parameters, runtime and the fitting-source code identity.
Store layout: ``<models>/<digest[:2]>/<digest>/{q2..q5}.ubj + record.json``. A hit requires the identical stored
identity and matching booster bytes; a partial/corrupt/conflicting entry is a TechnicalError (never refit
under the same identity). ``read_only`` (replay) refuses any fit.

Store discipline adapted from IPCCHYearlyGeoXGBExperiment/ipcch_yearly_xgb/modelstore.py at 6798df2.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ipcch_alphaearth.artifacts import array_digest, canonical, digest, sha256_bytes
from ipcch_alphaearth.errors import TechnicalError

TARGETS = ("q2", "q3", "q4", "q5")


def xgb_params(contract: dict, q: str) -> dict:
    """Effective XGBRegressor parameters for target q (recipe + fixed fields)."""
    recipe = dict(contract["backbone"]["recipes"][q])
    fixed = contract["backbone"]["fixed"]
    return {**recipe, "objective": fixed["objective"], "tree_method": fixed["tree_method"], "device": fixed["device"],
            "booster": fixed["booster"], "random_state": int(fixed["seed"]), "n_jobs": int(fixed["n_jobs"]),
            "scale_pos_weight": float(fixed["scale_pos_weight"])}


def fit_scalar(X: np.ndarray, y: np.ndarray, w: np.ndarray, params: dict):
    import xgboost as xgb  # noqa: PLC0415

    model = xgb.XGBRegressor(**params)
    model.fit(X, y, sample_weight=w, verbose=False)
    return model.get_booster()


def booster_from_raw(raw: bytes):
    import xgboost as xgb  # noqa: PLC0415

    booster = xgb.Booster()
    booster.load_model(bytearray(raw))
    return booster


def predict_booster(booster, X: np.ndarray, n_threads: int) -> np.ndarray:
    import xgboost as xgb  # noqa: PLC0415

    booster.set_param({"nthread": n_threads})
    out = booster.predict(xgb.DMatrix(np.ascontiguousarray(X, dtype=np.float64), missing=np.nan, nthread=n_threads))
    out = np.asarray(out, dtype=np.float64)
    if not np.isfinite(out).all():
        raise TechnicalError("nonfinite backbone prediction")
    return out


def fit_identity(*, arm: str, horizon: int, block_id: str, role: str, fit_origin: int, rows: np.ndarray, keys: np.ndarray,
                 X: np.ndarray, Y: np.ndarray, w: np.ndarray, features: list, contract: dict, env: dict, source: dict) -> dict:
    """Canonical identity of one quartet request, built from the actual fitting slices."""
    if len(rows) == 0:
        raise TechnicalError(f"empty backbone fitting pool for {arm} {block_id}: stop for review (no invented labels)")
    return {
        "protocol": "ipcch-alphaearth-backbone-v1",
        "arm": arm, "horizon": int(horizon), "block_id": block_id, "role": role, "fit_origin": int(fit_origin),
        "fit_pool": "valid canonical keys with target U <= fit origin (D29), all eligible history, no lower bound",
        "n_rows": int(len(rows)),
        "fit_rows": array_digest(np.asarray(rows, dtype=np.int64)),
        "fit_keys": array_digest(np.asarray(keys, dtype=np.int64)),
        "X_sha256": array_digest(np.ascontiguousarray(X, dtype=np.float64)),
        "X_shape": list(X.shape),
        "y_sha256": {q: array_digest(np.ascontiguousarray(Y[:, k], dtype=np.float64)) for k, q in enumerate(TARGETS)},
        "weights": {"formula": "2**(-(fit_origin - U)/24)", "sha256": array_digest(np.asarray(w, dtype=np.float64)),
                    "sum": float(np.sum(w))},
        "features_sha256": digest(list(features)), "n_features": len(features),
        "params": {q: xgb_params(contract, q) for q in TARGETS},
        "env": env, "source": source,
    }


def check_slices(identity: dict, X: np.ndarray, Y: np.ndarray, w: np.ndarray, contract: dict) -> None:
    """The actual fitting slices must be exactly the ones the identity was built from (before reuse or fit)."""
    bad = []
    if list(np.shape(X)) != identity["X_shape"] or array_digest(np.ascontiguousarray(X, dtype=np.float64)) != identity["X_sha256"]:
        bad.append("X")
    if len(Y) != identity["n_rows"] or any(array_digest(np.ascontiguousarray(Y[:, k], dtype=np.float64)) != identity["y_sha256"][q]
                                           for k, q in enumerate(TARGETS)):
        bad.append("Y")
    if array_digest(np.asarray(w, dtype=np.float64)) != identity["weights"]["sha256"]:
        bad.append("weights")
    if identity["params"] != {q: xgb_params(contract, q) for q in TARGETS}:
        bad.append("params")
    if bad:
        raise TechnicalError(f"fitting slices differ from the identity ({bad}); refusing to fit or reuse under it")


class ModelStore:
    def __init__(self, root: Path, read_only: bool = False):
        self.root = Path(root)
        self.read_only = read_only
        self.ledger: list[dict] = []

    def _dir(self, ident_digest: str) -> Path:
        return self.root / ident_digest[:2] / ident_digest

    def get_or_fit(self, identity: dict, X: np.ndarray, Y: np.ndarray, w: np.ndarray, contract: dict, purpose: str) -> tuple[dict, dict]:
        """Return ({q: booster}, record); fit once per identity (never in read-only mode)."""
        check_slices(identity, X, Y, w, contract)
        d = digest(identity)
        path = self._dir(d)
        tmp = path.with_name(path.name + ".partial")
        if tmp.exists():
            raise TechnicalError(f"partial model entry exists for {d}; refusing to fit or overwrite (technical stop)")
        if path.exists():
            boosters, record = self._load(path, d, identity)
            self.ledger.append({"identity": d, "purpose": purpose, "action": "reuse", **_short(identity)})
            return boosters, record
        if self.read_only:
            raise TechnicalError(f"read-only store has no model for identity {d} ({identity['arm']} {identity['block_id']}); replay never fits")
        t0 = time.time()
        boosters, payloads, per_target = {}, {}, {}
        for k, q in enumerate(TARGETS):
            params = xgb_params(contract, q)
            tq = time.time()
            booster = fit_scalar(X, Y[:, k], w, params)
            raw = bytes(booster.save_raw("ubj"))
            config = json.loads(booster.save_config())
            payloads[q] = raw
            boosters[q] = booster
            per_target[q] = {"booster_sha256": sha256_bytes(raw), "rounds": booster.num_boosted_rounds(),
                             "base_score": config["learner"]["learner_model_param"]["base_score"],
                             "resolved_config_sha256": sha256_bytes(canonical(config).encode()),
                             "updater": config["learner"]["gradient_booster"].get("gbtree_train_param", {}).get("tree_method"),
                             "resolved_config": config, "y_constant": bool(np.all(Y[:, k] == Y[0, k])),
                             "seconds": round(time.time() - tq, 3)}
        record = {"identity_digest": d, "identity": identity, "targets": per_target, "fit_seconds": round(time.time() - t0, 3)}
        tmp.mkdir(parents=True)
        for q, raw in payloads.items():
            (tmp / f"{q}.ubj").write_bytes(raw)
        (tmp / "record.json").write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
        tmp.rename(path)
        self.ledger.append({"identity": d, "purpose": purpose, "action": "fit", "seconds": record["fit_seconds"], **_short(identity)})
        return boosters, record

    def _load(self, path: Path, d: str, expected_identity: dict | None = None) -> tuple[dict, dict]:
        """Verify directory name, stored digest, recomputed identity digest, booster bytes and recoverable fit fields."""
        rec_path = path / "record.json"
        if not rec_path.exists():
            raise TechnicalError(f"model entry {d} lacks record.json (partial/corrupt)")
        record = json.loads(rec_path.read_text(encoding="utf-8"))
        if path.name != d or record.get("identity_digest") != d or digest(record.get("identity")) != d:
            raise TechnicalError(f"model entry {path.name}: directory, stored digest and recomputed identity digest disagree")
        if expected_identity is not None and canonical(record["identity"]) != canonical(expected_identity):
            raise TechnicalError(f"model entry {d} stores a conflicting identity")
        boosters = {}
        for q in TARGETS:
            t = record["targets"][q]
            raw = (path / f"{q}.ubj").read_bytes() if (path / f"{q}.ubj").exists() else None
            if raw is None or sha256_bytes(raw) != t["booster_sha256"]:
                raise TechnicalError(f"model entry {d} booster {q} missing or does not match its record")
            if sha256_bytes(canonical(t["resolved_config"]).encode()) != t["resolved_config_sha256"]:
                raise TechnicalError(f"model entry {d} {q}: resolved config differs from its fit-time digest")
            if t["resolved_config"]["learner"]["learner_model_param"]["base_score"] != t["base_score"]:
                raise TechnicalError(f"model entry {d} {q}: recorded base score differs from the resolved config")
            booster = booster_from_raw(raw)
            reloaded = json.loads(booster.save_config())
            if booster.num_boosted_rounds() != t["rounds"] or reloaded["learner"]["learner_model_param"]["base_score"] != t["base_score"]:
                raise TechnicalError(f"model entry {d} {q}: booster rounds/base score differ from the record")
            boosters[q] = booster
        return boosters, record

    def load_by_digest(self, d: str) -> tuple[dict, dict]:
        return self._load(self._dir(d), d)


def predict_quartet(boosters: dict, X: np.ndarray, n_threads: int) -> np.ndarray:
    return np.column_stack([predict_booster(boosters[q], X, n_threads) for q in TARGETS])


def _short(identity: dict) -> dict:
    return {k: identity[k] for k in ("arm", "horizon", "block_id", "role", "fit_origin", "n_rows")}


def clear_partial(root: Path) -> list[str]:
    """List (never delete) partial entries; a human decides (technical-stop evidence)."""
    return [str(p) for p in Path(root).rglob("*.partial")]

