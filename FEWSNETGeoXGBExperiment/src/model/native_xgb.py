"""Native XGBoost boosters with frozen shared prefixes (design D4, D14).

Every booster is ``xgb.train`` on the fixed four-class axis. A global/root booster is a
fresh fit; a local/child booster is a continuation of an immutable parent through
``xgb_model=parent`` that appends a fixed number of rounds. The parent's trees, leaf
values, missing directions and base score are never refreshed or updated:
``continue_booster`` checks the parent is byte-unchanged, the round count, the base
score, that the child's first rounds are structurally identical to the parent
(``prefix_identity``: serialized trees incl. missing directions and leaf weights,
tree_info, base score) and that they reproduce the parent's margins exactly. Every
record stores the booster's resolved ``save_config()``.

Input contract (D14): dense float matrices, +/-inf -> NaN, NaN passed to XGBoost as
missing; no imputer, no pseudo rows, no sample or class weights.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import xgboost as xgb

from src.experiment.plan import N_CLASSES, booster_params

PROBE_ROWS = 256


def clean(X) -> np.ndarray:
    """Dense float64 copy with +/-inf set to NaN; finite values and true zeros kept."""
    X = np.array(X, dtype=np.float64, copy=True)
    if X.ndim != 2:
        raise ValueError("model input must be a 2-D matrix")
    X[np.isinf(X)] = np.nan
    return X


def dmatrix(X, y=None) -> xgb.DMatrix:
    if y is not None:
        y = np.asarray(y, dtype=np.int64)
        if y.size and (y.min() < 0 or y.max() >= N_CLASSES):
            raise ValueError("labels must be class codes 0..3")
    return xgb.DMatrix(clean(X), label=y, missing=np.nan, nthread=4)


def check_sample_weight(sample_weight, n: int) -> np.ndarray:
    """D37: validated float32 row weights (length n, all finite, all > 0)."""
    w = np.asarray(sample_weight, dtype=np.float64)
    if w.ndim != 1 or len(w) != n:
        raise ValueError(f"sample_weight length {w.shape} != {n} fitting rows")
    if not np.all(np.isfinite(w)) or not np.all(w > 0):
        raise ValueError("sample_weight must be finite and > 0")
    w32 = w.astype(np.float32)
    if not np.all(np.isfinite(w32)) or not np.all(w32 > 0):
        raise ValueError("sample_weight not representable as positive finite float32")
    return w32


def weight_record(w32: np.ndarray) -> dict:
    w = w32.astype(np.float64)
    return {"dtype": str(w32.dtype), "sha256": hashlib.sha256(np.ascontiguousarray(w32).tobytes()).hexdigest(),
            "n": int(len(w)), "sum": float(w.sum()), "min": float(w.min()), "max": float(w.max()),
            "kish_ess": float(w.sum() ** 2 / np.sum(w ** 2))}


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


def num_class(booster: xgb.Booster) -> int:
    return int(json.loads(booster.save_config())["learner"]["learner_model_param"]["num_class"])


def resolved_config(booster: xgb.Booster) -> dict:
    """The booster's own parsed configuration (every resolved/default parameter)."""
    return json.loads(booster.save_config())


#: parent booster sha -> prefix identity, so a parent reused by many children is parsed once.
_PREFIX_CACHE: dict = {}


def prefix_identity(booster: xgb.Booster, rounds: int | None = None) -> dict:
    """Exact structural identity of the first ``rounds`` rounds (all rounds by default).

    Digest of the serialized trees (split indices/conditions, children, default_left
    missing directions, leaf/base weights, every stored field except the positional
    ``id``), their ``tree_info`` class slots, and the base score. Two boosters share a
    frozen prefix iff these digests are equal; the verifier recomputes them from the
    saved UBJ files.
    """
    model = json.loads(bytes(booster.save_raw("json")))["learner"]
    gb = model["gradient_booster"]["model"]
    total = booster.num_boosted_rounds()
    rounds = total if rounds is None else int(rounds)
    per_round = len(gb["trees"]) // total if total else N_CLASSES
    n_trees = rounds * per_round
    if rounds > total or len(gb["trees"]) != total * per_round:
        raise RuntimeError("tree count does not match the boosted rounds")
    payload = {"trees": [{k: v for k, v in t.items() if k != "id"} for t in gb["trees"][:n_trees]],
               "tree_info": gb["tree_info"][:n_trees],
               "base_score": model["learner_model_param"]["base_score"],
               "num_class": model["learner_model_param"]["num_class"]}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"rounds": rounds, "trees": n_trees, "sha256": digest}


def proba(booster: xgb.Booster, X) -> np.ndarray:
    """(n, 4) probabilities on the fixed class axis."""
    if len(X) == 0:
        return np.zeros((0, N_CLASSES))
    out = booster.predict(dmatrix(X))
    if out.ndim != 2 or out.shape[1] != N_CLASSES:
        raise RuntimeError(f"booster returned shape {out.shape}, not (n, 4)")
    return out.astype(np.float64)


def fit_global(X, y, config: dict, sample_weight=None) -> tuple[xgb.Booster, dict]:
    """Fresh fixed-four booster on real rows only.

    ``sample_weight`` (D37, default None = unchanged behaviour and record): validated row weights,
    passed to XGBoost as float32 and recorded in a ``sample_weight`` block."""
    params, rounds = booster_params(config)
    y = np.asarray(y, dtype=np.int64)
    if len(y) == 0:
        raise ValueError("empty fitting pool")
    w32 = None if sample_weight is None else check_sample_weight(sample_weight, len(y))
    dm = dmatrix(X, y)
    if w32 is not None:
        dm.set_weight(w32)
    booster = xgb.train(params, dm, num_boost_round=rounds)
    if booster.num_boosted_rounds() != rounds or num_class(booster) != N_CLASSES:
        raise RuntimeError("fresh booster has the wrong rounds or class axis")
    record = {"kind": "fresh", "rounds_total": rounds, "rounds_added": rounds,
              "params": params, "resolved_config": resolved_config(booster),
              "base_score": base_score(booster), "rows": int(len(y)),
              "class_counts": [int(np.sum(y == k)) for k in range(N_CLASSES)],
              "structure_sha256": prefix_identity(booster)["sha256"],
              "booster_sha256": sha(booster)}
    if w32 is not None:
        record["sample_weight"] = weight_record(w32)
    return booster, record


def continue_booster(parent: xgb.Booster, X, y, config: dict) -> tuple[xgb.Booster, dict]:
    """Append ``config['rounds']`` rounds to an immutable parent (default process_type)."""
    params, rounds = booster_params(config)
    if "process_type" in params or "updater" in params or "base_score" in params:
        raise ValueError("continuation must not refresh/update trees or override base_score")
    y = np.asarray(y, dtype=np.int64)
    if len(y) == 0:
        raise ValueError("empty fitting pool")
    before = raw(parent)
    n0 = parent.num_boosted_rounds()
    child = xgb.train(params, dmatrix(X, y), num_boost_round=rounds, xgb_model=parent)
    probe = np.asarray(X)[:PROBE_ROWS]
    problems = []
    if raw(parent) != before:
        problems.append("parent booster changed")
    if child.num_boosted_rounds() != n0 + rounds:
        problems.append(f"rounds {child.num_boosted_rounds()} != {n0} + {rounds}")
    if base_score(child) != base_score(parent) or num_class(child) != N_CLASSES:
        problems.append("base score or class axis changed")
    parent_sha = hashlib.sha256(before).hexdigest()
    if parent_sha not in _PREFIX_CACHE:
        _PREFIX_CACHE.clear()
        _PREFIX_CACHE[parent_sha] = prefix_identity(parent)
    parent_prefix = _PREFIX_CACHE[parent_sha]
    child_prefix = prefix_identity(child, n0)
    if child_prefix["sha256"] != parent_prefix["sha256"]:
        problems.append("child's first rounds differ structurally from the parent")
    prefix = child.predict(dmatrix(probe), output_margin=True, iteration_range=(0, n0))
    if not np.array_equal(prefix, parent.predict(dmatrix(probe), output_margin=True)):
        problems.append("child prefix does not reproduce the parent margins")
    if problems:
        raise RuntimeError(f"frozen-prefix continuation violated: {problems}")
    return child, {"kind": "continuation", "parent_sha256": parent_sha,
                   "parent_rounds": n0, "rounds_added": rounds, "rounds_total": n0 + rounds,
                   "params": params, "resolved_config": resolved_config(child),
                   "base_score": base_score(child), "rows": int(len(y)),
                   "class_counts": [int(np.sum(y == k)) for k in range(N_CLASSES)],
                   "parent_structure_sha256": parent_prefix["sha256"],
                   "child_prefix_structure_sha256": child_prefix["sha256"],
                   "structure_sha256": prefix_identity(child)["sha256"],
                   "prefix_check": (f"serialized trees/tree_info/base_score of the first {n0} rounds "
                                    f"equal the parent's; parent bytes unchanged; prefix margins equal "
                                    f"on {len(probe)} fitting rows"),
                   "booster_sha256": sha(child)}


def support(y, areas, months) -> dict:
    """Real-row support of one pool: rows, areas, label dates and four class counts."""
    y = np.asarray(y, dtype=np.int64)
    counts = [int(np.sum(y == k)) for k in range(N_CLASSES)]
    return {"rows": int(len(y)), "areas": int(np.unique(areas).size),
            "dates": int(np.unique(months).size), "classes": int(sum(c > 0 for c in counts)),
            "class_counts": counts}


def meets(record: dict, floor: dict) -> bool:
    return all(record[k] >= v for k, v in floor.items())


def keys_sha(areas, months) -> str:
    keys = np.column_stack([np.asarray(areas, dtype=np.int64), np.asarray(months, dtype=np.int64)])
    return hashlib.sha256(np.ascontiguousarray(keys).tobytes()).hexdigest()


#: Stage 1 child increment sources: ``parent`` (D4, accumulate ancestor increments) or
#: ``root`` (D28 / experiment-plan A3, one L increment on the shared root per child).
INCREMENT_SOURCES = ("parent", "root")


class XGBmodel:
    """Stage 1 checkpoint store with the RFmodel interface used by partition().

    ``load(b)`` makes branch ``b`` current; ``train`` then CONTINUES the current booster (parent
    mode) or the installed ROOT once (root mode, D28; ``path_rounds_added`` is then that single
    increment, while ``path_selection_rounds`` carries the search budget)
    (it never fits from scratch: the root is installed once with ``set_root``), and
    ``save(c)`` stores the current booster as branch ``c``. A parent copied to a child
    (``load(parent); save(child)``) therefore carries the parent's exact bytes.
    """

    name = "XGB"
    type = "static"
    mode = "classification"

    def __init__(self, path, local_config: dict, num_class=N_CLASSES, increment_source="parent"):
        if increment_source not in INCREMENT_SOURCES:
            raise ValueError(f"increment_source must be one of {INCREMENT_SOURCES}")
        self.increment_source = increment_source
        self._root = None  # (booster bytes, record) of the installed root, for root mode
        self.path = str(path)
        self.local_config = dict(local_config)
        self.num_class = num_class
        self.booster = None
        self.fit_record = None
        self.fit_log = []
        self.saved_log = []
        self._store = {}

    def _file(self, branch_id) -> str:
        return os.path.join(self.path, f"xgb_{branch_id or 'root'}")

    def set_root(self, booster: xgb.Booster, record: dict) -> None:
        self.booster = booster
        self.fit_record = {**record, "trained_under": None, "path_rounds_added": 0,
                           "increment_source": self.increment_source}
        if self.increment_source == "root":
            self.fit_record.update(path_selection_rounds=0, actual_local_rounds=0,
                                   shared_source=sha(booster), routing_parent=None)
            self._root = (raw(booster), dict(self.fit_record))
        self.fit_log.append(dict(self.fit_record, saved_as=None))
        self.save("")

    def train(self, X, y, branch_id, meta=None):
        if self.booster is None or self.fit_record is None:
            raise RuntimeError("no parent loaded: Stage 1 children are continuations only")
        parent = self.fit_record
        if parent.get("loaded_as") != (branch_id or ""):
            raise RuntimeError(f"train under {branch_id!r} but loaded {parent.get('loaded_as')!r}")
        if self.increment_source == "root":
            # D28 / A3: every new child continues the SHARED ROOT once; the current parent
            # stays the E1/E2 comparison and the fallback, but its increment is not inherited.
            root_payload, root_record = self._root
            child, record = continue_booster(from_raw(root_payload), X, y, self.local_config)
            self.booster = child
            self.fit_record = {**record, "trained_under": branch_id or "",
                               "path_rounds_added": record["rounds_added"],
                               "increment_source": "root", "shared_source": root_record["shared_source"],
                               "routing_parent": {"branch_id": branch_id or "",
                                                  "booster_sha256": parent.get("booster_sha256")},
                               "actual_local_rounds": record["rounds_added"],
                               "path_selection_rounds": int(parent["path_selection_rounds"]) + record["rounds_added"],
                               **(meta or {})}
            self.fit_log.append(dict(self.fit_record))
            return
        child, record = continue_booster(self.booster, X, y, self.local_config)
        self.booster = child
        self.fit_record = {**record, "trained_under": branch_id or "",
                           "path_rounds_added": parent["path_rounds_added"] + record["rounds_added"],
                           **(meta or {})}
        self.fit_log.append(dict(self.fit_record))

    def predict(self, X, prob=False):
        p = proba(self.booster, X)
        return p if prob else np.argmax(p, axis=1).astype(np.int64)

    def save(self, branch_id):
        payload = raw(self.booster)
        record = {k: v for k, v in self.fit_record.items() if k != "loaded_as"}
        os.makedirs(self.path, exist_ok=True)
        with open(self._file(branch_id) + ".ubj", "wb") as handle:
            handle.write(payload)
        with open(self._file(branch_id) + ".json", "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=1, default=str)
        self._store[branch_id or ""] = (payload, record)
        self.saved_log.append({"saved_as": branch_id or "root", **record})

    def load(self, branch_id, fresh=True):
        key = branch_id or ""
        if key not in self._store:
            with open(self._file(key) + ".ubj", "rb") as handle:
                payload = handle.read()
            with open(self._file(key) + ".json", encoding="utf-8") as handle:
                record = json.load(handle)
            self._store[key] = (payload, record)
        payload, record = self._store[key]
        self.booster = from_raw(payload)
        self.fit_record = {**record, "loaded_as": key}

    def path_rounds(self, branch_id) -> int:
        self.load(branch_id)
        if self.increment_source == "root":
            # Search-opportunity budget (A3): not the booster's actual 0/20 local rounds.
            return int(self.fit_record["path_selection_rounds"])
        return int(self.fit_record["path_rounds_added"])

    def predict_georf(self, X, X_group, s_branch, X_branch_id=None):
        from src.helper.helper import get_X_branch_id_by_group
        if X_branch_id is None:
            X_branch_id = get_X_branch_id_by_group(X_group, s_branch)
        out = np.zeros(X.shape[0], dtype=np.int64)
        for branch_id in np.unique(X_branch_id):
            rows = np.where(X_branch_id == branch_id)[0]
            self.load(branch_id)
            out[rows] = self.predict(X[rows])
        return out

    def predict_proba_georf(self, X, X_group, s_branch, X_branch_id=None):
        from src.helper.helper import get_X_branch_id_by_group
        if X_branch_id is None:
            X_branch_id = get_X_branch_id_by_group(X_group, s_branch)
        out = np.zeros((X.shape[0], self.num_class))
        for branch_id in np.unique(X_branch_id):
            rows = np.where(X_branch_id == branch_id)[0]
            self.load(branch_id)
            out[rows] = self.predict(X[rows], prob=True)
        return out
