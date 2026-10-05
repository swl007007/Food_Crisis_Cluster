"""Exact-identity quartet store and request ledger (R48).

A quartet is fitted once per complete identity and reused afterwards. The
identity (built by the caller) binds stage/fit scope, H, fitting origin/window
or Stage1 F membership, the ordered fitting keys, X/Y digests, unit weights,
feature schema, availability policy, recipe and parameters, seed and the
code/numerical environment; a local identity also binds region membership and
its global quartet's booster digests. Entries live under
``<root>/<digest[:2]>/<digest>/`` as four UBJ boosters plus ``record.json``.

A hit requires an equal stored identity and booster bytes matching the stored
digests. An absent entry is fitted. A partial, corrupt or conflicting entry
raises ``TechnicalError`` (R41): it is never overwritten, refitted or treated
as a fallback. Reuse returns the raw model only; gate decisions and routed
outputs are never cached here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Callable

import numpy as np

from ipcch_climate_geoxgb.errors import TechnicalError
from ipcch_climate_geoxgb.quartet import TARGETS, Quartet, base_score, from_raw, prefix_identity


def array_digest(array) -> str:
    """SHA256 of dtype, shape and C-ordered bytes (NaN bit patterns included).

    Arrays holding Python objects (including structured dtypes with object
    fields) are refused: their bytes are pointers, not values.
    """
    arr = np.ascontiguousarray(np.asarray(array))
    if arr.dtype.hasobject:
        raise TechnicalError("array_digest needs a numeric/fixed-width array, not one holding objects")
    header = f"{arr.dtype.str}|{arr.shape}|".encode()
    return hashlib.sha256(header + arr.tobytes()).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def target_digests(Y: np.ndarray) -> dict:
    """Per-target SHA256 of the float64 fitting targets (same bytes as fit records)."""
    Y = np.asarray(Y, dtype=np.float64)
    return {q: hashlib.sha256(np.ascontiguousarray(Y[:, k]).tobytes()).hexdigest() for k, q in enumerate(TARGETS)}


def identity_digest(identity: dict) -> str:
    return hashlib.sha256(canonical(identity).encode()).hexdigest()


REQUIRED_FIELDS = {
    "target", "kind", "rounds_total", "params", "resolved_config", "resolved_config_sha256", "base_score",
    "structure_sha256", "booster_sha256", "weights", "rows", "constant", "constant_value", "mean", "y_sha256",
}
LOCAL_FIELDS = {
    "parent_booster_sha256", "parent_rounds", "rounds_added",
    "parent_structure_sha256", "child_prefix_structure_sha256",
}
#: Identity fields every fit identity must bind (R48): requested parameters, rounds,
#: fitting row count and the per-target fitting-target digests.
IDENTITY_FIELDS = ("params", "rounds", "n_rows", "y_sha256")
_SHA = re.compile(r"^[0-9a-f]{64}$")
#: requested-param name -> (resolved-config section, key, kind)
_RESOLVED = {
    "max_depth": ("tree", "max_depth", "int"),
    "eta": ("tree", "eta", "f32"),
    "min_child_weight": ("tree", "min_child_weight", "f32"),
    "reg_lambda": ("tree", "reg_lambda", "f32"),
    "reg_alpha": ("tree", "reg_alpha", "f32"),
    "subsample": ("tree", "subsample", "f32"),
    "colsample_bytree": ("tree", "colsample_bytree", "f32"),
    "gamma": ("tree", "gamma", "f32"),
    "max_delta_step": ("tree", "max_delta_step", "f32"),
    "max_bin": ("tree", "max_bin", "int"),
    "grow_policy": ("tree", "grow_policy", "str"),
    "tree_method": ("gbtree", "tree_method", "str"),
    "num_parallel_tree": ("model", "num_parallel_tree", "int"),
    "seed": ("generic", "seed", "int"),
    "nthread": ("generic", "nthread", "int"),
    "device": ("generic", "device", "str"),
    "booster": ("train", "booster", "str"),
    "objective": ("train", "objective", "str"),
}


def _resolved_matches(params: dict, config: dict) -> list[str]:
    """Fields of the fit-time resolved config that disagree with the requested params."""
    try:
        learner = config["learner"]
        gb = learner["gradient_booster"]
        sections = {"tree": gb["tree_train_param"], "gbtree": gb["gbtree_train_param"],
                    "model": gb["gbtree_model_param"], "generic": learner["generic_param"],
                    "train": learner["learner_train_param"]}
    except (KeyError, TypeError):
        return ["resolved_config structure"]
    bad = []
    for name, (section, key, kind) in _RESOLVED.items():
        if name not in params:
            bad.append(f"requested {name} missing")
            continue
        value = sections[section].get(key)
        if value is None:
            bad.append(f"resolved {key} missing")
        elif kind == "f32":
            if np.float32(float(value)) != np.float32(params[name]):
                bad.append(f"{key}={value} vs {params[name]}")
        elif kind == "int":
            if int(value) != int(params[name]):
                bad.append(f"{key}={value} vs {params[name]}")
        elif str(value) != str(params[name]):
            bad.append(f"{key}={value} vs {params[name]}")
    if learner.get("objective", {}).get("name") != params.get("objective"):
        bad.append("objective.name")
    return bad


def validate_fit_records(identity: dict, payloads: dict, records: dict) -> None:
    """Required per-target fitting evidence, checked against identity and boosters.

    Called before an entry is written and whenever one is loaded; any missing,
    mistyped or inconsistent field is a TechnicalError (never a refit).
    """
    missing_identity = [f for f in IDENTITY_FIELDS if f not in identity]
    if missing_identity:
        raise TechnicalError(f"fit identity lacks {missing_identity}")
    expected_kind = "local" if str(identity.get("scope", "")).endswith("-local") else "global"
    for q in TARGETS:
        record = records.get(q)
        if not isinstance(record, dict):
            raise TechnicalError(f"fit record for {q} is missing")
        required = REQUIRED_FIELDS | (LOCAL_FIELDS if record.get("kind") == "local" else set())
        missing = sorted(required - set(record))
        if missing:
            raise TechnicalError(f"fit record for {q} lacks {missing}")
        if record["target"] != q or record["kind"] != expected_kind:
            raise TechnicalError(f"fit record for {q} has target/kind {record['target']}/{record['kind']}")
        if not isinstance(record["resolved_config"], dict) or not isinstance(record["params"], dict):
            raise TechnicalError(f"{q}: params/resolved_config are not objects")
        if record["params"] != identity["params"]:
            raise TechnicalError(f"{q}: recorded requested params differ from the identity")
        rounds_field = "rounds_added" if expected_kind == "local" else "rounds_total"
        if record[rounds_field] != identity["rounds"]:
            raise TechnicalError(f"{q}: recorded rounds differ from the identity")
        if not isinstance(record["rows"], int) or isinstance(record["rows"], bool) or record["rows"] != identity["n_rows"]:
            raise TechnicalError(f"{q}: recorded fitting rows differ from the identity")
        if record["weights"] != "unit":
            raise TechnicalError(f"{q}: weights must be unit")
        if not (isinstance(record["y_sha256"], str) and _SHA.match(record["y_sha256"])) or \
                record["y_sha256"] != identity["y_sha256"].get(q):
            raise TechnicalError(f"{q}: fitting-target digest differs from the identity")
        if not isinstance(record["constant"], bool) or (record["constant"] != (record["constant_value"] is not None)):
            raise TechnicalError(f"{q}: constant-target fields are inconsistent")
        from ipcch_climate_geoxgb.quartet import config_digest  # noqa: PLC0415

        if config_digest(record["resolved_config"]) != record["resolved_config_sha256"]:
            raise TechnicalError(f"{q}: resolved config does not match its fit-time digest")
        bad = _resolved_matches(record["params"] if expected_kind == "local" else
                                {k: v for k, v in record["params"].items() if k != "base_score"},
                                record["resolved_config"])
        if bad:
            raise TechnicalError(f"{q}: fit-time resolved config disagrees with requested params: {bad[:5]}")
        if hashlib.sha256(payloads[q]).hexdigest() != record["booster_sha256"]:
            raise TechnicalError(f"fit record for {q} does not match its booster bytes")
        booster = from_raw(payloads[q])
        if booster.num_boosted_rounds() != record["rounds_total"]:
            raise TechnicalError(f"{q}: booster rounds differ from the fit record")
        if base_score(booster) != record["base_score"]:
            raise TechnicalError(f"{q}: booster base score differs from the fit record")
        if prefix_identity(booster)["sha256"] != record["structure_sha256"]:
            raise TechnicalError(f"{q}: booster structure differs from the fit record")
        if record["resolved_config"]["learner"].get("learner_model_param", {}).get("base_score") != record["base_score"]:
            raise TechnicalError(f"{q}: resolved config base score differs from the fit record")
        if expected_kind == "global" and "base_score" in record["params"]:
            if np.float32(float(record["base_score"])) != np.float32(record["params"]["base_score"]):
                raise TechnicalError(f"{q}: global base score differs from the requested base_score")
        if record["kind"] == "local":
            if record["rounds_total"] != record["parent_rounds"] + record["rounds_added"]:
                raise TechnicalError(f"{q}: local rounds do not equal global + appended")
            prefix = prefix_identity(booster, record["parent_rounds"])["sha256"]
            if not (prefix == record["child_prefix_structure_sha256"] == record["parent_structure_sha256"]):
                raise TechnicalError(f"{q}: local prefix does not match its global booster")
            parents = identity.get("global_boosters")
            if parents is not None and parents.get(q) != record["parent_booster_sha256"]:
                raise TechnicalError(f"{q}: local parent digest differs from the identity's global booster")


class ModelStore:
    def __init__(self, root: Path | str, ledger_path: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger_path = Path(ledger_path)
        self.ledger_path.touch(exist_ok=True)  # an empty ledger is evidence of zero requests
        self.counts = {"requests": 0, "hits": 0, "fits": 0, "failed": 0}

    def _dir(self, digest: str) -> Path:
        return self.root / digest[:2] / digest

    def _log(self, entry: dict) -> None:
        with open(self.ledger_path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical(entry) + "\n")

    def _load(self, digest: str, identity: dict) -> Quartet:
        directory = self._dir(digest)
        record_path = directory / "record.json"
        if not record_path.is_file():
            raise TechnicalError(f"model entry {digest} exists without record.json (partial/corrupt)")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if canonical(record["identity"]) != canonical(identity):
            raise TechnicalError(f"model entry {digest} identity conflicts with the request")
        payloads = {}
        for q in TARGETS:
            path = directory / f"{q}.ubj"
            if not path.is_file():
                raise TechnicalError(f"model entry {digest} lacks {q}.ubj")
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != record["booster_sha256"][q]:
                raise TechnicalError(f"model entry {digest} {q} bytes do not match the recorded digest")
            payloads[q] = payload
        validate_fit_records(identity, payloads, record.get("fit_records") or {})
        return Quartet(payloads, record["fit_records"])

    def get_or_fit(self, identity: dict, fit: Callable[[], Quartet], use: dict) -> tuple[Quartet, dict]:
        """Return the quartet for ``identity``, fitting it once if absent.

        ``use`` describes this scientific request (stage/H/date/region/purpose)
        and is ledgered separately from the physical fit.
        """
        digest = identity_digest(identity)
        directory = self._dir(digest)
        self.counts["requests"] += 1
        try:
            quartet, status = self._get_or_fit(digest, directory, identity, fit)
        except Exception as error:
            self.counts["failed"] = self.counts.get("failed", 0) + 1
            self._log({"identity_sha256": digest, "status": "failed", "error_type": type(error).__name__,
                       "error": str(error), "notes": list(getattr(error, "__notes__", [])), **use})
            raise
        entry = {"identity_sha256": digest, "status": status, "booster_sha256": quartet.booster_shas(), **use}
        self._log(entry)
        return quartet, entry

    def _get_or_fit(self, digest, directory, identity, fit):
        if directory.exists():
            quartet = self._load(digest, identity)
            self.counts["hits"] += 1
            status = "hit"
        else:
            quartet = fit()
            validate_fit_records(identity, quartet.payloads, quartet.records)
            tmp = directory.with_name(directory.name + ".tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            for q in TARGETS:
                (tmp / f"{q}.ubj").write_bytes(quartet.payloads[q])
            record = {
                "identity": identity,
                "identity_sha256": digest,
                "booster_sha256": quartet.booster_shas(),
                "fit_records": quartet.records,
            }
            (tmp / "record.json").write_text(canonical(record), encoding="utf-8")
            os.replace(tmp, directory)
            self.counts["fits"] += 1
            status = "fit"
        return quartet, status
