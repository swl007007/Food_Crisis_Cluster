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
import shutil
from pathlib import Path
from typing import Callable

import numpy as np

from ipcch_geoxgb.errors import TechnicalError
from ipcch_geoxgb.quartet import TARGETS, Quartet, base_score, from_raw, prefix_identity


def array_digest(array) -> str:
    """SHA256 of dtype, shape and C-ordered bytes (NaN bit patterns included).

    Object arrays are refused: their bytes are pointers, not values.
    """
    arr = np.ascontiguousarray(np.asarray(array))
    if arr.dtype.kind == "O":
        raise TechnicalError("array_digest needs a numeric/fixed-width array, not an object array")
    header = f"{arr.dtype.str}|{arr.shape}|".encode()
    return hashlib.sha256(header + arr.tobytes()).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def identity_digest(identity: dict) -> str:
    return hashlib.sha256(canonical(identity).encode()).hexdigest()


REQUIRED_FIELDS = {
    "target", "kind", "rounds_total", "params", "resolved_config", "base_score",
    "structure_sha256", "booster_sha256", "weights", "rows", "constant", "y_sha256",
}
LOCAL_FIELDS = {
    "parent_booster_sha256", "parent_rounds", "rounds_added",
    "parent_structure_sha256", "child_prefix_structure_sha256",
}


def validate_fit_records(identity: dict, payloads: dict, records: dict) -> None:
    """Required per-target fitting evidence, checked against the actual boosters.

    Called before an entry is written and whenever one is loaded; any missing
    or inconsistent field is a TechnicalError (never a refit).
    """
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
        if hashlib.sha256(payloads[q]).hexdigest() != record["booster_sha256"]:
            raise TechnicalError(f"fit record for {q} does not match its booster bytes")
        booster = from_raw(payloads[q])
        if booster.num_boosted_rounds() != record["rounds_total"]:
            raise TechnicalError(f"{q}: booster rounds differ from the fit record")
        if base_score(booster) != record["base_score"]:
            raise TechnicalError(f"{q}: booster base score differs from the fit record")
        if prefix_identity(booster)["sha256"] != record["structure_sha256"]:
            raise TechnicalError(f"{q}: booster structure differs from the fit record")
        learner = record["resolved_config"].get("learner", {})
        if learner.get("objective", {}).get("name") != record["params"].get("objective"):
            raise TechnicalError(f"{q}: resolved config objective differs from the requested params")
        if learner.get("learner_model_param", {}).get("base_score") != record["base_score"]:
            raise TechnicalError(f"{q}: resolved config base score differs from the fit record")
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
        self.counts = {"requests": 0, "hits": 0, "fits": 0}

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
        entry = {"identity_sha256": digest, "status": status, "booster_sha256": quartet.booster_shas(), **use}
        self._log(entry)
        return quartet, entry
