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
from ipcch_geoxgb.quartet import TARGETS, Quartet


def array_digest(array) -> str:
    """SHA256 of dtype, shape and C-ordered bytes (NaN bit patterns included)."""
    arr = np.ascontiguousarray(np.asarray(array))
    header = f"{arr.dtype.str}|{arr.shape}|".encode()
    return hashlib.sha256(header + arr.tobytes()).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def identity_digest(identity: dict) -> str:
    return hashlib.sha256(canonical(identity).encode()).hexdigest()


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
