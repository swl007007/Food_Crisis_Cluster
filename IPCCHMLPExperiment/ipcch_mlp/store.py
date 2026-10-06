"""Exact-identity model store (design section 10).

Each scalar network entry lives in ``models/<d[:2]>/<d>/`` with ``state.pt``
(state_dict, ``torch.save``) and ``record.json`` (canonical identity, fit record,
initial/final canonical tensor digests). Entries are written to a temporary
directory and moved into place atomically. A hit requires an equal stored
identity and a state whose canonical digest equals the record; a corrupt or
conflicting entry stops the run and is never silently refitted. Transforms are
stored as ``transforms/<digest>.npz``. Request purposes go to an append-only
ledger, separate from the fitting identity.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from ipcch_mlp import nets, seeds
from ipcch_mlp.errors import TechnicalError
from ipcch_mlp.preprocess import Transform
from ipcch_mlp.runtime import torch


class ModelStore:
    def __init__(self, root: Path, ledger: Path, readonly: bool = False):
        self.readonly = readonly
        self.root = Path(root)
        (self.root / "models").mkdir(parents=True, exist_ok=True)
        (self.root / "transforms").mkdir(parents=True, exist_ok=True)
        self.ledger = Path(ledger)
        self.counts = {"requests": 0, "hits": 0, "fits": 0, "failed": 0}

    def _dir(self, digest: str) -> Path:
        return self.root / "models" / digest[:2] / digest

    def _log(self, entry: dict) -> None:
        with open(self.ledger, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")

    def load(self, digest: str, identity: dict | None = None) -> tuple[dict, dict]:
        d = self._dir(digest)
        try:
            record = json.loads((d / "record.json").read_text(encoding="utf-8"))
            state = torch.load(d / "state.pt", map_location="cpu", weights_only=True)
        except Exception as error:  # corrupt entry: stop
            raise TechnicalError(f"model entry {digest} is unreadable: {error}") from error
        if seeds.digest(record["identity"]) != digest:
            raise TechnicalError(f"model entry {digest}: stored identity does not hash to its key")
        if identity is not None and seeds.canonical(record["identity"]) != seeds.canonical(identity):
            raise TechnicalError(f"model entry {digest}: stored identity conflicts with the request")
        if nets.state_digest(state) != record["final_state_sha256"]:
            raise TechnicalError(f"model entry {digest}: tensor digest differs from its record")
        return state, record

    def get_or_fit(self, identity: dict, fit: Callable[[], tuple[dict, dict]], use: dict) -> tuple[dict, dict, str]:
        digest = seeds.digest(identity)
        self.counts["requests"] += 1
        if self._dir(digest).exists():
            state, record = self.load(digest, identity)
            self.counts["hits"] += 1
            self._log({"identity_sha256": digest, "status": "hit", **use})
            return state, record, digest
        if self.readonly:
            self.counts["failed"] += 1
            self._log({"identity_sha256": digest, "status": "missing_in_readonly_store", **use})
            raise TechnicalError(f"replay requested a model that was never stored: {digest} ({use})")
        try:
            state, fit_record = fit()
        except Exception as error:
            self.counts["failed"] += 1
            self._log({"identity_sha256": digest, "status": "failed", "error_type": type(error).__name__,
                       "error": str(error), **use})
            raise
        record = {"identity": identity, "identity_sha256": digest, **fit_record,
                  "final_state_sha256": nets.state_digest(state)}
        tmp = self.root / "models" / f".tmp-{uuid.uuid4().hex}"
        tmp.mkdir()
        torch.save(state, tmp / "state.pt")
        (tmp / "record.json").write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n",
                                         encoding="utf-8")
        target = self._dir(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():  # a concurrent writer is not part of the design: stop
            shutil.rmtree(tmp)
            raise TechnicalError(f"model entry {digest} appeared during the fit")
        os.replace(tmp, target)
        self.counts["fits"] += 1
        self._log({"identity_sha256": digest, "status": "fit", **use})
        return state, record, digest

    # ------------------------------------------------------------ transforms

    def put_transform(self, t: Transform) -> str:
        digest = t.digest()
        path = self.root / "transforms" / f"{digest}.npz"
        if path.exists():
            if self.get_transform(digest).digest() != digest:
                raise TechnicalError(f"stored transform {digest} is corrupt")
            return digest
        tmp = path.with_suffix(f".tmp-{uuid.uuid4().hex}.npz")
        np.savez(tmp, **t.to_arrays())
        os.replace(tmp, path)
        return digest

    def get_transform(self, digest: str) -> Transform:
        path = self.root / "transforms" / f"{digest}.npz"
        with np.load(path) as data:
            t = Transform.from_arrays({k: data[k] for k in data.files})
        if t.digest() != digest:
            raise TechnicalError(f"stored transform {digest} is corrupt")
        return t
