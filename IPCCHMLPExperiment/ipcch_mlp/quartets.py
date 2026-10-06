"""B/P/L quartets on one lawful fitting pool (PRD R7-R12, R16-R19; design sections 4-6, 10).

A *global fit* is one pool: its fitting-only transform, four independent scalar
B networks (one per q2..q5) and their evaluation-mode predictions on the pool.
Residual targets ``r_q = y_q - g_q(X)`` (float64 subtraction, cast to float32)
come from those stored predictions; the pooled residual quartet uses the whole
pool, a regional quartet the exact indexed regional subset. P and L each start
from fresh zero-output residual networks that correct the same frozen B; there
is no inheritance between P and L. Identities bind data, transform, parent B
state, residual targets, recipe, initialization and environment; seeds come
from a separate, environment-free seed identity.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from ipcch_mlp import nets, preprocess, seeds
from ipcch_mlp.errors import TechnicalError
from ipcch_mlp.store import ModelStore

TARGETS = ("q2", "q3", "q4", "q5")


def array_digest(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


@dataclass
class Quartet:
    """Four loaded scalar networks used together (atomic routing)."""

    role: str
    widths: list
    digests: dict  # q -> identity digest
    records: dict  # q -> record
    models: dict = field(default_factory=dict)  # q -> nets.ScalarMLP

    def predict(self, Xt: np.ndarray, device: str, batch: int) -> np.ndarray:
        cols = [nets.predict(self.models[q], Xt, device, batch) for q in TARGETS]
        return np.column_stack(cols) if len(Xt) else np.zeros((0, 4))

    def provider(self) -> str:
        return seeds.digest({"quartet": [self.digests[q] for q in TARGETS]})


@dataclass
class GlobalFit:
    key: tuple
    rows: np.ndarray  # pool rows (positions into the horizon arrays), sorted
    transform: preprocess.Transform
    transform_sha256: str
    Xt: np.ndarray  # float32 (n, 1122) pool inputs
    B: Quartet
    B_pool: np.ndarray  # (n, 4) float64 evaluation-mode predictions on the pool
    residual: np.ndarray  # (n, 4) float32 residual targets
    seed_base: dict
    identity_base: dict
    residuals: dict = field(default_factory=dict)  # (role, rid, node) -> Quartet


class Engine:
    def __init__(self, store: ModelStore, contract: dict, env: dict, device: str, cache_size: int = 12):
        self.store = store
        self.contract = contract
        self.train_cfg = contract["training"]
        self.arch = contract["architecture"]
        self.env = env
        self.device = device
        self.cache: OrderedDict = OrderedDict()
        self.cache_size = cache_size
        self.updates: list = []

    # ------------------------------------------------------------- scalar fit

    def _scalar(self, identity: dict, seed_identity: dict, widths: list, role: str, X: np.ndarray,
                y: np.ndarray, epochs: int, use: dict) -> tuple[nets.ScalarMLP, dict, str]:
        def fit():
            s = seeds.seed_triplet(seed_identity)
            net = nets.build(widths, role, s["init"])
            init_sha = nets.state_digest(nets.cpu_state(net))
            history = nets.train(net, X, y, epochs, s, self.train_cfg, self.device)
            pred = nets.predict(net, X, self.device, self.train_cfg["inference_batch_size"])
            fit_mse = float(np.mean((pred - y.astype(np.float64)) ** 2))
            return nets.cpu_state(net), {"seed_identity": seed_identity, "seeds": s, "widths": widths, "role": role,
                                         "initial_state_sha256": init_sha, "history": history,
                                         "fit_mse_eval_mode": fit_mse}
        state, record, digest = self.store.get_or_fit(identity, fit, use)
        net = nets.load(widths, state)
        return net, record, digest

    # ------------------------------------------------------------- global

    def global_fit(self, hz, stage: str, replicate: int, gid: str, origin_label, rows: np.ndarray,
                   use: dict) -> GlobalFit:
        rows = np.sort(np.asarray(rows, dtype=np.int64))
        if len(rows) == 0:
            raise TechnicalError(f"empty required global pool: H{hz.h} {stage} origin {origin_label}")
        key = (stage, hz.h, replicate, gid, str(origin_label), array_digest(rows))
        if key in self.cache:
            self.cache.move_to_end(key)
            gf = self.cache[key]
            for q in TARGETS:  # still a scientific request: record it in the ledger
                self.store.counts["requests"] += 1
                self.store.counts["hits"] += 1
                self.store._log({"identity_sha256": gf.B.digests[q], "status": "hit", **use, "target": q,
                                 "role": "global", "memory": True})
            return gf
        widths = self.arch["global_candidates"][gid]
        raw = hz.raw(rows)
        transform = preprocess.fit_transform(raw)
        t_sha = self.store.put_transform(transform)
        Xt = preprocess.apply(transform, raw)
        fit_keys = array_digest(hz.keys.iloc[rows][["admin_code", "target_ord"]].to_numpy(dtype=np.int64))
        seed_base = {"stage": stage, "H": hz.h, "replicate": replicate, "global_id": gid, "global_widths": widths,
                     "origin": str(origin_label), "fit_keys": fit_keys}
        identity_base = {"kind": "ipcch-mlp-scalar", "contract": self.contract["contract_version"], **seed_base,
                         "fit_rows": array_digest(rows), "n_fit": int(len(rows)),
                         "X_artifact_sha256": hz.x_sha256, "keys_artifact_sha256": hz.keys_sha256,
                         "transform_sha256": t_sha, "training": self.train_cfg, "dropout": self.arch["dropout"],
                         "env": self.env}
        models, digests, records = {}, {}, {}
        for j, q in enumerate(TARGETS):
            y = hz.Y[rows, j].astype(np.float32)
            ident = {**identity_base, "target": q, "role": "global", "init": "pytorch-default",
                     "y_sha256": array_digest(hz.Y[rows, j])}
            sid = {**seed_base, "target": q, "role": "global"}
            net, rec, dig = self._scalar(ident, sid, widths, "global", Xt, y, self.train_cfg["global_epochs"],
                                         {**use, "target": q, "role": "global"})
            models[q], digests[q], records[q] = net, dig, rec
        B = Quartet("global", widths, digests, records, models)
        B_pool = B.predict(Xt, self.device, self.train_cfg["inference_batch_size"])
        residual = (hz.Y[rows] - B_pool).astype(np.float32)
        if not np.isfinite(residual).all():
            raise TechnicalError("residual targets are not finite")
        gf = GlobalFit(key, rows, transform, t_sha, Xt, B, B_pool, residual, seed_base, identity_base)
        self.cache[key] = gf
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return gf

    # ------------------------------------------------------------- residual

    def residual_fit(self, hz, gf: GlobalFit, rid: str, node: str | None, use: dict) -> Quartet:
        """Pooled (node None) or regional residual quartet attached to ``gf``."""
        role = "pooled_residual" if node is None else "regional_residual"
        rkey = (role, rid, node)
        if rkey in gf.residuals:
            q4 = gf.residuals[rkey]
            for q in TARGETS:
                self.store.counts["requests"] += 1
                self.store.counts["hits"] += 1
                self.store._log({"identity_sha256": q4.digests[q], "status": "hit", **use, "target": q,
                                 "role": role, "memory": True})
            return q4
        widths = self.arch["residual_candidates"][rid]
        if node is None:
            idx = np.arange(len(gf.rows))
            region = {}
        else:
            idx = np.flatnonzero(np.isin(hz.area[gf.rows], hz.regions[node]))
            if len(idx) == 0:
                raise TechnicalError(f"regional residual requested for empty region {node}")
            region = {"region": node, "region_areas": array_digest(hz.regions[node]), "map_sha256": hz.map_sha256}
        sub_rows = gf.rows[idx]
        X = gf.Xt[idx]
        models, digests, records = {}, {}, {}
        for j, q in enumerate(TARGETS):
            r = np.ascontiguousarray(gf.residual[idx, j])
            ident = {**gf.identity_base, "target": q, "role": role, "residual_id": rid, "residual_widths": widths,
                     "init": "pytorch-default-hidden+zero-output", **region,
                     "residual_fit_rows": array_digest(sub_rows), "n_residual_fit": int(len(sub_rows)),
                     "parent_identity": gf.B.digests[q], "parent_state_sha256": gf.B.records[q]["final_state_sha256"],
                     "residual_target_sha256": array_digest(r)}
            sid = {**gf.seed_base, "target": q, "role": role, "residual_id": rid, "residual_widths": widths,
                   **({"region": node} if node else {})}
            net, rec, dig = self._scalar(ident, sid, widths, "residual", X, r, self.train_cfg["residual_epochs"],
                                         {**use, "target": q, "role": role, **({"region": node} if node else {})})
            models[q], digests[q], records[q] = net, dig, rec
        q4 = Quartet(role, widths, digests, records, models)
        gf.residuals[rkey] = q4
        return q4

    def inputs(self, hz, gf: GlobalFit, rows: np.ndarray) -> np.ndarray:
        return preprocess.apply(gf.transform, hz.raw(rows))
