"""Deterministic supervised AlphaEarth partition search (design section 9, D8/D10-D14/D19/D20).

One axis-aligned binary tree per H over the 128 frozen descriptors; whole admins route by saved thresholds
(left: value <= threshold), cross-country / disconnected groups allowed. Only admins with eligible mapped OOT
rows in the current pool determine thresholds and support. Best-first growth from one mapped root (B2-
equivalent): every supported cut of every splittable leaf (depth < 3) is scored by refitting the *whole*
joint shared/regional model; the largest quartet-objective reduction wins if it exceeds
1e-12 * max(1, |objective|). Gains within that tolerance of the best tie by node path (text), descriptor index,
then threshold order. One growth per lambda records the states reached at leaf budgets 2/4/8; an early
terminal state serves every later budget. Per-admin sufficient statistics are cached; scores are recomputed
after each accepted split because joint fits change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ipcch_alphaearth import correction as cr
from ipcch_alphaearth.errors import TechnicalError


# --------------------------------------------------------------------------- map structure


@dataclass
class PartitionMap:
    """Internal nodes {node_id: (descriptor_index, threshold)} and leaf ids; node ids are r, r0, r1, r00, ..."""

    splits: dict = field(default_factory=dict)
    leaves: list = field(default_factory=lambda: ["r"])
    descriptor_names: list = field(default_factory=list)

    @property
    def n_leaves(self) -> int:
        return len(self.leaves)

    @property
    def depth(self) -> int:
        return max(len(leaf) - 1 for leaf in self.leaves)

    def route(self, D: np.ndarray, routable: np.ndarray) -> np.ndarray:
        """Leaf id per admin row of D; ``None`` where the admin is not routable (missing/low-coverage AlphaEarth)."""
        D = np.asarray(D, dtype=np.float64)
        out = np.full(len(D), None, dtype=object)
        idx = np.flatnonzero(routable)
        if not np.isfinite(D[idx]).all():
            raise TechnicalError("a routable admin has a nonfinite descriptor")
        node = np.full(len(idx), "r", dtype=object)
        for _ in range(16):
            moved = False
            for nid, (j, t) in self.splits.items():
                here = node == nid
                if here.any():
                    left = D[idx[here], j] <= t
                    node[here] = np.where(left, nid + "0", nid + "1")
                    moved = True
            if not moved:
                break
        if not set(node.tolist()) <= set(self.leaves):
            raise TechnicalError("routing reached a node that is not a leaf of the map")
        out[idx] = node
        return out

    def record(self) -> dict:
        return {"splits": {k: {"descriptor_index": int(j), "descriptor": self.descriptor_names[j] if self.descriptor_names else None,
                               "threshold": float(t), "threshold_hex": float(t).hex()} for k, (j, t) in sorted(self.splits.items())},
                "leaves": sorted(self.leaves), "n_leaves": self.n_leaves, "depth": self.depth}

    @classmethod
    def from_record(cls, rec: dict, descriptor_names=None) -> "PartitionMap":
        splits = {k: (int(v["descriptor_index"]), float.fromhex(v["threshold_hex"])) for k, v in rec["splits"].items()}
        return cls(splits, list(rec["leaves"]), list(descriptor_names or []))


# --------------------------------------------------------------------------- pool for one correction context


@dataclass
class Pool:
    """Honest OOT correction rows of one fit (all rows; mapped rows carry an admin descriptor index)."""

    V: np.ndarray  # (n, 4, 8) design from the context transform
    R: np.ndarray  # (n, 4) residuals q - p
    w: np.ndarray  # (n,) raw D7 weights
    admin: np.ndarray  # (n,) admin codes
    target_ord: np.ndarray  # (n,)
    routable: np.ndarray  # (n,) bool: admin has a usable descriptor (coverage >= .80, 128 finite)

    def __post_init__(self):
        self.a = self.w / self.w.sum()
        self.total = cr.Stats.from_rows(self.V, self.R, self.a)


def cut_thresholds(values: np.ndarray, admin_codes: np.ndarray, n_cuts: int) -> list:
    """Rank ceil(j*n/10) observed values (j=1..9; one value per distinct admin, ties by admin code), deduplicated,
    without the maximum (empty right child)."""
    order = np.lexsort((admin_codes, values))
    v = values[order]
    n = len(v)
    vmax = v[-1]
    out = []
    for j in range(1, n_cuts + 1):
        t = v[math.ceil(j * n / (n_cuts + 1)) - 1]
        if t < vmax and (not out or t != out[-1]):
            out.append(t)
    return out


class Search:
    """Best-first growth for one context and one regional lambda."""

    def __init__(self, pool: Pool, D_admin: np.ndarray, admin_codes: np.ndarray, active: np.ndarray, contract: dict,
                 lam_region: float, descriptor_names=None):
        c = contract["correction"]
        self.c = c
        self.lam_shared = float(c["shared_lambda"])
        self.lam = float(lam_region)
        self.tol_solver = float(c["solver_residual_tolerance"])
        self.floor = c["support"]
        self.active = active
        self.pool = pool
        self.descriptor_names = list(descriptor_names or [])
        # mapped admins = routable admins with pool rows
        mapped_rows = pool.routable
        admins_mapped = np.unique(pool.admin[mapped_rows])
        pos = {a: i for i, a in enumerate(admin_codes)}
        missing = [a for a in admins_mapped if a not in pos]
        if missing:
            raise TechnicalError(f"{len(missing)} mapped admins lack descriptors")
        self.admins = admins_mapped  # sorted numeric
        self.D = np.asarray(D_admin, dtype=np.float64)[[pos[a] for a in admins_mapped]]
        if not np.isfinite(self.D).all():
            raise TechnicalError("mapped admin with nonfinite descriptor")
        ai = np.searchsorted(self.admins, pool.admin[mapped_rows])
        n_admin = len(self.admins)
        self.n_mapped_rows = int(mapped_rows.sum())
        self.S = cr.admin_stats(pool.V[mapped_rows], pool.R[mapped_rows], pool.a[mapped_rows], ai, n_admin)
        months, mi = np.unique(pool.target_ord[mapped_rows], return_inverse=True)
        self.month_hist = np.zeros((n_admin, len(months)))
        np.add.at(self.month_hist, (ai, mi), 1.0)
        w = pool.w[mapped_rows]
        self.sw = np.bincount(ai, weights=w, minlength=n_admin)
        self.sw2 = np.bincount(ai, weights=w * w, minlength=n_admin)
        self.ledger: list[dict] = []
        self.states: list[dict] = []

    # ---------------------------------------------------------------- helpers
    def _leaf_stats(self, member: np.ndarray) -> dict:
        flat = self.S[member].sum(axis=0)
        G, b, s, mass = cr.unflatten(flat)
        return {"G": G, "b": b, "mass": float(mass), "sw": float(self.sw[member].sum()), "sw2": float(self.sw2[member].sum()),
                "admins": int(member.sum()), "months": int((self.month_hist[member].sum(axis=0) > 0).sum())}

    def _objective_b2(self) -> dict:
        return cr.solve_b2(self.pool.total, self.active, self.lam_shared, self.tol_solver)

    def run(self, max_leaves: int, max_depth: int) -> list[dict]:
        b2 = self._objective_b2()
        membership = {"r": np.ones(len(self.admins), dtype=bool)}
        current = {"objective": b2["objective"], "fit": {"kind": "b2", **b2}}
        pmap = PartitionMap({}, ["r"], self.descriptor_names)
        self.states.append(self._state(pmap, current, step=0))
        step = 0
        if len(self.admins) == 0:  # no routable admin in the pool: the B2 root is the only state
            return self.states
        while pmap.n_leaves < max_leaves:
            step += 1
            best = self._best_split(pmap, membership, current["objective"], max_depth, step)
            if best is None:
                break
            nid, j, t, fit = best
            pmap = PartitionMap({**pmap.splits, nid: (j, t)},
                                sorted([leaf for leaf in pmap.leaves if leaf != nid] + [nid + "0", nid + "1"]), self.descriptor_names)
            left = membership[nid] & (self.D[:, j] <= t)
            right = membership[nid] & ~(self.D[:, j] <= t)
            membership = {k: v for k, v in membership.items() if k != nid}
            membership[nid + "0"], membership[nid + "1"] = left, right
            current = {"objective": fit["objective"], "fit": fit}
            self.states.append(self._state(pmap, current, step=step))
        return self.states

    def _state(self, pmap: PartitionMap, current: dict, step: int) -> dict:
        return {"step": step, "map": PartitionMap(dict(pmap.splits), list(pmap.leaves), self.descriptor_names),
                "objective": current["objective"], "fit": current["fit"]}

    def _best_split(self, pmap: PartitionMap, membership: dict, f_cur: float, max_depth: int, step: int):
        cands = []  # (node, j, order, t, left_mask)
        for nid in sorted(pmap.leaves):
            if len(nid) - 1 >= max_depth:
                continue
            member = membership[nid]
            if not member.any():
                continue
            codes = self.admins[member]
            for j in range(self.D.shape[1]):
                vals = self.D[member, j]
                if vals.min() == vals.max():
                    continue
                for order, t in enumerate(cut_thresholds(vals, codes, int(self.c["cuts_per_descriptor"]))):
                    cands.append((nid, j, order, t))
        if not cands:
            return None
        n_admin = len(self.admins)
        left_masks = np.zeros((len(cands), n_admin), dtype=bool)
        for i, (nid, j, _order, t) in enumerate(cands):
            left_masks[i] = membership[nid] & (self.D[:, j] <= t)
        parent_masks = np.array([membership[c[0]] for c in cands])
        right_masks = parent_masks & ~left_masks
        Lf, Rf = left_masks.astype(float), right_masks.astype(float)
        stats_L, stats_R = Lf @ self.S, Rf @ self.S
        mh_L, mh_R = Lf @ self.month_hist, Rf @ self.month_hist
        sw_L, sw_R = Lf @ self.sw, Rf @ self.sw
        sw2_L, sw2_R = Lf @ self.sw2, Rf @ self.sw2
        adm_L, adm_R = left_masks.sum(axis=1), right_masks.sum(axis=1)
        mon_L, mon_R = (mh_L > 0).sum(axis=1), (mh_R > 0).sum(axis=1)
        ok = cr.support_ok(sw_L, sw2_L, adm_L, mon_L, self.floor) & cr.support_ok(sw_R, sw2_R, adm_R, mon_R, self.floor) \
            & (adm_L > 0) & (adm_R > 0)
        objective = np.full(len(cands), np.nan)
        fits = {}
        others_cache = {}
        for nid in sorted({c[0] for c in cands}):
            rows = [i for i, c in enumerate(cands) if c[0] == nid and ok[i]]
            if not rows:
                continue
            others = [leaf for leaf in sorted(pmap.leaves) if leaf != nid]
            if nid not in others_cache:
                others_cache[nid] = [self._leaf_stats(membership[leaf]) for leaf in others]
            ost = others_cache[nid]
            C, K = len(rows), len(others) + 2
            leaf_G = np.zeros((C, K, cr.N_Q, cr.N_SLOTS, cr.N_SLOTS))
            leaf_b = np.zeros((C, K, cr.N_Q, cr.N_SLOTS))
            leaf_m = np.zeros((C, K))
            for k, st in enumerate(ost):
                leaf_G[:, k], leaf_b[:, k], leaf_m[:, k] = st["G"], st["b"], st["mass"]
            GL, bL, _sL, mL = cr.unflatten(stats_L[rows])
            GR, bR, _sR, mR = cr.unflatten(stats_R[rows])
            leaf_G[:, K - 2], leaf_b[:, K - 2], leaf_m[:, K - 2] = GL, bL, mL
            leaf_G[:, K - 1], leaf_b[:, K - 1], leaf_m[:, K - 1] = GR, bR, mR
            sol = cr.solve_joint_batch(self.pool.total, leaf_G, leaf_b, leaf_m, self.active, self.lam_shared, self.lam,
                                       self.tol_solver)
            objective[rows] = sol["objective"]
            for local, i in enumerate(rows):
                fits[i] = (others, sol, local)
        gain = f_cur - objective
        for i, (nid, j, order, t) in enumerate(cands):
            self.ledger.append({"step": step, "node": nid, "descriptor_index": j, "threshold_order": order, "threshold": t,
                                "left_admins": int(adm_L[i]), "right_admins": int(adm_R[i]),
                                "left_n_eff": float(cr.n_eff_of(sw_L[i], sw2_L[i])), "right_n_eff": float(cr.n_eff_of(sw_R[i], sw2_R[i])),
                                "left_months": int(mon_L[i]), "right_months": int(mon_R[i]), "supported": bool(ok[i]),
                                "objective": float(objective[i]), "gain": float(gain[i]), "current_objective": f_cur})
        if not ok.any():
            return None
        tol = float(self.c["gain_tolerance_relative"]) * max(1.0, abs(f_cur))
        best_gain = np.nanmax(gain)
        if not best_gain > tol:
            return None
        # ties within tolerance of the best, each itself strictly above the gain floor
        tied = [i for i in range(len(cands)) if ok[i] and gain[i] >= best_gain - tol and gain[i] > tol]
        pick = min(tied, key=lambda i: (cands[i][0], cands[i][1], cands[i][2]))
        nid, j, _order, t = cands[pick]
        others, sol, local = fits[pick]
        leaf_ids = others + [nid + "0", nid + "1"]
        order_leaves = np.argsort(leaf_ids, kind="stable")
        fit = {"kind": "joint", "beta": sol["beta"][local], "u": sol["u"][local][order_leaves],
               "leaf_ids": [leaf_ids[k] for k in order_leaves], "objective": float(sol["objective"][local]),
               "objective_q": sol["objective_q"][local]}
        self.ledger[-len(cands) + pick]["accepted"] = True
        return nid, j, t, fit

    def ledger_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.ledger)
        if "accepted" not in frame:
            frame["accepted"] = False
        frame["accepted"] = frame["accepted"].fillna(False).astype(bool)
        return frame


def state_at_budget(states: list[dict], budget: int) -> dict:
    """The last growth state with at most ``budget`` leaves (an early terminal state serves later budgets)."""
    eligible = [s for s in states if s["map"].n_leaves <= budget]
    return eligible[-1]


def fit_frozen_map(pool: Pool, transform_active: np.ndarray, pmap: PartitionMap, leaf_of_row: np.ndarray, contract: dict,
                   lam_region: float) -> dict:
    """Annual refit with a frozen map: per-leaf support, joint fit over supported leaves, exact B2 reuse below two."""
    c = contract["correction"]
    lam_shared, tol = float(c["shared_lambda"]), float(c["solver_residual_tolerance"])
    b2 = cr.solve_b2(pool.total, transform_active, lam_shared, tol)
    leaf_of_row = np.asarray(leaf_of_row, dtype=object)
    support, supported = {}, []
    for leaf in sorted(pmap.leaves):
        m = leaf_of_row == leaf
        w = pool.w[m]
        rec = {"rows": int(m.sum()), "admins": int(np.unique(pool.admin[m]).size), "target_months": int(np.unique(pool.target_ord[m]).size),
               "sum_w": float(w.sum()), "n_eff": float(cr.n_eff_of(w.sum(), (w * w).sum()))}
        rec["supported"] = bool(cr.support_ok(np.array([w.sum()]), np.array([(w * w).sum()]), np.array([rec["admins"]]),
                                              np.array([rec["target_months"]]), c["support"])[0])
        support[leaf] = rec
        if rec["supported"]:
            supported.append(leaf)
    if len(supported) < 2:
        return {"b2": b2, "g_kind": "b2_reuse", "support": support, "supported": supported}
    K = len(supported)
    leaf_G = np.zeros((1, K, cr.N_Q, cr.N_SLOTS, cr.N_SLOTS))
    leaf_b = np.zeros((1, K, cr.N_Q, cr.N_SLOTS))
    leaf_m = np.zeros((1, K))
    for k, leaf in enumerate(supported):
        m = leaf_of_row == leaf
        st = cr.Stats.from_rows(pool.V[m], pool.R[m], pool.a[m])
        leaf_G[0, k], leaf_b[0, k], leaf_m[0, k] = st.G, st.b, st.mass
    sol = cr.solve_joint_batch(pool.total, leaf_G, leaf_b, leaf_m, transform_active, lam_shared, lam_region, tol)
    return {"b2": b2, "g_kind": "joint", "support": support, "supported": supported, "beta_g": sol["beta"][0], "u": sol["u"][0],
            "g_objective": float(sol["objective"][0]), "max_stationarity": sol["max_stationarity"], "max_centering": sol["max_centering"]}
