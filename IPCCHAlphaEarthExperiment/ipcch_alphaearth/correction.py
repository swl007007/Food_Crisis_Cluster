"""Shared (B2) and shared + regional (G) low-dimensional ridge corrections (design section 8, D17-D20, D25).

Per target q with raw unprojected B1 prediction p and residual r = q - p on honest OOT rows:

    v = [1, z(p), z(prcp), z(tmean), z(violence), m(prcp), m(tmean), m(violence)]   (eight slots)

Weighted mean / population SD of the four core inputs are fitted on the correction pool only (D7 weights),
common to B2, G and every candidate map in that fit. Missing dynamic inputs take the training mean (zero after
scaling) plus an unscaled 0/1 flag. A wholly missing or constant column, and a constant flag, is *inactive*:
its standardized value is zero and its coefficient is fixed at zero (also at inference). Nonfinite p is a
technical error.

Objective per q (a_i = w_i / W over the whole pool, mapped or not):

    sum_i a_i (r_i - v_i'(beta + u_R(i)))^2 + .01 ||beta_nonintercept||^2 + lambda_region sum_R ||u_R||^2
    subject to sum_R m_R u_R = 0 coefficient-wise (m_R = mapped fitting weight of supported leaf R)

Rows without a supported leaf (missing AlphaEarth, unsupported leaf) have u = 0 and stay in the shared loss.
The quartet objective is the equal mean over q. Fewer than two supported leaves is exact B2 reuse. The KKT
system is solved per q and checked for stationarity and centering; a failed solve is a TechnicalError.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ipcch_alphaearth.errors import TechnicalError

N_Q = 4
N_SLOTS = 8
SLOT_NAMES = ("intercept", "z_pred", "z_prcp", "z_tmean", "z_violence", "m_prcp", "m_tmean", "m_violence")


# --------------------------------------------------------------------------- transforms


def _weighted_moments(x: np.ndarray, w: np.ndarray) -> tuple[float, float, bool]:
    """Weighted mean / population SD over finite entries; inactive when none, all identical or SD == 0."""
    ok = np.isfinite(x)
    if not ok.any():
        return 0.0, 0.0, False
    xv, wv = x[ok], w[ok]
    if np.all(xv == xv[0]):
        return float(xv[0]), 0.0, False
    sw = wv.sum()
    if sw <= 0:
        return 0.0, 0.0, False
    mean = float(np.sum(wv * xv) / sw)
    sd = float(np.sqrt(np.sum(wv * (xv - mean) ** 2) / sw))
    return mean, sd, bool(sd > 0 and np.isfinite(sd))


@dataclass
class Transform:
    """Fit-only scaling for one correction fit. ``p_*`` are per q; ``z_*`` are the three dynamic inputs."""

    p_mean: np.ndarray
    p_sd: np.ndarray
    p_active: np.ndarray
    z_mean: np.ndarray
    z_sd: np.ndarray
    z_active: np.ndarray
    flag_active: np.ndarray

    @classmethod
    def fit(cls, P: np.ndarray, Z: np.ndarray, w: np.ndarray) -> "Transform":
        P, Z, w = _check_inputs(P, Z, w)
        pm, ps, pa = zip(*(_weighted_moments(P[:, k], w) for k in range(N_Q)))
        zm, zs, za = zip(*(_weighted_moments(Z[:, j], w) for j in range(3)))
        miss = ~np.isfinite(Z)
        flag_active = np.array([bool(miss[:, j].any() and (~miss[:, j]).any()) for j in range(3)])
        return cls(np.array(pm), np.array(ps), np.array(pa, bool), np.array(zm), np.array(zs), np.array(za, bool), flag_active)

    def active(self) -> np.ndarray:
        """(4, 8) boolean slot mask per q (intercept always active)."""
        out = np.zeros((N_Q, N_SLOTS), dtype=bool)
        out[:, 0] = True
        out[:, 1] = self.p_active
        out[:, 2:5] = self.z_active[None, :]
        out[:, 5:8] = self.flag_active[None, :]
        return out

    def design(self, P: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """(n, 4, 8) design for prediction or fitting rows; inactive slots are exactly zero."""
        P, Z, _ = _check_inputs(P, Z, None)
        n = P.shape[0]
        V = np.zeros((n, N_Q, N_SLOTS))
        V[:, :, 0] = 1.0
        for k in range(N_Q):
            if self.p_active[k]:
                V[:, k, 1] = (P[:, k] - self.p_mean[k]) / self.p_sd[k]
        for j in range(3):
            ok = np.isfinite(Z[:, j])
            if self.z_active[j]:
                col = np.zeros(n)
                col[ok] = (Z[ok, j] - self.z_mean[j]) / self.z_sd[j]
                V[:, :, 2 + j] = col[:, None]
            if self.flag_active[j]:
                V[:, :, 5 + j] = (~ok).astype(float)[:, None]
        return V

    def record(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("p_mean", "p_sd", "p_active", "z_mean", "z_sd", "z_active", "flag_active")}

    @classmethod
    def from_record(cls, rec: dict) -> "Transform":
        return cls(*(np.array(rec[k], dtype=bool if k.endswith("active") else float)
                     for k in ("p_mean", "p_sd", "p_active", "z_mean", "z_sd", "z_active", "flag_active")))


def _check_inputs(P, Z, w):
    P = np.asarray(P, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != N_Q or Z.shape != (P.shape[0], 3):
        raise TechnicalError(f"correction inputs have shapes {P.shape} / {Z.shape}")
    if not np.isfinite(P).all():
        raise TechnicalError("nonfinite raw B1 prediction in a correction input (technical error, not imputable)")
    if np.isinf(Z).any():
        raise TechnicalError("infinite core correction input")
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
        if w.shape != (P.shape[0],) or not np.isfinite(w).all() or (w <= 0).any():
            raise TechnicalError("correction weights must be finite and positive")
    return P, Z, w


# --------------------------------------------------------------------------- sufficient statistics


@dataclass
class Stats:
    """Normalized-weight sufficient statistics per q for one row group: G (4,8,8), b (4,8), s (4,)."""

    G: np.ndarray
    b: np.ndarray
    s: np.ndarray
    mass: float  # sum of a_i (normalized weight)

    @classmethod
    def zeros(cls) -> "Stats":
        return cls(np.zeros((N_Q, N_SLOTS, N_SLOTS)), np.zeros((N_Q, N_SLOTS)), np.zeros(N_Q), 0.0)

    @classmethod
    def from_rows(cls, V: np.ndarray, R: np.ndarray, a: np.ndarray) -> "Stats":
        G = np.einsum("i,iqj,iqk->qjk", a, V, V)
        b = np.einsum("i,iqj,iq->qj", a, V, R)
        s = np.einsum("i,iq,iq->q", a, R, R)
        return cls(G, b, s, float(a.sum()))

    def __add__(self, other: "Stats") -> "Stats":
        return Stats(self.G + other.G, self.b + other.b, self.s + other.s, self.mass + other.mass)


def admin_stats(V: np.ndarray, R: np.ndarray, a: np.ndarray, admin_index: np.ndarray, n_admin: int) -> np.ndarray:
    """Flattened per-admin statistics (n_admin, 4*64 + 4*8 + 4 + 1): G, b, s, mass."""
    G = np.einsum("i,iqj,iqk->iqjk", a, V, V).reshape(len(a), N_Q * N_SLOTS * N_SLOTS)
    b = np.einsum("i,iqj,iq->iqj", a, V, R).reshape(len(a), N_Q * N_SLOTS)
    s = np.einsum("i,iq,iq->iq", a, R, R)
    flat = np.concatenate([G, b, s, a[:, None]], axis=1)
    out = np.zeros((n_admin, flat.shape[1]))
    np.add.at(out, admin_index, flat)
    return out


def unflatten(flat: np.ndarray) -> tuple:
    """(..., F) flattened statistics -> G (..., 4, 8, 8), b (..., 4, 8), s (..., 4), mass (...)."""
    g_end = N_Q * N_SLOTS * N_SLOTS
    b_end = g_end + N_Q * N_SLOTS
    G = flat[..., :g_end].reshape(*flat.shape[:-1], N_Q, N_SLOTS, N_SLOTS)
    b = flat[..., g_end:b_end].reshape(*flat.shape[:-1], N_Q, N_SLOTS)
    s = flat[..., b_end:b_end + N_Q]
    mass = flat[..., b_end + N_Q]
    return G, b, s, mass


STAT_WIDTH = N_Q * N_SLOTS * N_SLOTS + N_Q * N_SLOTS + N_Q + 1


# --------------------------------------------------------------------------- support (D20)


def support_ok(sum_w: np.ndarray, sum_w2: np.ndarray, admins: np.ndarray, months: np.ndarray, floor: dict) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        neff = np.where(sum_w2 > 0, sum_w ** 2 / np.where(sum_w2 > 0, sum_w2, 1.0), 0.0)
    return (neff >= floor["n_eff"]) & (admins >= floor["admins"]) & (months >= floor["target_months"])


def n_eff_of(sum_w, sum_w2):
    sum_w2 = np.asarray(sum_w2, dtype=float)
    return np.where(sum_w2 > 0, np.asarray(sum_w, dtype=float) ** 2 / np.where(sum_w2 > 0, sum_w2, 1.0), 0.0)


# --------------------------------------------------------------------------- solvers


def _penalty_diag(active_q: np.ndarray, lam_shared: float) -> np.ndarray:
    d = np.full(int(active_q.sum()), lam_shared)
    d[0] = 0.0  # unpenalized shared intercept (slot 0 is always active and first)
    return d


def solve_b2(total: Stats, active: np.ndarray, lam_shared: float, tol: float) -> dict:
    """Shared ridge per q; returns beta (4,8) with inactive slots zero, per-q objectives and quartet objective."""
    beta = np.zeros((N_Q, N_SLOTS))
    obj = np.zeros(N_Q)
    for q in range(N_Q):
        idx = np.flatnonzero(active[q])
        G = total.G[q][np.ix_(idx, idx)]
        b = total.b[q][idx]
        A = G + np.diag(_penalty_diag(active[q], lam_shared))
        x = _solve_checked(A, b, tol, f"B2 q{q + 2}")
        beta[q, idx] = x
        obj[q] = total.s[q] - 2 * x @ b + x @ G @ x + lam_shared * float(np.sum(x[1:] ** 2))
    return {"beta": beta, "objective_q": obj, "objective": float(obj.mean())}


def _solve_checked(A: np.ndarray, b: np.ndarray, tol: float, label: str) -> np.ndarray:
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as error:
        raise TechnicalError(f"{label}: singular correction system") from error
    resid = np.max(np.abs(A @ x - b), initial=0.0)
    if not np.isfinite(x).all() or resid > tol * max(1.0, float(np.max(np.abs(b), initial=0.0))):
        raise TechnicalError(f"{label}: correction solve failed stationarity check (residual {resid:.3e})")
    return x


def solve_joint_batch(total: Stats, leaf_G: np.ndarray, leaf_b: np.ndarray, leaf_m: np.ndarray, active: np.ndarray,
                      lam_shared: float, lam_region: float, tol: float) -> dict:
    """Joint shared + regional fit for C candidate maps with K >= 2 supported leaves each.

    leaf_G (C, K, 4, 8, 8), leaf_b (C, K, 4, 8), leaf_m (C, K) normalized mapped weight. Returns beta (C,4,8),
    u (C,K,4,8), objective_q (C,4), objective (C,), and the maximal stationarity/centering residuals.
    """
    C, K = leaf_m.shape
    if K < 2:
        raise ValueError("joint solve needs at least two supported leaves (fewer means exact B2 reuse)")
    beta = np.zeros((C, N_Q, N_SLOTS))
    u = np.zeros((C, K, N_Q, N_SLOTS))
    obj = np.zeros((C, N_Q))
    worst_stat, worst_center = 0.0, 0.0
    for q in range(N_Q):
        idx = np.flatnonzero(active[q])
        d = len(idx)
        Gt = total.G[q][np.ix_(idx, idx)]
        bt = total.b[q][idx]
        GR = leaf_G[:, :, q][:, :, idx][:, :, :, idx]  # (C, K, d, d)
        bR = leaf_b[:, :, q][:, :, idx]  # (C, K, d)
        n = d * (K + 2)
        M = np.zeros((C, n, n))
        rhs = np.zeros((C, n))
        Ps = np.diag(_penalty_diag(active[q], lam_shared))
        eye = np.eye(d)
        M[:, :d, :d] = Gt + Ps
        rhs[:, :d] = bt
        for k in range(K):
            s0 = d * (k + 1)
            M[:, :d, s0:s0 + d] = GR[:, k]
            M[:, s0:s0 + d, :d] = GR[:, k]
            M[:, s0:s0 + d, s0:s0 + d] = GR[:, k] + lam_region * eye
            c0 = d * (K + 1)
            M[:, s0:s0 + d, c0:c0 + d] = leaf_m[:, k, None, None] * eye
            M[:, c0:c0 + d, s0:s0 + d] = leaf_m[:, k, None, None] * eye
            rhs[:, s0:s0 + d] = bR[:, k]
        try:
            x = np.linalg.solve(M, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError as error:
            raise TechnicalError(f"joint q{q + 2}: singular KKT system") from error
        if not np.isfinite(x).all():
            raise TechnicalError(f"joint q{q + 2}: nonfinite KKT solution")
        stat = np.max(np.abs(np.einsum("cij,cj->ci", M, x) - rhs), axis=1)
        scale = np.maximum(1.0, np.max(np.abs(rhs), axis=1))
        uq = x[:, d:d * (K + 1)].reshape(C, K, d)
        center = np.max(np.abs(np.einsum("ck,ckj->cj", leaf_m, uq)), axis=1)
        worst_stat = max(worst_stat, float(np.max(stat / scale)))
        worst_center = max(worst_center, float(np.max(center)))
        if (stat > tol * scale).any() or (center > tol * np.maximum(1.0, leaf_m.sum(axis=1))).any():
            raise TechnicalError(f"joint q{q + 2}: KKT stationarity/centering check failed "
                                 f"(stationarity {float(np.max(stat / scale)):.3e}, centering {float(np.max(center)):.3e})")
        bq = x[:, :d]
        beta[:, q, idx] = bq
        u[:, :, q][:, :, idx] = uq
        # objective from sufficient statistics: rows outside supported leaves carry u = 0
        data = total.s[q] - 2 * (bq @ bt) + np.einsum("ci,ij,cj->c", bq, Gt, bq)
        data += -2 * np.einsum("ckj,ckj->c", uq, bR) + 2 * np.einsum("cki,ckij,cj->c", uq, GR, bq) \
            + np.einsum("cki,ckij,ckj->c", uq, GR, uq)
        pen = lam_shared * np.sum(bq[:, 1:] ** 2, axis=1) + lam_region * np.sum(uq ** 2, axis=(1, 2))
        obj[:, q] = data + pen
    return {"beta": beta, "u": u, "objective_q": obj, "objective": obj.mean(axis=1),
            "max_stationarity": worst_stat, "max_centering": worst_center}


# --------------------------------------------------------------------------- predictions


def correct(P: np.ndarray, V: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """p + v' coef per row and q; coef (4,8) shared or (n,4,8) row-specific."""
    if coef.ndim == 2:
        delta = np.einsum("iqj,qj->iq", V, coef)
    else:
        delta = np.einsum("iqj,iqj->iq", V, coef)
    out = np.asarray(P, dtype=np.float64) + delta
    if not np.isfinite(out).all():
        raise TechnicalError("nonfinite corrected prediction")
    return out


@dataclass
class CorrectionFit:
    """A fitted B2 + G pair for one correction context (map frozen or candidate)."""

    transform: Transform
    beta_b2: np.ndarray
    b2_objective: float
    g_kind: str  # "joint" | "b2_reuse"
    beta_g: np.ndarray | None = None
    leaf_ids: list = field(default_factory=list)  # supported leaves with deviations
    u: np.ndarray | None = None  # (K, 4, 8)
    g_objective: float | None = None
    unsupported_leaves: list = field(default_factory=list)

    @classmethod
    def identity(cls) -> "CorrectionFit":
        """Declared identity correction for a context with zero eligible OOT rows: B2 = B1 and G = B2 exactly."""
        return cls(None, np.zeros((N_Q, N_SLOTS)), None, "identity_empty_pool")

    def predict(self, P: np.ndarray, Z: np.ndarray, leaf_of_row: np.ndarray) -> dict:
        """B2/G raw quartets and per-row routes (``leaf:<id>``, ``unsupported_leaf:<id>``, ``missing_ae``, ``b2_reuse``,
        ``identity_empty_pool``)."""
        if self.g_kind == "identity_empty_pool":
            P = np.asarray(P, dtype=np.float64)
            if not np.isfinite(P).all():
                raise TechnicalError("nonfinite raw B1 prediction")
            return {"b2": P.copy(), "g": P.copy(), "route": np.full(len(P), "identity_empty_pool", dtype=object)}
        V = self.transform.design(P, Z)
        b2 = correct(P, V, self.beta_b2)
        leaf_of_row = np.asarray(leaf_of_row, dtype=object)
        if self.g_kind == "b2_reuse":
            return {"b2": b2, "g": b2.copy(), "route": np.full(len(P), "b2_reuse", dtype=object)}
        coef = np.repeat(self.beta_g[None], len(P), axis=0)
        route = np.empty(len(P), dtype=object)
        lookup = {leaf: k for k, leaf in enumerate(self.leaf_ids)}
        for i, leaf in enumerate(leaf_of_row):
            if leaf is None or leaf == "":
                route[i] = "missing_ae"
            elif leaf in lookup:
                coef[i] = coef[i] + self.u[lookup[leaf]]
                route[i] = f"leaf:{leaf}"
            else:
                route[i] = f"unsupported_leaf:{leaf}"
        return {"b2": b2, "g": correct(P, V, coef), "route": route}

    def record(self) -> dict:
        return {"transform": None if self.transform is None else self.transform.record(), "beta_b2": self.beta_b2.tolist(), "b2_objective": self.b2_objective,
                "g_kind": self.g_kind, "beta_g": None if self.beta_g is None else self.beta_g.tolist(),
                "leaf_ids": list(self.leaf_ids), "u": None if self.u is None else self.u.tolist(),
                "g_objective": self.g_objective, "unsupported_leaves": list(self.unsupported_leaves),
                "slots": list(SLOT_NAMES)}

    @classmethod
    def from_record(cls, rec: dict) -> "CorrectionFit":
        return cls(None if rec["transform"] is None else Transform.from_record(rec["transform"]), np.array(rec["beta_b2"]), rec["b2_objective"], rec["g_kind"],
                   None if rec["beta_g"] is None else np.array(rec["beta_g"]), list(rec["leaf_ids"]),
                   None if rec["u"] is None else np.array(rec["u"]), rec["g_objective"], list(rec["unsupported_leaves"]))
