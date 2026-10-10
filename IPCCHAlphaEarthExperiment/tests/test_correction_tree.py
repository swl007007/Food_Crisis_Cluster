"""Correction layer against a direct constrained reference solver; supervised search against brute force."""

import math
import unittest

import numpy as np

from helpers import intercept_active, small_contract
from ipcch_alphaearth import correction as cr, partition as pt
from ipcch_alphaearth.contract import load_contract
from ipcch_alphaearth.errors import TechnicalError


def reference_joint(V, R, w, leaf, leaves, active, lam_s, lam_r):
    """Row-level constrained ridge per q via a null-space parametrization of sum_R m_R u_R = 0 (independent of KKT)."""
    a = w / w.sum()
    K = len(leaves)
    beta = np.zeros((4, 8))
    u = np.zeros((K, 4, 8))
    obj = np.zeros(4)
    for q in range(4):
        idx = np.flatnonzero(active[q])
        d = len(idx)
        Vq = V[:, q][:, idx]
        m = np.array([a[leaf == L].sum() for L in leaves])
        # unknowns theta = [beta (d), u_1..u_K (d each)]; constraint C theta = 0 with C = [0, m_1 I, ..., m_K I]
        C = np.hstack([np.zeros((d, d))] + [m[k] * np.eye(d) for k in range(K)])
        _, _, vt = np.linalg.svd(C)
        N = vt[d:].T  # null space basis (rows of vt beyond rank d)
        X = np.zeros((len(a), d * (K + 1)))
        X[:, :d] = Vq
        for k, L in enumerate(leaves):
            sel = leaf == L
            X[sel, d * (k + 1):d * (k + 2)] = Vq[sel]
        pen = np.concatenate([[0.0], np.full(d - 1, lam_s), np.full(d * K, lam_r)])
        A = np.vstack([np.sqrt(a)[:, None] * X, np.sqrt(pen)[:, None] * np.eye(d * (K + 1))])
        y = np.concatenate([np.sqrt(a) * R[:, q], np.zeros(d * (K + 1))])
        z, *_ = np.linalg.lstsq(A @ N, y, rcond=None)
        theta = N @ z
        beta[q, idx] = theta[:d]
        for k in range(K):
            u[k, q, idx] = theta[d * (k + 1):d * (k + 2)]
        resid = R[:, q] - X @ theta
        obj[q] = np.sum(a * resid ** 2) + np.sum(pen * theta ** 2)
    return beta, u, obj


def random_case(seed, n=400):
    rng = np.random.default_rng(seed)
    P = rng.uniform(size=(n, 4))
    Z = rng.normal(size=(n, 3))
    Z[rng.uniform(size=Z.shape) < 0.1] = np.nan
    Z[:, 1] = 2.5  # constant core input -> inactive slot
    w = rng.uniform(0.1, 1.0, n)
    t = cr.Transform.fit(P, Z, w)
    V = t.design(P, Z)
    R = rng.normal(0, 0.1, (n, 4))
    leaf = np.array(["A", "B", "C", "U", "M"], dtype=object)[rng.integers(0, 5, n)]  # U unsupported, M missing AE
    return t, V, R, w, leaf


class CorrectionReference(unittest.TestCase):
    def test_joint_solution_matches_constrained_reference(self):
        t, V, R, w, leaf = random_case(1)
        active = t.active()
        self.assertFalse(active[:, 3].any())  # constant column is inactive
        leaves = ["A", "B", "C"]
        a = w / w.sum()
        total = cr.Stats.from_rows(V, R, a)
        G = np.zeros((1, 3, 4, 8, 8)); b = np.zeros((1, 3, 4, 8)); m = np.zeros((1, 3))
        for k, L in enumerate(leaves):
            st = cr.Stats.from_rows(V[leaf == L], R[leaf == L], a[leaf == L])
            G[0, k], b[0, k], m[0, k] = st.G, st.b, st.mass
        sol = cr.solve_joint_batch(total, G, b, m, active, 0.01, 0.1, 1e-9)
        rb, ru, robj = reference_joint(V, R, w, leaf, leaves, active, 0.01, 0.1)
        np.testing.assert_allclose(sol["beta"][0], rb, atol=1e-10)
        np.testing.assert_allclose(sol["u"][0], ru, atol=1e-10)
        np.testing.assert_allclose(sol["objective_q"][0], robj, atol=1e-12)
        self.assertLess(np.max(np.abs(np.einsum("k,kqj->qj", m[0], sol["u"][0]))), 1e-12)  # centering
        self.assertTrue(np.all(sol["beta"][0][~active] == 0) and np.all(sol["u"][0][:, ~active] == 0))

    def test_normalized_loss_and_unpenalized_intercept(self):
        t, V, R, w, leaf = random_case(2)
        active = t.active()
        a = w / w.sum()
        b1 = cr.solve_b2(cr.Stats.from_rows(V, R, a), active, 0.01, 1e-9)
        b2 = cr.solve_b2(cr.Stats.from_rows(V, R, (7 * w) / (7 * w).sum()), active, 0.01, 1e-9)
        np.testing.assert_allclose(b1["beta"], b2["beta"], atol=1e-14)  # weight scale is irrelevant
        only = intercept_active()
        b0 = cr.solve_b2(cr.Stats.from_rows(V, R, a), only, 0.01, 1e-9)
        np.testing.assert_allclose(b0["beta"][:, 0], (a[:, None] * R).sum(axis=0), atol=1e-14)  # unpenalized mean

    def test_unweighted_regional_penalty(self):
        # a leaf's shrinkage does not scale with its mass: objective equals the reference with lam * ||u||^2 per leaf
        t, V, R, w, leaf = random_case(3)
        w = np.where(leaf == "A", w * 20, w)
        leaves = ["A", "B"]
        active = t.active()
        a = w / w.sum()
        total = cr.Stats.from_rows(V, R, a)
        G = np.zeros((1, 2, 4, 8, 8)); b = np.zeros((1, 2, 4, 8)); m = np.zeros((1, 2))
        for k, L in enumerate(leaves):
            st = cr.Stats.from_rows(V[leaf == L], R[leaf == L], a[leaf == L])
            G[0, k], b[0, k], m[0, k] = st.G, st.b, st.mass
        sol = cr.solve_joint_batch(total, G, b, m, active, 0.01, 1.0, 1e-9)
        _, _, robj = reference_joint(V, R, w, leaf, leaves, active, 0.01, 1.0)
        np.testing.assert_allclose(sol["objective_q"][0], robj, atol=1e-12)

    def test_single_supported_leaf_reuses_b2_and_missing_rows_keep_shared_loss(self):
        contract = small_contract(**{"correction.support": {"n_eff": 1, "admins": 1, "target_months": 1}})
        t, V, R, w, leaf = random_case(4)
        n = len(w)
        pool = pt.Pool(V, R, w, np.arange(n) % 37, np.arange(n) % 11, np.ones(n, bool))
        pmap = pt.PartitionMap({"r": (0, 0.0)}, ["r0", "r1"])
        only_r0 = np.where(leaf == "A", "r0", None).astype(object)
        res = pt.fit_frozen_map(pool, t.active(), pmap, only_r0, contract, 0.1)
        self.assertEqual(res["g_kind"], "b2_reuse")
        two = np.where(leaf == "A", "r0", np.where(leaf == "B", "r1", None)).astype(object)
        res = pt.fit_frozen_map(pool, t.active(), pmap, two, contract, 0.1)
        self.assertEqual(res["g_kind"], "joint")
        b2 = res["b2"]["beta"]
        self.assertFalse(np.allclose(res["beta_g"], b2))  # G's shared response is its own fit (D18), not B2's

    def test_nonfinite_prediction_is_technical(self):
        P = np.array([[0.5, 0.4, 0.2, np.nan]])
        with self.assertRaises(TechnicalError):
            cr.Transform.fit(P, np.zeros((1, 3)), np.ones(1))


class BruteForceSearch(unittest.TestCase):
    def test_first_split_equals_exhaustive_reference(self):
        rng = np.random.default_rng(9)
        n_admin, per = 30, 6
        admin = np.repeat(np.arange(n_admin), per)
        months = np.tile(np.arange(per), n_admin)
        D = rng.normal(size=(n_admin, 3))
        Pq = rng.uniform(size=(len(admin), 4))
        Zq = rng.normal(size=(len(admin), 3))
        w = rng.uniform(0.5, 1.0, len(admin))
        t = cr.Transform.fit(Pq, Zq, w)
        V = t.design(Pq, Zq)
        R = 0.05 * rng.normal(size=(len(admin), 4)) + np.where(D[admin, 1] > 0.2, 0.08, -0.03)[:, None]
        contract = small_contract(**{"correction.support": {"n_eff": 20, "admins": 6, "target_months": 3}})
        pool = pt.Pool(V, R, w, admin, months, np.ones(len(admin), bool))
        search = pt.Search(pool, D, np.arange(n_admin), t.active(), contract, 0.1)
        states = search.run(2, 3)
        # exhaustive reference: every observed-rank cut, both-child support, joint objective by the null-space solver
        best = None
        f0 = cr.solve_b2(pool.total, t.active(), 0.01, 1e-9)["objective"]
        for j in range(3):
            for k, thr in enumerate(pt.cut_thresholds(D[:, j], np.arange(n_admin), 9)):
                left = D[admin, j] <= thr
                la, ra = np.unique(admin[left]), np.unique(admin[~left])
                if min(len(la), len(ra)) < 6:
                    continue
                neff = lambda x: x.sum() ** 2 / (x ** 2).sum()  # noqa: E731
                if neff(w[left]) < 20 or neff(w[~left]) < 20:
                    continue
                leaf = np.where(left, "L", "R").astype(object)
                _, _, robj = reference_joint(V, R, w, leaf, ["L", "R"], t.active(), 0.01, 0.1)
                gain = f0 - robj.mean()
                key = (-gain, j, k)
                if best is None or key < best[0]:
                    best = (key, j, thr, robj.mean())
        self.assertEqual(states[1]["map"].splits["r"], (best[1], best[2]))
        self.assertAlmostEqual(states[1]["objective"], best[3], places=12)

    def test_two_split_growth_equals_independent_reference_with_ties(self):
        """Independent best-first reference (own observed-rank thresholds, whole joint refit of every leaf by the
        null-space solver, strict floor, tie order node path -> descriptor -> threshold) through two splits.
        Descriptor 1 duplicates descriptor 0 (descriptor tie) and the residual pattern is mirror-symmetric across the
        root split (node tie at step 2)."""
        rng = np.random.default_rng(21)
        n_admin, per = 48, 5
        admin = np.repeat(np.arange(n_admin), per)
        months = np.tile(np.arange(per), n_admin)
        base = rng.permutation(n_admin).astype(float)
        D = np.column_stack([base, base, np.tile([0.0, 1.0], n_admin // 2)])
        a_sign = np.where(D[:, 0] < n_admin / 2, 1.0, -1.0)
        b_sign = np.where(D[:, 2] > 0.5, 1.0, -1.0)
        resid = (0.3 * a_sign + 0.1 * a_sign * b_sign)[admin]  # mirror: r0 and r1 split on descriptor 2 with equal gains
        V = np.zeros((len(admin), 4, 8)); V[:, :, 0] = 1.0
        R = np.repeat(resid[:, None], 4, axis=1)
        w = np.ones(len(admin))
        active = intercept_active()
        contract = small_contract(**{"correction.support": {"n_eff": 10, "admins": 6, "target_months": 2}})
        pool = pt.Pool(V, R, w, admin, months, np.ones(len(admin), bool))
        search = pt.Search(pool, D, np.arange(n_admin), active, contract, 0.01)
        states = search.run(3, 3)

        def ranks(vals, codes):
            pairs = sorted(zip(vals, codes))
            n, out = len(pairs), []
            for j in range(1, 10):
                t = pairs[math.ceil(j * n / 10) - 1][0]
                if t < pairs[-1][0] and t not in out:
                    out.append(t)
            return out

        def objective(leaf_of_row, leaves):
            if len(leaves) < 2:
                return cr.solve_b2(pool.total, active, 0.01, 1e-9)["objective"]
            _, _, obj = reference_joint(V, R, w, leaf_of_row, leaves, active, 0.01, 0.01)
            return obj.mean()

        membership = {"r": np.arange(n_admin)}
        f_cur = objective(None, ["r"])
        accepted = []
        for _step in range(2):
            cands = []
            for nid in sorted(membership):
                mem = membership[nid]
                for j in range(3):
                    vals = D[mem, j]
                    if vals.min() == vals.max():
                        continue
                    for order, thr in enumerate(ranks(vals.tolist(), mem.tolist())):
                        left = set(mem[vals <= thr].tolist())
                        right = set(mem.tolist()) - left
                        if min(len(left), len(right)) < 6:
                            continue
                        new = {k: v for k, v in membership.items() if k != nid}
                        new[nid + "0"], new[nid + "1"] = np.array(sorted(left)), np.array(sorted(right))
                        leaf = np.empty(len(admin), dtype=object)
                        for k, v in new.items():
                            leaf[np.isin(admin, v)] = k
                        cands.append((f_cur - objective(leaf, sorted(new)), nid, j, order, thr, new))
            tol = 1e-12 * max(1.0, abs(f_cur))
            best = max(c[0] for c in cands)
            tied = sorted([c for c in cands if c[0] >= best - tol and c[0] > tol], key=lambda c: (c[1], c[2], c[3]))
            pick = tied[0]
            accepted.append((pick[1], pick[2], pick[4]))
            membership, f_cur = pick[5], f_cur - pick[0]
        got = [(nid, states[k + 1]["map"].splits[nid][0], states[k + 1]["map"].splits[nid][1])
               for k, nid in enumerate([accepted[0][0], accepted[1][0]])]
        self.assertEqual(got, accepted)
        self.assertEqual(accepted[0][1], 0)  # descriptor 0 wins the exact tie with its duplicate column 1
        self.assertEqual(accepted[1][0], "r0")  # mirror-symmetric leaves: node path r0 precedes r1
        self.assertAlmostEqual(states[2]["objective"], f_cur, places=12)
        step2 = search.ledger_frame().query("step == 2")
        self.assertEqual(set(step2["node"]), {"r0", "r1"})  # every remaining leaf rescored under the joint refit
        g0 = step2[(step2.node == "r0") & (step2.descriptor_index == 2)]["gain"].max()
        g1 = step2[(step2.node == "r1") & (step2.descriptor_index == 2)]["gain"].max()
        self.assertAlmostEqual(g0, g1, delta=1e-12)

    def test_depth_limit_and_both_child_support(self):
        rng = np.random.default_rng(3)
        n_admin = 40
        admin = np.repeat(np.arange(n_admin), 5)
        D = rng.normal(size=(n_admin, 2))
        V = np.zeros((len(admin), 4, 8)); V[:, :, 0] = 1
        R = np.repeat((np.sign(D[admin, 0]) * 0.1 + np.sign(D[admin, 1]) * 0.05)[:, None], 4, axis=1)
        contract = small_contract(**{"correction.support": {"n_eff": 10, "admins": 15, "target_months": 1}})
        pool = pt.Pool(V, R, np.ones(len(admin)), admin, np.tile(np.arange(5), n_admin), np.ones(len(admin), bool))
        search = pt.Search(pool, D, np.arange(n_admin), intercept_active(), contract, 0.01)
        states = search.run(8, 1)
        self.assertEqual(states[-1]["map"].n_leaves, 2)  # depth limit 1 stops after the root split
        ledger = search.ledger_frame()
        small_side = ledger[(ledger["left_admins"] < 15) | (ledger["right_admins"] < 15)]
        self.assertTrue(len(small_side) > 0 and not small_side["supported"].any())
        self.assertFalse(ledger.loc[ledger["accepted"], "left_admins"].lt(15).any())


if __name__ == "__main__":
    unittest.main()
