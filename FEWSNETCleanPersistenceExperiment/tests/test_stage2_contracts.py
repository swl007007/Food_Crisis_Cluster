"""Contract tests for the Stage 2 consensus map (D55-D60 / R59-R64).

These exercise the boundaries where a defect would silently change which areas get a
learned partition: the released weight/affinity/sparsification formulas, the D57 core
selection rule and its three-node floor, the D19 versus eigengap-nc=1 distinction, the
bounded D59 completion, and the R62/R63 requirement that "other component" and "never
in the graph" stay separately identifiable. Every fixture is small and synthetic;
nothing here builds a real map or fits a model.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_pipeline as rp  # noqa: E402


def plan(name: str, f1: float, f1_base: float, labels) -> rp.ConsensusPlan:
    weight = max(rp._logit_clip(f1) - rp._logit_clip(f1_base), 0.0)
    return rp.ConsensusPlan(name, f1, f1_base, weight, np.asarray(labels, dtype=np.int32))


# --------------------------------------------------------------------------------------
# D56.1: the released weight transform, including its clip pathology
# --------------------------------------------------------------------------------------


class TestConsensusWeights(unittest.TestCase):
    def test_no_improvement_is_zero_weight_not_negative(self) -> None:
        """A plan that ties or loses to pooled must contribute nothing, never a
        negative edge that could cancel another plan's vote."""
        self.assertEqual(plan("tie", 0.4, 0.4, [0]).weight, 0.0)
        self.assertEqual(plan("worse", 0.2, 0.5, [0]).weight, 0.0)

    def test_weight_is_the_logit_difference(self) -> None:
        computed = plan("better", 0.6, 0.4, [0]).weight
        expected = float(np.log(0.6 / 0.4) - np.log(0.4 / 0.6))
        self.assertAlmostEqual(computed, expected, places=12)

    def test_zero_baseline_is_flagged_and_dominates(self) -> None:
        """D56/R60 mandate the released clip, so a plan whose pooled baseline is
        exactly 0 outranks a genuinely better plan. This is recorded, not repaired
        (IMPLEMENTATION_LOG.md L3); the flag is what makes it auditable."""
        noise = plan("noise", 0.033, 0.0, [0])
        solid = plan("solid", 0.26, 0.25, [0])
        self.assertTrue(noise.f1_base_is_zero)
        self.assertFalse(solid.f1_base_is_zero)
        self.assertGreater(noise.weight, 10.0 * solid.weight)

    def test_clip_is_symmetric_at_both_ends(self) -> None:
        # Exact in real arithmetic; the two clipped endpoints differ by ~3e-11 in
        # float64, so this asserts the symmetry rather than bit equality.
        self.assertAlmostEqual(rp._logit_clip(0.0), -rp._logit_clip(1.0), places=9)


# --------------------------------------------------------------------------------------
# D56.2: co-membership only from positive-weight, actually-assigned areas
# --------------------------------------------------------------------------------------


class TestCoMembership(unittest.TestCase):
    def test_zero_weight_plan_adds_no_edge(self) -> None:
        plans = [plan("zero", 0.4, 0.4, [0, 0, 0])]
        matrix = rp.accumulate_co_membership(plans, 3)
        self.assertTrue(np.all(matrix == 0.0))

    def test_unassigned_areas_never_co_group(self) -> None:
        """-1 is absence of assignment. Two areas both missing from a plan must not
        become similar to each other because of that shared absence."""
        plans = [plan("p", 0.6, 0.4, [0, -1, -1])]
        matrix = rp.accumulate_co_membership(plans, 3)
        self.assertGreater(matrix[0, 0], 0.0)
        self.assertEqual(matrix[1, 2], 0.0)
        self.assertEqual(matrix[2, 1], 0.0)

    def test_weights_accumulate_across_plans(self) -> None:
        a = plan("a", 0.6, 0.4, [0, 0])
        b = plan("b", 0.7, 0.5, [0, 0])
        matrix = rp.accumulate_co_membership([a, b], 2)
        self.assertAlmostEqual(float(matrix[0, 1]), a.weight + b.weight, places=5)

    def test_different_partitions_do_not_connect(self) -> None:
        matrix = rp.accumulate_co_membership([plan("p", 0.6, 0.4, [0, 1])], 2)
        self.assertEqual(matrix[0, 1], 0.0)


# --------------------------------------------------------------------------------------
# D56.2/D56.3: the released Gaussian, normalization and top-k conventions
# --------------------------------------------------------------------------------------


class TestAffinityShaping(unittest.TestCase):
    def test_gaussian_diagonal_is_one_and_decays_with_degrees(self) -> None:
        lat = np.array([0.0, 0.0, 0.0])
        lon = np.array([0.0, 5.0, 40.0])
        weight = rp.gaussian_spatial_weight(lat, lon, rp.SIGMA_DEGREES)
        self.assertAlmostEqual(float(weight[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(weight[0, 1]), float(np.exp(-0.5)), places=4)
        # Cross-continental separation is annihilated, which is why other continents
        # can only ever form their own components (see IMPLEMENTATION_LOG.md L8).
        self.assertLess(float(weight[0, 2]), 1e-13)

    def test_normalization_is_a_noop_without_positive_mass(self) -> None:
        zeros = np.zeros((2, 2), dtype=np.float32)
        self.assertTrue(np.array_equal(rp.normalize_by_max(zeros), zeros))

    def test_normalization_scales_the_maximum_to_one(self) -> None:
        matrix = np.array([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        self.assertAlmostEqual(float(rp.normalize_by_max(matrix).max()), 1.0, places=6)

    def test_small_graphs_retain_every_edge(self) -> None:
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
        sparse = rp.sparsify_top_k(matrix, rp.KNN_K)
        self.assertEqual(sparse.nnz, 4)

    def test_symmetric_union_keeps_one_sided_neighbours(self) -> None:
        """k is a per-row cap, but the symmetric maximum means a strong edge kept by
        only one endpoint survives for both. Degree is therefore not bounded by k."""
        matrix = np.array([
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.8],
            [0.1, 0.8, 1.0],
        ], dtype=np.float32)
        sparse = rp.sparsify_top_k(matrix, 2).toarray()
        self.assertAlmostEqual(float(sparse[1, 2]), 0.8, places=6)
        self.assertAlmostEqual(float(sparse[2, 1]), 0.8, places=6)


# --------------------------------------------------------------------------------------
# D57/R61: one core graph for both selection and fitting
# --------------------------------------------------------------------------------------


class TestCoreComponent(unittest.TestCase):
    @staticmethod
    def _sparse(matrix):
        from scipy.sparse import csr_matrix

        return csr_matrix(np.asarray(matrix, dtype=np.float32))

    def test_self_entries_do_not_join_nodes(self) -> None:
        """A node whose only positive entry is its own diagonal is isolated. If the
        diagonal were counted, every node would look connected."""
        affinity = self._sparse([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        core = rp.select_core_component(affinity, np.array([10, 11, 12]))
        self.assertEqual(core["component_count"], 3)
        self.assertEqual(core["largest_component_size"], 1)

    def test_largest_component_wins(self) -> None:
        affinity = self._sparse([
            [1.0, 0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        core = rp.select_core_component(affinity, np.array([7, 8, 9, 10]))
        self.assertEqual(core["largest_component_size"], 2)
        self.assertEqual(sorted(core["member_indices"].tolist()), [0, 1])

    def test_equal_size_tie_breaks_on_smallest_canonical_code(self) -> None:
        affinity = self._sparse([
            [1.0, 0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.5, 1.0],
        ])
        codes = np.array([101, 102, 3, 4])
        core = rp.select_core_component(affinity, codes)
        self.assertEqual(core["tied_components"], 2)
        self.assertEqual(sorted(codes[core["member_indices"]].tolist()), [3, 4])

    def test_two_node_core_is_an_incomplete_build_not_a_no_split(self) -> None:
        """R61's floor exists because k_eigen = max(2, min(20, n-2)) only satisfies
        k < n once n >= 3. Failing here must not be relabelled as evidence of no split."""
        affinity = self._sparse([[1.0, 0.5], [0.5, 1.0]])
        with self.assertRaises(rp.PipelineError) as caught:
            rp._eigengap_recommendation(affinity)
        self.assertIn("incomplete build", str(caught.exception))


# --------------------------------------------------------------------------------------
# D56.4: eigengap cluster selection
# --------------------------------------------------------------------------------------


class TestEigengapSelection(unittest.TestCase):
    def test_two_clear_blocks_are_recovered(self) -> None:
        from scipy.sparse import csr_matrix

        block = np.zeros((6, 6), dtype=np.float64)
        block[:3, :3] = 1.0
        block[3:, 3:] = 1.0
        np.fill_diagonal(block, 1.0)
        result = rp._eigengap_recommendation(csr_matrix(block))
        self.assertEqual(result["selected_nc"], 2)
        self.assertEqual(result["k_eigen"], max(2, min(20, 6 - 2)))

    def test_initialization_is_pinned_and_repeatable(self) -> None:
        """The released eigsh passes no v0. D56.5 requires a pinned start, so repeated
        calls must agree exactly rather than drift with ARPACK's random vector."""
        from scipy.sparse import csr_matrix

        block = np.zeros((8, 8), dtype=np.float64)
        block[:4, :4] = 1.0
        block[4:, 4:] = 1.0
        matrix = csr_matrix(block)
        first = rp._eigengap_recommendation(matrix)
        second = rp._eigengap_recommendation(matrix)
        self.assertEqual(first["eigenvalues"], second["eigenvalues"])
        self.assertEqual(first["selected_nc"], second["selected_nc"])


# --------------------------------------------------------------------------------------
# D59/R63: bounded nearest-donor completion
# --------------------------------------------------------------------------------------


class TestGeographicCompletion(unittest.TestCase):
    @staticmethod
    def _coords(rows) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["FEWSNET_admin_code", "lat", "lon"])

    def test_near_recipient_takes_its_donor_label(self) -> None:
        coords = self._coords([[1, 0.0, 0.0], [2, 0.0, 0.5]])  # ~55 km apart
        out = rp.complete_geographically(
            np.array([1]), np.array([7]), np.array([2]), coords
        )
        row = out.iloc[0]
        self.assertEqual(int(row["partition_id"]), 7)
        self.assertEqual(row["assignment_source"], rp.SUPPORT_COMPLETED)
        self.assertLess(float(row["donor_distance_km"]), rp.COMPLETION_MAX_KM)

    def test_far_recipient_stays_unassigned_but_keeps_its_evidence(self) -> None:
        """Beyond 100 km the area routes to pooled, and the rejected donor and its
        distance are still recorded so the decision is auditable."""
        coords = self._coords([[1, 0.0, 0.0], [2, 0.0, 10.0]])
        out = rp.complete_geographically(
            np.array([1]), np.array([7]), np.array([2]), coords
        )
        row = out.iloc[0]
        self.assertEqual(int(row["partition_id"]), rp.UNASSIGNED_PARTITION)
        self.assertEqual(row["assignment_source"], rp.SUPPORT_UNASSIGNED_FAR)
        self.assertEqual(int(row["donor_code"]), 1)
        self.assertGreater(float(row["donor_distance_km"]), rp.COMPLETION_MAX_KM)

    def test_exact_distance_ties_go_to_the_smallest_donor_code(self) -> None:
        coords = self._coords([[1, 0.0, -0.2], [5, 0.0, 0.2], [9, 0.0, 0.0]])
        out = rp.complete_geographically(
            np.array([1, 5]), np.array([11, 22]), np.array([9]), coords
        )
        self.assertEqual(int(out.iloc[0]["donor_code"]), 1)
        self.assertEqual(int(out.iloc[0]["partition_id"]), 11)

    def test_missing_coordinates_leave_the_area_unassigned(self) -> None:
        """D59.4 is an explicit approved difference from IPCCH: a missing coordinate
        routes to pooled rather than failing the build or imputing a location."""
        coords = self._coords([[1, 0.0, 0.0], [2, np.nan, np.nan]])
        out = rp.complete_geographically(
            np.array([1]), np.array([7]), np.array([2]), coords
        )
        row = out.iloc[0]
        self.assertEqual(int(row["partition_id"]), rp.UNASSIGNED_PARTITION)
        self.assertEqual(row["assignment_source"], rp.SUPPORT_UNASSIGNED_NO_COORD)
        self.assertIsNone(row["donor_code"])

    def test_completed_areas_never_become_donors(self) -> None:
        """A chain of three areas 60 km apart must not propagate: the far end is 120 km
        from the only real donor and stays unassigned."""
        coords = self._coords([[1, 0.0, 0.0], [2, 0.0, 0.54], [3, 0.0, 1.08]])
        out = rp.complete_geographically(
            np.array([1]), np.array([7]), np.array([2, 3]), coords
        ).set_index("FEWSNET_admin_code")
        self.assertEqual(int(out.loc[2, "partition_id"]), 7)
        self.assertEqual(int(out.loc[3, "partition_id"]), rp.UNASSIGNED_PARTITION)

    def test_unsorted_donor_codes_are_rejected(self) -> None:
        coords = self._coords([[1, 0.0, 0.0], [2, 0.0, 0.1], [3, 0.0, 0.2]])
        with self.assertRaises(rp.PipelineError):
            rp.complete_geographically(
                np.array([2, 1]), np.array([0, 1]), np.array([3]), coords
            )


# --------------------------------------------------------------------------------------
# D58/R62: node universe and support reasons
# --------------------------------------------------------------------------------------


class TestNodeUniverseAndSupport(unittest.TestCase):
    def _role_dir(self, root: Path, plans: dict, scores: dict) -> Path:
        role = root / "role"
        (role / "linked_tables" / "partitions").mkdir(parents=True)
        pd.DataFrame([
            {"name": name, "f1(1)": scores[name][0], "f1_base(1)": scores[name][1]}
            for name in plans
        ]).to_csv(role / "linked_tables" / "main_index.csv", index=False)
        for name, assignment in plans.items():
            pd.DataFrame({
                "FEWSNET_admin_code": list(assignment),
                "partition_id": list(assignment.values()),
            }).to_csv(
                role / "linked_tables" / "partitions" / f"{name}_partition.csv", index=False
            )
        return role

    def test_out_of_scope_areas_are_excluded_from_the_universe(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            role = self._role_dir(
                Path(tmp),
                {"a": {1: "00", 2: "s-1", 3: "01"}},
                {"a": (0.6, 0.4)},
            )
            _, codes, ledger = rp.load_consensus_plans(role)
            self.assertEqual(codes.tolist(), [1, 3])
            self.assertEqual(ledger["node_universe"], 2)

    def test_zero_weight_plans_contribute_nodes_but_no_edges(self) -> None:
        """D58 keeps their support in the ledger; D56 gives them no similarity. Both
        halves matter: dropping the node would hide real coverage, and adding the edge
        would fabricate evidence from a plan that beat nothing."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            role = self._role_dir(
                Path(tmp),
                {"zero": {1: "00", 2: "00"}, "pos": {3: "00", 4: "00"}},
                {"zero": (0.4, 0.4), "pos": (0.6, 0.4)},
            )
            plans, codes, ledger = rp.load_consensus_plans(role)
            self.assertEqual(codes.tolist(), [1, 2, 3, 4])
            self.assertEqual(ledger["zero_weight_plans"], 1)
            self.assertEqual(ledger["nodes_supported_only_by_zero_weight_plans"], 2)
            matrix = rp.accumulate_co_membership(plans, codes.size)
            self.assertEqual(float(matrix[0, 1]), 0.0)
            self.assertGreater(float(matrix[2, 3]), 0.0)

    def test_empty_pool_is_an_error_not_an_unsplit_result(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            role = Path(tmp) / "role"
            (role / "linked_tables" / "partitions").mkdir(parents=True)
            pd.DataFrame(columns=["name", "f1(1)", "f1_base(1)"]).to_csv(
                role / "linked_tables" / "main_index.csv", index=False
            )
            with self.assertRaises(rp.PipelineError) as caught:
                rp.load_consensus_plans(role)
            self.assertIn("not a no-split result", str(caught.exception))

    def test_null_assignments_never_create_co_membership(self) -> None:
        """A null partition_id is absence of assignment. Reading it as the string
        'nan' would intern it as a real partition and connect every unassigned area
        to every other, which D58 explicitly forbids."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            role = root / "role"
            (role / "linked_tables" / "partitions").mkdir(parents=True)
            pd.DataFrame([{"name": "a", "f1(1)": 0.6, "f1_base(1)": 0.4}]).to_csv(
                role / "linked_tables" / "main_index.csv", index=False
            )
            pd.DataFrame({
                "FEWSNET_admin_code": [1, 2, 3],
                "partition_id": ["00", None, None],
            }).to_csv(role / "linked_tables" / "partitions" / "a_partition.csv", index=False)

            plans, codes, _ = rp.load_consensus_plans(role)
            self.assertEqual(codes.tolist(), [1])
            self.assertEqual(plans[0].labels.tolist(), [0])
            matrix = rp.accumulate_co_membership(plans, codes.size)
            self.assertEqual(matrix.shape, (1, 1))


class TestSupportClassification(unittest.TestCase):
    """Behavioural checks on graph_support. Asserting only that the three labels are
    distinct strings would pass even if every area were classified wrongly."""

    def test_the_three_states_partition_the_master_cohort(self) -> None:
        """Calls the production classifier, not a copy of its expression."""
        core = np.array([1, 2])
        graph = np.array([1, 2, 3])              # 3 is in the graph, outside the core
        master = np.array([1, 2, 3, 4])          # 4 was never in the graph
        support = rp.classify_graph_support(master, core, graph)
        self.assertEqual(list(support), [
            rp.GRAPH_IN_CORE, rp.GRAPH_IN_CORE,
            rp.GRAPH_OTHER_COMPONENT, rp.GRAPH_NEVER_IN_GRAPH,
        ])

    def test_core_membership_wins_over_graph_membership(self) -> None:
        """Core areas are also graph areas; the order of the checks must not flip."""
        support = rp.classify_graph_support(
            np.array([7]), np.array([7]), np.array([7])
        )
        self.assertEqual(support[0], rp.GRAPH_IN_CORE)

    def test_classification_is_not_uniform(self) -> None:
        """Guards against a regression that labels everything the same way."""
        support = rp.classify_graph_support(
            np.array([1, 2, 3]), np.array([1]), np.array([1, 2])
        )
        self.assertEqual(len(set(support)), 3)

    def test_out_of_range_coordinates_are_not_merely_finite(self) -> None:
        """lon=360 is finite and wraps to lon=0 under haversine, landing ~1e-12 km from
        a donor there. Bounds, not finiteness, decide usability."""
        coords = pd.DataFrame(
            [[1, 0.0, 0.0], [2, 0.0, 360.0], [3, 95.0, 0.0]],
            columns=["FEWSNET_admin_code", "lat", "lon"],
        )
        out = rp.complete_geographically(
            np.array([1]), np.array([7]), np.array([2, 3]), coords
        ).set_index("FEWSNET_admin_code")
        for code in (2, 3):
            self.assertEqual(int(out.loc[code, "partition_id"]), rp.UNASSIGNED_PARTITION)
            self.assertEqual(
                out.loc[code, "assignment_source"], rp.SUPPORT_UNASSIGNED_NO_COORD
            )

    def test_invalid_core_coordinates_fail_the_build(self) -> None:
        coords = pd.DataFrame(
            [[1, 0.0, 999.0], [2, 0.0, 0.1]],
            columns=["FEWSNET_admin_code", "lat", "lon"],
        )
        with self.assertRaises(rp.PipelineError):
            rp.complete_geographically(
                np.array([1]), np.array([7]), np.array([2]), coords
            )


if __name__ == "__main__":
    unittest.main()
