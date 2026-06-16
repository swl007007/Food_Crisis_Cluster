import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "plot_georf_m2_adjacency_refinement.py"
spec = importlib.util.spec_from_file_location("plot_georf_m2_adjacency_refinement", SCRIPT_PATH)
refinement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refinement)


class GeoRFM2AdjacencyRefinementTests(unittest.TestCase):
    def test_reassignment_table_aligns_pre_and_post_clusters(self):
        pre = pd.DataFrame(
            {
                "FEWSNET_admin_code": ["1", "2", "3"],
                "cluster_id": [4, 4, 8],
            }
        )
        post = pd.DataFrame(
            {
                "FEWSNET_admin_code": ["1", "2", "3"],
                "cluster_id": [4, 5, 8],
                "cluster_id_original": [4, 4, 8],
            }
        )

        table = refinement.build_reassignment_table(pre, post)

        self.assertEqual(table["FEWSNET_admin_code"].tolist(), ["1", "2", "3"])
        self.assertEqual(table["cluster_id_before"].tolist(), [4, 4, 8])
        self.assertEqual(table["cluster_id_after"].tolist(), [4, 5, 8])
        self.assertEqual(table["reassigned"].tolist(), [False, True, False])

    def test_summary_reports_reassignment_rate_and_cluster_counts(self):
        table = pd.DataFrame(
            {
                "cluster_id_before": [1, 1, 2, 2],
                "cluster_id_after": [1, 3, 2, 2],
                "reassigned": [False, True, False, False],
            }
        )

        row = refinement.summarize_reassignment(table, iterations=3, iteration_move_count=5)

        self.assertEqual(row["mapping"], "GeoRF m2")
        self.assertEqual(row["iterations"], 3)
        self.assertEqual(row["n_polygons"], 4)
        self.assertEqual(row["n_final_changed_polygons"], 1)
        self.assertEqual(row["final_changed_pct"], 25.0)
        self.assertEqual(row["n_iteration_reassignment_moves"], 5)
        self.assertEqual(row["n_clusters_before"], 2)
        self.assertEqual(row["n_clusters_after"], 3)

    def test_parse_refinement_log_extracts_iterations_and_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.txt"
            path.write_text(
                "\n".join(
                    [
                        "Iterations: 3",
                        "Iteration 1/3: 29 polygons reassigned",
                        "Iteration 2/3: 10 polygons reassigned",
                        "Iteration 3/3: 6 polygons reassigned",
                        "  Total reassigned: 45 polygons",
                    ]
                ),
                encoding="utf-8",
            )

            parsed = refinement.parse_refinement_log(path)

        self.assertEqual(parsed["iterations"], 3)
        self.assertEqual(parsed["total_reassigned"], 45)
        self.assertEqual(parsed["per_iteration_reassigned"], [29, 10, 6])


if __name__ == "__main__":
    unittest.main()
