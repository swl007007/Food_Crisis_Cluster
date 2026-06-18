import importlib.util
import inspect
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "plot_global_cluster_map_2x2_refined.py"


def load_script_module():
    if "contextily" not in sys.modules:
        providers = types.SimpleNamespace(
            CartoDB=types.SimpleNamespace(PositronNoLabels=object()),
            OpenStreetMap=types.SimpleNamespace(Mapnik=object()),
        )
        sys.modules["contextily"] = types.SimpleNamespace(providers=providers)
    if "geopandas" not in sys.modules:
        sys.modules["geopandas"] = types.SimpleNamespace(read_file=lambda *_args, **_kwargs: None)

    spec = importlib.util.spec_from_file_location("plot_global_cluster_map_2x2_refined", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GlobalClusterMapSelectionTests(unittest.TestCase):
    def test_cli_description_uses_horizon_wording(self):
        module = load_script_module()
        source = inspect.getsource(module.parse_args)

        self.assertIn("4-month horizon", source)
        self.assertNotIn("4-month" + "-lag", source)

    def test_default_source_dir_is_repo_root_not_ablation_archive(self):
        module = load_script_module()

        self.assertEqual(module.DEFAULT_SOURCE_DIR, module.REPO_ROOT)
        self.assertNotIn("main_ablation_results", str(module.DEFAULT_SOURCE_DIR))
        self.assertNotIn("archived", str(module.DEFAULT_SOURCE_DIR))

    def test_choose_mapping_prefers_latest_file_over_stale_higher_nc(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmp:
            refined_dir = Path(tmp)
            stale = refined_dir / (
                "cluster_mapping_k40_nc17_general_refined_contig3_"
                "refined_contig3_refined_contig3.csv"
            )
            latest = refined_dir / "cluster_mapping_k40_nc15_general_refined_contig3.csv"
            stale.write_text("FEWSNET_admin_code,cluster_id\n1,1\n", encoding="utf-8")
            latest.write_text("FEWSNET_admin_code,cluster_id\n1,2\n", encoding="utf-8")
            os.utime(stale, (100.0, 100.0))
            os.utime(latest, (200.0, 200.0))

            self.assertEqual(module.choose_mapping(refined_dir, "general"), latest)

    def test_discover_model_csvs_supports_experiment_workspace_source(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping_dir = root / "GeoRFExperiment" / "knn_sparsification_results"
            mapping_dir.mkdir(parents=True)
            expected = {}
            for panel in module.PANEL_ORDER:
                path = mapping_dir / f"cluster_mapping_k40_nc13_{panel}.csv"
                path.write_text("FEWSNET_admin_code,cluster_id\n1,1\n", encoding="utf-8")
                expected[panel] = path

            self.assertEqual(module.discover_model_csvs(root, "GeoRF"), expected)

    def test_partition_styles_support_ten_east_africa_clusters(self):
        module = load_script_module()
        panel_data = {}
        for panel in module.PANEL_ORDER:
            if panel == "m10":
                panel_data[panel] = pd.DataFrame(
                    {
                        "cluster_id": list(range(10)),
                        "region_group": ["East Africa"] * 10,
                    }
                )
            else:
                panel_data[panel] = pd.DataFrame(
                    {
                        "cluster_id": [0],
                        "region_group": ["West Africa"],
                    }
                )
        cluster_regions = {
            (panel, int(row.cluster_id)): str(row.region_group)
            for panel, df in panel_data.items()
            for row in df.itertuples(index=False)
        }

        key_to_style, summary = module.build_partition_styles(panel_data, cluster_regions)

        self.assertEqual(len([key for key in key_to_style if key[0] == "m10"]), 10)
        self.assertEqual(summary["m10"][9], "East Africa")

    def test_partition_styles_support_many_east_africa_partitions_with_hatches(self):
        module = load_script_module()
        panel_data = {}
        for panel in module.PANEL_ORDER:
            if panel == "m10":
                panel_data[panel] = pd.DataFrame(
                    {
                        "cluster_id": list(range(12)),
                        "region_group": ["East Africa"] * 12,
                    }
                )
            else:
                panel_data[panel] = pd.DataFrame(
                    {
                        "cluster_id": [0],
                        "region_group": ["West Africa"],
                    }
                )
        cluster_regions = {
            (panel, int(row.cluster_id)): str(row.region_group)
            for panel, df in panel_data.items()
            for row in df.itertuples(index=False)
        }

        key_to_style, summary = module.build_partition_styles(panel_data, cluster_regions)

        m10_styles = [key_to_style[("m10", cluster_id)] for cluster_id in range(12)]
        self.assertEqual(len(m10_styles), 12)
        self.assertEqual(summary["m10"][11], "East Africa")
        self.assertGreater(len({style.hatch for style in m10_styles}), 1)
        self.assertEqual({style.facecolor for style in m10_styles}, {"#ffffff"})
        self.assertEqual(len({(style.facecolor, style.hatch) for style in m10_styles}), 12)

    def test_partition_styles_use_cluster_id_hatches_across_panels(self):
        module = load_script_module()
        panel_data = {
            "general": pd.DataFrame({"cluster_id": [3, 4], "region_group": ["West Africa", "West Africa"]}),
            "m2": pd.DataFrame({"cluster_id": [3], "region_group": ["East Africa"]}),
            "m6": pd.DataFrame({"cluster_id": [0], "region_group": ["West Africa"]}),
            "m10": pd.DataFrame({"cluster_id": [0], "region_group": ["West Africa"]}),
        }
        cluster_regions = {
            (panel, int(row.cluster_id)): str(row.region_group)
            for panel, df in panel_data.items()
            for row in df.itertuples(index=False)
        }

        key_to_style, _summary = module.build_partition_styles(panel_data, cluster_regions)

        self.assertEqual(key_to_style[("general", 3)].hatch, key_to_style[("m2", 3)].hatch)
        self.assertEqual(key_to_style[("general", 3)].facecolor, "#ffffff")
        self.assertEqual(key_to_style[("m2", 3)].facecolor, "#ffffff")
        self.assertEqual(key_to_style[("general", 3)].hatch, module.hatch_for_cluster_id(3))

    def test_cluster_id_hatches_are_unique_for_current_cluster_range(self):
        module = load_script_module()
        hatches = [module.hatch_for_cluster_id(cluster_id) for cluster_id in range(20)]

        self.assertEqual(len(set(hatches)), 20)

    def test_compact_legend_labels_are_cluster_ids_only(self):
        module = load_script_module()
        labels = module.compact_cluster_labels([0, 2, 13])

        self.assertEqual(labels, ["c0", "c2", "c13"])
        self.assertNotIn("m2", " ".join(labels))
        self.assertNotIn("WA", " ".join(labels))

    def test_plot_model_grid_uses_admin0_context_not_tile_basemap_or_polygon_boundaries(self):
        module = load_script_module()
        source = inspect.getsource(module.plot_model_grid)

        self.assertIn("load_admin0_basemap", source)
        self.assertIn("plot_admin0_context", source)
        self.assertIn("plot_admin0_outline", source)
        self.assertNotIn("cx.add_basemap", source)
        self.assertNotIn("boundary_layer", source)
        self.assertNotIn("main_boundary", source)

    def test_partition_layer_does_not_draw_fewsnet_polygon_boundary_overlay(self):
        module = load_script_module()
        source = inspect.getsource(module.plot_partition_layer)

        self.assertNotIn("boundary_gdf", source)
        self.assertNotIn(".boundary.plot", source)
        self.assertIn("dissolve_plot_layer", source)

    def test_admin0_basemap_simplification_is_crs_aware_for_latam_inset(self):
        module = load_script_module()
        source = inspect.getsource(module.load_admin0_basemap)

        self.assertIn("is_geographic", source)
        self.assertIn("PLOT_SIMPLIFY_TOLERANCE_DEG", source)
        self.assertIn("PLOT_SIMPLIFY_TOLERANCE_M", source)


if __name__ == "__main__":
    unittest.main()
