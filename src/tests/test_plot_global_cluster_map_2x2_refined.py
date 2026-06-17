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
    spec.loader.exec_module(module)
    return module


class GlobalClusterMapSelectionTests(unittest.TestCase):
    def test_cli_description_uses_horizon_wording(self):
        module = load_script_module()
        source = inspect.getsource(module.parse_args)

        self.assertIn("4-month horizon", source)
        self.assertNotIn("4-month" + "-lag", source)

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

    def test_partition_palette_supports_ten_east_africa_clusters(self):
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

        _cmap, key_to_idx, summary, _key_to_color = module.build_partition_palette(
            panel_data,
            cluster_regions,
        )

        self.assertEqual(len([key for key in key_to_idx if key[0] == "m10"]), 10)
        self.assertEqual(summary["m10"][9], "East Africa")


if __name__ == "__main__":
    unittest.main()
