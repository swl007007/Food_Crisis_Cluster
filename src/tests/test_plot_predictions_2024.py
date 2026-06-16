import importlib.util
import sys
import types
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "plot_predictions_2024.py"


def load_script_module():
    if "geopandas" not in sys.modules:
        sys.modules["geopandas"] = types.SimpleNamespace(
            GeoDataFrame=object,
            read_file=lambda *_args, **_kwargs: None,
        )
    else:
        sys.modules["geopandas"].GeoDataFrame = getattr(
            sys.modules["geopandas"], "GeoDataFrame", object
        )
    if "contextily" not in sys.modules:
        providers = types.SimpleNamespace(CartoDB=types.SimpleNamespace(Positron=object()))
        sys.modules["contextily"] = types.SimpleNamespace(providers=providers, add_basemap=lambda *_args, **_kwargs: None)

    spec = importlib.util.spec_from_file_location("plot_predictions_2024", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlotPredictions2024Tests(unittest.TestCase):
    def test_default_args_use_main_georf_fs2_and_global_shapefile(self):
        module = load_script_module()

        args = module.parse_args([])

        self.assertEqual(args.predictions.parent.name, "result_partition_k40_compare_GF_fs2")
        self.assertEqual(args.predictions.name, "predictions_monthly.csv")
        self.assertEqual(args.title_source, "result_partition_k40_compare_GF_fs2")
        self.assertTrue(str(args.shapefile).endswith("FEWS_Admin_LZ_v3.shp"))


if __name__ == "__main__":
    unittest.main()
