import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ARTIFACT_ROOT = REPO_ROOT / "scripts" / "paper_artifacts"


def load_script_module(module_name: str):
    if "geopandas" not in sys.modules:
        sys.modules["geopandas"] = types.SimpleNamespace(read_file=lambda *_args, **_kwargs: None)
    spec = importlib.util.spec_from_file_location(
        module_name,
        PAPER_ARTIFACT_ROOT / f"{module_name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PaperArtifactOutputDefaultTests(unittest.TestCase):
    def test_geodt_branch_location_output_preserves_pre_move_scripts_root(self):
        module = load_script_module("plot_geodt_branch_1_vs_011_locations")

        self.assertEqual(
            module.DEFAULT_OUTPUT,
            REPO_ROOT
            / "scripts"
            / "geodt_branch_1_vs_001_locations_2024-10_fs1_global.png",
        )

    def test_fewsnet_crisis_stack_output_preserves_pre_move_scripts_root(self):
        module = load_script_module("plot_fewsnet_crisis_stack_2018")

        self.assertEqual(
            module.DEFAULT_OUTPUT,
            REPO_ROOT / "scripts" / "fewsnet_crisis_stack_2018.png",
        )


if __name__ == "__main__":
    unittest.main()
