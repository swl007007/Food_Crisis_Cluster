"""IPCCH AlphaEarth supervised-partition experiment (task 10-10-ipcch-alphaearth-supervised-partition).

Four roles per horizon: B0 compact296 pooled XGB, B1 compact296 + AlphaEarth128 pooled XGB,
B2 = B1 + shared ridge calibration, G = B1 + jointly fitted shared calibration and shrunk regional
deviations on a frozen supervised AlphaEarth map. Contracts: config/experiment-contract.json.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = EXPERIMENT_ROOT / "config"
REPO_ROOT = EXPERIMENT_ROOT.parent
#: Run directories live on the WSL filesystem, never under Dropbox (design section 13).
DEFAULT_RUNS_ROOT = Path.home() / "ipcch-alphaearth-runs"
