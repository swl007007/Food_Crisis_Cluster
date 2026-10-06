"""Thin wrapper: ``<locked-python> run_experiment.py <command> --run-dir DIR`` (same as ``-m ipcch_mlp``)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ipcch_mlp.cli import main  # noqa: E402

raise SystemExit(main())
