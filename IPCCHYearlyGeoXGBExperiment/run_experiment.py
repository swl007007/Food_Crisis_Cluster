"""``<P6-locked-python> run_experiment.py [--config DIR] --run-dir DIR <command>`` (same as ``-m ipcch_yearly_xgb``).

Run from this directory, or set PYTHONPATH to it; no sys.path injection.
"""
from ipcch_yearly_xgb.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
