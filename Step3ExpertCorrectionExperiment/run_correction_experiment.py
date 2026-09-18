#!/usr/bin/env python3
"""Entrypoint for the isolated Step 3 expert selective-correction experiment.

Run from the repository root, e.g.:

    .venv-geodt-diagnostic/bin/python \
        Step3ExpertCorrectionExperiment/run_correction_experiment.py \
        --run-id my_run --scopes 1 2

See ``Step3ExpertCorrectionExperiment/README.md`` for the contract gates that can
halt a run and for the exact full-experiment command.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from step3correction.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
