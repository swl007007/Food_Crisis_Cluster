#!/usr/bin/env bash
# Full bounded GeoXGBoost experiment (task experiment-plan.md v1.0) into one run directory
# outside Dropbox. Phases are idempotent only at completed-record granularity; nothing
# is overwritten. The tests and the verifier are separate acceptance commands.
set -euo pipefail
RUN=${1:?run directory}
W=${WORKERS:-6}
PY=${PYTHON_EXE:-/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe}
cd "$(dirname "$0")"
$PY -B scripts/prepare_fourclass.py --run-dir "$RUN"
$PY -B scripts/run_experiment.py --run-dir "$RUN" gscreen --workers "$W"
$PY -B scripts/run_stage1.py --run-dir "$RUN" --workers "$W"
for phase in maps develop; do $PY -B scripts/run_experiment.py --run-dir "$RUN" $phase --workers "$W"; done
$PY -B scripts/run_experiment.py --run-dir "$RUN" select
$PY -B scripts/run_experiment.py --run-dir "$RUN" oldmap --workers "$W"
$PY -B scripts/run_experiment.py --run-dir "$RUN" freeze
$PY -B scripts/run_experiment.py --run-dir "$RUN" final --workers "$W"
$PY -B scripts/report_fourclass.py --run-dir "$RUN"
