#!/usr/bin/env bash
# Sequential approved scenario phases (single launcher): e.g. `phase_launch.sh scen-freeze` or
# `phase_launch.sh scen-historical scen-report`. Pinned interpreter, diagnostic-only env (no
# numerical change), stops on the first failure; no retry.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: phase_launch.sh PHASE..." >&2; exit 2; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
G=/mnt/c/Users/swl00/geoxgb_runs; RUN='C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1'
PY=/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export WSLENV="${WSLENV:+$WSLENV:}PYTHONFAULTHANDLER/w:PYTHONUNBUFFERED/w"
cd "$REPO/FEWSNETGeoXGBExperiment"
[ -z "$(git status --porcelain -- .)" ] || { echo "package tree not clean" >&2; exit 3; }
for phase in "$@"; do
  echo "$(date -Is) HEAD=$(git rev-parse --short HEAD) CMD: $PY -B scripts/run_experiment.py --run-dir '$RUN' $phase" >> "$G/scen-b43ef6a-v1.commands.log"
  $PY -B scripts/run_experiment.py --run-dir "$RUN" "$phase" >> "$G/scen-b43ef6a-v1.${phase#scen-}.log" 2>&1
done
echo "$(date -Is) HEAD=$(git rev-parse --short HEAD) $* done" >> "$G/scen-b43ef6a-v1.commands.log"
