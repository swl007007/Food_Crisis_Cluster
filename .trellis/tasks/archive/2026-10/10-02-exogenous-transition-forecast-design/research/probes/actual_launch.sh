#!/usr/bin/env bash
# scen-actual single launcher (prediction-only; truth never loaded). Usage:
#   actual_launch.sh 'C:\...\actual_availability.csv' 'C:\...\extension_manifest.v2.json'
# Pinned interpreter, diagnostic-only env (no numerical change), clean package tree, fresh output
# (refuses if scenario_actual exists), stops on failure; no retry. Writes the PID via the caller's exec.
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: actual_launch.sh TABLE_CSV_WINPATH MANIFEST_JSON_WINPATH" >&2; exit 2; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
G=/mnt/c/Users/swl00/geoxgb_runs; RUN='C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1'
PY=/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export WSLENV="${WSLENV:+$WSLENV:}PYTHONFAULTHANDLER/w:PYTHONUNBUFFERED/w"
cd "$REPO/FEWSNETGeoXGBExperiment"
[ -z "$(git status --porcelain -- .)" ] || { echo "package tree not clean" >&2; exit 3; }
[ ! -e "$G/scen-b43ef6a-v1/scenario_actual" ] || { echo "scenario_actual exists; refusing" >&2; exit 4; }
echo "$(date -Is) HEAD=$(git rev-parse --short HEAD) CMD: $PY -B scripts/run_experiment.py --run-dir '$RUN' scen-actual --actual-availability '$1' --actual-scaffold '$2'" >> "$G/scen-b43ef6a-v1.commands.log"
$PY -B scripts/run_experiment.py --run-dir "$RUN" scen-actual --actual-availability "$1" --actual-scaffold "$2" >> "$G/scen-b43ef6a-v1.actual.log" 2>&1
echo "$(date -Is) HEAD=$(git rev-parse --short HEAD) scen-actual done" >> "$G/scen-b43ef6a-v1.commands.log"
