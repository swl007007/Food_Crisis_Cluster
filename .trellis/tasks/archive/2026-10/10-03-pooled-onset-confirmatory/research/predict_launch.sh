#!/usr/bin/env bash
set -euo pipefail
cd "/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster/FEWSNETGeoXGBExperiment"
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export WSLENV="${WSLENV:+$WSLENV:}PYTHONFAULTHANDLER/w:PYTHONUNBUFFERED/w"
/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe -B '..\.trellis\tasks\10-03-pooled-onset-confirmatory\research\run_confirm.py' --run-dir 'C:\Users\swl00\geoxgb_runs\confirm-onset-v1' predict
