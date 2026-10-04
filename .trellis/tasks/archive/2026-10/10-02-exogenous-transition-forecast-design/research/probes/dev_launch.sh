#!/usr/bin/env bash
# Approved sequential development launch: save the Stage 1 acceptance summary, then scen-develop
# (72 folds) and scen-select. Diagnostic-only Windows env (faulthandler, unbuffered); no numerical
# change. Stops on the first failure; no retry.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
G=/mnt/c/Users/swl00/geoxgb_runs; RUN='C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1'
PY=/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export WSLENV="${WSLENV:+$WSLENV:}PYTHONFAULTHANDLER/w:PYTHONUNBUFFERED/w"
cd "$REPO/FEWSNETGeoXGBExperiment"
log() { echo "$(date -Is) HEAD=$(git rev-parse --short HEAD) $*" >> "$G/scen-b43ef6a-v1.commands.log"; }
log "ACCEPT accept_scenario_stage1 -> scen-b43ef6a-v1.stage1_acceptance.json"
$PY -B -c "
import json, collections, datetime
from pathlib import Path
from src.utils.acceptance import accept_scenario_stage1
r = accept_scenario_stage1(Path(r'$RUN'))
st = collections.Counter(v['status'] for v in r.values())
assert len(r) == 648 and st == {'completed': 648}, (len(r), st)
Path(r'C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1.stage1_acceptance.json').write_text(json.dumps({
  'accepted_at': datetime.datetime.now().isoformat(), 'function': 'src.utils.acceptance.accept_scenario_stage1',
  'scheduled': 648, 'accepted': len(r), 'status_counts': dict(st),
  'attempts': {'total_candidate_attempts': 669, 'attempt1_completed': 627, 'attempt1_failed_0xC0000005': 21,
               'resume1_completed': 21, 'resume1_failed': 0},
  'native_failure_attribution': 'unknown',
  'candidates': {k: v['status'] for k, v in sorted(r.items())}}, indent=1))
print('accepted', len(r), dict(st))
" >> "$G/scen-b43ef6a-v1.develop.log" 2>&1
log "CMD: $PY -B scripts/run_experiment.py --run-dir '$RUN' scen-develop"
$PY -B scripts/run_experiment.py --run-dir "$RUN" scen-develop >> "$G/scen-b43ef6a-v1.develop.log" 2>&1
log "CMD: $PY -B scripts/run_experiment.py --run-dir '$RUN' scen-select"
$PY -B scripts/run_experiment.py --run-dir "$RUN" scen-select >> "$G/scen-b43ef6a-v1.select.log" 2>&1
log "develop+select done"
