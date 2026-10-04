#!/usr/bin/env bash
# Approved single Stage 1 resume pass (coordinator 2026-10-03), FAIL-CLOSED: every preservation,
# snapshot, reconcile, cd or env-check failure aborts BEFORE any fit. Sole resume owner.
# Waits for the original batch, preserves + verifies failed temp trees, snapshots attempt-1
# evidence (ledger and log byte-checked), reconciles IDs (informational), runs ONE identical resume
# pass (diagnostic env only), then preserves/verifies resume failures in their OWN destination and
# reconciles again. Acceptance is done separately (accept_scenario_stage1).
# Sourcing the file defines the helpers only (synthetic helper checks); paths are overridable.
set -euo pipefail
G=${RECOVERY_G:-/mnt/c/Users/swl00/geoxgb_runs}
TMP=${RECOVERY_TMP:-/mnt/c/Users/swl00/AppData/Local/Temp/geoxgb_stage1/scen-b43ef6a-v1}
RUN_W='C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1'
RUN=$G/scen-b43ef6a-v1
PKG="/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster/FEWSNETGeoXGBExperiment"
PY=/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe
REC=$G/scen-b43ef6a-v1.recovery.log
EARLIER=$G/scen-b43ef6a-v1.failed_stage1
abort() { echo "$(date -Is) ABORT (no fit started): $*" >> "$REC"; exit 2; }
failed_roots() {  # every ledger line must be valid JSON; prints failed roots (possibly none)
  python3 - "$1" <<'PY'
import json, sys
for n, line in enumerate(open(sys.argv[1], encoding="utf-8"), 1):
    if not line.strip():
        continue
    rec = json.loads(line)            # malformed line -> nonzero exit
    if rec["status"] == "failed":
        print(rec["root"])
PY
}
preserve() {  # $1 destination dir, $2 ledger file, $3 "reuse-earlier" only for attempt 1
  local roots r dest
  roots=$(failed_roots "$2") || abort "cannot parse ledger $2"
  mkdir -p "$1" || abort "mkdir $1"
  for r in $roots; do
    dest="$1/$r"
    if [ "${3:-}" = "reuse-earlier" ] && [ -d "$EARLIER/$r" ]; then dest="$EARLIER/$r"; fi
    if [ -d "$TMP/$r" ]; then
      [ -d "$dest" ] || cp -a "$TMP/$r" "$dest" || abort "copy $r"
      diff -rq "$TMP/$r" "$dest" > /dev/null || abort "backup of $r differs from its temp tree"
    else
      [ -d "$dest" ] || abort "failed root $r has neither a temp tree nor a backup"
    fi
    echo "preserved $r -> $dest" >> "$REC"
  done
}
reconcile() {  # informational counts only
  python3 - "$RUN" "$1" >> "$REC" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
run, label = Path(sys.argv[1]), sys.argv[2]
roots = {r["root"] for r in json.loads((run / "prepared/manifests/schedule.json").read_text())["stage1_scenario_roots"]}
done = {p.parent.name: json.loads(p.read_text())["status"] for p in (run / "stage1_scenario/roots").glob("*/completion.json")}
print(json.dumps({"label": label, "scheduled": len(roots), "unique_completed_roots": len(set(done) & roots),
                  "unexpected": sorted(set(done) - roots), "completion_status": dict(Counter(done.values())),
                  "missing": sorted(roots - set(done))}))
PY
}
main() {
  local BATCH_PID=$1 rc=0 n1 ENV_OK
  while kill -0 "$BATCH_PID" 2>/dev/null; do sleep 60; done
  echo "$(date -Is) batch $BATCH_PID exited" >> "$REC"
  preserve "$G/scen-b43ef6a-v1.failed_stage1_attempt1" "$RUN/stage1_scenario/ledger.jsonl" reuse-earlier
  cp -a "$RUN/stage1_scenario/ledger.jsonl" "$G/scen-b43ef6a-v1.stage1.attempt1.ledger.jsonl" || abort "ledger snapshot"
  cp -a "$G/scen-b43ef6a-v1.stage1.log" "$G/scen-b43ef6a-v1.stage1.attempt1.log" || abort "log snapshot"
  cmp -s "$RUN/stage1_scenario/ledger.jsonl" "$G/scen-b43ef6a-v1.stage1.attempt1.ledger.jsonl" || abort "ledger snapshot differs"
  cmp -s "$G/scen-b43ef6a-v1.stage1.log" "$G/scen-b43ef6a-v1.stage1.attempt1.log" || abort "log snapshot differs"
  reconcile after_attempt1 || abort "reconcile after attempt1"
  cd "$PKG" || abort "cd package"
  export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 WSLENV="${WSLENV:+$WSLENV:}PYTHONFAULTHANDLER/w:PYTHONUNBUFFERED/w"
  ENV_OK=$("$PY" -c "import os,faulthandler;print(os.environ.get('PYTHONFAULTHANDLER')=='1' and os.environ.get('PYTHONUNBUFFERED')=='1' and faulthandler.is_enabled())" | tr -d '\r') \
    || abort "env check failed to run"
  [ "$ENV_OK" = "True" ] || abort "diagnostic env not visible in the Windows interpreter ($ENV_OK)"
  local CMD="$PY -B scripts/run_stage1.py --run-dir '$RUN_W' --split-mode scen --workers 1"
  echo "$(date -Is) RESUME CMD (env PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1, WSLENV appended): $CMD" >> "$G/scen-b43ef6a-v1.commands.log"
  bash -c "$CMD" > "$G/scen-b43ef6a-v1.stage1.resume1.log" 2>&1 || rc=$?
  echo "$(date -Is) resume exit=$rc (nonzero expected if any root fails again)" >> "$REC"
  n1=$(wc -l < "$G/scen-b43ef6a-v1.stage1.attempt1.ledger.jsonl")
  tail -n +$((n1 + 1)) "$RUN/stage1_scenario/ledger.jsonl" > "$G/scen-b43ef6a-v1.stage1.resume1.ledger.jsonl" \
    || abort "resume ledger slice"
  preserve "$G/scen-b43ef6a-v1.failed_stage1_resume1" "$G/scen-b43ef6a-v1.stage1.resume1.ledger.jsonl"
  reconcile after_resume1 || abort "reconcile after resume1"
  echo "$(date -Is) recovery done (resume exit=$rc); acceptance via accept_scenario_stage1 before development" >> "$REC"
}
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
