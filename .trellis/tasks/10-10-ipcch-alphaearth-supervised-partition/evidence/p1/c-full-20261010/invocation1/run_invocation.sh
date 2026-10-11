#!/usr/bin/env bash
# One released invocation of the D30 full extraction (release codex-D30-full-20261010). flock enforces one at a time.
set -u
F=/home/swl007007/ipcch-alphaearth-runs/p1-c-full-20261010
REPO="/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster/IPCCHAlphaEarthExperiment"
cd "$REPO" || exit 3
exec 9>"$F/invocation.lock"
flock -n 9 || { echo "another invocation holds the lock $(date -u +%FT%TZ)" >> "$F/invocations.exit.log"; exit 4; }
n=$(( $(wc -l < "$F/invocations.exit.log" 2>/dev/null || echo 0) + 1 ))
echo "start attempt-line $n $(date -u +%FT%TZ)" >> "$F/invocations.exit.log"
timeout 22200 /opt/earthengine/bin/python scripts/ee_pilot_c.py --request "$F/inputs/production_manifest.json" --out "$F/out" >> "$F/invocations.stdout.log" 2>&1
rc=$?
echo "exit $rc $(date -u +%FT%TZ)" >> "$F/invocations.exit.log"
/opt/earthengine/bin/python scripts/check_c_outputs.py --request "$F/inputs/production_manifest.json" --out "$F/out" --json "$F/check-$(date -u +%Y%m%dT%H%M%SZ).json" > /dev/null 2>&1
echo "checker exit $? $(date -u +%FT%TZ)" >> "$F/invocations.exit.log"
