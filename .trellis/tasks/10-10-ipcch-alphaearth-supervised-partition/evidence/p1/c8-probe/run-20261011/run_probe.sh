#!/usr/bin/env bash
# Released c8 probe: ONE invocation, only after precheck passes, holding the full run's invocation.lock throughout.
set -u
R=/home/swl007007/ipcch-alphaearth-runs/p1-c8-probe-20261011
F=/home/swl007007/ipcch-alphaearth-runs/p1-c-full-20261010
REPO="/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster/IPCCHAlphaEarthExperiment"
exec 9>"$F/invocation.lock"
flock -n 9 || { echo "full invocation.lock busy $(date -u +%FT%TZ)" >> "$R/probe.exit.log"; exit 4; }
[ -e "$R/probe.started" ] && { echo "probe already started once; refusing" >> "$R/probe.exit.log"; exit 5; }
python3 "$R/precheck.py" > "$R/precheck.json" 2>&1 || { echo "precheck FAILED $(date -u +%FT%TZ)" >> "$R/probe.exit.log"; exit 6; }
date -u +%FT%TZ > "$R/probe.started"
cd "$REPO" || exit 3
timeout 900 /opt/earthengine/bin/python scripts/ee_pilot_c.py --request "$R/inputs/C8_probe_manifest.json" --out "$R/out" > "$R/probe.stdout.log" 2>&1
echo "exit $? $(date -u +%FT%TZ)" >> "$R/probe.exit.log"
/opt/earthengine/bin/python scripts/check_c_outputs.py --request "$R/inputs/C8_probe_manifest.json" --out "$R/out" --json "$R/check.json" > /dev/null 2>&1
echo "checker exit $? $(date -u +%FT%TZ)" >> "$R/probe.exit.log"
