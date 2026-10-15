#!/usr/bin/env bash
# P2 develop, ONCE (receipt codex-D30-formal-P2-development-mapfreeze-20261014).
cd "/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster/IPCCHAlphaEarthExperiment" || exit 3
echo "develop start $(date -u +%FT%TZ)" >> /home/swl007007/ipcch-alphaearth-runs/p1-formal-d30-logs/stages.log
/home/swl007007/.venvs/ipcch-geo/bin/python -m ipcch_alphaearth develop --run /home/swl007007/ipcch-alphaearth-runs/p1-formal-d30-20261014 > /home/swl007007/ipcch-alphaearth-runs/p1-formal-d30-logs/develop.out 2> /home/swl007007/ipcch-alphaearth-runs/p1-formal-d30-logs/develop.err
rc=$?
echo "develop exit $rc $(date -u +%FT%TZ)" >> /home/swl007007/ipcch-alphaearth-runs/p1-formal-d30-logs/stages.log
