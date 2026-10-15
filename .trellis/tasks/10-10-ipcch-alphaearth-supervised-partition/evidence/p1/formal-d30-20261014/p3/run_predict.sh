#!/usr/bin/env bash
# P3 predict, ONCE (receipt codex-D30-formal-P3-predict-report-replay-20261014).
cd "/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster/IPCCHAlphaEarthExperiment" || exit 3
echo "predict start $(date -u +%FT%TZ)" >> /home/swl007007/ipcch-alphaearth-runs/p1-formal-d30-logs/stages.log
/home/swl007007/.venvs/ipcch-geo/bin/python -m ipcch_alphaearth predict --run /home/swl007007/ipcch-alphaearth-runs/p1-formal-d30-20261014 > /home/swl007007/ipcch-alphaearth-runs/p1-formal-d30-logs/predict.out 2> /home/swl007007/ipcch-alphaearth-runs/p1-formal-d30-logs/predict.err
rc=$?
echo "predict exit $rc $(date -u +%FT%TZ)" >> /home/swl007007/ipcch-alphaearth-runs/p1-formal-d30-logs/stages.log
