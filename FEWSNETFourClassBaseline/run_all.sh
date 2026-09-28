#!/usr/bin/env bash
# Full four-class run: prepare -> Stage 1 -> Stage 2 -> Stage 3 (3 horizons) -> report.
set -euo pipefail
RUN=${1:?run directory}
PY=${PYTHON_EXE:-/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe}
SCHEMA=feature-schema.json
cd "$(dirname "$0")"
$PY -B scripts/prepare_fourclass.py --run-dir "$RUN"
# Keep first and last supported fold checkpoints per scope for replay (A8).
$PY -B scripts/run_stage1.py --run-dir "$RUN" --workers 3 \
    --retain fs1:2018-02,fs1:2020-10,fs2:2018-02,fs2:2020-10,fs3:2018-02,fs3:2020-10
$PY -B scripts/run_stage2.py --run-dir "$RUN"
declare -A START=([1]=2021-05 [2]=2021-09 [3]=2022-01)
declare -A H=([1]=4 [2]=8 [3]=12)
for s in 1 2 3; do
  $PY -B scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py --data "$RUN/prepared/snapshot_h${H[$s]}.parquet" \
      --schema "$SCHEMA" --consensus "$RUN/stage2/consensus.json" \
      --observations "$RUN/prepared/ledgers/observations.csv" --out-dir "$RUN/stage3/h${H[$s]}" \
      --start-month "${START[$s]}" --end-month 2024-12 --forecasting-scope $s
done
$PY -B scripts/report_fourclass.py --run-dir "$RUN"
