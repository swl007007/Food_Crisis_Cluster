#!/usr/bin/env bash
# Local IPCCH MLflow server: start | stop | status   (localhost only; no autostart, no linger)
set -euo pipefail
VENV=/home/swl007007/.venvs/ipcch-mlflow
ROOT=/home/swl007007/.local/share/ipcch-mlflow
UNIT=ipcch-mlflow
HOST=127.0.0.1
PORT=5000
export MLFLOW_DISABLE_AGENT_HINT=1
case "${1:-status}" in
  start)
    mkdir -p "$ROOT/artifacts" "$ROOT/logs"
    if systemctl --user is-active --quiet "$UNIT"; then echo "already running"; exit 0; fi
    systemd-run --user --unit="$UNIT" --collect \
      --setenv=MLFLOW_DISABLE_AGENT_HINT=1 \
      -p StandardOutput=append:"$ROOT/logs/server.log" -p StandardError=append:"$ROOT/logs/server.log" \
      "$VENV/bin/mlflow" server \
        --backend-store-uri "sqlite:///$ROOT/mlflow.db" \
        --artifacts-destination "$ROOT/artifacts" --serve-artifacts \
        --host "$HOST" --port "$PORT" --workers 1
    for i in $(seq 1 60); do
      if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then echo "started: http://$HOST:$PORT"; exit 0; fi
      sleep 1
    done
    echo "server did not become healthy; see $ROOT/logs/server.log" >&2; exit 1 ;;
  stop)
    systemctl --user stop "$UNIT" 2>/dev/null || true
    echo "stopped" ;;
  status)
    if systemctl --user is-active --quiet "$UNIT"; then state=active; else state=inactive; fi
    health=$(curl -fsS "http://$HOST:$PORT/health" 2>/dev/null || echo unreachable)
    echo "unit=$UNIT state=$state health=$health url=http://$HOST:$PORT root=$ROOT" ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
