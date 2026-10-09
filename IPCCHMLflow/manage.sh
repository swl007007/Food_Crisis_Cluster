#!/usr/bin/env bash
# Local IPCCH MLflow server: start | stop | status   (localhost only; no autostart, no linger)
# Uses a transient systemd user unit when the user bus is reachable; otherwise (WSL login not
# registered with logind -> "Failed to connect to bus") falls back to a setsid process + pid file.
set -euo pipefail
VENV=/home/swl007007/.venvs/ipcch-mlflow
ROOT=/home/swl007007/.local/share/ipcch-mlflow
UNIT=ipcch-mlflow
HOST=127.0.0.1
PORT=5000
PIDFILE="$ROOT/server.pid"
export MLFLOW_DISABLE_AGENT_HINT=1
SERVER_ARGS=(server
  --backend-store-uri "sqlite:///$ROOT/mlflow.db"
  --artifacts-destination "$ROOT/artifacts" --serve-artifacts
  --host "$HOST" --port "$PORT" --workers 1)

have_user_bus() { systemctl --user show-environment >/dev/null 2>&1; }
unit_active() { have_user_bus && systemctl --user is-active --quiet "$UNIT"; }
pid_alive() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
healthy() { curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; }

case "${1:-status}" in
  start)
    mkdir -p "$ROOT/artifacts" "$ROOT/logs"
    if unit_active || pid_alive; then echo "already running"; exit 0; fi
    if healthy; then echo "port $PORT already serves /health (unmanaged process?)" >&2; exit 1; fi
    if have_user_bus; then
      systemd-run --user --unit="$UNIT" --collect \
        --setenv=MLFLOW_DISABLE_AGENT_HINT=1 \
        -p StandardOutput=append:"$ROOT/logs/server.log" -p StandardError=append:"$ROOT/logs/server.log" \
        "$VENV/bin/mlflow" "${SERVER_ARGS[@]}"
    else
      echo "no systemd user bus; starting as background process (pid file $PIDFILE)"
      setsid nohup "$VENV/bin/mlflow" "${SERVER_ARGS[@]}" >>"$ROOT/logs/server.log" 2>&1 < /dev/null &
      echo $! > "$PIDFILE"
    fi
    for i in $(seq 1 60); do
      if healthy; then echo "started: http://$HOST:$PORT"; exit 0; fi
      sleep 1
    done
    echo "server did not become healthy; see $ROOT/logs/server.log" >&2; exit 1 ;;
  stop)
    if have_user_bus; then systemctl --user stop "$UNIT" 2>/dev/null || true; fi
    if pid_alive; then
      pid=$(cat "$PIDFILE")
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      for i in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
    echo "stopped" ;;
  status)
    if unit_active; then state="active(systemd)"
    elif pid_alive; then state="active(pid $(cat "$PIDFILE"))"
    else state=inactive; fi
    health=$(curl -fsS "http://$HOST:$PORT/health" 2>/dev/null || echo unreachable)
    echo "unit=$UNIT state=$state health=$health url=http://$HOST:$PORT root=$ROOT" ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
