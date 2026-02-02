#!/usr/bin/env bash
set -euo pipefail
BASE_PORT=8000

for GPU in 0 1 2 3 4 5 6 7; do
  PORT=$((BASE_PORT + GPU))
  PIDFILE="/tmp/opencua_pids/${PORT}.pid"

  if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE" || true)
    if [ -n "${PID:-}" ] && ps -p "$PID" >/dev/null 2>&1; then
      echo "Stopping port ${PORT}, PID ${PID} ..."
      kill "$PID" || true
      sleep 1
      # 兜底：还在就 -9
      ps -p "$PID" >/dev/null 2>&1 && kill -9 "$PID" || true
    else
      echo "No live process for port ${PORT} (PID file stale or empty)."
    fi
    rm -f "$PIDFILE"
  else
    echo "No PID file for port ${PORT}."
  fi
done
echo "Stopped all."
