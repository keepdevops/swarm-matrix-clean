#!/usr/bin/env bash
# matrix-3-shutdown.sh — stop backend, frontend, and any spawned llama-server.
# Idempotent: missing pids and dead processes are not errors.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUNDIR="$ROOT/.matrix"
BACKEND_PID="$RUNDIR/backend.pid"
FRONTEND_PID="$RUNDIR/frontend.pid"

C_GRN=$'\033[32m'; C_YEL=$'\033[33m'; C_RST=$'\033[0m'
say()  { printf "[matrix] %s\n" "$*"; }
warn() { printf "%s[matrix] %s%s\n" "$C_YEL" "$*" "$C_RST"; }

stop_pidfile() {
  local label="$1" pidfile="$2"
  if [ ! -f "$pidfile" ]; then
    warn "$label — no pidfile ($pidfile); skipping"
    return 0
  fi
  local pid; pid="$(cat "$pidfile" 2>/dev/null || true)"
  rm -f "$pidfile"
  if [ -z "$pid" ]; then
    warn "$label — empty pidfile"
    return 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    warn "$label — pid $pid already dead"
    return 0
  fi
  say "stopping $label (pid $pid)"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || { say "  ${C_GRN}stopped${C_RST}"; return 0; }
    sleep 0.25
  done
  warn "$label did not exit cleanly; escalating to SIGKILL"
  kill -9 "$pid" 2>/dev/null || true
}

stop_pidfile "frontend" "$FRONTEND_PID"
stop_pidfile "backend"  "$BACKEND_PID"

# Sweep any orphaned llama-server processes the backend spawned.
# (FastAPI's shutdown hook should have killed them, but belt+suspenders.)
llama_pids=$(pgrep -f "/Users/Shared/llama/llama-server" 2>/dev/null || true)
if [ -n "$llama_pids" ]; then
  say "sweeping orphaned llama-server processes: $(echo $llama_pids | tr '\n' ' ')"
  for pid in $llama_pids; do
    kill "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in $llama_pids; do
    if kill -0 "$pid" 2>/dev/null; then
      warn "llama-server pid $pid still alive; SIGKILL"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
fi

# Sweep any orphaned uvicorn / vite processes on our ports as final guarantee.
for port in "${BACKEND_PORT:-8765}" "${FRONTEND_PORT:-5173}"; do
  pid=$(lsof -iTCP:"$port" -sTCP:LISTEN -P -n -t 2>/dev/null | head -1 || true)
  if [ -n "$pid" ]; then
    warn "port $port still held by pid $pid; killing"
    kill "$pid" 2>/dev/null || true
    sleep 0.5
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
  fi
done

say "${C_GRN}shutdown complete${C_RST}"
