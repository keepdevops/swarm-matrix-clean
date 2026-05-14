#!/usr/bin/env bash
# matrix-2-launch.sh — start backend + frontend in background, wait for ready.
# Writes pids + logs under .matrix/. Idempotent: refuses to start if PID is alive.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BACKEND_PORT="${BACKEND_PORT:-8765}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
RUNDIR="$ROOT/.matrix"
mkdir -p "$RUNDIR"

BACKEND_PID="$RUNDIR/backend.pid"
FRONTEND_PID="$RUNDIR/frontend.pid"
BACKEND_LOG="$RUNDIR/backend.log"
FRONTEND_LOG="$RUNDIR/frontend.log"

C_GRN=$'\033[32m'; C_RED=$'\033[31m'; C_RST=$'\033[0m'
say()  { printf "[matrix] %s\n" "$*"; }
die()  { printf "%s[matrix] %s%s\n" "$C_RED" "$*" "$C_RST" >&2; exit 1; }

# Refuse to start if a recorded pid is still alive.
is_alive() {
  local pidfile="$1"
  [ -f "$pidfile" ] || return 1
  local pid; pid="$(cat "$pidfile" 2>/dev/null || true)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

if is_alive "$BACKEND_PID"; then
  die "backend already running (pid $(cat $BACKEND_PID)). Run matrix-3-shutdown.sh first."
fi
if is_alive "$FRONTEND_PID"; then
  die "frontend already running (pid $(cat $FRONTEND_PID)). Run matrix-3-shutdown.sh first."
fi

# Refuse if ports are taken by anything else.
for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  if lsof -iTCP:"$port" -sTCP:LISTEN -P -n 2>/dev/null | grep -q LISTEN; then
    die "port $port is in use by another process — free it and retry."
  fi
done

# ---- Start backend ----
say "starting FastAPI server on :$BACKEND_PORT (log: $BACKEND_LOG)"
PORT="$BACKEND_PORT" LOG_LEVEL="${LOG_LEVEL:-INFO}" \
  nohup python3 -m server >"$BACKEND_LOG" 2>&1 &
echo "$!" > "$BACKEND_PID"

# Wait for /api/health.
say "waiting for backend health…"
for i in $(seq 1 40); do
  if curl -sf "http://127.0.0.1:$BACKEND_PORT/api/health" >/dev/null 2>&1; then
    say "${C_GRN}backend ready${C_RST} (pid $(cat $BACKEND_PID))"
    break
  fi
  if ! kill -0 "$(cat $BACKEND_PID)" 2>/dev/null; then
    tail -20 "$BACKEND_LOG" >&2
    die "backend exited during startup — see $BACKEND_LOG"
  fi
  sleep 0.5
done
if ! curl -sf "http://127.0.0.1:$BACKEND_PORT/api/health" >/dev/null; then
  tail -20 "$BACKEND_LOG" >&2
  die "backend never responded — see $BACKEND_LOG"
fi

# ---- Start frontend ----
say "starting Vite dev server on :$FRONTEND_PORT (log: $FRONTEND_LOG)"
( cd "$ROOT/frontend" && nohup npm run dev -- --port "$FRONTEND_PORT" \
    >"$FRONTEND_LOG" 2>&1 & echo "$!" > "$FRONTEND_PID" )

# Wait for Vite to bind the port.
say "waiting for frontend…"
for i in $(seq 1 40); do
  if curl -sf "http://127.0.0.1:$FRONTEND_PORT/" >/dev/null 2>&1; then
    say "${C_GRN}frontend ready${C_RST} (pid $(cat $FRONTEND_PID))"
    break
  fi
  if ! kill -0 "$(cat $FRONTEND_PID)" 2>/dev/null; then
    tail -20 "$FRONTEND_LOG" >&2
    die "frontend exited during startup — see $FRONTEND_LOG"
  fi
  sleep 0.5
done

echo
say "stack is up:"
printf "    frontend  →  http://localhost:%s/\n" "$FRONTEND_PORT"
printf "    backend   →  http://127.0.0.1:%s/api/health\n" "$BACKEND_PORT"
printf "    logs      →  %s, %s\n" "$BACKEND_LOG" "$FRONTEND_LOG"
echo
say "stop with: scripts/matrix-3-shutdown.sh"
