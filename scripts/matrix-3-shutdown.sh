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

# Engine subprocesses (llama-server, vLLM) are spawned as direct children of the
# backend, so snapshot them while the parent is still alive — once it exits there
# is nothing left to walk. Do NOT try to find them by binary path afterwards:
# argv[0] is whatever `llama_server_path` happened to be, which is a symlink on
# one machine and the resolved path on another.
ENGINE_PIDS=""
if [ -f "$BACKEND_PID" ]; then
  bpid="$(cat "$BACKEND_PID" 2>/dev/null || true)"
  if [ -n "$bpid" ] && kill -0 "$bpid" 2>/dev/null; then
    ENGINE_PIDS="$(pgrep -P "$bpid" 2>/dev/null || true)"
    [ -n "$ENGINE_PIDS" ] && say "engines to reap: $(echo $ENGINE_PIDS | tr '\n' ' ')"
  fi
fi

stop_pidfile "frontend" "$FRONTEND_PID"
stop_pidfile "backend"  "$BACKEND_PID"

# The backend's FastAPI shutdown hook releases engines on its way out. Verify
# that rather than trust it: anything still alive here is ours and is now an
# orphan. These are known pids, so no pattern matching is involved.
for pid in $ENGINE_PIDS; do
  kill -0 "$pid" 2>/dev/null || continue
  warn "engine pid $pid outlived the backend; terminating"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.25
  done
  if kill -0 "$pid" 2>/dev/null; then
    warn "engine pid $pid did not exit; SIGKILL"
    kill -9 "$pid" 2>/dev/null || true
  fi
done

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

# Engines orphaned by an *earlier* run — the backend was SIGKILLed, crashed, or
# its pidfile is gone — have no parent to walk, so they have to be discovered.
# Use the one invariant that holds: BackendConfig.port is schema-bounded to
# 9000-9090 (backends/base.py), so every engine we ever spawn listens in that
# range. 9000/9001 are the service ports, already handled above.
#
# Killing requires BOTH conditions, because neither is sufficient alone: an
# unrelated dev server may hold 9042, and cofiswarm's llama-servers on 8083-8090
# are engines that are not ours.
ENGINE_RE='llama-server|vllm\.entrypoints\.openai\.api_server'
# One lsof for the whole range; 89 individual probes would be needlessly slow.
while read -r pid port; do
  [ -n "$pid" ] || continue
  kill -0 "$pid" 2>/dev/null || continue          # already gone, or listed twice
  cmdline="$(ps -o command= -p "$pid" 2>/dev/null || true)"
  if ! printf '%s' "$cmdline" | grep -Eq "$ENGINE_RE"; then
    warn "port $port held by pid $pid, but it is not an engine — leaving it alone"
    continue
  fi
  warn "orphaned engine from an earlier run on port $port (pid $pid); terminating"
  kill "$pid" 2>/dev/null || true
  sleep 0.5
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
done < <(lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | awk '
  NR > 1 { split($9, a, ":"); p = a[length(a)] + 0
           if (p >= 9002 && p <= 9090) print $2, p }' | sort -u)

say "${C_GRN}shutdown complete${C_RST}"
