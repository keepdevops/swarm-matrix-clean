#!/usr/bin/env bash
# matrix-1-check.sh — preflight checks before launching the stack.
# Exits non-zero on any missing prerequisite. No silent failures.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BACKEND_PORT="${BACKEND_PORT:-8765}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
# Must match matrix-2-launch.sh, or this script checks deps in one interpreter
# while the launcher starts the server in another.
PYTHON="${PYTHON:-python3}"

# ANSI without hardcoded values per global "design tokens" spirit.
C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'; C_RST=$'\033[0m'

fails=0
ok()    { printf "  %sOK%s    %s\n" "$C_GRN" "$C_RST" "$1"; }
warn()  { printf "  %sWARN%s  %s\n" "$C_YEL" "$C_RST" "$1"; }
fail()  { printf "  %sFAIL%s  %s\n" "$C_RED" "$C_RST" "$1"; fails=$((fails+1)); }

section() { printf "\n== %s ==\n" "$1"; }

# -------- 1. Toolchain --------
section "Toolchain"
probe_version() {
  case "$1" in
    lsof) lsof -v 2>&1 | awk -F: '/revision:/ {print "lsof " $2; exit}' ;;
    *)    "$1" --version 2>&1 | head -1 ;;
  esac
}
for bin in "$PYTHON" node npm curl lsof; do
  if command -v "$bin" >/dev/null 2>&1; then
    ok "$bin ($(probe_version "$bin"))"
  else
    fail "missing: $bin"
  fi
done

# -------- 2. Python deps --------
section "Python deps"
"$PYTHON" - <<'PY'
import importlib, sys
required = ["fastapi", "uvicorn", "httpx", "pydantic"]
missing = []
for m in required:
    try: importlib.import_module(m)
    except ImportError: missing.append(m)
if missing:
    print(f"MISSING:{','.join(missing)}")
    sys.exit(2)
PY
status=$?
if [ $status -eq 0 ]; then
  ok "fastapi, uvicorn, httpx, pydantic importable"
else
  fail "Python deps missing — run: pip install -r requirements.txt"
fi

# -------- 3. Frontend deps --------
section "Frontend deps"
if [ -d "$ROOT/frontend/node_modules" ]; then
  ok "frontend/node_modules present"
else
  fail "frontend/node_modules missing — run: cd frontend && npm install"
fi

# -------- 4. Agent configs --------
section "Agent configs"
if [ ! -d "$ROOT/config/agents" ]; then
  fail "config/agents directory missing"
else
  for cfg in "$ROOT"/config/agents/*.json; do
    [ -e "$cfg" ] || continue
    name="$(basename "$cfg")"
    if "$PYTHON" -c "import json,sys; json.load(open(sys.argv[1]))" "$cfg" 2>/dev/null; then
      # validate referenced paths
      msg=$("$PYTHON" - "$cfg" <<'PY'
import json, sys, pathlib
cfg = json.load(open(sys.argv[1]))
problems = []
for key in ("model_path", "llama_server_path", "vllm_python"):
    v = cfg.get(key)
    if v and not pathlib.Path(v).exists():
        problems.append(f"{key}={v}")
print(";".join(problems))
PY
)
      if [ -z "$msg" ]; then
        ok "$name"
      else
        warn "$name — referenced paths missing: $msg"
      fi
    else
      fail "$name — invalid JSON"
    fi
  done
fi

# -------- 5. Ports --------
section "Ports"
for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  if lsof -iTCP:"$port" -sTCP:LISTEN -P -n 2>/dev/null | grep -q LISTEN; then
    pid=$(lsof -iTCP:"$port" -sTCP:LISTEN -P -n -t 2>/dev/null | head -1)
    warn "port $port already in use (pid $pid) — matrix-3-shutdown.sh first"
  else
    ok "port $port free"
  fi
done

# -------- 6. Summary --------
section "Summary"
if [ $fails -gt 0 ]; then
  printf "%s%d check(s) failed.%s Fix the items above and rerun.\n" "$C_RED" "$fails" "$C_RST"
  exit 1
fi
printf "%sAll checks passed.%s Run scripts/matrix-2-launch.sh next.\n" "$C_GRN" "$C_RST"
