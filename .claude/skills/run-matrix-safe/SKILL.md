---
name: run-matrix-safe
description: Build, launch, drive, screenshot, and smoke-test the matrix-safe stack (FastAPI control plane on :9000 + Vite/React editor UI on :9001) and the run_agent.py CLI. Use when asked to run, start, restart, stop, debug, screenshot, or verify matrix-safe, its server, its frontend, or a local LLM agent config.
---

# Running matrix-safe

Python control plane (FastAPI, `:9000`) that hot-swaps six local inference
backends, plus a React/CodeMirror editor UI (Vite dev server, `:9001`) that
proxies `/api` to the backend. Three shell scripts own the lifecycle;
`driver.mjs` is how an agent pokes the running stack.

**All paths below are relative to the repo root (`/Users/caribou/matrix-safe`).**
Verified on macOS 25.6 (Apple Silicon), Node 26, Python 3.11.

## Prerequisites

The interpreter split is the single biggest trap here — **two different Python
environments, each missing what the other has**:

| Purpose | Interpreter | Has |
|---|---|---|
| **Running the app** | `python3` → `~/miniforge3/envs/mlx-env/bin/python3` | fastapi, uvicorn, **mlx_lm** |
| **Running the tests** | `.venv/bin/python3` (3.14) | fastapi, uvicorn, **pytest**, ruff |

`.venv/` has no `pytest` in `mlx-env`, and no `mlx_lm` in `.venv`. Do not
activate `.venv` and then launch — the stack starts fine and every `mlx` agent
fails at generate time. Launch with the default `python3`.

Models live under `MATRIX_SAFE_MODELS_DIR` (default `/Users/Shared/llama/models`);
`llama_cpp_binary` also needs the `llama-server` binary at
`/Users/Shared/llama/llama-server` (a symlink into `llama.cpp-master/build/bin/`).
Both are already present on this machine — nothing to install.

Frontend deps are installed (`frontend/node_modules`). If missing:
`cd frontend && npm install`.

## Run (agent path)

```bash
./scripts/matrix-1-check.sh      # preflight: toolchain, deps, configs, ports free
./scripts/matrix-2-launch.sh     # backgrounds both servers, waits for health
node .claude/skills/run-matrix-safe/driver.mjs smoke
./scripts/matrix-3-shutdown.sh   # stop both + sweep spawned llama-server
```

`matrix-2-launch.sh` writes pids and logs to `.matrix/` and **refuses to start
if a recorded pid is still alive** — always shut down before relaunching.

### driver.mjs

Zero-dependency Node 18+ script. Exits non-zero on any failure, and treats an
SSE `error` event or a zero-token stream as a hard failure (the server returns
those on a **200**, so a naive `curl` looks like it succeeded).

```bash
node .claude/skills/run-matrix-safe/driver.mjs health          # probe :9000 + :9001
node .claude/skills/run-matrix-safe/driver.mjs backends        # registered engine keys
node .claude/skills/run-matrix-safe/driver.mjs agents          # agent configs + model paths
node .claude/skills/run-matrix-safe/driver.mjs generate local-mlx.json --max-tokens 32
node .claude/skills/run-matrix-safe/driver.mjs chat local_mlx_agent
node .claude/skills/run-matrix-safe/driver.mjs smoke           # all of the above
```

`generate` takes an agent **filename**; `chat` (the OpenAI-compatible route)
takes an **agent_id**. Flags: `--prompt --max-tokens --temperature --backend`.
Env: `MATRIX_BACKEND_URL MATRIX_FRONTEND_URL MATRIX_TIMEOUT_MS SMOKE_AGENT
SMOKE_MODEL MATRIX_DEBUG`.

Measured cold acquire times: `mlx` ≈ 3s, `llama_cpp_binary` ≈ 8.6s (spawns
`llama-server` and loads a 4.7 GB GGUF). Warm re-acquire is ~3 ms. The driver's
default timeout is 300 s for that reason.

**Use `local-mlx.json` / `local_mlx_agent` for any smoke check** — it is by far
the fastest and it is the default.

### Driving the UI

Use the Playwright MCP browser tools. The flow that works:

1. `browser_tabs` with `action: "new"`, `url: "http://localhost:9001/"` —
   **always open a dedicated tab** (see Gotchas).
2. `browser_fill_form` — select `Agent` = `Local MLX (Llama-3.2-1B, fast) (mlx)`
   and set `Max tok` = `64`.
3. `browser_click` the `Run ⌘↩` button.
4. `browser_wait_for` text `done (` — **not** `done (stop)`; with a low
   `Max tok` the status reads `done (length)`.
5. `browser_take_screenshot` with a filename under `.playwright-mcp/`.

Screenshots can only be written inside the repo — the MCP server rejects paths
outside its allowed roots with `File access denied`.

## Direct invocation (no server)

Most backend changes need only this — no stack, no browser:

```bash
python3 run_agent.py --list-backends
python3 run_agent.py -a config/agents/local-mlx.json --prompt "// mutex"
python3 run_agent.py -a config/agents/local-mlx.json --backend mlx --temperature 0.0
cat README.md | python3 run_agent.py -a config/agents/local-mlx.json
```

## Run (human path)

Two terminals: `PORT=9000 python3 -m server`, then `cd frontend && npm run dev`.
Open `http://localhost:9001/`. The launch script does this better; use it.

## Test

```bash
.venv/bin/python3 -m pytest        # 61 passed, 7 deselected, ~0.1s
.venv/bin/python3 -m ruff check .  # All checks passed!
```

The 7 deselected are the opt-in real-model integration tier (`-m integration`),
gated on `MATRIX_SAFE_GGUF_PATH` etc. — see the README table.

## Gotchas

- **`python3` is not `.venv/bin/python3`.** See Prerequisites. `matrix-1-check.sh`
  only probes fastapi/uvicorn/httpx/pydantic, so it passes green under `.venv`
  and the failure surfaces much later as
  `acquire failed: backend 'mlx' failed to initialize`.
- **Only 2 of the 7 agent configs actually run in this environment.** `mlx` and
  `llama_cpp_binary` work. Anything targeting `llama_cpp_python`
  (`docker-gemma2b.json`, `docker-coder-14b.json`) fails with
  `acquire failed: backend 'llama_cpp_python' failed to initialize` — the
  `llama-cpp-python` package is installed in neither interpreter. `vllm` /
  `exllamav2` need CUDA and cannot run here at all.
- **Engine cleanup keys off the port range, never the binary path.**
  `BackendConfig.port` is schema-bounded to **9000–9090** (`backends/base.py:27`),
  so every engine subprocess matrix-safe spawns — `llama-server` *and* vLLM —
  listens in that range. `matrix-3-shutdown.sh` relies on that: it snapshots the
  backend's children before killing it, then sweeps 9002–9090 for orphans left by
  a previous crash, killing only processes that are *both* in range and actually
  an engine. Do not reintroduce path matching: `argv[0]` is whatever
  `llama_server_path` was set to (a symlink on one machine, the resolved
  `llama.cpp-master/build/bin/` path on another), which is exactly why the
  original sweep matched nothing. The two-condition guard also keeps the sweep off
  the unrelated **cofiswarm** stack's five `llama-server` processes on 8083–8090
  and off any non-engine process that happens to hold a port in range.
- **Vite binds IPv6.** Probe the frontend as `http://localhost:9001/`, never
  `http://127.0.0.1:9001/` — the IPv4 probe never connects even while the dev
  server serves fine. Both `matrix-1-check.sh` and `driver.mjs` already do this.
- **Errors arrive on a 200.** `/api/generate` streams `event: error` inside a
  successful SSE response. `curl -f` will not catch it; use the driver.
- **The Playwright MCP browser is a shared, live Chrome profile.** Its "current
  tab" drifts to whatever the human is browsing — repeatedly a 1×1 ad-tracking
  pixel, which yields `Cannot take screenshot with 0 width` and
  `TypeError: Cannot read properties of undefined (reading 'height')` from
  element-collection scripts. That error is **not** a bug in this app. Always
  open your own tab and re-select it before acting.
- **`timeout(1)` does not exist** on this macOS box (no coreutils). Use the
  driver's `MATRIX_TIMEOUT_MS` or `curl -m`.
- `favicon.ico` 404s in the browser console on every page load. Harmless.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `port 9000 is in use by another process` | `./scripts/matrix-3-shutdown.sh`, then relaunch. |
| `backend already running (pid N)` | Same — the launcher is deliberately non-idempotent. |
| `acquire failed: backend 'mlx' failed to initialize` | Launched under `.venv`. Relaunch with the default `python3`. |
| `acquire failed: backend 'llama_cpp_python' failed to initialize` | Expected; that engine isn't installed. Use an `mlx` or `llama_cpp_binary` agent. |
| `agent config not found: X` (HTTP 404) | `generate` wants a filename (`local-mlx.json`), not an agent_id. |
| `unknown model: X` (HTTP 404) | `chat` wants an agent_id (`local_mlx_agent`), not a filename. |
| `No module named pytest` | You used `python3`. Tests run under `.venv/bin/python3`. |
| Generate hangs >30 s | Normal cold GGUF load. Watch `.matrix/backend.log`. |
| Frontend up but agent dropdown empty | Backend down or the `/api` proxy is broken — run `driver.mjs health`. |
