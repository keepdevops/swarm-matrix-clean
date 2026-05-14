# matrix-safe

Air-gapped, multi-engine inference scaffolding for local LLMs. One Python
control plane drives six interchangeable backends; swapping engines is a
config-only operation.

## Why

Local LLM stacks fragment fast: llama.cpp wants a subprocess, vLLM wants a
CUDA host with an OpenAI-compatible server, MLX wants Apple Silicon, EXL2 wants
its own quantization tooling, and Hugging Face Transformers wants to be the
universal fallback. This repo gives them all a single `InferenceBackend`
contract and a hot-swap loader so the rest of your system never knows or
cares which one is running.

## Architecture

```
                       ┌──────────────────────────────┐
                       │      run_agent.py CLI        │
                       └──────────────┬───────────────┘
                                      │
                       ┌──────────────▼───────────────┐
                       │ orchestration/manager.py     │
                       │   • LRU hot-swap loader       │
                       │   • asyncio.Lock around swaps │
                       │   • Pydantic-validated config │
                       └──────────────┬───────────────┘
                                      │
                         backends/InferenceBackend ABC
   ┌──────────────┬──────────────┬────┴────┬──────────────┬──────────────┐
   ▼              ▼              ▼         ▼              ▼              ▼
llama_cpp     llama_python    mlx       vllm         exllamav2     transformers
(subprocess)  (in-process)  (in-proc)  (subprocess)  (in-proc)     (in-proc)
```

Every backend emits OpenAI-shaped delta chunks:
`{"choices": [{"delta": {"content": "..."}, "finish_reason": null}]}`.

## Backends

| Key                 | Format        | Hardware           | Notes                                            |
| ------------------- | ------------- | ------------------ | ------------------------------------------------ |
| `llama_cpp_binary`  | GGUF          | CPU / Metal / CUDA | Spawns `llama-server` on 127.0.0.1               |
| `llama_cpp_python`  | GGUF          | CPU / Metal / CUDA | `llama-cpp-python` bindings, single process      |
| `mlx`               | MLX weights   | Apple Silicon      | Fastest path on M-series                         |
| `vllm`              | HF dir        | CUDA / ROCm        | PagedAttention, high concurrency                 |
| `exllamav2`         | EXL2 dir      | CUDA / ROCm        | Tight VRAM, great single-stream throughput       |
| `transformers`      | HF dir        | CUDA/ROCm/MPS/CPU  | Most portable; slowest reference path            |

## Install

Runtime deps only (the control plane, no engines):

```bash
pip install -r requirements.txt
```

Add the engines you actually want — each is commented in `requirements.txt`:

```bash
pip install llama-cpp-python    # backends/llama_python.py
pip install mlx-lm              # backends/mlx_engine.py (Apple Silicon)
pip install vllm                # backends/vllm_engine.py (CUDA/ROCm)
pip install exllamav2           # backends/exllamav2_engine.py (CUDA/ROCm)
pip install transformers torch  # backends/transformers_engine.py
```

`backends/llama_cpp.py` only needs a built `llama-server` binary — point at it
via the agent config's `llama_server_path`.

## Quick start

```bash
# See what's wired in
python run_agent.py --list-backends

# Run an agent config end-to-end
python run_agent.py -a config/agents/developer.json \
    --prompt "// fast inverse square root\n"

# Override a setting from the CLI
python run_agent.py -a config/agents/developer.json \
    --backend llama_cpp_python --temperature 0.0

# Pipe a prompt
cat README.md | python run_agent.py -a config/agents/portable.json
```

Switching engines is a one-field edit:

```jsonc
// config/agents/developer.json
{
  "backend_target": "llama_cpp_binary",           // ← change this
  "model_path": "/Users/Shared/models/gguf/codellama-7b.Q5_K_M.gguf",
  "llama_server_path": "/Users/Shared/llama/llama-server",
  "max_context_tokens": 8192
}
```

## Programmatic use

```python
import asyncio, logging
from orchestration import BackendManager
from orchestration.manager import load_agent_config

logging.basicConfig(level=logging.INFO)

async def main():
    mgr = BackendManager(max_resident=1)
    backend = await mgr.acquire(load_agent_config("config/agents/developer.json"))
    async for chunk in backend.generate_stream({"prompt": "// inv sqrt\n"}):
        if "error" in chunk:
            print("ERR:", chunk["error"]); break
        print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
    await mgr.release_all()

asyncio.run(main())
```

Hot-swap policy: `max_resident=1` keeps one engine live and evicts the LRU on
acquire of a different target. Bump to 2+ if you have memory headroom and want
to keep multiple engines warm for low-latency agent handoffs.

## Testing

Fast suite (42 tests, ~130 ms, no real models):

```bash
pytest
```

Real-model integration tests — opt-in, env-var gated, no auto-downloads:

```bash
MATRIX_SAFE_GGUF_PATH=/path/to/tinyllama-1.1B.Q4_K_M.gguf \
  pytest -m integration -k llama_cpp_python
```

Recognized env vars:

| Var                          | Used by                              |
| ---------------------------- | ------------------------------------ |
| `MATRIX_SAFE_GGUF_PATH`      | `llama_cpp_python`, `llama_cpp_binary` |
| `MATRIX_SAFE_LLAMA_SERVER`   | `llama_cpp_binary` (server binary)   |
| `MATRIX_SAFE_MLX_PATH`       | `mlx`                                |
| `MATRIX_SAFE_HF_PATH`        | `transformers`                       |
| `MATRIX_SAFE_EXL2_PATH`      | `exllamav2`                          |
| `MATRIX_SAFE_VLLM_PATH`      | `vllm`                               |
| `MATRIX_SAFE_VLLM_PYTHON`    | (optional) vLLM-only Python interpreter |
| `MATRIX_SAFE_NGL`            | (optional) llama.cpp GPU layers      |
| `MATRIX_SAFE_DTYPE`          | (optional) transformers dtype        |
| `MATRIX_SAFE_VLLM_MEM`       | (optional) vLLM GPU memory util      |
| `MATRIX_SAFE_INT_PORT`       | (optional) starting port for servers |

Tests skip loudly with the missing prerequisite, so it's obvious which ran.

## CI

`.github/workflows/ci.yml` runs on every push/PR plus manual dispatch:

- **`test` matrix**: `{ubuntu, macos} × py{3.10, 3.11, 3.12}` — six cells.
- **Gates**: syntax check on every `.py`, **300-LOC enforcement** per the
  modular constraint, full fast test suite, CLI smoke test.
- **`lint` job**: `ruff check` + `ruff format --check`.
- **Concurrency**: in-progress runs cancel when a newer push lands.

Integration tests are auto-deselected in CI (no env vars set).

## Layout

```
matrix-safe/
├── run_agent.py                # Unified CLI loader
├── pyproject.toml              # ruff config
├── pytest.ini                  # asyncio_mode=auto, -m "not integration"
├── requirements.txt            # core deps + commented optional engines
├── backends/
│   ├── base.py                 # InferenceBackend ABC + Pydantic BackendConfig
│   ├── _sse.py                 # llama-server SSE parser
│   ├── llama_cpp.py            # subprocess llama-server
│   ├── llama_python.py         # in-process llama-cpp-python
│   ├── mlx_engine.py           # Apple MLX
│   ├── vllm_engine.py          # subprocess vLLM (OpenAI server)
│   ├── exllamav2_engine.py     # in-process EXL2
│   └── transformers_engine.py  # in-process HF transformers
├── orchestration/
│   └── manager.py              # LRU hot-swap BackendManager
├── config/agents/
│   ├── developer.json          # llama_cpp_binary
│   ├── reviewer.json           # mlx
│   ├── scanner.json            # vllm
│   ├── refactor.json           # exllamav2
│   └── portable.json           # transformers
├── tests/
│   ├── conftest.py             # FakeBackend, fake_registry fixture
│   ├── test_base_config.py     # Pydantic validation
│   ├── test_sse.py             # SSE parser + vLLM translator
│   ├── test_manager.py         # acquire / LRU / cancel
│   ├── test_run_agent_cli.py   # CLI smoke
│   └── test_integration_real_models.py  # opt-in real-model tier
└── .github/workflows/ci.yml
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the CI gates your PR has
to clear, and the checklist for adding a new backend.

## License

Apache License 2.0 — see [LICENSE](LICENSE). Copyright 2026 keepdevops.

## Design constraints

These are non-negotiable across every backend:

- **No silent failures.** Every `except` logs with `exc_info=True`. Health
  probes, pipe drains, and worker threads all surface their errors.
- **Schema-validated inputs.** Pydantic `BackendConfig` checks paths exist,
  port ranges, dtype names, and numeric bounds before any engine starts.
- **Cancellation is honored.** `asyncio.CancelledError` unwinds cleanly:
  subprocess backends close upstream streams; in-process backends signal a
  thread `stop_flag` and `join` with a bounded timeout.
- **Air-gap safe.** Servers bind 127.0.0.1, no network in tests, no
  auto-downloads. Egress must be enforced at the OS/firewall layer.
- **250–300 LOC per file.** CI enforces this; if a module grows past 300 it
  gets split, not bloated.
