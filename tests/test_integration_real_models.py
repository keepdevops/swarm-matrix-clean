"""Real-model integration tests.

These are OPT-IN — they only run when:
  1. invoked with `pytest -m integration`, AND
  2. the relevant env var points at a real local model, AND
  3. the backend's Python deps are importable.

Each test skips loudly with the missing prerequisite so you can tell *why*
nothing ran. No network. No auto-downloads. Air-gap safe.

Env vars (set whichever backends you actually have weights for):
  MATRIX_SAFE_GGUF_PATH       absolute path to a .gguf file
  MATRIX_SAFE_LLAMA_SERVER    absolute path to a built llama-server binary
  MATRIX_SAFE_MLX_PATH        directory containing MLX weights
  MATRIX_SAFE_HF_PATH         directory containing HF weights (config + safetensors)
  MATRIX_SAFE_EXL2_PATH       directory containing EXL2 weights
  MATRIX_SAFE_VLLM_PATH       directory containing HF weights for vLLM
  MATRIX_SAFE_VLLM_PYTHON     python interpreter that has vllm installed (optional)
  MATRIX_SAFE_INT_PORT        starting port for spawned servers (default 18080)

Each test caps generation at a few tokens so a slow CPU box still finishes
in seconds, not minutes.
"""

from __future__ import annotations

import asyncio
import importlib
import os
from typing import Any

import pytest

from orchestration.manager import BackendManager

pytestmark = pytest.mark.integration

PROMPT = "The capital of France is"
MAX_TOKENS = 16
_PORT_COUNTER = int(os.environ.get("MATRIX_SAFE_INT_PORT", "18080"))


def _next_port() -> int:
    global _PORT_COUNTER
    _PORT_COUNTER += 1
    return _PORT_COUNTER


def _require_module(modname: str) -> None:
    try:
        importlib.import_module(modname)
    except ImportError as exc:
        pytest.skip(f"{modname} not installed: {exc}")


def _require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        pytest.skip(f"env var {var} not set")
    return val


async def _drain_stream(backend, payload: dict[str, Any]) -> list[str]:
    """Collect content chunks. Asserts at least one non-error chunk arrives."""
    pieces: list[str] = []
    finish_seen = False
    async for chunk in backend.generate_stream(payload):
        if "error" in chunk:
            pytest.fail(f"backend yielded error: {chunk['error']}")
        choice = chunk["choices"][0]
        text = (choice.get("delta") or {}).get("content", "")
        if text:
            pieces.append(text)
        if choice.get("finish_reason"):
            finish_seen = True
            break
    assert pieces, "no content chunks produced"
    assert finish_seen, "stream ended without finish_reason"
    return pieces


async def _run_one(target: str, config_extras: dict[str, Any]) -> str:
    mgr = BackendManager(max_resident=1)
    config = {
        "backend_target": target,
        "max_context_tokens": 2048,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        **config_extras,
    }
    try:
        backend = await mgr.acquire(config)
        pieces = await _drain_stream(backend, {"prompt": PROMPT})
        return "".join(pieces)
    finally:
        await mgr.release_all()


# ----------------------------- llama_cpp_python ------------------------------


@pytest.mark.asyncio
async def test_llama_cpp_python_generates():
    _require_module("llama_cpp")
    model_path = _require_env("MATRIX_SAFE_GGUF_PATH")
    text = await _run_one("llama_cpp_python", {"model_path": model_path})
    assert text.strip(), f"empty output: {text!r}"


# ----------------------------- llama_cpp_binary ------------------------------


@pytest.mark.asyncio
async def test_llama_cpp_binary_generates():
    model_path = _require_env("MATRIX_SAFE_GGUF_PATH")
    server_path = _require_env("MATRIX_SAFE_LLAMA_SERVER")
    text = await _run_one(
        "llama_cpp_binary",
        {
            "model_path": model_path,
            "llama_server_path": server_path,
            "port": _next_port(),
            "startup_timeout_s": 60.0,
            "n_gpu_layers": int(os.environ.get("MATRIX_SAFE_NGL", "0")),
        },
    )
    assert text.strip()


# ----------------------------- mlx -------------------------------------------


@pytest.mark.asyncio
async def test_mlx_generates():
    _require_module("mlx_lm")
    model_dir = _require_env("MATRIX_SAFE_MLX_PATH")
    text = await _run_one("mlx", {"model_path": model_dir})
    assert text.strip()


# ----------------------------- transformers ----------------------------------


@pytest.mark.asyncio
async def test_transformers_generates():
    _require_module("transformers")
    _require_module("torch")
    model_dir = _require_env("MATRIX_SAFE_HF_PATH")
    text = await _run_one(
        "transformers",
        {"model_path": model_dir, "dtype": os.environ.get("MATRIX_SAFE_DTYPE", "auto")},
    )
    assert text.strip()


# ----------------------------- exllamav2 -------------------------------------


@pytest.mark.asyncio
async def test_exllamav2_generates():
    _require_module("exllamav2")
    model_dir = _require_env("MATRIX_SAFE_EXL2_PATH")
    text = await _run_one("exllamav2", {"model_path": model_dir})
    assert text.strip()


# ----------------------------- vllm ------------------------------------------


@pytest.mark.asyncio
async def test_vllm_generates():
    model_dir = _require_env("MATRIX_SAFE_VLLM_PATH")
    extras: dict[str, Any] = {
        "model_path": model_dir,
        "port": _next_port(),
        "startup_timeout_s": 240.0,
        "gpu_memory_utilization": float(os.environ.get("MATRIX_SAFE_VLLM_MEM", "0.6")),
    }
    py_bin: str | None = os.environ.get("MATRIX_SAFE_VLLM_PYTHON")
    if py_bin:
        extras["vllm_python"] = py_bin
    text = await _run_one("vllm", extras)
    assert text.strip()


# ----------------------------- cancellation path -----------------------------


@pytest.mark.asyncio
async def test_real_backend_cancellation_is_clean():
    """Pick the lightest available backend and verify CancelledError unwinds
    without leaving resources hung."""
    _require_module("llama_cpp")
    model_path = _require_env("MATRIX_SAFE_GGUF_PATH")
    mgr = BackendManager(max_resident=1)
    backend = await mgr.acquire(
        {
            "backend_target": "llama_cpp_python",
            "model_path": model_path,
            "max_tokens": 256,
            "temperature": 0.0,
        }
    )

    async def consume():
        async for _ in backend.generate_stream({"prompt": PROMPT}):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.1)  # let generation start
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Shutdown must still succeed after a cancelled stream.
    await mgr.release_all()
