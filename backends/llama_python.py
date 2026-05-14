"""In-process llama-cpp-python bindings adapter.

Runs the blocking generator in a worker thread and bridges chunks back to the
event loop via an asyncio.Queue so the control plane never blocks.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncGenerator
from typing import Any

from backends.base import BackendConfig, InferenceBackend

logger = logging.getLogger(__name__)

_SENTINEL_DONE: dict[str, Any] = {"__done__": True}


class LlamaCppPythonBackend(InferenceBackend):
    """Single-process llama.cpp via Python bindings. GGUF input."""

    name = "llama_cpp_python"

    def __init__(self) -> None:
        self.llm: Any | None = None
        self._cfg: BackendConfig | None = None
        self._gen_lock = asyncio.Lock()  # serialize concurrent generations

    async def initialize(self, config: dict[str, Any]) -> bool:
        try:
            cfg = self.parse_config(config)
        except Exception:
            logger.error("invalid config for llama_cpp_python", exc_info=True)
            return False

        if not cfg.model_path:
            logger.error("model_path is required for llama_cpp_python")
            return False

        try:
            from llama_cpp import Llama  # deferred import
        except ImportError:
            logger.error(
                "package 'llama-cpp-python' not installed; pip install llama-cpp-python",
                exc_info=True,
            )
            return False

        try:
            self.llm = await asyncio.to_thread(
                Llama,
                model_path=cfg.model_path,
                n_ctx=cfg.max_context_tokens,
                n_gpu_layers=cfg.n_gpu_layers,
                verbose=False,
            )
        except Exception:
            logger.error("Llama() construction failed", exc_info=True)
            return False

        self._cfg = cfg
        logger.info("llama_cpp_python loaded %s (ctx=%d)", cfg.model_path, cfg.max_context_tokens)
        return True

    async def generate_stream(
        self, payload: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        if self.llm is None or self._cfg is None:
            yield {"error": "backend not initialized"}
            return

        cfg = self._cfg
        prompt = payload.get("prompt", "")
        max_tokens = int(payload.get("max_tokens", cfg.max_tokens))
        temperature = float(payload.get("temperature", cfg.temperature))

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)
        stop_flag = threading.Event()

        def worker() -> None:
            try:
                stream = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )
                for chunk in stream:
                    if stop_flag.is_set():
                        break
                    text = chunk["choices"][0].get("text", "")
                    finish = chunk["choices"][0].get("finish_reason")
                    item = {"choices": [{"delta": {"content": text}, "finish_reason": finish}]}
                    asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()
            except Exception as exc:
                logger.error("llama_cpp_python worker failed", exc_info=True)
                asyncio.run_coroutine_threadsafe(
                    queue.put({"error": f"generation fault: {exc}"}), loop
                ).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL_DONE), loop)

        async with self._gen_lock:
            thread = threading.Thread(target=worker, daemon=True, name="llama-py-gen")
            thread.start()
            try:
                while True:
                    item = await queue.get()
                    if item is _SENTINEL_DONE:
                        return
                    yield item
            except asyncio.CancelledError:
                logger.info("generate_stream cancelled; signaling worker stop")
                stop_flag.set()
                raise
            finally:
                stop_flag.set()
                await asyncio.to_thread(thread.join, 5.0)
                if thread.is_alive():
                    logger.error("llama_cpp_python worker did not exit within 5s")

    async def shutdown(self) -> bool:
        if self.llm is not None:
            try:
                del self.llm
            except Exception:
                logger.error("error releasing Llama handle", exc_info=True)
            self.llm = None
        logger.info("llama_cpp_python backend shut down")
        return True
