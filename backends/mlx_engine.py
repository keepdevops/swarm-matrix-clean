"""Apple MLX backend via mlx-lm. Fast path for M-series Silicon.

MLX weights live as a directory (HF-style): config.json + weights.safetensors +
tokenizer files. Point `model_path` at that directory.
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


class MLXBackend(InferenceBackend):
    """In-process MLX inference. GGUF not supported here — use llama_cpp_* for that."""

    name = "mlx"

    def __init__(self) -> None:
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self._stream_generate: Any | None = None
        self._cfg: BackendConfig | None = None
        self._gen_lock = asyncio.Lock()

    async def initialize(self, config: dict[str, Any]) -> bool:
        try:
            cfg = self.parse_config(config)
        except Exception:
            logger.error("invalid config for mlx", exc_info=True)
            return False

        if not cfg.model_path:
            logger.error("model_path (MLX weights directory) is required")
            return False

        try:
            from mlx_lm import load, stream_generate  # deferred import
        except ImportError:
            logger.error(
                "package 'mlx-lm' not installed; pip install mlx-lm",
                exc_info=True,
            )
            return False

        try:
            model, tokenizer = await asyncio.to_thread(load, cfg.model_path)
        except Exception:
            logger.error("mlx_lm.load failed for %s", cfg.model_path, exc_info=True)
            return False

        self.model = model
        self.tokenizer = tokenizer
        self._stream_generate = stream_generate
        self._cfg = cfg
        logger.info("MLX backend loaded %s (ctx=%d)", cfg.model_path, cfg.max_context_tokens)
        return True

    async def generate_stream(
        self, payload: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        if self.model is None or self.tokenizer is None or self._cfg is None:
            yield {"error": "backend not initialized"}
            return

        cfg = self._cfg
        prompt = payload.get("prompt", "")
        max_tokens = int(payload.get("max_tokens", cfg.max_tokens))
        temperature = float(payload.get("temperature", cfg.temperature))

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)
        stop_flag = threading.Event()

        stream_generate = self._stream_generate
        model = self.model
        tokenizer = self.tokenizer

        def worker() -> None:
            try:
                # mlx-lm >=0.18 takes sampler kwargs; older versions accept temp directly.
                # We pass temperature via the broadly-compatible `temp` kwarg.
                kwargs: dict[str, Any] = {"max_tokens": max_tokens}
                try:
                    from mlx_lm.sample_utils import make_sampler

                    kwargs["sampler"] = make_sampler(temp=temperature)
                except Exception:
                    kwargs["temp"] = temperature  # legacy fallback

                last_finish: str | None = None
                for response in stream_generate(model, tokenizer, prompt, **kwargs):
                    if stop_flag.is_set():
                        break
                    text = getattr(response, "text", None)
                    if text is None and isinstance(response, str):
                        text = response  # very-legacy mlx-lm
                    if text is None:
                        continue
                    last_finish = getattr(response, "finish_reason", None)
                    item = {"choices": [{"delta": {"content": text}, "finish_reason": None}]}
                    asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()

                terminal = {
                    "choices": [{"delta": {"content": ""}, "finish_reason": last_finish or "stop"}]
                }
                asyncio.run_coroutine_threadsafe(queue.put(terminal), loop).result()
            except Exception as exc:
                logger.error("MLX worker failed", exc_info=True)
                asyncio.run_coroutine_threadsafe(
                    queue.put({"error": f"generation fault: {exc}"}), loop
                ).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL_DONE), loop)

        async with self._gen_lock:
            thread = threading.Thread(target=worker, daemon=True, name="mlx-gen")
            thread.start()
            try:
                while True:
                    item = await queue.get()
                    if item is _SENTINEL_DONE:
                        return
                    yield item
            except asyncio.CancelledError:
                logger.info("generate_stream cancelled; signaling MLX worker stop")
                stop_flag.set()
                raise
            finally:
                stop_flag.set()
                await asyncio.to_thread(thread.join, 5.0)
                if thread.is_alive():
                    logger.error("MLX worker did not exit within 5s")

    async def shutdown(self) -> bool:
        self.model = None
        self.tokenizer = None
        self._stream_generate = None
        logger.info("MLX backend shut down")
        return True
