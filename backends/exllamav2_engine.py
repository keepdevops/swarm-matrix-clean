"""ExLlamaV2 backend.

In-process EXL2 inference on CUDA/ROCm. Point `model_path` at an EXL2 model
directory (config.json + tokenizer + *.safetensors with EXL2 quantization).
Not supported on Apple Silicon — use `mlx` or `llama_cpp_*` there.
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


class ExLlamaV2Backend(InferenceBackend):
    """ExLlamaV2 streaming generator wrapped in an asyncio bridge."""

    name = "exllamav2"

    def __init__(self) -> None:
        self.model: Any | None = None
        self.cache: Any | None = None
        self.tokenizer: Any | None = None
        self.generator: Any | None = None
        self._sampler_settings_cls: Any | None = None
        self._cfg: BackendConfig | None = None
        self._gen_lock = asyncio.Lock()

    async def initialize(self, config: dict[str, Any]) -> bool:
        try:
            cfg = self.parse_config(config)
        except Exception:
            logger.error("invalid config for exllamav2", exc_info=True)
            return False

        if not cfg.model_path:
            logger.error("model_path (EXL2 model directory) is required")
            return False

        try:
            from exllamav2 import (
                ExLlamaV2,
                ExLlamaV2Cache,
                ExLlamaV2Config,
                ExLlamaV2Tokenizer,
            )
            from exllamav2.generator import (
                ExLlamaV2Sampler,
                ExLlamaV2StreamingGenerator,
            )
        except ImportError:
            logger.error(
                "package 'exllamav2' not installed; pip install exllamav2",
                exc_info=True,
            )
            return False

        def _build() -> tuple:
            ex_config = ExLlamaV2Config(cfg.model_path)
            ex_config.max_seq_len = cfg.max_context_tokens
            model = ExLlamaV2(ex_config)
            cache = ExLlamaV2Cache(model, lazy=True)
            model.load_autosplit(cache)
            tokenizer = ExLlamaV2Tokenizer(ex_config)
            generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
            generator.warmup()
            return model, cache, tokenizer, generator

        try:
            model, cache, tokenizer, generator = await asyncio.to_thread(_build)
        except Exception:
            logger.error("ExLlamaV2 init failed for %s", cfg.model_path, exc_info=True)
            return False

        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        self.generator = generator
        self._sampler_settings_cls = ExLlamaV2Sampler.Settings
        self._cfg = cfg
        logger.info("ExLlamaV2 loaded %s (ctx=%d)", cfg.model_path, cfg.max_context_tokens)
        return True

    async def generate_stream(
        self, payload: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        if self.generator is None or self.tokenizer is None or self._cfg is None:
            yield {"error": "backend not initialized"}
            return

        cfg = self._cfg
        prompt = payload.get("prompt", "")
        max_tokens = int(payload.get("max_tokens", cfg.max_tokens))
        temperature = float(payload.get("temperature", cfg.temperature))

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)
        stop_flag = threading.Event()

        generator = self.generator
        tokenizer = self.tokenizer
        settings_cls = self._sampler_settings_cls
        assert settings_cls is not None

        def worker() -> None:
            try:
                settings = settings_cls()
                settings.temperature = temperature
                input_ids = tokenizer.encode(prompt)
                generator.set_stop_conditions([tokenizer.eos_token_id])
                generator.begin_stream(input_ids, settings)

                produced = 0
                finish = "stop"
                while produced < max_tokens:
                    if stop_flag.is_set():
                        finish = "cancelled"
                        break
                    chunk, eos, _ = generator.stream()
                    if chunk:
                        item = {"choices": [{"delta": {"content": chunk}, "finish_reason": None}]}
                        asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()
                    produced += 1
                    if eos:
                        finish = "stop"
                        break
                else:
                    finish = "length"

                terminal = {"choices": [{"delta": {"content": ""}, "finish_reason": finish}]}
                asyncio.run_coroutine_threadsafe(queue.put(terminal), loop).result()
            except Exception as exc:
                logger.error("ExLlamaV2 worker failed", exc_info=True)
                asyncio.run_coroutine_threadsafe(
                    queue.put({"error": f"generation fault: {exc}"}), loop
                ).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL_DONE), loop)

        async with self._gen_lock:
            thread = threading.Thread(target=worker, daemon=True, name="exl2-gen")
            thread.start()
            try:
                while True:
                    item = await queue.get()
                    if item is _SENTINEL_DONE:
                        return
                    yield item
            except asyncio.CancelledError:
                logger.info("ExLlamaV2 generate_stream cancelled")
                stop_flag.set()
                raise
            finally:
                stop_flag.set()
                await asyncio.to_thread(thread.join, 5.0)
                if thread.is_alive():
                    logger.error("ExLlamaV2 worker did not exit within 5s")

    async def shutdown(self) -> bool:
        # Order matters: release generator → cache → model so CUDA tensors
        # are freed in dependency order.
        self.generator = None
        if self.cache is not None:
            try:
                del self.cache
            except Exception:
                logger.error("error releasing ExLlamaV2 cache", exc_info=True)
            self.cache = None
        if self.model is not None:
            try:
                unload = getattr(self.model, "unload", None)
                if callable(unload):
                    unload()
                del self.model
            except Exception:
                logger.error("error releasing ExLlamaV2 model", exc_info=True)
            self.model = None
        self.tokenizer = None
        self._sampler_settings_cls = None
        logger.info("ExLlamaV2 backend shut down")
        return True
