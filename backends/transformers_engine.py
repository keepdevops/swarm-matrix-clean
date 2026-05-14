"""Hugging Face Transformers backend.

In-process inference via AutoModelForCausalLM + TextIteratorStreamer.
Works on CUDA, ROCm, MPS (Apple Silicon), and CPU — slowest path but the most
portable. Point `model_path` at a local HF model directory.
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

_DTYPE_MAP: dict[str, str] = {
    "auto": "auto",
    "float16": "float16",
    "fp16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float32": "float32",
    "fp32": "float32",
}


def _resolve_torch_dtype(name: str) -> Any:
    """Map a config string to a torch dtype or the literal 'auto'."""
    import torch  # local import; transformers always pulls torch

    key = _DTYPE_MAP.get(name.lower(), "auto")
    if key == "auto":
        return "auto"
    return getattr(torch, key)


class TransformersBackend(InferenceBackend):
    """HF Transformers with TextIteratorStreamer + StoppingCriteria for cancel."""

    name = "transformers"

    def __init__(self) -> None:
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self._cfg: BackendConfig | None = None
        self._gen_lock = asyncio.Lock()

    async def initialize(self, config: dict[str, Any]) -> bool:
        try:
            cfg = self.parse_config(config)
        except Exception:
            logger.error("invalid config for transformers", exc_info=True)
            return False

        if not cfg.model_path:
            logger.error("model_path (HF model directory) is required for transformers")
            return False

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        except ImportError:
            logger.error(
                "package 'transformers' not installed; pip install transformers torch",
                exc_info=True,
            )
            return False

        def _build() -> tuple:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch_dtype = _resolve_torch_dtype(cfg.dtype)
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            model.eval()
            return model, tokenizer

        try:
            model, tokenizer = await asyncio.to_thread(_build)
        except Exception:
            logger.error("transformers init failed for %s", cfg.model_path, exc_info=True)
            return False

        self.model = model
        self.tokenizer = tokenizer
        self._cfg = cfg
        logger.info(
            "transformers loaded %s (ctx=%d, dtype=%s)",
            cfg.model_path,
            cfg.max_context_tokens,
            cfg.dtype,
        )
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

        model = self.model
        tokenizer = self.tokenizer

        def worker() -> None:
            gen_thread: threading.Thread | None = None
            try:
                from transformers import (
                    StoppingCriteria,
                    StoppingCriteriaList,
                    TextIteratorStreamer,
                )

                class _CancelCriteria(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kw):
                        return stop_flag.is_set()

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                streamer = TextIteratorStreamer(
                    tokenizer, skip_prompt=True, skip_special_tokens=True
                )

                gen_kwargs: dict[str, Any] = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0.0,
                    temperature=max(temperature, 1e-5),
                    stopping_criteria=StoppingCriteriaList([_CancelCriteria()]),
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

                gen_thread = threading.Thread(
                    target=model.generate,
                    kwargs=gen_kwargs,
                    daemon=True,
                    name="hf-generate",
                )
                gen_thread.start()

                for text in streamer:
                    if stop_flag.is_set():
                        break
                    if not text:
                        continue
                    item = {"choices": [{"delta": {"content": text}, "finish_reason": None}]}
                    asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()

                finish = "cancelled" if stop_flag.is_set() else "stop"
                terminal = {"choices": [{"delta": {"content": ""}, "finish_reason": finish}]}
                asyncio.run_coroutine_threadsafe(queue.put(terminal), loop).result()
            except Exception as exc:
                logger.error("transformers worker failed", exc_info=True)
                asyncio.run_coroutine_threadsafe(
                    queue.put({"error": f"generation fault: {exc}"}), loop
                ).result()
            finally:
                if gen_thread is not None:
                    gen_thread.join(timeout=10.0)
                    if gen_thread.is_alive():
                        logger.error("hf-generate thread did not exit within 10s")
                asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL_DONE), loop)

        async with self._gen_lock:
            thread = threading.Thread(target=worker, daemon=True, name="hf-bridge")
            thread.start()
            try:
                while True:
                    item = await queue.get()
                    if item is _SENTINEL_DONE:
                        return
                    yield item
            except asyncio.CancelledError:
                logger.info("transformers generate_stream cancelled")
                stop_flag.set()
                raise
            finally:
                stop_flag.set()
                await asyncio.to_thread(thread.join, 15.0)
                if thread.is_alive():
                    logger.error("transformers bridge thread did not exit within 15s")

    async def shutdown(self) -> bool:
        if self.model is not None:
            try:
                del self.model
            except Exception:
                logger.error("error releasing transformers model", exc_info=True)
            self.model = None
        self.tokenizer = None
        try:
            import torch

            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
        except Exception:
            logger.error("torch cache flush failed", exc_info=True)
        logger.info("transformers backend shut down")
        return True
