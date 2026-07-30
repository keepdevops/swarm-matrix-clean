"""Apple MLX backend via mlx-lm. Fast path for M-series Silicon.

MLX weights live as a directory (HF-style): config.json + weights.safetensors +
tokenizer files. Point `model_path` at that directory.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import AsyncGenerator
from typing import Any

from backends.base import BackendConfig, InferenceBackend

logger = logging.getLogger(__name__)

_SENTINEL_DONE: dict[str, Any] = {"__done__": True}

# How often a blocked worker re-checks stop_flag while handing off a chunk.
_EMIT_POLL_SECONDS = 0.1
# How long a signalled worker gets to unwind before we call the thread wedged.
_WORKER_EXIT_GRACE_SECONDS = 5.0


class MLXBackend(InferenceBackend):
    """In-process MLX inference. GGUF not supported here — use llama_cpp_* for that."""

    name = "mlx"

    def __init__(self) -> None:
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self._stream_generate: Any | None = None
        self._cfg: BackendConfig | None = None
        self._gen_lock = asyncio.Lock()
        # MLX's default GPU stream is thread-local, so the model load and every
        # generation must run on the SAME thread. A single-worker executor pins
        # all MLX work to one thread; otherwise generation on a fresh thread
        # raises "There is no Stream(gpu, N) in current thread".
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        # Set if a worker never returned. The pinned thread owns the model, so
        # there is nothing to restart it with — later calls fail fast instead.
        self._wedged = False

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

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx"
        )

        def _load_on_pinned() -> tuple[Any, Any]:
            # Establish this thread's default GPU stream, then load the model on
            # it so generation (also on this thread) shares the same stream.
            try:
                import mlx.core as mx

                mx.set_default_device(mx.gpu)
            except Exception:
                logger.error("could not set MLX default device", exc_info=True)
            return load(cfg.model_path)

        loop = asyncio.get_running_loop()
        try:
            model, tokenizer = await loop.run_in_executor(self._executor, _load_on_pinned)
        except Exception:
            logger.error("mlx_lm.load failed for %s", cfg.model_path, exc_info=True)
            self._executor.shutdown(wait=False)
            self._executor = None
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
        if (
            self.model is None
            or self.tokenizer is None
            or self._cfg is None
            or self._executor is None
        ):
            yield {"error": "backend not initialized"}
            return

        if self._wedged:
            # A previous worker never came back and still holds the executor's
            # only slot, so a new one would queue behind it and hang forever.
            yield {"error": "MLX worker thread is wedged; restart the backend"}
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

        def emit(item: dict[str, Any]) -> bool:
            """Hand one chunk to the event loop. False means stop generating.

            Never waits indefinitely. A plain `.result()` here is what wedged the
            backend: with a full queue and a consumer that had stopped draining,
            the pinned thread parked forever, and because the executor has a
            single worker, every later generation queued behind it and hung too.
            Polling lets a blocked worker notice stop_flag and unwind.
            """
            put = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            while True:
                try:
                    put.result(timeout=_EMIT_POLL_SECONDS)
                    return True
                except concurrent.futures.TimeoutError:
                    if stop_flag.is_set():
                        put.cancel()
                        return False
                except Exception:
                    logger.error("could not hand MLX chunk to event loop", exc_info=True)
                    return False

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
            # Run on the pinned single-thread executor (same thread as load) so
            # MLX's thread-local GPU stream is valid for generation.
            future = loop.run_in_executor(self._executor, worker)
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
                try:
                    await asyncio.wait_for(asyncio.shield(future), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.error("MLX worker did not finish within 5s")
                except Exception:
                    logger.error("MLX worker errored on shutdown", exc_info=True)

    async def shutdown(self) -> bool:
        self.model = None
        self.tokenizer = None
        self._stream_generate = None
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        logger.info("MLX backend shut down")
        return True
