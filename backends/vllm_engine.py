"""vLLM backend via the OpenAI-compatible HTTP server.

Spawns `python -m vllm.entrypoints.openai.api_server` on 127.0.0.1 and streams
/v1/completions. Use this on CUDA hosts for high-throughput multi-request
generation (PagedAttention). Not suitable for Apple Silicon — pick `mlx` there.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from backends.base import BackendConfig, InferenceBackend

logger = logging.getLogger(__name__)


def _translate_openai_chunk(line: str) -> dict[str, Any] | None:
    """Parse one OpenAI-style SSE line into our delta shape.

    vLLM /v1/completions emits: {"choices":[{"text":"...","finish_reason":...}]}
    We re-shape into the project-wide {"choices":[{"delta":{"content":...}}]}.
    """
    if not line or not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        evt = json.loads(payload)
    except json.JSONDecodeError:
        logger.error("malformed vLLM SSE payload: %r", payload, exc_info=True)
        return None
    choices = evt.get("choices") or []
    if not choices:
        return None
    c0 = choices[0]
    text = c0.get("text", "") or (c0.get("delta") or {}).get("content", "")
    finish = c0.get("finish_reason")
    return {"choices": [{"delta": {"content": text}, "finish_reason": finish}]}


class VLLMBackend(InferenceBackend):
    """vLLM OpenAI-compatible server subprocess adapter."""

    name = "vllm"

    def __init__(self) -> None:
        self.process: asyncio.subprocess.Process | None = None
        self.client: httpx.AsyncClient | None = None
        self.base_url: str = ""
        self._drain_tasks: list[asyncio.Task] = []
        self._cfg: BackendConfig | None = None
        self._served_name: str = ""

    async def initialize(self, config: dict[str, Any]) -> bool:
        try:
            cfg = self.parse_config(config)
        except Exception:
            logger.error("invalid config for vllm", exc_info=True)
            return False

        if not cfg.model_path:
            logger.error("model_path (local model dir) is required for vllm")
            return False

        self._cfg = cfg
        self._served_name = cfg.served_model_name or cfg.model_path
        self.base_url = f"http://127.0.0.1:{cfg.port}"
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=None)

        python_bin = cfg.vllm_python or sys.executable
        cmd = [
            python_bin,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            cfg.model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(cfg.port),
            "--max-model-len",
            str(cfg.max_context_tokens),
            "--tensor-parallel-size",
            str(cfg.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(cfg.gpu_memory_utilization),
            "--dtype",
            cfg.dtype,
            "--served-model-name",
            self._served_name,
            *cfg.vllm_extra_args,
        ]

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError):
            logger.error("failed to spawn vLLM server: %s", cmd, exc_info=True)
            return False

        self._drain_tasks = [
            asyncio.create_task(self._drain(self.process.stdout, logging.DEBUG)),
            asyncio.create_task(self._drain(self.process.stderr, logging.INFO)),
        ]

        # vLLM warm-up can be slow (weight loading + CUDA graphs).
        timeout = max(cfg.startup_timeout_s, 60.0)
        if not await self._wait_ready(timeout):
            logger.error("vLLM server failed health check at %s", self.base_url)
            await self.shutdown()
            return False

        logger.info("vLLM server online at %s (model=%s)", self.base_url, self._served_name)
        return True

    async def _drain(self, stream: asyncio.StreamReader | None, level: int) -> None:
        if stream is None:
            return
        try:
            while True:
                line = await stream.readline()
                if not line:
                    return
                logger.log(level, "[vllm] %s", line.decode(errors="replace").rstrip())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("vllm pipe drain failed", exc_info=True)

    async def _wait_ready(self, timeout_s: float) -> bool:
        assert self.client is not None
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            if self.process and self.process.returncode is not None:
                logger.error("vLLM exited early rc=%s", self.process.returncode)
                return False
            try:
                r = await self.client.get("/health", timeout=2.0)
                if r.status_code == 200:
                    return True
                logger.warning("vLLM health non-200: %s", r.status_code)
            except httpx.RequestError:
                pass
            except Exception:
                logger.error("vLLM health probe raised", exc_info=True)
            await asyncio.sleep(1.0)
        return False

    async def generate_stream(
        self, payload: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        if self.client is None or self._cfg is None:
            yield {"error": "backend not initialized"}
            return

        cfg = self._cfg
        api_payload = {
            "model": self._served_name,
            "prompt": payload.get("prompt", ""),
            "stream": True,
            "temperature": payload.get("temperature", cfg.temperature),
            "max_tokens": payload.get("max_tokens", cfg.max_tokens),
        }

        try:
            async with self.client.stream(
                "POST", "/v1/completions", json=api_payload, timeout=None
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    logger.error(
                        "vLLM /v1/completions %s: %s",
                        response.status_code,
                        body[:512],
                    )
                    yield {"error": f"upstream status {response.status_code}"}
                    return
                async for line in response.aiter_lines():
                    chunk = _translate_openai_chunk(line)
                    if chunk is None:
                        continue
                    yield chunk
                    if chunk["choices"][0]["finish_reason"]:
                        return
        except asyncio.CancelledError:
            logger.info("vLLM generate_stream cancelled")
            raise
        except httpx.HTTPError as exc:
            logger.error("vLLM stream failure", exc_info=True)
            yield {"error": f"stream fault: {exc}"}

    async def shutdown(self) -> bool:
        if self.process is not None and self.process.returncode is None:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.error("vLLM did not terminate; killing")
                try:
                    self.process.kill()
                    await self.process.wait()
                except ProcessLookupError:
                    pass
            except Exception:
                logger.error("vLLM shutdown error", exc_info=True)
        self.process = None

        for t in self._drain_tasks:
            t.cancel()
        self._drain_tasks = []

        if self.client is not None:
            try:
                await self.client.aclose()
            except Exception:
                logger.error("vLLM client close error", exc_info=True)
            self.client = None

        logger.info("vLLM backend shut down")
        return True
