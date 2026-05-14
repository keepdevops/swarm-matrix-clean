"""Native llama.cpp llama-server subprocess adapter (air-gapped, M3 Metal)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from backends._sse import iter_completion_chunks
from backends.base import BackendConfig, InferenceBackend

logger = logging.getLogger(__name__)


class LlamaCppServerBackend(InferenceBackend):
    """Spawns llama-server on 127.0.0.1 and streams /completion responses."""

    name = "llama_cpp_binary"

    def __init__(self) -> None:
        self.process: asyncio.subprocess.Process | None = None
        self.client: httpx.AsyncClient | None = None
        self.base_url: str = ""
        self._drain_tasks: list[asyncio.Task] = []
        self._cfg: BackendConfig | None = None

    async def initialize(self, config: dict[str, Any]) -> bool:
        try:
            cfg = self.parse_config(config)
        except Exception:
            logger.error("invalid config for llama_cpp_binary", exc_info=True)
            return False

        if not cfg.llama_server_path or not cfg.model_path:
            logger.error(
                "llama_server_path and model_path are required (got %r, %r)",
                cfg.llama_server_path,
                cfg.model_path,
            )
            return False

        self._cfg = cfg
        self.base_url = f"http://127.0.0.1:{cfg.port}"
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=None)

        cmd = [
            cfg.llama_server_path,
            "--model",
            cfg.model_path,
            "--port",
            str(cfg.port),
            "--ctx-size",
            str(cfg.max_context_tokens),
            "--gpu-layers",
            str(cfg.n_gpu_layers),
            "--ubatch-size",
            str(cfg.ubatch_size),
            "--host",
            "127.0.0.1",
        ]

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError):
            logger.error("failed to spawn llama-server: %s", cmd, exc_info=True)
            return False

        # Drain pipes to prevent buffer-full deadlock.
        self._drain_tasks = [
            asyncio.create_task(self._drain(self.process.stdout, logging.DEBUG)),
            asyncio.create_task(self._drain(self.process.stderr, logging.INFO)),
        ]

        if not await self._wait_ready(cfg.startup_timeout_s):
            logger.error("llama-server failed health check at %s", self.base_url)
            await self.shutdown()
            return False

        logger.info("llama.cpp server online at %s", self.base_url)
        return True

    async def _drain(self, stream: asyncio.StreamReader | None, level: int) -> None:
        if stream is None:
            return
        try:
            while True:
                line = await stream.readline()
                if not line:
                    return
                logger.log(level, "[llama-server] %s", line.decode(errors="replace").rstrip())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("pipe drain failed", exc_info=True)

    async def _wait_ready(self, timeout_s: float) -> bool:
        assert self.client is not None
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            if self.process and self.process.returncode is not None:
                logger.error("llama-server exited early rc=%s", self.process.returncode)
                return False
            try:
                r = await self.client.get("/health", timeout=2.0)
                if r.status_code == 200:
                    return True
                logger.warning("health probe non-200: %s", r.status_code)
            except httpx.RequestError:
                pass
            except Exception:
                logger.error("health probe raised", exc_info=True)
            await asyncio.sleep(0.5)
        return False

    async def generate_stream(
        self, payload: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        if self.client is None or self._cfg is None:
            yield {"error": "backend not initialized"}
            return

        cfg = self._cfg
        api_payload = {
            "prompt": payload.get("prompt", ""),
            "stream": True,
            "temperature": payload.get("temperature", cfg.temperature),
            "n_predict": payload.get("max_tokens", cfg.max_tokens),
        }

        try:
            async with self.client.stream(
                "POST", "/completion", json=api_payload, timeout=None
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    logger.error(
                        "llama-server /completion %s: %s",
                        response.status_code,
                        body[:512],
                    )
                    yield {"error": f"upstream status {response.status_code}"}
                    return
                async for chunk in iter_completion_chunks(response.aiter_lines()):
                    yield chunk
        except asyncio.CancelledError:
            logger.info("generate_stream cancelled; upstream closed")
            raise
        except httpx.HTTPError as exc:
            logger.error("llama-server stream failure", exc_info=True)
            yield {"error": f"stream fault: {exc}"}

    async def shutdown(self) -> bool:
        if self.process is not None and self.process.returncode is None:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.error("llama-server did not terminate; killing")
                try:
                    self.process.kill()
                    await self.process.wait()
                except ProcessLookupError:
                    pass
            except Exception:
                logger.error("shutdown error", exc_info=True)
        self.process = None

        for t in self._drain_tasks:
            t.cancel()
        self._drain_tasks = []

        if self.client is not None:
            try:
                await self.client.aclose()
            except Exception:
                logger.error("client close error", exc_info=True)
            self.client = None

        logger.info("llama.cpp backend shut down")
        return True
