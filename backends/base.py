"""Abstract base + validated config for all inference backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class BackendConfig(BaseModel):
    """Schema-validated config passed to every backend's initialize()."""

    backend_target: str = Field(..., description="Registry key, e.g. 'llama_cpp_binary'.")
    model_path: str | None = Field(None, description="Path to GGUF or model directory.")
    max_context_tokens: int = Field(4096, ge=256, le=131072)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=1, le=32768)

    # llama-server binary specifics
    llama_server_path: str | None = None
    port: int = Field(8080, ge=1024, le=65535)
    n_gpu_layers: int = Field(99, ge=-1, description="-1=auto, 0=CPU only, 99=all to GPU.")
    ubatch_size: int = Field(512, ge=16, le=4096)
    startup_timeout_s: float = Field(20.0, ge=1.0, le=300.0)

    # vLLM specifics
    vllm_python: str | None = Field(None, description="Python interpreter with vllm installed.")
    served_model_name: str | None = Field(None, description="Name advertised by /v1/models.")
    tensor_parallel_size: int = Field(1, ge=1, le=16)
    gpu_memory_utilization: float = Field(0.9, ge=0.1, le=0.98)
    dtype: str = Field("auto", description="auto|float16|bfloat16|float32.")
    vllm_extra_args: list[str] = Field(default_factory=list)

    @field_validator("model_path", "llama_server_path", "vllm_python")
    @classmethod
    def _path_exists_if_set(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return v
        if not Path(v).exists():
            raise ValueError(f"path does not exist: {v}")
        return v


class InferenceBackend(ABC):
    """All engine adapters MUST implement this contract.

    Lifecycle: initialize() -> generate_stream()* -> shutdown().
    Errors must be logged with logger.error(exc_info=True) — never silently swallowed.
    """

    name: str = "abstract"

    @abstractmethod
    async def initialize(self, config: dict[str, Any]) -> bool:
        """Load the engine. Return True on success, False on failure (and log)."""

    @abstractmethod
    async def generate_stream(
        self, payload: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Yield OpenAI-shaped chunks: {'choices': [{'delta': {'content': str}}]}.

        On error, yield a single {'error': str} and return.
        Must honor asyncio.CancelledError by stopping upstream generation.
        """

    @abstractmethod
    async def shutdown(self) -> bool:
        """Release resources. Must be idempotent."""

    @staticmethod
    def parse_config(raw: dict[str, Any]) -> BackendConfig:
        """Validate raw dict into BackendConfig. Raises pydantic.ValidationError on failure."""
        return BackendConfig.model_validate(raw)
