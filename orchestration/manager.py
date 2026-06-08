"""BackendManager: hot-swappable LRU loader for inference backends.

Default policy: one active backend at a time (max_resident=1). Switch to
max_resident>=2 if you have memory headroom for concurrent engines.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

from backends.base import InferenceBackend
from backends.exllamav2_engine import ExLlamaV2Backend
from backends.llama_cpp import LlamaCppServerBackend
from backends.llama_python import LlamaCppPythonBackend
from backends.mlx_engine import MLXBackend
from backends.transformers_engine import TransformersBackend
from backends.vllm_engine import VLLMBackend
from orchestration.paths import resolve_paths

logger = logging.getLogger(__name__)


class BackendManager:
    """Lifecycle-aware backend registry with LRU eviction.

    Adding a new engine = one line in `_registry`. The orchestration API
    (acquire / release_all) does not change.
    """

    _registry: dict[str, type[InferenceBackend]] = {
        "llama_cpp_binary": LlamaCppServerBackend,
        "llama_cpp_python": LlamaCppPythonBackend,
        "mlx": MLXBackend,
        "vllm": VLLMBackend,
        "exllamav2": ExLlamaV2Backend,
        "transformers": TransformersBackend,
    }

    def __init__(self, max_resident: int = 1) -> None:
        if max_resident < 1:
            raise ValueError("max_resident must be >= 1")
        self._active: OrderedDict[str, InferenceBackend] = OrderedDict()
        self._lock = asyncio.Lock()
        self._max = max_resident

    @classmethod
    def register(cls, key: str, backend_cls: type[InferenceBackend]) -> None:
        """Register an additional backend at runtime."""
        cls._registry[key] = backend_cls

    @staticmethod
    def _cache_key(agent_config: dict[str, Any]) -> str:
        """LRU key. Includes model_path so two agents sharing a backend class
        but pointing at different weights get distinct resident instances."""
        return f"{agent_config.get('backend_target')}::{agent_config.get('model_path', '')}"

    async def acquire(self, agent_config: dict[str, Any]) -> InferenceBackend:
        """Return an initialized backend for the given agent config.

        Reuses a live backend if the same (backend_target, model_path) is
        already resident; otherwise evicts LRU entries until under
        `max_resident` and constructs a new one. Raises on unknown target
        or init failure (fail loudly).
        """
        target = agent_config.get("backend_target")
        if not target:
            raise ValueError("agent_config missing 'backend_target'")
        key = self._cache_key(agent_config)

        async with self._lock:
            if key in self._active:
                self._active.move_to_end(key)
                return self._active[key]

            while len(self._active) >= self._max:
                victim_key, victim = self._active.popitem(last=False)
                logger.info("evicting backend %s (LRU)", victim_key)
                try:
                    await victim.shutdown()
                except Exception:
                    logger.error("eviction shutdown failed for %s", victim_key, exc_info=True)

            cls = self._registry.get(target)
            if cls is None:
                raise ValueError(
                    f"unknown backend_target {target!r}; registered: {sorted(self._registry)}"
                )

            backend = cls()
            ok = await backend.initialize(agent_config)
            if not ok:
                raise RuntimeError(f"backend {target!r} failed to initialize")
            self._active[key] = backend
            logger.info("backend %s resident (active=%d)", key, len(self._active))
            return backend

    async def release_all(self) -> None:
        """Shutdown every resident backend. Idempotent."""
        async with self._lock:
            for key, backend in list(self._active.items()):
                try:
                    await backend.shutdown()
                except Exception:
                    logger.error("shutdown failed for %s", key, exc_info=True)
            self._active.clear()

    def resident(self) -> list[str]:
        return list(self._active.keys())


def load_agent_config(path: str | Path) -> dict[str, Any]:
    """Load, JSON-parse, and path-resolve an agent config file.

    Model/binary paths are resolved against env-configurable base dirs (see
    `orchestration.paths`); deeper validation happens in the backend.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"agent config not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except json.JSONDecodeError:
        logger.error("invalid JSON in %s", p, exc_info=True)
        raise
    return resolve_paths(cfg)
