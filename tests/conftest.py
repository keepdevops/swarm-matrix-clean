"""Shared pytest fixtures + a fake backend for manager/CLI tests."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest

# Make project root importable when tests are run from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.base import InferenceBackend  # noqa: E402


class FakeBackend(InferenceBackend):
    """In-memory backend that yields a fixed token list. No external deps."""

    name = "fake"

    # class-level instrumentation so tests can inspect lifecycle
    instances: list[FakeBackend] = []

    def __init__(self) -> None:
        self.initialized = False
        self.shutdown_called = False
        self.cancelled = False
        self._tokens: list[str] = ["hello", " ", "world"]
        FakeBackend.instances.append(self)

    async def initialize(self, config: dict[str, Any]) -> bool:
        if config.get("force_fail"):
            return False
        self._tokens = list(config.get("tokens", self._tokens))
        self.initialized = True
        return True

    async def generate_stream(
        self, payload: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        try:
            for tok in self._tokens:
                await asyncio.sleep(0)  # cooperative cancel point
                yield {"choices": [{"delta": {"content": tok}, "finish_reason": None}]}
            yield {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]}
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def shutdown(self) -> bool:
        self.shutdown_called = True
        return True


class SlowFakeBackend(FakeBackend):
    """Sleeps between tokens so cancellation tests have a window to act."""

    name = "fake_slow"

    async def generate_stream(
        self, payload: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        try:
            for tok in self._tokens:
                await asyncio.sleep(0.05)
                yield {"choices": [{"delta": {"content": tok}, "finish_reason": None}]}
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.fixture(autouse=True)
def _reset_fake_instances():
    FakeBackend.instances.clear()
    yield
    FakeBackend.instances.clear()


@pytest.fixture
def fake_registry(monkeypatch):
    """Swap BackendManager._registry with fake backends for the test."""
    from orchestration.manager import BackendManager

    original = BackendManager._registry
    monkeypatch.setattr(
        BackendManager,
        "_registry",
        {"fake": FakeBackend, "fake_slow": SlowFakeBackend},
    )
    yield BackendManager
    monkeypatch.setattr(BackendManager, "_registry", original)
