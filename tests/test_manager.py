"""BackendManager acquire / LRU / shutdown / cancellation tests."""

from __future__ import annotations

import asyncio
import json

import pytest

from tests.conftest import FakeBackend, SlowFakeBackend


@pytest.mark.asyncio
async def test_acquire_initializes_backend(fake_registry):
    mgr = fake_registry(max_resident=1)
    backend = await mgr.acquire({"backend_target": "fake"})
    assert backend.initialized
    assert mgr.resident() == ["fake"]
    await mgr.release_all()
    assert backend.shutdown_called


@pytest.mark.asyncio
async def test_acquire_returns_same_instance(fake_registry):
    mgr = fake_registry(max_resident=1)
    b1 = await mgr.acquire({"backend_target": "fake"})
    b2 = await mgr.acquire({"backend_target": "fake"})
    assert b1 is b2
    await mgr.release_all()


@pytest.mark.asyncio
async def test_unknown_target_raises(fake_registry):
    mgr = fake_registry(max_resident=1)
    with pytest.raises(ValueError, match="unknown backend_target"):
        await mgr.acquire({"backend_target": "nope"})


@pytest.mark.asyncio
async def test_missing_target_raises(fake_registry):
    mgr = fake_registry()
    with pytest.raises(ValueError, match="missing 'backend_target'"):
        await mgr.acquire({})


@pytest.mark.asyncio
async def test_init_failure_raises_loudly(fake_registry):
    mgr = fake_registry()
    with pytest.raises(RuntimeError, match="failed to initialize"):
        await mgr.acquire({"backend_target": "fake", "force_fail": True})


@pytest.mark.asyncio
async def test_lru_evicts_when_over_capacity(fake_registry):
    mgr = fake_registry(max_resident=1)
    a = await mgr.acquire({"backend_target": "fake"})
    b = await mgr.acquire({"backend_target": "fake_slow"})
    assert a.shutdown_called  # evicted
    assert b.initialized
    assert mgr.resident() == ["fake_slow"]
    await mgr.release_all()


@pytest.mark.asyncio
async def test_max_resident_two_keeps_both(fake_registry):
    mgr = fake_registry(max_resident=2)
    a = await mgr.acquire({"backend_target": "fake"})
    await mgr.acquire({"backend_target": "fake_slow"})
    assert not a.shutdown_called
    assert mgr.resident() == ["fake", "fake_slow"]
    await mgr.release_all()


@pytest.mark.asyncio
async def test_max_resident_zero_rejected(fake_registry):
    with pytest.raises(ValueError):
        fake_registry(max_resident=0)


@pytest.mark.asyncio
async def test_generate_stream_yields_tokens(fake_registry):
    mgr = fake_registry()
    backend = await mgr.acquire({"backend_target": "fake", "tokens": ["a", "b", "c"]})
    out = []
    async for chunk in backend.generate_stream({"prompt": "p"}):
        out.append(chunk["choices"][0]["delta"]["content"])
    await mgr.release_all()
    assert "".join(out) == "abc"


@pytest.mark.asyncio
async def test_cancellation_propagates_to_backend(fake_registry):
    mgr = fake_registry()
    backend: SlowFakeBackend = await mgr.acquire({"backend_target": "fake_slow"})

    async def consume():
        async for _ in backend.generate_stream({"prompt": "p"}):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend.cancelled is True
    await mgr.release_all()


@pytest.mark.asyncio
async def test_release_all_is_idempotent(fake_registry):
    mgr = fake_registry()
    await mgr.acquire({"backend_target": "fake"})
    await mgr.release_all()
    await mgr.release_all()  # no exception
    assert mgr.resident() == []


def test_register_adds_backend(fake_registry):
    fake_registry.register("fake_extra", FakeBackend)
    assert "fake_extra" in fake_registry._registry


def test_load_agent_config_roundtrip(tmp_path):
    from orchestration.manager import load_agent_config

    p = tmp_path / "agent.json"
    p.write_text(json.dumps({"backend_target": "fake", "max_tokens": 16}))
    cfg = load_agent_config(p)
    assert cfg["backend_target"] == "fake"
    assert cfg["max_tokens"] == 16


def test_load_agent_config_missing_file(tmp_path):
    from orchestration.manager import load_agent_config

    with pytest.raises(FileNotFoundError):
        load_agent_config(tmp_path / "nope.json")


def test_load_agent_config_invalid_json(tmp_path):
    from orchestration.manager import load_agent_config

    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        load_agent_config(p)
