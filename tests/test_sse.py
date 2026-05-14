"""SSE parser tests for backends/_sse.py and vllm translator."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from backends._sse import iter_completion_chunks, parse_sse_line
from backends.vllm_engine import _translate_openai_chunk


def test_parse_sse_data_line():
    evt = parse_sse_line('data: {"content":"hi","stop":false}')
    assert evt == {"content": "hi", "stop": False}


def test_parse_sse_keepalive_returns_none():
    assert parse_sse_line("") is None
    assert parse_sse_line(": keep-alive") is None
    assert parse_sse_line("event: ping") is None


def test_parse_sse_done_returns_none():
    assert parse_sse_line("data: [DONE]") is None


def test_parse_sse_malformed_returns_none(caplog):
    caplog.set_level("ERROR")
    assert parse_sse_line("data: {not json") is None
    assert any("malformed SSE" in r.message for r in caplog.records)


async def _alines(items: list[str]) -> AsyncIterator[str]:
    for s in items:
        yield s


@pytest.mark.asyncio
async def test_iter_completion_chunks_translates_to_delta():
    raw = [
        'data: {"content":"hel","stop":false}',
        'data: {"content":"lo","stop":false}',
        'data: {"content":"","stop":true}',
        "data: [DONE]",
    ]
    out = [c async for c in iter_completion_chunks(_alines(raw))]
    assert len(out) == 3
    assert out[0]["choices"][0]["delta"]["content"] == "hel"
    assert out[1]["choices"][0]["delta"]["content"] == "lo"
    assert out[2]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_iter_completion_chunks_stops_after_stop_true():
    raw = [
        'data: {"content":"a","stop":true}',
        'data: {"content":"unreachable","stop":false}',
    ]
    out = [c async for c in iter_completion_chunks(_alines(raw))]
    assert len(out) == 1
    assert out[0]["choices"][0]["finish_reason"] == "stop"


def test_vllm_translator_completions_shape():
    line = 'data: {"choices":[{"text":"hello","finish_reason":null}]}'
    chunk = _translate_openai_chunk(line)
    assert chunk == {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]}


def test_vllm_translator_chat_delta_shape():
    line = 'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}'
    chunk = _translate_openai_chunk(line)
    assert chunk["choices"][0]["delta"]["content"] == "hi"


def test_vllm_translator_done_and_malformed():
    assert _translate_openai_chunk("data: [DONE]") is None
    assert _translate_openai_chunk("") is None
    assert _translate_openai_chunk("data: not-json") is None


def test_vllm_translator_finish_reason_propagates():
    line = 'data: {"choices":[{"text":"","finish_reason":"length"}]}'
    chunk = _translate_openai_chunk(line)
    assert chunk["choices"][0]["finish_reason"] == "length"
