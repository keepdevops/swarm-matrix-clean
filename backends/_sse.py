"""SSE parsing helper for llama-server /completion streaming responses."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

DONE_SENTINEL = "[DONE]"


def parse_sse_line(line: str) -> dict | None:
    """Parse one SSE line. Returns:
      - None for keep-alive / non-data / [DONE] lines
      - dict with parsed JSON payload otherwise
    Logs and returns None on malformed JSON (caller decides whether to abort).
    """
    if not line or not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == DONE_SENTINEL:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        logger.error("malformed SSE payload: %r", payload, exc_info=exc)
        return None


async def iter_completion_chunks(
    lines: AsyncIterator[str],
) -> AsyncIterator[dict]:
    """Convert llama-server SSE lines into OpenAI-shaped delta chunks.

    llama-server emits {'content': str, 'stop': bool, ...} per event.
    We translate to {'choices': [{'delta': {'content': str}, 'finish_reason': ...}]}.
    """
    async for line in lines:
        evt = parse_sse_line(line)
        if evt is None:
            continue
        content = evt.get("content", "")
        stop = bool(evt.get("stop", False))
        finish = "stop" if stop else None
        yield {
            "choices": [
                {
                    "delta": {"content": content},
                    "finish_reason": finish,
                }
            ]
        }
        if stop:
            return
