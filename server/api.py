"""FastAPI app: list backends/agents, stream generation over SSE."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from orchestration.manager import BackendManager, load_agent_config

logger = logging.getLogger(__name__)

AGENTS_DIR = Path(__file__).resolve().parents[1] / "config" / "agents"


def _config_error_detail(agent: str, exc: Exception) -> str:
    """Build a 400 detail that names *why* an agent config was rejected.

    Includes the underlying message (and therefore local filesystem paths) so a
    bad config is diagnosable from the client. Safe here because the API binds
    to localhost; revisit if this server is ever exposed beyond the machine.
    """
    if isinstance(exc, json.JSONDecodeError):
        kind = "is not valid JSON"
    elif isinstance(exc, FileNotFoundError):
        kind = "references a missing file"
    else:
        kind = "could not be loaded"
    return f"agent config {agent} {kind}: {exc}"


class GenerateRequest(BaseModel):
    agent: str = Field(..., description="Agent config filename (e.g. 'developer.json').")
    prompt: str = Field("", description="Prompt text from the editor.")
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=1, le=8192)
    backend_override: str | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="matrix-safe", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.state.manager = BackendManager(max_resident=1)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/backends")
    async def list_backends() -> dict[str, list[str]]:
        return {"backends": sorted(BackendManager._registry.keys())}

    @app.get("/api/agents")
    async def list_agents() -> dict[str, list[dict[str, Any]]]:
        if not AGENTS_DIR.exists():
            return {"agents": []}
        out: list[dict[str, Any]] = []
        for p in sorted(AGENTS_DIR.glob("*.json")):
            try:
                cfg = json.loads(p.read_text())
            except json.JSONDecodeError:
                logger.error("skipping malformed agent config %s", p, exc_info=True)
                continue
            out.append({
                "file": p.name,
                "agent_id": cfg.get("agent_id"),
                "name": cfg.get("name"),
                "backend_target": cfg.get("backend_target"),
                "model_path": cfg.get("model_path"),
            })
        return {"agents": out}

    @app.post("/api/generate")
    async def generate(req: GenerateRequest, request: Request) -> StreamingResponse:
        cfg_path = AGENTS_DIR / req.agent
        if not cfg_path.exists() or cfg_path.parent != AGENTS_DIR:
            raise HTTPException(404, f"agent config not found: {req.agent}")
        try:
            cfg = load_agent_config(cfg_path)
        except Exception as exc:
            logger.error("agent config load failed: %s", cfg_path, exc_info=True)
            raise HTTPException(400, _config_error_detail(req.agent, exc)) from exc

        if req.temperature is not None:
            cfg["temperature"] = req.temperature
        if req.max_tokens is not None:
            cfg["max_tokens"] = req.max_tokens
        if req.backend_override:
            cfg["backend_target"] = req.backend_override

        mgr: BackendManager = app.state.manager
        return StreamingResponse(
            _sse_stream(mgr, cfg, req.prompt, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.on_event("shutdown")
    async def _release() -> None:
        await app.state.manager.release_all()

    return app


async def _sse_stream(
    mgr: BackendManager,
    cfg: dict[str, Any],
    prompt: str,
    request: Request,
) -> AsyncGenerator[bytes, None]:
    """Yield Server-Sent Events for a single generate call.

    Event shapes:
      event: ready    data: {"backend": "..."}
      event: token    data: {"content": "..."}
      event: done     data: {"finish_reason": "stop"}
      event: error    data: {"error": "..."}
    """
    def _sse(event: str, data: dict[str, Any]) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    try:
        backend = await mgr.acquire(cfg)
    except Exception as exc:
        logger.error("backend acquire failed", exc_info=True)
        yield _sse("error", {"error": f"acquire failed: {exc}"})
        return

    yield _sse("ready", {"backend": cfg.get("backend_target", "?")})

    try:
        async for chunk in backend.generate_stream({"prompt": prompt}):
            if await request.is_disconnected():
                logger.info("client disconnected; stopping stream")
                return
            if "error" in chunk:
                yield _sse("error", {"error": chunk["error"]})
                return
            choice = (chunk.get("choices") or [{}])[0]
            text = (choice.get("delta") or {}).get("content", "")
            if text:
                yield _sse("token", {"content": text})
            finish = choice.get("finish_reason")
            if finish:
                yield _sse("done", {"finish_reason": finish})
                return
        yield _sse("done", {"finish_reason": "stop"})
    except Exception as exc:
        logger.error("stream error", exc_info=True)
        yield _sse("error", {"error": str(exc)})
