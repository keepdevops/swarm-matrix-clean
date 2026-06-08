"""Resolve model and binary paths from env-configurable base dirs.

Agent configs store paths *relative* to a base directory so no machine-specific
absolute path is baked into version control. An absolute path in a config is
still honored as-is (escape hatch); a relative path resolves against the base.

Env vars (with defaults):
    MATRIX_SAFE_MODELS_DIR    base dir for model files
                              (default: /Users/Shared/llama/models)
    MATRIX_SAFE_LLAMA_SERVER  llama-server binary path
                              (default: /Users/Shared/llama/llama-server)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_MODELS_DIR = "MATRIX_SAFE_MODELS_DIR"
ENV_LLAMA_SERVER = "MATRIX_SAFE_LLAMA_SERVER"

DEFAULT_MODELS_DIR = "/Users/Shared/llama/models"
DEFAULT_LLAMA_SERVER = "/Users/Shared/llama/llama-server"


def models_dir() -> Path:
    """Base directory for model files (env-overridable)."""
    return Path(os.environ.get(ENV_MODELS_DIR, DEFAULT_MODELS_DIR)).expanduser()


def llama_server_default() -> str:
    """Default llama-server binary path (env-overridable)."""
    return os.environ.get(ENV_LLAMA_SERVER, DEFAULT_LLAMA_SERVER)


def _resolve(value: str, base: Path) -> str:
    """Expand ~ and join relative paths against `base`; absolute paths pass through."""
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else (base / p))


def resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve `model_path` (and `llama_server_path` for llama_cpp) in place.

    Relative paths resolve against the env-configured base dirs; absolute paths
    are kept as given. Raises FileNotFoundError when a resolved path is missing
    so a bad path fails here rather than as a cryptic backend startup error.
    """
    base = models_dir()

    mp = cfg.get("model_path")
    if mp:
        resolved = _resolve(mp, base)
        cfg["model_path"] = resolved
        if not Path(resolved).exists():
            logger.error("model_path does not exist: %s (base=%s)", resolved, base)
            raise FileNotFoundError(f"model_path not found: {resolved}")

    if cfg.get("backend_target") == "llama_cpp_binary":
        server = cfg.get("llama_server_path") or llama_server_default()
        server = _resolve(server, base)
        cfg["llama_server_path"] = server
        if not Path(server).exists():
            logger.error("llama_server_path does not exist: %s", server)
            raise FileNotFoundError(f"llama_server_path not found: {server}")

    return cfg
