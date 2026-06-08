"""Tests for env-configurable model/binary path resolution."""

from __future__ import annotations

import pytest

from orchestration import paths


def test_relative_model_path_resolves_against_models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_MODELS_DIR, str(tmp_path))
    (tmp_path / "GGUF").mkdir()
    model = tmp_path / "GGUF" / "m.gguf"
    model.write_text("x")

    cfg = paths.resolve_paths({"backend_target": "mlx", "model_path": "GGUF/m.gguf"})
    assert cfg["model_path"] == str(model)


def test_absolute_model_path_passes_through(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_MODELS_DIR, "/somewhere/else")
    model = tmp_path / "m.gguf"
    model.write_text("x")

    cfg = paths.resolve_paths({"backend_target": "mlx", "model_path": str(model)})
    assert cfg["model_path"] == str(model)


def test_missing_model_path_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_MODELS_DIR, str(tmp_path))
    with pytest.raises(FileNotFoundError, match="model_path not found"):
        paths.resolve_paths({"backend_target": "mlx", "model_path": "nope.gguf"})


def test_llama_server_resolved_from_env_default(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_MODELS_DIR, str(tmp_path))
    model = tmp_path / "m.gguf"
    model.write_text("x")
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n")
    monkeypatch.setenv(paths.ENV_LLAMA_SERVER, str(server))

    cfg = paths.resolve_paths(
        {"backend_target": "llama_cpp_binary", "model_path": "m.gguf"}
    )
    assert cfg["llama_server_path"] == str(server)


def test_missing_llama_server_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_MODELS_DIR, str(tmp_path))
    model = tmp_path / "m.gguf"
    model.write_text("x")
    monkeypatch.setenv(paths.ENV_LLAMA_SERVER, str(tmp_path / "absent"))

    with pytest.raises(FileNotFoundError, match="llama_server_path not found"):
        paths.resolve_paths(
            {"backend_target": "llama_cpp_binary", "model_path": "m.gguf"}
        )


def test_no_model_path_is_noop():
    cfg = paths.resolve_paths({"backend_target": "fake"})
    assert "model_path" not in cfg
