"""Pydantic validation for BackendConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backends.base import BackendConfig, InferenceBackend


def test_minimal_config_accepted():
    cfg = BackendConfig(backend_target="fake")
    assert cfg.backend_target == "fake"
    assert cfg.max_context_tokens == 4096
    assert cfg.temperature == 0.2
    assert cfg.port == 8080


def test_temperature_bounds():
    with pytest.raises(ValidationError):
        BackendConfig(backend_target="x", temperature=-0.1)
    with pytest.raises(ValidationError):
        BackendConfig(backend_target="x", temperature=2.5)


def test_context_bounds():
    with pytest.raises(ValidationError):
        BackendConfig(backend_target="x", max_context_tokens=128)
    with pytest.raises(ValidationError):
        BackendConfig(backend_target="x", max_context_tokens=10_000_000)


def test_port_bounds():
    with pytest.raises(ValidationError):
        BackendConfig(backend_target="x", port=80)  # reserved
    with pytest.raises(ValidationError):
        BackendConfig(backend_target="x", port=70_000)


def test_path_validator_rejects_missing(tmp_path):
    missing = tmp_path / "nope.gguf"
    with pytest.raises(ValidationError):
        BackendConfig(backend_target="x", model_path=str(missing))


def test_path_validator_accepts_existing(tmp_path):
    p = tmp_path / "model.gguf"
    p.write_bytes(b"fake")
    cfg = BackendConfig(backend_target="x", model_path=str(p))
    assert cfg.model_path == str(p)


def test_empty_path_is_treated_as_none():
    cfg = BackendConfig(backend_target="x", model_path="")
    assert cfg.model_path == ""  # validator allows empty as sentinel


def test_parse_config_helper(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"x")
    cfg = InferenceBackend.parse_config({"backend_target": "fake", "model_path": str(p)})
    assert isinstance(cfg, BackendConfig)
    assert cfg.backend_target == "fake"


def test_dtype_default():
    cfg = BackendConfig(backend_target="x")
    assert cfg.dtype == "auto"
    assert cfg.tensor_parallel_size == 1
    assert cfg.vllm_extra_args == []


def test_gpu_memory_utilization_bounds():
    with pytest.raises(ValidationError):
        BackendConfig(backend_target="x", gpu_memory_utilization=0.99)
    with pytest.raises(ValidationError):
        BackendConfig(backend_target="x", gpu_memory_utilization=0.05)
