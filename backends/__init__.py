"""Inference backend adapters. Each backend implements InferenceBackend."""

from backends.base import BackendConfig, InferenceBackend

__all__ = ["InferenceBackend", "BackendConfig"]
