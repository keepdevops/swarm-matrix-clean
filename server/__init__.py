"""HTTP server exposing BackendManager over SSE for the React frontend."""
from server.api import create_app

__all__ = ["create_app"]
