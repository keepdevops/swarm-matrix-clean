"""Run with: python -m server  (port 8765 by default)."""
from __future__ import annotations

import logging
import os

import uvicorn

from server.api import create_app


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )
    port = int(os.environ.get("PORT", "8765"))
    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
