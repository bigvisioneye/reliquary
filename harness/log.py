from __future__ import annotations

import logging
import sys
from typing import Literal

LogLevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class _FlushStreamHandler(logging.StreamHandler):
    """Emit log lines and flush immediately (works with nohup tail -f)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
            self.flush()
        except (ValueError, OSError):
            # pytest may close captured stdout between handler setup and emit.
            pass


def configure_harness_logging(level: LogLevelName = "INFO") -> logging.Logger:
    """Configure the ``harness`` logger tree for CLI progress output."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    root = logging.getLogger("harness")
    if not root.handlers:
        handler = _FlushStreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"harness.{name}")
