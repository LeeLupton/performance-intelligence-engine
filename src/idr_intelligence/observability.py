"""Structured, opt-in operational logging for the inference paths.

The library is silent by default (a NullHandler on the package logger in
``__init__``), so importing ``idr_intelligence`` never emits output and scoring
stays byte-for-byte deterministic on stdout. The CLI calls
:func:`configure_logging` to turn on JSON-lines logs to **stderr** — model
provenance on load, per-run counts and timing, finding summaries, drift flags,
evictions, and rejections. This is the operational signal a deployment needs
(shippable to any log aggregator as-is) without touching the finding contract:
logs never go to stdout, so the finding JSON a consumer parses is unchanged.
"""

from __future__ import annotations

import logging
import sys
from typing import IO, Any

PACKAGE_LOGGER = "idr_intelligence"
LEVELS = ("debug", "info", "warning", "error")


class JsonLineFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, event, then structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "idr_fields", None)
        if fields:
            payload.update(fields)
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "warning", stream: IO[str] | None = None) -> logging.Logger:
    """Route the package logger to a JSON-lines stderr handler at ``level``.

    Idempotent: replaces any handler this function previously installed, so
    repeated CLI invocations in one process do not stack handlers. The stream
    is bound at call time, which is what lets test capture (and a supervisor's
    redirected stderr) see the records.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown log level {level!r}; choose from {', '.join(LEVELS)}")
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(getattr(logging, level.upper()))
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonLineFormatter())
    logger.handlers = [handler]
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """A child logger under the package namespace (e.g. ``get_logger("cli")``)."""
    return logging.getLogger(f"{PACKAGE_LOGGER}.{name}")


def log_event(logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one structured record: a stable ``event`` name plus keyword fields."""
    logger.log(level, event, extra={"idr_fields": fields})
