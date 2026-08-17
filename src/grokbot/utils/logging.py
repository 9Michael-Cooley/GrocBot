"""Logging setup.

Structured output when GROKBOT_LOG_FORMAT=json (what the cluster ran), human
output otherwise. Request ids come from a contextvar so handlers don't have to
thread them through every call.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import time

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_COLORS = {
    logging.DEBUG: "\033[38;5;244m",
    logging.INFO: "\033[38;5;39m",
    logging.WARNING: "\033[38;5;214m",
    logging.ERROR: "\033[38;5;203m",
    logging.CRITICAL: "\033[48;5;203;38;5;231m",
}
_RESET = "\033[0m"


def set_request_id(rid: str) -> None:
    _request_id.set(rid)


def get_request_id() -> str:
    return _request_id.get()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 6),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": _request_id.get(),
        }
        for key, val in getattr(record, "extra_fields", {}).items():
            payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"), default=str)


class HumanFormatter(logging.Formatter):
    def __init__(self, color: bool = True):
        super().__init__()
        self.color = color
        self.start = time.time()

    def format(self, record: logging.LogRecord) -> str:
        elapsed = record.created - self.start
        rid = _request_id.get()
        prefix = f"[{elapsed:8.3f}s] {record.levelname:<7} {record.name:<24}"
        if rid != "-":
            prefix += f" ({rid})"
        line = f"{prefix}  {record.getMessage()}"
        if self.color:
            line = f"{_COLORS.get(record.levelno, '')}{line}{_RESET}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


_configured = False


def configure(level: str = "info", fmt: str | None = None) -> None:
    """Install the handler and set the level.

    Re-entrant on purpose. Module-level `log = get_logger(__name__)` runs at
    import and auto-configures at the default level, so by the time main()
    parses --log-level the logger already exists. Returning early here made
    --log-level and telemetry.log_level silently do nothing.
    """
    global _configured
    root = logging.getLogger("grokbot")

    if _configured:
        root.setLevel(_LEVELS.get(level.lower(), logging.INFO))
        return

    fmt = fmt or os.environ.get("GROKBOT_LOG_FORMAT", "human")
    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(HumanFormatter(color=sys.stderr.isatty()))
    root.handlers[:] = [handler]
    root.setLevel(_LEVELS.get(level.lower(), logging.INFO))
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    if not _configured:
        configure(os.environ.get("GROKBOT_LOG_LEVEL", "info"))
    return logging.getLogger(name if name.startswith("grokbot") else f"grokbot.{name}")
