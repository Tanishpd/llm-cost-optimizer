"""Structured logging.

Plain-text logs are fine to read and useless to query. On a gateway whose whole
product is visibility into its decisions, "which requests escalated twice last
Tuesday" should be a filter, not a grep.

JSON is opt-in via `LOG_FORMAT=json` so local development keeps human-readable
output. The trace id is attached automatically from a context variable, so a log
line emitted deep in the cascade carries the same id the caller received in
`X-HLR-Trace-Id` — without every function having to thread it through.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from typing import Any

# Set once per request in the middleware; read by the formatter. A ContextVar
# is the right tool here because it follows the async task, so concurrent
# requests cannot read each other's id.
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")

# Fields already on every LogRecord; anything else the caller passed via
# `extra=` is worth emitting.
_STANDARD = frozenset(
    """args asctime created exc_info exc_text filename funcName levelname levelno
    lineno module msecs message msg name pathname process processName relativeCreated
    stack_info thread threadName taskName""".split()
)


class JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        trace_id = trace_id_var.get()
        if trace_id:
            payload["trace_id"] = trace_id

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Anything passed as `extra=` — the structured half.
        for key, value in record.__dict__.items():
            if key not in _STANDARD and not key.startswith("_"):
                payload[key] = value

        # default=str so an un-serializable value degrades to its repr rather
        # than throwing inside the logger and losing the line entirely.
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable, with the trace id when there is one."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        trace_id = trace_id_var.get()
        return f"{base}  [trace={trace_id}]" if trace_id else base


def configure(*, level: str = "INFO", fmt: str = "text") -> None:
    """Install a handler on the `hlr` logger tree.

    Scoped to `hlr` rather than the root logger: uvicorn configures its own
    loggers, and taking over the root would either duplicate its output or
    fight it. `propagate = False` is what stops the duplication.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if fmt.lower() == "json"
        else TextFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )

    root = logging.getLogger("hlr")
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False


def set_trace_id(trace_id: str) -> None:
    trace_id_var.set(trace_id)
