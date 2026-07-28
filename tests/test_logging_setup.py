"""Tests for app/logging_setup.py.

Plain-text logs are fine to read and useless to query. On a gateway whose whole
product is visibility into its decisions, "which requests escalated twice last
Tuesday" should be a filter, not a grep.
"""

from __future__ import annotations

import json
import logging

from app.logging_setup import JsonFormatter, configure, set_trace_id, trace_id_var


def make_record(msg="hello", **extra) -> logging.LogRecord:
    record = logging.LogRecord("hlr.test", logging.INFO, __file__, 1, msg, None, None)
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_output_is_one_json_object_per_line():
    payload = json.loads(JsonFormatter().format(make_record()))
    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "hlr.test"
    assert "ts" in payload


def test_extra_fields_are_emitted():
    """The structured half — a grep can't filter on `tier`, a query can."""
    payload = json.loads(JsonFormatter().format(
        make_record(tier="local_large", confidence=0.62, escalated=True)
    ))
    assert payload["tier"] == "local_large"
    assert payload["confidence"] == 0.62
    assert payload["escalated"] is True


def test_trace_id_is_attached_from_context():
    """A line emitted deep in the cascade carries the same id the caller got
    back, without every function threading it through."""
    token = trace_id_var.set("abc123")
    try:
        assert json.loads(JsonFormatter().format(make_record()))["trace_id"] == "abc123"
    finally:
        trace_id_var.reset(token)


def test_no_trace_id_when_unset():
    token = trace_id_var.set("")
    try:
        assert "trace_id" not in json.loads(JsonFormatter().format(make_record()))
    finally:
        trace_id_var.reset(token)


def test_unserializable_values_do_not_lose_the_line():
    """Throwing inside the logger would discard the very line you needed."""
    payload = json.loads(JsonFormatter().format(make_record(obj=object())))
    assert "obj" in payload
    assert isinstance(payload["obj"], str)


def test_exceptions_are_included():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = make_record("failed")
        record.exc_info = sys.exc_info()
        payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exception"]


def test_configure_installs_exactly_one_handler():
    """Called twice — on reload, or in tests — it must not duplicate output."""
    configure(level="INFO", fmt="json")
    configure(level="INFO", fmt="json")
    assert len(logging.getLogger("hlr").handlers) == 1


def test_configure_does_not_propagate_to_root():
    """uvicorn owns the root logger; propagating would duplicate every line."""
    configure(level="INFO", fmt="text")
    assert logging.getLogger("hlr").propagate is False


def test_level_is_respected():
    configure(level="WARNING", fmt="text")
    assert logging.getLogger("hlr").level == logging.WARNING
    configure(level="INFO", fmt="text")  # restore


def test_set_trace_id_helper():
    set_trace_id("xyz")
    assert trace_id_var.get() == "xyz"
    set_trace_id("")
