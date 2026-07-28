"""Shared test configuration.

Set **before** any test module imports `mock_frontier.main`, which reads these
at import time. Without this the mock sleeps for its default 1800 ms per call
and the suite crawls — a fast test suite is a suite people actually run.
"""

from __future__ import annotations

import os

os.environ.setdefault("MOCK_LATENCY_MS", "0")
os.environ.setdefault("MOCK_JITTER_MS", "0")
os.environ.setdefault("MOCK_FAILURE_RATE", "0.0")
# Seeded so any test touching jitter or failure injection is deterministic.
os.environ.setdefault("MOCK_SEED", "1337")
