"""Circuit breaking and retries.

Both exist because of a specific failure shape this gateway has: when a tier is
down, *every* request still walks to it, waits for its timeout, and only then
escalates. With a 120 s Ollama timeout, one dead model turns every request into
a two-minute request. The cascade degrades correctly but agonisingly slowly.

A breaker converts that into a fast skip. After `failure_threshold` consecutive
failures the tier is marked open and skipped immediately, until a cooldown
elapses and one probe request is allowed through to test recovery.

**Retries are deliberately narrow.** Only timeouts and connection errors are
retried, only once by default, and only for tiers that have somewhere to
escalate to. Retrying a frontier call that already burned tokens spends money
twice for one answer, so the frontier is retried only on failures that could
not have been billed.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from app.providers.base import ProviderError, ProviderTimeout, ProviderUnavailable

logger = logging.getLogger("hlr.resilience")

State = Literal["closed", "open", "half_open"]


@dataclass
class BreakerStats:
    state: State = "closed"
    consecutive_failures: int = 0
    opened_at: float = 0.0
    trips: int = 0
    short_circuited: int = 0

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "consecutive_failures": self.consecutive_failures,
            "trips": self.trips,
            "short_circuited": self.short_circuited,
        }


class CircuitBreaker:
    """One breaker per tier.

    Counts *consecutive* failures rather than a rate: a tier that fails once
    an hour is not broken, and a rate window would need traffic to decay. Two
    failures in a row on a local model almost always means the model is gone.
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 3,
        cooldown_s: float = 30.0,
    ) -> None:
        self.name = name
        self._threshold = failure_threshold
        self._cooldown = cooldown_s
        self._stats = BreakerStats()

    @property
    def stats(self) -> BreakerStats:
        return self._stats

    def _now(self) -> float:
        return time.monotonic()

    def allows(self, *, now: float | None = None) -> bool:
        """Should a request be attempted against this tier?

        An open breaker lets exactly one probe through once the cooldown has
        elapsed. That probe is what closes the breaker if the tier recovered —
        without it, a breaker that opens never closes.
        """
        now = self._now() if now is None else now
        s = self._stats
        if s.state == "closed":
            return True
        if s.state == "open":
            if now - s.opened_at >= self._cooldown:
                s.state = "half_open"
                return True
            s.short_circuited += 1
            return False
        # half_open: a probe is already in flight; hold everything else back so
        # a recovering tier is not immediately flooded.
        s.short_circuited += 1
        return False

    def record_success(self) -> None:
        s = self._stats
        if s.state != "closed":
            logger.info("circuit %s closed after a successful probe", self.name)
        s.state = "closed"
        s.consecutive_failures = 0

    def record_failure(self, *, now: float | None = None) -> None:
        now = self._now() if now is None else now
        s = self._stats
        s.consecutive_failures += 1
        if s.state == "half_open":
            # The probe failed — back to open, cooldown restarts.
            s.state = "open"
            s.opened_at = now
            logger.warning("circuit %s re-opened: probe failed", self.name)
            return
        if s.consecutive_failures >= self._threshold and s.state == "closed":
            s.state = "open"
            s.opened_at = now
            s.trips += 1
            logger.warning(
                "circuit %s opened after %d consecutive failures; skipping for %.0fs",
                self.name, s.consecutive_failures, self._cooldown,
            )

    def reset(self) -> None:
        self._stats = BreakerStats()


class BreakerRegistry:
    """The breakers, one per tier, exposed for /healthz and /metrics."""

    def __init__(self, *, failure_threshold: int = 3, cooldown_s: float = 30.0) -> None:
        self._threshold = failure_threshold
        self._cooldown = cooldown_s
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name, failure_threshold=self._threshold, cooldown_s=self._cooldown
            )
        return self._breakers[name]

    def snapshot(self) -> dict[str, dict]:
        return {name: b.stats.as_dict() for name, b in self._breakers.items()}

    def open_circuits(self) -> list[str]:
        return [n for n, b in self._breakers.items() if b.stats.state == "open"]

    def reset(self) -> None:
        for b in self._breakers.values():
            b.reset()


@dataclass
class RetryPolicy:
    """Narrow on purpose.

    `retry_billable` is the load-bearing flag: a frontier call that timed out
    may still have been billed, so retrying it can pay twice for one answer.
    Connection errors are safe to retry because the request never landed.
    """

    attempts: int = 2
    base_delay_s: float = 0.25
    max_delay_s: float = 2.0
    retry_billable: bool = False
    jitter: bool = True

    def delay_for(self, attempt: int, *, rand: Callable[[], float] = random.random) -> float:
        """Exponential backoff with full jitter.

        Without jitter, N clients that fail together retry together and
        reproduce the thundering herd that caused the failure.
        """
        raw = min(self.base_delay_s * (2 ** (attempt - 1)), self.max_delay_s)
        return raw * rand() if self.jitter else raw

    def should_retry(self, exc: BaseException, *, billable: bool) -> bool:
        if isinstance(exc, ProviderUnavailable):
            # Never landed, or the model is missing — retrying a missing model
            # is pointless, but the caller distinguishes that via the breaker.
            return True
        if isinstance(exc, ProviderTimeout):
            # The request may have been processed. Only safe when nothing was
            # billed for it.
            return self.retry_billable or not billable
        return False


async def call_with_resilience[T](
    fn: Callable[[], Awaitable[T]],
    *,
    breaker: CircuitBreaker,
    policy: RetryPolicy,
    billable: bool = False,
) -> T:
    """Run `fn`, honouring the breaker and retrying narrowly.

    Raises `ProviderUnavailable` immediately when the breaker is open — that is
    the whole value: a dead tier costs microseconds instead of a full timeout.
    """
    if not breaker.allows():
        raise ProviderUnavailable(
            f"{breaker.name} circuit is open after repeated failures; skipping",
            fix=f"Check {breaker.name}; the circuit retries automatically after a cooldown.",
        )

    last: ProviderError | None = None
    for attempt in range(1, max(policy.attempts, 1) + 1):
        try:
            result = await fn()
            breaker.record_success()
            return result
        except ProviderError as exc:
            last = exc
            breaker.record_failure()
            if attempt >= policy.attempts or not policy.should_retry(exc, billable=billable):
                raise
            delay = policy.delay_for(attempt)
            logger.info(
                "%s attempt %d/%d failed (%s); retrying in %.2fs",
                breaker.name, attempt, policy.attempts, type(exc).__name__, delay,
            )
            await asyncio.sleep(delay)

    assert last is not None  # unreachable: the loop either returns or raises
    raise last
