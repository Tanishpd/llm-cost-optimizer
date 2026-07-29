"""Tests for app/resilience.py.

The failure this addresses is specific: when a tier is down, every request
still walks to it, waits out its timeout, and only then escalates. With a 120 s
Ollama timeout one dead model turns every request into a two-minute request.
The cascade degrades correctly but agonisingly slowly.
"""

from __future__ import annotations

import pytest

from app.providers.base import ProviderTimeout, ProviderUnavailable
from app.resilience import (
    BreakerRegistry,
    CircuitBreaker,
    RetryPolicy,
    call_with_resilience,
)

# --------------------------------------------------------------------------
# breaker state machine
# --------------------------------------------------------------------------


def test_starts_closed():
    assert CircuitBreaker("t").allows() is True


def test_opens_only_after_consecutive_failures():
    """Counting consecutive rather than total: a tier that fails once an hour
    is not broken."""
    b = CircuitBreaker("t", failure_threshold=3)
    b.record_failure(now=0.0)
    b.record_failure(now=0.0)
    assert b.allows(now=0.0) is True   # 2 of 3
    b.record_failure(now=0.0)
    assert b.allows(now=0.0) is False  # tripped


def test_a_success_resets_the_count():
    b = CircuitBreaker("t", failure_threshold=3)
    b.record_failure()
    b.record_failure()
    b.record_success()
    b.record_failure()
    assert b.allows() is True  # the streak was broken, so 1 of 3


def test_open_circuit_short_circuits_until_cooldown():
    b = CircuitBreaker("t", failure_threshold=1, cooldown_s=30)
    b.record_failure(now=100.0)
    assert b.allows(now=105.0) is False
    assert b.allows(now=129.9) is False
    assert b.stats.short_circuited == 2


def test_cooldown_admits_exactly_one_probe():
    """Without a probe a breaker that opens never closes. Admitting *one*
    stops a recovering tier being immediately flooded."""
    b = CircuitBreaker("t", failure_threshold=1, cooldown_s=30)
    b.record_failure(now=100.0)
    assert b.allows(now=131.0) is True   # the probe
    assert b.stats.state == "half_open"
    assert b.allows(now=131.0) is False  # everything else waits


def test_successful_probe_closes_the_circuit():
    b = CircuitBreaker("t", failure_threshold=1, cooldown_s=30)
    b.record_failure(now=100.0)
    b.allows(now=131.0)
    b.record_success()
    assert b.stats.state == "closed"
    assert b.allows(now=131.0) is True


def test_failed_probe_reopens_and_restarts_the_cooldown():
    b = CircuitBreaker("t", failure_threshold=1, cooldown_s=30)
    b.record_failure(now=100.0)
    b.allows(now=131.0)          # probe admitted
    b.record_failure(now=131.0)  # probe failed
    assert b.stats.state == "open"
    assert b.allows(now=140.0) is False   # cooldown restarted from 131
    assert b.allows(now=162.0) is True


def test_trips_are_counted_for_observability():
    b = CircuitBreaker("t", failure_threshold=1, cooldown_s=0)
    b.record_failure(now=0.0)
    assert b.stats.trips == 1
    assert b.stats.as_dict()["state"] == "open"


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def test_registry_isolates_tiers():
    """One dead tier must not short-circuit the others."""
    reg = BreakerRegistry(failure_threshold=1)
    reg.get("frontier").record_failure()
    assert reg.get("frontier").allows() is False
    assert reg.get("local_small").allows() is True


def test_registry_reports_open_circuits():
    reg = BreakerRegistry(failure_threshold=1)
    reg.get("frontier").record_failure()
    assert reg.open_circuits() == ["frontier"]
    assert "frontier" in reg.snapshot()


# --------------------------------------------------------------------------
# retry policy — narrow on purpose
# --------------------------------------------------------------------------


def test_connection_errors_are_retried():
    assert RetryPolicy().should_retry(ProviderUnavailable("refused"), billable=False) is True


def test_billable_timeouts_are_not_retried():
    """A frontier call that timed out may still have been billed — retrying
    can pay twice for one answer."""
    policy = RetryPolicy(retry_billable=False)
    assert policy.should_retry(ProviderTimeout("slow"), billable=True) is False
    assert policy.should_retry(ProviderTimeout("slow"), billable=False) is True


def test_billable_timeouts_can_be_opted_into():
    assert RetryPolicy(retry_billable=True).should_retry(
        ProviderTimeout("slow"), billable=True
    ) is True


def test_unknown_errors_are_not_retried():
    assert RetryPolicy().should_retry(ValueError("bug"), billable=False) is False


def test_backoff_is_exponential_and_capped():
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=4.0, jitter=False)
    assert policy.delay_for(1) == 1.0
    assert policy.delay_for(2) == 2.0
    assert policy.delay_for(3) == 4.0
    assert policy.delay_for(9) == 4.0  # capped


def test_jitter_spreads_retries():
    """Without it, N clients that failed together retry together and reproduce
    the herd that caused the failure."""
    policy = RetryPolicy(base_delay_s=1.0, jitter=True)
    delays = {policy.delay_for(1) for _ in range(50)}
    assert len(delays) > 1
    assert all(0 <= d <= 1.0 for d in delays)


# --------------------------------------------------------------------------
# call_with_resilience
# --------------------------------------------------------------------------


async def test_success_passes_through():
    breaker = CircuitBreaker("t")
    result = await call_with_resilience(
        lambda: _returns("ok"), breaker=breaker, policy=RetryPolicy(attempts=1)
    )
    assert result == "ok"
    assert breaker.stats.state == "closed"


async def _returns(value):
    return value


async def test_open_breaker_fails_fast_without_calling():
    """The whole point: a dead tier costs microseconds, not a full timeout."""
    breaker = CircuitBreaker("frontier", failure_threshold=1, cooldown_s=999)
    breaker.record_failure()
    calls = []

    async def never():
        calls.append(1)
        return "should not happen"

    with pytest.raises(ProviderUnavailable, match="circuit is open"):
        await call_with_resilience(never, breaker=breaker, policy=RetryPolicy())
    assert calls == []


async def test_retry_then_succeed():
    attempts = []

    async def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise ProviderUnavailable("connection refused")
        return "recovered"

    result = await call_with_resilience(
        flaky,
        breaker=CircuitBreaker("t", failure_threshold=10),
        policy=RetryPolicy(attempts=2, base_delay_s=0.001),
    )
    assert result == "recovered"
    assert len(attempts) == 2


async def test_non_retryable_failure_raises_immediately():
    attempts = []

    async def billable_timeout():
        attempts.append(1)
        raise ProviderTimeout("slow")

    with pytest.raises(ProviderTimeout):
        await call_with_resilience(
            billable_timeout,
            breaker=CircuitBreaker("frontier", failure_threshold=10),
            policy=RetryPolicy(attempts=3, base_delay_s=0.001),
            billable=True,
        )
    assert len(attempts) == 1, "a billable timeout was retried"


async def test_exhausted_retries_raise_the_last_error():
    async def always_fails():
        raise ProviderUnavailable("still down")

    with pytest.raises(ProviderUnavailable, match="still down"):
        await call_with_resilience(
            always_fails,
            breaker=CircuitBreaker("t", failure_threshold=10),
            policy=RetryPolicy(attempts=2, base_delay_s=0.001),
        )


async def test_repeated_failures_eventually_open_the_breaker():
    breaker = CircuitBreaker("t", failure_threshold=2, cooldown_s=999)

    async def always_fails():
        raise ProviderUnavailable("down")

    for _ in range(2):
        with pytest.raises(ProviderUnavailable):
            await call_with_resilience(
                always_fails, breaker=breaker, policy=RetryPolicy(attempts=1)
            )
    assert breaker.stats.state == "open"
    # And now it short-circuits rather than calling again.
    with pytest.raises(ProviderUnavailable, match="circuit is open"):
        await call_with_resilience(
            always_fails, breaker=breaker, policy=RetryPolicy(attempts=1)
        )


def test_an_open_circuit_reports_what_broke_it():
    """"Circuit is open" alone says something failed repeatedly, not what.

    A 401 from a misconfigured key and a genuinely dead upstream look
    identical at the point an operator reads the error — and the message that
    named the cause was the one being discarded. This cost real debugging
    time: clearing FRONTIER_API_KEY made the local mock return 401, the
    breaker opened, and the surfaced advice was "check frontier" rather than
    "check FRONTIER_API_KEY", which the provider had already said.
    """
    import pytest

    from app.providers.base import ProviderUnavailable
    from app.resilience import CircuitBreaker

    breaker = CircuitBreaker("frontier", failure_threshold=2, cooldown_s=60)
    original = ProviderUnavailable("frontier rejected the API key (401)",
                                   fix="Check FRONTIER_API_KEY")
    for _ in range(2):
        breaker.record_failure(error=original)

    assert breaker.allows() is False
    assert breaker.stats.last_error.startswith("frontier rejected the API key")
    assert breaker.stats.last_fix == "Check FRONTIER_API_KEY"

    async def never_called():  # pragma: no cover - the breaker short-circuits
        raise AssertionError("the open circuit should not have called through")

    from app.resilience import RetryPolicy, call_with_resilience

    async def run():
        return await call_with_resilience(
            never_called, breaker=breaker, policy=RetryPolicy(attempts=1))

    import asyncio
    with pytest.raises(ProviderUnavailable) as exc:
        asyncio.run(run())
    assert "401" in str(exc.value), "the cause must survive the breaker"
    assert exc.value.fix == "Check FRONTIER_API_KEY", "and so must the remedy"


def test_a_breaker_with_no_recorded_cause_still_advises_something():
    """Older call sites may not pass the error; the generic advice remains."""
    import asyncio

    import pytest

    from app.providers.base import ProviderUnavailable
    from app.resilience import CircuitBreaker, RetryPolicy, call_with_resilience

    breaker = CircuitBreaker("ollama", failure_threshold=1, cooldown_s=60)
    breaker.record_failure()
    assert breaker.allows() is False

    async def fn():  # pragma: no cover - short-circuited
        return None

    with pytest.raises(ProviderUnavailable) as exc:
        asyncio.run(call_with_resilience(fn, breaker=breaker, policy=RetryPolicy(attempts=1)))
    assert "circuit is open" in str(exc.value)
    assert "retries automatically" in exc.value.fix
