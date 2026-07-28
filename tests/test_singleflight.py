"""Tests for app/cache/singleflight.py.

The property that matters is *not* "the lock always works". It is that the lock
**never costs correctness or availability** — a waiter that times out proceeds,
a Redis failure proceeds, a crashed holder's key expires. Deduplication is an
efficiency measure and is allowed to fail open.
"""

from __future__ import annotations

import asyncio

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.cache.singleflight import LOCK_PREFIX, SingleFlight


class FakeRedis:
    def __init__(self, *, fail_on: set[str] | None = None):
        self.store: dict[str, str] = {}
        self.fail_on = fail_on or set()

    def _check(self, op):
        if op in self.fail_on:
            raise RedisConnectionError(f"simulated {op} failure")

    async def set(self, key, value, nx=False, px=None):
        self._check("set")
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def exists(self, key):
        self._check("exists")
        return 1 if key in self.store else 0

    async def eval(self, script, numkeys, key, token):
        self._check("eval")
        if self.store.get(key) == token:
            del self.store[key]
            return 1
        return 0


@pytest.fixture
def sf():
    return SingleFlight(FakeRedis(), ttl_ms=5000, wait_ms=200)


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------


async def test_first_caller_wins(sf):
    lease = await sf.acquire("abc")
    assert lease.acquired is True
    assert lease.token


async def test_second_caller_waits_and_is_told_to_recheck(sf):
    """The winner finishes quickly, so the waiter should look in the cache
    rather than repeat the work."""
    first = await sf.acquire("abc")

    async def finish_soon():
        await asyncio.sleep(0.06)
        await sf.release("abc", first.token)

    task = asyncio.create_task(finish_soon())
    second = await sf.acquire("abc")
    await task

    assert second.acquired is False
    assert second.should_recheck is True
    assert second.waited_ms > 0


async def test_release_frees_the_key_for_the_next_caller(sf):
    first = await sf.acquire("abc")
    await sf.release("abc", first.token)
    assert (await sf.acquire("abc")).acquired is True


async def test_different_digests_do_not_block_each_other(sf):
    await sf.acquire("aaa")
    assert (await sf.acquire("bbb")).acquired is True


# --------------------------------------------------------------------------
# failing open — the important half
# --------------------------------------------------------------------------


async def test_waiter_that_times_out_proceeds_anyway():
    """A duplicate model call costs milliseconds. A request wedged behind a
    lock whose holder died costs the gateway."""
    sf = SingleFlight(FakeRedis(), ttl_ms=5000, wait_ms=120)
    await sf.acquire("abc")  # holder never releases
    lease = await sf.acquire("abc")
    assert lease.acquired is False
    assert lease.should_recheck is False  # timed out — do the work
    assert lease.waited_ms >= 100


async def test_redis_failure_lets_the_request_through():
    """A lock we cannot take is not a reason to refuse the request."""
    sf = SingleFlight(FakeRedis(fail_on={"set"}))
    assert (await sf.acquire("abc")).acquired is True


async def test_no_redis_means_every_caller_is_a_winner():
    """Nowhere to coordinate, and the gateway still has to serve."""
    sf = SingleFlight(None)
    assert sf.enabled is False
    assert (await sf.acquire("abc")).acquired is True
    assert (await sf.acquire("abc")).acquired is True
    await sf.release("abc", "tok")  # must not raise


async def test_release_never_raises():
    sf = SingleFlight(FakeRedis(fail_on={"eval"}))
    lease = await sf.acquire("abc")
    await sf.release("abc", lease.token)  # TTL will clear it


async def test_release_with_no_token_is_a_noop(sf):
    await sf.release("abc", "")


# --------------------------------------------------------------------------
# lock hygiene
# --------------------------------------------------------------------------


async def test_release_only_deletes_our_own_lease(sf):
    """A lock that expired mid-flight may have been re-acquired by someone
    else; deleting theirs would defeat the whole purpose."""
    await sf.acquire("abc")
    await sf.release("abc", "someone-elses-token")
    assert (await sf.acquire("abc")).acquired is False  # the real holder still has it


async def test_lock_is_set_with_an_expiry(sf):
    """A crashed holder must not wedge a key forever."""
    seen = {}

    original = sf._redis.set

    async def spy(key, value, nx=False, px=None):
        seen["px"], seen["nx"] = px, nx
        return await original(key, value, nx=nx, px=px)

    sf._redis.set = spy
    await sf.acquire("abc")
    assert seen["nx"] is True
    assert seen["px"] == 5000


async def test_lock_key_is_namespaced(sf):
    await sf.acquire("abc")
    assert f"{LOCK_PREFIX}abc" in sf._redis.store


# --------------------------------------------------------------------------
# concurrency
# --------------------------------------------------------------------------


async def test_only_one_of_many_concurrent_callers_wins():
    """The whole point: ten identical requests, one model call.

    On an 8 GB machine ten concurrent 3B calls do not merely cost ten times —
    they contend for memory and become slower than one call would have been.
    """
    sf = SingleFlight(FakeRedis(), ttl_ms=5000, wait_ms=80)
    leases = await asyncio.gather(*(sf.acquire("same") for _ in range(10)))
    assert sum(1 for lease in leases if lease.acquired) == 1
