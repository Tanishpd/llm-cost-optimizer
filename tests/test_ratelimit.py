"""Tests for app/ratelimit.py — the shared-state limiter.

The bug this exists to fix is silent: with N workers each enforcing its own
window, a configured 60/min becomes 60N and nothing reports it. So the tests
care most about (a) the window really being shared, and (b) the fallback being
loud rather than silently permissive.
"""

from __future__ import annotations

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.ratelimit import KEY_PREFIX, RedisRateLimiter
from app.security import RateLimiter

pytestmark = pytest.mark.asyncio


class FakeRedis:
    """Enough of a sorted set to run the sliding-window logic in Python,
    standing in for the Lua script."""

    def __init__(self, *, fail: bool = False):
        self.zsets: dict[str, dict[str, float]] = {}
        self.fail = fail
        self.calls = 0

    def register_script(self, _lua):
        async def run(keys, args):
            self.calls += 1
            if self.fail:
                raise RedisConnectionError("simulated failure")
            key = keys[0]
            now, window, limit, member = float(args[0]), float(args[1]), int(args[2]), args[3]
            bucket = self.zsets.setdefault(key, {})
            for m, score in list(bucket.items()):
                if score <= now - window:
                    del bucket[m]
            used = len(bucket)
            if used >= limit:
                oldest = min(bucket.values()) if bucket else now
                retry_ms = max(0, int((oldest + window - now) * 1000))
                return [0, 0, retry_ms]
            bucket[member] = now
            return [1, limit - used - 1, 0]

        return run

    async def delete(self, key):
        self.zsets.pop(key, None)
        return 1


async def test_requests_under_the_limit_pass():
    rl = RedisRateLimiter(FakeRedis(), limit=3)
    for _ in range(3):
        allowed, _, _ = await rl.check("alice")
        assert allowed is True


async def test_the_request_over_the_limit_is_blocked():
    rl = RedisRateLimiter(FakeRedis(), limit=3)
    for _ in range(3):
        await rl.check("alice")
    allowed, remaining, retry_after = await rl.check("alice")
    assert allowed is False
    assert remaining == 0
    assert retry_after > 0


async def test_state_is_shared_not_per_instance():
    """The whole point: two limiter objects — standing in for two worker
    processes — must consume one budget, not one each."""
    shared = FakeRedis()
    worker_a = RedisRateLimiter(shared, limit=4)
    worker_b = RedisRateLimiter(shared, limit=4)

    assert (await worker_a.check("alice"))[0] is True
    assert (await worker_b.check("alice"))[0] is True
    assert (await worker_a.check("alice"))[0] is True
    assert (await worker_b.check("alice"))[0] is True
    # Budget of 4 is now spent, regardless of which "worker" asks.
    assert (await worker_a.check("alice"))[0] is False
    assert (await worker_b.check("alice"))[0] is False


async def test_principals_have_independent_budgets():
    rl = RedisRateLimiter(FakeRedis(), limit=2)
    await rl.check("alice")
    await rl.check("alice")
    assert (await rl.check("alice"))[0] is False
    assert (await rl.check("bob"))[0] is True


async def test_remaining_counts_down():
    rl = RedisRateLimiter(FakeRedis(), limit=3)
    assert (await rl.check("alice"))[1] == 2
    assert (await rl.check("alice"))[1] == 1
    assert (await rl.check("alice"))[1] == 0


async def test_zero_limit_disables_and_never_touches_redis():
    redis = FakeRedis()
    rl = RedisRateLimiter(redis, limit=0)
    assert rl.enabled is False
    for _ in range(50):
        assert (await rl.check("alice"))[0] is True
    assert redis.calls == 0


# --------------------------------------------------------------------------
# degradation — fall back, but never fail open silently
# --------------------------------------------------------------------------


async def test_redis_failure_falls_back_to_local_enforcement():
    """A limit that multiplies by worker count still beats no limit at all."""
    rl = RedisRateLimiter(FakeRedis(fail=True), limit=2,
                          fallback=RateLimiter(limit=2, window_s=60))
    assert (await rl.check("alice"))[0] is True
    assert (await rl.check("alice"))[0] is True
    assert (await rl.check("alice"))[0] is False  # still enforced, just locally


async def test_degraded_state_is_visible():
    rl = RedisRateLimiter(FakeRedis(fail=True), limit=2)
    assert rl.distributed is True   # believed shared until proven otherwise
    await rl.check("alice")
    assert rl.distributed is False  # now known to be per-process


async def test_recovery_is_detected():
    redis = FakeRedis(fail=True)
    rl = RedisRateLimiter(redis, limit=5)
    await rl.check("alice")
    assert rl.distributed is False
    redis.fail = False
    await rl.check("alice")
    assert rl.distributed is True


async def test_no_redis_uses_the_local_limiter():
    rl = RedisRateLimiter(None, limit=2, fallback=RateLimiter(limit=2, window_s=60))
    assert rl.distributed is False
    assert (await rl.check("alice"))[0] is True
    assert (await rl.check("alice"))[0] is True
    assert (await rl.check("alice"))[0] is False


async def test_key_is_namespaced():
    redis = FakeRedis()
    await RedisRateLimiter(redis, limit=5).check("alice")
    assert f"{KEY_PREFIX}alice" in redis.zsets


async def test_reset_clears_a_principal():
    redis = FakeRedis()
    rl = RedisRateLimiter(redis, limit=1)
    await rl.check("alice")
    assert (await rl.check("alice"))[0] is False
    await rl.reset("alice")
    assert (await rl.check("alice"))[0] is True


# --------------------------------------------------------------------------
# integration — the real Lua script against a real Redis
# --------------------------------------------------------------------------


@pytest.mark.integration
async def test_live_redis_sliding_window():
    """The Lua path is never exercised by the fake, so it needs its own test —
    a script that errors would silently fall back to per-process limiting."""
    import uuid

    from redis.asyncio import Redis

    redis = Redis.from_url("redis://localhost:6379/15", decode_responses=True)
    principal = f"test-{uuid.uuid4().hex[:8]}"
    rl = RedisRateLimiter(redis, limit=3, window_s=60)
    try:
        for _ in range(3):
            assert (await rl.check(principal))[0] is True
        allowed, remaining, retry_after = await rl.check(principal)
        assert allowed is False
        assert remaining == 0
        assert 0 < retry_after <= 60
        assert rl.distributed is True, "fell back to local — the Lua script failed"
    finally:
        await rl.reset(principal)
        await redis.aclose()


@pytest.mark.integration
async def test_live_redis_shares_state_between_instances():
    """Two limiter objects on one Redis = two workers on one deployment."""
    import uuid

    from redis.asyncio import Redis

    redis = Redis.from_url("redis://localhost:6379/15", decode_responses=True)
    principal = f"test-{uuid.uuid4().hex[:8]}"
    a = RedisRateLimiter(redis, limit=2, window_s=60)
    b = RedisRateLimiter(redis, limit=2, window_s=60)
    try:
        assert (await a.check(principal))[0] is True
        assert (await b.check(principal))[0] is True
        assert (await a.check(principal))[0] is False
        assert (await b.check(principal))[0] is False
    finally:
        await a.reset(principal)
        await redis.aclose()
