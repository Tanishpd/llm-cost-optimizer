"""Tests for app/cache/exact.py.

Unit tests use a fake Redis so they run with no server. Integration tests hit
the real one — a cache is exactly the kind of component where an in-memory fake
can pass while the real client fails on encoding or TTL semantics.
"""

from __future__ import annotations

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError

from app.cache.exact import BILLED_TIERS, CachedAnswer, ExactCache
from app.cache.keys import build_key

BASE = dict(model="qwen2.5:3b", temperature=0.0, top_p=1.0, max_tokens=512)


def make_answer(**overrides) -> CachedAnswer:
    return CachedAnswer(
        **{
            "content": "Paris",
            "model": "qwen2.5:3b",
            "tier": "local_large",
            "tokens_in": 41,
            "tokens_out": 2,
            "baseline_micro_usd": 123,
            **overrides,
        }
    )


class FakeRedis:
    """Just enough Redis. `fail_on` makes any operation raise, which is how the
    degradation paths get exercised."""

    def __init__(self, *, fail_on: set[str] | None = None):
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.fail_on = fail_on or set()

    def _check(self, op: str):
        if op in self.fail_on:
            raise RedisConnectionError(f"simulated {op} failure")

    async def get(self, key):
        self._check("get")
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self._check("set")
        self.store[key] = value
        if ex:
            self.ttls[key] = ex
        return True

    async def delete(self, *keys):
        self._check("delete")
        return sum(1 for k in keys if self.store.pop(k, None) is not None)

    async def ping(self):
        self._check("ping")
        return True

    async def scan_iter(self, match=None, count=None):
        self._check("scan")
        prefix = (match or "").rstrip("*")
        for k in list(self.store):
            if k.startswith(prefix):
                yield k


@pytest.fixture
def cache():
    return ExactCache(FakeRedis())


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------


async def test_set_then_get_round_trips(cache):
    key = build_key(prompt="capital?", **BASE)
    assert await cache.set(key, make_answer()) is True
    hit = await cache.get(key)
    assert hit is not None
    assert hit.content == "Paris"
    assert hit.tier == "local_large"
    assert hit.baseline_micro_usd == 123


async def test_miss_returns_none(cache):
    assert await cache.get(build_key(prompt="never stored", **BASE)) is None


async def test_different_params_do_not_share_an_entry(cache):
    """The correctness property the key exists for."""
    hot = build_key(prompt="q", **{**BASE, "temperature": 1.0})
    cold = build_key(prompt="q", **{**BASE, "temperature": 0.0})
    await cache.set(cold, make_answer(content="deterministic"))
    assert await cache.get(hot) is None


# --------------------------------------------------------------------------
# TTL — expensive answers earn more shelf space
# --------------------------------------------------------------------------


def test_frontier_answers_get_the_longer_ttl():
    c = ExactCache(FakeRedis(), ttl_local=3600, ttl_frontier=86400)
    assert c.ttl_for("frontier") == 86400
    assert c.ttl_for("local_small") == 3600
    assert c.ttl_for("local_large") == 3600
    assert c.ttl_for("cache_semantic") == 3600


def test_billed_tiers_is_the_single_source_of_truth():
    assert "frontier" in BILLED_TIERS
    assert "local_large" not in BILLED_TIERS


async def test_ttl_is_applied_on_write():
    fake = FakeRedis()
    c = ExactCache(fake, ttl_local=60, ttl_frontier=600)
    key = build_key(prompt="q", **BASE)
    await c.set(key, make_answer(tier="frontier"))
    assert list(fake.ttls.values()) == [600]


# --------------------------------------------------------------------------
# degradation — invariant 4: a cache must never take the request path down
# --------------------------------------------------------------------------


async def test_read_failure_is_a_miss_not_an_exception():
    c = ExactCache(FakeRedis(fail_on={"get"}))
    assert await c.get(build_key(prompt="q", **BASE)) is None
    assert c.available is False


async def test_write_failure_returns_false_not_an_exception():
    c = ExactCache(FakeRedis(fail_on={"set"}))
    assert await c.set(build_key(prompt="q", **BASE), make_answer()) is False
    assert c.available is False


async def test_no_redis_configured_is_a_silent_no_op():
    """Gateway must run with Redis absent entirely."""
    c = ExactCache(None)
    assert c.available is False
    assert await c.get(build_key(prompt="q", **BASE)) is None
    assert await c.set(build_key(prompt="q", **BASE), make_answer()) is False
    assert await c.flush() == 0
    assert await c.size() == 0
    ok, detail = await c.ping()
    assert ok is False and detail == "not configured"


async def test_ping_never_raises():
    ok, detail = await ExactCache(FakeRedis(fail_on={"ping"})).ping()
    assert ok is False
    assert "ConnectionError" in detail


async def test_flush_survives_a_mid_scan_failure():
    """Partial flush is acceptable; an exception on the request path is not."""
    c = ExactCache(FakeRedis(fail_on={"scan"}))
    assert await c.flush() == 0


# --------------------------------------------------------------------------
# corrupt entries are misses, never exceptions
# --------------------------------------------------------------------------


def test_unparseable_entry_is_discarded():
    assert CachedAnswer.from_json("{not json") is None


def test_valid_json_of_the_wrong_type_is_discarded():
    """`[]` and `"str"` parse fine but aren't mappings. An unguarded .items()
    on them raises AttributeError, which would escape the handler and crash
    the request path — the one thing a cache must never do."""
    assert CachedAnswer.from_json("[]") is None
    assert CachedAnswer.from_json('"a string"') is None
    assert CachedAnswer.from_json("42") is None
    assert CachedAnswer.from_json("null") is None


def test_entry_missing_required_fields_is_discarded():
    assert CachedAnswer.from_json('{"content": "only this"}') is None


def test_entry_from_an_older_schema_is_tolerated():
    """Extra keys from a future/past version must not crash the request path."""
    raw = make_answer().to_json().replace("{", '{"unknown_future_field": 1,', 1)
    parsed = CachedAnswer.from_json(raw)
    assert parsed is not None
    assert parsed.content == "Paris"


async def test_corrupt_stored_value_reads_as_a_miss():
    fake = FakeRedis()
    key = build_key(prompt="q", **BASE)
    from app.cache.keys import redis_key

    fake.store[redis_key(key.digest)] = "garbage"
    assert await ExactCache(fake).get(key) is None


# --------------------------------------------------------------------------
# flush is scoped
# --------------------------------------------------------------------------


async def test_flush_only_removes_exact_cache_keys():
    """Must not wipe telemetry, semantic vectors, or anything else sharing
    the database."""
    fake = FakeRedis()
    c = ExactCache(fake)
    await c.set(build_key(prompt="a", **BASE), make_answer())
    await c.set(build_key(prompt="b", **BASE), make_answer())
    fake.store["hlr:telemetry"] = "keep me"
    fake.store["hlr:sem:vec:1"] = "keep me too"

    assert await c.flush() == 2
    assert "hlr:telemetry" in fake.store
    assert "hlr:sem:vec:1" in fake.store


async def test_size_counts_only_exact_entries():
    fake = FakeRedis()
    c = ExactCache(fake)
    await c.set(build_key(prompt="a", **BASE), make_answer())
    fake.store["hlr:sem:vec:1"] = "x"
    assert await c.size() == 1


# --------------------------------------------------------------------------
# integration — real Redis
# --------------------------------------------------------------------------


@pytest.mark.integration
async def test_live_redis_round_trip_and_ttl():
    from redis.asyncio import Redis

    redis = Redis.from_url("redis://localhost:6379/0", decode_responses=True)
    c = ExactCache(redis, ttl_local=60, ttl_frontier=600)
    key = build_key(prompt="integration probe — exact cache", **BASE)
    try:
        assert await c.set(key, make_answer(tier="frontier")) is True
        hit = await c.get(key)
        assert hit is not None and hit.content == "Paris"

        from app.cache.keys import redis_key

        ttl = await redis.ttl(redis_key(key.digest))
        assert 0 < ttl <= 600  # frontier TTL applied by the real server

        ok, _ = await c.ping()
        assert ok is True
    finally:
        await c.delete(key)
        await redis.aclose()
