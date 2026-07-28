"""Tests for app/telemetry.py.

The centrepiece is the reconciliation identities from docs/API.md. Those exist
so the headline savings figure can be checked rather than trusted, and this is
where that check lives.
"""

from __future__ import annotations

import time

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.telemetry import RequestRecord, Telemetry

PRICE_IN, PRICE_OUT = 2.50, 10.00


class FakeRedis:
    """Hash/stream subset, with a pipeline that mirrors redis-py's API."""

    def __init__(self, *, fail_on: set[str] | None = None):
        self.hashes: dict[str, dict[str, int]] = {}
        self.stream: list[tuple[str, dict[str, str]]] = []
        self.fail_on = fail_on or set()
        self._seq = 0

    def _check(self, op):
        if op in self.fail_on:
            raise RedisConnectionError(f"simulated {op} failure")

    def pipeline(self, transaction=False):
        return FakePipeline(self)

    async def hgetall(self, key):
        self._check("hgetall")
        return {k: str(v) for k, v in self.hashes.get(key, {}).items()}

    async def xrevrange(self, key, count=None):
        self._check("xrevrange")
        return list(reversed(self.stream))[:count]

    async def delete(self, *keys):
        self._check("delete")
        n = 0
        for k in keys:
            if self.hashes.pop(k, None) is not None:
                n += 1
            elif k == "hlr:telemetry" and self.stream:
                self.stream.clear()
                n += 1
        return n


class FakePipeline:
    def __init__(self, redis: FakeRedis):
        self.r = redis
        self.ops: list = []

    def xadd(self, key, fields, maxlen=None, approximate=True):
        self.ops.append(("xadd", key, fields))
        return self

    def hincrby(self, key, field, amount):
        self.ops.append(("hincrby", key, field, amount))
        return self

    async def execute(self):
        self.r._check("execute")
        for op in self.ops:
            if op[0] == "xadd":
                self.r._seq += 1
                self.r.stream.append((f"{self.r._seq}-0", op[2]))
            else:
                _, key, field, amount = op
                self.r.hashes.setdefault(key, {})
                self.r.hashes[key][field] = self.r.hashes[key].get(field, 0) + amount
        return [True] * len(self.ops)


def record(**overrides) -> RequestRecord:
    base = dict(
        trace_id="t1", tier="local_large", path=["local_large"], reason="test",
        latency_ms={"total": 10.0}, tokens_in=41, tokens_out=156,
        cost_micro_usd=0, baseline_micro_usd=1663, created_at=time.time(),
    )
    base.update(overrides)
    return RequestRecord(**base)  # type: ignore[arg-type]


@pytest.fixture
def telemetry():
    return Telemetry(FakeRedis())


# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------


async def test_record_writes_stream_and_counters(telemetry):
    assert await telemetry.record(record()) is True
    m = await telemetry.metrics()
    assert m["requests"] == 1
    assert m["tiers"]["served"]["local_large"] == 1
    assert len(await telemetry.recent()) == 1


async def test_served_counts_every_tier_separately(telemetry):
    for tier in ("cache_exact", "cache_exact", "local_small", "frontier"):
        await telemetry.record(record(tier=tier, path=[tier]))
    served = (await telemetry.metrics())["tiers"]["served"]
    assert served["cache_exact"] == 2
    assert served["local_small"] == 1
    assert served["frontier"] == 1
    assert served["local_large"] == 0


async def test_attempted_counts_the_whole_path_not_just_the_winner(telemetry):
    """A request that walked 0.5B -> 3B -> frontier paid for all three."""
    await telemetry.record(
        record(tier="frontier", path=["local_small", "local_large", "frontier"])
    )
    m = await telemetry.metrics()
    assert m["tiers"]["attempted"] == {"local_small": 1, "local_large": 1, "frontier": 1}
    assert m["tiers"]["served"]["frontier"] == 1
    assert m["tiers"]["served"]["local_small"] == 0


async def test_cache_tiers_are_not_counted_as_model_attempts(telemetry):
    """A cache hit attempts no model."""
    await telemetry.record(record(tier="cache_exact", path=["cache_exact"], cache_hit="exact"))
    assert (await telemetry.metrics())["tiers"]["attempted"] == {
        "local_small": 0, "local_large": 0, "frontier": 0
    }


async def test_gate_verdicts_are_counted(telemetry):
    await telemetry.record(record(gates=[
        {"gate": "small", "verdict": "escalate"},
        {"gate": "large", "verdict": "accept"},
    ]))
    g = (await telemetry.metrics())["gates"]
    assert g["small"] == {"evaluated": 1, "accepted": 0, "escalated": 1, "escalation_rate": 1.0}
    assert g["large"] == {"evaluated": 1, "accepted": 1, "escalated": 0, "escalation_rate": 0.0}


async def test_cold_loads_tracked_apart_from_latency(telemetry):
    await telemetry.record(record(cold_load_ms=1957.6))
    await telemetry.record(record(cold_load_ms=0.0))
    loads = (await telemetry.metrics())["model_loads"]
    assert loads["cold_loads"] == 1
    assert loads["mean_load_ms"] == pytest.approx(1957.0, abs=1.0)


async def test_guard_rejections_are_counted(telemetry):
    await telemetry.record(record(guard_rejected="entity_mismatch"))
    await telemetry.record(record(guard_rejected="entity_mismatch"))
    await telemetry.record(record(guard_rejected="number_mismatch"))
    rejections = (await telemetry.metrics())["cache"]["guard_rejections"]
    assert rejections == {"entity_mismatch": 2, "number_mismatch": 1}


# --------------------------------------------------------------------------
# reconciliation identities — docs/API.md
# --------------------------------------------------------------------------


async def test_served_counts_sum_to_requests(telemetry):
    for tier in ("cache_exact", "local_small", "local_large", "frontier"):
        await telemetry.record(record(tier=tier, path=[tier]))
    m = await telemetry.metrics()
    assert sum(m["tiers"]["served"].values()) == m["requests"] == 4


async def test_gate_accepted_plus_escalated_equals_evaluated(telemetry):
    for verdict in ("accept", "accept", "escalate"):
        await telemetry.record(record(gates=[{"gate": "small", "verdict": verdict}]))
    g = (await telemetry.metrics())["gates"]["small"]
    assert g["accepted"] + g["escalated"] == g["evaluated"] == 3


async def test_saved_equals_baseline_minus_actual(telemetry):
    await telemetry.record(record(tier="frontier", cost_micro_usd=1663, baseline_micro_usd=1663))
    await telemetry.record(record(tier="local_large", cost_micro_usd=0, baseline_micro_usd=1663))
    c = (await telemetry.metrics())["cost"]
    assert c["saved_usd"] == pytest.approx(c["baseline_usd"] - c["actual_usd"], abs=1e-9)


async def test_actual_cost_derives_from_frontier_tokens_only(telemetry):
    """The billed/baseline/throughput split: only the frontier request
    contributes frontier tokens."""
    await telemetry.record(record(tier="frontier", tokens_in=100, tokens_out=200,
                                  cost_micro_usd=2250, baseline_micro_usd=2250))
    await telemetry.record(record(tier="local_large", tokens_in=50, tokens_out=60,
                                  cost_micro_usd=0, baseline_micro_usd=725))
    t = (await telemetry.metrics())["tokens"]
    assert (t["frontier_in"], t["frontier_out"]) == (100, 200)
    assert (t["baseline_in"], t["baseline_out"]) == (150, 260)
    # baseline computed from baseline tokens at the documented rates
    expected = 150 / 1e6 * PRICE_IN + 260 / 1e6 * PRICE_OUT
    assert (await telemetry.metrics())["cost"]["baseline_usd"] == pytest.approx(expected, abs=1e-6)


async def test_saved_pct_matches_the_ratio(telemetry):
    await telemetry.record(record(cost_micro_usd=0, baseline_micro_usd=1_000_000))
    await telemetry.record(record(tier="frontier", cost_micro_usd=500_000,
                                  baseline_micro_usd=500_000))
    c = (await telemetry.metrics())["cost"]
    assert c["saved_pct"] == pytest.approx(c["saved_usd"] / c["baseline_usd"] * 100, abs=0.1)


async def test_hit_rate_is_hits_over_requests(telemetry):
    await telemetry.record(record(tier="cache_exact", path=["cache_exact"], cache_hit="exact"))
    await telemetry.record(record(tier="cache_semantic", path=["cache_semantic"],
                                  cache_hit="semantic"))
    await telemetry.record(record())
    cache = (await telemetry.metrics())["cache"]
    assert cache["hit_rate"] == pytest.approx(2 / 3, abs=0.001)
    assert cache["exact_hit_rate"] == pytest.approx(1 / 3, abs=0.001)
    assert cache["semantic_hit_rate"] == pytest.approx(1 / 3, abs=0.001)


# --------------------------------------------------------------------------
# degradation — telemetry must never fail a served request
# --------------------------------------------------------------------------


async def test_record_failure_is_swallowed():
    t = Telemetry(FakeRedis(fail_on={"execute"}))
    assert await t.record(record()) is False  # no exception


async def test_no_redis_is_a_silent_no_op():
    t = Telemetry(None)
    assert t.available is False
    assert await t.record(record()) is False
    assert await t.recent() == []
    assert await t.reset() == 0


async def test_metrics_without_redis_is_structurally_complete():
    """A dashboard should render "no data", not KeyError."""
    m = await Telemetry(None).metrics()
    assert m["requests"] == 0
    assert m["degraded"] == ["redis_unavailable"]
    for section in ("tiers", "router", "cache", "gates", "cost", "tokens", "model_loads"):
        assert section in m
    assert set(m["tiers"]["served"]) == {
        "cache_exact", "cache_semantic", "local_small", "local_large", "frontier"
    }
    assert m["cost"]["saved_pct"] == 0.0


async def test_metrics_read_failure_degrades_to_zeros():
    m = await Telemetry(FakeRedis(fail_on={"hgetall"})).metrics()
    assert m["requests"] == 0


async def test_empty_metrics_has_no_divide_by_zero(telemetry):
    m = await telemetry.metrics()
    assert m["cache"]["hit_rate"] == 0.0
    assert m["gates"]["small"]["escalation_rate"] == 0.0
    assert m["cost"]["saved_pct"] == 0.0
    assert m["model_loads"]["mean_load_ms"] == 0.0


# --------------------------------------------------------------------------
# stream
# --------------------------------------------------------------------------


async def test_recent_returns_newest_first(telemetry):
    for i in range(3):
        await telemetry.record(record(trace_id=f"t{i}"))
    assert [r["trace_id"] for r in await telemetry.recent()] == ["t2", "t1", "t0"]


async def test_recent_round_trips_structured_fields(telemetry):
    await telemetry.record(record(
        path=["local_small", "local_large"],
        latency_ms={"local_small": 1.0, "total": 2.0},
        gates=[{"gate": "small", "verdict": "escalate"}],
    ))
    rec = (await telemetry.recent())[0]
    assert rec["path"] == ["local_small", "local_large"]
    assert rec["latency_ms"]["total"] == 2.0
    assert rec["gates"][0]["gate"] == "small"


async def test_reset_clears_counters(telemetry):
    await telemetry.record(record())
    await telemetry.reset()
    m = await telemetry.metrics()
    assert m["requests"] == 0
    assert sum(m["tiers"]["served"].values()) == 0


# --------------------------------------------------------------------------
# integration
# --------------------------------------------------------------------------


@pytest.mark.integration
async def test_live_redis_stream_and_counters():
    from redis.asyncio import Redis

    redis = Redis.from_url("redis://localhost:6379/15", decode_responses=True)
    t = Telemetry(redis)
    try:
        await t.reset()
        await t.record(record(tier="local_large", path=["local_small", "local_large"]))
        await t.record(record(tier="frontier", path=["frontier"],
                              cost_micro_usd=1663, baseline_micro_usd=1663))
        m = await t.metrics()
        assert m["requests"] == 2
        assert sum(m["tiers"]["served"].values()) == 2
        assert m["tiers"]["attempted"]["local_small"] == 1
        assert m["cost"]["saved_usd"] == pytest.approx(
            m["cost"]["baseline_usd"] - m["cost"]["actual_usd"], abs=1e-9
        )
        assert len(await t.recent()) == 2
    finally:
        await t.reset()
        await redis.aclose()
