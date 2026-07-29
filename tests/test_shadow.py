"""Tests for app/shadow.py and the sampled frontier comparison.

Shadow mode is the only thing in this project that measures its central claim
*continuously*. Everything else validated it once, offline, on 67 prompts — and
gate `large` failed that validation.

Which means the properties worth pinning are the ones that would make it a
liability rather than an instrument:

- it **cannot slow or break** the request it measures,
- it **cannot outspend** the budget it is measuring,
- it **samples the right thing** — accepted local answers, not cache hits or
  frontier answers, which would be paying to compare something with itself,
- and it is **off by default**, because spending money to measure is a decision.
"""

from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.audit import AuditLog
from app.budgets import GLOBAL_BUCKET, BudgetLedger, Tenant, TenantRegistry, window_key
from app.cache.exact import ExactCache
from app.cache.semantic import SemanticCache
from app.cache.singleflight import SingleFlight
from app.config import Settings
from app.costs import usd_to_micro
from app.main import app
from app.providers.base import ProviderUnavailable
from app.ratelimit import RedisRateLimiter
from app.resilience import BreakerRegistry, RetryPolicy
from app.security import ApiKeyAuth, RequestLimits
from app.shadow import ShadowSampler, ShadowStats, similarity
from app.telemetry import Telemetry
from tests.test_budgets import FakeRedis
from tests.test_gateway import StubFrontier, StubOllama

# --------------------------------------------------------------------------
# similarity — a tripwire, not a grade
# --------------------------------------------------------------------------


def test_identical_answers_score_one():
    assert similarity("Paris is the capital", "Paris is the capital") == 1.0


def test_reordering_and_rewording_stays_high():
    """Two correct answers routinely differ in phrasing and order. A sequence
    measure would punish that as though it were disagreement."""
    assert similarity("The capital of France is Paris.",
                      "Paris is the capital city of France.") > 0.7


def test_an_unrelated_answer_scores_low():
    """The M0 hallucination: asked about the Treaty of Kadesh, the 0.5B
    answered about 1956 and Egypt–Israel. That is what divergence looks like."""
    assert similarity(
        "The Treaty of Kadesh was signed c. 1259 BC between Egypt and the Hittites.",
        "It was signed in 1956 between Egypt and Israel, ending the Six-Day War.",
    ) < 0.35


@pytest.mark.parametrize("a,b,expected", [
    ("", "", 1.0),
    ("something", "", 0.0),
    ("", "something", 0.0),
])
def test_empty_answers(a, b, expected):
    assert similarity(a, b) == expected


def test_similarity_is_symmetric():
    a, b = "the quick brown fox", "a brown fox, quick"
    assert similarity(a, b) == similarity(b, a)


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def test_off_by_default():
    """Spending money to measure has to be switched on deliberately."""
    assert Settings(_env_file=None).shadow_sample_rate == 0.0
    assert not ShadowSampler(rate=0.0).enabled


@pytest.mark.parametrize("rate", [0.01, 0.1, 0.5])
def test_the_sample_rate_is_honoured(rate):
    s = ShadowSampler(rate=rate)
    n = 20_000
    hit = sum(s.should_sample(uuid.uuid4().hex[:12], tier="local_small", gate_accepted=True)
              for _ in range(n))
    assert abs(hit / n - rate) < 0.02, f"sampled {hit / n:.3f}, wanted {rate}"


def test_sequential_trace_ids_are_sampled_uniformly_too():
    """`int(trace_id, 16) % N` is only uniform if the ids are. Sequential ids
    put every early request in the same bucket — the first version of this
    sampled 2000/2000 at rate 0.5."""
    s = ShadowSampler(rate=0.1)
    hit = sum(s.should_sample(f"{i:012x}", tier="local_small", gate_accepted=True)
              for i in range(20_000))
    assert abs(hit / 20_000 - 0.1) < 0.02


def test_sampling_is_deterministic_in_the_trace_id():
    """So an anomaly is reproducible rather than a thing that happened once."""
    s = ShadowSampler(rate=0.5)
    tid = uuid.uuid4().hex[:12]
    assert all(s.should_sample(tid, tier="local_small", gate_accepted=True)
               == s.should_sample(tid, tier="local_small", gate_accepted=True)
               for _ in range(5))


@pytest.mark.parametrize("tier", ["cache_exact", "cache_semantic", "frontier"])
def test_only_local_answers_are_shadowed(tier):
    """A cache hit was already checked when produced, and a frontier answer
    *is* the reference — shadowing either pays to compare something with
    itself."""
    s = ShadowSampler(rate=1.0)
    assert not s.should_sample("abc", tier=tier, gate_accepted=True)


def test_an_escalated_answer_is_not_shadowed():
    """It already went to the frontier. There is nothing to compare."""
    s = ShadowSampler(rate=1.0)
    assert not s.should_sample("abc", tier="local_small", gate_accepted=False)


def test_stats_arithmetic():
    st = ShadowStats(completed=4, divergent=1, similarity_sum=2.0)
    assert st.mean_similarity == 0.5
    assert st.divergence_rate == 0.25
    assert ShadowStats().mean_similarity == 0.0        # no division by zero
    assert ShadowStats().divergence_rate == 0.0


# --------------------------------------------------------------------------
# through the gateway
# --------------------------------------------------------------------------


@pytest.fixture
def shadow_client(tmp_path):
    def _make(*, rate=1.0, frontier=None, redis=None, budget=Decimal("0"),
              ceiling=Decimal("0"), captured=None):
        reg = TenantRegistry(
            [Tenant("acme", keys=frozenset({"sk-acme"}), daily_budget_usd=budget)])
        c = TestClient(app)
        st = c.app.state
        st.settings = Settings(_env_file=None, shadow_sample_rate=rate)
        st.ollama = StubOllama()
        st.http = None
        st.redis = None
        st.exact_cache = ExactCache(None)
        st.telemetry = Telemetry(None)
        st.frontier = frontier or StubFrontier(content="the frontier's answer")
        st.semantic_cache = SemanticCache(None, enabled=False)
        st.singleflight = SingleFlight(None)
        st.tenants = reg
        st.budgets = BudgetLedger(redis, reg, max_daily_spend_usd=ceiling)
        st.auth = ApiKeyAuth({"sk-acme"}, tenant_for=lambda k: reg.resolve(k).name)
        st.rate_limiter = RedisRateLimiter(None, limit=0)
        st.limits = RequestLimits()
        st.breakers = BreakerRegistry(failure_threshold=10**6)
        st.retry_policy = RetryPolicy(attempts=1)
        st.calibrators = {}

        class Capturing(AuditLog):
            def write(self, record):
                if captured is not None:
                    captured.append(record)
                return True

        st.audit = Capturing(tmp_path, enabled=True)
        st.shadow = ShadowSampler(rate=rate, divergence_threshold=0.35)
        return c
    return _make


def _ask(c, content="What is the capital of France?"):
    return c.post("/v1/chat/completions",
                  json={"messages": [{"role": "user", "content": content}]},
                  headers={"X-API-Key": "sk-acme"})


async def _settle(c):
    """Let the detached comparison finish."""
    await c.app.state.shadow.drain(timeout=5.0)


def test_the_caller_never_sees_the_shadow(shadow_client):
    """It runs after the response and must not alter it."""
    c = shadow_client(rate=1.0)
    r = _ask(c)
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Paris"   # the local answer
    assert "shadow" not in body["x_hlr"]
    assert "the frontier's answer" not in r.text


def test_a_shadowed_request_records_a_comparison(shadow_client):
    captured: list[dict] = []
    c = shadow_client(rate=1.0, captured=captured)
    _ask(c)
    asyncio.run(_settle(c))

    shadows = [e for e in captured if e.get("outcome") == "shadow"]
    assert len(shadows) == 1
    assert 0.0 <= shadows[0]["similarity"] <= 1.0
    assert shadows[0]["tier"] == "local_small"
    assert c.app.state.shadow.stats.completed == 1


def test_a_divergent_answer_is_flagged(shadow_client):
    """The signal: the gate accepted something that does not resemble what the
    expensive model would have said."""
    captured: list[dict] = []
    c = shadow_client(rate=1.0, captured=captured,
                      frontier=StubFrontier(content="Something entirely unrelated indeed"))
    _ask(c)
    asyncio.run(_settle(c))

    shadow = [e for e in captured if e.get("outcome") == "shadow"][0]
    assert shadow["diverged"] is True
    assert c.app.state.shadow.stats.divergent == 1


def test_an_agreeing_answer_is_not_flagged(shadow_client):
    captured: list[dict] = []
    c = shadow_client(rate=1.0, captured=captured, frontier=StubFrontier(content="Paris"))
    _ask(c)
    asyncio.run(_settle(c))
    assert [e for e in captured if e.get("outcome") == "shadow"][0]["diverged"] is False


def test_rate_zero_shadows_nothing(shadow_client):
    """The shipped default must cost nothing and do nothing."""
    captured: list[dict] = []
    c = shadow_client(rate=0.0, captured=captured)
    _ask(c)
    asyncio.run(_settle(c))
    assert not [e for e in captured if e.get("outcome") == "shadow"]
    assert c.app.state.shadow.stats.sampled == 0


def test_a_failing_frontier_costs_a_measurement_not_a_request(shadow_client):
    """The most important negative: the instrument must not break the thing."""
    c = shadow_client(rate=1.0, frontier=StubFrontier(error=ProviderUnavailable("down")))
    r = _ask(c)
    asyncio.run(_settle(c))

    assert r.status_code == 200, "a failing shadow call broke the request"
    assert r.json()["choices"][0]["message"]["content"] == "Paris"
    assert c.app.state.shadow.stats.failed == 1
    assert c.app.state.shadow.stats.completed == 0


def test_shadow_spend_is_billed_to_the_tenant(shadow_client):
    """It is a real frontier call. Hiding its cost would make the savings
    figure this project reports a lie."""
    redis = FakeRedis()
    c = shadow_client(rate=1.0, redis=redis, budget=Decimal("5.00"))
    _ask(c)
    asyncio.run(_settle(c))

    assert redis.store.get(window_key("acme"), 0) > 0
    assert c.app.state.shadow.stats.cost_micro_usd > 0


def test_shadow_is_skipped_when_the_tenant_is_out_of_budget(shadow_client):
    """Measurement that could exhaust the budget it measures is a poor trade."""
    redis = FakeRedis()
    redis.store[window_key("acme")] = usd_to_micro(Decimal("1.00"))
    c = shadow_client(rate=1.0, redis=redis, budget=Decimal("1.00"))
    _ask(c)
    asyncio.run(_settle(c))

    assert c.app.state.shadow.stats.skipped_budget == 1
    assert c.app.state.shadow.stats.completed == 0


def test_shadow_is_skipped_under_the_gateway_ceiling(shadow_client):
    redis = FakeRedis()
    redis.store[window_key(GLOBAL_BUCKET)] = usd_to_micro(Decimal("10.00"))
    c = shadow_client(rate=1.0, redis=redis, ceiling=Decimal("10.00"))
    _ask(c)
    asyncio.run(_settle(c))
    assert c.app.state.shadow.stats.skipped_budget == 1


def test_drain_does_not_hang_when_nothing_is_in_flight(shadow_client):
    c = shadow_client(rate=0.0)
    asyncio.run(_settle(c))


def test_the_task_set_does_not_grow_without_bound(shadow_client):
    """A held reference per request would be the same slow leak the rate
    limiter's pruner exists for."""
    c = shadow_client(rate=1.0)
    for _ in range(5):
        _ask(c)
    asyncio.run(_settle(c))
    assert len(c.app.state.shadow._tasks) == 0
