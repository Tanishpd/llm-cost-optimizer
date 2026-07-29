"""Tests for app/budgets.py and per-tenant spend caps.

The interesting assertions are not "does the counter add up" — they are the
three trade-offs the module docstring names, each of which could plausibly have
been decided the other way and so needs pinning:

1. **Over budget degrades, it does not reject.** A tenant past their cap still
   gets local answers.
2. **Check-then-spend, with a bounded overshoot.** The bound is asserted, not
   asserted-to-be-small.
3. **Redis down fails open, loudly.** Silent fail-open would be the bug.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from app.audit import AuditLog
from app.budgets import (
    ANONYMOUS_TENANT,
    DEFAULT_TENANT,
    GLOBAL_BUCKET,
    BudgetLedger,
    Tenant,
    TenantRegistry,
    max_overshoot_micro,
    window_key,
)
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
from app.shadow import ShadowSampler
from app.telemetry import Telemetry
from tests.test_gateway import StubFrontier, StubOllama


class FakeRedis:
    """Enough of Redis for the ledger: GET, INCRBY, EXPIRE, and a pipeline.

    Failures raise `redis.exceptions.ConnectionError`, not the builtin of the
    same name — the builtin is not a `RedisError`, so a fake that raised it
    would test a code path the real client can never reach.
    """

    def __init__(self, *, fail: bool = False) -> None:
        self.store: dict[str, int] = {}
        self.fail = fail
        self.expires: dict[str, int] = {}

    async def get(self, key):
        if self.fail:
            raise RedisConnectionError("redis down")
        v = self.store.get(key)
        return None if v is None else str(v)

    def pipeline(self, transaction=False):
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self._r = redis
        self._ops: list = []

    def incrby(self, key, amount):
        self._ops.append(("incrby", key, amount))
        return self

    def expire(self, key, ttl):
        self._ops.append(("expire", key, ttl))
        return self

    async def execute(self):
        if self._r.fail:
            raise RedisConnectionError("redis down")
        out = []
        for op, key, arg in self._ops:
            if op == "incrby":
                self._r.store[key] = self._r.store.get(key, 0) + arg
                out.append(self._r.store[key])
            else:
                self._r.expires[key] = arg
                out.append(True)
        return out


def registry(**budgets: str) -> TenantRegistry:
    return TenantRegistry([
        Tenant(name, keys=frozenset({f"sk-{name}"}), daily_budget_usd=Decimal(v))
        for name, v in budgets.items()
    ])


# --------------------------------------------------------------------------
# tenant resolution
# --------------------------------------------------------------------------


def test_a_key_resolves_to_the_tenant_that_owns_it():
    r = registry(acme="5.00", globex="1.00")
    assert r.resolve("sk-acme").name == "acme"
    assert r.resolve("sk-globex").name == "globex"


def test_an_unknown_key_falls_back_to_default_not_to_a_budgeted_tenant():
    """A key that is authenticated but unmapped must not inherit someone
    else's cap — in either direction. It pays from `default`."""
    r = registry(acme="5.00")
    assert r.resolve("sk-unlisted").name == DEFAULT_TENANT


def test_no_key_is_anonymous():
    assert registry(acme="5.00").resolve(None).name == ANONYMOUS_TENANT


def test_a_key_claimed_by_two_tenants_is_a_configuration_error():
    """Silently picking one would mean spend landing on the wrong account."""
    with pytest.raises(ValueError, match="exactly one tenant"):
        TenantRegistry([
            Tenant("acme", keys=frozenset({"sk-shared"})),
            Tenant("globex", keys=frozenset({"sk-shared"})),
        ])


def test_the_same_key_listed_twice_for_one_tenant_is_fine():
    TenantRegistry([Tenant("acme", keys=frozenset({"sk-a"}))])


def test_default_budget_applies_to_default_and_anonymous():
    r = TenantRegistry([], default_daily_budget_usd=Decimal("2.00"))
    assert r.get(DEFAULT_TENANT).daily_budget_usd == Decimal("2.00")
    assert r.get(ANONYMOUS_TENANT).daily_budget_usd == Decimal("2.00")


def test_configured_excludes_the_implicit_tenants():
    """`/healthz` lists these; padding it with two names nobody configured
    would make a real tenant harder to spot."""
    assert [t.name for t in registry(acme="1").configured] == ["acme"]


# --------------------------------------------------------------------------
# the ledger
# --------------------------------------------------------------------------


async def test_an_unlimited_tenant_never_touches_redis():
    """Budgets are opt-in and must cost nothing when unused — this is what
    makes it safe to check on the cache-hit path."""
    redis = FakeRedis(fail=True)          # any call raises
    ledger = BudgetLedger(redis, TenantRegistry())
    v = await ledger.check(DEFAULT_TENANT)
    assert v.allowed and not v.unavailable


async def test_spend_accumulates_and_then_blocks():
    redis = FakeRedis()
    ledger = BudgetLedger(redis, registry(acme="1.00"))

    assert (await ledger.check("acme")).allowed
    await ledger.record("acme", usd_to_micro(Decimal("0.60")))
    assert (await ledger.check("acme")).allowed

    await ledger.record("acme", usd_to_micro(Decimal("0.45")))
    v = await ledger.check("acme")
    assert not v.allowed
    assert "budget exhausted" in v.reason
    assert v.remaining_micro == 0


async def test_exactly_at_the_cap_blocks():
    """`>=`, not `>`: a $1.00 cap means you may not spend the 1000001st micro."""
    redis = FakeRedis()
    ledger = BudgetLedger(redis, registry(acme="1.00"))
    await ledger.record("acme", usd_to_micro(Decimal("1.00")))
    assert not (await ledger.check("acme")).allowed


async def test_spend_is_scoped_per_tenant():
    redis = FakeRedis()
    ledger = BudgetLedger(redis, registry(acme="1.00", globex="1.00"))
    await ledger.record("acme", usd_to_micro(Decimal("1.00")))
    assert not (await ledger.check("acme")).allowed
    assert (await ledger.check("globex")).allowed


async def test_spend_is_scoped_per_utc_day():
    redis = FakeRedis()
    ledger = BudgetLedger(redis, registry(acme="1.00"))
    day1 = datetime(2026, 7, 28, 23, 59, tzinfo=UTC)
    day2 = datetime(2026, 7, 29, 0, 1, tzinfo=UTC)

    await ledger.record("acme", usd_to_micro(Decimal("1.00")), now=day1)
    assert not (await ledger.check("acme", now=day1)).allowed
    assert (await ledger.check("acme", now=day2)).allowed, "a new day is a new budget"


def test_window_key_is_utc_regardless_of_the_input_timezone():
    from datetime import timedelta, timezone

    late = datetime(2026, 7, 28, 20, 0, tzinfo=timezone(timedelta(hours=-8)))
    assert window_key("acme", now=late).endswith("2026-07-29")


async def test_recording_sets_a_ttl_so_windows_do_not_leak():
    """One key per tenant per day, forever, is a slow leak that only shows up
    on a long-running instance."""
    redis = FakeRedis()
    ledger = BudgetLedger(redis, registry(acme="1.00"))
    await ledger.record("acme", 500)
    # Two keys: the tenant's window and the gateway-wide one, both TTL'd in the
    # same pipeline so neither can be left immortal.
    assert sorted(redis.expires) == sorted([window_key("acme"), window_key(GLOBAL_BUCKET)])
    assert set(redis.expires.values()) == {60 * 60 * 48}


async def test_redis_down_fails_open_and_says_so():
    """Failing closed would turn a cache outage into a total frontier outage.
    Failing open silently would be the actual bug — hence `unavailable`."""
    ledger = BudgetLedger(FakeRedis(fail=True), registry(acme="1.00"))
    v = await ledger.check("acme")
    assert v.allowed
    assert v.unavailable
    assert "not enforced" in v.reason


async def test_no_redis_at_all_fails_open_and_says_so():
    ledger = BudgetLedger(None, registry(acme="1.00"))
    v = await ledger.check("acme")
    assert v.allowed and v.unavailable


async def test_a_failed_record_does_not_raise():
    """The provider already billed; losing the record must not also lose the
    response the caller already paid for."""
    ledger = BudgetLedger(FakeRedis(fail=True), registry(acme="1.00"))
    assert await ledger.record("acme", 1000) == -1


async def test_usage_hides_unlimited_tenants_with_no_spend():
    redis = FakeRedis()
    ledger = BudgetLedger(redis, registry(acme="1.00"))
    usage = await ledger.usage()
    assert "acme" in usage
    assert DEFAULT_TENANT not in usage, "zero-spend, no-cap tenants are noise"


async def test_usage_reports_dollars_not_micros():
    redis = FakeRedis()
    ledger = BudgetLedger(redis, registry(acme="5.00"))
    await ledger.record("acme", usd_to_micro(Decimal("1.25")))
    row = (await ledger.usage())["acme"]
    assert row["spent_usd"] == pytest.approx(1.25)
    assert row["limit_usd"] == pytest.approx(5.00)
    assert row["remaining_usd"] == pytest.approx(3.75)
    assert row["enforced"] is True


# --------------------------------------------------------------------------
# the overshoot bound
# --------------------------------------------------------------------------


def test_max_overshoot_is_one_request_of_output_at_the_clamped_ceiling():
    """"Approximately enforced" is only acceptable when the approximation has
    a number attached. 4096 tokens at $10/1M = $0.04096."""
    micro = max_overshoot_micro(4096, Decimal("10.00"))
    assert micro == usd_to_micro(Decimal("0.04096"))


async def test_the_documented_overshoot_bound_actually_holds():
    """Spend right up to the cap, then let one maximal request through."""
    redis = FakeRedis()
    ledger = BudgetLedger(redis, registry(acme="1.00"))
    await ledger.record("acme", usd_to_micro(Decimal("0.999999")))
    assert (await ledger.check("acme")).allowed        # still under, by a micro

    worst = max_overshoot_micro(4096, Decimal("10.00"))
    await ledger.record("acme", worst)

    over = int(redis.store[window_key("acme")]) - usd_to_micro(Decimal("1.00"))
    assert 0 < over <= worst


# --------------------------------------------------------------------------
# through the gateway
# --------------------------------------------------------------------------


@pytest.fixture
def budget_client():
    def _make(*, tenants, redis=None, keys=None, frontier=None, ollama=None,
              max_daily_spend_usd=Decimal("0")):
        c = TestClient(app)
        c.app.state.settings = Settings(_env_file=None)
        c.app.state.ollama = ollama or StubOllama()
        c.app.state.http = None
        c.app.state.redis = None
        c.app.state.exact_cache = ExactCache(None)
        c.app.state.telemetry = Telemetry(None)
        c.app.state.frontier = frontier or StubFrontier()
        c.app.state.semantic_cache = SemanticCache(None, enabled=False)
        c.app.state.singleflight = SingleFlight(None)
        c.app.state.tenants = tenants
        c.app.state.budgets = BudgetLedger(
            redis, tenants, max_daily_spend_usd=max_daily_spend_usd)
        c.app.state.auth = ApiKeyAuth(
            keys or set(), tenant_for=lambda k: tenants.resolve(k).name)
        c.app.state.rate_limiter = RedisRateLimiter(None, limit=0)
        c.app.state.limits = RequestLimits()
        c.app.state.breakers = BreakerRegistry(failure_threshold=10**6)
        c.app.state.retry_policy = RetryPolicy(attempts=1)
        # Disabled: the request path must be exercised without any test
        # writing files. tests/test_audit.py covers the enabled path.
        c.app.state.audit = AuditLog("unused", enabled=False)
        # Off by default, as shipped. tests/test_shadow.py covers it on.
        c.app.state.shadow = ShadowSampler(rate=0.0)
        c.app.state.calibrators = {}
        return c
    return _make


def _ask(c, content="q", key="sk-acme", **extra):
    return c.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": content}], **extra},
        headers={"X-API-Key": key},
    )


def test_x_hlr_reports_who_paid(budget_client):
    reg = registry(acme="5.00")
    c = budget_client(tenants=reg, redis=FakeRedis(), keys={"sk-acme"})
    hlr = _ask(c).json()["x_hlr"]
    assert hlr["budget"]["tenant"] == "acme"
    assert hlr["budget"]["limit_usd"] == pytest.approx(5.0)
    assert hlr["budget"]["enforced"] is True


def test_an_exhausted_tenant_still_gets_an_answer(budget_client):
    """The load-bearing behaviour: degrade to local, do not reject."""
    reg = registry(acme="1.00")
    redis = FakeRedis()
    redis.store[window_key("acme")] = usd_to_micro(Decimal("1.00"))

    c = budget_client(tenants=reg, redis=redis, keys={"sk-acme"})
    r = _ask(c, "Prove every finite integral domain is a field.")

    assert r.status_code == 200
    hlr = r.json()["x_hlr"]
    assert hlr["tier"] != "frontier"
    assert "frontier" not in hlr["path"]
    assert "budget_exhausted:acme" in hlr["degraded"]
    assert "budget exhausted" in hlr["reason"]


def test_an_exhausted_tenant_forcing_frontier_gets_402(budget_client):
    """The one case with no cheaper fallback, because the caller removed it."""
    reg = registry(acme="1.00")
    redis = FakeRedis()
    redis.store[window_key("acme")] = usd_to_micro(Decimal("1.00"))

    c = budget_client(tenants=reg, redis=redis, keys={"sk-acme"})
    r = _ask(c, "q", model="gateway:frontier")

    assert r.status_code == 402
    assert r.json()["error"]["code"] == "budget_exhausted"


def test_a_tenant_within_budget_may_still_reach_the_frontier(budget_client):
    reg = registry(acme="5.00")
    c = budget_client(tenants=reg, redis=FakeRedis(), keys={"sk-acme"})
    hlr = _ask(c, "q", model="gateway:frontier").json()["x_hlr"]
    assert hlr["tier"] == "frontier"


def test_frontier_spend_lands_on_the_calling_tenant(budget_client):
    reg = registry(acme="5.00", globex="5.00")
    redis = FakeRedis()
    c = budget_client(tenants=reg, redis=redis, keys={"sk-acme", "sk-globex"})

    _ask(c, "q", key="sk-acme", model="gateway:frontier")

    assert redis.store.get(window_key("acme"), 0) > 0
    assert redis.store.get(window_key("globex"), 0) == 0


def test_local_answers_cost_the_tenant_nothing(budget_client):
    """Local tiers book $0 (D13), so they must not consume budget."""
    reg = registry(acme="5.00")
    redis = FakeRedis()
    c = budget_client(tenants=reg, redis=redis, keys={"sk-acme"})
    _ask(c, "q")
    assert redis.store.get(window_key("acme"), 0) == 0


def test_budget_remaining_reflects_this_request_not_the_one_before(budget_client):
    reg = registry(acme="5.00")
    redis = FakeRedis()
    c = budget_client(tenants=reg, redis=redis, keys={"sk-acme"})
    hlr = _ask(c, "q", model="gateway:frontier").json()["x_hlr"]
    assert hlr["budget"]["spent_usd"] > 0, "spend from this very request is visible"
    assert hlr["budget"]["remaining_usd"] < 5.0


def test_redis_down_serves_and_flags_rather_than_blocking(budget_client):
    reg = registry(acme="1.00")
    c = budget_client(tenants=reg, redis=FakeRedis(fail=True), keys={"sk-acme"})
    r = _ask(c, "q", model="gateway:frontier")
    assert r.status_code == 200
    hlr = r.json()["x_hlr"]
    assert "budget_unavailable" in hlr["degraded"]
    assert hlr["budget"]["enforced"] is False


def test_no_budgets_configured_changes_nothing(budget_client):
    """The shipped default. Budgets must be invisible until switched on."""
    c = budget_client(tenants=TenantRegistry(), redis=FakeRedis())
    r = _ask(c, "q", key="")
    assert r.status_code == 200
    hlr = r.json()["x_hlr"]
    assert hlr["budget"]["tenant"] == "anonymous"
    assert hlr["budget"]["limit_usd"] is None
    assert hlr["budget"]["enforced"] is False
    assert not [d for d in hlr["degraded"] if "budget" in d], (
        "redis_unavailable from the cache is expected here; a budget marker is not")


def test_metrics_attributes_spend_per_tenant(budget_client):
    reg = registry(acme="5.00")
    redis = FakeRedis()
    c = budget_client(tenants=reg, redis=redis, keys={"sk-acme"})
    _ask(c, "q", model="gateway:frontier")

    m = c.get("/metrics", headers={"X-API-Key": "sk-acme"}).json()
    assert m["tenants"]["window"] == "utc_day"
    assert m["tenants"]["usage"]["acme"]["spent_usd"] > 0


def test_healthz_says_whether_budgets_are_actually_enforced(budget_client):
    reg = registry(acme="5.00")
    c = budget_client(tenants=reg, redis=None, keys={"sk-acme"})
    cfg = c.get("/healthz").json()["config"]
    assert cfg["tenants"] == ["acme"]
    assert cfg["budgets_enforced"] is False, "caps configured but Redis unreachable"


# --------------------------------------------------------------------------
# config parsing
# --------------------------------------------------------------------------


def test_tenant_keys_authenticate_without_being_listed_twice():
    s = Settings(_env_file=None, tenants={"acme": {"keys": ["sk-a"]}})
    assert "sk-a" in s.api_key_set


def test_flat_keys_and_tenant_keys_coexist():
    s = Settings(_env_file=None, gateway_api_keys="sk-flat",
                 tenants={"acme": {"keys": ["sk-a"]}})
    assert s.api_key_set == {"sk-flat", "sk-a"}


def test_a_malformed_tenant_raises_rather_than_loading_with_no_budget():
    """Skipping a bad entry would produce a tenant with no cap — the exact
    outcome the setting exists to prevent."""
    with pytest.raises(ValueError, match="must be a list"):
        Settings(_env_file=None, tenants={"acme": {"keys": "sk-a"}}).tenant_list()

    with pytest.raises(ValueError, match="not a number"):
        Settings(_env_file=None,
                 tenants={"acme": {"daily_budget_usd": "free"}}).tenant_list()

    with pytest.raises(ValueError, match="negative"):
        Settings(_env_file=None,
                 tenants={"acme": {"daily_budget_usd": "-1"}}).tenant_list()


def test_budgets_survive_the_json_float_round_trip():
    """A budget arriving as JSON is a float. `Decimal(0.1)` is 0.1000000000
    000000055511151231257827021181583404541015625, and comparing accumulated
    integer micros against that would be off by one."""
    s = Settings(_env_file=None, tenants={"acme": {"daily_budget_usd": 0.1}})
    assert s.tenant_list()[0].daily_budget_micro == 100_000


# --------------------------------------------------------------------------
# the gateway-wide ceiling
#
# Unlike the per-tenant budgets above, this ships ARMED. It exists because
# invariant 4 ("Ollama down -> escalate everything") keeps the gateway
# answering at roughly $138/day per client at the shipped rate limit, and
# before this the only protection was opt-in — present, and unarmed.
# --------------------------------------------------------------------------


def test_the_ceiling_is_armed_by_default():
    """A safety limit that defaults to off is how you learn about it from an
    invoice. This is the one budget setting that is deliberately not 0."""
    assert Settings(_env_file=None).max_daily_spend_usd == Decimal("10.00")


async def test_global_spend_accumulates_across_tenants():
    """The point of a *gateway* ceiling: no single tenant need exceed anything
    for the gateway to have spent too much."""
    redis = FakeRedis()
    ledger = BudgetLedger(redis, registry(a="100", b="100"),
                          max_daily_spend_usd=Decimal("1.00"))
    await ledger.record("a", usd_to_micro(Decimal("0.60")))
    assert (await ledger.check_global()).allowed
    await ledger.record("b", usd_to_micro(Decimal("0.50")))

    v = await ledger.check_global()
    assert not v.allowed
    assert "ceiling reached" in v.reason
    # Neither tenant is anywhere near their own $100.
    assert (await ledger.check("a")).allowed
    assert (await ledger.check("b")).allowed


async def test_zero_disables_the_ceiling():
    redis = FakeRedis()
    ledger = BudgetLedger(redis, registry(), max_daily_spend_usd=Decimal("0"))
    await ledger.record(DEFAULT_TENANT, usd_to_micro(Decimal("999")))
    assert (await ledger.check_global()).allowed


async def test_the_ceiling_fails_open_and_says_so():
    """Same trade as the tenant budgets (D25): failing closed would turn a
    Redis blip into a total frontier outage."""
    ledger = BudgetLedger(FakeRedis(fail=True), registry(),
                          max_daily_spend_usd=Decimal("1.00"))
    v = await ledger.check_global()
    assert v.allowed and v.unavailable


async def test_usage_reports_the_ceiling_separately_from_tenants():
    redis = FakeRedis()
    ledger = BudgetLedger(redis, registry(acme="5.00"),
                          max_daily_spend_usd=Decimal("20.00"))
    await ledger.record("acme", usd_to_micro(Decimal("1.00")))
    usage = await ledger.usage()
    assert usage[GLOBAL_BUCKET]["spent_usd"] == pytest.approx(1.0)
    assert usage[GLOBAL_BUCKET]["limit_usd"] == pytest.approx(20.0)
    assert usage["acme"]["spent_usd"] == pytest.approx(1.0)


def test_the_ceiling_degrades_to_local_like_a_tenant_budget(budget_client):
    reg = registry(acme="100.00")
    redis = FakeRedis()
    redis.store[window_key(GLOBAL_BUCKET)] = usd_to_micro(Decimal("10.00"))

    c = budget_client(tenants=reg, redis=redis, keys={"sk-acme"},
                      max_daily_spend_usd=Decimal("10.00"))
    r = _ask(c, "Prove every finite integral domain is a field.")

    assert r.status_code == 200
    hlr = r.json()["x_hlr"]
    assert "frontier" not in hlr["path"]
    assert "spend_ceiling_reached" in hlr["degraded"]
    assert "ceiling reached" in hlr["reason"]


def test_the_ceiling_outranks_a_tenant_who_is_within_budget(budget_client):
    """acme has $100 of headroom and still may not spend: an operator limit is
    not something a caller can be within."""
    reg = registry(acme="100.00")
    redis = FakeRedis()
    redis.store[window_key(GLOBAL_BUCKET)] = usd_to_micro(Decimal("10.00"))

    c = budget_client(tenants=reg, redis=redis, keys={"sk-acme"},
                      max_daily_spend_usd=Decimal("10.00"))
    hlr = _ask(c, "q").json()["x_hlr"]

    # The tenant's own position is still reported truthfully...
    assert hlr["budget"]["tenant"] == "acme"
    assert hlr["budget"]["remaining_usd"] == pytest.approx(100.0)
    # ...but the ceiling is what governed.
    assert "spend_ceiling_reached" in hlr["degraded"]


def test_pinned_frontier_under_the_ceiling_is_503_not_402(budget_client):
    """The caller's account is fine and they can do nothing about it, so
    billing them for the refusal would be the wrong story."""
    reg = registry(acme="100.00")
    redis = FakeRedis()
    redis.store[window_key(GLOBAL_BUCKET)] = usd_to_micro(Decimal("10.00"))

    c = budget_client(tenants=reg, redis=redis, keys={"sk-acme"},
                      max_daily_spend_usd=Decimal("10.00"))
    r = _ask(c, "q", model="gateway:frontier")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "spend_ceiling_reached"


def test_the_ollama_outage_scenario_is_bounded(budget_client):
    """The scenario this whole setting exists for.

    Ollama is down, so invariant 4 sends every request to the frontier. Without
    a ceiling that is unbounded; with one it stops, and says why.
    """
    reg = registry()
    redis = FakeRedis()
    c = budget_client(
        tenants=reg, redis=redis, keys=set(),
        ollama=StubOllama(error=ProviderUnavailable("unreachable", fix="ollama serve &")),
        max_daily_spend_usd=Decimal("0.0005"),
    )
    codes, tiers = [], []
    for i in range(6):
        r = _ask(c, f"question {i}", key="")
        codes.append(r.status_code)
        if r.status_code == 200:
            tiers.append(r.json()["x_hlr"]["tier"])

    assert "frontier" in tiers, "the first requests should escalate as designed"
    spent = redis.store.get(window_key(GLOBAL_BUCKET), 0)
    assert spent <= usd_to_micro(Decimal("0.0005")) + max_overshoot_micro(
        4096, Decimal("10.00")), "spend exceeded the ceiling by more than one request"
    # Once the ceiling binds and Ollama is still down, there is nothing left to
    # serve with — a loud 503 rather than a quiet invoice.
    assert 503 in codes
