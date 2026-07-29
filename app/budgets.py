"""Per-tenant spend caps.

Auth answers *who is calling*. This answers *what they are allowed to spend*.
Before it existed, a key was a credential rather than an account: every key
drew on the same frontier budget, and `/metrics` could tell you what a request
cost but not who owed for it.

Three decisions worth knowing about, each with a real trade-off:

**Over budget degrades, it does not reject.** A tenant past their cap keeps
getting answers — just only from the local tiers, with `budget_exhausted` in
`degraded` and the reason saying so. On a gateway whose entire thesis is "most
prompts don't need the expensive model", refusing service when the expensive
model is unaffordable would be the wrong instinct. A 402 is returned only when
the frontier was the *only* option left, which means every local tier was
already down.

**The check is before the call, the record is after it**, because the cost is
not knowable until the provider replies. So a tenant can overshoot their cap by
at most one request. That bound is computable and small — at the shipped
`MAX_OUTPUT_TOKENS=4096` and $10/1M output tokens, worst case is about **$0.04**
— and `max_overshoot_usd()` reports it rather than leaving it to be discovered.
Reserving an estimate up front would remove the overshoot and introduce a worse
failure: requests denied against tokens they were never going to use.

**Redis down means budgets are not enforced.** Spend control that fails open is
uncomfortable, but the alternative — failing closed — turns a cache outage into
a total frontier outage, and the rate limiter already made the same call. It is
loud rather than silent: `degraded` says `budget_unavailable`, and `/healthz`
reports it.

Windows are UTC calendar days. A rolling 24h window would need a sorted set per
tenant and answers a question nobody asks about a budget; "what did we spend
today" is the question, and a day boundary is what a human means by it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.costs import micro_to_usd, usd_to_micro

logger = logging.getLogger("hlr.budgets")

# Two days, so a request landing either side of midnight UTC always finds its
# own window and yesterday's is still readable for a report.
WINDOW_TTL_S = 60 * 60 * 48

DEFAULT_TENANT = "default"
ANONYMOUS_TENANT = "anonymous"

# The whole gateway's spend, across every tenant. Not a tenant itself — the
# leading underscores keep it out of any namespace an operator would configure,
# and `usage()` reports it separately so it is never mistaken for an account.
GLOBAL_BUCKET = "__all__"


def window_key(tenant: str, *, now: datetime | None = None) -> str:
    """`hlr:budget:{tenant}:{YYYY-MM-DD}` — UTC, so the boundary does not move
    with the server's timezone or with daylight saving."""
    stamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%d")
    return f"hlr:budget:{tenant}:{stamp}"


@dataclass(frozen=True)
class Tenant:
    """An account. `daily_budget_usd` of 0 means unlimited."""

    name: str
    keys: frozenset[str] = field(default_factory=frozenset)
    daily_budget_usd: Decimal = Decimal("0")

    @property
    def unlimited(self) -> bool:
        return self.daily_budget_usd <= 0

    @property
    def daily_budget_micro(self) -> int:
        return usd_to_micro(self.daily_budget_usd)


@dataclass(frozen=True)
class BudgetVerdict:
    """The answer to "may this tenant reach the frontier right now?"."""

    allowed: bool
    tenant: str
    spent_micro: int = 0
    limit_micro: int = 0
    reason: str = ""
    # True when Redis was unreachable, i.e. `allowed` is a fail-open default
    # rather than a measurement. The caller surfaces this in `degraded`.
    unavailable: bool = False

    @property
    def remaining_micro(self) -> int:
        if self.limit_micro <= 0:
            return -1                      # unlimited
        return max(0, self.limit_micro - self.spent_micro)

    def as_dict(self) -> dict:
        return {
            "tenant": self.tenant,
            "spent_usd": float(micro_to_usd(self.spent_micro)),
            "limit_usd": float(micro_to_usd(self.limit_micro)) if self.limit_micro > 0 else None,
            "remaining_usd": (
                float(micro_to_usd(self.remaining_micro)) if self.remaining_micro >= 0 else None
            ),
            "enforced": not self.unavailable and self.limit_micro > 0,
        }


class TenantRegistry:
    """Maps an API key to the account that pays for it.

    Keys configured through `GATEWAY_API_KEYS` belong to the `default` tenant,
    so adding budgets to an existing deployment changes nothing until a named
    tenant is configured. An open gateway bills everything to `anonymous`.
    """

    def __init__(
        self,
        tenants: list[Tenant] | None = None,
        *,
        default_daily_budget_usd: Decimal = Decimal("0"),
    ) -> None:
        self._tenants = {t.name: t for t in (tenants or [])}
        for implicit in (DEFAULT_TENANT, ANONYMOUS_TENANT):
            self._tenants.setdefault(
                implicit, Tenant(implicit, daily_budget_usd=default_daily_budget_usd)
            )
        # Reverse index, built once. A key claimed by two tenants is a
        # configuration error rather than a tie to break silently.
        self._by_key: dict[str, str] = {}
        for tenant in self._tenants.values():
            for key in tenant.keys:
                if key in self._by_key and self._by_key[key] != tenant.name:
                    raise ValueError(
                        f"API key is claimed by both {self._by_key[key]!r} and "
                        f"{tenant.name!r} — a key must belong to exactly one tenant"
                    )
                self._by_key[key] = tenant.name

    @property
    def names(self) -> list[str]:
        return sorted(self._tenants)

    @property
    def configured(self) -> list[Tenant]:
        """Explicitly configured tenants, excluding the two implicit ones."""
        return [t for n, t in sorted(self._tenants.items())
                if n not in (DEFAULT_TENANT, ANONYMOUS_TENANT)]

    def all_keys(self) -> set[str]:
        return set(self._by_key)

    def get(self, name: str) -> Tenant:
        return self._tenants.get(name) or self._tenants[DEFAULT_TENANT]

    def resolve(self, key: str | None) -> Tenant:
        """Which account pays for a request made with this key."""
        if not key:
            return self._tenants[ANONYMOUS_TENANT]
        return self._tenants.get(self._by_key.get(key, DEFAULT_TENANT),
                                 self._tenants[DEFAULT_TENANT])


class BudgetLedger:
    """Reads and writes per-tenant spend, and the global ceiling. Never raises.

    The global ceiling exists because the *safety* behaviour is also the
    expensive one. Invariant 4 says "Ollama down -> escalate everything", which
    keeps the gateway answering and, at the shipped rate limit, spends about
    $138/day per client while doing it. Tenant budgets can cap that, but they
    are opt-in — so before this existed the protection was present and unarmed.

    A daily ceiling is deliberately blunt. It cannot distinguish a legitimate
    traffic spike from a local-tier outage; it only bounds the damage either
    way. The sharper instrument — tripping on a sudden jump in the *rate* of
    frontier escalations, which is the actual signature of an outage — is worth
    building later and is not a reason to ship without a bound now.
    """

    def __init__(
        self,
        redis: Redis | None,
        registry: TenantRegistry,
        *,
        max_daily_spend_usd: Decimal = Decimal("0"),
    ) -> None:
        self._redis = redis
        self._registry = registry
        self._global_limit = usd_to_micro(max_daily_spend_usd) if max_daily_spend_usd > 0 else 0

    @property
    def global_limit_micro(self) -> int:
        return self._global_limit

    async def check_global(self, *, now: datetime | None = None) -> BudgetVerdict:
        """Has the gateway as a whole spent its daily allowance?

        Checked before the tenant's own budget: an operator ceiling is not
        something a caller can be within.
        """
        if self._global_limit <= 0:
            return BudgetVerdict(True, GLOBAL_BUCKET, reason="no global ceiling configured")
        if self._redis is None:
            return BudgetVerdict(
                True, GLOBAL_BUCKET, limit_micro=self._global_limit, unavailable=True,
                reason="global ceiling not enforced: Redis unreachable",
            )
        try:
            raw = await self._redis.get(window_key(GLOBAL_BUCKET, now=now))
            spent = int(raw or 0)
        except (RedisError, ValueError, TypeError) as exc:
            logger.warning("global budget read failed, allowing: %s", exc)
            return BudgetVerdict(
                True, GLOBAL_BUCKET, limit_micro=self._global_limit, unavailable=True,
                reason="global ceiling not enforced: Redis read failed",
            )
        if spent >= self._global_limit:
            return BudgetVerdict(
                False, GLOBAL_BUCKET, spent_micro=spent, limit_micro=self._global_limit,
                reason=(
                    f"gateway daily spend ceiling reached: ${micro_to_usd(spent)} of "
                    f"${micro_to_usd(self._global_limit)} (UTC day, all tenants)"
                ),
            )
        return BudgetVerdict(
            True, GLOBAL_BUCKET, spent_micro=spent, limit_micro=self._global_limit,
            reason=f"${micro_to_usd(self._global_limit - spent)} of today's ceiling remaining",
        )

    @property
    def available(self) -> bool:
        return self._redis is not None

    async def check(self, tenant_name: str, *, now: datetime | None = None) -> BudgetVerdict:
        """May this tenant reach a billed tier right now?

        Read-only — nothing is reserved. See the module docstring on why the
        one-request overshoot is the accepted trade.
        """
        tenant = self._registry.get(tenant_name)
        if tenant.unlimited:
            return BudgetVerdict(True, tenant.name, reason="no budget configured")

        limit = tenant.daily_budget_micro
        if self._redis is None:
            return BudgetVerdict(
                True, tenant.name, limit_micro=limit, unavailable=True,
                reason="budget not enforced: Redis unreachable",
            )
        try:
            raw = await self._redis.get(window_key(tenant.name, now=now))
            spent = int(raw or 0)
        except (RedisError, ValueError, TypeError) as exc:
            logger.warning("budget read failed for %s, allowing: %s", tenant.name, exc)
            return BudgetVerdict(
                True, tenant.name, limit_micro=limit, unavailable=True,
                reason="budget not enforced: Redis read failed",
            )

        if spent >= limit:
            return BudgetVerdict(
                False, tenant.name, spent_micro=spent, limit_micro=limit,
                reason=(
                    f"daily budget exhausted: ${micro_to_usd(spent)} of "
                    f"${micro_to_usd(limit)} spent today (UTC)"
                ),
            )
        return BudgetVerdict(
            True, tenant.name, spent_micro=spent, limit_micro=limit,
            reason=f"${micro_to_usd(limit - spent)} of today's budget remaining",
        )

    async def record(
        self, tenant_name: str, cost_micro_usd: int, *, now: datetime | None = None
    ) -> int:
        """Add spend to today's window. Returns the new total, or -1 if unrecorded.

        `INCRBY` + `EXPIRE` in one pipeline: the counter cannot end up without
        a TTL and leak a key per tenant per day forever.
        """
        if self._redis is None or cost_micro_usd <= 0:
            return -1
        key = window_key(tenant_name, now=now)
        gkey = window_key(GLOBAL_BUCKET, now=now)
        try:
            # Both counters in one pipeline: a tenant's spend and the gateway's
            # total can never disagree about a request that happened.
            pipe = self._redis.pipeline(transaction=False)
            pipe.incrby(key, cost_micro_usd)
            pipe.expire(key, WINDOW_TTL_S)
            pipe.incrby(gkey, cost_micro_usd)
            pipe.expire(gkey, WINDOW_TTL_S)
            total, _, _, _ = await pipe.execute()
            return int(total)
        except RedisError as exc:
            # The request already succeeded and the provider already billed.
            # Losing the record understates spend, which is why this is a
            # warning and not a debug line.
            logger.warning("budget record failed for %s (%d micro-USD unrecorded): %s",
                           tenant_name, cost_micro_usd, exc)
            return -1

    async def usage(self, *, now: datetime | None = None) -> dict[str, dict]:
        """Today's spend for every known tenant — for `/metrics`."""
        if self._redis is None:
            return {}
        out: dict[str, dict] = {}
        if self._global_limit > 0:
            try:
                spent = int(await self._redis.get(window_key(GLOBAL_BUCKET, now=now)) or 0)
                out[GLOBAL_BUCKET] = BudgetVerdict(
                    spent < self._global_limit, GLOBAL_BUCKET,
                    spent_micro=spent, limit_micro=self._global_limit).as_dict()
            except (RedisError, ValueError, TypeError):
                pass
        for name in self._registry.names:
            tenant = self._registry.get(name)
            try:
                raw = await self._redis.get(window_key(name, now=now))
                spent = int(raw or 0)
            except (RedisError, ValueError, TypeError):
                continue
            # Skip tenants with neither spend nor a cap: listing every implicit
            # name with zeros makes the real ones harder to see.
            if spent == 0 and tenant.unlimited:
                continue
            out[name] = BudgetVerdict(
                spent < tenant.daily_budget_micro or tenant.unlimited,
                name, spent_micro=spent,
                limit_micro=0 if tenant.unlimited else tenant.daily_budget_micro,
            ).as_dict()
        return out


def max_overshoot_micro(max_output_tokens: int, price_out: Decimal) -> int:
    """The most a tenant can exceed their cap, given check-then-spend.

    One request's worth of output at the clamped maximum. Input tokens are
    bounded by `MAX_PROMPT_CHARS` and cost far less, so this is the term that
    matters. Documented rather than hand-waved, because "approximately
    enforced" is only acceptable when the approximation has a number.
    """
    return usd_to_micro(Decimal(max_output_tokens) * price_out / Decimal(1_000_000))
