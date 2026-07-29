"""Rate limiting that is correct across workers.

The in-process limiter in `security.py` is correct for one worker and silently
wrong for more: with N workers each enforces its own window, so a configured 60
requests/minute becomes 60N. On a gateway that spends money per request that is
not a rounding error.

This implementation keeps the window in Redis as a sorted set per principal, so
every worker consults the same state. The whole check-and-record is one Lua
script — without atomicity, N workers can each read "59 used" and each admit a
request, which is the exact race the limiter exists to prevent.

Degradation: if Redis is unreachable the limiter falls back to the in-process
one rather than failing open. A local limit that multiplies by worker count is
still far better than no limit at all while Redis is down.
"""

from __future__ import annotations

import logging
import time
import uuid

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.security import RateLimiter

logger = logging.getLogger("hlr.ratelimit")

KEY_PREFIX = "hlr:rl:"

# Sliding-window log. Returns {allowed, remaining, retry_after_ms}.
#
# Redis Lua truncates numeric returns to integers, so retry is returned in
# milliseconds rather than as a fractional second.
#
# `now` is supplied by the caller rather than read from redis.call('TIME'):
# workers share a host clock, and a client-supplied timestamp keeps the script
# deterministic.
_SLIDING_WINDOW_LUA = """
local key    = KEYS[1]
local now    = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit  = tonumber(ARGV[3])
local member = ARGV[4]

redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local used = redis.call('ZCARD', key)

if used >= limit then
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry = 0
    if oldest[2] then
        retry = math.ceil((tonumber(oldest[2]) + window - now) * 1000)
    end
    if retry < 0 then retry = 0 end
    return {0, 0, retry}
end

redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, math.ceil(window))
return {1, limit - used - 1, 0}
"""


class RedisRateLimiter:
    """Sliding-window limiter shared by every worker.

    Exposes the same `check(principal) -> (allowed, remaining, retry_after)`
    shape as the in-process limiter, so the middleware does not care which is
    in play.
    """

    def __init__(
        self,
        redis: Redis | None,
        *,
        limit: int = 60,
        window_s: float = 60.0,
        fallback: RateLimiter | None = None,
    ) -> None:
        self._redis = redis
        self._limit = limit
        self._window = window_s
        # Used only while Redis is unreachable. Wrong across workers, but a
        # limit that multiplies by worker count beats no limit at all.
        self._fallback = fallback or RateLimiter(limit=limit, window_s=window_s)
        self._script = None
        self._degraded = False

    @property
    def enabled(self) -> bool:
        return self._limit > 0

    @property
    def distributed(self) -> bool:
        """True when the shared Redis window is actually in use."""
        return self._redis is not None and not self._degraded

    async def check(
        self, principal: str, *, limit: int | None = None
    ) -> tuple[bool, int, float]:
        """`limit` overrides the configured ceiling for this call.

        That is how a per-tenant limit reaches the shared window without one
        limiter instance per tenant — the Lua script already takes the limit as
        an argument, so the window itself is untouched by the override.
        """
        effective = self._limit if limit is None else limit
        if effective <= 0:
            return True, -1, 0.0
        if self._redis is None:
            return self._fallback.check(principal, limit=effective)

        if self._script is None:
            self._script = self._redis.register_script(_SLIDING_WINDOW_LUA)

        try:
            allowed, remaining, retry_ms = await self._script(
                keys=[f"{KEY_PREFIX}{principal}"],
                args=[time.time(), self._window, effective, uuid.uuid4().hex],
            )
            if self._degraded:
                logger.info("rate limiter back on shared Redis state")
                self._degraded = False
            return bool(allowed), int(remaining), round(int(retry_ms) / 1000, 1)
        except RedisError as exc:
            if not self._degraded:
                logger.warning(
                    "rate limiter falling back to per-process state (limit now effectively "
                    "multiplies by worker count): %s",
                    exc,
                )
                self._degraded = True
            return self._fallback.check(principal, limit=effective)

    async def reset(self, principal: str | None = None) -> None:
        """Clear one principal's window, or the fallback's. Test convenience."""
        self._fallback.reset()
        if self._redis is None or principal is None:
            return
        try:
            await self._redis.delete(f"{KEY_PREFIX}{principal}")
        except RedisError:
            pass

    def prune(self) -> int:
        """Redis expires its own keys; only the fallback needs sweeping."""
        return self._fallback.prune()
