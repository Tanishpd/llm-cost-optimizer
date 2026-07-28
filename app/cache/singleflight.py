"""Stampede protection: N identical concurrent prompts produce one model call.

Without it, a cold cache under concurrent load is the worst case this gateway
has. Ten identical requests arrive, all miss, all run the cascade, all pay — and
on an 8 GB machine ten concurrent 3B calls do not just cost ten times, they
contend for memory and get slower than one call would have been.

The shape is deliberately modest:

- **One winner runs; the others wait briefly and re-check the cache.** They do
  not queue on the winner's result — that would need a pub/sub round trip and
  couples every waiter's fate to the winner's.
- **A waiter that times out proceeds anyway.** A duplicate model call is a
  wasted few hundred milliseconds; a request that hangs waiting on a lock whose
  holder died is a broken gateway. Correctness of the *answer* never depends on
  the lock — it is purely an efficiency measure.
- **The lock always expires.** `SET NX PX` with a TTL, so a crashed holder
  cannot wedge a key forever.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger("hlr.cache.singleflight")

LOCK_PREFIX = "hlr:lock:"

# Long enough to cover a full cascade walk (0.5B -> 3B -> frontier), short
# enough that a crashed holder frees the key quickly.
DEFAULT_TTL_MS = 30_000
# A waiter gives up well before the lock TTL — it would rather do duplicate work
# than sit on a request.
DEFAULT_WAIT_MS = 2_500
POLL_INTERVAL_S = 0.05

# Release only if we still hold it: a lock that expired mid-flight may have been
# re-acquired by someone else, and deleting theirs would defeat the purpose.
_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


@dataclass
class Lease:
    """The outcome of trying to be the one that does the work."""

    acquired: bool
    token: str = ""
    waited_ms: float = 0.0
    # True when we waited for another holder and the answer appeared. The caller
    # should re-check the cache rather than run the cascade.
    should_recheck: bool = False


class SingleFlight:
    """Best-effort deduplication of concurrent identical requests."""

    def __init__(
        self,
        redis: Redis | None,
        *,
        ttl_ms: int = DEFAULT_TTL_MS,
        wait_ms: int = DEFAULT_WAIT_MS,
    ) -> None:
        self._redis = redis
        self._ttl_ms = ttl_ms
        self._wait_ms = wait_ms

    @property
    def enabled(self) -> bool:
        return self._redis is not None

    async def acquire(self, digest: str) -> Lease:
        """Try to become the single flight for `digest`.

        Returns `acquired=True` if this caller should do the work. Otherwise it
        has already waited for the holder; `should_recheck` says whether the
        wait ended because the lock cleared (so an answer is probably cached) or
        because it timed out (so proceed and do the work anyway).

        With no Redis, every caller is a winner — there is nowhere to
        coordinate, and the gateway still has to serve.
        """
        if self._redis is None:
            return Lease(acquired=True)

        key = f"{LOCK_PREFIX}{digest}"
        token = uuid.uuid4().hex
        try:
            won = await self._redis.set(key, token, nx=True, px=self._ttl_ms)
        except RedisError as exc:
            # A lock we cannot take is not a reason to refuse the request.
            logger.warning("single-flight unavailable, proceeding unguarded: %s", exc)
            return Lease(acquired=True)

        if won:
            return Lease(acquired=True, token=token)

        waited = await self._wait_for(key)
        return Lease(
            acquired=False,
            waited_ms=round(waited * 1000, 1),
            should_recheck=waited * 1000 < self._wait_ms,
        )

    async def _wait_for(self, key: str) -> float:
        """Poll until the holder releases or we give up. Returns seconds waited."""
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + self._wait_ms / 1000
        while loop.time() < deadline:
            await asyncio.sleep(POLL_INTERVAL_S)
            try:
                if not await self._redis.exists(key):  # type: ignore[union-attr]
                    break
            except RedisError:
                break
        return loop.time() - started

    async def release(self, digest: str, token: str) -> None:
        """Release a lease we hold. Never raises."""
        if self._redis is None or not token:
            return
        try:
            await self._redis.eval(_RELEASE_LUA, 1, f"{LOCK_PREFIX}{digest}", token)
        except RedisError as exc:
            # The TTL will clean up; a failed release costs one stampede window.
            logger.warning("single-flight release failed (TTL will clear it): %s", exc)
